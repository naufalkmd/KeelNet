"""Stage 3 confidence-calibration evaluation for QA and support scores."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from keelnet.config import (
    DEFAULT_CALIBRATION_BINS,
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_N_BEST_SIZE,
    DEFAULT_SEED,
    DEFAULT_SUPPORT_MATCH_F1,
    DEFAULT_SUPPORT_THRESHOLD_MAX,
    DEFAULT_SUPPORT_THRESHOLD_MIN,
    DEFAULT_SUPPORT_THRESHOLD_STEP,
    DEFAULT_TEMPERATURE_MAX,
    DEFAULT_TEMPERATURE_MIN,
    DEFAULT_TEMPERATURE_STEP,
    DEFAULT_THRESHOLD_GAP_MIN_COUNT,
    DEFAULT_VALIDATION_SIZE,
    RUN_MODE_ABSTAIN,
    RUN_MODES,
)
from keelnet.metrics import (
    adaptive_expected_calibration_error,
    brier_score,
    build_adaptive_reliability_bins,
    build_reliability_bins,
    expected_calibration_error,
    is_supported_answer,
    maximum_calibration_error,
    pearson_correlation,
    threshold_calibration_gap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-model-path", type=Path, required=True)
    parser.add_argument("--qa-mode", choices=RUN_MODES, required=True)
    parser.add_argument("--verifier-model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--qa-threshold-min", type=float, default=-5.0)
    parser.add_argument("--qa-threshold-max", type=float, default=5.0)
    parser.add_argument("--qa-threshold-step", type=float, default=0.5)
    parser.add_argument(
        "--support-threshold-min",
        type=float,
        default=DEFAULT_SUPPORT_THRESHOLD_MIN,
    )
    parser.add_argument(
        "--support-threshold-max",
        type=float,
        default=DEFAULT_SUPPORT_THRESHOLD_MAX,
    )
    parser.add_argument(
        "--support-threshold-step",
        type=float,
        default=DEFAULT_SUPPORT_THRESHOLD_STEP,
    )
    parser.add_argument(
        "--match-f1-threshold",
        type=float,
        default=DEFAULT_SUPPORT_MATCH_F1,
    )
    parser.add_argument(
        "--qa-temperature-min",
        type=float,
        default=DEFAULT_TEMPERATURE_MIN,
    )
    parser.add_argument(
        "--qa-temperature-max",
        type=float,
        default=DEFAULT_TEMPERATURE_MAX,
    )
    parser.add_argument(
        "--qa-temperature-step",
        type=float,
        default=DEFAULT_TEMPERATURE_STEP,
    )
    parser.add_argument(
        "--support-temperature-min",
        type=float,
        default=DEFAULT_TEMPERATURE_MIN,
    )
    parser.add_argument(
        "--support-temperature-max",
        type=float,
        default=DEFAULT_TEMPERATURE_MAX,
    )
    parser.add_argument(
        "--support-temperature-step",
        type=float,
        default=DEFAULT_TEMPERATURE_STEP,
    )
    parser.add_argument("--calibration-bins", type=int, default=DEFAULT_CALIBRATION_BINS)
    parser.add_argument(
        "--threshold-gap-min-count",
        type=int,
        default=DEFAULT_THRESHOLD_GAP_MIN_COUNT,
    )
    return parser


def sigmoid_scores(scores: np.ndarray | list[float], *, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    scaled = np.asarray(scores, dtype=float) / float(temperature)
    scaled = np.clip(scaled, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def logit_probabilities(probabilities: np.ndarray | list[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_temperature_scaler(
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    *,
    temperature_min: float,
    temperature_max: float,
    temperature_step: float,
    num_bins: int,
) -> float:
    labels_array = np.asarray(labels, dtype=float)
    scores_array = np.asarray(scores, dtype=float)
    if len(scores_array) == 0 or len(np.unique(labels_array)) < 2:
        return 1.0

    best_temperature = 1.0
    best_brier = float("inf")
    best_ece = float("inf")
    current = float(temperature_min)
    while current <= temperature_max + 1e-9:
        probabilities = sigmoid_scores(scores_array, temperature=current).tolist()
        current_brier = brier_score(probabilities, labels_array.tolist())
        current_ece = expected_calibration_error(probabilities, labels_array.tolist(), num_bins=num_bins)
        if current_brier < best_brier - 1e-12 or (
            abs(current_brier - best_brier) <= 1e-12 and current_ece < best_ece
        ):
            best_temperature = current
            best_brier = current_brier
            best_ece = current_ece
        current += temperature_step

    return float(round(best_temperature, 10))


def summarize_binary_calibration(
    probabilities: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    *,
    num_bins: int,
    threshold_gap_min_count: int,
) -> dict[str, Any]:
    probability_list = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6).tolist()
    label_list = np.asarray(labels, dtype=float).tolist()
    sample_count = len(probability_list)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "accuracy": 0.0,
            "positive_rate": 0.0,
            "mean_confidence": 0.0,
            "ece": 0.0,
            "adaptive_ece": 0.0,
            "mce": 0.0,
            "brier_score": 0.0,
            "correlation": 0.0,
            "threshold_gap": 0.0,
            "reliability_bins": [],
            "adaptive_reliability_bins": [],
        }

    predicted_positive = [1.0 if probability >= 0.5 else 0.0 for probability in probability_list]
    accuracy = sum(
        1.0 if prediction == float(label) else 0.0
        for prediction, label in zip(predicted_positive, label_list, strict=False)
    ) / sample_count
    mean_confidence = sum(probability_list) / sample_count
    positive_rate = sum(label_list) / sample_count

    return {
        "sample_count": sample_count,
        "accuracy": accuracy,
        "positive_rate": positive_rate,
        "mean_confidence": mean_confidence,
        "ece": expected_calibration_error(probability_list, label_list, num_bins=num_bins),
        "adaptive_ece": adaptive_expected_calibration_error(probability_list, label_list, num_bins=num_bins),
        "mce": maximum_calibration_error(probability_list, label_list, num_bins=num_bins),
        "brier_score": brier_score(probability_list, label_list),
        "correlation": pearson_correlation(probability_list, label_list),
        "threshold_gap": threshold_calibration_gap(
            probability_list,
            label_list,
            min_count=threshold_gap_min_count,
        ),
        "reliability_bins": build_reliability_bins(probability_list, label_list, num_bins=num_bins),
        "adaptive_reliability_bins": build_adaptive_reliability_bins(
            probability_list,
            label_list,
            num_bins=num_bins,
        ),
    }


def _save_reliability_plot(
    output_path: Path,
    *,
    title: str,
    raw_bins: list[dict[str, float]],
    calibrated_bins: list[dict[str, float]],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    def _curve_points(bins: list[dict[str, float]]) -> tuple[list[float], list[float], list[float]]:
        filtered_bins = [bin_stats for bin_stats in bins if bin_stats["count"] > 0.0]
        x_values = [bin_stats["mean_confidence"] for bin_stats in filtered_bins]
        y_values = [bin_stats["accuracy"] for bin_stats in filtered_bins]
        marker_sizes = [30.0 + (bin_stats["count"] * 1.5) for bin_stats in filtered_bins]
        return x_values, y_values, marker_sizes

    figure, axis = plt.subplots(figsize=(6.0, 4.5))
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#6b7280", linewidth=1.5, label="Perfect calibration")

    for bins, label, color in (
        (raw_bins, "Raw", "#b91c1c"),
        (calibrated_bins, "Calibrated", "#1d4ed8"),
    ):
        x_values, y_values, marker_sizes = _curve_points(bins)
        if not x_values:
            continue
        axis.plot(x_values, y_values, color=color, marker="o", linewidth=2.0, label=label)
        axis.scatter(x_values, y_values, s=marker_sizes, color=color, alpha=0.7)

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title(title)
    axis.grid(True, alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _qa_decision_score(prediction: Mapping[str, Any], *, qa_threshold: float) -> float:
    decision = str(prediction.get("decision", "abstain")).lower()
    margin = float(prediction.get("scores", {}).get("margin", 0.0))
    if decision == "abstain":
        return margin - qa_threshold
    return qa_threshold - margin


def _qa_decision_label(
    prediction: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    match_f1_threshold: float,
) -> float:
    decision = str(prediction.get("decision", "abstain")).lower()
    if decision == "abstain":
        return 1.0 if not bool(reference.get("is_answerable", False)) else 0.0

    if not bool(reference.get("is_answerable", False)):
        return 0.0
    answer_text = str(prediction.get("answer", ""))
    return 1.0 if is_supported_answer(answer_text, list(reference.get("answers", [])), match_f1_threshold=match_f1_threshold) else 0.0


def _collect_qa_confidence_records(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    qa_threshold: float,
    match_f1_threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example_id, reference in references.items():
        prediction = predictions.get(example_id, {"decision": "abstain", "answer": "", "scores": {}})
        records.append(
            {
                "id": example_id,
                "score": _qa_decision_score(prediction, qa_threshold=qa_threshold),
                "label": _qa_decision_label(
                    prediction,
                    reference,
                    match_f1_threshold=match_f1_threshold,
                ),
                "decision": str(prediction.get("decision", "abstain")).lower(),
            }
        )
    return records


def _collect_support_confidence_records(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    match_f1_threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example_id, reference in references.items():
        prediction = predictions.get(example_id, {"decision": "abstain", "answer": "", "support": {}})
        if str(prediction.get("decision", "abstain")).lower() != "answer":
            continue

        raw_probability = float(prediction.get("support", {}).get("score", 0.0))
        answer_text = str(prediction.get("answer", ""))
        label = (
            1.0
            if bool(reference.get("is_answerable", False))
            and is_supported_answer(
                answer_text,
                list(reference.get("answers", [])),
                match_f1_threshold=match_f1_threshold,
            )
            else 0.0
        )
        records.append(
            {
                "id": example_id,
                "score": float(logit_probabilities([raw_probability])[0]),
                "label": label,
                "raw_probability": raw_probability,
            }
        )
    return records


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    temperature: float,
    num_bins: int,
    threshold_gap_min_count: int,
) -> dict[str, Any]:
    scores = np.asarray([record["score"] for record in records], dtype=float)
    labels = np.asarray([record["label"] for record in records], dtype=float)
    probabilities = sigmoid_scores(scores, temperature=temperature)
    return summarize_binary_calibration(
        probabilities,
        labels,
        num_bins=num_bins,
        threshold_gap_min_count=threshold_gap_min_count,
    )


def main() -> None:
    args = build_parser().parse_args()

    from transformers import (
        AutoModelForQuestionAnswering,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from keelnet.data import build_reference_index, load_stage1_splits
    from keelnet.evaluate import (
        predict_raw_qa_outputs,
        prepare_qa_eval_artifacts,
        search_abstain_threshold,
    )
    from keelnet.hf_compat import trainer_processing_kwargs
    from keelnet.metrics import compute_stage1_metrics, compute_stage2_support_metrics
    from keelnet.postprocess import postprocess_qa_predictions
    from keelnet.verify import (
        _finalize_support_predictions,
        _gate_predictions_with_support,
        _score_verifier_predictions,
        _search_support_threshold,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_splits = load_stage1_splits(
        validation_size=args.validation_size,
        seed=args.seed,
        answer_only_train=False,
        max_eval_samples=args.max_eval_samples,
    )

    qa_tokenizer = AutoTokenizer.from_pretrained(args.qa_model_path, use_fast=True)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model_path)
    qa_trainer = Trainer(
        model=qa_model,
        args=TrainingArguments(
            output_dir=str(args.output_path.parent / "tmp-qa-calibration"),
            per_device_eval_batch_size=args.eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, qa_tokenizer),
    )

    validation_examples = stage1_splits["validation"]
    dev_examples = stage1_splits["dev"]
    validation_features, validation_model_dataset = prepare_qa_eval_artifacts(
        validation_examples,
        qa_tokenizer,
        args.max_length,
        args.doc_stride,
    )
    dev_features, dev_model_dataset = prepare_qa_eval_artifacts(
        dev_examples,
        qa_tokenizer,
        args.max_length,
        args.doc_stride,
    )

    validation_raw_predictions = predict_raw_qa_outputs(qa_trainer, validation_model_dataset)
    dev_raw_predictions = predict_raw_qa_outputs(qa_trainer, dev_model_dataset)

    validation_references = build_reference_index(validation_examples)
    dev_references = build_reference_index(dev_examples)

    qa_threshold = 0.0
    qa_validation_metrics: dict[str, float] | None = None
    qa_threshold_sweep: dict[str, list[dict[str, float]]] | None = None
    allow_abstain = args.qa_mode == RUN_MODE_ABSTAIN
    if allow_abstain:
        qa_threshold, qa_validation_metrics, qa_validation_sweep = search_abstain_threshold(
            validation_examples,
            validation_features,
            validation_raw_predictions,
            validation_references,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            threshold_min=args.qa_threshold_min,
            threshold_max=args.qa_threshold_max,
            threshold_step=args.qa_threshold_step,
        )
        _, _, qa_dev_sweep = search_abstain_threshold(
            dev_examples,
            dev_features,
            dev_raw_predictions,
            dev_references,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            threshold_min=args.qa_threshold_min,
            threshold_max=args.qa_threshold_max,
            threshold_step=args.qa_threshold_step,
        )
        qa_threshold_sweep = {"validation": qa_validation_sweep, "dev": qa_dev_sweep}

    validation_qa_predictions = postprocess_qa_predictions(
        validation_examples,
        validation_features,
        validation_raw_predictions,
        allow_abstain=allow_abstain,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=qa_threshold,
    )
    dev_qa_predictions = postprocess_qa_predictions(
        dev_examples,
        dev_features,
        dev_raw_predictions,
        allow_abstain=allow_abstain,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=qa_threshold,
    )

    qa_validation_records = _collect_qa_confidence_records(
        validation_qa_predictions,
        validation_references,
        qa_threshold=qa_threshold,
        match_f1_threshold=args.match_f1_threshold,
    )
    qa_dev_records = _collect_qa_confidence_records(
        dev_qa_predictions,
        dev_references,
        qa_threshold=qa_threshold,
        match_f1_threshold=args.match_f1_threshold,
    )

    qa_temperature = fit_temperature_scaler(
        [record["score"] for record in qa_validation_records],
        [record["label"] for record in qa_validation_records],
        temperature_min=args.qa_temperature_min,
        temperature_max=args.qa_temperature_max,
        temperature_step=args.qa_temperature_step,
        num_bins=args.calibration_bins,
    )

    verifier_tokenizer = AutoTokenizer.from_pretrained(args.verifier_model_path, use_fast=True)
    verifier_model = AutoModelForSequenceClassification.from_pretrained(args.verifier_model_path)
    verifier_trainer = Trainer(
        model=verifier_model,
        args=TrainingArguments(
            output_dir=str(args.output_path.parent / "tmp-verifier-calibration"),
            per_device_eval_batch_size=args.eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, verifier_tokenizer),
    )

    validation_scored_predictions = _score_verifier_predictions(
        verifier_trainer,
        verifier_tokenizer,
        validation_examples,
        validation_qa_predictions,
        max_length=args.max_length,
    )
    dev_scored_predictions = _score_verifier_predictions(
        verifier_trainer,
        verifier_tokenizer,
        dev_examples,
        dev_qa_predictions,
        max_length=args.max_length,
    )

    support_threshold, validation_support_metrics, validation_gated_summary, support_sweep = _search_support_threshold(
        validation_scored_predictions,
        validation_references,
        threshold_min=args.support_threshold_min,
        threshold_max=args.support_threshold_max,
        threshold_step=args.support_threshold_step,
        support_match_f1_threshold=args.match_f1_threshold,
    )

    finalized_dev_predictions = _finalize_support_predictions(
        dev_scored_predictions,
        dev_references,
        support_threshold=support_threshold,
        support_match_f1_threshold=args.match_f1_threshold,
    )
    raw_gated_dev_metrics = compute_stage1_metrics(
        _gate_predictions_with_support(finalized_dev_predictions),
        dev_references,
    )
    raw_support_dev_metrics = compute_stage2_support_metrics(
        finalized_dev_predictions,
        dev_references,
        support_threshold=support_threshold,
        support_match_f1_threshold=args.match_f1_threshold,
    )

    support_validation_records = _collect_support_confidence_records(
        validation_scored_predictions,
        validation_references,
        match_f1_threshold=args.match_f1_threshold,
    )
    support_dev_records = _collect_support_confidence_records(
        dev_scored_predictions,
        dev_references,
        match_f1_threshold=args.match_f1_threshold,
    )

    support_temperature = fit_temperature_scaler(
        [record["score"] for record in support_validation_records],
        [record["label"] for record in support_validation_records],
        temperature_min=args.support_temperature_min,
        temperature_max=args.support_temperature_max,
        temperature_step=args.support_temperature_step,
        num_bins=args.calibration_bins,
    )

    validation_qa_raw_metrics = _summarize_records(
        qa_validation_records,
        temperature=1.0,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )
    validation_qa_calibrated_metrics = _summarize_records(
        qa_validation_records,
        temperature=qa_temperature,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )
    dev_qa_raw_metrics = _summarize_records(
        qa_dev_records,
        temperature=1.0,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )
    dev_qa_calibrated_metrics = _summarize_records(
        qa_dev_records,
        temperature=qa_temperature,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )
    validation_support_raw_metrics = _summarize_records(
        support_validation_records,
        temperature=1.0,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )
    validation_support_calibrated_metrics = _summarize_records(
        support_validation_records,
        temperature=support_temperature,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )
    dev_support_raw_metrics = _summarize_records(
        support_dev_records,
        temperature=1.0,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )
    dev_support_calibrated_metrics = _summarize_records(
        support_dev_records,
        temperature=support_temperature,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )

    reliability_plot_paths: dict[str, str] = {}
    for plot_key, title, raw_metrics, calibrated_metrics in (
        (
            "qa_dev",
            "Stage 3 QA Reliability (Dev)",
            dev_qa_raw_metrics,
            dev_qa_calibrated_metrics,
        ),
        (
            "support_dev",
            "Stage 3 Support Reliability (Dev)",
            dev_support_raw_metrics,
            dev_support_calibrated_metrics,
        ),
    ):
        plot_path = args.output_path.parent / f"{plot_key}_reliability.png"
        saved = _save_reliability_plot(
            plot_path,
            title=title,
            raw_bins=raw_metrics["reliability_bins"],
            calibrated_bins=calibrated_metrics["reliability_bins"],
        )
        if saved:
            reliability_plot_paths[plot_key] = str(plot_path)

    output = {
        "stage": "stage3-calibration-eval",
        "qa_mode": args.qa_mode,
        "qa_model_path": str(args.qa_model_path),
        "verifier_model_path": str(args.verifier_model_path),
        "qa_selected_threshold": qa_threshold,
        "support_selected_threshold": support_threshold,
        "match_f1_threshold": args.match_f1_threshold,
        "qa_temperature": qa_temperature,
        "support_temperature": support_temperature,
        "qa_validation_metrics": qa_validation_metrics,
        "qa_threshold_sweep": qa_threshold_sweep,
        "support_threshold_sweep": support_sweep,
        "reliability_plot_paths": reliability_plot_paths,
        "validation_support_reference_metrics": validation_support_metrics,
        "validation_gated_reference_metrics": validation_gated_summary,
        "dev_gated_reference_metrics": raw_gated_dev_metrics,
        "dev_support_reference_metrics": raw_support_dev_metrics,
        "validation_qa_raw_metrics": validation_qa_raw_metrics,
        "validation_qa_calibrated_metrics": validation_qa_calibrated_metrics,
        "dev_qa_raw_metrics": dev_qa_raw_metrics,
        "dev_qa_calibrated_metrics": dev_qa_calibrated_metrics,
        "validation_support_raw_metrics": validation_support_raw_metrics,
        "validation_support_calibrated_metrics": validation_support_calibrated_metrics,
        "dev_support_raw_metrics": dev_support_raw_metrics,
        "dev_support_calibrated_metrics": dev_support_calibrated_metrics,
    }
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
