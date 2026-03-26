"""Stage 4 fixed unsupported-confidence controller evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from keelnet.calibration import _qa_decision_score, logit_probabilities, sigmoid_scores
from keelnet.config import (
    DEFAULT_CONTROL_ALPHA_MAX,
    DEFAULT_CONTROL_ALPHA_MIN,
    DEFAULT_CONTROL_ALPHA_STEP,
    DEFAULT_CONTROL_JOINT_THRESHOLD_MAX,
    DEFAULT_CONTROL_JOINT_THRESHOLD_MIN,
    DEFAULT_CONTROL_JOINT_THRESHOLD_STEP,
    DEFAULT_CONTROL_MAX_UNSUPPORTED_ANSWER_RATE,
    DEFAULT_CONTROL_QA_THRESHOLD_MAX,
    DEFAULT_CONTROL_QA_THRESHOLD_MIN,
    DEFAULT_CONTROL_QA_THRESHOLD_STEP,
    DEFAULT_CONTROL_SUPPORT_THRESHOLD_MAX,
    DEFAULT_CONTROL_SUPPORT_THRESHOLD_MIN,
    DEFAULT_CONTROL_SUPPORT_THRESHOLD_STEP,
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_N_BEST_SIZE,
    DEFAULT_SEED,
    DEFAULT_SUPPORT_MATCH_F1,
    DEFAULT_VALIDATION_SIZE,
    RUN_MODE_ABSTAIN,
    RUN_MODES,
)
from keelnet.metrics import compute_answer_support_mix, compute_stage1_metrics


@dataclass(frozen=True)
class ControlConfig:
    support_threshold: float
    qa_threshold: float
    joint_threshold: float
    alpha: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--qa-model-path", type=Path, default=None)
    parser.add_argument("--qa-mode", choices=RUN_MODES, default=None)
    parser.add_argument("--verifier-model-path", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)
    parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_CONTROL_MAX_UNSUPPORTED_ANSWER_RATE,
    )
    parser.add_argument(
        "--support-threshold-min",
        type=float,
        default=DEFAULT_CONTROL_SUPPORT_THRESHOLD_MIN,
    )
    parser.add_argument(
        "--support-threshold-max",
        type=float,
        default=DEFAULT_CONTROL_SUPPORT_THRESHOLD_MAX,
    )
    parser.add_argument(
        "--support-threshold-step",
        type=float,
        default=DEFAULT_CONTROL_SUPPORT_THRESHOLD_STEP,
    )
    parser.add_argument(
        "--qa-threshold-min",
        type=float,
        default=DEFAULT_CONTROL_QA_THRESHOLD_MIN,
    )
    parser.add_argument(
        "--qa-threshold-max",
        type=float,
        default=DEFAULT_CONTROL_QA_THRESHOLD_MAX,
    )
    parser.add_argument(
        "--qa-threshold-step",
        type=float,
        default=DEFAULT_CONTROL_QA_THRESHOLD_STEP,
    )
    parser.add_argument(
        "--joint-threshold-min",
        type=float,
        default=DEFAULT_CONTROL_JOINT_THRESHOLD_MIN,
    )
    parser.add_argument(
        "--joint-threshold-max",
        type=float,
        default=DEFAULT_CONTROL_JOINT_THRESHOLD_MAX,
    )
    parser.add_argument(
        "--joint-threshold-step",
        type=float,
        default=DEFAULT_CONTROL_JOINT_THRESHOLD_STEP,
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=DEFAULT_CONTROL_ALPHA_MIN,
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=DEFAULT_CONTROL_ALPHA_MAX,
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=DEFAULT_CONTROL_ALPHA_STEP,
    )
    return parser


def _frange(minimum: float, maximum: float, step: float) -> list[float]:
    values: list[float] = []
    current = float(minimum)
    while current <= maximum + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def _load_calibration_settings(
    calibration_path: Path,
    *,
    qa_model_path: Path | None,
    qa_mode: str | None,
    verifier_model_path: Path | None,
    match_f1_threshold: float,
) -> dict[str, Any]:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    resolved_qa_model_path = qa_model_path or Path(calibration["qa_model_path"])
    resolved_verifier_model_path = verifier_model_path or Path(calibration["verifier_model_path"])
    resolved_qa_mode = qa_mode or calibration.get("qa_mode", RUN_MODE_ABSTAIN)

    return {
        "calibration": calibration,
        "qa_model_path": Path(resolved_qa_model_path),
        "verifier_model_path": Path(resolved_verifier_model_path),
        "qa_mode": resolved_qa_mode,
        "qa_selected_threshold": float(calibration["qa_selected_threshold"]),
        "support_selected_threshold": float(calibration["support_selected_threshold"]),
        "qa_temperature": float(calibration["qa_temperature"]),
        "support_temperature": float(calibration["support_temperature"]),
        "match_f1_threshold": float(calibration.get("match_f1_threshold", match_f1_threshold)),
    }


def build_controller_scores(
    predictions: dict[str, dict[str, Any]],
    *,
    qa_selected_threshold: float,
    qa_temperature: float,
    support_temperature: float,
) -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    for example_id, prediction in predictions.items():
        qa_score = _qa_decision_score(prediction, qa_threshold=qa_selected_threshold)
        qa_confidence = float(sigmoid_scores([qa_score], temperature=qa_temperature)[0])
        support_score = float(prediction.get("support", {}).get("score", 0.0))
        support_logit = float(logit_probabilities([support_score])[0])
        support_confidence = float(sigmoid_scores([support_logit], temperature=support_temperature)[0])
        records[example_id] = {
            "qa_confidence": qa_confidence,
            "support_confidence": support_confidence,
        }
    return records


def apply_fixed_controller(
    predictions: dict[str, dict[str, Any]],
    controller_scores: dict[str, dict[str, float]],
    *,
    config: ControlConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    controlled_predictions: dict[str, dict[str, Any]] = {}
    reason_counts = {
        "qa_model": 0,
        "support_gate": 0,
        "qa_gate": 0,
        "joint_gate": 0,
        "answer": 0,
    }

    for example_id, prediction in predictions.items():
        decision = str(prediction.get("decision", "abstain")).lower()
        score_record = controller_scores.get(example_id, {"qa_confidence": 0.0, "support_confidence": 0.0})
        qa_confidence = float(score_record["qa_confidence"])
        support_confidence = float(score_record["support_confidence"])
        joint_score = float(config.alpha * support_confidence + (1.0 - config.alpha) * qa_confidence)

        support_pass = support_confidence >= config.support_threshold
        qa_pass = qa_confidence >= config.qa_threshold
        joint_pass = joint_score >= config.joint_threshold

        if decision != "answer":
            abstain_reason = "qa_model"
            final_decision = "abstain"
        elif not support_pass:
            abstain_reason = "support_gate"
            final_decision = "abstain"
        elif not qa_pass:
            abstain_reason = "qa_gate"
            final_decision = "abstain"
        elif not joint_pass:
            abstain_reason = "joint_gate"
            final_decision = "abstain"
        else:
            abstain_reason = None
            final_decision = "answer"

        if final_decision == "answer":
            reason_counts["answer"] += 1
        else:
            reason_counts[str(abstain_reason)] += 1

        controlled_predictions[example_id] = {
            **prediction,
            "decision": final_decision,
            "answer": str(prediction.get("answer", "")) if final_decision == "answer" else "",
            "control": {
                "qa_confidence": qa_confidence,
                "support_confidence": support_confidence,
                "joint_score": joint_score,
                "qa_pass": qa_pass,
                "support_pass": support_pass,
                "joint_pass": joint_pass,
                "selected_thresholds": asdict(config),
            },
        }
        if final_decision == "abstain":
            controlled_predictions[example_id]["abstain_reason"] = abstain_reason

    total_predictions = len(controlled_predictions)
    summary = {
        "answer_count": reason_counts["answer"],
        "answer_rate": 100.0 * reason_counts["answer"] / total_predictions if total_predictions else 0.0,
        "abstain_reason_counts": {key: value for key, value in reason_counts.items() if key != "answer"},
    }
    return controlled_predictions, summary


def _evaluate_controller_config(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    controller_scores: dict[str, dict[str, float]],
    *,
    config: ControlConfig,
    match_f1_threshold: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    controlled_predictions, summary = apply_fixed_controller(
        predictions,
        controller_scores,
        config=config,
    )
    stage1_metrics = compute_stage1_metrics(controlled_predictions, references)
    answer_mix = compute_answer_support_mix(
        controlled_predictions,
        references,
        match_f1_threshold=match_f1_threshold,
    )
    result = {
        **asdict(config),
        **stage1_metrics,
        **answer_mix,
        "constraint_satisfied": False,
        "summary": summary,
    }
    return result, controlled_predictions


def _select_best_controller(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    satisfied = [entry for entry in sweep if entry["constraint_satisfied"]]
    if satisfied:
        return max(
            satisfied,
            key=lambda item: (
                item["overall_f1"],
                item["answerable_f1"],
                item["supported_answer_rate"],
                -item["unsupported_answer_rate"],
            ),
        )

    return max(
        sweep,
        key=lambda item: (
            -item["unsupported_answer_rate"],
            item["overall_f1"],
            item["answerable_f1"],
            item["supported_answer_rate"],
        ),
    )


def search_control_config(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    controller_scores: dict[str, dict[str, float]],
    *,
    max_unsupported_answer_rate: float,
    support_threshold_min: float,
    support_threshold_max: float,
    support_threshold_step: float,
    qa_threshold_min: float,
    qa_threshold_max: float,
    qa_threshold_step: float,
    joint_threshold_min: float,
    joint_threshold_max: float,
    joint_threshold_step: float,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    match_f1_threshold: float,
) -> tuple[ControlConfig, dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sweep: list[dict[str, Any]] = []
    predictions_by_signature: dict[tuple[float, float, float, float], dict[str, dict[str, Any]]] = {}

    for support_threshold in _frange(support_threshold_min, support_threshold_max, support_threshold_step):
        for qa_threshold in _frange(qa_threshold_min, qa_threshold_max, qa_threshold_step):
            for joint_threshold in _frange(joint_threshold_min, joint_threshold_max, joint_threshold_step):
                for alpha in _frange(alpha_min, alpha_max, alpha_step):
                    config = ControlConfig(
                        support_threshold=support_threshold,
                        qa_threshold=qa_threshold,
                        joint_threshold=joint_threshold,
                        alpha=alpha,
                    )
                    result, controlled_predictions = _evaluate_controller_config(
                        predictions,
                        references,
                        controller_scores,
                        config=config,
                        match_f1_threshold=match_f1_threshold,
                    )
                    result["constraint_satisfied"] = (
                        result["unsupported_answer_rate"] <= max_unsupported_answer_rate
                    )
                    sweep.append(result)
                    predictions_by_signature[(support_threshold, qa_threshold, joint_threshold, alpha)] = controlled_predictions

    best_entry = _select_best_controller(sweep)
    best_config = ControlConfig(
        support_threshold=float(best_entry["support_threshold"]),
        qa_threshold=float(best_entry["qa_threshold"]),
        joint_threshold=float(best_entry["joint_threshold"]),
        alpha=float(best_entry["alpha"]),
    )
    best_predictions = predictions_by_signature[
        (
            best_config.support_threshold,
            best_config.qa_threshold,
            best_config.joint_threshold,
            best_config.alpha,
        )
    ]
    return best_config, best_entry, best_predictions, sweep


def main() -> None:
    args = build_parser().parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    from keelnet.data import build_reference_index, load_stage1_splits
    from keelnet.evaluate import prepare_qa_eval_artifacts, predict_raw_qa_outputs
    from keelnet.hf_compat import trainer_processing_kwargs
    from keelnet.postprocess import postprocess_qa_predictions
    from keelnet.verify import _finalize_support_predictions, _gate_predictions_with_support, _score_verifier_predictions
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    settings = _load_calibration_settings(
        args.calibration_path,
        qa_model_path=args.qa_model_path,
        qa_mode=args.qa_mode,
        verifier_model_path=args.verifier_model_path,
        match_f1_threshold=args.match_f1_threshold,
    )

    stage1_splits = load_stage1_splits(
        validation_size=args.validation_size,
        seed=args.seed,
        answer_only_train=False,
        max_eval_samples=args.max_eval_samples,
    )

    qa_tokenizer = AutoTokenizer.from_pretrained(settings["qa_model_path"], use_fast=True)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(settings["qa_model_path"])
    qa_trainer = Trainer(
        model=qa_model,
        args=TrainingArguments(
            output_dir=str(args.output_path.parent / "tmp-qa-control"),
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

    allow_abstain = settings["qa_mode"] == RUN_MODE_ABSTAIN
    validation_qa_predictions = postprocess_qa_predictions(
        validation_examples,
        validation_features,
        validation_raw_predictions,
        allow_abstain=allow_abstain,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=settings["qa_selected_threshold"],
    )
    dev_qa_predictions = postprocess_qa_predictions(
        dev_examples,
        dev_features,
        dev_raw_predictions,
        allow_abstain=allow_abstain,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=settings["qa_selected_threshold"],
    )

    base_validation_metrics = compute_stage1_metrics(validation_qa_predictions, validation_references)
    base_dev_metrics = compute_stage1_metrics(dev_qa_predictions, dev_references)

    verifier_tokenizer = AutoTokenizer.from_pretrained(settings["verifier_model_path"], use_fast=True)
    verifier_model = AutoModelForSequenceClassification.from_pretrained(settings["verifier_model_path"])
    verifier_trainer = Trainer(
        model=verifier_model,
        args=TrainingArguments(
            output_dir=str(args.output_path.parent / "tmp-verifier-control"),
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

    validation_stage2_predictions = _finalize_support_predictions(
        validation_scored_predictions,
        validation_references,
        support_threshold=settings["support_selected_threshold"],
        support_match_f1_threshold=settings["match_f1_threshold"],
    )
    dev_stage2_predictions = _finalize_support_predictions(
        dev_scored_predictions,
        dev_references,
        support_threshold=settings["support_selected_threshold"],
        support_match_f1_threshold=settings["match_f1_threshold"],
    )

    stage2_gated_validation_predictions = _gate_predictions_with_support(validation_stage2_predictions)
    stage2_gated_dev_predictions = _gate_predictions_with_support(dev_stage2_predictions)
    stage2_gated_validation_metrics = compute_stage1_metrics(
        stage2_gated_validation_predictions,
        validation_references,
    )
    stage2_gated_dev_metrics = compute_stage1_metrics(
        stage2_gated_dev_predictions,
        dev_references,
    )
    stage2_gated_validation_mix = compute_answer_support_mix(
        stage2_gated_validation_predictions,
        validation_references,
        match_f1_threshold=settings["match_f1_threshold"],
    )
    stage2_gated_dev_mix = compute_answer_support_mix(
        stage2_gated_dev_predictions,
        dev_references,
        match_f1_threshold=settings["match_f1_threshold"],
    )

    validation_controller_scores = build_controller_scores(
        validation_scored_predictions,
        qa_selected_threshold=settings["qa_selected_threshold"],
        qa_temperature=settings["qa_temperature"],
        support_temperature=settings["support_temperature"],
    )
    dev_controller_scores = build_controller_scores(
        dev_scored_predictions,
        qa_selected_threshold=settings["qa_selected_threshold"],
        qa_temperature=settings["qa_temperature"],
        support_temperature=settings["support_temperature"],
    )

    best_config, best_validation_entry, best_validation_predictions, control_sweep = search_control_config(
        validation_scored_predictions,
        validation_references,
        validation_controller_scores,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
        support_threshold_min=args.support_threshold_min,
        support_threshold_max=args.support_threshold_max,
        support_threshold_step=args.support_threshold_step,
        qa_threshold_min=args.qa_threshold_min,
        qa_threshold_max=args.qa_threshold_max,
        qa_threshold_step=args.qa_threshold_step,
        joint_threshold_min=args.joint_threshold_min,
        joint_threshold_max=args.joint_threshold_max,
        joint_threshold_step=args.joint_threshold_step,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        match_f1_threshold=settings["match_f1_threshold"],
    )
    dev_control_predictions, dev_control_summary = apply_fixed_controller(
        dev_scored_predictions,
        dev_controller_scores,
        config=best_config,
    )
    dev_control_metrics = compute_stage1_metrics(dev_control_predictions, dev_references)
    dev_control_mix = compute_answer_support_mix(
        dev_control_predictions,
        dev_references,
        match_f1_threshold=settings["match_f1_threshold"],
    )

    output = {
        "stage": "stage4-fixed-control-eval",
        "calibration_path": str(args.calibration_path),
        "qa_model_path": str(settings["qa_model_path"]),
        "verifier_model_path": str(settings["verifier_model_path"]),
        "qa_mode": settings["qa_mode"],
        "qa_selected_threshold": settings["qa_selected_threshold"],
        "support_selected_threshold": settings["support_selected_threshold"],
        "qa_temperature": settings["qa_temperature"],
        "support_temperature": settings["support_temperature"],
        "match_f1_threshold": settings["match_f1_threshold"],
        "max_unsupported_answer_rate": args.max_unsupported_answer_rate,
        "selected_config": asdict(best_config),
        "selected_validation_metrics": best_validation_entry,
        "base_validation_metrics": base_validation_metrics,
        "base_dev_metrics": base_dev_metrics,
        "stage2_gated_validation_metrics": stage2_gated_validation_metrics,
        "stage2_gated_dev_metrics": stage2_gated_dev_metrics,
        "stage2_gated_validation_mix": stage2_gated_validation_mix,
        "stage2_gated_dev_mix": stage2_gated_dev_mix,
        "control_validation_metrics": compute_stage1_metrics(best_validation_predictions, validation_references),
        "control_dev_metrics": dev_control_metrics,
        "control_validation_mix": compute_answer_support_mix(
            best_validation_predictions,
            validation_references,
            match_f1_threshold=settings["match_f1_threshold"],
        ),
        "control_dev_mix": dev_control_mix,
        "control_dev_summary": dev_control_summary,
        "validation_control_sweep": control_sweep,
        "dev_predictions": dev_control_predictions,
    }
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
