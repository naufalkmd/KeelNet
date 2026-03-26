"""Stage 2 support-verification training and evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

from keelnet.config import (
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_BEST_SIZE,
    DEFAULT_SEED,
    DEFAULT_SUPPORT_MATCH_F1,
    DEFAULT_SUPPORT_THRESHOLD_MAX,
    DEFAULT_SUPPORT_THRESHOLD_MIN,
    DEFAULT_SUPPORT_THRESHOLD_STEP,
    DEFAULT_VALIDATION_SIZE,
    DEFAULT_VERIFICATION_NEGATIVES_PER_ANSWERABLE,
    DEFAULT_VERIFICATION_NEGATIVES_PER_UNANSWERABLE,
    RUN_MODE_ABSTAIN,
    RUN_MODES,
)
from keelnet.data import (
    build_reference_index,
    build_stage2_verification_splits,
    load_stage1_splits,
    prepare_verification_features,
)
from keelnet.evaluate import (
    predict_raw_qa_outputs,
    prepare_qa_eval_artifacts,
    search_abstain_threshold,
)
from keelnet.hf_compat import trainer_processing_kwargs
from keelnet.metrics import compute_stage1_metrics, compute_stage2_support_metrics, is_supported_answer
from keelnet.postprocess import postprocess_qa_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Stage 2 support verifier.")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    train_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--num-train-epochs", type=float, default=3.0)
    train_parser.add_argument("--train-batch-size", type=int, default=16)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.0)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-eval-samples", type=int, default=None)
    train_parser.add_argument(
        "--negatives-per-answerable",
        type=int,
        default=DEFAULT_VERIFICATION_NEGATIVES_PER_ANSWERABLE,
    )
    train_parser.add_argument(
        "--negatives-per-unanswerable",
        type=int,
        default=DEFAULT_VERIFICATION_NEGATIVES_PER_UNANSWERABLE,
    )
    train_parser.add_argument("--logging-steps", type=int, default=50)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained support verifier on top of Stage 1 QA predictions.",
    )
    eval_parser.add_argument("--qa-model-path", type=Path, required=True)
    eval_parser.add_argument("--qa-mode", choices=RUN_MODES, required=True)
    eval_parser.add_argument("--verifier-model-path", type=Path, required=True)
    eval_parser.add_argument("--output-path", type=Path, required=True)
    eval_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    eval_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    eval_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    eval_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    eval_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    eval_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    eval_parser.add_argument("--eval-batch-size", type=int, default=16)
    eval_parser.add_argument("--max-eval-samples", type=int, default=None)
    eval_parser.add_argument("--qa-threshold-min", type=float, default=-5.0)
    eval_parser.add_argument("--qa-threshold-max", type=float, default=5.0)
    eval_parser.add_argument("--qa-threshold-step", type=float, default=0.5)
    eval_parser.add_argument(
        "--support-threshold-min",
        type=float,
        default=DEFAULT_SUPPORT_THRESHOLD_MIN,
    )
    eval_parser.add_argument(
        "--support-threshold-max",
        type=float,
        default=DEFAULT_SUPPORT_THRESHOLD_MAX,
    )
    eval_parser.add_argument(
        "--support-threshold-step",
        type=float,
        default=DEFAULT_SUPPORT_THRESHOLD_STEP,
    )
    eval_parser.add_argument(
        "--support-match-f1-threshold",
        type=float,
        default=DEFAULT_SUPPORT_MATCH_F1,
    )

    return parser


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _train_verifier(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    stage1_splits = load_stage1_splits(
        validation_size=args.validation_size,
        seed=args.seed,
        answer_only_train=False,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    verification_splits = build_stage2_verification_splits(
        stage1_splits,
        seed=args.seed,
        negatives_per_answerable=args.negatives_per_answerable,
        negatives_per_unanswerable=args.negatives_per_unanswerable,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = verification_splits["train"].map(
        lambda batch: prepare_verification_features(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=verification_splits["train"].column_names,
        desc="Tokenizing verification training data",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DefaultDataCollator(),
        **trainer_processing_kwargs(Trainer, tokenizer),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    run_metadata = {
        "stage": "stage2-verifier-train",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "validation_size": args.validation_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "warmup_ratio": args.warmup_ratio,
        "negatives_per_answerable": args.negatives_per_answerable,
        "negatives_per_unanswerable": args.negatives_per_unanswerable,
        "raw_train_examples": len(stage1_splits["train"]),
        "raw_validation_examples": len(stage1_splits["validation"]),
        "raw_dev_examples": len(stage1_splits["dev"]),
        "verification_train_examples": len(verification_splits["train"]),
        "verification_validation_examples": len(verification_splits["validation"]),
        "verification_dev_examples": len(verification_splits["dev"]),
    }
    metadata_path = args.output_dir / "run_config.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")


def _build_verifier_records(examples, qa_predictions: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for example in examples:
        example_id = str(example["id"])
        prediction = qa_predictions.get(example_id, {"decision": "abstain", "answer": ""})
        if str(prediction.get("decision", "abstain")).lower() != "answer":
            continue
        records.append(
            {
                "id": example_id,
                "question": str(example["question"]),
                "context": str(example["context"]),
                "candidate_answer": str(prediction.get("answer", "")),
            }
        )
    return records


def _score_verifier_predictions(
    trainer: Trainer,
    tokenizer,
    examples,
    qa_predictions: dict[str, dict[str, Any]],
    *,
    max_length: int,
) -> dict[str, dict[str, Any]]:
    records = _build_verifier_records(examples, qa_predictions)
    support_scores: dict[str, float] = {}
    if records:
        verifier_dataset = Dataset.from_list(records)
        verifier_features = verifier_dataset.map(
            lambda batch: prepare_verification_features(batch, tokenizer, max_length),
            batched=True,
            remove_columns=verifier_dataset.column_names,
            desc="Tokenizing verification evaluation data",
        )
        output = trainer.predict(verifier_features)
        probabilities = _softmax(np.asarray(output.predictions))
        for record, score in zip(records, probabilities[:, 1], strict=False):
            support_scores[str(record["id"])] = float(score)

    scored_predictions: dict[str, dict[str, Any]] = {}
    for example in examples:
        example_id = str(example["id"])
        prediction = qa_predictions.get(example_id, {"decision": "abstain", "answer": "", "scores": {}})
        support_score = support_scores.get(example_id, 0.0)
        support_info = {"score": support_score}
        scored_predictions[example_id] = {
            **prediction,
            "support": support_info,
        }
    return scored_predictions


def _finalize_support_predictions(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    support_threshold: float,
    support_match_f1_threshold: float,
) -> dict[str, dict[str, Any]]:
    finalized_predictions: dict[str, dict[str, Any]] = {}
    for example_id, prediction in predictions.items():
        reference = references[example_id]
        decision = str(prediction.get("decision", "abstain")).lower()
        answer_text = str(prediction.get("answer", ""))
        support_info = dict(prediction.get("support", {}))
        support_score = float(support_info.get("score", 0.0))

        gold_supported: str | None = None
        predicted_supported = False
        if decision == "answer":
            predicted_supported = support_score >= support_threshold
            gold_supported = (
                "supported"
                if bool(reference.get("is_answerable", False))
                and is_supported_answer(
                    answer_text,
                    list(reference.get("answers", [])),
                    match_f1_threshold=support_match_f1_threshold,
                )
                else "unsupported"
            )

        support_info.update(
            {
                "label": "supported" if predicted_supported else "unsupported",
                "gold_label": gold_supported,
                "threshold": support_threshold,
            }
        )
        finalized_predictions[example_id] = {
            **prediction,
            "support": support_info,
            "gated_decision": "answer" if decision == "answer" and predicted_supported else "abstain",
        }

    return finalized_predictions


def _gate_predictions_with_support(
    predictions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    gated_predictions: dict[str, dict[str, Any]] = {}
    for example_id, prediction in predictions.items():
        gated_decision = str(prediction.get("gated_decision", prediction.get("decision", "abstain"))).lower()
        if gated_decision == "answer":
            gated_predictions[example_id] = {
                "decision": "answer",
                "answer": str(prediction.get("answer", "")),
                "scores": dict(prediction.get("scores", {})),
                "support": dict(prediction.get("support", {})),
            }
            continue

        gated_predictions[example_id] = {
            "decision": "abstain",
            "answer": "",
            "scores": dict(prediction.get("scores", {})),
            "support": dict(prediction.get("support", {})),
            "abstain_reason": (
                "support_verifier"
                if str(prediction.get("decision", "abstain")).lower() == "answer"
                else "qa_model"
            ),
        }
    return gated_predictions


def _search_support_threshold(
    validation_predictions: dict[str, dict[str, Any]],
    validation_references: dict[str, dict[str, Any]],
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    support_match_f1_threshold: float,
) -> tuple[float, dict[str, float], dict[str, float], list[dict[str, float]]]:
    sweep: list[dict[str, float]] = []
    current = threshold_min
    while current <= threshold_max + 1e-9:
        finalized_predictions = _finalize_support_predictions(
            validation_predictions,
            validation_references,
            support_threshold=current,
            support_match_f1_threshold=support_match_f1_threshold,
        )
        support_metrics = compute_stage2_support_metrics(
            finalized_predictions,
            validation_references,
            support_threshold=current,
            support_match_f1_threshold=support_match_f1_threshold,
        )
        gated_metrics = compute_stage1_metrics(
            _gate_predictions_with_support(finalized_predictions),
            validation_references,
        )
        sweep.append(
            {
                "support_threshold": round(float(current), 10),
                **support_metrics,
                "gated_answerable_f1": gated_metrics["answerable_f1"],
                "gated_overall_f1": gated_metrics["overall_f1"],
                "gated_unsupported_answer_rate": gated_metrics["unsupported_answer_rate"],
            }
        )
        current += threshold_step

    best_entry = max(
        sweep,
        key=lambda item: (
            item["support_f1"],
            item["gated_overall_f1"],
            -item["gated_unsupported_answer_rate"],
        ),
    )
    best_threshold = float(best_entry["support_threshold"])
    best_support_metrics = {
        key: value
        for key, value in best_entry.items()
        if key
        not in {"support_threshold", "gated_answerable_f1", "gated_overall_f1", "gated_unsupported_answer_rate"}
    }
    best_gated_metrics = {
        "answerable_f1": best_entry["gated_answerable_f1"],
        "overall_f1": best_entry["gated_overall_f1"],
        "unsupported_answer_rate": best_entry["gated_unsupported_answer_rate"],
    }
    return best_threshold, best_support_metrics, best_gated_metrics, sweep


def _evaluate_verifier(args: argparse.Namespace) -> None:
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
            output_dir=str(args.output_path.parent / "tmp-qa-eval"),
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
        qa_threshold_sweep = {
            "validation": qa_validation_sweep,
            "dev": qa_dev_sweep,
        }

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

    base_validation_metrics = compute_stage1_metrics(validation_qa_predictions, validation_references)
    base_dev_metrics = compute_stage1_metrics(dev_qa_predictions, dev_references)

    verifier_tokenizer = AutoTokenizer.from_pretrained(args.verifier_model_path, use_fast=True)
    verifier_model = AutoModelForSequenceClassification.from_pretrained(args.verifier_model_path)
    verifier_trainer = Trainer(
        model=verifier_model,
        args=TrainingArguments(
            output_dir=str(args.output_path.parent / "tmp-verifier-eval"),
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
        support_match_f1_threshold=args.support_match_f1_threshold,
    )

    finalized_validation_predictions = _finalize_support_predictions(
        validation_scored_predictions,
        validation_references,
        support_threshold=support_threshold,
        support_match_f1_threshold=args.support_match_f1_threshold,
    )
    finalized_dev_predictions = _finalize_support_predictions(
        dev_scored_predictions,
        dev_references,
        support_threshold=support_threshold,
        support_match_f1_threshold=args.support_match_f1_threshold,
    )

    dev_support_metrics = compute_stage2_support_metrics(
        finalized_dev_predictions,
        dev_references,
        support_threshold=support_threshold,
        support_match_f1_threshold=args.support_match_f1_threshold,
    )
    gated_validation_metrics = compute_stage1_metrics(
        _gate_predictions_with_support(finalized_validation_predictions),
        validation_references,
    )
    gated_dev_metrics = compute_stage1_metrics(
        _gate_predictions_with_support(finalized_dev_predictions),
        dev_references,
    )

    output = {
        "stage": "stage2-verifier-eval",
        "qa_mode": args.qa_mode,
        "qa_model_path": str(args.qa_model_path),
        "verifier_model_path": str(args.verifier_model_path),
        "qa_selected_threshold": qa_threshold,
        "support_selected_threshold": support_threshold,
        "support_match_f1_threshold": args.support_match_f1_threshold,
        "qa_validation_metrics": qa_validation_metrics,
        "qa_threshold_sweep": qa_threshold_sweep,
        "base_validation_metrics": base_validation_metrics,
        "base_dev_metrics": base_dev_metrics,
        "validation_support_metrics": validation_support_metrics,
        "dev_support_metrics": dev_support_metrics,
        "validation_gated_summary": validation_gated_summary,
        "gated_validation_metrics": gated_validation_metrics,
        "gated_dev_metrics": gated_dev_metrics,
        "support_threshold_sweep": support_sweep,
        "dev_predictions": finalized_dev_predictions,
    }
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _train_verifier(args)
        return
    if args.command == "evaluate":
        _evaluate_verifier(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
