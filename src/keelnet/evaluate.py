"""Evaluation entrypoint for the Stage 1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

from keelnet.config import (
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_N_BEST_SIZE,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_SIZE,
    RUN_MODE_ABSTAIN,
    RUN_MODES,
)
from keelnet.data import build_reference_index, load_stage1_splits, prepare_eval_features
from keelnet.hf_compat import trainer_processing_kwargs
from keelnet.metrics import compute_stage1_metrics
from keelnet.postprocess import postprocess_qa_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mode", choices=RUN_MODES, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--threshold-min", type=float, default=-5.0)
    parser.add_argument("--threshold-max", type=float, default=5.0)
    parser.add_argument("--threshold-step", type=float, default=0.5)
    return parser


def _predict_raw(trainer: Trainer, eval_dataset):
    output = trainer.predict(eval_dataset)
    if isinstance(output.predictions, tuple):
        return tuple(np.asarray(item) for item in output.predictions[:2])
    raise ValueError("Expected tuple predictions with start/end logits.")


def _prepare_eval_artifacts(raw_dataset, tokenizer, max_length: int, doc_stride: int):
    eval_features = raw_dataset.map(
        lambda batch: prepare_eval_features(batch, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing evaluation data",
    )
    eval_dataset_for_model = eval_features.remove_columns(["example_id", "offset_mapping", "cls_index"])
    return eval_features, eval_dataset_for_model


def _search_threshold(
    examples,
    features,
    raw_predictions,
    references,
    *,
    n_best_size: int,
    max_answer_length: int,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    sweep: list[dict[str, float]] = []
    current = threshold_min
    while current <= threshold_max + 1e-9:
        predictions = postprocess_qa_predictions(
            examples,
            features,
            raw_predictions,
            allow_abstain=True,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            null_score_diff_threshold=current,
        )
        metrics = compute_stage1_metrics(predictions, references)
        sweep.append({"threshold": round(float(current), 10), **metrics})
        current += threshold_step

    best_entry = max(sweep, key=lambda item: (item["abstain_f1"], item["answerable_f1"]))
    best_threshold = float(best_entry["threshold"])
    best_metrics = {key: value for key, value in best_entry.items() if key != "threshold"}
    return best_threshold, best_metrics, sweep


def main() -> None:
    args = build_parser().parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = load_stage1_splits(
        validation_size=args.validation_size,
        seed=args.seed,
        answer_only_train=False,
        max_eval_samples=args.max_eval_samples,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_path.parent / "tmp-eval"),
            per_device_eval_batch_size=args.eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, tokenizer),
    )

    validation_examples = splits["validation"]
    dev_examples = splits["dev"]
    validation_features, validation_model_dataset = _prepare_eval_artifacts(
        validation_examples,
        tokenizer,
        args.max_length,
        args.doc_stride,
    )
    dev_features, dev_model_dataset = _prepare_eval_artifacts(
        dev_examples,
        tokenizer,
        args.max_length,
        args.doc_stride,
    )

    validation_raw_predictions = _predict_raw(trainer, validation_model_dataset)
    dev_raw_predictions = _predict_raw(trainer, dev_model_dataset)

    validation_references = build_reference_index(validation_examples)
    dev_references = build_reference_index(dev_examples)

    threshold = 0.0
    validation_metrics: dict[str, float] | None = None
    threshold_sweep: dict[str, list[dict[str, float]]] | None = None
    allow_abstain = args.mode == RUN_MODE_ABSTAIN
    if allow_abstain:
        threshold, validation_metrics, validation_sweep = _search_threshold(
            validation_examples,
            validation_features,
            validation_raw_predictions,
            validation_references,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        _, _, dev_sweep = _search_threshold(
            dev_examples,
            dev_features,
            dev_raw_predictions,
            dev_references,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        threshold_sweep = {
            "validation": validation_sweep,
            "dev": dev_sweep,
        }

    dev_predictions = postprocess_qa_predictions(
        dev_examples,
        dev_features,
        dev_raw_predictions,
        allow_abstain=allow_abstain,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=threshold,
    )
    dev_metrics = compute_stage1_metrics(dev_predictions, dev_references)

    output = {
        "mode": args.mode,
        "model_path": str(args.model_path),
        "selected_threshold": threshold,
        "validation_metrics": validation_metrics,
        "threshold_sweep": threshold_sweep,
        "dev_metrics": dev_metrics,
        "dev_predictions": dev_predictions,
    }
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
