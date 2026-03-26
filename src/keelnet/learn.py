"""Stage 5 support-constrained learning training and evaluation entrypoint."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import ModelOutput

from keelnet.config import (
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_BEST_SIZE,
    DEFAULT_SEED,
    DEFAULT_STAGE5_KEEP_LOSS_WEIGHT,
    DEFAULT_STAGE5_KEEP_NEGATIVE_WEIGHT,
    DEFAULT_STAGE5_KEEP_POSITIVE_WEIGHT,
    DEFAULT_STAGE5_KEEP_THRESHOLD_MAX,
    DEFAULT_STAGE5_KEEP_THRESHOLD_MIN,
    DEFAULT_STAGE5_KEEP_THRESHOLD_STEP,
    DEFAULT_STAGE5_MAX_UNSUPPORTED_ANSWER_RATE,
    DEFAULT_STAGE5_SUPPORT_LOSS_WEIGHT,
    DEFAULT_SUPPORT_MATCH_F1,
    DEFAULT_VALIDATION_SIZE,
)
from keelnet.data import (
    build_reference_index,
    load_stage1_splits,
    prepare_eval_features,
    prepare_stage5_train_features,
)
from keelnet.hf_compat import trainer_processing_kwargs
from keelnet.metrics import compute_answer_support_mix, compute_stage1_metrics

MODEL_STATE_FILENAME = "stage5_model.pt"
MODEL_CONFIG_FILENAME = "stage5_model_config.json"


@dataclass
class SupportConstrainedQAOutput(ModelOutput):
    """Outputs for the Stage 5 learner."""

    loss: torch.Tensor | None = None
    start_logits: torch.Tensor | None = None
    end_logits: torch.Tensor | None = None
    abstain_logits: torch.Tensor | None = None
    support_logits: torch.Tensor | None = None


def keep_probability_from_logits(abstain_logit: float, support_logit: float) -> float:
    abstain_probability = 1.0 / (1.0 + np.exp(-float(abstain_logit)))
    support_probability = 1.0 / (1.0 + np.exp(-float(support_logit)))
    keep_probability = (1.0 - abstain_probability) * support_probability
    return float(min(1.0 - 1e-6, max(1e-6, keep_probability)))


class SupportConstrainedQAModel(nn.Module):
    """Shared-encoder QA model with abstention and support heads."""

    def __init__(
        self,
        model_name: str,
        *,
        keep_loss_weight: float,
        support_loss_weight: float,
        keep_positive_weight: float,
        keep_negative_weight: float,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.keep_loss_weight = float(keep_loss_weight)
        self.support_loss_weight = float(support_loss_weight)
        self.keep_positive_weight = float(keep_positive_weight)
        self.keep_negative_weight = float(keep_negative_weight)

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "dim", None)
        if hidden_size is None:
            raise ValueError("Could not determine encoder hidden size.")

        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.abstain_classifier = nn.Linear(hidden_size, 1)
        self.support_classifier = nn.Linear(hidden_size, 1)
        self._supports_token_type_ids = "token_type_ids" in inspect.signature(self.encoder.forward).parameters

    def _encode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        encoder_kwargs: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            encoder_kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None and self._supports_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        keep_labels: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
    ) -> SupportConstrainedQAOutput:
        if input_ids is None:
            raise ValueError("input_ids are required.")

        sequence_output = self._encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        cls_output = sequence_output[:, 0]
        abstain_logits = self.abstain_classifier(cls_output).squeeze(-1)
        support_logits = self.support_classifier(cls_output).squeeze(-1)

        loss = None
        if (
            start_positions is not None
            and end_positions is not None
            and keep_labels is not None
            and support_labels is not None
        ):
            keep_targets = keep_labels.float()
            support_targets = support_labels.float()

            start_loss = F.cross_entropy(start_logits, start_positions, reduction="none")
            end_loss = F.cross_entropy(end_logits, end_positions, reduction="none")
            positive_count = keep_targets.sum().clamp_min(1.0)
            span_loss = (((start_loss + end_loss) / 2.0) * keep_targets).sum() / positive_count

            abstain_probabilities = torch.sigmoid(abstain_logits)
            support_probabilities = torch.sigmoid(support_logits)
            keep_probabilities = ((1.0 - abstain_probabilities) * support_probabilities).clamp(1e-6, 1.0 - 1e-6)
            keep_loss = -(
                self.keep_positive_weight * keep_targets * torch.log(keep_probabilities)
                + self.keep_negative_weight * (1.0 - keep_targets) * torch.log(1.0 - keep_probabilities)
            ).mean()
            support_loss = F.binary_cross_entropy_with_logits(support_logits, support_targets)

            loss = span_loss + self.keep_loss_weight * keep_loss + self.support_loss_weight * support_loss

        return SupportConstrainedQAOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            abstain_logits=abstain_logits,
            support_logits=support_logits,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Stage 5 support-constrained learner.")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    train_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    train_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--num-train-epochs", type=float, default=3.0)
    train_parser.add_argument("--train-batch-size", type=int, default=8)
    train_parser.add_argument("--eval-batch-size", type=int, default=8)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.0)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-eval-samples", type=int, default=None)
    train_parser.add_argument("--keep-loss-weight", type=float, default=DEFAULT_STAGE5_KEEP_LOSS_WEIGHT)
    train_parser.add_argument("--support-loss-weight", type=float, default=DEFAULT_STAGE5_SUPPORT_LOSS_WEIGHT)
    train_parser.add_argument("--keep-positive-weight", type=float, default=DEFAULT_STAGE5_KEEP_POSITIVE_WEIGHT)
    train_parser.add_argument("--keep-negative-weight", type=float, default=DEFAULT_STAGE5_KEEP_NEGATIVE_WEIGHT)
    train_parser.add_argument("--logging-steps", type=int, default=50)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the Stage 5 support-constrained learner.")
    eval_parser.add_argument("--model-path", type=Path, required=True)
    eval_parser.add_argument("--output-path", type=Path, required=True)
    eval_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    eval_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    eval_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    eval_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    eval_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    eval_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    eval_parser.add_argument("--eval-batch-size", type=int, default=8)
    eval_parser.add_argument("--max-eval-samples", type=int, default=None)
    eval_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)
    eval_parser.add_argument("--keep-threshold-min", type=float, default=DEFAULT_STAGE5_KEEP_THRESHOLD_MIN)
    eval_parser.add_argument("--keep-threshold-max", type=float, default=DEFAULT_STAGE5_KEEP_THRESHOLD_MAX)
    eval_parser.add_argument("--keep-threshold-step", type=float, default=DEFAULT_STAGE5_KEEP_THRESHOLD_STEP)
    eval_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE5_MAX_UNSUPPORTED_ANSWER_RATE,
    )

    return parser


def _save_stage5_model(model: SupportConstrainedQAModel, output_dir: Path, tokenizer) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / MODEL_STATE_FILENAME)
    config_payload = {
        "model_name": model.model_name,
        "keep_loss_weight": model.keep_loss_weight,
        "support_loss_weight": model.support_loss_weight,
        "keep_positive_weight": model.keep_positive_weight,
        "keep_negative_weight": model.keep_negative_weight,
    }
    (output_dir / MODEL_CONFIG_FILENAME).write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(output_dir)


def load_stage5_model(model_path: Path) -> SupportConstrainedQAModel:
    config = json.loads((model_path / MODEL_CONFIG_FILENAME).read_text(encoding="utf-8"))
    model = SupportConstrainedQAModel(
        config["model_name"],
        keep_loss_weight=config["keep_loss_weight"],
        support_loss_weight=config["support_loss_weight"],
        keep_positive_weight=config["keep_positive_weight"],
        keep_negative_weight=config["keep_negative_weight"],
    )
    state_dict = torch.load(model_path / MODEL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def prepare_stage5_eval_artifacts(raw_dataset, tokenizer, max_length: int, doc_stride: int):
    eval_features = raw_dataset.map(
        lambda batch: prepare_eval_features(batch, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing evaluation data",
    )
    eval_dataset_for_model = eval_features.remove_columns(["example_id", "offset_mapping", "cls_index"])
    return eval_features, eval_dataset_for_model


def predict_stage5_raw_outputs(trainer: Trainer, eval_dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output = trainer.predict(eval_dataset)
    predictions = output.predictions
    if isinstance(predictions, tuple) and len(predictions) >= 4:
        return tuple(np.asarray(item) for item in predictions[:4])  # type: ignore[return-value]
    raise ValueError("Expected tuple predictions with start/end/abstain/support logits.")


def _best_feature_span(
    context: str,
    feature,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    *,
    n_best_size: int,
    max_answer_length: int,
) -> tuple[str, float]:
    offsets = feature["offset_mapping"]
    best_span_text = ""
    best_span_score = float("-inf")

    start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
    end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]
    for start_index in start_indexes:
        for end_index in end_indexes:
            if start_index >= len(offsets) or end_index >= len(offsets):
                continue
            if offsets[start_index] is None or offsets[end_index] is None:
                continue
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_length:
                continue

            start_char = offsets[start_index][0]
            end_char = offsets[end_index][1]
            if start_char is None or end_char is None:
                continue

            score = float(start_logits[start_index] + end_logits[end_index])
            if score > best_span_score:
                best_span_score = score
                best_span_text = context[start_char:end_char]

    return best_span_text, best_span_score


def postprocess_stage5_predictions(
    examples,
    features,
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    keep_threshold: float,
    n_best_size: int,
    max_answer_length: int,
) -> dict[str, dict[str, Any]]:
    all_start_logits, all_end_logits, all_abstain_logits, all_support_logits = predictions
    features_per_example: dict[str, list[int]] = defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_per_example[str(feature["example_id"])].append(feature_index)

    final_predictions: dict[str, dict[str, Any]] = {}
    for example in examples:
        example_id = str(example["id"])
        context = str(example["context"])

        best_feature_index: int | None = None
        best_span_text = ""
        best_span_score = float("-inf")
        for feature_index in features_per_example[example_id]:
            span_text, span_score = _best_feature_span(
                context,
                features[feature_index],
                all_start_logits[feature_index],
                all_end_logits[feature_index],
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
            )
            if span_score > best_span_score:
                best_feature_index = feature_index
                best_span_text = span_text
                best_span_score = span_score

        if best_feature_index is None or not best_span_text.strip():
            final_predictions[example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "best_span": best_span_score,
                    "keep_probability": 0.0,
                    "abstain_probability": 1.0,
                    "support_probability": 0.0,
                },
                "support": {"score": 0.0},
                "abstain_reason": "no_valid_span",
            }
            continue

        abstain_logit = float(all_abstain_logits[best_feature_index])
        support_logit = float(all_support_logits[best_feature_index])
        abstain_probability = float(1.0 / (1.0 + np.exp(-abstain_logit)))
        support_probability = float(1.0 / (1.0 + np.exp(-support_logit)))
        keep_probability = keep_probability_from_logits(abstain_logit, support_logit)

        decision = "answer" if keep_probability >= keep_threshold else "abstain"
        final_predictions[example_id] = {
            "decision": decision,
            "answer": best_span_text if decision == "answer" else "",
            "scores": {
                "best_span": best_span_score,
                "keep_probability": keep_probability,
                "abstain_probability": abstain_probability,
                "support_probability": support_probability,
            },
            "support": {"score": support_probability},
        }
        if decision == "abstain":
            final_predictions[example_id]["abstain_reason"] = "keep_gate"

    return final_predictions


def select_stage5_threshold_entry(
    sweep: list[dict[str, float]],
    *,
    max_unsupported_answer_rate: float,
) -> dict[str, float]:
    if not sweep:
        raise ValueError("Threshold sweep is empty.")

    constrained = [
        entry for entry in sweep if bool(entry.get("constraint_satisfied", False))
    ]
    pool = constrained or sweep
    return max(
        pool,
        key=lambda entry: (
            float(entry["overall_f1"]),
            float(entry["answerable_f1"]),
            -float(entry["unsupported_answer_rate"]),
            float(entry["supported_answer_rate"]),
        ),
    )


def search_keep_threshold(
    examples,
    features,
    raw_predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    references: dict[str, dict[str, Any]],
    *,
    n_best_size: int,
    max_answer_length: int,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    match_f1_threshold: float,
    max_unsupported_answer_rate: float,
) -> tuple[float, dict[str, float], dict[str, float], list[dict[str, float]]]:
    sweep: list[dict[str, float]] = []
    current = threshold_min
    while current <= threshold_max + 1e-9:
        predictions = postprocess_stage5_predictions(
            examples,
            features,
            raw_predictions,
            keep_threshold=current,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
        )
        metrics = compute_stage1_metrics(predictions, references)
        mix = compute_answer_support_mix(
            predictions,
            references,
            match_f1_threshold=match_f1_threshold,
        )
        sweep.append(
            {
                "threshold": round(float(current), 10),
                "constraint_satisfied": metrics["unsupported_answer_rate"] <= max_unsupported_answer_rate,
                **metrics,
                **mix,
            }
        )
        current += threshold_step

    best_entry = select_stage5_threshold_entry(
        sweep,
        max_unsupported_answer_rate=max_unsupported_answer_rate,
    )
    best_threshold = float(best_entry["threshold"])
    best_metrics = {
        key: value
        for key, value in best_entry.items()
        if key not in {"threshold", "constraint_satisfied", "answer_rate", "supported_answer_rate", "unsupported_among_answers_rate", "answered_count", "supported_answers_count", "unsupported_answers_count"}
    }
    best_mix = {
        "answer_rate": float(best_entry["answer_rate"]),
        "supported_answer_rate": float(best_entry["supported_answer_rate"]),
        "unsupported_among_answers_rate": float(best_entry["unsupported_among_answers_rate"]),
        "answered_count": float(best_entry["answered_count"]),
        "supported_answers_count": float(best_entry["supported_answers_count"]),
        "unsupported_answers_count": float(best_entry["unsupported_answers_count"]),
    }
    return best_threshold, best_metrics, best_mix, sweep


def _train_model(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    splits = load_stage1_splits(
        validation_size=args.validation_size,
        seed=args.seed,
        answer_only_train=False,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = SupportConstrainedQAModel(
        args.model_name,
        keep_loss_weight=args.keep_loss_weight,
        support_loss_weight=args.support_loss_weight,
        keep_positive_weight=args.keep_positive_weight,
        keep_negative_weight=args.keep_negative_weight,
    )

    train_dataset = splits["train"].map(
        lambda batch: prepare_stage5_train_features(batch, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=splits["train"].column_names,
        desc="Tokenizing Stage 5 training data",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer-tmp"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="no",
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

    _save_stage5_model(model, args.output_dir, tokenizer)
    run_metadata = {
        "stage": "stage5-support-constrained-train",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "doc_stride": args.doc_stride,
        "validation_size": args.validation_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "warmup_ratio": args.warmup_ratio,
        "keep_loss_weight": args.keep_loss_weight,
        "support_loss_weight": args.support_loss_weight,
        "keep_positive_weight": args.keep_positive_weight,
        "keep_negative_weight": args.keep_negative_weight,
        "train_examples": len(splits["train"]),
        "validation_examples": len(splits["validation"]),
        "dev_examples": len(splits["dev"]),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")


def _evaluate_model(args: argparse.Namespace) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    splits = load_stage1_splits(
        validation_size=args.validation_size,
        seed=args.seed,
        answer_only_train=False,
        max_eval_samples=args.max_eval_samples,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = load_stage5_model(args.model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_path.parent / "tmp-stage5-eval"),
            per_device_eval_batch_size=args.eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, tokenizer),
    )

    validation_examples = splits["validation"]
    dev_examples = splits["dev"]
    validation_features, validation_model_dataset = prepare_stage5_eval_artifacts(
        validation_examples,
        tokenizer,
        args.max_length,
        args.doc_stride,
    )
    dev_features, dev_model_dataset = prepare_stage5_eval_artifacts(
        dev_examples,
        tokenizer,
        args.max_length,
        args.doc_stride,
    )

    validation_raw_predictions = predict_stage5_raw_outputs(trainer, validation_model_dataset)
    dev_raw_predictions = predict_stage5_raw_outputs(trainer, dev_model_dataset)
    validation_references = build_reference_index(validation_examples)
    dev_references = build_reference_index(dev_examples)

    threshold, validation_metrics, validation_mix, threshold_sweep = search_keep_threshold(
        validation_examples,
        validation_features,
        validation_raw_predictions,
        validation_references,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        threshold_min=args.keep_threshold_min,
        threshold_max=args.keep_threshold_max,
        threshold_step=args.keep_threshold_step,
        match_f1_threshold=args.match_f1_threshold,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
    )

    dev_predictions = postprocess_stage5_predictions(
        dev_examples,
        dev_features,
        dev_raw_predictions,
        keep_threshold=threshold,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )
    dev_metrics = compute_stage1_metrics(dev_predictions, dev_references)
    dev_mix = compute_answer_support_mix(
        dev_predictions,
        dev_references,
        match_f1_threshold=args.match_f1_threshold,
    )

    output = {
        "stage": "stage5-support-constrained-eval",
        "model_path": str(args.model_path),
        "selected_keep_threshold": threshold,
        "max_unsupported_answer_rate": args.max_unsupported_answer_rate,
        "validation_metrics": validation_metrics,
        "validation_mix": validation_mix,
        "threshold_sweep": threshold_sweep,
        "dev_metrics": dev_metrics,
        "dev_mix": dev_mix,
        "dev_predictions": dev_predictions,
    }
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        _train_model(args)
        return
    if args.command == "evaluate":
        _evaluate_model(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
