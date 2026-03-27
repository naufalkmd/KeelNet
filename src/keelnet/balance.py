"""Stage 6 adaptive candidate balancing on top of Stage 5 predictions."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from keelnet.config import (
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_N_BEST_SIZE,
    DEFAULT_SEED,
    DEFAULT_STAGE5_KEEP_THRESHOLD_MAX,
    DEFAULT_STAGE5_KEEP_THRESHOLD_MIN,
    DEFAULT_STAGE5_KEEP_THRESHOLD_STEP,
    DEFAULT_STAGE5_MAX_UNSUPPORTED_ANSWER_RATE,
    DEFAULT_SUPPORT_MATCH_F1,
    DEFAULT_VALIDATION_SIZE,
)
from keelnet.data import build_reference_index, load_stage1_splits
from keelnet.metrics import (
    compute_answer_support_mix,
    compute_stage1_metrics,
    is_supported_answer,
    normalize_answer,
)

CONTROLLER_STATE_FILENAME = "stage6_balance_controller.pt"
CONTROLLER_CONFIG_FILENAME = "stage6_balance_config.json"
TRAINING_HISTORY_FILENAME = "stage6_balance_training_history.json"


@dataclass(frozen=True)
class CandidateRecord:
    example_id: str
    answer_text: str
    span_score: float
    score_gap_to_best: float
    score_margin_to_next: float
    keep_probability: float
    support_probability: float
    abstain_probability: float
    answer_length_tokens: float
    normalized_rank: float
    label: float
    hard_negative_weight: float


@dataclass(frozen=True)
class CandidateBundle:
    feature_names: list[str]
    all_example_ids: list[str]
    records: list[CandidateRecord]
    features: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray


@dataclass(frozen=True)
class AdaptiveBalanceConfig:
    stage5_model_path: str
    input_dim: int
    hidden_size: int
    dropout: float
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    n_best_size: int
    max_answer_length: int
    max_candidates_per_example: int
    max_candidates_per_feature: int
    validation_size: float
    seed: int
    max_unsupported_answer_rate: float
    train_batch_size: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    positive_weight: float
    negative_weight_init: float
    negative_weight_min: float
    negative_weight_max: float
    adaptive_weight_step: float
    hard_negative_weight: float
    max_train_samples: int | None
    max_eval_samples: int | None


FEATURE_NAMES = [
    "span_score",
    "score_gap_to_best",
    "score_margin_to_next",
    "keep_probability",
    "support_probability",
    "abstain_probability",
    "answer_length_tokens",
    "normalized_rank",
]


class AdaptiveBalanceController(nn.Module):
    """Small MLP that scores answer candidates for answer-vs-abstain selection."""

    def __init__(self, input_dim: int, *, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.dropout_rate = float(dropout)

        if hidden_size <= 0:
            self.network = nn.Linear(input_dim, 1)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def keep_probability_from_logits(abstain_logit: float, support_logit: float) -> float:
    abstain_probability = 1.0 / (1.0 + np.exp(-float(abstain_logit)))
    support_probability = 1.0 / (1.0 + np.exp(-float(support_logit)))
    keep_probability = (1.0 - abstain_probability) * support_probability
    return float(min(1.0 - 1e-6, max(1e-6, keep_probability)))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _feature_candidate_spans(
    context: str,
    feature,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    *,
    n_best_size: int,
    max_answer_length: int,
    max_candidates_per_feature: int,
) -> list[dict[str, float | str]]:
    offsets = feature["offset_mapping"]
    candidates: list[dict[str, float | str]] = []

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

            answer_text = context[start_char:end_char].strip()
            if not normalize_answer(answer_text):
                continue

            candidates.append(
                {
                    "answer_text": answer_text,
                    "span_score": float(start_logits[start_index] + end_logits[end_index]),
                }
            )

    candidates.sort(key=lambda item: float(item["span_score"]), reverse=True)
    deduped: list[dict[str, float | str]] = []
    seen_answers: set[str] = set()
    for candidate in candidates:
        normalized = normalize_answer(str(candidate["answer_text"]))
        if normalized in seen_answers:
            continue
        seen_answers.add(normalized)
        deduped.append(candidate)
        if len(deduped) >= max_candidates_per_feature:
            break
    return deduped


def _candidate_feature_vector(record: CandidateRecord) -> list[float]:
    return [
        float(record.span_score),
        float(record.score_gap_to_best),
        float(record.score_margin_to_next),
        float(record.keep_probability),
        float(record.support_probability),
        float(record.abstain_probability),
        float(record.answer_length_tokens),
        float(record.normalized_rank),
    ]


def build_candidate_bundle(
    examples,
    features,
    raw_predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    references: dict[str, dict[str, Any]],
    *,
    n_best_size: int,
    max_answer_length: int,
    max_candidates_per_example: int,
    max_candidates_per_feature: int,
    match_f1_threshold: float,
    hard_negative_weight: float,
) -> CandidateBundle:
    all_start_logits, all_end_logits, all_abstain_logits, all_support_logits = raw_predictions
    features_per_example: dict[str, list[int]] = defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_per_example[str(feature["example_id"])].append(feature_index)

    records: list[CandidateRecord] = []
    all_example_ids = [str(example["id"]) for example in examples]
    for example in examples:
        example_id = str(example["id"])
        context = str(example["context"])
        gold_answers = list(references.get(example_id, {}).get("answers", []))
        candidates_by_answer: dict[str, dict[str, float | str]] = {}

        for feature_index in features_per_example.get(example_id, []):
            abstain_logit = float(all_abstain_logits[feature_index])
            support_logit = float(all_support_logits[feature_index])
            abstain_probability = _sigmoid(abstain_logit)
            support_probability = _sigmoid(support_logit)
            keep_probability = keep_probability_from_logits(abstain_logit, support_logit)

            feature_candidates = _feature_candidate_spans(
                context,
                features[feature_index],
                np.asarray(all_start_logits[feature_index]),
                np.asarray(all_end_logits[feature_index]),
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
                max_candidates_per_feature=max_candidates_per_feature,
            )
            for candidate in feature_candidates:
                answer_text = str(candidate["answer_text"])
                normalized = normalize_answer(answer_text)
                payload = {
                    "answer_text": answer_text,
                    "span_score": float(candidate["span_score"]),
                    "keep_probability": keep_probability,
                    "support_probability": support_probability,
                    "abstain_probability": abstain_probability,
                }
                existing = candidates_by_answer.get(normalized)
                if existing is None or float(payload["span_score"]) > float(existing["span_score"]):
                    candidates_by_answer[normalized] = payload

        ordered_candidates = sorted(
            candidates_by_answer.values(),
            key=lambda item: float(item["span_score"]),
            reverse=True,
        )[:max_candidates_per_example]
        if not ordered_candidates:
            continue

        best_score = float(ordered_candidates[0]["span_score"])
        for index, candidate in enumerate(ordered_candidates):
            next_score = (
                float(ordered_candidates[index + 1]["span_score"])
                if index + 1 < len(ordered_candidates)
                else float(candidate["span_score"])
            )
            answer_text = str(candidate["answer_text"])
            label = 1.0 if is_supported_answer(
                answer_text,
                gold_answers,
                match_f1_threshold=match_f1_threshold,
            ) else 0.0
            normalized_rank = float(index) / max(1, len(ordered_candidates) - 1)
            hard_weight = 1.0
            if label < 0.5:
                hard_weight += float(hard_negative_weight) * float((1.0 - normalized_rank) * candidate["keep_probability"])

            records.append(
                CandidateRecord(
                    example_id=example_id,
                    answer_text=answer_text,
                    span_score=float(candidate["span_score"]),
                    score_gap_to_best=float(best_score - float(candidate["span_score"])),
                    score_margin_to_next=float(float(candidate["span_score"]) - next_score),
                    keep_probability=float(candidate["keep_probability"]),
                    support_probability=float(candidate["support_probability"]),
                    abstain_probability=float(candidate["abstain_probability"]),
                    answer_length_tokens=float(max(1, len(normalize_answer(answer_text).split()))),
                    normalized_rank=normalized_rank,
                    label=label,
                    hard_negative_weight=hard_weight,
                )
            )

    if records:
        features_matrix = np.asarray([_candidate_feature_vector(record) for record in records], dtype=np.float32)
        labels = np.asarray([record.label for record in records], dtype=np.float32)
        sample_weights = np.asarray([record.hard_negative_weight for record in records], dtype=np.float32)
    else:
        features_matrix = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.float32)
        sample_weights = np.zeros((0,), dtype=np.float32)

    return CandidateBundle(
        feature_names=list(FEATURE_NAMES),
        all_example_ids=all_example_ids,
        records=records,
        features=features_matrix,
        labels=labels,
        sample_weights=sample_weights,
    )


def fit_feature_standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.size == 0:
        raise ValueError("Cannot fit standardization on an empty feature matrix.")
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features.astype(np.float32)
    return ((features - mean) / std).astype(np.float32)


def postprocess_stage6_predictions(
    bundle: CandidateBundle,
    candidate_probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    records_by_example: dict[str, list[tuple[CandidateRecord, float]]] = defaultdict(list)
    for record, probability in zip(bundle.records, candidate_probabilities, strict=False):
        records_by_example[record.example_id].append((record, float(probability)))

    predictions: dict[str, dict[str, Any]] = {}
    for example_id in bundle.all_example_ids:
        candidates = records_by_example.get(example_id, [])
        if not candidates:
            predictions[example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "controller_probability": 0.0,
                    "span_score": float("-inf"),
                    "keep_probability": 0.0,
                    "support_probability": 0.0,
                    "abstain_probability": 1.0,
                },
                "abstain_reason": "no_candidate",
            }
            continue

        best_record, best_probability = max(
            candidates,
            key=lambda item: (item[1], item[0].span_score, item[0].keep_probability),
        )
        decision = "answer" if best_probability >= threshold else "abstain"
        predictions[example_id] = {
            "decision": decision,
            "answer": best_record.answer_text if decision == "answer" else "",
            "scores": {
                "controller_probability": best_probability,
                "span_score": best_record.span_score,
                "keep_probability": best_record.keep_probability,
                "support_probability": best_record.support_probability,
                "abstain_probability": best_record.abstain_probability,
            },
            "support": {"score": best_record.support_probability},
            "balance": {
                "selected_answer": best_record.answer_text,
                "selected_rank": best_record.normalized_rank,
                "score_gap_to_best": best_record.score_gap_to_best,
                "score_margin_to_next": best_record.score_margin_to_next,
            },
        }
        if decision == "abstain":
            predictions[example_id]["abstain_reason"] = "controller_gate"

    return predictions


def select_stage6_threshold_entry(
    sweep: list[dict[str, float]],
    *,
    max_unsupported_answer_rate: float,
) -> dict[str, float]:
    if not sweep:
        raise ValueError("Threshold sweep is empty.")

    constrained = [entry for entry in sweep if bool(entry.get("constraint_satisfied", False))]
    pool = constrained or sweep
    return max(
        pool,
        key=lambda entry: (
            float(entry["overall_f1"]),
            float(entry["answerable_f1"]),
            -float(entry["unsupported_answer_rate"]),
            float(entry["supported_answer_rate"]),
            float(entry["abstain_f1"]),
        ),
    )


def search_balance_threshold(
    bundle: CandidateBundle,
    candidate_probabilities: np.ndarray,
    references: dict[str, dict[str, Any]],
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    match_f1_threshold: float,
    max_unsupported_answer_rate: float,
) -> tuple[float, dict[str, float], dict[str, float], list[dict[str, float]]]:
    sweep: list[dict[str, float]] = []
    current = float(threshold_min)
    while current <= threshold_max + 1e-9:
        predictions = postprocess_stage6_predictions(
            bundle,
            candidate_probabilities,
            threshold=current,
        )
        metrics = compute_stage1_metrics(predictions, references)
        mix = compute_answer_support_mix(
            predictions,
            references,
            match_f1_threshold=match_f1_threshold,
        )
        sweep.append(
            {
                "threshold": round(current, 10),
                "constraint_satisfied": metrics["unsupported_answer_rate"] <= max_unsupported_answer_rate,
                **metrics,
                **mix,
            }
        )
        current += threshold_step

    best_entry = select_stage6_threshold_entry(
        sweep,
        max_unsupported_answer_rate=max_unsupported_answer_rate,
    )
    best_threshold = float(best_entry["threshold"])
    best_metrics = {
        key: value
        for key, value in best_entry.items()
        if key
        not in {
            "threshold",
            "constraint_satisfied",
            "answer_rate",
            "supported_answer_rate",
            "unsupported_among_answers_rate",
            "answered_count",
            "supported_answers_count",
            "unsupported_answers_count",
        }
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


def _prepare_stage5_prediction_artifacts(
    *,
    model_path: Path,
    validation_size: float,
    seed: int,
    max_length: int,
    doc_stride: int,
    eval_batch_size: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    from keelnet.hf_compat import trainer_processing_kwargs
    from keelnet.learn import (
        load_stage5_model,
        predict_stage5_raw_outputs,
        prepare_stage5_eval_artifacts,
    )

    splits = load_stage1_splits(
        validation_size=validation_size,
        seed=seed,
        answer_only_train=False,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = load_stage5_model(model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_path / "tmp-stage6-balance"),
            per_device_eval_batch_size=eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, tokenizer),
    )

    artifacts: dict[str, Any] = {"splits": splits}
    for split_name in ("train", "validation", "dev"):
        split_examples = splits[split_name]
        eval_features, eval_model_dataset = prepare_stage5_eval_artifacts(
            split_examples,
            tokenizer,
            max_length,
            doc_stride,
        )
        raw_predictions = predict_stage5_raw_outputs(trainer, eval_model_dataset)
        artifacts[split_name] = {
            "examples": split_examples,
            "features": eval_features,
            "raw_predictions": raw_predictions,
            "references": build_reference_index(split_examples),
        }
    return artifacts


def _bundle_to_tensors(bundle: CandidateBundle) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(bundle.features, dtype=torch.float32),
        torch.tensor(bundle.labels, dtype=torch.float32),
        torch.tensor(bundle.sample_weights, dtype=torch.float32),
    )


def _predict_candidate_probabilities(
    model: AdaptiveBalanceController,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0,), dtype=np.float32)

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(batch)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            outputs.append(np.asarray(probabilities, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def _evaluate_candidate_loss(
    model: AdaptiveBalanceController,
    bundle: CandidateBundle,
    *,
    batch_size: int,
    device: torch.device,
    positive_weight: float,
    negative_weight: float,
) -> float:
    if bundle.features.size == 0:
        return 0.0

    features_tensor, labels_tensor, sample_weights_tensor = _bundle_to_tensors(bundle)
    dataset = TensorDataset(features_tensor, labels_tensor, sample_weights_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_features, batch_labels, batch_sample_weights in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_sample_weights = batch_sample_weights.to(device)

            logits = model(batch_features)
            class_weights = torch.where(
                batch_labels > 0.5,
                torch.full_like(batch_labels, float(positive_weight)),
                torch.full_like(batch_labels, float(negative_weight)),
            )
            losses = F.binary_cross_entropy_with_logits(logits, batch_labels, reduction="none")
            weighted = losses * class_weights * batch_sample_weights
            total_loss += float(weighted.sum().detach().cpu())
            total_count += len(batch_labels)

    if total_count == 0:
        return 0.0
    return total_loss / total_count


def _is_better_validation_entry(
    candidate: dict[str, float],
    best: dict[str, float] | None,
    *,
    max_unsupported_answer_rate: float,
) -> bool:
    if best is None:
        return True

    candidate_satisfied = candidate["validation_unsupported_answer_rate"] <= max_unsupported_answer_rate
    best_satisfied = best["validation_unsupported_answer_rate"] <= max_unsupported_answer_rate
    if candidate_satisfied != best_satisfied:
        return candidate_satisfied
    if candidate_satisfied:
        return (
            candidate["validation_overall_f1"],
            candidate["validation_answerable_f1"],
            -candidate["validation_unsupported_answer_rate"],
            candidate["validation_abstain_f1"],
        ) > (
            best["validation_overall_f1"],
            best["validation_answerable_f1"],
            -best["validation_unsupported_answer_rate"],
            best["validation_abstain_f1"],
        )
    return (
        -candidate["validation_unsupported_answer_rate"],
        candidate["validation_overall_f1"],
        candidate["validation_answerable_f1"],
        candidate["validation_abstain_f1"],
    ) > (
        -best["validation_unsupported_answer_rate"],
        best["validation_overall_f1"],
        best["validation_answerable_f1"],
        best["validation_abstain_f1"],
    )


def _save_controller(
    model: AdaptiveBalanceController,
    output_dir: Path,
    config: AdaptiveBalanceConfig,
    training_history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / CONTROLLER_STATE_FILENAME)
    (output_dir / CONTROLLER_CONFIG_FILENAME).write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / TRAINING_HISTORY_FILENAME).write_text(
        json.dumps(training_history, indent=2) + "\n",
        encoding="utf-8",
    )


def load_controller(
    controller_path: Path,
) -> tuple[AdaptiveBalanceController, AdaptiveBalanceConfig]:
    config_payload = json.loads((controller_path / CONTROLLER_CONFIG_FILENAME).read_text(encoding="utf-8"))
    config = AdaptiveBalanceConfig(**config_payload)
    model = AdaptiveBalanceController(
        config.input_dim,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    )
    state_dict = torch.load(controller_path / CONTROLLER_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    return model, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Stage 6 adaptive balance controller.")
    train_parser.add_argument("--stage5-model-path", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    train_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    train_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    train_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    train_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--eval-batch-size", type=int, default=8)
    train_parser.add_argument("--train-batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--num-train-epochs", type=int, default=10)
    train_parser.add_argument("--hidden-size", type=int, default=32)
    train_parser.add_argument("--dropout", type=float, default=0.10)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-eval-samples", type=int, default=None)
    train_parser.add_argument("--max-candidates-per-example", type=int, default=6)
    train_parser.add_argument("--max-candidates-per-feature", type=int, default=3)
    train_parser.add_argument("--positive-weight", type=float, default=1.0)
    train_parser.add_argument("--negative-weight-init", type=float, default=1.0)
    train_parser.add_argument("--negative-weight-min", type=float, default=0.5)
    train_parser.add_argument("--negative-weight-max", type=float, default=6.0)
    train_parser.add_argument("--adaptive-weight-step", type=float, default=0.5)
    train_parser.add_argument("--hard-negative-weight", type=float, default=1.5)
    train_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE5_MAX_UNSUPPORTED_ANSWER_RATE,
    )
    train_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the Stage 6 adaptive balance controller.")
    eval_parser.add_argument("--controller-path", type=Path, required=True)
    eval_parser.add_argument("--output-path", type=Path, required=True)
    eval_parser.add_argument("--stage5-model-path", type=Path, default=None)
    eval_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    eval_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    eval_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    eval_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    eval_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    eval_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    eval_parser.add_argument("--eval-batch-size", type=int, default=8)
    eval_parser.add_argument("--max-eval-samples", type=int, default=None)
    eval_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)
    eval_parser.add_argument(
        "--candidate-threshold-min",
        type=float,
        default=DEFAULT_STAGE5_KEEP_THRESHOLD_MIN,
    )
    eval_parser.add_argument(
        "--candidate-threshold-max",
        type=float,
        default=DEFAULT_STAGE5_KEEP_THRESHOLD_MAX,
    )
    eval_parser.add_argument(
        "--candidate-threshold-step",
        type=float,
        default=DEFAULT_STAGE5_KEEP_THRESHOLD_STEP,
    )
    eval_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE5_MAX_UNSUPPORTED_ANSWER_RATE,
    )
    return parser


def _train_controller(args: argparse.Namespace) -> None:
    from transformers import set_seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts = _prepare_stage5_prediction_artifacts(
        model_path=args.stage5_model_path,
        validation_size=args.validation_size,
        seed=args.seed,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        eval_batch_size=args.eval_batch_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    train_bundle = build_candidate_bundle(
        artifacts["train"]["examples"],
        artifacts["train"]["features"],
        artifacts["train"]["raw_predictions"],
        artifacts["train"]["references"],
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        max_candidates_per_example=args.max_candidates_per_example,
        max_candidates_per_feature=args.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=args.hard_negative_weight,
    )
    validation_bundle = build_candidate_bundle(
        artifacts["validation"]["examples"],
        artifacts["validation"]["features"],
        artifacts["validation"]["raw_predictions"],
        artifacts["validation"]["references"],
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        max_candidates_per_example=args.max_candidates_per_example,
        max_candidates_per_feature=args.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=args.hard_negative_weight,
    )

    if train_bundle.features.size == 0:
        raise ValueError("No Stage 6 training candidates were generated.")
    if validation_bundle.features.size == 0:
        raise ValueError("No Stage 6 validation candidates were generated.")

    feature_mean, feature_std = fit_feature_standardization(train_bundle.features)
    standardized_train_bundle = CandidateBundle(
        feature_names=train_bundle.feature_names,
        all_example_ids=train_bundle.all_example_ids,
        records=train_bundle.records,
        features=standardize_features(train_bundle.features, feature_mean, feature_std),
        labels=train_bundle.labels,
        sample_weights=train_bundle.sample_weights,
    )
    standardized_validation_bundle = CandidateBundle(
        feature_names=validation_bundle.feature_names,
        all_example_ids=validation_bundle.all_example_ids,
        records=validation_bundle.records,
        features=standardize_features(validation_bundle.features, feature_mean, feature_std),
        labels=validation_bundle.labels,
        sample_weights=validation_bundle.sample_weights,
    )

    model = AdaptiveBalanceController(
        standardized_train_bundle.features.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_dataset = TensorDataset(*_bundle_to_tensors(standardized_train_bundle))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    references = artifacts["validation"]["references"]
    negative_weight = float(args.negative_weight_init)
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_entry: dict[str, float] | None = None
    training_history: list[dict[str, float]] = []

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        epoch_loss = 0.0
        example_count = 0
        for batch_features, batch_labels, batch_sample_weights in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_sample_weights = batch_sample_weights.to(device)

            logits = model(batch_features)
            class_weights = torch.where(
                batch_labels > 0.5,
                torch.full_like(batch_labels, float(args.positive_weight)),
                torch.full_like(batch_labels, negative_weight),
            )
            loss_vector = F.binary_cross_entropy_with_logits(logits, batch_labels, reduction="none")
            loss = (loss_vector * class_weights * batch_sample_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * len(batch_labels)
            example_count += len(batch_labels)

        train_loss = epoch_loss / max(1, example_count)
        validation_probabilities = _predict_candidate_probabilities(
            model,
            standardized_validation_bundle.features,
            batch_size=args.train_batch_size,
            device=device,
        )
        validation_predictions = postprocess_stage6_predictions(
            standardized_validation_bundle,
            validation_probabilities,
            threshold=0.50,
        )
        validation_metrics = compute_stage1_metrics(validation_predictions, references)
        validation_mix = compute_answer_support_mix(
            validation_predictions,
            references,
            match_f1_threshold=args.match_f1_threshold,
        )
        validation_loss = _evaluate_candidate_loss(
            model,
            standardized_validation_bundle,
            batch_size=args.train_batch_size,
            device=device,
            positive_weight=args.positive_weight,
            negative_weight=negative_weight,
        )
        history_entry = {
            "epoch": float(epoch),
            "negative_weight": float(negative_weight),
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "validation_overall_f1": float(validation_metrics["overall_f1"]),
            "validation_answerable_f1": float(validation_metrics["answerable_f1"]),
            "validation_unsupported_answer_rate": float(validation_metrics["unsupported_answer_rate"]),
            "validation_abstain_f1": float(validation_metrics["abstain_f1"]),
            "validation_answer_rate": float(validation_mix["answer_rate"]),
            "validation_supported_answer_rate": float(validation_mix["supported_answer_rate"]),
        }
        training_history.append(history_entry)

        if _is_better_validation_entry(
            history_entry,
            best_entry,
            max_unsupported_answer_rate=args.max_unsupported_answer_rate,
        ):
            best_entry = history_entry
            best_state_dict = copy.deepcopy(model.state_dict())

        target = max(float(args.max_unsupported_answer_rate), 1.0)
        gap_ratio = (float(validation_metrics["unsupported_answer_rate"]) - float(args.max_unsupported_answer_rate)) / target
        adjusted_weight = negative_weight * (1.0 + float(args.adaptive_weight_step) * gap_ratio)
        negative_weight = float(
            min(
                float(args.negative_weight_max),
                max(float(args.negative_weight_min), adjusted_weight),
            )
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    config = AdaptiveBalanceConfig(
        stage5_model_path=str(args.stage5_model_path),
        input_dim=standardized_train_bundle.features.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        feature_names=list(standardized_train_bundle.feature_names),
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        max_candidates_per_example=args.max_candidates_per_example,
        max_candidates_per_feature=args.max_candidates_per_feature,
        validation_size=args.validation_size,
        seed=args.seed,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        positive_weight=args.positive_weight,
        negative_weight_init=args.negative_weight_init,
        negative_weight_min=args.negative_weight_min,
        negative_weight_max=args.negative_weight_max,
        adaptive_weight_step=args.adaptive_weight_step,
        hard_negative_weight=args.hard_negative_weight,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    _save_controller(model.cpu(), args.output_dir, config, training_history)

    run_metadata = {
        "stage": "stage6-adaptive-balance-train",
        "stage5_model_path": str(args.stage5_model_path),
        "train_examples": len(artifacts["train"]["examples"]),
        "validation_examples": len(artifacts["validation"]["examples"]),
        "train_candidate_count": len(train_bundle.records),
        "validation_candidate_count": len(validation_bundle.records),
        "best_epoch": best_entry["epoch"] if best_entry is not None else None,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _evaluate_controller(args: argparse.Namespace) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    controller, config = load_controller(args.controller_path)
    resolved_stage5_model_path = args.stage5_model_path or Path(config.stage5_model_path)
    feature_mean = np.asarray(config.feature_mean, dtype=np.float32)
    feature_std = np.asarray(config.feature_std, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = controller.to(device)

    artifacts = _prepare_stage5_prediction_artifacts(
        model_path=Path(resolved_stage5_model_path),
        validation_size=args.validation_size,
        seed=args.seed,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        eval_batch_size=args.eval_batch_size,
        max_train_samples=None,
        max_eval_samples=args.max_eval_samples,
    )
    validation_bundle = build_candidate_bundle(
        artifacts["validation"]["examples"],
        artifacts["validation"]["features"],
        artifacts["validation"]["raw_predictions"],
        artifacts["validation"]["references"],
        n_best_size=config.n_best_size,
        max_answer_length=config.max_answer_length,
        max_candidates_per_example=config.max_candidates_per_example,
        max_candidates_per_feature=config.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=config.hard_negative_weight,
    )
    dev_bundle = build_candidate_bundle(
        artifacts["dev"]["examples"],
        artifacts["dev"]["features"],
        artifacts["dev"]["raw_predictions"],
        artifacts["dev"]["references"],
        n_best_size=config.n_best_size,
        max_answer_length=config.max_answer_length,
        max_candidates_per_example=config.max_candidates_per_example,
        max_candidates_per_feature=config.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=config.hard_negative_weight,
    )

    standardized_validation_bundle = CandidateBundle(
        feature_names=validation_bundle.feature_names,
        all_example_ids=validation_bundle.all_example_ids,
        records=validation_bundle.records,
        features=standardize_features(validation_bundle.features, feature_mean, feature_std),
        labels=validation_bundle.labels,
        sample_weights=validation_bundle.sample_weights,
    )
    standardized_dev_bundle = CandidateBundle(
        feature_names=dev_bundle.feature_names,
        all_example_ids=dev_bundle.all_example_ids,
        records=dev_bundle.records,
        features=standardize_features(dev_bundle.features, feature_mean, feature_std),
        labels=dev_bundle.labels,
        sample_weights=dev_bundle.sample_weights,
    )

    validation_probabilities = _predict_candidate_probabilities(
        controller,
        standardized_validation_bundle.features,
        batch_size=max(32, config.train_batch_size),
        device=device,
    )
    dev_probabilities = _predict_candidate_probabilities(
        controller,
        standardized_dev_bundle.features,
        batch_size=max(32, config.train_batch_size),
        device=device,
    )

    selected_threshold, validation_metrics, validation_mix, threshold_sweep = search_balance_threshold(
        standardized_validation_bundle,
        validation_probabilities,
        artifacts["validation"]["references"],
        threshold_min=args.candidate_threshold_min,
        threshold_max=args.candidate_threshold_max,
        threshold_step=args.candidate_threshold_step,
        match_f1_threshold=args.match_f1_threshold,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
    )

    dev_predictions = postprocess_stage6_predictions(
        standardized_dev_bundle,
        dev_probabilities,
        threshold=selected_threshold,
    )
    dev_metrics = compute_stage1_metrics(dev_predictions, artifacts["dev"]["references"])
    dev_mix = compute_answer_support_mix(
        dev_predictions,
        artifacts["dev"]["references"],
        match_f1_threshold=args.match_f1_threshold,
    )

    output = {
        "stage": "stage6-adaptive-balance-eval",
        "controller_path": str(args.controller_path),
        "stage5_model_path": str(resolved_stage5_model_path),
        "selected_candidate_threshold": selected_threshold,
        "max_unsupported_answer_rate": args.max_unsupported_answer_rate,
        "validation_metrics": validation_metrics,
        "validation_mix": validation_mix,
        "threshold_sweep": threshold_sweep,
        "dev_metrics": dev_metrics,
        "dev_mix": dev_mix,
        "dev_predictions": dev_predictions,
        "feature_names": list(config.feature_names),
    }
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        _train_controller(args)
        return
    if args.command == "evaluate":
        _evaluate_controller(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
