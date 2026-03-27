"""Stage 8 hybrid calibrated control on top of Stage 5 candidates."""

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
from datasets import Dataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from keelnet.balance import (
    CandidateBundle,
    CandidateRecord,
    _prepare_stage5_prediction_artifacts,
    build_candidate_bundle,
    fit_feature_standardization,
    standardize_features,
)
from keelnet.calibration import logit_probabilities, sigmoid_scores
from keelnet.config import (
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_N_BEST_SIZE,
    DEFAULT_SEED,
    DEFAULT_STAGE8_CANDIDATE_THRESHOLD_MAX,
    DEFAULT_STAGE8_CANDIDATE_THRESHOLD_MIN,
    DEFAULT_STAGE8_CANDIDATE_THRESHOLD_STEP,
    DEFAULT_STAGE8_MAX_UNSUPPORTED_ANSWER_RATE,
    DEFAULT_SUPPORT_MATCH_F1,
    DEFAULT_VALIDATION_SIZE,
)
from keelnet.data import prepare_verification_features
from keelnet.hf_compat import trainer_processing_kwargs
from keelnet.metrics import compute_answer_support_mix, compute_stage1_metrics

HYBRID_MODEL_STATE_FILENAME = "stage8_hybrid_controller.pt"
HYBRID_MODEL_CONFIG_FILENAME = "stage8_hybrid_config.json"
HYBRID_TRAINING_HISTORY_FILENAME = "stage8_hybrid_training_history.json"


HYBRID_FEATURE_NAMES = [
    "span_score",
    "score_gap_to_best",
    "score_margin_to_next",
    "keep_probability",
    "stage5_support_probability",
    "abstain_probability",
    "raw_verifier_support_probability",
    "calibrated_support_probability",
    "support_gap",
    "support_gate_pass",
    "answer_length_tokens",
    "normalized_rank",
]


@dataclass(frozen=True)
class HybridCandidate:
    example_id: str
    answer_text: str
    span_score: float
    score_gap_to_best: float
    score_margin_to_next: float
    keep_probability: float
    stage5_support_probability: float
    abstain_probability: float
    raw_verifier_support_probability: float
    calibrated_support_probability: float
    support_gate_pass: float
    answer_length_tokens: float
    normalized_rank: float
    label: float
    hard_negative_weight: float


@dataclass(frozen=True)
class HybridBundle:
    feature_names: list[str]
    all_example_ids: list[str]
    records: list[HybridCandidate]
    features: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray


@dataclass(frozen=True)
class HybridControllerConfig:
    stage5_model_path: str
    control_path: str
    verifier_model_path: str
    support_temperature: float
    hard_support_threshold: float
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
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    positive_weight: float
    negative_weight: float
    hard_negative_weight: float
    max_train_samples: int | None
    max_eval_samples: int | None


class HybridSafetyController(nn.Module):
    """Small MLP that scores Stage 5 candidate answers for final keep/abstain."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Stage 8 hybrid controller.")
    train_parser.add_argument("--stage5-model-path", type=Path, required=True)
    train_parser.add_argument("--control-path", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--verifier-model-path", type=Path, default=None)
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    train_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    train_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    train_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    train_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--train-batch-size", type=int, default=64)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--num-train-epochs", type=float, default=10.0)
    train_parser.add_argument("--hidden-size", type=int, default=32)
    train_parser.add_argument("--dropout", type=float, default=0.10)
    train_parser.add_argument("--positive-weight", type=float, default=1.0)
    train_parser.add_argument("--negative-weight", type=float, default=2.0)
    train_parser.add_argument("--hard-negative-weight", type=float, default=1.5)
    train_parser.add_argument("--max-candidates-per-example", type=int, default=6)
    train_parser.add_argument("--max-candidates-per-feature", type=int, default=3)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-eval-samples", type=int, default=None)
    train_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)
    train_parser.add_argument("--hard-support-threshold", type=float, default=None)
    train_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE8_MAX_UNSUPPORTED_ANSWER_RATE,
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the Stage 8 hybrid controller.")
    eval_parser.add_argument("--model-path", type=Path, required=True)
    eval_parser.add_argument("--stage5-model-path", type=Path, default=None)
    eval_parser.add_argument("--control-path", type=Path, default=None)
    eval_parser.add_argument("--verifier-model-path", type=Path, default=None)
    eval_parser.add_argument("--output-path", type=Path, required=True)
    eval_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    eval_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    eval_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    eval_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    eval_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    eval_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    eval_parser.add_argument("--eval-batch-size", type=int, default=16)
    eval_parser.add_argument("--max-eval-samples", type=int, default=None)
    eval_parser.add_argument("--max-train-samples", type=int, default=None)
    eval_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)
    eval_parser.add_argument("--candidate-threshold-min", type=float, default=DEFAULT_STAGE8_CANDIDATE_THRESHOLD_MIN)
    eval_parser.add_argument("--candidate-threshold-max", type=float, default=DEFAULT_STAGE8_CANDIDATE_THRESHOLD_MAX)
    eval_parser.add_argument("--candidate-threshold-step", type=float, default=DEFAULT_STAGE8_CANDIDATE_THRESHOLD_STEP)
    eval_parser.add_argument("--hard-support-threshold", type=float, default=None)
    eval_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE8_MAX_UNSUPPORTED_ANSWER_RATE,
    )

    return parser


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _load_control_settings(
    control_path: Path,
    *,
    verifier_model_path: Path | None,
    hard_support_threshold: float | None,
) -> dict[str, Any]:
    payload = json.loads(control_path.read_text(encoding="utf-8"))
    selected_config = payload.get("selected_config", {}) or {}
    resolved_verifier_model_path = verifier_model_path or Path(payload["verifier_model_path"])
    resolved_support_threshold = (
        float(hard_support_threshold)
        if hard_support_threshold is not None
        else float(selected_config.get("support_threshold", payload.get("support_selected_threshold", 0.5)))
    )
    return {
        "control_path": control_path,
        "verifier_model_path": Path(resolved_verifier_model_path),
        "support_temperature": float(payload["support_temperature"]),
        "hard_support_threshold": resolved_support_threshold,
    }


def _candidate_feature_vector(record: HybridCandidate) -> list[float]:
    return [
        float(record.span_score),
        float(record.score_gap_to_best),
        float(record.score_margin_to_next),
        float(record.keep_probability),
        float(record.stage5_support_probability),
        float(record.abstain_probability),
        float(record.raw_verifier_support_probability),
        float(record.calibrated_support_probability),
        float(record.calibrated_support_probability - record.stage5_support_probability),
        float(record.support_gate_pass),
        float(record.answer_length_tokens),
        float(record.normalized_rank),
    ]


def _candidate_records_to_dataset(examples, bundle: CandidateBundle) -> Dataset:
    if not bundle.records:
        return Dataset.from_list([])

    examples_by_id = {str(example["id"]): example for example in examples}
    records: list[dict[str, str | int]] = []
    for index, record in enumerate(bundle.records):
        example = examples_by_id[record.example_id]
        records.append(
            {
                "candidate_index": index,
                "question": str(example["question"]),
                "context": str(example["context"]),
                "candidate_answer": str(record.answer_text),
            }
        )
    return Dataset.from_list(records)


def score_bundle_with_verifier(
    trainer: Trainer,
    tokenizer,
    examples,
    bundle: CandidateBundle,
    *,
    max_length: int,
    support_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not bundle.records:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    verifier_dataset = _candidate_records_to_dataset(examples, bundle)
    verifier_features = verifier_dataset.map(
        lambda batch: prepare_verification_features(batch, tokenizer, max_length),
        batched=True,
        remove_columns=verifier_dataset.column_names,
        desc="Tokenizing hybrid verification data",
    )
    output = trainer.predict(verifier_features)
    probabilities = _softmax(np.asarray(output.predictions))
    raw_support = np.asarray(probabilities[:, 1], dtype=np.float32)
    calibrated_support = np.asarray(
        sigmoid_scores(logit_probabilities(raw_support), temperature=support_temperature),
        dtype=np.float32,
    )
    return raw_support, calibrated_support


def build_hybrid_bundle(
    candidate_bundle: CandidateBundle,
    raw_support_probabilities: np.ndarray,
    calibrated_support_probabilities: np.ndarray,
    *,
    hard_support_threshold: float,
    hard_negative_weight: float,
) -> HybridBundle:
    if len(raw_support_probabilities) != len(candidate_bundle.records):
        raise ValueError("Raw support probabilities must align with candidate records.")
    if len(calibrated_support_probabilities) != len(candidate_bundle.records):
        raise ValueError("Calibrated support probabilities must align with candidate records.")

    records: list[HybridCandidate] = []
    for record, raw_support, calibrated_support in zip(
        candidate_bundle.records,
        raw_support_probabilities,
        calibrated_support_probabilities,
        strict=False,
    ):
        sample_weight = float(record.hard_negative_weight)
        if float(record.label) < 0.5:
            sample_weight += float(hard_negative_weight) * max(
                float(record.keep_probability),
                float(calibrated_support),
            )

        records.append(
            HybridCandidate(
                example_id=record.example_id,
                answer_text=record.answer_text,
                span_score=float(record.span_score),
                score_gap_to_best=float(record.score_gap_to_best),
                score_margin_to_next=float(record.score_margin_to_next),
                keep_probability=float(record.keep_probability),
                stage5_support_probability=float(record.support_probability),
                abstain_probability=float(record.abstain_probability),
                raw_verifier_support_probability=float(raw_support),
                calibrated_support_probability=float(calibrated_support),
                support_gate_pass=1.0 if float(calibrated_support) >= float(hard_support_threshold) else 0.0,
                answer_length_tokens=float(record.answer_length_tokens),
                normalized_rank=float(record.normalized_rank),
                label=float(record.label),
                hard_negative_weight=float(sample_weight),
            )
        )

    if records:
        features_matrix = np.asarray([_candidate_feature_vector(record) for record in records], dtype=np.float32)
        labels = np.asarray([record.label for record in records], dtype=np.float32)
        sample_weights = np.asarray([record.hard_negative_weight for record in records], dtype=np.float32)
    else:
        features_matrix = np.zeros((0, len(HYBRID_FEATURE_NAMES)), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.float32)
        sample_weights = np.zeros((0,), dtype=np.float32)

    return HybridBundle(
        feature_names=list(HYBRID_FEATURE_NAMES),
        all_example_ids=list(candidate_bundle.all_example_ids),
        records=records,
        features=features_matrix,
        labels=labels,
        sample_weights=sample_weights,
    )


def postprocess_hybrid_predictions(
    bundle: HybridBundle,
    candidate_probabilities: np.ndarray,
    *,
    threshold: float,
    hard_support_threshold: float,
) -> dict[str, dict[str, Any]]:
    records_by_example: dict[str, list[tuple[HybridCandidate, float]]] = defaultdict(list)
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
                    "calibrated_support_probability": 0.0,
                    "raw_verifier_support_probability": 0.0,
                    "span_score": float("-inf"),
                },
                "abstain_reason": "no_candidate",
            }
            continue

        ranked_candidates = sorted(
            candidates,
            key=lambda item: (item[1], item[0].span_score, item[0].keep_probability),
            reverse=True,
        )
        safe_candidates = [
            item
            for item in ranked_candidates
            if item[0].calibrated_support_probability >= hard_support_threshold
        ]

        if safe_candidates and safe_candidates[0][1] >= threshold:
            selected_record, selected_probability = safe_candidates[0]
            predictions[example_id] = {
                "decision": "answer",
                "answer": selected_record.answer_text,
                "scores": {
                    "controller_probability": float(selected_probability),
                    "calibrated_support_probability": float(selected_record.calibrated_support_probability),
                    "raw_verifier_support_probability": float(selected_record.raw_verifier_support_probability),
                    "span_score": float(selected_record.span_score),
                    "keep_probability": float(selected_record.keep_probability),
                    "stage5_support_probability": float(selected_record.stage5_support_probability),
                },
                "support": {"score": float(selected_record.calibrated_support_probability)},
                "hybrid": {
                    "selected_answer": selected_record.answer_text,
                    "selected_rank": float(selected_record.normalized_rank),
                    "support_gate_pass": True,
                },
            }
            continue

        best_record, best_probability = ranked_candidates[0]
        abstain_reason = "controller_gate"
        if best_probability >= threshold and best_record.calibrated_support_probability < hard_support_threshold:
            abstain_reason = "support_shield"

        predictions[example_id] = {
            "decision": "abstain",
            "answer": "",
            "scores": {
                "controller_probability": float(best_probability),
                "calibrated_support_probability": float(best_record.calibrated_support_probability),
                "raw_verifier_support_probability": float(best_record.raw_verifier_support_probability),
                "span_score": float(best_record.span_score),
            },
            "abstain_reason": abstain_reason,
        }

    return predictions


def select_stage8_threshold_entry(
    sweep: list[dict[str, float]],
    *,
    max_unsupported_answer_rate: float,
) -> dict[str, float]:
    if not sweep:
        raise ValueError("Threshold sweep is empty.")

    constrained = [entry for entry in sweep if bool(entry.get("constraint_satisfied", False))]
    pool = constrained or sweep
    if constrained:
        return max(
            pool,
            key=lambda entry: (
                float(entry["overall_f1"]),
                float(entry["answerable_f1"]),
                float(entry["supported_answer_rate"]),
                -float(entry["unsupported_answer_rate"]),
                float(entry["abstain_f1"]),
            ),
        )

    return max(
        pool,
        key=lambda entry: (
            -float(entry["unsupported_answer_rate"]),
            float(entry["overall_f1"]),
            float(entry["answerable_f1"]),
            float(entry["supported_answer_rate"]),
        ),
    )


def search_hybrid_threshold(
    bundle: HybridBundle,
    candidate_probabilities: np.ndarray,
    references: dict[str, dict[str, Any]],
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    hard_support_threshold: float,
    match_f1_threshold: float,
    max_unsupported_answer_rate: float,
) -> tuple[float, dict[str, float], dict[str, float], list[dict[str, float]]]:
    sweep: list[dict[str, float]] = []
    current = float(threshold_min)
    while current <= threshold_max + 1e-9:
        predictions = postprocess_hybrid_predictions(
            bundle,
            candidate_probabilities,
            threshold=current,
            hard_support_threshold=hard_support_threshold,
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

    best_entry = select_stage8_threshold_entry(
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


def _bundle_to_tensors(bundle: HybridBundle) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(bundle.features, dtype=torch.float32),
        torch.tensor(bundle.labels, dtype=torch.float32),
        torch.tensor(bundle.sample_weights, dtype=torch.float32),
    )


def _predict_candidate_probabilities(
    model: HybridSafetyController,
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
    model: HybridSafetyController,
    bundle: HybridBundle,
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


def _build_hybrid_bundle_for_split(
    split_artifact: dict[str, Any],
    verifier_trainer: Trainer,
    verifier_tokenizer,
    control_settings: dict[str, Any],
    *,
    n_best_size: int,
    max_answer_length: int,
    max_candidates_per_example: int,
    max_candidates_per_feature: int,
    match_f1_threshold: float,
    hard_negative_weight: float,
    max_length: int,
) -> HybridBundle:
    candidate_bundle = build_candidate_bundle(
        split_artifact["examples"],
        split_artifact["features"],
        split_artifact["raw_predictions"],
        split_artifact["references"],
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        max_candidates_per_example=max_candidates_per_example,
        max_candidates_per_feature=max_candidates_per_feature,
        match_f1_threshold=match_f1_threshold,
        hard_negative_weight=hard_negative_weight,
    )
    raw_support, calibrated_support = score_bundle_with_verifier(
        verifier_trainer,
        verifier_tokenizer,
        split_artifact["examples"],
        candidate_bundle,
        max_length=max_length,
        support_temperature=float(control_settings["support_temperature"]),
    )
    return build_hybrid_bundle(
        candidate_bundle,
        raw_support,
        calibrated_support,
        hard_support_threshold=float(control_settings["hard_support_threshold"]),
        hard_negative_weight=hard_negative_weight,
    )


def _save_hybrid_controller(
    model: HybridSafetyController,
    config: HybridControllerConfig,
    output_dir: Path,
    *,
    training_history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / HYBRID_MODEL_STATE_FILENAME)
    (output_dir / HYBRID_MODEL_CONFIG_FILENAME).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / HYBRID_TRAINING_HISTORY_FILENAME).write_text(
        json.dumps(training_history, indent=2) + "\n",
        encoding="utf-8",
    )


def load_hybrid_controller(model_path: Path) -> tuple[HybridSafetyController, HybridControllerConfig]:
    config_payload = json.loads((model_path / HYBRID_MODEL_CONFIG_FILENAME).read_text(encoding="utf-8"))
    config = HybridControllerConfig(**config_payload)
    model = HybridSafetyController(
        config.input_dim,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    )
    state_dict = torch.load(model_path / HYBRID_MODEL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    return model, config


def _train_controller(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    control_settings = _load_control_settings(
        args.control_path,
        verifier_model_path=args.verifier_model_path,
        hard_support_threshold=args.hard_support_threshold,
    )

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

    verifier_tokenizer = AutoTokenizer.from_pretrained(control_settings["verifier_model_path"], use_fast=True)
    verifier_model = AutoModelForSequenceClassification.from_pretrained(control_settings["verifier_model_path"])
    verifier_trainer = Trainer(
        model=verifier_model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "tmp-stage8-verifier"),
            per_device_eval_batch_size=args.eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, verifier_tokenizer),
    )

    train_bundle = _build_hybrid_bundle_for_split(
        artifacts["train"],
        verifier_trainer,
        verifier_tokenizer,
        control_settings,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        max_candidates_per_example=args.max_candidates_per_example,
        max_candidates_per_feature=args.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=args.hard_negative_weight,
        max_length=args.max_length,
    )
    validation_bundle = _build_hybrid_bundle_for_split(
        artifacts["validation"],
        verifier_trainer,
        verifier_tokenizer,
        control_settings,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        max_candidates_per_example=args.max_candidates_per_example,
        max_candidates_per_feature=args.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=args.hard_negative_weight,
        max_length=args.max_length,
    )

    if train_bundle.features.size == 0:
        raise ValueError("Hybrid training bundle is empty; Stage 5 did not produce any candidates.")

    feature_mean, feature_std = fit_feature_standardization(train_bundle.features)
    train_bundle = HybridBundle(
        feature_names=train_bundle.feature_names,
        all_example_ids=train_bundle.all_example_ids,
        records=train_bundle.records,
        features=standardize_features(train_bundle.features, feature_mean, feature_std),
        labels=train_bundle.labels,
        sample_weights=train_bundle.sample_weights,
    )
    validation_bundle = HybridBundle(
        feature_names=validation_bundle.feature_names,
        all_example_ids=validation_bundle.all_example_ids,
        records=validation_bundle.records,
        features=standardize_features(validation_bundle.features, feature_mean, feature_std),
        labels=validation_bundle.labels,
        sample_weights=validation_bundle.sample_weights,
    )

    features_tensor, labels_tensor, sample_weights_tensor = _bundle_to_tensors(train_bundle)
    train_dataset = TensorDataset(features_tensor, labels_tensor, sample_weights_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridSafetyController(
        train_bundle.features.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state_dict = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    training_history: list[dict[str, float]] = []

    for epoch in range(int(round(args.num_train_epochs))):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0

        for batch_features, batch_labels, batch_sample_weights in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_sample_weights = batch_sample_weights.to(device)

            logits = model(batch_features)
            class_weights = torch.where(
                batch_labels > 0.5,
                torch.full_like(batch_labels, float(args.positive_weight)),
                torch.full_like(batch_labels, float(args.negative_weight)),
            )
            losses = F.binary_cross_entropy_with_logits(logits, batch_labels, reduction="none")
            weighted_loss = (losses * class_weights * batch_sample_weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_train_loss += float(weighted_loss.detach().cpu()) * len(batch_labels)
            total_train_examples += len(batch_labels)

        average_train_loss = total_train_loss / max(1, total_train_examples)
        validation_loss = _evaluate_candidate_loss(
            model,
            validation_bundle,
            batch_size=args.eval_batch_size,
            device=device,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
        )

        training_history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(average_train_loss),
                "validation_loss": float(validation_loss),
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    config = HybridControllerConfig(
        stage5_model_path=str(args.stage5_model_path),
        control_path=str(args.control_path),
        verifier_model_path=str(control_settings["verifier_model_path"]),
        support_temperature=float(control_settings["support_temperature"]),
        hard_support_threshold=float(control_settings["hard_support_threshold"]),
        input_dim=train_bundle.features.shape[1],
        hidden_size=int(args.hidden_size),
        dropout=float(args.dropout),
        feature_names=list(train_bundle.feature_names),
        feature_mean=feature_mean.astype(float).tolist(),
        feature_std=feature_std.astype(float).tolist(),
        n_best_size=int(args.n_best_size),
        max_answer_length=int(args.max_answer_length),
        max_candidates_per_example=int(args.max_candidates_per_example),
        max_candidates_per_feature=int(args.max_candidates_per_feature),
        validation_size=float(args.validation_size),
        seed=int(args.seed),
        max_unsupported_answer_rate=float(args.max_unsupported_answer_rate),
        train_batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.eval_batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        num_train_epochs=int(round(args.num_train_epochs)),
        positive_weight=float(args.positive_weight),
        negative_weight=float(args.negative_weight),
        hard_negative_weight=float(args.hard_negative_weight),
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    _save_hybrid_controller(
        model.cpu(),
        config,
        args.output_dir,
        training_history=training_history,
    )

    run_metadata = {
        "stage": "stage8-hybrid-train",
        "stage5_model_path": str(args.stage5_model_path),
        "control_path": str(args.control_path),
        "verifier_model_path": str(control_settings["verifier_model_path"]),
        "hard_support_threshold": float(control_settings["hard_support_threshold"]),
        "support_temperature": float(control_settings["support_temperature"]),
        "max_length": args.max_length,
        "doc_stride": args.doc_stride,
        "max_answer_length": args.max_answer_length,
        "n_best_size": args.n_best_size,
        "validation_size": args.validation_size,
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "positive_weight": args.positive_weight,
        "negative_weight": args.negative_weight,
        "hard_negative_weight": args.hard_negative_weight,
        "max_candidates_per_example": args.max_candidates_per_example,
        "max_candidates_per_feature": args.max_candidates_per_feature,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_unsupported_answer_rate": args.max_unsupported_answer_rate,
        "train_candidate_count": len(train_bundle.records),
        "validation_candidate_count": len(validation_bundle.records),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")


def _evaluate_controller(args: argparse.Namespace) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    model, config = load_hybrid_controller(args.model_path)
    resolved_stage5_model_path = args.stage5_model_path or Path(config.stage5_model_path)
    resolved_control_path = args.control_path or Path(config.control_path)
    resolved_verifier_model_path = args.verifier_model_path or Path(config.verifier_model_path)

    control_settings = _load_control_settings(
        Path(resolved_control_path),
        verifier_model_path=Path(resolved_verifier_model_path),
        hard_support_threshold=args.hard_support_threshold or config.hard_support_threshold,
    )

    artifacts = _prepare_stage5_prediction_artifacts(
        model_path=Path(resolved_stage5_model_path),
        validation_size=args.validation_size,
        seed=args.seed,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        eval_batch_size=args.eval_batch_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    verifier_tokenizer = AutoTokenizer.from_pretrained(control_settings["verifier_model_path"], use_fast=True)
    verifier_model = AutoModelForSequenceClassification.from_pretrained(control_settings["verifier_model_path"])
    verifier_trainer = Trainer(
        model=verifier_model,
        args=TrainingArguments(
            output_dir=str(args.output_path.parent / "tmp-stage8-verifier-eval"),
            per_device_eval_batch_size=args.eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, verifier_tokenizer),
    )

    validation_bundle = _build_hybrid_bundle_for_split(
        artifacts["validation"],
        verifier_trainer,
        verifier_tokenizer,
        control_settings,
        n_best_size=config.n_best_size,
        max_answer_length=config.max_answer_length,
        max_candidates_per_example=config.max_candidates_per_example,
        max_candidates_per_feature=config.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=config.hard_negative_weight,
        max_length=args.max_length,
    )
    dev_bundle = _build_hybrid_bundle_for_split(
        artifacts["dev"],
        verifier_trainer,
        verifier_tokenizer,
        control_settings,
        n_best_size=config.n_best_size,
        max_answer_length=config.max_answer_length,
        max_candidates_per_example=config.max_candidates_per_example,
        max_candidates_per_feature=config.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=config.hard_negative_weight,
        max_length=args.max_length,
    )

    feature_mean = np.asarray(config.feature_mean, dtype=np.float32)
    feature_std = np.asarray(config.feature_std, dtype=np.float32)
    validation_bundle = HybridBundle(
        feature_names=validation_bundle.feature_names,
        all_example_ids=validation_bundle.all_example_ids,
        records=validation_bundle.records,
        features=standardize_features(validation_bundle.features, feature_mean, feature_std),
        labels=validation_bundle.labels,
        sample_weights=validation_bundle.sample_weights,
    )
    dev_bundle = HybridBundle(
        feature_names=dev_bundle.feature_names,
        all_example_ids=dev_bundle.all_example_ids,
        records=dev_bundle.records,
        features=standardize_features(dev_bundle.features, feature_mean, feature_std),
        labels=dev_bundle.labels,
        sample_weights=dev_bundle.sample_weights,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    validation_candidate_probabilities = _predict_candidate_probabilities(
        model,
        validation_bundle.features,
        batch_size=args.eval_batch_size,
        device=device,
    )
    dev_candidate_probabilities = _predict_candidate_probabilities(
        model,
        dev_bundle.features,
        batch_size=args.eval_batch_size,
        device=device,
    )

    threshold, validation_metrics, validation_mix, threshold_sweep = search_hybrid_threshold(
        validation_bundle,
        validation_candidate_probabilities,
        artifacts["validation"]["references"],
        threshold_min=args.candidate_threshold_min,
        threshold_max=args.candidate_threshold_max,
        threshold_step=args.candidate_threshold_step,
        hard_support_threshold=float(control_settings["hard_support_threshold"]),
        match_f1_threshold=args.match_f1_threshold,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
    )

    dev_predictions = postprocess_hybrid_predictions(
        dev_bundle,
        dev_candidate_probabilities,
        threshold=threshold,
        hard_support_threshold=float(control_settings["hard_support_threshold"]),
    )
    dev_metrics = compute_stage1_metrics(dev_predictions, artifacts["dev"]["references"])
    dev_mix = compute_answer_support_mix(
        dev_predictions,
        artifacts["dev"]["references"],
        match_f1_threshold=args.match_f1_threshold,
    )

    output = {
        "stage": "stage8-hybrid-eval",
        "model_path": str(args.model_path),
        "stage5_model_path": str(resolved_stage5_model_path),
        "control_path": str(resolved_control_path),
        "verifier_model_path": str(control_settings["verifier_model_path"]),
        "support_temperature": float(control_settings["support_temperature"]),
        "hard_support_threshold": float(control_settings["hard_support_threshold"]),
        "selected_candidate_threshold": float(threshold),
        "max_unsupported_answer_rate": float(args.max_unsupported_answer_rate),
        "validation_metrics": validation_metrics,
        "validation_mix": validation_mix,
        "threshold_sweep": threshold_sweep,
        "dev_metrics": dev_metrics,
        "dev_mix": dev_mix,
        "dev_predictions": dev_predictions,
    }
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        _train_controller(args)
        return
    if args.command == "evaluate":
        _evaluate_controller(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
