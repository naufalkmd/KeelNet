"""Stage 7 risk-budgeted action learning on top of Stage 5 candidates."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from keelnet.balance import (
    CandidateBundle,
    CandidateRecord,
    _predict_candidate_probabilities,
    _prepare_stage5_prediction_artifacts,
    build_candidate_bundle,
    fit_feature_standardization,
    load_controller,
    standardize_features,
)
from keelnet.config import (
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_N_BEST_SIZE,
    DEFAULT_SEED,
    DEFAULT_STAGE7_MAX_OVERABSTAIN_RATE,
    DEFAULT_STAGE7_MAX_UNSUPPORTED_ANSWER_RATE,
    DEFAULT_STAGE7_RISK_THRESHOLD_MAX,
    DEFAULT_STAGE7_RISK_THRESHOLD_MIN,
    DEFAULT_STAGE7_RISK_THRESHOLD_STEP,
    DEFAULT_SUPPORT_MATCH_F1,
    DEFAULT_VALIDATION_SIZE,
)
from keelnet.metrics import (
    compute_answer_support_mix,
    compute_stage1_metrics,
    normalize_answer,
)

ACTION_MODEL_STATE_FILENAME = "stage7_action_model.pt"
ACTION_MODEL_CONFIG_FILENAME = "stage7_action_config.json"
ACTION_TRAINING_HISTORY_FILENAME = "stage7_action_training_history.json"


ACTION_FEATURE_NAMES = [
    "span_score",
    "score_gap_to_best",
    "score_margin_to_next",
    "keep_probability",
    "support_probability",
    "abstain_probability",
    "stage6_controller_probability",
    "answer_length_tokens",
    "normalized_rank",
    "question_overlap",
    "keep_support_gap",
    "keep_uncertainty",
]


@dataclass(frozen=True)
class ActionCandidate:
    example_id: str
    answer_text: str
    span_score: float
    score_gap_to_best: float
    score_margin_to_next: float
    keep_probability: float
    stage5_support_probability: float
    support_probability: float
    abstain_probability: float
    stage6_controller_probability: float
    raw_verifier_support_probability: float
    calibrated_support_probability: float
    support_gate_pass: float
    answer_length_tokens: float
    normalized_rank: float
    question_overlap: float
    label: float
    hard_negative_weight: float
    model_features: tuple[float, ...]


@dataclass(frozen=True)
class ActionSet:
    example_id: str
    question: str
    answerable: bool
    target_action_index: int
    candidates: tuple[ActionCandidate, ...]


@dataclass(frozen=True)
class RiskBudgetedActionConfig:
    stage5_model_path: str
    stage6_controller_path: str | None
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
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    utility_loss_weight: float
    risk_loss_weight: float
    action_loss_weight: float
    tail_risk_weight: float
    max_unsupported_answer_rate: float
    max_overabstain_rate: float
    unsafe_dual: float
    overabstain_dual: float
    unsafe_dual_lr: float
    overabstain_dual_lr: float
    hard_risk_threshold: float
    max_train_samples: int | None
    max_eval_samples: int | None
    clean_splitting: bool = False
    max_test_samples: int | None = None
    control_path: str | None = None
    verifier_model_path: str | None = None
    support_temperature: float | None = None
    hard_support_threshold: float | None = None


class RiskBudgetedActionModel(nn.Module):
    """Group-wise action learner over answer candidates plus abstain."""

    def __init__(self, input_dim: int, *, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.dropout_rate = float(dropout)

        hidden = max(16, int(hidden_size))
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.utility_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 1)
        self.abstain_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        unsafe_dual: float,
        overabstain_dual: float,
    ) -> dict[str, torch.Tensor]:
        hidden = self.candidate_encoder(candidate_features)
        utility_logits = self.utility_head(hidden).squeeze(-1)
        risk_logits = self.risk_head(hidden).squeeze(-1)
        risk_probabilities = torch.sigmoid(risk_logits)

        mask_float = candidate_mask.float()
        masked_hidden = hidden * mask_float.unsqueeze(-1)
        candidate_count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = masked_hidden.sum(dim=1) / candidate_count

        max_hidden = hidden.masked_fill(~candidate_mask.unsqueeze(-1), -1e9)
        max_pool = max_hidden.max(dim=1).values
        no_candidate_mask = candidate_mask.sum(dim=1) == 0
        max_pool = torch.where(no_candidate_mask.unsqueeze(-1), torch.zeros_like(max_pool), max_pool)

        abstain_features = torch.cat([mean_pool, max_pool], dim=-1)
        abstain_logits = self.abstain_head(abstain_features).squeeze(-1) - float(overabstain_dual)

        adjusted_candidate_logits = utility_logits - float(unsafe_dual) * risk_probabilities
        adjusted_candidate_logits = adjusted_candidate_logits.masked_fill(~candidate_mask, -1e9)
        action_logits = torch.cat([adjusted_candidate_logits, abstain_logits.unsqueeze(1)], dim=1)
        action_probabilities = torch.softmax(action_logits, dim=1)

        return {
            "utility_logits": utility_logits,
            "risk_logits": risk_logits,
            "risk_probabilities": risk_probabilities,
            "adjusted_candidate_logits": adjusted_candidate_logits,
            "abstain_logits": abstain_logits,
            "action_logits": action_logits,
            "action_probabilities": action_probabilities,
        }


def _tokenize_to_set(text: str) -> set[str]:
    normalized = normalize_answer(text)
    if not normalized:
        return set()
    return set(normalized.split())


def _question_overlap(question: str, answer_text: str) -> float:
    answer_tokens = _tokenize_to_set(answer_text)
    if not answer_tokens:
        return 0.0
    question_tokens = _tokenize_to_set(question)
    return float(len(answer_tokens & question_tokens)) / float(len(answer_tokens))


def _candidate_feature_vector(
    record: CandidateRecord,
    *,
    question: str,
    stage6_probability: float,
    support_probability: float,
) -> list[float]:
    keep_uncertainty = 1.0 - abs(float(record.keep_probability) - 0.5) * 2.0
    return [
        float(record.span_score),
        float(record.score_gap_to_best),
        float(record.score_margin_to_next),
        float(record.keep_probability),
        float(support_probability),
        float(record.abstain_probability),
        float(stage6_probability),
        float(record.answer_length_tokens),
        float(record.normalized_rank),
        float(_question_overlap(question, record.answer_text)),
        float(record.keep_probability - support_probability),
        float(max(0.0, keep_uncertainty)),
    ]


def _prepare_control_runtime(
    *,
    control_path: Path | None,
    verifier_model_path: Path | None,
    hard_support_threshold: float | None,
    eval_batch_size: int,
    output_dir: Path,
) -> dict[str, Any] | None:
    if control_path is None:
        return None

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    from keelnet.hf_compat import trainer_processing_kwargs
    from keelnet.hybrid import _load_control_settings

    control_settings = _load_control_settings(
        control_path,
        verifier_model_path=verifier_model_path,
        hard_support_threshold=hard_support_threshold,
    )
    verifier_tokenizer = AutoTokenizer.from_pretrained(control_settings["verifier_model_path"], use_fast=True)
    verifier_model = AutoModelForSequenceClassification.from_pretrained(control_settings["verifier_model_path"])
    verifier_trainer = Trainer(
        model=verifier_model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_eval_batch_size=eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, verifier_tokenizer),
    )
    return {
        "control_settings": control_settings,
        "verifier_tokenizer": verifier_tokenizer,
        "verifier_trainer": verifier_trainer,
    }


def _score_bundle_with_control(
    examples,
    bundle: CandidateBundle,
    *,
    control_runtime: dict[str, Any] | None,
    max_length: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if control_runtime is None:
        return None, None

    from keelnet.hybrid import score_bundle_with_verifier

    raw_support, calibrated_support = score_bundle_with_verifier(
        control_runtime["verifier_trainer"],
        control_runtime["verifier_tokenizer"],
        examples,
        bundle,
        max_length=max_length,
        support_temperature=float(control_runtime["control_settings"]["support_temperature"]),
    )
    return raw_support, calibrated_support


def build_action_sets(
    examples,
    bundle: CandidateBundle,
    *,
    stage6_candidate_probabilities: np.ndarray | None = None,
    raw_verifier_support_probabilities: np.ndarray | None = None,
    calibrated_support_probabilities: np.ndarray | None = None,
    hard_support_threshold: float | None = None,
) -> list[ActionSet]:
    if stage6_candidate_probabilities is not None and len(stage6_candidate_probabilities) != len(bundle.records):
        raise ValueError("Stage 6 candidate probabilities must align with bundle records.")
    if raw_verifier_support_probabilities is not None and len(raw_verifier_support_probabilities) != len(bundle.records):
        raise ValueError("Raw verifier support probabilities must align with bundle records.")
    if calibrated_support_probabilities is not None and len(calibrated_support_probabilities) != len(bundle.records):
        raise ValueError("Calibrated support probabilities must align with bundle records.")

    records_by_example: dict[str, list[tuple[CandidateRecord, float, float | None, float | None]]] = defaultdict(list)
    if stage6_candidate_probabilities is None:
        stage6_candidate_probabilities = np.zeros((len(bundle.records),), dtype=np.float32)

    if raw_verifier_support_probabilities is None:
        raw_verifier_support_probabilities = np.full((len(bundle.records),), np.nan, dtype=np.float32)
    if calibrated_support_probabilities is None:
        calibrated_support_probabilities = np.full((len(bundle.records),), np.nan, dtype=np.float32)

    for record, stage6_probability, raw_support, calibrated_support in zip(
        bundle.records,
        stage6_candidate_probabilities,
        raw_verifier_support_probabilities,
        calibrated_support_probabilities,
        strict=False,
    ):
        records_by_example[record.example_id].append(
            (
                record,
                float(stage6_probability),
                None if np.isnan(raw_support) else float(raw_support),
                None if np.isnan(calibrated_support) else float(calibrated_support),
            )
        )

    action_sets: list[ActionSet] = []
    for example in examples:
        example_id = str(example["id"])
        question = str(example["question"])
        answerable = bool(example["answers"]["text"])
        raw_candidates = records_by_example.get(example_id, [])

        candidates = tuple(
            ActionCandidate(
                example_id=example_id,
                answer_text=record.answer_text,
                span_score=float(record.span_score),
                score_gap_to_best=float(record.score_gap_to_best),
                score_margin_to_next=float(record.score_margin_to_next),
                keep_probability=float(record.keep_probability),
                stage5_support_probability=float(record.support_probability),
                support_probability=float(
                    calibrated_support
                    if calibrated_support is not None
                    else record.support_probability
                ),
                abstain_probability=float(record.abstain_probability),
                stage6_controller_probability=float(stage6_probability),
                raw_verifier_support_probability=float(
                    raw_support
                    if raw_support is not None
                    else record.support_probability
                ),
                calibrated_support_probability=float(
                    calibrated_support
                    if calibrated_support is not None
                    else record.support_probability
                ),
                support_gate_pass=float(
                    1.0
                    if calibrated_support is None or hard_support_threshold is None
                    else float(calibrated_support >= hard_support_threshold)
                ),
                answer_length_tokens=float(record.answer_length_tokens),
                normalized_rank=float(record.normalized_rank),
                question_overlap=float(_question_overlap(question, record.answer_text)),
                label=float(record.label),
                hard_negative_weight=float(record.hard_negative_weight),
                model_features=tuple(
                    _candidate_feature_vector(
                        record,
                        question=question,
                        stage6_probability=float(stage6_probability),
                        support_probability=float(
                            calibrated_support
                            if calibrated_support is not None
                            else record.support_probability
                        ),
                    )
                ),
            )
            for record, stage6_probability, raw_support, calibrated_support in raw_candidates
        )

        supported_indexes = [index for index, candidate in enumerate(candidates) if candidate.label > 0.5]
        if supported_indexes:
            target_action_index = max(
                supported_indexes,
                key=lambda index: (
                    candidates[index].support_gate_pass,
                    candidates[index].stage6_controller_probability,
                    candidates[index].keep_probability,
                    candidates[index].support_probability,
                    candidates[index].span_score,
                    -candidates[index].normalized_rank,
                ),
            )
        else:
            target_action_index = len(candidates)

        action_sets.append(
            ActionSet(
                example_id=example_id,
                question=question,
                answerable=answerable,
                target_action_index=target_action_index,
                candidates=candidates,
            )
        )

    return action_sets


def flatten_action_features(action_sets: list[ActionSet]) -> np.ndarray:
    rows: list[list[float]] = []
    for action_set in action_sets:
        for candidate in action_set.candidates:
            rows.append(list(candidate.model_features))
    if not rows:
        return np.zeros((0, len(ACTION_FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def standardize_action_sets(
    action_sets: list[ActionSet],
    mean: np.ndarray,
    std: np.ndarray,
) -> list[ActionSet]:
    matrix = flatten_action_features(action_sets)
    if matrix.size == 0:
        return action_sets

    standardized = standardize_features(matrix, mean, std)
    cursor = 0
    output: list[ActionSet] = []
    for action_set in action_sets:
        candidates: list[ActionCandidate] = []
        for candidate in action_set.candidates:
            candidates.append(replace(candidate, model_features=tuple(standardized[cursor].tolist())))
            cursor += 1
        output.append(replace(action_set, candidates=tuple(candidates)))
    return output


def collate_action_sets(action_sets: list[ActionSet]) -> dict[str, Any]:
    batch_size = len(action_sets)
    max_candidates = max((len(action_set.candidates) for action_set in action_sets), default=0)
    max_candidates = max(1, max_candidates)
    feature_dim = len(ACTION_FEATURE_NAMES)

    candidate_features = torch.zeros((batch_size, max_candidates, feature_dim), dtype=torch.float32)
    candidate_labels = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    candidate_risk_labels = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    candidate_weights = torch.ones((batch_size, max_candidates), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    target_indices = torch.zeros((batch_size,), dtype=torch.long)

    for batch_index, action_set in enumerate(action_sets):
        target_indices[batch_index] = (
            int(action_set.target_action_index)
            if int(action_set.target_action_index) < len(action_set.candidates)
            else max_candidates
        )
        for candidate_index, candidate in enumerate(action_set.candidates):
            candidate_features[batch_index, candidate_index] = torch.tensor(candidate.model_features, dtype=torch.float32)
            candidate_labels[batch_index, candidate_index] = float(candidate.label)
            candidate_risk_labels[batch_index, candidate_index] = 1.0 - float(candidate.label)
            candidate_weights[batch_index, candidate_index] = float(candidate.hard_negative_weight)
            candidate_mask[batch_index, candidate_index] = True

    return {
        "action_sets": action_sets,
        "candidate_features": candidate_features,
        "candidate_labels": candidate_labels,
        "candidate_risk_labels": candidate_risk_labels,
        "candidate_weights": candidate_weights,
        "candidate_mask": candidate_mask,
        "target_indices": target_indices,
    }


def _masked_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = losses * weights * mask.float()
    return masked.sum() / mask.float().sum().clamp_min(1.0)


def compute_action_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    action_loss_weight: float,
    utility_loss_weight: float,
    risk_loss_weight: float,
    tail_risk_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    action_loss = F.cross_entropy(outputs["action_logits"], batch["target_indices"])

    utility_weights = torch.where(
        batch["candidate_labels"] > 0.5,
        torch.ones_like(batch["candidate_weights"]),
        batch["candidate_weights"],
    )
    utility_loss = _masked_weighted_bce(
        outputs["utility_logits"],
        batch["candidate_labels"],
        utility_weights,
        batch["candidate_mask"],
    )

    risk_weights = torch.where(
        batch["candidate_risk_labels"] > 0.5,
        batch["candidate_weights"] * float(tail_risk_weight),
        torch.ones_like(batch["candidate_weights"]),
    )
    risk_loss = _masked_weighted_bce(
        outputs["risk_logits"],
        batch["candidate_risk_labels"],
        risk_weights,
        batch["candidate_mask"],
    )

    total_loss = (
        float(action_loss_weight) * action_loss
        + float(utility_loss_weight) * utility_loss
        + float(risk_loss_weight) * risk_loss
    )
    return total_loss, {
        "action_loss": float(action_loss.detach().cpu()),
        "utility_loss": float(utility_loss.detach().cpu()),
        "risk_loss": float(risk_loss.detach().cpu()),
    }


def _predict_action_outputs(
    model: RiskBudgetedActionModel,
    action_sets: list[ActionSet],
    *,
    batch_size: int,
    device: torch.device,
    unsafe_dual: float,
    overabstain_dual: float,
) -> list[dict[str, Any]]:
    if not action_sets:
        return []

    loader = DataLoader(action_sets, batch_size=batch_size, shuffle=False, collate_fn=collate_action_sets)
    outputs: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_features = batch["candidate_features"].to(device)
            batch_mask = batch["candidate_mask"].to(device)
            batch_outputs = model(
                batch_features,
                batch_mask,
                unsafe_dual=unsafe_dual,
                overabstain_dual=overabstain_dual,
            )
            action_probabilities = batch_outputs["action_probabilities"].detach().cpu().numpy()
            utility_logits = batch_outputs["utility_logits"].detach().cpu().numpy()
            risk_logits = batch_outputs["risk_logits"].detach().cpu().numpy()
            risk_probabilities = batch_outputs["risk_probabilities"].detach().cpu().numpy()
            adjusted_candidate_logits = batch_outputs["adjusted_candidate_logits"].detach().cpu().numpy()
            abstain_logits = batch_outputs["abstain_logits"].detach().cpu().numpy()

            for index, action_set in enumerate(batch["action_sets"]):
                candidate_count = len(action_set.candidates)
                outputs.append(
                    {
                        "candidate_action_probabilities": action_probabilities[index, :candidate_count].astype(float).tolist(),
                        "abstain_probability": float(action_probabilities[index, -1]),
                        "candidate_utility_logits": utility_logits[index, :candidate_count].astype(float).tolist(),
                        "candidate_risk_logits": risk_logits[index, :candidate_count].astype(float).tolist(),
                        "candidate_risk_probabilities": risk_probabilities[index, :candidate_count].astype(float).tolist(),
                        "candidate_action_scores": adjusted_candidate_logits[index, :candidate_count].astype(float).tolist(),
                        "abstain_logit": float(abstain_logits[index]),
                    }
                )
    return outputs


def _evaluate_action_model(
    model: RiskBudgetedActionModel,
    action_sets: list[ActionSet],
    *,
    batch_size: int,
    device: torch.device,
    unsafe_dual: float,
    overabstain_dual: float,
    action_loss_weight: float,
    utility_loss_weight: float,
    risk_loss_weight: float,
    tail_risk_weight: float,
) -> dict[str, float]:
    if not action_sets:
        return {
            "total_loss": 0.0,
            "action_loss": 0.0,
            "utility_loss": 0.0,
            "risk_loss": 0.0,
        }

    loader = DataLoader(action_sets, batch_size=batch_size, shuffle=False, collate_fn=collate_action_sets)
    model.eval()
    total = {"total_loss": 0.0, "action_loss": 0.0, "utility_loss": 0.0, "risk_loss": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch_features = batch["candidate_features"].to(device)
            batch_labels = batch["candidate_labels"].to(device)
            batch_risk_labels = batch["candidate_risk_labels"].to(device)
            batch_weights = batch["candidate_weights"].to(device)
            batch_mask = batch["candidate_mask"].to(device)
            batch_targets = batch["target_indices"].to(device)
            outputs = model(
                batch_features,
                batch_mask,
                unsafe_dual=unsafe_dual,
                overabstain_dual=overabstain_dual,
            )
            loss, components = compute_action_losses(
                outputs,
                {
                    "candidate_labels": batch_labels,
                    "candidate_risk_labels": batch_risk_labels,
                    "candidate_weights": batch_weights,
                    "candidate_mask": batch_mask,
                    "target_indices": batch_targets,
                },
                action_loss_weight=action_loss_weight,
                utility_loss_weight=utility_loss_weight,
                risk_loss_weight=risk_loss_weight,
                tail_risk_weight=tail_risk_weight,
            )
            total["total_loss"] += float(loss.detach().cpu())
            for key, value in components.items():
                total[key] += float(value)
            batches += 1

    if batches == 0:
        return total
    return {key: value / batches for key, value in total.items()}


def compute_overabstain_stats(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> dict[str, float]:
    answerable_total = 0
    overabstain_count = 0

    for example_id, reference in references.items():
        if not bool(reference.get("is_answerable", False)):
            continue
        answerable_total += 1
        decision = str(predictions.get(example_id, {}).get("decision", "abstain")).lower()
        if decision == "abstain":
            overabstain_count += 1

    rate = 0.0
    if answerable_total > 0:
        rate = 100.0 * overabstain_count / answerable_total
    return {
        "overabstain_rate": float(rate),
        "overabstain_count": float(overabstain_count),
    }


def postprocess_action_predictions(
    action_sets: list[ActionSet],
    action_outputs: list[dict[str, Any]],
    *,
    risk_threshold: float,
    hard_support_threshold: float | None = None,
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for action_set, action_output in zip(action_sets, action_outputs, strict=False):
        candidate_count = len(action_set.candidates)
        if candidate_count == 0:
            predictions[action_set.example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "selected_action_probability": float(action_output["abstain_probability"]),
                    "selected_risk_probability": 1.0,
                    "best_candidate_score": float("-inf"),
                    "abstain_score": float(action_output["abstain_logit"]),
                },
                "abstain_reason": "no_candidate",
            }
            continue

        ranked_candidate_indexes = sorted(
            range(candidate_count),
            key=lambda index: (
                float(action_output["candidate_action_scores"][index]),
                float(action_output["candidate_utility_logits"][index]),
                action_set.candidates[index].span_score,
            ),
            reverse=True,
        )
        top_candidate_index = ranked_candidate_indexes[0]
        top_candidate_score = float(action_output["candidate_action_scores"][top_candidate_index])
        abstain_probability = float(action_output["abstain_probability"])
        abstain_score = float(action_output["abstain_logit"])

        risk_safe_candidate_indexes = [
            index
            for index in ranked_candidate_indexes
            if float(action_output["candidate_risk_probabilities"][index]) < risk_threshold
        ]
        if hard_support_threshold is not None:
            safe_candidate_indexes = [
                index
                for index in risk_safe_candidate_indexes
                if float(action_set.candidates[index].calibrated_support_probability) >= hard_support_threshold
            ]
        else:
            safe_candidate_indexes = risk_safe_candidate_indexes
        selected_candidate_index = safe_candidate_indexes[0] if safe_candidate_indexes else None

        if selected_candidate_index is None:
            abstain_reason = "support_shield" if risk_safe_candidate_indexes and hard_support_threshold is not None else "risk_shield"
            predictions[action_set.example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "selected_action_probability": abstain_probability,
                    "selected_risk_probability": float(action_output["candidate_risk_probabilities"][top_candidate_index]),
                    "selected_support_probability": float(action_set.candidates[top_candidate_index].calibrated_support_probability),
                    "best_candidate_score": top_candidate_score,
                    "abstain_score": abstain_score,
                },
                "abstain_reason": abstain_reason,
            }
            continue

        selected_candidate = action_set.candidates[selected_candidate_index]
        selected_candidate_score = float(action_output["candidate_action_scores"][selected_candidate_index])
        selected_risk_probability = float(action_output["candidate_risk_probabilities"][selected_candidate_index])
        selected_probability = float(action_output["candidate_action_probabilities"][selected_candidate_index])

        if abstain_score >= selected_candidate_score:
            predictions[action_set.example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "selected_action_probability": abstain_probability,
                    "selected_risk_probability": selected_risk_probability,
                    "selected_support_probability": float(selected_candidate.calibrated_support_probability),
                    "best_candidate_score": selected_candidate_score,
                    "abstain_score": abstain_score,
                },
                "abstain_reason": "abstain_action",
            }
            continue

        predictions[action_set.example_id] = {
            "decision": "answer",
            "answer": selected_candidate.answer_text,
            "scores": {
                "selected_action_probability": selected_probability,
                "selected_risk_probability": selected_risk_probability,
                "best_candidate_score": selected_candidate_score,
                "abstain_score": abstain_score,
                "span_score": selected_candidate.span_score,
                "keep_probability": selected_candidate.keep_probability,
                "support_probability": selected_candidate.support_probability,
                "stage5_support_probability": selected_candidate.stage5_support_probability,
                "stage6_controller_probability": selected_candidate.stage6_controller_probability,
                "raw_verifier_support_probability": selected_candidate.raw_verifier_support_probability,
                "calibrated_support_probability": selected_candidate.calibrated_support_probability,
            },
            "support": {"score": selected_candidate.calibrated_support_probability},
            "risk": {"score": selected_risk_probability},
            "action": {
                "selected_answer": selected_candidate.answer_text,
                "selected_rank": selected_candidate.normalized_rank,
                "question_overlap": selected_candidate.question_overlap,
            },
        }

    return predictions


def select_stage7_threshold_entry(
    sweep: list[dict[str, float]],
    *,
    max_unsupported_answer_rate: float,
    max_overabstain_rate: float,
) -> dict[str, float]:
    if not sweep:
        raise ValueError("Threshold sweep is empty.")

    constrained = [
        entry
        for entry in sweep
        if bool(entry.get("constraint_satisfied", False))
    ]
    pool = constrained or sweep
    if constrained:
        return max(
            pool,
            key=lambda entry: (
                float(entry["overall_f1"]),
                float(entry["answerable_f1"]),
                float(entry["supported_answer_rate"]),
                -float(entry["unsupported_answer_rate"]),
                -float(entry["overabstain_rate"]),
                float(entry["abstain_f1"]),
            ),
        )
    return max(
        pool,
        key=lambda entry: (
            -float(entry["unsupported_answer_rate"]),
            -float(entry["overabstain_rate"]),
            float(entry["overall_f1"]),
            float(entry["answerable_f1"]),
            float(entry["supported_answer_rate"]),
        ),
    )


def search_risk_threshold(
    action_sets: list[ActionSet],
    action_outputs: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    match_f1_threshold: float,
    max_unsupported_answer_rate: float,
    max_overabstain_rate: float,
    hard_support_threshold: float | None = None,
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float], list[dict[str, float]]]:
    sweep: list[dict[str, float]] = []
    current = float(threshold_min)
    while current <= threshold_max + 1e-9:
        predictions = postprocess_action_predictions(
            action_sets,
            action_outputs,
            risk_threshold=current,
            hard_support_threshold=hard_support_threshold,
        )
        metrics = compute_stage1_metrics(predictions, references)
        mix = compute_answer_support_mix(
            predictions,
            references,
            match_f1_threshold=match_f1_threshold,
        )
        overabstain = compute_overabstain_stats(predictions, references)
        sweep.append(
            {
                "threshold": round(current, 10),
                "constraint_satisfied": (
                    metrics["unsupported_answer_rate"] <= max_unsupported_answer_rate
                    and overabstain["overabstain_rate"] <= max_overabstain_rate
                ),
                **metrics,
                **mix,
                **overabstain,
            }
        )
        current += threshold_step

    best_entry = select_stage7_threshold_entry(
        sweep,
        max_unsupported_answer_rate=max_unsupported_answer_rate,
        max_overabstain_rate=max_overabstain_rate,
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
            "overabstain_rate",
            "overabstain_count",
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
    best_overabstain = {
        "overabstain_rate": float(best_entry["overabstain_rate"]),
        "overabstain_count": float(best_entry["overabstain_count"]),
    }
    return best_threshold, best_metrics, best_mix, best_overabstain, sweep


def _is_better_validation_entry(
    candidate: dict[str, float],
    best: dict[str, float] | None,
    *,
    max_unsupported_answer_rate: float,
    max_overabstain_rate: float,
) -> bool:
    if best is None:
        return True

    candidate_satisfied = (
        candidate["validation_unsupported_answer_rate"] <= max_unsupported_answer_rate
        and candidate["validation_overabstain_rate"] <= max_overabstain_rate
    )
    best_satisfied = (
        best["validation_unsupported_answer_rate"] <= max_unsupported_answer_rate
        and best["validation_overabstain_rate"] <= max_overabstain_rate
    )
    if candidate_satisfied != best_satisfied:
        return candidate_satisfied
    if candidate_satisfied:
        return (
            candidate["validation_overall_f1"],
            candidate["validation_answerable_f1"],
            candidate["validation_supported_answer_rate"],
            -candidate["validation_unsupported_answer_rate"],
            -candidate["validation_overabstain_rate"],
            candidate["validation_abstain_f1"],
        ) > (
            best["validation_overall_f1"],
            best["validation_answerable_f1"],
            best["validation_supported_answer_rate"],
            -best["validation_unsupported_answer_rate"],
            -best["validation_overabstain_rate"],
            best["validation_abstain_f1"],
        )
    return (
        -candidate["validation_unsupported_answer_rate"],
        -candidate["validation_overabstain_rate"],
        candidate["validation_overall_f1"],
        candidate["validation_answerable_f1"],
        candidate["validation_supported_answer_rate"],
    ) > (
        -best["validation_unsupported_answer_rate"],
        -best["validation_overabstain_rate"],
        best["validation_overall_f1"],
        best["validation_answerable_f1"],
        best["validation_supported_answer_rate"],
    )


def _save_action_model(
    model: RiskBudgetedActionModel,
    output_dir: Path,
    config: RiskBudgetedActionConfig,
    training_history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / ACTION_MODEL_STATE_FILENAME)
    (output_dir / ACTION_MODEL_CONFIG_FILENAME).write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / ACTION_TRAINING_HISTORY_FILENAME).write_text(
        json.dumps(training_history, indent=2) + "\n",
        encoding="utf-8",
    )


def load_action_model(
    model_path: Path,
) -> tuple[RiskBudgetedActionModel, RiskBudgetedActionConfig]:
    config_payload = json.loads((model_path / ACTION_MODEL_CONFIG_FILENAME).read_text(encoding="utf-8"))
    config = RiskBudgetedActionConfig(**config_payload)
    model = RiskBudgetedActionModel(
        config.input_dim,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    )
    state_dict = torch.load(model_path / ACTION_MODEL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    return model, config


def _resolve_stage6_candidate_probabilities(
    bundle: CandidateBundle,
    *,
    stage6_controller_path: Path | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray | None:
    if stage6_controller_path is None:
        return None
    if bundle.features.size == 0:
        return np.zeros((0,), dtype=np.float32)

    controller, controller_config = load_controller(stage6_controller_path)
    controller = controller.to(device)
    feature_mean = np.asarray(controller_config.feature_mean, dtype=np.float32)
    feature_std = np.asarray(controller_config.feature_std, dtype=np.float32)
    standardized = standardize_features(bundle.features, feature_mean, feature_std)
    return _predict_candidate_probabilities(
        controller,
        standardized,
        batch_size=batch_size,
        device=device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Stage 7 risk-budgeted action learner.")
    train_parser.add_argument("--stage5-model-path", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--stage6-controller-path", type=Path, default=None)
    train_parser.add_argument("--control-path", type=Path, default=None)
    train_parser.add_argument("--verifier-model-path", type=Path, default=None)
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    train_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    train_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    train_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    train_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--eval-batch-size", type=int, default=32)
    train_parser.add_argument("--train-batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=2e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--num-train-epochs", type=int, default=12)
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--dropout", type=float, default=0.10)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-eval-samples", type=int, default=None)
    train_parser.add_argument(
        "--clean-splitting",
        action="store_true",
        help="Use train/validation/test splits and keep the official SQuAD validation split untouched for final testing.",
    )
    train_parser.add_argument("--max-test-samples", type=int, default=None)
    train_parser.add_argument("--max-candidates-per-example", type=int, default=6)
    train_parser.add_argument("--max-candidates-per-feature", type=int, default=3)
    train_parser.add_argument("--action-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--utility-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--risk-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--tail-risk-weight", type=float, default=2.0)
    train_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE7_MAX_UNSUPPORTED_ANSWER_RATE,
    )
    train_parser.add_argument(
        "--max-overabstain-rate",
        type=float,
        default=DEFAULT_STAGE7_MAX_OVERABSTAIN_RATE,
    )
    train_parser.add_argument("--unsafe-dual-init", type=float, default=1.0)
    train_parser.add_argument("--overabstain-dual-init", type=float, default=1.0)
    train_parser.add_argument("--unsafe-dual-lr", type=float, default=0.25)
    train_parser.add_argument("--overabstain-dual-lr", type=float, default=0.25)
    train_parser.add_argument("--hard-risk-threshold", type=float, default=0.35)
    train_parser.add_argument("--hard-support-threshold", type=float, default=None)
    train_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the Stage 7 risk-budgeted action learner.")
    eval_parser.add_argument("--model-path", type=Path, required=True)
    eval_parser.add_argument("--output-path", type=Path, required=True)
    eval_parser.add_argument("--stage5-model-path", type=Path, default=None)
    eval_parser.add_argument("--stage6-controller-path", type=Path, default=None)
    eval_parser.add_argument("--control-path", type=Path, default=None)
    eval_parser.add_argument("--verifier-model-path", type=Path, default=None)
    eval_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    eval_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    eval_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    eval_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    eval_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    eval_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    eval_parser.add_argument("--eval-batch-size", type=int, default=32)
    eval_parser.add_argument("--max-eval-samples", type=int, default=None)
    eval_parser.add_argument(
        "--clean-splitting",
        action="store_true",
        help="Use train/validation/test splits and report final results on the untouched test split.",
    )
    eval_parser.add_argument("--max-test-samples", type=int, default=None)
    eval_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)
    eval_parser.add_argument(
        "--risk-threshold-min",
        type=float,
        default=DEFAULT_STAGE7_RISK_THRESHOLD_MIN,
    )
    eval_parser.add_argument(
        "--risk-threshold-max",
        type=float,
        default=DEFAULT_STAGE7_RISK_THRESHOLD_MAX,
    )
    eval_parser.add_argument(
        "--risk-threshold-step",
        type=float,
        default=DEFAULT_STAGE7_RISK_THRESHOLD_STEP,
    )
    eval_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE7_MAX_UNSUPPORTED_ANSWER_RATE,
    )
    eval_parser.add_argument(
        "--max-overabstain-rate",
        type=float,
        default=DEFAULT_STAGE7_MAX_OVERABSTAIN_RATE,
    )
    eval_parser.add_argument("--hard-support-threshold", type=float, default=None)
    return parser


def _train_action_learner(args: argparse.Namespace) -> None:
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
        clean_splitting=args.clean_splitting,
        max_test_samples=args.max_test_samples,
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
        hard_negative_weight=args.tail_risk_weight,
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
        hard_negative_weight=args.tail_risk_weight,
    )

    if train_bundle.features.size == 0:
        raise ValueError("No Stage 7 training candidates were generated.")
    if validation_bundle.features.size == 0:
        raise ValueError("No Stage 7 validation candidates were generated.")

    control_runtime = _prepare_control_runtime(
        control_path=args.control_path,
        verifier_model_path=args.verifier_model_path,
        hard_support_threshold=args.hard_support_threshold,
        eval_batch_size=args.eval_batch_size,
        output_dir=args.output_dir / "tmp-stage7-control-train",
    )
    stage6_train_probabilities = _resolve_stage6_candidate_probabilities(
        train_bundle,
        stage6_controller_path=args.stage6_controller_path,
        batch_size=args.eval_batch_size,
        device=device,
    )
    stage6_validation_probabilities = _resolve_stage6_candidate_probabilities(
        validation_bundle,
        stage6_controller_path=args.stage6_controller_path,
        batch_size=args.eval_batch_size,
        device=device,
    )
    train_raw_support_probabilities, train_calibrated_support_probabilities = _score_bundle_with_control(
        artifacts["train"]["examples"],
        train_bundle,
        control_runtime=control_runtime,
        max_length=args.max_length,
    )
    validation_raw_support_probabilities, validation_calibrated_support_probabilities = _score_bundle_with_control(
        artifacts["validation"]["examples"],
        validation_bundle,
        control_runtime=control_runtime,
        max_length=args.max_length,
    )

    train_action_sets = build_action_sets(
        artifacts["train"]["examples"],
        train_bundle,
        stage6_candidate_probabilities=stage6_train_probabilities,
        raw_verifier_support_probabilities=train_raw_support_probabilities,
        calibrated_support_probabilities=train_calibrated_support_probabilities,
        hard_support_threshold=(
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
    )
    validation_action_sets = build_action_sets(
        artifacts["validation"]["examples"],
        validation_bundle,
        stage6_candidate_probabilities=stage6_validation_probabilities,
        raw_verifier_support_probabilities=validation_raw_support_probabilities,
        calibrated_support_probabilities=validation_calibrated_support_probabilities,
        hard_support_threshold=(
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
    )

    train_feature_matrix = flatten_action_features(train_action_sets)
    if train_feature_matrix.size == 0:
        raise ValueError("Stage 7 training action sets did not contain any candidates.")
    feature_mean, feature_std = fit_feature_standardization(train_feature_matrix)
    standardized_train_sets = standardize_action_sets(train_action_sets, feature_mean, feature_std)
    standardized_validation_sets = standardize_action_sets(validation_action_sets, feature_mean, feature_std)

    model = RiskBudgetedActionModel(
        len(ACTION_FEATURE_NAMES),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = DataLoader(
        standardized_train_sets,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_action_sets,
    )

    unsafe_dual = float(args.unsafe_dual_init)
    overabstain_dual = float(args.overabstain_dual_init)
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_entry: dict[str, float] | None = None
    best_unsafe_dual = unsafe_dual
    best_overabstain_dual = overabstain_dual
    training_history: list[dict[str, float]] = []
    validation_references = artifacts["validation"]["references"]

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        total_loss = 0.0
        total_action_loss = 0.0
        total_utility_loss = 0.0
        total_risk_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            batch_features = batch["candidate_features"].to(device)
            batch_labels = batch["candidate_labels"].to(device)
            batch_risk_labels = batch["candidate_risk_labels"].to(device)
            batch_weights = batch["candidate_weights"].to(device)
            batch_mask = batch["candidate_mask"].to(device)
            batch_targets = batch["target_indices"].to(device)

            outputs = model(
                batch_features,
                batch_mask,
                unsafe_dual=unsafe_dual,
                overabstain_dual=overabstain_dual,
            )
            loss, components = compute_action_losses(
                outputs,
                {
                    "candidate_labels": batch_labels,
                    "candidate_risk_labels": batch_risk_labels,
                    "candidate_weights": batch_weights,
                    "candidate_mask": batch_mask,
                    "target_indices": batch_targets,
                },
                action_loss_weight=args.action_loss_weight,
                utility_loss_weight=args.utility_loss_weight,
                risk_loss_weight=args.risk_loss_weight,
                tail_risk_weight=args.tail_risk_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_action_loss += float(components["action_loss"])
            total_utility_loss += float(components["utility_loss"])
            total_risk_loss += float(components["risk_loss"])
            batch_count += 1

        validation_outputs = _predict_action_outputs(
            model,
            standardized_validation_sets,
            batch_size=args.eval_batch_size,
            device=device,
            unsafe_dual=unsafe_dual,
            overabstain_dual=overabstain_dual,
        )
        validation_predictions = postprocess_action_predictions(
            standardized_validation_sets,
            validation_outputs,
            risk_threshold=args.hard_risk_threshold,
            hard_support_threshold=(
                None
                if control_runtime is None
                else float(control_runtime["control_settings"]["hard_support_threshold"])
            ),
        )
        validation_metrics = compute_stage1_metrics(validation_predictions, validation_references)
        validation_mix = compute_answer_support_mix(
            validation_predictions,
            validation_references,
            match_f1_threshold=args.match_f1_threshold,
        )
        validation_overabstain = compute_overabstain_stats(validation_predictions, validation_references)
        validation_losses = _evaluate_action_model(
            model,
            standardized_validation_sets,
            batch_size=args.eval_batch_size,
            device=device,
            unsafe_dual=unsafe_dual,
            overabstain_dual=overabstain_dual,
            action_loss_weight=args.action_loss_weight,
            utility_loss_weight=args.utility_loss_weight,
            risk_loss_weight=args.risk_loss_weight,
            tail_risk_weight=args.tail_risk_weight,
        )

        history_entry = {
            "epoch": float(epoch),
            "unsafe_dual": float(unsafe_dual),
            "overabstain_dual": float(overabstain_dual),
            "train_total_loss": float(total_loss / max(1, batch_count)),
            "train_action_loss": float(total_action_loss / max(1, batch_count)),
            "train_utility_loss": float(total_utility_loss / max(1, batch_count)),
            "train_risk_loss": float(total_risk_loss / max(1, batch_count)),
            "validation_total_loss": float(validation_losses["total_loss"]),
            "validation_action_loss": float(validation_losses["action_loss"]),
            "validation_utility_loss": float(validation_losses["utility_loss"]),
            "validation_risk_loss": float(validation_losses["risk_loss"]),
            "validation_overall_f1": float(validation_metrics["overall_f1"]),
            "validation_answerable_f1": float(validation_metrics["answerable_f1"]),
            "validation_unsupported_answer_rate": float(validation_metrics["unsupported_answer_rate"]),
            "validation_supported_answer_rate": float(validation_mix["supported_answer_rate"]),
            "validation_answer_rate": float(validation_mix["answer_rate"]),
            "validation_abstain_f1": float(validation_metrics["abstain_f1"]),
            "validation_overabstain_rate": float(validation_overabstain["overabstain_rate"]),
        }
        training_history.append(history_entry)

        if _is_better_validation_entry(
            history_entry,
            best_entry,
            max_unsupported_answer_rate=args.max_unsupported_answer_rate,
            max_overabstain_rate=args.max_overabstain_rate,
        ):
            best_entry = history_entry
            best_state_dict = copy.deepcopy(model.state_dict())
            best_unsafe_dual = float(unsafe_dual)
            best_overabstain_dual = float(overabstain_dual)

        unsafe_target = max(float(args.max_unsupported_answer_rate), 1.0)
        unsafe_gap = float(validation_metrics["unsupported_answer_rate"]) - float(args.max_unsupported_answer_rate)
        unsafe_dual = float(min(20.0, max(0.0, unsafe_dual + float(args.unsafe_dual_lr) * (unsafe_gap / unsafe_target))))

        overabstain_target = max(float(args.max_overabstain_rate), 1.0)
        overabstain_gap = float(validation_overabstain["overabstain_rate"]) - float(args.max_overabstain_rate)
        overabstain_dual = float(
            min(
                20.0,
                max(0.0, overabstain_dual + float(args.overabstain_dual_lr) * (overabstain_gap / overabstain_target)),
            )
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    config = RiskBudgetedActionConfig(
        stage5_model_path=str(args.stage5_model_path),
        stage6_controller_path=str(args.stage6_controller_path) if args.stage6_controller_path is not None else None,
        input_dim=len(ACTION_FEATURE_NAMES),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        feature_names=list(ACTION_FEATURE_NAMES),
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        max_candidates_per_example=args.max_candidates_per_example,
        max_candidates_per_feature=args.max_candidates_per_feature,
        validation_size=args.validation_size,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        utility_loss_weight=args.utility_loss_weight,
        risk_loss_weight=args.risk_loss_weight,
        action_loss_weight=args.action_loss_weight,
        tail_risk_weight=args.tail_risk_weight,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
        max_overabstain_rate=args.max_overabstain_rate,
        unsafe_dual=best_unsafe_dual,
        overabstain_dual=best_overabstain_dual,
        unsafe_dual_lr=args.unsafe_dual_lr,
        overabstain_dual_lr=args.overabstain_dual_lr,
        hard_risk_threshold=args.hard_risk_threshold,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        clean_splitting=args.clean_splitting,
        max_test_samples=args.max_test_samples,
        control_path=str(args.control_path) if args.control_path is not None else None,
        verifier_model_path=(
            None
            if control_runtime is None
            else str(control_runtime["control_settings"]["verifier_model_path"])
        ),
        support_temperature=(
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["support_temperature"])
        ),
        hard_support_threshold=(
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
    )
    _save_action_model(model.cpu(), args.output_dir, config, training_history)

    run_metadata = {
        "stage": "stage7-risk-budgeted-action-train",
        "stage5_model_path": str(args.stage5_model_path),
        "stage6_controller_path": str(args.stage6_controller_path) if args.stage6_controller_path is not None else None,
        "control_path": str(args.control_path) if args.control_path is not None else None,
        "train_examples": len(artifacts["train"]["examples"]),
        "validation_examples": len(artifacts["validation"]["examples"]),
        "train_candidate_count": len(train_bundle.records),
        "validation_candidate_count": len(validation_bundle.records),
        "best_epoch": best_entry["epoch"] if best_entry is not None else None,
        "unsafe_dual": best_unsafe_dual,
        "overabstain_dual": best_overabstain_dual,
        "hard_support_threshold": (
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
        "clean_splitting": bool(args.clean_splitting),
        "max_test_samples": args.max_test_samples,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _evaluate_action_learner(args: argparse.Namespace) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model, config = load_action_model(args.model_path)
    resolved_stage5_model_path = args.stage5_model_path or Path(config.stage5_model_path)
    resolved_stage6_controller_path = (
        args.stage6_controller_path
        if args.stage6_controller_path is not None
        else (Path(config.stage6_controller_path) if config.stage6_controller_path is not None else None)
    )
    resolved_control_path = (
        args.control_path
        if args.control_path is not None
        else (Path(config.control_path) if getattr(config, "control_path", None) is not None else None)
    )
    resolved_verifier_model_path = (
        args.verifier_model_path
        if args.verifier_model_path is not None
        else (
            Path(config.verifier_model_path)
            if getattr(config, "verifier_model_path", None) is not None
            else None
        )
    )
    resolved_hard_support_threshold = (
        args.hard_support_threshold
        if args.hard_support_threshold is not None
        else getattr(config, "hard_support_threshold", None)
    )
    clean_splitting = bool(args.clean_splitting or getattr(config, "clean_splitting", False))
    resolved_max_test_samples = (
        args.max_test_samples
        if args.max_test_samples is not None
        else getattr(config, "max_test_samples", None)
    )
    final_split_name = "test" if clean_splitting else "dev"
    feature_mean = np.asarray(config.feature_mean, dtype=np.float32)
    feature_std = np.asarray(config.feature_std, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    artifacts = _prepare_stage5_prediction_artifacts(
        model_path=Path(resolved_stage5_model_path),
        validation_size=args.validation_size,
        seed=args.seed,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        eval_batch_size=args.eval_batch_size,
        max_train_samples=None,
        max_eval_samples=args.max_eval_samples,
        clean_splitting=clean_splitting,
        max_test_samples=resolved_max_test_samples,
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
        hard_negative_weight=config.tail_risk_weight,
    )
    final_bundle = build_candidate_bundle(
        artifacts[final_split_name]["examples"],
        artifacts[final_split_name]["features"],
        artifacts[final_split_name]["raw_predictions"],
        artifacts[final_split_name]["references"],
        n_best_size=config.n_best_size,
        max_answer_length=config.max_answer_length,
        max_candidates_per_example=config.max_candidates_per_example,
        max_candidates_per_feature=config.max_candidates_per_feature,
        match_f1_threshold=args.match_f1_threshold,
        hard_negative_weight=config.tail_risk_weight,
    )
    control_runtime = _prepare_control_runtime(
        control_path=resolved_control_path,
        verifier_model_path=resolved_verifier_model_path,
        hard_support_threshold=resolved_hard_support_threshold,
        eval_batch_size=args.eval_batch_size,
        output_dir=args.output_path.parent / "tmp-stage7-control-eval",
    )

    validation_stage6_probabilities = _resolve_stage6_candidate_probabilities(
        validation_bundle,
        stage6_controller_path=resolved_stage6_controller_path,
        batch_size=args.eval_batch_size,
        device=device,
    )
    final_stage6_probabilities = _resolve_stage6_candidate_probabilities(
        final_bundle,
        stage6_controller_path=resolved_stage6_controller_path,
        batch_size=args.eval_batch_size,
        device=device,
    )
    validation_raw_support_probabilities, validation_calibrated_support_probabilities = _score_bundle_with_control(
        artifacts["validation"]["examples"],
        validation_bundle,
        control_runtime=control_runtime,
        max_length=args.max_length,
    )
    final_raw_support_probabilities, final_calibrated_support_probabilities = _score_bundle_with_control(
        artifacts[final_split_name]["examples"],
        final_bundle,
        control_runtime=control_runtime,
        max_length=args.max_length,
    )

    validation_action_sets = build_action_sets(
        artifacts["validation"]["examples"],
        validation_bundle,
        stage6_candidate_probabilities=validation_stage6_probabilities,
        raw_verifier_support_probabilities=validation_raw_support_probabilities,
        calibrated_support_probabilities=validation_calibrated_support_probabilities,
        hard_support_threshold=(
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
    )
    final_action_sets = build_action_sets(
        artifacts[final_split_name]["examples"],
        final_bundle,
        stage6_candidate_probabilities=final_stage6_probabilities,
        raw_verifier_support_probabilities=final_raw_support_probabilities,
        calibrated_support_probabilities=final_calibrated_support_probabilities,
        hard_support_threshold=(
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
    )
    standardized_validation_sets = standardize_action_sets(validation_action_sets, feature_mean, feature_std)
    standardized_final_sets = standardize_action_sets(final_action_sets, feature_mean, feature_std)

    validation_outputs = _predict_action_outputs(
        model,
        standardized_validation_sets,
        batch_size=config.eval_batch_size,
        device=device,
        unsafe_dual=config.unsafe_dual,
        overabstain_dual=config.overabstain_dual,
    )
    final_outputs = _predict_action_outputs(
        model,
        standardized_final_sets,
        batch_size=config.eval_batch_size,
        device=device,
        unsafe_dual=config.unsafe_dual,
        overabstain_dual=config.overabstain_dual,
    )

    selected_threshold, validation_metrics, validation_mix, validation_overabstain, threshold_sweep = search_risk_threshold(
        standardized_validation_sets,
        validation_outputs,
        artifacts["validation"]["references"],
        threshold_min=args.risk_threshold_min,
        threshold_max=args.risk_threshold_max,
        threshold_step=args.risk_threshold_step,
        match_f1_threshold=args.match_f1_threshold,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
        max_overabstain_rate=args.max_overabstain_rate,
        hard_support_threshold=(
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
    )

    final_predictions = postprocess_action_predictions(
        standardized_final_sets,
        final_outputs,
        risk_threshold=selected_threshold,
        hard_support_threshold=(
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
    )
    final_metrics = compute_stage1_metrics(final_predictions, artifacts[final_split_name]["references"])
    final_mix = compute_answer_support_mix(
        final_predictions,
        artifacts[final_split_name]["references"],
        match_f1_threshold=args.match_f1_threshold,
    )
    final_overabstain = compute_overabstain_stats(final_predictions, artifacts[final_split_name]["references"])

    output = {
        "stage": "stage7-risk-budgeted-action-eval",
        "model_path": str(args.model_path),
        "stage5_model_path": str(resolved_stage5_model_path),
        "stage6_controller_path": str(resolved_stage6_controller_path) if resolved_stage6_controller_path is not None else None,
        "control_path": str(resolved_control_path) if resolved_control_path is not None else None,
        "verifier_model_path": (
            None
            if control_runtime is None
            else str(control_runtime["control_settings"]["verifier_model_path"])
        ),
        "support_temperature": (
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["support_temperature"])
        ),
        "hard_support_threshold": (
            None
            if control_runtime is None
            else float(control_runtime["control_settings"]["hard_support_threshold"])
        ),
        "clean_splitting": clean_splitting,
        "final_eval_split": final_split_name,
        "selected_risk_threshold": selected_threshold,
        "max_unsupported_answer_rate": args.max_unsupported_answer_rate,
        "max_overabstain_rate": args.max_overabstain_rate,
        "unsafe_dual": config.unsafe_dual,
        "overabstain_dual": config.overabstain_dual,
        "validation_metrics": validation_metrics,
        "validation_mix": validation_mix,
        "validation_overabstain": validation_overabstain,
        "threshold_sweep": threshold_sweep,
        "final_metrics": final_metrics,
        "final_mix": final_mix,
        "final_overabstain": final_overabstain,
        "final_predictions": final_predictions,
        "feature_names": list(config.feature_names),
    }
    if final_split_name == "dev":
        output["dev_metrics"] = final_metrics
        output["dev_mix"] = final_mix
        output["dev_overabstain"] = final_overabstain
        output["dev_predictions"] = final_predictions
    else:
        output["test_metrics"] = final_metrics
        output["test_mix"] = final_mix
        output["test_overabstain"] = final_overabstain
        output["test_predictions"] = final_predictions
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        _train_action_learner(args)
        return
    if args.command == "evaluate":
        _evaluate_action_learner(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
