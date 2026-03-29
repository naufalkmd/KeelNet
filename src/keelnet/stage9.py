"""Stage 9 exact risk-generalization architecture with a self-contained proposal and support stack."""

from __future__ import annotations

import argparse
import copy
import inspect
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
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoModel, AutoTokenizer
from transformers.utils import ModelOutput

STAGE9_MODEL_STATE_FILENAME = "stage9_model.pt"
STAGE9_MODEL_CONFIG_FILENAME = "stage9_config.json"
STAGE9_TRAINING_HISTORY_FILENAME = "stage9_training_history.json"
STAGE9_PROPOSAL_DIRNAME = "proposal-model"
STAGE9_SUPPORT_DIRNAME = "support-scorer"
STAGE9_PROPOSAL_STATE_FILENAME = "stage9_proposal_model.pt"
STAGE9_PROPOSAL_CONFIG_FILENAME = "stage9_proposal_model_config.json"

DEFAULT_MAX_LENGTH = 384
DEFAULT_DOC_STRIDE = 128
DEFAULT_MAX_ANSWER_LENGTH = 30
DEFAULT_N_BEST_SIZE = 20
DEFAULT_VALIDATION_SIZE = 0.1
DEFAULT_SEED = 42
DEFAULT_STAGE9_BASE_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_SUPPORT_MATCH_F1 = 0.5
DEFAULT_STAGE9_MAX_UNSUPPORTED_ANSWER_RATE = 20.0
DEFAULT_STAGE9_MAX_OVERABSTAIN_RATE = 20.0
DEFAULT_STAGE9_RISK_THRESHOLD_MIN = 0.10
DEFAULT_STAGE9_RISK_THRESHOLD_MAX = 0.90
DEFAULT_STAGE9_RISK_THRESHOLD_STEP = 0.05
DEFAULT_STAGE9_RANDOMIZATION_SCALE = 0.10
DEFAULT_STAGE9_SUPPORT_THRESHOLD_MIN = 0.30
DEFAULT_STAGE9_SUPPORT_THRESHOLD_MAX = 0.90
DEFAULT_STAGE9_SUPPORT_THRESHOLD_STEP = 0.05

STAGE9_DEFAULT_CALIBRATION_BINS = 10
STAGE9_DEFAULT_TEMPERATURE_MIN = 0.25
STAGE9_DEFAULT_TEMPERATURE_MAX = 5.0
STAGE9_DEFAULT_TEMPERATURE_STEP = 0.05
STAGE9_DEFAULT_THRESHOLD_GAP_MIN_COUNT = 25

STAGE9_CANDIDATE_FEATURE_NAMES = [
    "span_score",
    "score_gap_to_best",
    "score_margin_to_next",
    "keep_probability",
    "proposal_support_probability",
    "abstain_probability",
    "raw_support_scorer_probability",
    "calibrated_support_probability",
    "support_gap",
    "support_gate_pass",
    "answer_length_tokens",
    "normalized_rank",
    "question_overlap",
    "keep_uncertainty",
    "support_uncertainty",
    "keep_support_disagreement",
    "keep_gap_to_best",
    "support_gap_to_best",
]

STAGE9_INTERACTION_FEATURE_NAMES = [
    "candidate_count",
    "top_keep_gap",
    "top_support_gap",
    "keep_entropy",
    "support_entropy",
    "keep_spread",
    "support_spread",
    "mean_keep_support_disagreement",
    "max_keep_support_disagreement",
    "safe_candidate_fraction",
]

QUESTION_TYPE_PREFIXES = ("who", "what", "where", "when", "which", "how", "why", "other")

STAGE9_DOMAIN_FEATURE_NAMES = [
    "question_type_who",
    "question_type_what",
    "question_type_where",
    "question_type_when",
    "question_type_which",
    "question_type_how",
    "question_type_why",
    "question_type_other",
    "question_length_norm",
    "context_length_norm",
    "question_has_digit",
    "context_has_digit",
]


def trainer_processing_kwargs(trainer_cls: type, processor: Any) -> dict[str, Any]:
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": processor}
    if "tokenizer" in parameters:
        return {"tokenizer": processor}
    return {}


def training_arguments_eval_strategy_kwargs(training_arguments_cls: type, strategy: str) -> dict[str, Any]:
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in parameters:
        return {"eval_strategy": strategy}
    if "evaluation_strategy" in parameters:
        return {"evaluation_strategy": strategy}
    return {}


def normalize_answer(text: str) -> str:
    import re
    import string

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        punctuation = set(string.punctuation)
        return "".join(ch for ch in value if ch not in punctuation)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    num_same = 0
    for token in truth_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, truth) for truth in ground_truths)


def _percentage(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def is_supported_answer(
    answer_text: str,
    gold_answers: list[str],
    *,
    match_f1_threshold: float,
) -> bool:
    normalized_answer = normalize_answer(answer_text)
    if not normalized_answer or not gold_answers:
        return False
    return metric_max_over_ground_truths(f1_score, answer_text, gold_answers) >= match_f1_threshold


def compute_stage1_metrics(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> dict[str, float]:
    answerable_total = 0
    unanswerable_total = 0
    answerable_em_sum = 0.0
    answerable_f1_sum = 0.0
    overall_em_sum = 0.0
    overall_f1_sum = 0.0
    unsupported_answers = 0
    abstain_tp = 0
    abstain_fp = 0
    abstain_fn = 0

    for example_id, reference in references.items():
        prediction = predictions.get(example_id, {"decision": "abstain", "answer": "", "scores": {}})
        decision = str(prediction.get("decision", "abstain")).lower()
        answer_text = str(prediction.get("answer", ""))
        predicted_abstain = decision == "abstain"
        gold_answers = list(reference.get("answers", []))
        gold_abstain = not bool(reference.get("is_answerable", False))

        em = 0.0
        f1 = 0.0
        if predicted_abstain and gold_abstain:
            em = 1.0
            f1 = 1.0
        elif not predicted_abstain and gold_answers:
            em = metric_max_over_ground_truths(exact_match_score, answer_text, gold_answers)
            f1 = metric_max_over_ground_truths(f1_score, answer_text, gold_answers)

        overall_em_sum += em
        overall_f1_sum += f1

        if gold_abstain:
            unanswerable_total += 1
            if not predicted_abstain:
                unsupported_answers += 1
        else:
            answerable_total += 1
            answerable_em_sum += em
            answerable_f1_sum += f1

        if predicted_abstain and gold_abstain:
            abstain_tp += 1
        elif predicted_abstain and not gold_abstain:
            abstain_fp += 1
        elif (not predicted_abstain) and gold_abstain:
            abstain_fn += 1

    abstain_precision = abstain_tp / (abstain_tp + abstain_fp) if abstain_tp + abstain_fp > 0 else 0.0
    abstain_recall = abstain_tp / (abstain_tp + abstain_fn) if abstain_tp + abstain_fn > 0 else 0.0
    abstain_f1 = (
        2 * abstain_precision * abstain_recall / (abstain_precision + abstain_recall)
        if abstain_precision + abstain_recall > 0
        else 0.0
    )

    total_examples = len(references)
    return {
        "answerable_em": _percentage(answerable_em_sum, answerable_total),
        "answerable_f1": _percentage(answerable_f1_sum, answerable_total),
        "overall_em": _percentage(overall_em_sum, total_examples),
        "overall_f1": _percentage(overall_f1_sum, total_examples),
        "unsupported_answer_rate": _percentage(unsupported_answers, unanswerable_total),
        "abstain_precision": 100.0 * abstain_precision,
        "abstain_recall": 100.0 * abstain_recall,
        "abstain_f1": 100.0 * abstain_f1,
        "answerable_count": float(answerable_total),
        "unanswerable_count": float(unanswerable_total),
    }


def compute_answer_support_mix(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    match_f1_threshold: float,
) -> dict[str, float]:
    total_examples = len(references)
    answered_total = 0
    supported_answers = 0
    unsupported_answers = 0

    for example_id, reference in references.items():
        prediction = predictions.get(example_id, {"decision": "abstain", "answer": ""})
        if str(prediction.get("decision", "abstain")).lower() != "answer":
            continue
        answered_total += 1
        answer_text = str(prediction.get("answer", ""))
        gold_answers = list(reference.get("answers", []))
        supported = bool(reference.get("is_answerable", False)) and is_supported_answer(
            answer_text,
            gold_answers,
            match_f1_threshold=match_f1_threshold,
        )
        if supported:
            supported_answers += 1
        else:
            unsupported_answers += 1

    return {
        "answer_rate": _percentage(answered_total, total_examples),
        "supported_answer_rate": _percentage(supported_answers, answered_total),
        "unsupported_among_answers_rate": _percentage(unsupported_answers, answered_total),
        "answered_count": float(answered_total),
        "supported_answers_count": float(supported_answers),
        "unsupported_answers_count": float(unsupported_answers),
    }


def is_answerable(example: dict[str, Any]) -> bool:
    return len(example["answers"]["text"]) > 0


def build_stage1_splits_from_raw(
    *,
    train: Dataset,
    eval_source: Dataset,
    validation_size: float,
    seed: int,
    clean_splitting: bool,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetDict:
    answerable_train = train.filter(is_answerable, desc="Filtering answerable train examples")
    unanswerable_train = train.filter(lambda example: not is_answerable(example), desc="Filtering unanswerable train examples")

    answerable_split = answerable_train.train_test_split(test_size=validation_size, seed=seed)
    unanswerable_split = unanswerable_train.train_test_split(test_size=validation_size, seed=seed)

    train_split = concatenate_datasets([answerable_split["train"], unanswerable_split["train"]]).shuffle(seed=seed)
    validation_split = concatenate_datasets([answerable_split["test"], unanswerable_split["test"]]).shuffle(seed=seed)

    if max_train_samples is not None:
        train_split = train_split.select(range(min(max_train_samples, len(train_split))))
    if max_eval_samples is not None:
        validation_split = validation_split.select(range(min(max_eval_samples, len(validation_split))))

    split_dict: dict[str, Dataset] = {"train": train_split, "validation": validation_split}
    if clean_splitting:
        test_split = eval_source
        capped_test_samples = max_test_samples if max_test_samples is not None else max_eval_samples
        if capped_test_samples is not None:
            test_split = test_split.select(range(min(capped_test_samples, len(test_split))))
        split_dict["test"] = test_split
    else:
        dev_split = eval_source
        if max_eval_samples is not None:
            dev_split = dev_split.select(range(min(max_eval_samples, len(dev_split))))
        split_dict["dev"] = dev_split
    return DatasetDict(split_dict)


def load_stage1_splits(
    *,
    validation_size: float,
    seed: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    raw = load_dataset("squad_v2")
    return build_stage1_splits_from_raw(
        train=raw["train"],
        eval_source=raw["validation"],
        validation_size=validation_size,
        seed=seed,
        clean_splitting=False,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def load_stage1_clean_splits(
    *,
    validation_size: float,
    seed: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetDict:
    raw = load_dataset("squad_v2")
    return build_stage1_splits_from_raw(
        train=raw["train"],
        eval_source=raw["validation"],
        validation_size=validation_size,
        seed=seed,
        clean_splitting=True,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        max_test_samples=max_test_samples,
    )


def build_reference_index(dataset: Dataset) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for example in dataset:
        references[str(example["id"])] = {
            "answers": list(example["answers"]["text"]),
            "is_answerable": is_answerable(example),
        }
    return references


def prepare_eval_features(examples, tokenizer, max_length: int, doc_stride: int):
    pad_on_right = tokenizer.padding_side == "right"
    questions = [question.lstrip() for question in examples["question"]]
    tokenized_examples = tokenizer(
        questions if pad_on_right else examples["context"],
        examples["context"] if pad_on_right else questions,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    example_ids = []
    cls_indices = []
    context_index = 1 if pad_on_right else 0
    for feature_index in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]
        example_ids.append(examples["id"][sample_index])
        cls_indices.append(tokenized_examples["input_ids"][feature_index].index(tokenizer.cls_token_id))
        tokenized_examples["offset_mapping"][feature_index] = [
            offset if sequence_ids[token_index] == context_index else None
            for token_index, offset in enumerate(tokenized_examples["offset_mapping"][feature_index])
        ]
    tokenized_examples["example_id"] = example_ids
    tokenized_examples["cls_index"] = cls_indices
    return tokenized_examples


def prepare_train_features(examples, tokenizer, max_length: int, doc_stride: int):
    pad_on_right = tokenizer.padding_side == "right"
    questions = [question.lstrip() for question in examples["question"]]
    tokenized_examples = tokenizer(
        questions if pad_on_right else examples["context"],
        examples["context"] if pad_on_right else questions,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    start_positions = []
    end_positions = []
    keep_labels = []
    support_labels = []
    context_index = 1 if pad_on_right else 0

    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            keep_labels.append(0.0)
            support_labels.append(0.0)
            continue

        start_char = int(answers["answer_start"][0])
        end_char = start_char + len(str(answers["text"][0]))

        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            keep_labels.append(0.0)
            support_labels.append(0.0)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

        keep_labels.append(1.0)
        support_labels.append(1.0)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    tokenized_examples["keep_labels"] = keep_labels
    tokenized_examples["support_labels"] = support_labels
    return tokenized_examples


def prepare_support_scoring_features(examples, tokenizer, max_length: int):
    queries = [
        f"Question: {question}\nAnswer: {candidate_answer}"
        for question, candidate_answer in zip(examples["question"], examples["candidate_answer"], strict=False)
    ]
    return tokenizer(
        queries,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    )


def prepare_support_training_features(examples, tokenizer, max_length: int):
    features = prepare_support_scoring_features(examples, tokenizer, max_length)
    features["labels"] = list(examples["labels"])
    return features


@dataclass
class Stage9FrozenQAOutput(ModelOutput):
    """Outputs for the internally trained proposal model inside isolated Stage 9."""

    loss: torch.Tensor | None = None
    start_logits: torch.Tensor | None = None
    end_logits: torch.Tensor | None = None
    abstain_logits: torch.Tensor | None = None
    support_logits: torch.Tensor | None = None


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
class Stage9Candidate:
    example_id: str
    answer_text: str
    span_score: float
    score_gap_to_best: float
    score_margin_to_next: float
    keep_probability: float
    proposal_support_probability: float
    abstain_probability: float
    raw_support_scorer_probability: float
    calibrated_support_probability: float
    support_gate_pass: float
    answer_length_tokens: float
    normalized_rank: float
    question_overlap: float
    label: float
    hard_negative_weight: float
    candidate_features: tuple[float, ...]


@dataclass(frozen=True)
class Stage9Set:
    example_id: str
    question: str
    answerable: bool
    target_action_index: int
    candidates: tuple[Stage9Candidate, ...]
    interaction_features: tuple[float, ...]
    domain_features: tuple[float, ...]


@dataclass(frozen=True)
class Stage9Config:
    base_model_name: str
    support_temperature: float
    hard_support_threshold: float
    candidate_input_dim: int
    interaction_input_dim: int
    domain_input_dim: int
    hidden_size: int
    dropout: float
    candidate_feature_names: list[str]
    interaction_feature_names: list[str]
    domain_feature_names: list[str]
    candidate_feature_mean: list[float]
    candidate_feature_std: list[float]
    interaction_feature_mean: list[float]
    interaction_feature_std: list[float]
    domain_feature_mean: list[float]
    domain_feature_std: list[float]
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
    abstain_loss_weight: float
    joint_loss_weight: float
    tail_risk_weight: float
    randomization_scale: float
    risk_penalty: float
    max_unsupported_answer_rate: float
    max_overabstain_rate: float
    best_validation_risk_temperature: float
    best_validation_risk_threshold: float
    best_validation_abstain_margin: float
    max_train_samples: int | None
    max_eval_samples: int | None
    clean_splitting: bool = False
    max_test_samples: int | None = None
    use_domain_features: bool = True


class FrozenStage9QAModel(nn.Module):
    """Proposal model trained inside Stage 9, then frozen for candidate generation."""

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
    ) -> Stage9FrozenQAOutput:
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

        return Stage9FrozenQAOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            abstain_logits=abstain_logits,
            support_logits=support_logits,
        )


class ExactStage9Model(nn.Module):
    """Shared candidate encoder with utility, risk, and abstain heads."""

    def __init__(
        self,
        candidate_input_dim: int,
        interaction_input_dim: int,
        domain_input_dim: int,
        *,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = max(16, int(hidden_size))
        self.candidate_input_dim = int(candidate_input_dim)
        self.interaction_input_dim = int(interaction_input_dim)
        self.domain_input_dim = int(domain_input_dim)
        self.hidden_size = hidden
        self.dropout_rate = float(dropout)

        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.utility_head = nn.Linear(hidden, 1)
        risk_input_dim = hidden + max(0, self.domain_input_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(risk_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        abstain_input_dim = hidden * 2 + self.interaction_input_dim
        self.abstain_head = nn.Sequential(
            nn.Linear(abstain_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        interaction_features: torch.Tensor,
        domain_features: torch.Tensor,
        *,
        risk_penalty: float,
    ) -> dict[str, torch.Tensor]:
        hidden = self.candidate_encoder(candidate_features)
        utility_logits = self.utility_head(hidden).squeeze(-1)

        if self.domain_input_dim > 0:
            repeated_domain = domain_features.unsqueeze(1).expand(-1, candidate_features.size(1), -1)
            risk_inputs = torch.cat([hidden, repeated_domain], dim=-1)
        else:
            risk_inputs = hidden
        risk_logits = self.risk_head(risk_inputs).squeeze(-1)
        risk_probabilities = torch.sigmoid(risk_logits)

        mask_float = candidate_mask.float()
        masked_hidden = hidden * mask_float.unsqueeze(-1)
        candidate_count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = masked_hidden.sum(dim=1) / candidate_count
        max_hidden = hidden.masked_fill(~candidate_mask.unsqueeze(-1), -1e9)
        max_pool = max_hidden.max(dim=1).values
        no_candidate_mask = candidate_mask.sum(dim=1) == 0
        max_pool = torch.where(no_candidate_mask.unsqueeze(-1), torch.zeros_like(max_pool), max_pool)

        abstain_inputs = torch.cat([mean_pool, max_pool, interaction_features], dim=-1)
        abstain_logits = self.abstain_head(abstain_inputs).squeeze(-1)

        candidate_scores = utility_logits - float(risk_penalty) * risk_probabilities
        candidate_scores = candidate_scores.masked_fill(~candidate_mask, -1e9)
        action_logits = torch.cat([candidate_scores, abstain_logits.unsqueeze(1)], dim=1)
        action_probabilities = torch.softmax(action_logits, dim=1)

        return {
            "utility_logits": utility_logits,
            "risk_logits": risk_logits,
            "risk_probabilities": risk_probabilities,
            "candidate_scores": candidate_scores,
            "abstain_logits": abstain_logits,
            "action_logits": action_logits,
            "action_probabilities": action_probabilities,
        }


def sigmoid_scores(scores: np.ndarray | list[float], *, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    scaled = np.asarray(scores, dtype=float) / float(temperature)
    scaled = np.clip(scaled, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def logit_probabilities(probabilities: np.ndarray | list[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    if len(probabilities) == 0:
        return 0.0
    return float(np.mean((probabilities - labels) ** 2))


def _expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, *, num_bins: int) -> float:
    if len(probabilities) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    total = len(probabilities)
    error = 0.0
    for start, end in zip(bins[:-1], bins[1:], strict=False):
        if end >= 1.0:
            mask = (probabilities >= start) & (probabilities <= end)
        else:
            mask = (probabilities >= start) & (probabilities < end)
        if not np.any(mask):
            continue
        bin_prob = probabilities[mask]
        bin_labels = labels[mask]
        error += abs(float(np.mean(bin_prob)) - float(np.mean(bin_labels))) * (len(bin_prob) / total)
    return float(error)


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
        probabilities = sigmoid_scores(scores_array, temperature=current)
        current_brier = _brier_score(probabilities, labels_array)
        current_ece = _expected_calibration_error(probabilities, labels_array, num_bins=num_bins)
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
    probability_array = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    label_array = np.asarray(labels, dtype=float)
    sample_count = len(probability_array)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "accuracy": 0.0,
            "positive_rate": 0.0,
            "mean_confidence": 0.0,
            "ece": 0.0,
            "brier_score": 0.0,
            "threshold_gap": 0.0,
        }

    predicted_positive = (probability_array >= 0.5).astype(np.float32)
    accuracy = float(np.mean(predicted_positive == label_array))
    threshold_slice = probability_array[:threshold_gap_min_count] if sample_count >= threshold_gap_min_count else probability_array
    label_slice = label_array[:threshold_gap_min_count] if sample_count >= threshold_gap_min_count else label_array
    return {
        "sample_count": int(sample_count),
        "accuracy": accuracy,
        "positive_rate": float(np.mean(label_array)),
        "mean_confidence": float(np.mean(probability_array)),
        "ece": _expected_calibration_error(probability_array, label_array, num_bins=num_bins),
        "brier_score": _brier_score(probability_array, label_array),
        "threshold_gap": float(abs(np.mean(threshold_slice) - np.mean(label_slice))),
    }


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


def keep_probability_from_logits(abstain_logit: float, support_logit: float) -> float:
    abstain_probability = 1.0 / (1.0 + np.exp(-float(abstain_logit)))
    support_probability = 1.0 / (1.0 + np.exp(-float(support_logit)))
    keep_probability = (1.0 - abstain_probability) * support_probability
    return float(min(1.0 - 1e-6, max(1e-6, keep_probability)))


def save_stage9_proposal_model(
    model: FrozenStage9QAModel,
    tokenizer,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    (output_dir / STAGE9_PROPOSAL_CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "model_name": model.model_name,
                "keep_loss_weight": model.keep_loss_weight,
                "support_loss_weight": model.support_loss_weight,
                "keep_positive_weight": model.keep_positive_weight,
                "keep_negative_weight": model.keep_negative_weight,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output_dir / STAGE9_PROPOSAL_STATE_FILENAME)


def load_stage9_proposal_model(model_path: Path) -> FrozenStage9QAModel:
    config = json.loads((model_path / STAGE9_PROPOSAL_CONFIG_FILENAME).read_text(encoding="utf-8"))
    model = FrozenStage9QAModel(
        config["model_name"],
        keep_loss_weight=config["keep_loss_weight"],
        support_loss_weight=config["support_loss_weight"],
        keep_positive_weight=config["keep_positive_weight"],
        keep_negative_weight=config["keep_negative_weight"],
    )
    state_dict = torch.load(model_path / STAGE9_PROPOSAL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def train_stage9_proposal_model(
    *,
    base_model_name: str,
    train_examples: Dataset,
    validation_examples: Dataset,
    output_dir: Path,
    max_length: int,
    doc_stride: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_train_epochs: int,
    seed: int,
) -> Path:
    from transformers import Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = FrozenStage9QAModel(
        base_model_name,
        keep_loss_weight=1.0,
        support_loss_weight=1.0,
        keep_positive_weight=1.0,
        keep_negative_weight=1.0,
    )
    train_features = train_examples.map(
        lambda batch: prepare_train_features(batch, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=train_examples.column_names,
        desc="Tokenizing Stage 9 proposal train data",
    )
    validation_features = validation_examples.map(
        lambda batch: prepare_train_features(batch, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=validation_examples.column_names,
        desc="Tokenizing Stage 9 proposal validation data",
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "trainer-output"),
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            save_strategy="no",
            logging_strategy="epoch",
            report_to=[],
            seed=seed,
            **training_arguments_eval_strategy_kwargs(TrainingArguments, "epoch"),
        ),
        train_dataset=train_features,
        eval_dataset=validation_features,
        **trainer_processing_kwargs(Trainer, tokenizer),
    )
    trainer.train()
    save_stage9_proposal_model(model.cpu(), tokenizer, output_dir)
    return output_dir


def prepare_proposal_eval_artifacts(raw_dataset, tokenizer, max_length: int, doc_stride: int):
    eval_features = raw_dataset.map(
        lambda batch: prepare_eval_features(batch, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing Stage 9 proposal evaluation data",
    )
    eval_dataset_for_model = eval_features.remove_columns(["example_id", "offset_mapping", "cls_index"])
    return eval_features, eval_dataset_for_model


def predict_proposal_raw_outputs(trainer, eval_dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output = trainer.predict(eval_dataset)
    predictions = output.predictions
    if isinstance(predictions, tuple) and len(predictions) >= 4:
        return tuple(np.asarray(item) for item in predictions[:4])  # type: ignore[return-value]
    raise ValueError("Expected tuple predictions with start/end/abstain/support logits.")


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


def _candidate_bundle_feature_vector(record: CandidateRecord) -> list[float]:
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


def _hard_negative_borderline_weight(
    *,
    normalized_rank: float,
    keep_probability: float,
    support_probability: float,
    hard_negative_weight: float,
) -> float:
    """Explicit Stage 9 mining pressure for hard negatives and borderline unsafe answers."""
    rank_pressure = max(0.0, 1.0 - float(normalized_rank))
    risky_answer_pressure = max(0.0, min(1.0, float(keep_probability)))
    borderline_pressure = max(
        _sigmoid_uncertainty(float(keep_probability)),
        _sigmoid_uncertainty(float(support_probability)),
    )
    tail_risk_pressure = 0.40 * rank_pressure + 0.30 * risky_answer_pressure + 0.30 * borderline_pressure
    return 1.0 + float(hard_negative_weight) * tail_risk_pressure


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
            abstain_probability = 1.0 / (1.0 + np.exp(-abstain_logit))
            support_probability = 1.0 / (1.0 + np.exp(-support_logit))
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
            sample_weight = 1.0
            if label < 0.5:
                sample_weight = _hard_negative_borderline_weight(
                    normalized_rank=normalized_rank,
                    keep_probability=float(candidate["keep_probability"]),
                    support_probability=float(candidate["support_probability"]),
                    hard_negative_weight=hard_negative_weight,
                )

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
                    hard_negative_weight=sample_weight,
                )
            )

    if records:
        features_matrix = np.asarray([_candidate_bundle_feature_vector(record) for record in records], dtype=np.float32)
        labels = np.asarray([record.label for record in records], dtype=np.float32)
        sample_weights = np.asarray([record.hard_negative_weight for record in records], dtype=np.float32)
    else:
        features_matrix = np.zeros((0, 8), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.float32)
        sample_weights = np.zeros((0,), dtype=np.float32)

    return CandidateBundle(
        feature_names=[
            "span_score",
            "score_gap_to_best",
            "score_margin_to_next",
            "keep_probability",
            "support_probability",
            "abstain_probability",
            "answer_length_tokens",
            "normalized_rank",
        ],
        all_example_ids=all_example_ids,
        records=records,
        features=features_matrix,
        labels=labels,
        sample_weights=sample_weights,
    )


def build_support_training_dataset(examples, bundle: CandidateBundle) -> Dataset:
    examples_by_id = {str(example["id"]): example for example in examples}
    rows = []
    for record in bundle.records:
        example = examples_by_id[str(record.example_id)]
        rows.append(
            {
                "question": str(example["question"]),
                "context": str(example["context"]),
                "candidate_answer": str(record.answer_text),
                "labels": int(record.label > 0.5),
            }
        )
    return Dataset.from_list(rows)


def fit_support_threshold(
    probabilities: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> float:
    labels_array = np.asarray(labels, dtype=np.float32)
    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    if len(probabilities_array) == 0 or len(np.unique(labels_array)) < 2:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0
    best_accuracy = -1.0
    current = float(threshold_min)
    while current <= threshold_max + 1e-9:
        predictions = (probabilities_array >= current).astype(np.float32)
        tp = float(np.sum((predictions == 1.0) & (labels_array == 1.0)))
        fp = float(np.sum((predictions == 1.0) & (labels_array == 0.0)))
        fn = float(np.sum((predictions == 0.0) & (labels_array == 1.0)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        accuracy = float(np.mean(predictions == labels_array))
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and accuracy > best_accuracy):
            best_threshold = current
            best_f1 = f1
            best_accuracy = accuracy
        current += threshold_step
    return float(round(best_threshold, 10))


def train_stage9_support_scorer(
    *,
    base_model_name: str,
    train_examples,
    validation_examples,
    train_bundle: CandidateBundle,
    validation_bundle: CandidateBundle,
    output_dir: Path,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_train_epochs: int,
    seed: int,
    calibration_bins: int,
    threshold_gap_min_count: int,
    support_threshold_min: float,
    support_threshold_max: float,
    support_threshold_step: float,
) -> tuple[Path, float, float]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    train_dataset = build_support_training_dataset(train_examples, train_bundle)
    validation_dataset = build_support_training_dataset(validation_examples, validation_bundle)
    train_features = train_dataset.map(
        lambda batch: prepare_support_training_features(batch, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing Stage 9 support train data",
    )
    validation_features = validation_dataset.map(
        lambda batch: prepare_support_training_features(batch, tokenizer, max_length),
        batched=True,
        remove_columns=validation_dataset.column_names,
        desc="Tokenizing Stage 9 support validation data",
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "trainer-output"),
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            save_strategy="no",
            logging_strategy="epoch",
            report_to=[],
            seed=seed,
            **training_arguments_eval_strategy_kwargs(TrainingArguments, "epoch"),
        ),
        train_dataset=train_features,
        eval_dataset=validation_features,
        **trainer_processing_kwargs(Trainer, tokenizer),
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    validation_raw_support, _ = score_bundle_with_support_scorer(
        trainer,
        tokenizer,
        validation_examples,
        validation_bundle,
        max_length=max_length,
        support_temperature=1.0,
    )
    support_labels = np.asarray(validation_bundle.labels, dtype=np.float32)
    support_temperature = fit_temperature_scaler(
        logit_probabilities(validation_raw_support),
        support_labels,
        temperature_min=STAGE9_DEFAULT_TEMPERATURE_MIN,
        temperature_max=STAGE9_DEFAULT_TEMPERATURE_MAX,
        temperature_step=STAGE9_DEFAULT_TEMPERATURE_STEP,
        num_bins=calibration_bins,
    )
    calibrated_support = np.asarray(
        sigmoid_scores(logit_probabilities(validation_raw_support), temperature=support_temperature),
        dtype=np.float32,
    )
    _ = summarize_binary_calibration(
        calibrated_support,
        support_labels,
        num_bins=calibration_bins,
        threshold_gap_min_count=threshold_gap_min_count,
    )
    hard_support_threshold = fit_support_threshold(
        calibrated_support,
        support_labels,
        threshold_min=support_threshold_min,
        threshold_max=support_threshold_max,
        threshold_step=support_threshold_step,
    )
    return output_dir, float(support_temperature), float(hard_support_threshold)


def score_bundle_with_support_scorer(
    trainer,
    tokenizer,
    examples,
    bundle: CandidateBundle,
    *,
    max_length: int,
    support_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    from datasets import Dataset

    if not bundle.records:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    examples_by_id = {str(example["id"]): example for example in examples}
    support_rows = []
    for index, record in enumerate(bundle.records):
        example = examples_by_id[record.example_id]
        support_rows.append(
            {
                "candidate_index": index,
                "question": str(example["question"]),
                "context": str(example["context"]),
                "candidate_answer": str(record.answer_text),
            }
        )
    support_dataset = Dataset.from_list(support_rows)
    support_features = support_dataset.map(
        lambda batch: prepare_support_scoring_features(batch, tokenizer, max_length),
        batched=True,
        remove_columns=support_dataset.column_names,
        desc="Tokenizing Stage 9 support-scoring data",
    )
    output = trainer.predict(support_features)
    probabilities = np.asarray(output.predictions, dtype=np.float32)
    if probabilities.ndim == 2:
        probabilities = probabilities - probabilities.max(axis=1, keepdims=True)
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities), axis=1, keepdims=True)
        raw_support = np.asarray(probabilities[:, 1], dtype=np.float32)
    else:
        raise ValueError("Expected support-scorer predictions shaped like [N, 2].")
    calibrated_support = np.asarray(
        sigmoid_scores(logit_probabilities(raw_support), temperature=support_temperature),
        dtype=np.float32,
    )
    return raw_support, calibrated_support


def _prepare_proposal_prediction_artifacts(
    *,
    model_path: Path,
    validation_size: float,
    seed: int,
    max_length: int,
    doc_stride: int,
    eval_batch_size: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    clean_splitting: bool = False,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    if clean_splitting:
        splits = load_stage1_clean_splits(
            validation_size=validation_size,
            seed=seed,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            max_test_samples=max_test_samples,
        )
    else:
        splits = load_stage1_splits(
            validation_size=validation_size,
            seed=seed,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = load_stage9_proposal_model(model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_path / "tmp-stage9-proposal-eval"),
            per_device_eval_batch_size=eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, tokenizer),
    )

    artifacts: dict[str, Any] = {"splits": splits}
    for split_name in splits.keys():
        split_examples = splits[split_name]
        eval_features, eval_model_dataset = prepare_proposal_eval_artifacts(
            split_examples,
            tokenizer,
            max_length,
            doc_stride,
        )
        raw_predictions = predict_proposal_raw_outputs(trainer, eval_model_dataset)
        artifacts[split_name] = {
            "examples": split_examples,
            "features": eval_features,
            "raw_predictions": raw_predictions,
            "references": build_reference_index(split_examples),
        }
    return artifacts


def _tokenize_to_set(text: str) -> set[str]:
    normalized = normalize_answer(text)
    if not normalized:
        return set()
    return set(normalized.split())


def _tokenize_to_list(text: str) -> list[str]:
    normalized = normalize_answer(text)
    if not normalized:
        return []
    return normalized.split()


def _question_overlap(question: str, answer_text: str) -> float:
    answer_tokens = _tokenize_to_set(answer_text)
    if not answer_tokens:
        return 0.0
    question_tokens = _tokenize_to_set(question)
    return float(len(answer_tokens & question_tokens)) / float(len(answer_tokens))


def _sigmoid_uncertainty(probability: float) -> float:
    return float(max(0.0, 1.0 - abs(float(probability) - 0.5) * 2.0))


def _normalized_entropy(values: list[float]) -> float:
    if not values:
        return 0.0
    clipped = np.clip(np.asarray(values, dtype=np.float32), 1e-6, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return 0.0
    probabilities = clipped / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    max_entropy = float(np.log(len(probabilities))) if len(probabilities) > 1 else 1.0
    if max_entropy <= 0.0:
        return 0.0
    return float(entropy / max_entropy)


def _question_type_one_hot(question: str) -> list[float]:
    normalized = normalize_answer(question)
    tokens = normalized.split()
    prefix = tokens[0] if tokens else "other"
    values = [0.0] * len(QUESTION_TYPE_PREFIXES)
    try:
        index = QUESTION_TYPE_PREFIXES.index(prefix)
    except ValueError:
        index = QUESTION_TYPE_PREFIXES.index("other")
    values[index] = 1.0
    return values


def _domain_feature_vector(example: dict[str, Any]) -> list[float]:
    question = str(example.get("question", ""))
    context = str(example.get("context", ""))
    question_tokens = _tokenize_to_list(question)
    context_tokens = _tokenize_to_list(context)
    return [
        *_question_type_one_hot(question),
        float(min(len(question_tokens), 40) / 40.0),
        float(min(len(context_tokens), 256) / 256.0),
        float(any(char.isdigit() for char in question)),
        float(any(char.isdigit() for char in context)),
    ]


def _interaction_feature_vector(candidates: tuple[Stage9Candidate, ...]) -> list[float]:
    if not candidates:
        return [0.0] * len(STAGE9_INTERACTION_FEATURE_NAMES)

    keep_values = [float(candidate.keep_probability) for candidate in candidates]
    support_values = [float(candidate.calibrated_support_probability) for candidate in candidates]
    disagreement = [
        abs(float(candidate.keep_probability) - float(candidate.calibrated_support_probability))
        for candidate in candidates
    ]
    sorted_keep = sorted(keep_values, reverse=True)
    sorted_support = sorted(support_values, reverse=True)
    top_keep_gap = float(sorted_keep[0] - sorted_keep[1]) if len(sorted_keep) > 1 else float(sorted_keep[0])
    top_support_gap = (
        float(sorted_support[0] - sorted_support[1]) if len(sorted_support) > 1 else float(sorted_support[0])
    )
    return [
        float(len(candidates)),
        top_keep_gap,
        top_support_gap,
        _normalized_entropy(keep_values),
        _normalized_entropy(support_values),
        float(max(keep_values) - min(keep_values)),
        float(max(support_values) - min(support_values)),
        float(np.mean(disagreement)),
        float(np.max(disagreement)),
        float(np.mean([candidate.support_gate_pass for candidate in candidates])),
    ]


def _candidate_feature_vector(
    record: CandidateRecord,
    *,
    question: str,
    raw_support_probability: float,
    calibrated_support_probability: float,
    best_keep_probability: float,
    best_support_probability: float,
    support_gate_pass: float,
) -> list[float]:
    return [
        float(record.span_score),
        float(record.score_gap_to_best),
        float(record.score_margin_to_next),
        float(record.keep_probability),
        float(record.support_probability),
        float(record.abstain_probability),
        float(raw_support_probability),
        float(calibrated_support_probability),
        float(calibrated_support_probability - record.support_probability),
        float(support_gate_pass),
        float(record.answer_length_tokens),
        float(record.normalized_rank),
        float(_question_overlap(question, record.answer_text)),
        _sigmoid_uncertainty(record.keep_probability),
        _sigmoid_uncertainty(calibrated_support_probability),
        float(abs(record.keep_probability - calibrated_support_probability)),
        float(best_keep_probability - record.keep_probability),
        float(best_support_probability - calibrated_support_probability),
    ]


def _fit_or_default_standardization(matrix: np.ndarray, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.zeros((feature_count,), dtype=np.float32), np.ones((feature_count,), dtype=np.float32)
    return fit_feature_standardization(matrix)


def build_stage9_sets(
    examples,
    bundle: CandidateBundle,
    *,
    raw_support_probabilities: np.ndarray,
    calibrated_support_probabilities: np.ndarray,
    hard_support_threshold: float,
    use_domain_features: bool,
) -> list[Stage9Set]:
    if len(raw_support_probabilities) != len(bundle.records):
        raise ValueError("Raw support probabilities must align with candidate records.")
    if len(calibrated_support_probabilities) != len(bundle.records):
        raise ValueError("Calibrated support probabilities must align with candidate records.")

    records_by_example: dict[str, list[tuple[CandidateRecord, float, float]]] = defaultdict(list)
    for record, raw_support, calibrated_support in zip(
        bundle.records,
        raw_support_probabilities,
        calibrated_support_probabilities,
        strict=False,
    ):
        records_by_example[str(record.example_id)].append(
            (record, float(raw_support), float(calibrated_support))
        )

    action_sets: list[Stage9Set] = []
    for example in examples:
        example_id = str(example["id"])
        question = str(example["question"])
        answerable = bool(example["answers"]["text"])
        raw_candidates = records_by_example.get(example_id, [])

        best_keep_probability = max((float(record.keep_probability) for record, _, _ in raw_candidates), default=0.0)
        best_support_probability = max((float(calibrated_support) for _, _, calibrated_support in raw_candidates), default=0.0)

        candidates = tuple(
            Stage9Candidate(
                example_id=example_id,
                answer_text=str(record.answer_text),
                span_score=float(record.span_score),
                score_gap_to_best=float(record.score_gap_to_best),
                score_margin_to_next=float(record.score_margin_to_next),
                keep_probability=float(record.keep_probability),
                proposal_support_probability=float(record.support_probability),
                abstain_probability=float(record.abstain_probability),
                raw_support_scorer_probability=float(raw_support),
                calibrated_support_probability=float(calibrated_support),
                support_gate_pass=float(calibrated_support >= hard_support_threshold),
                answer_length_tokens=float(record.answer_length_tokens),
                normalized_rank=float(record.normalized_rank),
                question_overlap=float(_question_overlap(question, record.answer_text)),
                label=float(record.label),
                hard_negative_weight=float(record.hard_negative_weight),
                candidate_features=tuple(
                    _candidate_feature_vector(
                        record,
                        question=question,
                        raw_support_probability=float(raw_support),
                        calibrated_support_probability=float(calibrated_support),
                        best_keep_probability=best_keep_probability,
                        best_support_probability=best_support_probability,
                        support_gate_pass=float(calibrated_support >= hard_support_threshold),
                    )
                ),
            )
            for record, raw_support, calibrated_support in raw_candidates
        )

        supported_indexes = [index for index, candidate in enumerate(candidates) if candidate.label > 0.5]
        if supported_indexes:
            target_action_index = max(
                supported_indexes,
                key=lambda index: (
                    candidates[index].support_gate_pass,
                    candidates[index].calibrated_support_probability,
                    candidates[index].keep_probability,
                    candidates[index].span_score,
                    -candidates[index].normalized_rank,
                ),
            )
        else:
            target_action_index = len(candidates)

        action_sets.append(
            Stage9Set(
                example_id=example_id,
                question=question,
                answerable=answerable,
                target_action_index=target_action_index,
                candidates=candidates,
                interaction_features=tuple(_interaction_feature_vector(candidates)),
                domain_features=tuple(_domain_feature_vector(example) if use_domain_features else []),
            )
        )

    return action_sets


def flatten_candidate_features(action_sets: list[Stage9Set]) -> np.ndarray:
    rows: list[list[float]] = []
    for action_set in action_sets:
        for candidate in action_set.candidates:
            rows.append(list(candidate.candidate_features))
    if not rows:
        return np.zeros((0, len(STAGE9_CANDIDATE_FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def flatten_interaction_features(action_sets: list[Stage9Set]) -> np.ndarray:
    if not action_sets:
        return np.zeros((0, len(STAGE9_INTERACTION_FEATURE_NAMES)), dtype=np.float32)
    return np.asarray([list(action_set.interaction_features) for action_set in action_sets], dtype=np.float32)


def flatten_domain_features(action_sets: list[Stage9Set]) -> np.ndarray:
    if not action_sets:
        return np.zeros((0, len(STAGE9_DOMAIN_FEATURE_NAMES)), dtype=np.float32)
    if not action_sets[0].domain_features:
        return np.zeros((len(action_sets), 0), dtype=np.float32)
    return np.asarray([list(action_set.domain_features) for action_set in action_sets], dtype=np.float32)


def standardize_stage9_sets(
    action_sets: list[Stage9Set],
    *,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
    interaction_mean: np.ndarray,
    interaction_std: np.ndarray,
    domain_mean: np.ndarray,
    domain_std: np.ndarray,
) -> list[Stage9Set]:
    candidate_matrix = flatten_candidate_features(action_sets)
    interaction_matrix = flatten_interaction_features(action_sets)
    domain_matrix = flatten_domain_features(action_sets)

    standardized_candidates = standardize_features(candidate_matrix, candidate_mean, candidate_std)
    standardized_interactions = (
        standardize_features(interaction_matrix, interaction_mean, interaction_std)
        if interaction_matrix.size
        else interaction_matrix.astype(np.float32)
    )
    standardized_domains = (
        standardize_features(domain_matrix, domain_mean, domain_std)
        if domain_matrix.size
        else domain_matrix.astype(np.float32)
    )

    candidate_cursor = 0
    output: list[Stage9Set] = []
    for set_index, action_set in enumerate(action_sets):
        standardized_candidates_for_set: list[Stage9Candidate] = []
        for candidate in action_set.candidates:
            standardized_candidates_for_set.append(
                replace(
                    candidate,
                    candidate_features=tuple(standardized_candidates[candidate_cursor].tolist()),
                )
            )
            candidate_cursor += 1

        interaction_features = (
            tuple(standardized_interactions[set_index].tolist())
            if standardized_interactions.size
            else tuple(action_set.interaction_features)
        )
        if standardized_domains.size:
            domain_features = tuple(standardized_domains[set_index].tolist())
        elif domain_matrix.shape[1] == 0:
            domain_features = tuple()
        else:
            domain_features = tuple(action_set.domain_features)

        output.append(
            replace(
                action_set,
                candidates=tuple(standardized_candidates_for_set),
                interaction_features=interaction_features,
                domain_features=domain_features,
            )
        )
    return output


def collate_stage9_sets(action_sets: list[Stage9Set]) -> dict[str, Any]:
    batch_size = len(action_sets)
    max_candidates = max((len(action_set.candidates) for action_set in action_sets), default=0)
    max_candidates = max(1, max_candidates)
    candidate_feature_dim = len(STAGE9_CANDIDATE_FEATURE_NAMES)
    interaction_dim = len(STAGE9_INTERACTION_FEATURE_NAMES)
    domain_dim = len(action_sets[0].domain_features) if action_sets and action_sets[0].domain_features else 0

    candidate_features = torch.zeros((batch_size, max_candidates, candidate_feature_dim), dtype=torch.float32)
    candidate_labels = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    candidate_risk_labels = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    candidate_weights = torch.ones((batch_size, max_candidates), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    target_indices = torch.zeros((batch_size,), dtype=torch.long)
    abstain_targets = torch.zeros((batch_size,), dtype=torch.float32)
    interaction_features_tensor = torch.zeros((batch_size, interaction_dim), dtype=torch.float32)
    domain_features_tensor = torch.zeros((batch_size, domain_dim), dtype=torch.float32)

    for batch_index, action_set in enumerate(action_sets):
        if int(action_set.target_action_index) < len(action_set.candidates):
            target_indices[batch_index] = int(action_set.target_action_index)
        else:
            target_indices[batch_index] = max_candidates
            abstain_targets[batch_index] = 1.0

        if action_set.interaction_features:
            interaction_features_tensor[batch_index] = torch.tensor(
                action_set.interaction_features,
                dtype=torch.float32,
            )
        if domain_dim > 0 and action_set.domain_features:
            domain_features_tensor[batch_index] = torch.tensor(action_set.domain_features, dtype=torch.float32)

        for candidate_index, candidate in enumerate(action_set.candidates):
            candidate_features[batch_index, candidate_index] = torch.tensor(candidate.candidate_features, dtype=torch.float32)
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
        "abstain_targets": abstain_targets,
        "interaction_features": interaction_features_tensor,
        "domain_features": domain_features_tensor,
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


def _apply_randomization_support(
    risk_weights: torch.Tensor,
    risk_labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    randomization_scale: float,
) -> torch.Tensor:
    """Training-only stochastic weighting for Mermaid Randomization Support -> Tail-Risk Training."""
    if randomization_scale <= 0.0 or not bool(candidate_mask.any()):
        return risk_weights
    unsafe_mask = (risk_labels > 0.5) & candidate_mask
    if not bool(unsafe_mask.any()):
        return risk_weights
    jitter = 1.0 + float(randomization_scale) * (2.0 * torch.rand_like(risk_weights) - 1.0)
    jitter = torch.where(unsafe_mask, jitter, torch.ones_like(jitter))
    return (risk_weights * jitter.clamp_min(0.10)).clamp_min(0.0)


def compute_stage9_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    joint_loss_weight: float,
    utility_loss_weight: float,
    risk_loss_weight: float,
    abstain_loss_weight: float,
    tail_risk_weight: float,
    randomization_scale: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    joint_loss = F.cross_entropy(outputs["action_logits"], batch["target_indices"])

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
    risk_weights = _apply_randomization_support(
        risk_weights,
        batch["candidate_risk_labels"],
        batch["candidate_mask"],
        randomization_scale=randomization_scale,
    )
    risk_loss = _masked_weighted_bce(
        outputs["risk_logits"],
        batch["candidate_risk_labels"],
        risk_weights,
        batch["candidate_mask"],
    )

    abstain_loss = F.binary_cross_entropy_with_logits(
        outputs["abstain_logits"],
        batch["abstain_targets"],
    )

    total_loss = (
        float(joint_loss_weight) * joint_loss
        + float(utility_loss_weight) * utility_loss
        + float(risk_loss_weight) * risk_loss
        + float(abstain_loss_weight) * abstain_loss
    )
    return total_loss, {
        "joint_loss": float(joint_loss.detach().cpu()),
        "utility_loss": float(utility_loss.detach().cpu()),
        "risk_loss": float(risk_loss.detach().cpu()),
        "abstain_loss": float(abstain_loss.detach().cpu()),
    }


def _prepare_support_runtime(
    *,
    support_model_path: Path,
    eval_batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    support_tokenizer = AutoTokenizer.from_pretrained(support_model_path, use_fast=True)
    support_model = AutoModelForSequenceClassification.from_pretrained(support_model_path)
    support_trainer = Trainer(
        model=support_model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_eval_batch_size=eval_batch_size,
            report_to=[],
        ),
        **trainer_processing_kwargs(Trainer, support_tokenizer),
    )
    return {
        "support_model_path": support_model_path,
        "support_tokenizer": support_tokenizer,
        "support_trainer": support_trainer,
    }


def _predict_stage9_outputs(
    model: ExactStage9Model,
    action_sets: list[Stage9Set],
    *,
    batch_size: int,
    device: torch.device,
    risk_penalty: float,
) -> list[dict[str, Any]]:
    if not action_sets:
        return []

    loader = DataLoader(action_sets, batch_size=batch_size, shuffle=False, collate_fn=collate_stage9_sets)
    outputs: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_outputs = model(
                batch["candidate_features"].to(device),
                batch["candidate_mask"].to(device),
                batch["interaction_features"].to(device),
                batch["domain_features"].to(device),
                risk_penalty=risk_penalty,
            )
            utility_logits = batch_outputs["utility_logits"].detach().cpu().numpy()
            risk_logits = batch_outputs["risk_logits"].detach().cpu().numpy()
            raw_risk_probabilities = batch_outputs["risk_probabilities"].detach().cpu().numpy()
            candidate_scores = batch_outputs["candidate_scores"].detach().cpu().numpy()
            abstain_logits = batch_outputs["abstain_logits"].detach().cpu().numpy()
            action_probabilities = batch_outputs["action_probabilities"].detach().cpu().numpy()

            for index, action_set in enumerate(batch["action_sets"]):
                candidate_count = len(action_set.candidates)
                outputs.append(
                    {
                        "candidate_utility_logits": utility_logits[index, :candidate_count].astype(float).tolist(),
                        "candidate_risk_logits": risk_logits[index, :candidate_count].astype(float).tolist(),
                        "candidate_raw_risk_probabilities": raw_risk_probabilities[index, :candidate_count].astype(float).tolist(),
                        "candidate_raw_scores": candidate_scores[index, :candidate_count].astype(float).tolist(),
                        "abstain_logit": float(abstain_logits[index]),
                        "raw_action_probabilities": action_probabilities[index, : candidate_count + 1].astype(float).tolist(),
                    }
                )
    return outputs


def _evaluate_stage9_model(
    model: ExactStage9Model,
    action_sets: list[Stage9Set],
    *,
    batch_size: int,
    device: torch.device,
    risk_penalty: float,
    joint_loss_weight: float,
    utility_loss_weight: float,
    risk_loss_weight: float,
    abstain_loss_weight: float,
    tail_risk_weight: float,
    randomization_scale: float = 0.0,
) -> dict[str, float]:
    if not action_sets:
        return {
            "total_loss": 0.0,
            "joint_loss": 0.0,
            "utility_loss": 0.0,
            "risk_loss": 0.0,
            "abstain_loss": 0.0,
        }

    loader = DataLoader(action_sets, batch_size=batch_size, shuffle=False, collate_fn=collate_stage9_sets)
    model.eval()
    total = {
        "total_loss": 0.0,
        "joint_loss": 0.0,
        "utility_loss": 0.0,
        "risk_loss": 0.0,
        "abstain_loss": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["candidate_features"].to(device),
                batch["candidate_mask"].to(device),
                batch["interaction_features"].to(device),
                batch["domain_features"].to(device),
                risk_penalty=risk_penalty,
            )
            loss, components = compute_stage9_losses(
                outputs,
                {
                    "candidate_labels": batch["candidate_labels"].to(device),
                    "candidate_risk_labels": batch["candidate_risk_labels"].to(device),
                    "candidate_weights": batch["candidate_weights"].to(device),
                    "candidate_mask": batch["candidate_mask"].to(device),
                    "target_indices": batch["target_indices"].to(device),
                    "abstain_targets": batch["abstain_targets"].to(device),
                },
                joint_loss_weight=joint_loss_weight,
                utility_loss_weight=utility_loss_weight,
                risk_loss_weight=risk_loss_weight,
                abstain_loss_weight=abstain_loss_weight,
                tail_risk_weight=tail_risk_weight,
                randomization_scale=randomization_scale,
            )
            total["total_loss"] += float(loss.detach().cpu())
            for key, value in components.items():
                total[key] += float(value)
            batches += 1
    return {key: value / max(1, batches) for key, value in total.items()}


def _collect_risk_calibration_arrays(
    action_sets: list[Stage9Set],
    action_outputs: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[float] = []
    labels: list[float] = []
    for action_set, action_output in zip(action_sets, action_outputs, strict=False):
        for candidate, risk_logit in zip(
            action_set.candidates,
            action_output["candidate_risk_logits"],
            strict=False,
        ):
            logits.append(float(risk_logit))
            labels.append(1.0 - float(candidate.label))
    return np.asarray(logits, dtype=np.float32), np.asarray(labels, dtype=np.float32)


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


def _masked_softmax(scores: list[float], abstain_score: float) -> np.ndarray:
    vector = np.asarray([*scores, float(abstain_score)], dtype=np.float64)
    shifted = vector - np.max(vector)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def postprocess_stage9_predictions(
    action_sets: list[Stage9Set],
    action_outputs: list[dict[str, Any]],
    *,
    risk_temperature: float,
    risk_penalty: float,
    risk_threshold: float,
    abstain_margin: float,
    hard_support_threshold: float,
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for action_set, action_output in zip(action_sets, action_outputs, strict=False):
        candidate_count = len(action_set.candidates)
        abstain_logit = float(action_output["abstain_logit"])
        if candidate_count == 0:
            predictions[action_set.example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "selected_action_probability": 1.0,
                    "selected_risk_probability": 1.0,
                    "best_candidate_score": float("-inf"),
                    "abstain_score": abstain_logit,
                },
                "abstain_reason": "no_candidate",
            }
            continue

        risk_logits = np.asarray(action_output["candidate_risk_logits"], dtype=np.float32)
        utility_logits = np.asarray(action_output["candidate_utility_logits"], dtype=np.float32)
        calibrated_risk = sigmoid_scores(risk_logits, temperature=risk_temperature)
        calibrated_scores = utility_logits - float(risk_penalty) * calibrated_risk

        ranked_candidate_indexes = sorted(
            range(candidate_count),
            key=lambda index: (
                float(calibrated_scores[index]),
                float(utility_logits[index]),
                action_set.candidates[index].span_score,
            ),
            reverse=True,
        )

        safe_candidates = [
            index
            for index in ranked_candidate_indexes
            if float(action_set.candidates[index].calibrated_support_probability) >= float(hard_support_threshold)
            and float(calibrated_risk[index]) <= float(risk_threshold)
        ]
        top_candidate_index = ranked_candidate_indexes[0]
        top_candidate_score = float(calibrated_scores[top_candidate_index])

        if not safe_candidates:
            top_candidate = action_set.candidates[top_candidate_index]
            support_reason = (
                top_candidate.calibrated_support_probability < float(hard_support_threshold)
                and calibrated_risk[top_candidate_index] <= float(risk_threshold)
            )
            predictions[action_set.example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "selected_action_probability": 1.0,
                    "selected_risk_probability": float(calibrated_risk[top_candidate_index]),
                    "selected_support_probability": float(top_candidate.calibrated_support_probability),
                    "best_candidate_score": top_candidate_score,
                    "abstain_score": abstain_logit,
                },
                "abstain_reason": "support_shield" if support_reason else "risk_shield",
            }
            continue

        selected_candidate_index = safe_candidates[0]
        selected_candidate = action_set.candidates[selected_candidate_index]
        selected_candidate_score = float(calibrated_scores[selected_candidate_index])

        calibrated_action_probabilities = _masked_softmax(calibrated_scores.tolist(), abstain_logit)
        selected_candidate_probability = float(calibrated_action_probabilities[selected_candidate_index])
        abstain_probability = float(calibrated_action_probabilities[-1])

        if abstain_logit + float(abstain_margin) >= selected_candidate_score:
            predictions[action_set.example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "selected_action_probability": abstain_probability,
                    "selected_risk_probability": float(calibrated_risk[selected_candidate_index]),
                    "selected_support_probability": float(selected_candidate.calibrated_support_probability),
                    "best_candidate_score": selected_candidate_score,
                    "abstain_score": abstain_logit,
                    "abstain_margin": float(abstain_margin),
                },
                "abstain_reason": "abstain_margin",
            }
            continue

        predictions[action_set.example_id] = {
            "decision": "answer",
            "answer": selected_candidate.answer_text,
            "scores": {
                "selected_action_probability": selected_candidate_probability,
                "selected_risk_probability": float(calibrated_risk[selected_candidate_index]),
                "selected_support_probability": float(selected_candidate.calibrated_support_probability),
                "best_candidate_score": selected_candidate_score,
                "abstain_score": abstain_logit,
                "abstain_margin": float(abstain_margin),
                "span_score": float(selected_candidate.span_score),
                "keep_probability": float(selected_candidate.keep_probability),
                "proposal_support_probability": float(selected_candidate.proposal_support_probability),
                "raw_support_scorer_probability": float(selected_candidate.raw_support_scorer_probability),
                "calibrated_support_probability": float(selected_candidate.calibrated_support_probability),
            },
            "support": {"score": float(selected_candidate.calibrated_support_probability)},
            "risk": {"score": float(calibrated_risk[selected_candidate_index])},
            "stage9": {
                "selected_answer": selected_candidate.answer_text,
                "selected_rank": float(selected_candidate.normalized_rank),
                "question_overlap": float(selected_candidate.question_overlap),
            },
        }
    return predictions


def select_stage9_boundary_entry(
    sweep: list[dict[str, float]],
    *,
    max_unsupported_answer_rate: float,
    max_overabstain_rate: float,
) -> dict[str, float]:
    if not sweep:
        raise ValueError("Boundary sweep is empty.")

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


def search_stage9_boundary(
    action_sets: list[Stage9Set],
    action_outputs: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    risk_temperature: float,
    risk_penalty: float,
    risk_threshold_min: float,
    risk_threshold_max: float,
    risk_threshold_step: float,
    abstain_margin_min: float,
    abstain_margin_max: float,
    abstain_margin_step: float,
    match_f1_threshold: float,
    max_unsupported_answer_rate: float,
    max_overabstain_rate: float,
    hard_support_threshold: float,
) -> tuple[float, float, dict[str, float], dict[str, float], dict[str, float], list[dict[str, float]]]:
    sweep: list[dict[str, float]] = []
    risk_threshold = float(risk_threshold_min)
    while risk_threshold <= risk_threshold_max + 1e-9:
        abstain_margin = float(abstain_margin_min)
        while abstain_margin <= abstain_margin_max + 1e-9:
            predictions = postprocess_stage9_predictions(
                action_sets,
                action_outputs,
                risk_temperature=risk_temperature,
                risk_penalty=risk_penalty,
                risk_threshold=risk_threshold,
                abstain_margin=abstain_margin,
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
                    "risk_threshold": round(risk_threshold, 10),
                    "abstain_margin": round(abstain_margin, 10),
                    "constraint_satisfied": (
                        metrics["unsupported_answer_rate"] <= max_unsupported_answer_rate
                        and overabstain["overabstain_rate"] <= max_overabstain_rate
                    ),
                    **metrics,
                    **mix,
                    **overabstain,
                }
            )
            abstain_margin += abstain_margin_step
        risk_threshold += risk_threshold_step

    best_entry = select_stage9_boundary_entry(
        sweep,
        max_unsupported_answer_rate=max_unsupported_answer_rate,
        max_overabstain_rate=max_overabstain_rate,
    )
    best_metrics = {
        key: value
        for key, value in best_entry.items()
        if key
        not in {
            "risk_threshold",
            "abstain_margin",
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
    return (
        float(best_entry["risk_threshold"]),
        float(best_entry["abstain_margin"]),
        best_metrics,
        best_mix,
        best_overabstain,
        sweep,
    )


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


def _save_stage9_model(
    model: ExactStage9Model,
    output_dir: Path,
    config: Stage9Config,
    training_history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / STAGE9_MODEL_STATE_FILENAME)
    (output_dir / STAGE9_MODEL_CONFIG_FILENAME).write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / STAGE9_TRAINING_HISTORY_FILENAME).write_text(
        json.dumps(training_history, indent=2) + "\n",
        encoding="utf-8",
    )


def load_stage9_model(model_path: Path) -> tuple[ExactStage9Model, Stage9Config]:
    config_payload = json.loads((model_path / STAGE9_MODEL_CONFIG_FILENAME).read_text(encoding="utf-8"))
    config = Stage9Config(**config_payload)
    model = ExactStage9Model(
        config.candidate_input_dim,
        config.interaction_input_dim,
        config.domain_input_dim,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    )
    state_dict = torch.load(model_path / STAGE9_MODEL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    return model, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the exact Stage 9 risk-generalization model.")
    train_parser.add_argument("--base-model-name", type=str, default=DEFAULT_STAGE9_BASE_MODEL_NAME)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    train_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    train_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    train_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    train_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--eval-batch-size", type=int, default=32)
    train_parser.add_argument("--train-batch-size", type=int, default=16)
    train_parser.add_argument("--proposal-train-batch-size", type=int, default=8)
    train_parser.add_argument("--support-train-batch-size", type=int, default=16)
    train_parser.add_argument("--proposal-learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--support-learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--proposal-weight-decay", type=float, default=0.01)
    train_parser.add_argument("--support-weight-decay", type=float, default=0.01)
    train_parser.add_argument("--proposal-num-train-epochs", type=int, default=2)
    train_parser.add_argument("--support-num-train-epochs", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=2e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--num-train-epochs", type=int, default=16)
    train_parser.add_argument("--hidden-size", type=int, default=96)
    train_parser.add_argument("--dropout", type=float, default=0.15)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-eval-samples", type=int, default=None)
    train_parser.add_argument("--clean-splitting", action="store_true")
    train_parser.add_argument("--max-test-samples", type=int, default=None)
    train_parser.add_argument("--max-candidates-per-example", type=int, default=6)
    train_parser.add_argument("--max-candidates-per-feature", type=int, default=3)
    train_parser.add_argument("--joint-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--utility-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--risk-loss-weight", type=float, default=1.25)
    train_parser.add_argument("--abstain-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--tail-risk-weight", type=float, default=3.0)
    train_parser.add_argument("--randomization-scale", type=float, default=DEFAULT_STAGE9_RANDOMIZATION_SCALE)
    train_parser.add_argument("--risk-penalty", type=float, default=1.0)
    train_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE9_MAX_UNSUPPORTED_ANSWER_RATE,
    )
    train_parser.add_argument(
        "--max-overabstain-rate",
        type=float,
        default=DEFAULT_STAGE9_MAX_OVERABSTAIN_RATE,
    )
    train_parser.add_argument("--hard-support-threshold", type=float, default=None)
    train_parser.add_argument("--support-threshold-min", type=float, default=DEFAULT_STAGE9_SUPPORT_THRESHOLD_MIN)
    train_parser.add_argument("--support-threshold-max", type=float, default=DEFAULT_STAGE9_SUPPORT_THRESHOLD_MAX)
    train_parser.add_argument("--support-threshold-step", type=float, default=DEFAULT_STAGE9_SUPPORT_THRESHOLD_STEP)
    train_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)
    train_parser.add_argument("--risk-temperature-min", type=float, default=STAGE9_DEFAULT_TEMPERATURE_MIN)
    train_parser.add_argument("--risk-temperature-max", type=float, default=STAGE9_DEFAULT_TEMPERATURE_MAX)
    train_parser.add_argument("--risk-temperature-step", type=float, default=STAGE9_DEFAULT_TEMPERATURE_STEP)
    train_parser.add_argument("--calibration-bins", type=int, default=STAGE9_DEFAULT_CALIBRATION_BINS)
    train_parser.add_argument("--threshold-gap-min-count", type=int, default=STAGE9_DEFAULT_THRESHOLD_GAP_MIN_COUNT)
    train_parser.add_argument(
        "--risk-threshold-min",
        type=float,
        default=DEFAULT_STAGE9_RISK_THRESHOLD_MIN,
    )
    train_parser.add_argument(
        "--risk-threshold-max",
        type=float,
        default=DEFAULT_STAGE9_RISK_THRESHOLD_MAX,
    )
    train_parser.add_argument(
        "--risk-threshold-step",
        type=float,
        default=DEFAULT_STAGE9_RISK_THRESHOLD_STEP,
    )
    train_parser.add_argument("--abstain-margin-min", type=float, default=0.0)
    train_parser.add_argument("--abstain-margin-max", type=float, default=0.50)
    train_parser.add_argument("--abstain-margin-step", type=float, default=0.05)
    train_parser.add_argument(
        "--disable-domain-features",
        action="store_true",
        help="Disable the optional domain/context feature path into the risk head.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the exact Stage 9 risk-generalization model.")
    eval_parser.add_argument("--model-path", type=Path, required=True)
    eval_parser.add_argument("--output-path", type=Path, required=True)
    eval_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    eval_parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    eval_parser.add_argument("--max-answer-length", type=int, default=DEFAULT_MAX_ANSWER_LENGTH)
    eval_parser.add_argument("--n-best-size", type=int, default=DEFAULT_N_BEST_SIZE)
    eval_parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    eval_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    eval_parser.add_argument("--eval-batch-size", type=int, default=32)
    eval_parser.add_argument("--max-eval-samples", type=int, default=None)
    eval_parser.add_argument("--clean-splitting", action="store_true")
    eval_parser.add_argument("--max-test-samples", type=int, default=None)
    eval_parser.add_argument("--hard-support-threshold", type=float, default=None)
    eval_parser.add_argument("--match-f1-threshold", type=float, default=DEFAULT_SUPPORT_MATCH_F1)
    eval_parser.add_argument("--risk-temperature-min", type=float, default=STAGE9_DEFAULT_TEMPERATURE_MIN)
    eval_parser.add_argument("--risk-temperature-max", type=float, default=STAGE9_DEFAULT_TEMPERATURE_MAX)
    eval_parser.add_argument("--risk-temperature-step", type=float, default=STAGE9_DEFAULT_TEMPERATURE_STEP)
    eval_parser.add_argument("--calibration-bins", type=int, default=STAGE9_DEFAULT_CALIBRATION_BINS)
    eval_parser.add_argument("--threshold-gap-min-count", type=int, default=STAGE9_DEFAULT_THRESHOLD_GAP_MIN_COUNT)
    eval_parser.add_argument(
        "--risk-threshold-min",
        type=float,
        default=DEFAULT_STAGE9_RISK_THRESHOLD_MIN,
    )
    eval_parser.add_argument(
        "--risk-threshold-max",
        type=float,
        default=DEFAULT_STAGE9_RISK_THRESHOLD_MAX,
    )
    eval_parser.add_argument(
        "--risk-threshold-step",
        type=float,
        default=DEFAULT_STAGE9_RISK_THRESHOLD_STEP,
    )
    eval_parser.add_argument("--abstain-margin-min", type=float, default=0.0)
    eval_parser.add_argument("--abstain-margin-max", type=float, default=0.50)
    eval_parser.add_argument("--abstain-margin-step", type=float, default=0.05)
    eval_parser.add_argument(
        "--max-unsupported-answer-rate",
        type=float,
        default=DEFAULT_STAGE9_MAX_UNSUPPORTED_ANSWER_RATE,
    )
    eval_parser.add_argument(
        "--max-overabstain-rate",
        type=float,
        default=DEFAULT_STAGE9_MAX_OVERABSTAIN_RATE,
    )
    return parser


def _train_stage9(args: argparse.Namespace) -> None:
    from transformers import set_seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.clean_splitting:
        splits = load_stage1_clean_splits(
            validation_size=args.validation_size,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            max_test_samples=args.max_test_samples,
        )
    else:
        splits = load_stage1_splits(
            validation_size=args.validation_size,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )

    proposal_model_dir = args.output_dir / STAGE9_PROPOSAL_DIRNAME
    support_model_dir = args.output_dir / STAGE9_SUPPORT_DIRNAME
    train_examples = splits["train"]
    validation_examples = splits["validation"]

    train_stage9_proposal_model(
        base_model_name=args.base_model_name,
        train_examples=train_examples,
        validation_examples=validation_examples,
        output_dir=proposal_model_dir,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        train_batch_size=args.proposal_train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.proposal_learning_rate,
        weight_decay=args.proposal_weight_decay,
        num_train_epochs=args.proposal_num_train_epochs,
        seed=args.seed,
    )

    artifacts = _prepare_proposal_prediction_artifacts(
        model_path=proposal_model_dir,
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

    (
        _support_model_dir,
        support_temperature,
        learned_hard_support_threshold,
    ) = train_stage9_support_scorer(
        base_model_name=args.base_model_name,
        train_examples=artifacts["train"]["examples"],
        validation_examples=artifacts["validation"]["examples"],
        train_bundle=train_bundle,
        validation_bundle=validation_bundle,
        output_dir=support_model_dir,
        max_length=args.max_length,
        train_batch_size=args.support_train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.support_learning_rate,
        weight_decay=args.support_weight_decay,
        num_train_epochs=args.support_num_train_epochs,
        seed=args.seed,
        calibration_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
        support_threshold_min=args.support_threshold_min,
        support_threshold_max=args.support_threshold_max,
        support_threshold_step=args.support_threshold_step,
    )

    resolved_hard_support_threshold = (
        float(args.hard_support_threshold)
        if args.hard_support_threshold is not None
        else float(learned_hard_support_threshold)
    )
    support_runtime = _prepare_support_runtime(
        support_model_path=support_model_dir,
        eval_batch_size=args.eval_batch_size,
        output_dir=args.output_dir / "tmp-stage9-support-train",
    )

    train_raw_support, train_calibrated_support = score_bundle_with_support_scorer(
        support_runtime["support_trainer"],
        support_runtime["support_tokenizer"],
        artifacts["train"]["examples"],
        train_bundle,
        max_length=args.max_length,
        support_temperature=support_temperature,
    )
    validation_raw_support, validation_calibrated_support = score_bundle_with_support_scorer(
        support_runtime["support_trainer"],
        support_runtime["support_tokenizer"],
        artifacts["validation"]["examples"],
        validation_bundle,
        max_length=args.max_length,
        support_temperature=support_temperature,
    )

    train_sets = build_stage9_sets(
        artifacts["train"]["examples"],
        train_bundle,
        raw_support_probabilities=train_raw_support,
        calibrated_support_probabilities=train_calibrated_support,
        hard_support_threshold=resolved_hard_support_threshold,
        use_domain_features=not args.disable_domain_features,
    )
    validation_sets = build_stage9_sets(
        artifacts["validation"]["examples"],
        validation_bundle,
        raw_support_probabilities=validation_raw_support,
        calibrated_support_probabilities=validation_calibrated_support,
        hard_support_threshold=resolved_hard_support_threshold,
        use_domain_features=not args.disable_domain_features,
    )

    candidate_mean, candidate_std = _fit_or_default_standardization(
        flatten_candidate_features(train_sets),
        len(STAGE9_CANDIDATE_FEATURE_NAMES),
    )
    interaction_mean, interaction_std = _fit_or_default_standardization(
        flatten_interaction_features(train_sets),
        len(STAGE9_INTERACTION_FEATURE_NAMES),
    )
    domain_matrix = flatten_domain_features(train_sets)
    domain_dim = domain_matrix.shape[1]
    domain_mean, domain_std = _fit_or_default_standardization(domain_matrix, domain_dim)

    standardized_train_sets = standardize_stage9_sets(
        train_sets,
        candidate_mean=candidate_mean,
        candidate_std=candidate_std,
        interaction_mean=interaction_mean,
        interaction_std=interaction_std,
        domain_mean=domain_mean,
        domain_std=domain_std,
    )
    standardized_validation_sets = standardize_stage9_sets(
        validation_sets,
        candidate_mean=candidate_mean,
        candidate_std=candidate_std,
        interaction_mean=interaction_mean,
        interaction_std=interaction_std,
        domain_mean=domain_mean,
        domain_std=domain_std,
    )

    model = ExactStage9Model(
        len(STAGE9_CANDIDATE_FEATURE_NAMES),
        len(STAGE9_INTERACTION_FEATURE_NAMES),
        domain_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        standardized_train_sets,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_stage9_sets,
    )

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_entry: dict[str, float] | None = None
    best_risk_temperature = 1.0
    best_risk_threshold = args.risk_threshold_max
    best_abstain_margin = args.abstain_margin_min
    training_history: list[dict[str, float]] = []

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        total_loss = 0.0
        total_joint_loss = 0.0
        total_utility_loss = 0.0
        total_risk_loss = 0.0
        total_abstain_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            outputs = model(
                batch["candidate_features"].to(device),
                batch["candidate_mask"].to(device),
                batch["interaction_features"].to(device),
                batch["domain_features"].to(device),
                risk_penalty=args.risk_penalty,
            )
            loss, components = compute_stage9_losses(
                outputs,
                {
                    "candidate_labels": batch["candidate_labels"].to(device),
                    "candidate_risk_labels": batch["candidate_risk_labels"].to(device),
                    "candidate_weights": batch["candidate_weights"].to(device),
                    "candidate_mask": batch["candidate_mask"].to(device),
                    "target_indices": batch["target_indices"].to(device),
                    "abstain_targets": batch["abstain_targets"].to(device),
                },
                joint_loss_weight=args.joint_loss_weight,
                utility_loss_weight=args.utility_loss_weight,
                risk_loss_weight=args.risk_loss_weight,
                abstain_loss_weight=args.abstain_loss_weight,
                tail_risk_weight=args.tail_risk_weight,
                randomization_scale=args.randomization_scale,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_joint_loss += float(components["joint_loss"])
            total_utility_loss += float(components["utility_loss"])
            total_risk_loss += float(components["risk_loss"])
            total_abstain_loss += float(components["abstain_loss"])
            batch_count += 1

        validation_outputs = _predict_stage9_outputs(
            model,
            standardized_validation_sets,
            batch_size=args.eval_batch_size,
            device=device,
            risk_penalty=args.risk_penalty,
        )
        risk_logits, risk_labels = _collect_risk_calibration_arrays(standardized_validation_sets, validation_outputs)
        risk_temperature = fit_temperature_scaler(
            risk_logits,
            risk_labels,
            temperature_min=args.risk_temperature_min,
            temperature_max=args.risk_temperature_max,
            temperature_step=args.risk_temperature_step,
            num_bins=args.calibration_bins,
        )
        (
            selected_risk_threshold,
            selected_abstain_margin,
            validation_metrics,
            validation_mix,
            validation_overabstain,
            _,
        ) = search_stage9_boundary(
            standardized_validation_sets,
            validation_outputs,
            artifacts["validation"]["references"],
            risk_temperature=risk_temperature,
            risk_penalty=args.risk_penalty,
            risk_threshold_min=args.risk_threshold_min,
            risk_threshold_max=args.risk_threshold_max,
            risk_threshold_step=args.risk_threshold_step,
            abstain_margin_min=args.abstain_margin_min,
            abstain_margin_max=args.abstain_margin_max,
            abstain_margin_step=args.abstain_margin_step,
            match_f1_threshold=args.match_f1_threshold,
            max_unsupported_answer_rate=args.max_unsupported_answer_rate,
            max_overabstain_rate=args.max_overabstain_rate,
            hard_support_threshold=resolved_hard_support_threshold,
        )
        validation_losses = _evaluate_stage9_model(
            model,
            standardized_validation_sets,
            batch_size=args.eval_batch_size,
            device=device,
            risk_penalty=args.risk_penalty,
            joint_loss_weight=args.joint_loss_weight,
            utility_loss_weight=args.utility_loss_weight,
            risk_loss_weight=args.risk_loss_weight,
            abstain_loss_weight=args.abstain_loss_weight,
            tail_risk_weight=args.tail_risk_weight,
            randomization_scale=0.0,
        )
        calibration_summary = summarize_binary_calibration(
            sigmoid_scores(risk_logits, temperature=risk_temperature),
            risk_labels,
            num_bins=args.calibration_bins,
            threshold_gap_min_count=args.threshold_gap_min_count,
        )

        history_entry = {
            "epoch": float(epoch),
            "train_total_loss": float(total_loss / max(1, batch_count)),
            "train_joint_loss": float(total_joint_loss / max(1, batch_count)),
            "train_utility_loss": float(total_utility_loss / max(1, batch_count)),
            "train_risk_loss": float(total_risk_loss / max(1, batch_count)),
            "train_abstain_loss": float(total_abstain_loss / max(1, batch_count)),
            "validation_total_loss": float(validation_losses["total_loss"]),
            "validation_joint_loss": float(validation_losses["joint_loss"]),
            "validation_utility_loss": float(validation_losses["utility_loss"]),
            "validation_risk_loss": float(validation_losses["risk_loss"]),
            "validation_abstain_loss": float(validation_losses["abstain_loss"]),
            "validation_risk_temperature": float(risk_temperature),
            "validation_selected_risk_threshold": float(selected_risk_threshold),
            "validation_selected_abstain_margin": float(selected_abstain_margin),
            "validation_risk_brier": float(calibration_summary["brier_score"]),
            "validation_risk_ece": float(calibration_summary["ece"]),
            "validation_overall_f1": float(validation_metrics["overall_f1"]),
            "validation_answerable_f1": float(validation_metrics["answerable_f1"]),
            "validation_unsupported_answer_rate": float(validation_metrics["unsupported_answer_rate"]),
            "validation_supported_answer_rate": float(validation_mix["supported_answer_rate"]),
            "validation_answer_rate": float(validation_mix["answer_rate"]),
            "validation_abstain_f1": float(validation_metrics["abstain_f1"]),
            "validation_overabstain_rate": float(validation_overabstain["overabstain_rate"]),
            "randomization_scale": float(args.randomization_scale),
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
            best_risk_temperature = float(risk_temperature)
            best_risk_threshold = float(selected_risk_threshold)
            best_abstain_margin = float(selected_abstain_margin)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    config = Stage9Config(
        base_model_name=str(args.base_model_name),
        support_temperature=float(support_temperature),
        hard_support_threshold=float(resolved_hard_support_threshold),
        candidate_input_dim=len(STAGE9_CANDIDATE_FEATURE_NAMES),
        interaction_input_dim=len(STAGE9_INTERACTION_FEATURE_NAMES),
        domain_input_dim=domain_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        candidate_feature_names=list(STAGE9_CANDIDATE_FEATURE_NAMES),
        interaction_feature_names=list(STAGE9_INTERACTION_FEATURE_NAMES),
        domain_feature_names=list(STAGE9_DOMAIN_FEATURE_NAMES[:domain_dim]),
        candidate_feature_mean=candidate_mean.tolist(),
        candidate_feature_std=candidate_std.tolist(),
        interaction_feature_mean=interaction_mean.tolist(),
        interaction_feature_std=interaction_std.tolist(),
        domain_feature_mean=domain_mean.tolist(),
        domain_feature_std=domain_std.tolist(),
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
        abstain_loss_weight=args.abstain_loss_weight,
        joint_loss_weight=args.joint_loss_weight,
        tail_risk_weight=args.tail_risk_weight,
        randomization_scale=args.randomization_scale,
        risk_penalty=args.risk_penalty,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
        max_overabstain_rate=args.max_overabstain_rate,
        best_validation_risk_temperature=best_risk_temperature,
        best_validation_risk_threshold=best_risk_threshold,
        best_validation_abstain_margin=best_abstain_margin,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        clean_splitting=bool(args.clean_splitting),
        max_test_samples=args.max_test_samples,
        use_domain_features=not args.disable_domain_features,
    )
    _save_stage9_model(model.cpu(), args.output_dir, config, training_history)

    run_metadata = {
        "stage": "stage9-risk-generalization-train",
        "base_model_name": str(args.base_model_name),
        "proposal_model_dir": str(proposal_model_dir),
        "support_model_dir": str(support_model_dir),
        "train_examples": len(artifacts["train"]["examples"]),
        "validation_examples": len(artifacts["validation"]["examples"]),
        "train_candidate_count": len(train_bundle.records),
        "validation_candidate_count": len(validation_bundle.records),
        "best_epoch": best_entry["epoch"] if best_entry is not None else None,
        "best_validation_risk_temperature": best_risk_temperature,
        "best_validation_risk_threshold": best_risk_threshold,
        "best_validation_abstain_margin": best_abstain_margin,
        "support_temperature": float(support_temperature),
        "hard_support_threshold": float(resolved_hard_support_threshold),
        "clean_splitting": bool(args.clean_splitting),
        "max_test_samples": args.max_test_samples,
        "use_domain_features": bool(not args.disable_domain_features),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _evaluate_stage9(args: argparse.Namespace) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model, config = load_stage9_model(args.model_path)
    resolved_hard_support_threshold = args.hard_support_threshold or config.hard_support_threshold
    clean_splitting = bool(args.clean_splitting or config.clean_splitting)
    resolved_max_test_samples = args.max_test_samples if args.max_test_samples is not None else config.max_test_samples
    final_split_name = "test" if clean_splitting else "dev"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    proposal_model_dir = args.model_path / STAGE9_PROPOSAL_DIRNAME
    support_model_dir = args.model_path / STAGE9_SUPPORT_DIRNAME

    artifacts = _prepare_proposal_prediction_artifacts(
        model_path=proposal_model_dir,
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
    support_runtime = _prepare_support_runtime(
        support_model_path=support_model_dir,
        eval_batch_size=args.eval_batch_size,
        output_dir=args.output_path.parent / "tmp-stage9-support-eval",
    )
    validation_raw_support, validation_calibrated_support = score_bundle_with_support_scorer(
        support_runtime["support_trainer"],
        support_runtime["support_tokenizer"],
        artifacts["validation"]["examples"],
        validation_bundle,
        max_length=args.max_length,
        support_temperature=float(config.support_temperature),
    )
    final_raw_support, final_calibrated_support = score_bundle_with_support_scorer(
        support_runtime["support_trainer"],
        support_runtime["support_tokenizer"],
        artifacts[final_split_name]["examples"],
        final_bundle,
        max_length=args.max_length,
        support_temperature=float(config.support_temperature),
    )

    validation_sets = build_stage9_sets(
        artifacts["validation"]["examples"],
        validation_bundle,
        raw_support_probabilities=validation_raw_support,
        calibrated_support_probabilities=validation_calibrated_support,
        hard_support_threshold=float(resolved_hard_support_threshold),
        use_domain_features=bool(config.use_domain_features),
    )
    final_sets = build_stage9_sets(
        artifacts[final_split_name]["examples"],
        final_bundle,
        raw_support_probabilities=final_raw_support,
        calibrated_support_probabilities=final_calibrated_support,
        hard_support_threshold=float(resolved_hard_support_threshold),
        use_domain_features=bool(config.use_domain_features),
    )

    standardized_validation_sets = standardize_stage9_sets(
        validation_sets,
        candidate_mean=np.asarray(config.candidate_feature_mean, dtype=np.float32),
        candidate_std=np.asarray(config.candidate_feature_std, dtype=np.float32),
        interaction_mean=np.asarray(config.interaction_feature_mean, dtype=np.float32),
        interaction_std=np.asarray(config.interaction_feature_std, dtype=np.float32),
        domain_mean=np.asarray(config.domain_feature_mean, dtype=np.float32),
        domain_std=np.asarray(config.domain_feature_std, dtype=np.float32),
    )
    standardized_final_sets = standardize_stage9_sets(
        final_sets,
        candidate_mean=np.asarray(config.candidate_feature_mean, dtype=np.float32),
        candidate_std=np.asarray(config.candidate_feature_std, dtype=np.float32),
        interaction_mean=np.asarray(config.interaction_feature_mean, dtype=np.float32),
        interaction_std=np.asarray(config.interaction_feature_std, dtype=np.float32),
        domain_mean=np.asarray(config.domain_feature_mean, dtype=np.float32),
        domain_std=np.asarray(config.domain_feature_std, dtype=np.float32),
    )

    validation_outputs = _predict_stage9_outputs(
        model,
        standardized_validation_sets,
        batch_size=config.eval_batch_size,
        device=device,
        risk_penalty=config.risk_penalty,
    )
    final_outputs = _predict_stage9_outputs(
        model,
        standardized_final_sets,
        batch_size=config.eval_batch_size,
        device=device,
        risk_penalty=config.risk_penalty,
    )

    risk_logits, risk_labels = _collect_risk_calibration_arrays(standardized_validation_sets, validation_outputs)
    risk_temperature = fit_temperature_scaler(
        risk_logits,
        risk_labels,
        temperature_min=args.risk_temperature_min,
        temperature_max=args.risk_temperature_max,
        temperature_step=args.risk_temperature_step,
        num_bins=args.calibration_bins,
    )
    risk_calibration_summary = summarize_binary_calibration(
        sigmoid_scores(risk_logits, temperature=risk_temperature),
        risk_labels,
        num_bins=args.calibration_bins,
        threshold_gap_min_count=args.threshold_gap_min_count,
    )

    (
        selected_risk_threshold,
        selected_abstain_margin,
        validation_metrics,
        validation_mix,
        validation_overabstain,
        threshold_sweep,
    ) = search_stage9_boundary(
        standardized_validation_sets,
        validation_outputs,
        artifacts["validation"]["references"],
        risk_temperature=risk_temperature,
        risk_penalty=config.risk_penalty,
        risk_threshold_min=args.risk_threshold_min,
        risk_threshold_max=args.risk_threshold_max,
        risk_threshold_step=args.risk_threshold_step,
        abstain_margin_min=args.abstain_margin_min,
        abstain_margin_max=args.abstain_margin_max,
        abstain_margin_step=args.abstain_margin_step,
        match_f1_threshold=args.match_f1_threshold,
        max_unsupported_answer_rate=args.max_unsupported_answer_rate,
        max_overabstain_rate=args.max_overabstain_rate,
        hard_support_threshold=float(resolved_hard_support_threshold),
    )

    final_predictions = postprocess_stage9_predictions(
        standardized_final_sets,
        final_outputs,
        risk_temperature=risk_temperature,
        risk_penalty=config.risk_penalty,
        risk_threshold=selected_risk_threshold,
        abstain_margin=selected_abstain_margin,
        hard_support_threshold=float(resolved_hard_support_threshold),
    )
    final_metrics = compute_stage1_metrics(final_predictions, artifacts[final_split_name]["references"])
    final_mix = compute_answer_support_mix(
        final_predictions,
        artifacts[final_split_name]["references"],
        match_f1_threshold=args.match_f1_threshold,
    )
    final_overabstain = compute_overabstain_stats(final_predictions, artifacts[final_split_name]["references"])

    output = {
        "stage": "stage9-risk-generalization-eval",
        "model_path": str(args.model_path),
        "proposal_model_path": str(proposal_model_dir),
        "support_model_path": str(support_model_dir),
        "support_temperature": float(config.support_temperature),
        "hard_support_threshold": float(resolved_hard_support_threshold),
        "clean_splitting": clean_splitting,
        "final_eval_split": final_split_name,
        "risk_temperature": float(risk_temperature),
        "risk_calibration_summary": risk_calibration_summary,
        "selected_risk_threshold": float(selected_risk_threshold),
        "selected_abstain_margin": float(selected_abstain_margin),
        "max_unsupported_answer_rate": args.max_unsupported_answer_rate,
        "max_overabstain_rate": args.max_overabstain_rate,
        "risk_penalty": config.risk_penalty,
        "validation_metrics": validation_metrics,
        "validation_mix": validation_mix,
        "validation_overabstain": validation_overabstain,
        "threshold_sweep": threshold_sweep,
        "final_metrics": final_metrics,
        "final_mix": final_mix,
        "final_overabstain": final_overabstain,
        "final_predictions": final_predictions,
        "candidate_feature_names": list(config.candidate_feature_names),
        "interaction_feature_names": list(config.interaction_feature_names),
        "domain_feature_names": list(config.domain_feature_names),
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
        _train_stage9(args)
        return
    if args.command == "evaluate":
        _evaluate_stage9(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
