"""Dataset loading and preprocessing for Stage 1 and Stage 2."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from keelnet.metrics import f1_score, is_supported_answer, metric_max_over_ground_truths, normalize_answer

_WORD_PATTERN = re.compile(r"\b\w+(?:[-']\w+)*\b")


def is_answerable(example: Mapping[str, Any]) -> bool:
    return len(example["answers"]["text"]) > 0


def load_stage1_splits(
    *,
    validation_size: float,
    seed: int,
    answer_only_train: bool,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    """Load SQuAD v2 and create deterministic train/validation/dev splits."""

    raw = load_dataset("squad_v2")
    return build_stage1_splits_from_raw(
        train=raw["train"],
        eval_source=raw["validation"],
        validation_size=validation_size,
        seed=seed,
        answer_only_train=answer_only_train,
        clean_splitting=False,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def load_stage1_clean_splits(
    *,
    validation_size: float,
    seed: int,
    answer_only_train: bool,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetDict:
    """Load SQuAD v2 and create deterministic train/validation/test splits.

    This variant keeps the official SQuAD validation split untouched as the final
    `test` split instead of exposing it as `dev`.
    """

    raw = load_dataset("squad_v2")
    return build_stage1_splits_from_raw(
        train=raw["train"],
        eval_source=raw["validation"],
        validation_size=validation_size,
        seed=seed,
        answer_only_train=answer_only_train,
        clean_splitting=True,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        max_test_samples=max_test_samples,
    )


def build_stage1_splits_from_raw(
    *,
    train: Dataset,
    eval_source: Dataset,
    validation_size: float,
    seed: int,
    answer_only_train: bool,
    clean_splitting: bool,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetDict:
    """Build Stage 1 splits from raw datasets.

    `clean_splitting=False` returns `train` / `validation` / `dev`.
    `clean_splitting=True` returns `train` / `validation` / `test`.
    """

    answerable_train = train.filter(is_answerable, desc="Filtering answerable train examples")
    unanswerable_train = train.filter(
        lambda example: not is_answerable(example),
        desc="Filtering unanswerable train examples",
    )

    answerable_split = answerable_train.train_test_split(test_size=validation_size, seed=seed)
    unanswerable_split = unanswerable_train.train_test_split(test_size=validation_size, seed=seed)

    train_split = concatenate_datasets(
        [answerable_split["train"], unanswerable_split["train"]]
    ).shuffle(seed=seed)
    validation_split = concatenate_datasets(
        [answerable_split["test"], unanswerable_split["test"]]
    ).shuffle(seed=seed)

    if answer_only_train:
        train_split = train_split.filter(is_answerable, desc="Keeping answerable train examples only")

    if max_train_samples is not None:
        train_split = train_split.select(range(min(max_train_samples, len(train_split))))

    if max_eval_samples is not None:
        validation_split = validation_split.select(range(min(max_eval_samples, len(validation_split))))

    split_dict: dict[str, Dataset] = {
        "train": train_split,
        "validation": validation_split,
    }

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


def build_reference_index(dataset: Dataset) -> dict[str, dict[str, Any]]:
    """Build an id -> reference mapping for metrics."""

    references: dict[str, dict[str, Any]] = {}
    for example in dataset:
        references[str(example["id"])] = {
            "answers": list(example["answers"]["text"]),
            "is_answerable": is_answerable(example),
        }
    return references


def format_verification_query(question: str, candidate_answer: str) -> str:
    """Format the Stage 2 verifier prompt."""

    return f"Question: {question}\nAnswer: {candidate_answer}"


def _sample_negative_answer(
    context: str,
    gold_answers: list[str],
    rng: random.Random,
    *,
    max_tokens: int = 6,
    max_overlap_f1: float = 0.5,
) -> str:
    """Sample a deterministic but diverse unsupported answer span from the context."""

    matches = list(_WORD_PATTERN.finditer(context))
    if not matches:
        return "unsupported"

    normalized_gold = {
        normalize_answer(answer)
        for answer in gold_answers
        if normalize_answer(answer)
    }
    max_tokens = max(1, min(max_tokens, len(matches)))

    for _ in range(64):
        start_index = rng.randrange(len(matches))
        span_length = rng.randint(1, min(max_tokens, len(matches) - start_index))
        end_index = start_index + span_length - 1
        candidate = context[matches[start_index].start() : matches[end_index].end()].strip()
        normalized_candidate = normalize_answer(candidate)
        if not normalized_candidate or normalized_candidate in normalized_gold:
            continue
        if gold_answers and metric_max_over_ground_truths(f1_score, candidate, gold_answers) >= max_overlap_f1:
            continue
        return candidate

    for match in matches:
        candidate = match.group(0).strip()
        normalized_candidate = normalize_answer(candidate)
        if not normalized_candidate or normalized_candidate in normalized_gold:
            continue
        if gold_answers and metric_max_over_ground_truths(f1_score, candidate, gold_answers) >= max_overlap_f1:
            continue
        return candidate

    return "unsupported"


def build_stage2_verification_splits(
    splits: DatasetDict,
    *,
    seed: int,
    negatives_per_answerable: int,
    negatives_per_unanswerable: int,
    qa_predictions_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    support_match_f1_threshold: float = 0.5,
) -> DatasetDict:
    """Create Stage 2 verification examples from Stage 1 raw splits."""

    verification_splits: dict[str, Dataset] = {}
    for split_offset, split_name in enumerate(("train", "validation", "dev")):
        rng = random.Random(seed + split_offset)
        records: list[dict[str, Any]] = []

        for example in splits[split_name]:
            example_id = str(example["id"])
            question = str(example["question"])
            context = str(example["context"])
            gold_answers = [str(answer) for answer in example["answers"]["text"]]
            answerable = bool(gold_answers)

            if answerable:
                records.append(
                    {
                        "id": f"{example_id}::supported",
                        "source_example_id": example_id,
                        "question": question,
                        "context": context,
                        "candidate_answer": gold_answers[0],
                        "support_label": 1,
                        "source_kind": "gold-answer",
                    }
                )
                for negative_index in range(negatives_per_answerable):
                    records.append(
                        {
                            "id": f"{example_id}::negative::{negative_index}",
                            "source_example_id": example_id,
                            "question": question,
                            "context": context,
                            "candidate_answer": _sample_negative_answer(context, gold_answers, rng),
                            "support_label": 0,
                            "source_kind": "sampled-answerable-negative",
                        }
                    )
            else:
                for negative_index in range(negatives_per_unanswerable):
                    records.append(
                        {
                            "id": f"{example_id}::unsupported::{negative_index}",
                            "source_example_id": example_id,
                            "question": question,
                            "context": context,
                            "candidate_answer": _sample_negative_answer(context, [], rng),
                            "support_label": 0,
                            "source_kind": "sampled-unanswerable-negative",
                        }
                    )

            qa_predictions = None if qa_predictions_by_split is None else qa_predictions_by_split.get(split_name, {})
            if qa_predictions is None:
                continue

            prediction = qa_predictions.get(example_id, {"decision": "abstain", "answer": ""})
            if str(prediction.get("decision", "abstain")).lower() != "answer":
                continue

            candidate_answer = str(prediction.get("answer", "")).strip()
            if not normalize_answer(candidate_answer):
                continue

            qa_supported = answerable and is_supported_answer(
                candidate_answer,
                gold_answers,
                match_f1_threshold=support_match_f1_threshold,
            )
            records.append(
                {
                    "id": f"{example_id}::qa-prediction",
                    "source_example_id": example_id,
                    "question": question,
                    "context": context,
                    "candidate_answer": candidate_answer,
                    "support_label": 1 if qa_supported else 0,
                    "source_kind": "qa-prediction-supported" if qa_supported else "qa-prediction-unsupported",
                }
            )

        verification_splits[split_name] = Dataset.from_list(records)

    return DatasetDict(verification_splits)


def prepare_train_features(examples, tokenizer, max_length: int, doc_stride: int):
    """Tokenize train examples using the standard extractive QA procedure."""

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

    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        context_index = 1 if pad_on_right else 0

        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def prepare_stage5_train_features(examples, tokenizer, max_length: int, doc_stride: int):
    """Tokenize Stage 5 train examples with span, keep, and support labels.

    The keep/support labels are defined at the feature level. If an answerable
    example overflows into a feature chunk that does not actually contain the
    gold span, that feature receives keep/support label 0.
    """

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

    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            keep_labels.append(0)
            support_labels.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        context_index = 1 if pad_on_right else 0

        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        answer_in_feature = not (
            offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char
        )
        if not answer_in_feature:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            keep_labels.append(0)
            support_labels.append(0)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)
        keep_labels.append(1)
        support_labels.append(1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    tokenized_examples["keep_labels"] = keep_labels
    tokenized_examples["support_labels"] = support_labels
    return tokenized_examples


def prepare_eval_features(examples, tokenizer, max_length: int, doc_stride: int):
    """Tokenize evaluation examples and keep metadata for post-processing."""

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


def prepare_verification_features(examples, tokenizer, max_length: int):
    """Tokenize Stage 2 verifier examples."""

    queries = [
        format_verification_query(question, candidate_answer)
        for question, candidate_answer in zip(
            examples["question"],
            examples["candidate_answer"],
            strict=False,
        )
    ]
    tokenized_examples = tokenizer(
        queries,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    )
    if "support_label" in examples:
        tokenized_examples["labels"] = examples["support_label"]
    return tokenized_examples
