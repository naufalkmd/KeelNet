"""Dataset loading and preprocessing for Stage 1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


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
    train = raw["train"]
    dev = raw["validation"]

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
        dev = dev.select(range(min(max_eval_samples, len(dev))))

    return DatasetDict(
        {
            "train": train_split,
            "validation": validation_split,
            "dev": dev,
        }
    )


def build_reference_index(dataset: Dataset) -> dict[str, dict[str, Any]]:
    """Build an id -> reference mapping for metrics."""

    references: dict[str, dict[str, Any]] = {}
    for example in dataset:
        references[str(example["id"])] = {
            "answers": list(example["answers"]["text"]),
            "is_answerable": is_answerable(example),
        }
    return references


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
