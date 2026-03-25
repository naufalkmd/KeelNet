"""Training entrypoint for the Stage 1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

from keelnet.config import (
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_SIZE,
    RUN_MODE_ABSTAIN,
    RUN_MODE_BASELINE,
    RUN_MODES,
)
from keelnet.data import load_stage1_splits, prepare_train_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=RUN_MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--doc-stride", type=int, default=DEFAULT_DOC_STRIDE)
    parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    answer_only_train = args.mode == RUN_MODE_BASELINE
    splits = load_stage1_splits(
        validation_size=args.validation_size,
        seed=args.seed,
        answer_only_train=answer_only_train,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    train_dataset = splits["train"].map(
        lambda batch: prepare_train_features(batch, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=splits["train"].column_names,
        desc="Tokenizing training data",
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
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    run_metadata = {
        "mode": args.mode,
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
        "train_examples": len(splits["train"]),
        "validation_examples": len(splits["validation"]),
        "dev_examples": len(splits["dev"]),
    }
    metadata_path = args.output_dir / "run_config.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
