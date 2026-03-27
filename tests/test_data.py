from pathlib import Path
import sys
import unittest

from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.data import build_stage1_splits_from_raw, build_stage2_verification_splits


def _make_raw_train_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": "t1",
                "question": "Who wrote Hamlet?",
                "context": "William Shakespeare wrote Hamlet.",
                "answers": {"text": ["William Shakespeare"], "answer_start": [0]},
            },
            {
                "id": "t2",
                "question": "Who discovered penicillin?",
                "context": "Alexander Fleming discovered penicillin.",
                "answers": {"text": ["Alexander Fleming"], "answer_start": [0]},
            },
            {
                "id": "t3",
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France.",
                "answers": {"text": ["Paris"], "answer_start": [0]},
            },
            {
                "id": "t4",
                "question": "Who painted Starry Night?",
                "context": "Vincent van Gogh painted Starry Night.",
                "answers": {"text": ["Vincent van Gogh"], "answer_start": [0]},
            },
            {
                "id": "t5",
                "question": "Who wrote Hamlet on Mars?",
                "context": "The passage says nothing about Mars.",
                "answers": {"text": [], "answer_start": []},
            },
            {
                "id": "t6",
                "question": "Who discovered penicillin on Mars?",
                "context": "The passage says nothing about Mars.",
                "answers": {"text": [], "answer_start": []},
            },
            {
                "id": "t7",
                "question": "What is the capital of France on Mars?",
                "context": "The passage says nothing about Mars.",
                "answers": {"text": [], "answer_start": []},
            },
            {
                "id": "t8",
                "question": "Who painted Starry Night on Mars?",
                "context": "The passage says nothing about Mars.",
                "answers": {"text": [], "answer_start": []},
            },
        ]
    )


def _make_raw_eval_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": "e1",
                "question": "Who developed relativity?",
                "context": "Albert Einstein developed relativity.",
                "answers": {"text": ["Albert Einstein"], "answer_start": [0]},
            },
            {
                "id": "e2",
                "question": "Where is the Eiffel Tower?",
                "context": "The Eiffel Tower is in Paris.",
                "answers": {"text": ["Paris"], "answer_start": [28]},
            },
            {
                "id": "e3",
                "question": "Who developed relativity on Mars?",
                "context": "The passage says nothing about Mars.",
                "answers": {"text": [], "answer_start": []},
            },
            {
                "id": "e4",
                "question": "Where is the Eiffel Tower on Mars?",
                "context": "The passage says nothing about Mars.",
                "answers": {"text": [], "answer_start": []},
            },
        ]
    )


class Stage1SplitTest(unittest.TestCase):
    def test_build_stage1_clean_splits_from_raw(self):
        splits = build_stage1_splits_from_raw(
            train=_make_raw_train_dataset(),
            eval_source=_make_raw_eval_dataset(),
            validation_size=0.25,
            seed=42,
            answer_only_train=False,
            clean_splitting=True,
        )

        self.assertEqual(set(splits.keys()), {"train", "validation", "test"})
        self.assertEqual(len(splits["train"]), 6)
        self.assertEqual(len(splits["validation"]), 2)
        self.assertEqual(len(splits["test"]), 4)

        train_ids = set(splits["train"]["id"])
        validation_ids = set(splits["validation"]["id"])
        test_ids = set(splits["test"]["id"])

        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))

    def test_answer_only_train_only_filters_training_split(self):
        splits = build_stage1_splits_from_raw(
            train=_make_raw_train_dataset(),
            eval_source=_make_raw_eval_dataset(),
            validation_size=0.25,
            seed=42,
            answer_only_train=True,
            clean_splitting=True,
        )

        self.assertTrue(all(len(example["answers"]["text"]) > 0 for example in splits["train"]))
        self.assertTrue(any(len(example["answers"]["text"]) == 0 for example in splits["validation"]))
        self.assertTrue(any(len(example["answers"]["text"]) == 0 for example in splits["test"]))


class Stage2DataTest(unittest.TestCase):
    def test_build_stage2_verification_splits(self):
        raw_split = Dataset.from_list(
            [
                {
                    "id": "a1",
                    "question": "Who discovered penicillin?",
                    "context": "Alexander Fleming discovered penicillin in 1928.",
                    "answers": {"text": ["Alexander Fleming"], "answer_start": [0]},
                },
                {
                    "id": "u1",
                    "question": "Who discovered penicillin on Mars?",
                    "context": "The passage discusses penicillin but says nothing about Mars.",
                    "answers": {"text": [], "answer_start": []},
                },
            ]
        )
        splits = DatasetDict({"train": raw_split, "validation": raw_split, "dev": raw_split})

        verification_splits = build_stage2_verification_splits(
            splits,
            seed=42,
            negatives_per_answerable=1,
            negatives_per_unanswerable=1,
        )

        self.assertEqual(set(verification_splits.keys()), {"train", "validation", "dev"})
        train_records = verification_splits["train"]
        self.assertEqual(len(train_records), 3)

        positive_records = [record for record in train_records if record["support_label"] == 1]
        negative_records = [record for record in train_records if record["support_label"] == 0]

        self.assertEqual(len(positive_records), 1)
        self.assertEqual(len(negative_records), 2)
        self.assertEqual(positive_records[0]["candidate_answer"], "Alexander Fleming")
        self.assertTrue(all(record["candidate_answer"] for record in negative_records))


if __name__ == "__main__":
    unittest.main()
