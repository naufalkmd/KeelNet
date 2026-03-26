from pathlib import Path
import sys
import unittest

from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.data import build_stage2_verification_splits


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
