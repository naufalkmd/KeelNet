from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.learn import (
    keep_probability_from_logits,
    postprocess_stage5_predictions,
    search_keep_threshold,
)


class Stage5LearningTests(unittest.TestCase):
    def test_keep_probability_from_logits_penalizes_high_abstention(self):
        confident_keep = keep_probability_from_logits(-2.0, 2.0)
        hesitant_keep = keep_probability_from_logits(2.0, 2.0)

        self.assertGreater(confident_keep, 0.70)
        self.assertLess(hesitant_keep, 0.15)

    def test_postprocess_stage5_predictions_applies_keep_gate(self):
        examples = [{"id": "ex-1", "context": "alpha beta gamma"}]
        features = [
            {
                "example_id": "ex-1",
                "offset_mapping": [None, (0, 5), (6, 10), (11, 16)],
                "cls_index": 0,
            }
        ]
        raw_predictions = (
            [[-5.0, 0.0, 6.0, 0.0]],
            [[-5.0, 0.0, 6.0, 0.0]],
            [2.0],
            [2.0],
        )

        predictions = postprocess_stage5_predictions(
            examples,
            features,
            raw_predictions,
            keep_threshold=0.50,
            n_best_size=2,
            max_answer_length=4,
        )

        self.assertEqual(predictions["ex-1"]["decision"], "abstain")
        self.assertEqual(predictions["ex-1"]["abstain_reason"], "keep_gate")
        self.assertAlmostEqual(predictions["ex-1"]["scores"]["support_probability"], 0.8808, places=3)

    def test_search_keep_threshold_prefers_constraint_satisfying_value(self):
        examples = [
            {"id": "a1", "context": "alpha beta gamma"},
            {"id": "u1", "context": "paris london rome"},
        ]
        features = [
            {
                "example_id": "a1",
                "offset_mapping": [None, (0, 5), (6, 10), (11, 16)],
                "cls_index": 0,
            },
            {
                "example_id": "u1",
                "offset_mapping": [None, (0, 5), (6, 12), (13, 17)],
                "cls_index": 0,
            },
        ]
        raw_predictions = (
            [
                [-5.0, 0.0, 6.0, 0.0],
                [-5.0, 0.0, 6.0, 0.0],
            ],
            [
                [-5.0, 0.0, 6.0, 0.0],
                [-5.0, 0.0, 6.0, 0.0],
            ],
            [-2.0, -0.5],
            [2.0, 1.0],
        )
        references = {
            "a1": {"answers": ["beta"], "is_answerable": True},
            "u1": {"answers": [], "is_answerable": False},
        }

        threshold, metrics, mix, sweep = search_keep_threshold(
            examples,
            features,
            raw_predictions,
            references,
            n_best_size=2,
            max_answer_length=4,
            threshold_min=0.30,
            threshold_max=0.70,
            threshold_step=0.20,
            match_f1_threshold=0.5,
            max_unsupported_answer_rate=0.0,
        )

        self.assertTrue(any(entry["constraint_satisfied"] for entry in sweep))
        self.assertGreaterEqual(threshold, 0.50)
        self.assertEqual(metrics["unsupported_answer_rate"], 0.0)
        self.assertEqual(mix["supported_answers_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
