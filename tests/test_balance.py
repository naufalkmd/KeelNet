from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.balance import (
    CandidateBundle,
    CandidateRecord,
    build_candidate_bundle,
    postprocess_stage6_predictions,
    search_balance_threshold,
)


class Stage6BalanceTests(unittest.TestCase):
    def test_build_candidate_bundle_marks_supported_candidates(self):
        examples = [{"id": "a1", "context": "alpha beta gamma delta"}]
        features = [
            {
                "example_id": "a1",
                "offset_mapping": [None, (0, 5), (6, 10), (11, 16), (17, 22)],
                "cls_index": 0,
            }
        ]
        raw_predictions = (
            [[-5.0, 0.0, 8.0, 7.0, 0.0]],
            [[-5.0, 0.0, 8.0, 7.0, 0.0]],
            [-2.0],
            [2.0],
        )
        references = {"a1": {"answers": ["beta"], "is_answerable": True}}

        bundle = build_candidate_bundle(
            examples,
            features,
            raw_predictions,
            references,
            n_best_size=3,
            max_answer_length=4,
            max_candidates_per_example=3,
            max_candidates_per_feature=3,
            match_f1_threshold=0.5,
            hard_negative_weight=1.5,
        )

        answers_to_labels = {record.answer_text: record.label for record in bundle.records}
        self.assertEqual(answers_to_labels["beta"], 1.0)
        self.assertEqual(answers_to_labels["gamma"], 0.0)
        self.assertGreaterEqual(len(bundle.records), 2)

    def test_postprocess_stage6_predictions_applies_controller_gate(self):
        bundle = CandidateBundle(
            feature_names=["f"] * 8,
            all_example_ids=["a1"],
            records=[
                CandidateRecord(
                    example_id="a1",
                    answer_text="beta",
                    span_score=9.0,
                    score_gap_to_best=0.0,
                    score_margin_to_next=2.0,
                    keep_probability=0.8,
                    support_probability=0.9,
                    abstain_probability=0.1,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=1.0,
                    hard_negative_weight=1.0,
                )
            ],
            features=np.zeros((1, 8), dtype=np.float32),
            labels=np.asarray([1.0], dtype=np.float32),
            sample_weights=np.asarray([1.0], dtype=np.float32),
        )

        predictions = postprocess_stage6_predictions(
            bundle,
            np.asarray([0.45], dtype=np.float32),
            threshold=0.50,
        )

        self.assertEqual(predictions["a1"]["decision"], "abstain")
        self.assertEqual(predictions["a1"]["abstain_reason"], "controller_gate")
        self.assertAlmostEqual(predictions["a1"]["scores"]["keep_probability"], 0.8)

    def test_search_balance_threshold_prefers_constraint_satisfying_value(self):
        bundle = CandidateBundle(
            feature_names=["f"] * 8,
            all_example_ids=["a1", "u1"],
            records=[
                CandidateRecord(
                    example_id="a1",
                    answer_text="beta",
                    span_score=9.0,
                    score_gap_to_best=0.0,
                    score_margin_to_next=2.0,
                    keep_probability=0.8,
                    support_probability=0.9,
                    abstain_probability=0.1,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=1.0,
                    hard_negative_weight=1.0,
                ),
                CandidateRecord(
                    example_id="u1",
                    answer_text="paris",
                    span_score=8.5,
                    score_gap_to_best=0.0,
                    score_margin_to_next=1.0,
                    keep_probability=0.7,
                    support_probability=0.6,
                    abstain_probability=0.2,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=0.0,
                    hard_negative_weight=1.5,
                ),
            ],
            features=np.zeros((2, 8), dtype=np.float32),
            labels=np.asarray([1.0, 0.0], dtype=np.float32),
            sample_weights=np.asarray([1.0, 1.5], dtype=np.float32),
        )
        references = {
            "a1": {"answers": ["beta"], "is_answerable": True},
            "u1": {"answers": [], "is_answerable": False},
        }

        threshold, metrics, mix, sweep = search_balance_threshold(
            bundle,
            np.asarray([0.95, 0.45], dtype=np.float32),
            references,
            threshold_min=0.40,
            threshold_max=0.80,
            threshold_step=0.20,
            match_f1_threshold=0.5,
            max_unsupported_answer_rate=0.0,
        )

        self.assertTrue(any(entry["constraint_satisfied"] for entry in sweep))
        self.assertGreaterEqual(threshold, 0.60)
        self.assertEqual(metrics["unsupported_answer_rate"], 0.0)
        self.assertEqual(mix["supported_answers_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
