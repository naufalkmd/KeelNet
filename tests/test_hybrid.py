from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.hybrid import (
    HybridBundle,
    HybridCandidate,
    build_hybrid_bundle,
    postprocess_hybrid_predictions,
    search_hybrid_threshold,
)
from keelnet.balance import CandidateBundle, CandidateRecord


class Stage8HybridTests(unittest.TestCase):
    def test_build_hybrid_bundle_adds_calibrated_support_features(self):
        candidate_bundle = CandidateBundle(
            feature_names=[],
            all_example_ids=["a1"],
            records=[
                CandidateRecord(
                    example_id="a1",
                    answer_text="beta",
                    span_score=9.0,
                    score_gap_to_best=0.0,
                    score_margin_to_next=2.0,
                    keep_probability=0.8,
                    support_probability=0.6,
                    abstain_probability=0.2,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=1.0,
                    hard_negative_weight=1.0,
                ),
                CandidateRecord(
                    example_id="a1",
                    answer_text="gamma",
                    span_score=8.0,
                    score_gap_to_best=1.0,
                    score_margin_to_next=1.0,
                    keep_probability=0.7,
                    support_probability=0.4,
                    abstain_probability=0.3,
                    answer_length_tokens=1.0,
                    normalized_rank=1.0,
                    label=0.0,
                    hard_negative_weight=1.2,
                ),
            ],
            features=np.zeros((2, 1), dtype=np.float32),
            labels=np.asarray([1.0, 0.0], dtype=np.float32),
            sample_weights=np.asarray([1.0, 1.2], dtype=np.float32),
        )

        bundle = build_hybrid_bundle(
            candidate_bundle,
            np.asarray([0.75, 0.20], dtype=np.float32),
            np.asarray([0.82, 0.35], dtype=np.float32),
            hard_support_threshold=0.50,
            hard_negative_weight=1.5,
        )

        self.assertAlmostEqual(bundle.records[0].calibrated_support_probability, 0.82, places=6)
        self.assertEqual(bundle.records[0].support_gate_pass, 1.0)
        self.assertEqual(bundle.records[1].support_gate_pass, 0.0)
        self.assertEqual(bundle.features.shape[1], len(bundle.feature_names))
        self.assertGreater(bundle.sample_weights[1], 1.2)

    def test_postprocess_hybrid_predictions_falls_back_to_safe_candidate(self):
        bundle = HybridBundle(
            feature_names=[],
            all_example_ids=["a1"],
            records=[
                HybridCandidate(
                    example_id="a1",
                    answer_text="unsafe",
                    span_score=9.0,
                    score_gap_to_best=0.0,
                    score_margin_to_next=1.0,
                    keep_probability=0.9,
                    stage5_support_probability=0.8,
                    abstain_probability=0.1,
                    raw_verifier_support_probability=0.30,
                    calibrated_support_probability=0.35,
                    support_gate_pass=0.0,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=0.0,
                    hard_negative_weight=2.0,
                ),
                HybridCandidate(
                    example_id="a1",
                    answer_text="safe",
                    span_score=8.0,
                    score_gap_to_best=1.0,
                    score_margin_to_next=1.0,
                    keep_probability=0.7,
                    stage5_support_probability=0.6,
                    abstain_probability=0.2,
                    raw_verifier_support_probability=0.70,
                    calibrated_support_probability=0.75,
                    support_gate_pass=1.0,
                    answer_length_tokens=1.0,
                    normalized_rank=1.0,
                    label=1.0,
                    hard_negative_weight=1.0,
                ),
            ],
            features=np.zeros((2, 12), dtype=np.float32),
            labels=np.asarray([0.0, 1.0], dtype=np.float32),
            sample_weights=np.asarray([2.0, 1.0], dtype=np.float32),
        )

        predictions = postprocess_hybrid_predictions(
            bundle,
            np.asarray([0.92, 0.71], dtype=np.float32),
            threshold=0.60,
            hard_support_threshold=0.50,
        )

        self.assertEqual(predictions["a1"]["decision"], "answer")
        self.assertEqual(predictions["a1"]["answer"], "safe")
        self.assertAlmostEqual(predictions["a1"]["support"]["score"], 0.75, places=6)

    def test_search_hybrid_threshold_prefers_constraint_satisfying_value(self):
        bundle = HybridBundle(
            feature_names=[],
            all_example_ids=["a1", "u1"],
            records=[
                HybridCandidate(
                    example_id="a1",
                    answer_text="beta",
                    span_score=9.0,
                    score_gap_to_best=0.0,
                    score_margin_to_next=2.0,
                    keep_probability=0.8,
                    stage5_support_probability=0.7,
                    abstain_probability=0.2,
                    raw_verifier_support_probability=0.85,
                    calibrated_support_probability=0.90,
                    support_gate_pass=1.0,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=1.0,
                    hard_negative_weight=1.0,
                ),
                HybridCandidate(
                    example_id="u1",
                    answer_text="paris",
                    span_score=8.5,
                    score_gap_to_best=0.0,
                    score_margin_to_next=1.0,
                    keep_probability=0.7,
                    stage5_support_probability=0.6,
                    abstain_probability=0.2,
                    raw_verifier_support_probability=0.25,
                    calibrated_support_probability=0.30,
                    support_gate_pass=0.0,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=0.0,
                    hard_negative_weight=2.0,
                ),
            ],
            features=np.zeros((2, 12), dtype=np.float32),
            labels=np.asarray([1.0, 0.0], dtype=np.float32),
            sample_weights=np.asarray([1.0, 2.0], dtype=np.float32),
        )
        references = {
            "a1": {"answers": ["beta"], "is_answerable": True},
            "u1": {"answers": [], "is_answerable": False},
        }

        threshold, metrics, mix, sweep = search_hybrid_threshold(
            bundle,
            np.asarray([0.95, 0.65], dtype=np.float32),
            references,
            threshold_min=0.40,
            threshold_max=0.80,
            threshold_step=0.20,
            hard_support_threshold=0.50,
            match_f1_threshold=0.5,
            max_unsupported_answer_rate=0.0,
        )

        self.assertTrue(any(entry["constraint_satisfied"] for entry in sweep))
        self.assertGreaterEqual(threshold, 0.40)
        self.assertEqual(metrics["unsupported_answer_rate"], 0.0)
        self.assertEqual(mix["supported_answers_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
