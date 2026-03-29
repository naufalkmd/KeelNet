from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.stage9 import (
    CandidateBundle,
    CandidateRecord,
    STAGE9_CANDIDATE_FEATURE_NAMES,
    STAGE9_DOMAIN_FEATURE_NAMES,
    STAGE9_INTERACTION_FEATURE_NAMES,
    build_stage9_sets,
    postprocess_stage9_predictions,
    search_stage9_boundary,
)


class Stage9ExactArchitectureTests(unittest.TestCase):
    def test_build_stage9_sets_creates_targets_and_feature_paths(self):
        examples = [
            {
                "id": "a1",
                "question": "Who wrote Hamlet?",
                "context": "William Shakespeare wrote Hamlet.",
                "answers": {"text": ["William Shakespeare"]},
            },
            {
                "id": "u1",
                "question": "Where is Mars City?",
                "context": "The passage says nothing about that.",
                "answers": {"text": []},
            },
        ]
        bundle = CandidateBundle(
            feature_names=[],
            all_example_ids=["a1", "u1"],
            records=[
                CandidateRecord(
                    example_id="a1",
                    answer_text="William Shakespeare",
                    span_score=9.0,
                    score_gap_to_best=0.0,
                    score_margin_to_next=1.5,
                    keep_probability=0.9,
                    support_probability=0.8,
                    abstain_probability=0.1,
                    answer_length_tokens=2.0,
                    normalized_rank=0.0,
                    label=1.0,
                    hard_negative_weight=1.0,
                ),
                CandidateRecord(
                    example_id="a1",
                    answer_text="Christopher Marlowe",
                    span_score=7.5,
                    score_gap_to_best=1.5,
                    score_margin_to_next=0.5,
                    keep_probability=0.6,
                    support_probability=0.3,
                    abstain_probability=0.2,
                    answer_length_tokens=2.0,
                    normalized_rank=1.0,
                    label=0.0,
                    hard_negative_weight=2.0,
                ),
                CandidateRecord(
                    example_id="u1",
                    answer_text="Mars City",
                    span_score=8.0,
                    score_gap_to_best=0.0,
                    score_margin_to_next=0.0,
                    keep_probability=0.7,
                    support_probability=0.4,
                    abstain_probability=0.3,
                    answer_length_tokens=2.0,
                    normalized_rank=0.0,
                    label=0.0,
                    hard_negative_weight=2.0,
                ),
            ],
            features=np.zeros((3, 8), dtype=np.float32),
            labels=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            sample_weights=np.asarray([1.0, 2.0, 2.0], dtype=np.float32),
        )

        action_sets = build_stage9_sets(
            examples,
            bundle,
            raw_support_probabilities=np.asarray([0.82, 0.25, 0.30], dtype=np.float32),
            calibrated_support_probabilities=np.asarray([0.78, 0.20, 0.28], dtype=np.float32),
            hard_support_threshold=0.60,
            use_domain_features=True,
        )

        self.assertEqual(action_sets[0].target_action_index, 0)
        self.assertEqual(action_sets[1].target_action_index, len(action_sets[1].candidates))
        self.assertEqual(len(action_sets[0].candidates[0].candidate_features), len(STAGE9_CANDIDATE_FEATURE_NAMES))
        self.assertEqual(len(action_sets[0].interaction_features), len(STAGE9_INTERACTION_FEATURE_NAMES))
        self.assertEqual(len(action_sets[0].domain_features), len(STAGE9_DOMAIN_FEATURE_NAMES))
        self.assertEqual(action_sets[0].candidates[0].support_gate_pass, 1.0)
        self.assertEqual(action_sets[0].candidates[1].support_gate_pass, 0.0)

    def test_postprocess_stage9_predictions_prefers_safe_candidate(self):
        action_sets = build_stage9_sets(
            [
                {
                    "id": "a1",
                    "question": "Who wrote Hamlet?",
                    "context": "William Shakespeare wrote Hamlet.",
                    "answers": {"text": ["William Shakespeare"]},
                }
            ],
            CandidateBundle(
                feature_names=[],
                all_example_ids=["a1"],
                records=[
                    CandidateRecord(
                        example_id="a1",
                        answer_text="unsafe",
                        span_score=9.0,
                        score_gap_to_best=0.0,
                        score_margin_to_next=1.0,
                        keep_probability=0.9,
                        support_probability=0.7,
                        abstain_probability=0.1,
                        answer_length_tokens=1.0,
                        normalized_rank=0.0,
                        label=0.0,
                        hard_negative_weight=2.0,
                    ),
                    CandidateRecord(
                        example_id="a1",
                        answer_text="safe",
                        span_score=8.5,
                        score_gap_to_best=0.5,
                        score_margin_to_next=0.0,
                        keep_probability=0.8,
                        support_probability=0.75,
                        abstain_probability=0.2,
                        answer_length_tokens=1.0,
                        normalized_rank=1.0,
                        label=1.0,
                        hard_negative_weight=1.0,
                    ),
                ],
                features=np.zeros((2, 8), dtype=np.float32),
                labels=np.asarray([0.0, 1.0], dtype=np.float32),
                sample_weights=np.asarray([2.0, 1.0], dtype=np.float32),
            ),
            raw_support_probabilities=np.asarray([0.65, 0.80], dtype=np.float32),
            calibrated_support_probabilities=np.asarray([0.70, 0.82], dtype=np.float32),
            hard_support_threshold=0.60,
            use_domain_features=False,
        )
        outputs = [
            {
                "candidate_utility_logits": [2.4, 2.2],
                "candidate_risk_logits": [2.2, -2.0],
                "candidate_raw_risk_probabilities": [0.90, 0.12],
                "candidate_raw_scores": [1.5, 2.0],
                "abstain_logit": 1.2,
                "raw_action_probabilities": [0.5, 0.3, 0.2],
            }
        ]

        predictions = postprocess_stage9_predictions(
            action_sets,
            outputs,
            risk_temperature=1.0,
            risk_penalty=1.0,
            risk_threshold=0.50,
            abstain_margin=0.10,
            hard_support_threshold=0.60,
        )

        self.assertEqual(predictions["a1"]["decision"], "answer")
        self.assertEqual(predictions["a1"]["answer"], "safe")
        self.assertAlmostEqual(predictions["a1"]["support"]["score"], 0.82, places=6)

    def test_search_stage9_boundary_finds_constraint_satisfying_pair(self):
        action_sets = build_stage9_sets(
            [
                {
                    "id": "a1",
                    "question": "Who wrote Hamlet?",
                    "context": "William Shakespeare wrote Hamlet.",
                    "answers": {"text": ["William Shakespeare"]},
                },
                {
                    "id": "u1",
                    "question": "Where is Mars City?",
                    "context": "The passage says nothing about that.",
                    "answers": {"text": []},
                },
            ],
            CandidateBundle(
                feature_names=[],
                all_example_ids=["a1", "u1"],
                records=[
                    CandidateRecord(
                        example_id="a1",
                        answer_text="William Shakespeare",
                        span_score=9.0,
                        score_gap_to_best=0.0,
                        score_margin_to_next=0.0,
                        keep_probability=0.9,
                        support_probability=0.8,
                        abstain_probability=0.1,
                        answer_length_tokens=2.0,
                        normalized_rank=0.0,
                        label=1.0,
                        hard_negative_weight=1.0,
                    ),
                    CandidateRecord(
                        example_id="u1",
                        answer_text="Mars City",
                        span_score=8.5,
                        score_gap_to_best=0.0,
                        score_margin_to_next=0.0,
                        keep_probability=0.8,
                        support_probability=0.7,
                        abstain_probability=0.1,
                        answer_length_tokens=2.0,
                        normalized_rank=0.0,
                        label=0.0,
                        hard_negative_weight=2.0,
                    ),
                ],
                features=np.zeros((2, 8), dtype=np.float32),
                labels=np.asarray([1.0, 0.0], dtype=np.float32),
                sample_weights=np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            raw_support_probabilities=np.asarray([0.85, 0.72], dtype=np.float32),
            calibrated_support_probabilities=np.asarray([0.88, 0.75], dtype=np.float32),
            hard_support_threshold=0.60,
            use_domain_features=False,
        )
        outputs = [
            {
                "candidate_utility_logits": [2.0],
                "candidate_risk_logits": [-2.5],
                "candidate_raw_risk_probabilities": [0.08],
                "candidate_raw_scores": [1.92],
                "abstain_logit": 0.5,
                "raw_action_probabilities": [0.8, 0.2],
            },
            {
                "candidate_utility_logits": [1.8],
                "candidate_risk_logits": [1.5],
                "candidate_raw_risk_probabilities": [0.82],
                "candidate_raw_scores": [0.98],
                "abstain_logit": 0.4,
                "raw_action_probabilities": [0.7, 0.3],
            },
        ]
        references = {
            "a1": {"answers": ["William Shakespeare"], "is_answerable": True},
            "u1": {"answers": [], "is_answerable": False},
        }

        risk_threshold, abstain_margin, metrics, mix, overabstain, sweep = search_stage9_boundary(
            action_sets,
            outputs,
            references,
            risk_temperature=1.0,
            risk_penalty=1.0,
            risk_threshold_min=0.10,
            risk_threshold_max=0.90,
            risk_threshold_step=0.40,
            abstain_margin_min=0.0,
            abstain_margin_max=0.20,
            abstain_margin_step=0.20,
            match_f1_threshold=0.5,
            max_unsupported_answer_rate=0.0,
            max_overabstain_rate=0.0,
            hard_support_threshold=0.60,
        )

        self.assertTrue(any(entry["constraint_satisfied"] for entry in sweep))
        self.assertAlmostEqual(risk_threshold, 0.10, places=6)
        self.assertAlmostEqual(abstain_margin, 0.0, places=6)
        self.assertEqual(metrics["unsupported_answer_rate"], 0.0)
        self.assertEqual(overabstain["overabstain_rate"], 0.0)
        self.assertEqual(mix["supported_answers_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
