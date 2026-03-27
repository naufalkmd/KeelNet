from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.action import (
    ActionCandidate,
    ActionSet,
    build_action_sets,
    compute_overabstain_stats,
    postprocess_action_predictions,
    search_risk_threshold,
)
from keelnet.balance import CandidateBundle, CandidateRecord


class Stage7ActionTests(unittest.TestCase):
    def test_build_action_sets_uses_supported_candidate_or_abstain_target(self):
        examples = [
            {
                "id": "a1",
                "question": "Which token is correct?",
                "answers": {"text": ["beta"]},
            },
            {
                "id": "u1",
                "question": "What city is mentioned?",
                "answers": {"text": []},
            },
        ]
        bundle = CandidateBundle(
            feature_names=[],
            all_example_ids=["a1", "u1"],
            records=[
                CandidateRecord(
                    example_id="a1",
                    answer_text="beta",
                    span_score=9.0,
                    score_gap_to_best=0.0,
                    score_margin_to_next=2.0,
                    keep_probability=0.9,
                    support_probability=0.8,
                    abstain_probability=0.1,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=1.0,
                    hard_negative_weight=1.0,
                ),
                CandidateRecord(
                    example_id="a1",
                    answer_text="gamma",
                    span_score=7.0,
                    score_gap_to_best=2.0,
                    score_margin_to_next=1.0,
                    keep_probability=0.6,
                    support_probability=0.3,
                    abstain_probability=0.2,
                    answer_length_tokens=1.0,
                    normalized_rank=1.0,
                    label=0.0,
                    hard_negative_weight=2.0,
                ),
                CandidateRecord(
                    example_id="u1",
                    answer_text="paris",
                    span_score=8.5,
                    score_gap_to_best=0.0,
                    score_margin_to_next=1.0,
                    keep_probability=0.7,
                    support_probability=0.4,
                    abstain_probability=0.2,
                    answer_length_tokens=1.0,
                    normalized_rank=0.0,
                    label=0.0,
                    hard_negative_weight=2.0,
                ),
            ],
            features=np.zeros((3, 1), dtype=np.float32),
            labels=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            sample_weights=np.asarray([1.0, 2.0, 2.0], dtype=np.float32),
        )

        action_sets = build_action_sets(
            examples,
            bundle,
            stage6_candidate_probabilities=np.asarray([0.95, 0.20, 0.40], dtype=np.float32),
        )

        self.assertEqual(action_sets[0].target_action_index, 0)
        self.assertEqual(action_sets[0].candidates[0].answer_text, "beta")
        self.assertAlmostEqual(action_sets[0].candidates[0].stage6_controller_probability, 0.95, places=6)
        self.assertEqual(action_sets[1].target_action_index, len(action_sets[1].candidates))

    def test_postprocess_action_predictions_applies_risk_shield(self):
        action_sets = [
            ActionSet(
                example_id="a1",
                question="Which token is correct?",
                answerable=True,
                target_action_index=0,
                candidates=(
                    ActionCandidate(
                        example_id="a1",
                        answer_text="beta",
                        span_score=9.0,
                        score_gap_to_best=0.0,
                        score_margin_to_next=2.0,
                        keep_probability=0.9,
                        support_probability=0.8,
                        abstain_probability=0.1,
                        stage6_controller_probability=0.7,
                        answer_length_tokens=1.0,
                        normalized_rank=0.0,
                        question_overlap=0.0,
                        label=1.0,
                        hard_negative_weight=1.0,
                        model_features=(0.0,) * 12,
                    ),
                ),
            )
        ]
        outputs = [
            {
                "candidate_action_probabilities": [0.72],
                "abstain_probability": 0.28,
                "candidate_utility_logits": [2.0],
                "candidate_risk_logits": [1.4],
                "candidate_risk_probabilities": [0.80],
                "candidate_action_scores": [0.91],
                "abstain_logit": 0.10,
            }
        ]

        predictions = postprocess_action_predictions(
            action_sets,
            outputs,
            risk_threshold=0.50,
        )

        self.assertEqual(predictions["a1"]["decision"], "abstain")
        self.assertEqual(predictions["a1"]["abstain_reason"], "risk_shield")
        self.assertAlmostEqual(predictions["a1"]["scores"]["selected_risk_probability"], 0.80, places=6)

    def test_search_risk_threshold_prefers_constraint_satisfying_tradeoff(self):
        action_sets = [
            ActionSet(
                example_id="a1",
                question="Which token is correct?",
                answerable=True,
                target_action_index=0,
                candidates=(
                    ActionCandidate(
                        example_id="a1",
                        answer_text="beta",
                        span_score=9.0,
                        score_gap_to_best=0.0,
                        score_margin_to_next=2.0,
                        keep_probability=0.9,
                        support_probability=0.8,
                        abstain_probability=0.1,
                        stage6_controller_probability=0.7,
                        answer_length_tokens=1.0,
                        normalized_rank=0.0,
                        question_overlap=0.0,
                        label=1.0,
                        hard_negative_weight=1.0,
                        model_features=(0.0,) * 12,
                    ),
                ),
            ),
            ActionSet(
                example_id="u1",
                question="What city is mentioned?",
                answerable=False,
                target_action_index=1,
                candidates=(
                    ActionCandidate(
                        example_id="u1",
                        answer_text="paris",
                        span_score=8.5,
                        score_gap_to_best=0.0,
                        score_margin_to_next=1.0,
                        keep_probability=0.8,
                        support_probability=0.4,
                        abstain_probability=0.2,
                        stage6_controller_probability=0.5,
                        answer_length_tokens=1.0,
                        normalized_rank=0.0,
                        question_overlap=0.0,
                        label=0.0,
                        hard_negative_weight=2.0,
                        model_features=(0.0,) * 12,
                    ),
                ),
            ),
        ]
        outputs = [
            {
                "candidate_action_probabilities": [0.80],
                "abstain_probability": 0.20,
                "candidate_utility_logits": [1.5],
                "candidate_risk_logits": [-1.4],
                "candidate_risk_probabilities": [0.20],
                "candidate_action_scores": [0.80],
                "abstain_logit": 0.10,
            },
            {
                "candidate_action_probabilities": [0.75],
                "abstain_probability": 0.25,
                "candidate_utility_logits": [1.2],
                "candidate_risk_logits": [0.40],
                "candidate_risk_probabilities": [0.60],
                "candidate_action_scores": [0.70],
                "abstain_logit": 0.10,
            },
        ]
        references = {
            "a1": {"answers": ["beta"], "is_answerable": True},
            "u1": {"answers": [], "is_answerable": False},
        }

        threshold, metrics, mix, overabstain, sweep = search_risk_threshold(
            action_sets,
            outputs,
            references,
            threshold_min=0.10,
            threshold_max=0.80,
            threshold_step=0.30,
            match_f1_threshold=0.5,
            max_unsupported_answer_rate=0.0,
            max_overabstain_rate=0.0,
        )

        self.assertTrue(any(entry["constraint_satisfied"] for entry in sweep))
        self.assertAlmostEqual(threshold, 0.40, places=6)
        self.assertEqual(metrics["unsupported_answer_rate"], 0.0)
        self.assertEqual(overabstain["overabstain_rate"], 0.0)
        self.assertEqual(mix["supported_answers_count"], 1.0)

    def test_compute_overabstain_stats_counts_answerable_abstentions(self):
        predictions = {
            "a1": {"decision": "abstain", "answer": ""},
            "u1": {"decision": "abstain", "answer": ""},
        }
        references = {
            "a1": {"answers": ["beta"], "is_answerable": True},
            "u1": {"answers": [], "is_answerable": False},
        }

        stats = compute_overabstain_stats(predictions, references)

        self.assertEqual(stats["overabstain_count"], 1.0)
        self.assertEqual(stats["overabstain_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
