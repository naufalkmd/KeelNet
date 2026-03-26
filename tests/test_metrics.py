from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.metrics import (
    adaptive_expected_calibration_error,
    build_adaptive_reliability_bins,
    build_reliability_bins,
    brier_score,
    compute_stage1_metrics,
    compute_stage2_support_metrics,
    expected_calibration_error,
    exact_match_score,
    f1_score,
    maximum_calibration_error,
    normalize_answer,
    pearson_correlation,
    threshold_calibration_gap,
)


class MetricsTest(unittest.TestCase):
    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("The, Quick Brown Fox!"), "quick brown fox")

    def test_exact_match_score(self):
        self.assertEqual(exact_match_score("The Eiffel Tower", "eiffel tower"), 1.0)
        self.assertEqual(exact_match_score("Paris", "Lyon"), 0.0)

    def test_f1_score(self):
        self.assertAlmostEqual(f1_score("Alexander Fleming", "Fleming"), 2 / 3)
        self.assertEqual(f1_score("", ""), 1.0)
        self.assertEqual(f1_score("penicillin", ""), 0.0)

    def test_compute_stage1_metrics(self):
        predictions = {
            "a1": {"decision": "answer", "answer": "Alexander Fleming"},
            "a2": {"decision": "abstain", "answer": ""},
            "u1": {"decision": "abstain", "answer": ""},
            "u2": {"decision": "answer", "answer": "made up"},
        }
        references = {
            "a1": {"answers": ["Alexander Fleming"], "is_answerable": True},
            "a2": {"answers": ["Penicillin"], "is_answerable": True},
            "u1": {"answers": [], "is_answerable": False},
            "u2": {"answers": [], "is_answerable": False},
        }

        metrics = compute_stage1_metrics(predictions, references)

        self.assertEqual(metrics["answerable_count"], 2.0)
        self.assertEqual(metrics["unanswerable_count"], 2.0)
        self.assertAlmostEqual(metrics["answerable_em"], 50.0)
        self.assertAlmostEqual(metrics["answerable_f1"], 50.0)
        self.assertAlmostEqual(metrics["overall_em"], 50.0)
        self.assertAlmostEqual(metrics["overall_f1"], 50.0)
        self.assertAlmostEqual(metrics["unsupported_answer_rate"], 50.0)
        self.assertAlmostEqual(metrics["abstain_precision"], 50.0)
        self.assertAlmostEqual(metrics["abstain_recall"], 50.0)
        self.assertAlmostEqual(metrics["abstain_f1"], 50.0)

    def test_compute_stage2_support_metrics(self):
        predictions = {
            "a1": {
                "decision": "answer",
                "answer": "Alexander Fleming",
                "support": {"score": 0.9},
            },
            "a2": {
                "decision": "answer",
                "answer": "made up",
                "support": {"score": 0.8},
            },
            "u1": {
                "decision": "answer",
                "answer": "confident guess",
                "support": {"score": 0.7},
            },
            "u2": {
                "decision": "abstain",
                "answer": "",
                "support": {"score": 0.1},
            },
        }
        references = {
            "a1": {"answers": ["Alexander Fleming"], "is_answerable": True},
            "a2": {"answers": ["Penicillin"], "is_answerable": True},
            "u1": {"answers": [], "is_answerable": False},
            "u2": {"answers": [], "is_answerable": False},
        }

        metrics = compute_stage2_support_metrics(
            predictions,
            references,
            support_threshold=0.5,
            support_match_f1_threshold=0.5,
        )

        self.assertEqual(metrics["answered_count"], 3.0)
        self.assertEqual(metrics["gold_supported_count"], 1.0)
        self.assertAlmostEqual(metrics["support_accuracy"], 100.0 / 3.0, places=4)
        self.assertAlmostEqual(metrics["support_precision"], 100.0 / 3.0, places=4)
        self.assertAlmostEqual(metrics["support_recall"], 100.0, places=4)
        self.assertAlmostEqual(metrics["support_f1"], 50.0, places=4)
        self.assertAlmostEqual(metrics["supported_answer_rate"], 100.0 / 3.0, places=4)
        self.assertAlmostEqual(metrics["predicted_supported_rate"], 100.0, places=4)
        self.assertAlmostEqual(metrics["contradiction_rate"], 200.0 / 3.0, places=4)

    def test_calibration_metrics(self):
        probabilities = [0.9, 0.8, 0.2, 0.1]
        labels = [1.0, 1.0, 0.0, 0.0]

        bins = build_reliability_bins(probabilities, labels, num_bins=2)
        adaptive_bins = build_adaptive_reliability_bins(probabilities, labels, num_bins=2)
        self.assertEqual(len(bins), 2)
        self.assertEqual(len(adaptive_bins), 2)
        self.assertEqual(bins[0]["count"], 2.0)
        self.assertEqual(bins[1]["count"], 2.0)
        self.assertAlmostEqual(bins[0]["mean_confidence"], 0.15, places=6)
        self.assertAlmostEqual(bins[1]["mean_confidence"], 0.85, places=6)
        self.assertAlmostEqual(bins[0]["accuracy"], 0.0, places=6)
        self.assertAlmostEqual(bins[1]["accuracy"], 1.0, places=6)
        self.assertAlmostEqual(adaptive_bins[0]["mean_confidence"], 0.15, places=6)
        self.assertAlmostEqual(adaptive_bins[1]["mean_confidence"], 0.85, places=6)

        self.assertAlmostEqual(brier_score(probabilities, labels), 0.025, places=6)
        self.assertAlmostEqual(
            expected_calibration_error(probabilities, labels, num_bins=2),
            0.15,
            places=6,
        )
        self.assertAlmostEqual(
            adaptive_expected_calibration_error(probabilities, labels, num_bins=2),
            0.15,
            places=6,
        )
        self.assertAlmostEqual(
            maximum_calibration_error(probabilities, labels, num_bins=2),
            0.15,
            places=6,
        )
        self.assertGreater(pearson_correlation(probabilities, labels), 0.95)

    def test_threshold_calibration_gap(self):
        probabilities = [0.95, 0.85, 0.75, 0.65, 0.55]
        labels = [1.0, 1.0, 1.0, 0.0, 0.0]
        gap = threshold_calibration_gap(
            probabilities,
            labels,
            thresholds=[0.5, 0.7, 0.9],
            min_count=1,
        )
        self.assertAlmostEqual(gap, (0.1 + 0.3 + 0.1) / 3.0, places=6)

if __name__ == "__main__":
    unittest.main()
