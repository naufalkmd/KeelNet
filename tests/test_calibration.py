from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.calibration import (
    fit_temperature_scaler,
    logit_probabilities,
    sigmoid_scores,
    summarize_binary_calibration,
)


class CalibrationTest(unittest.TestCase):
    def test_sigmoid_scores_invert_logit_probabilities(self):
        probabilities = [0.2, 0.8]
        recovered = sigmoid_scores(logit_probabilities(probabilities))
        self.assertAlmostEqual(float(recovered[0]), probabilities[0], places=6)
        self.assertAlmostEqual(float(recovered[1]), probabilities[1], places=6)

    def test_fit_temperature_scaler_softens_overconfident_scores(self):
        scores = [4.0, -4.0, 4.0, -4.0]
        labels = [1.0, 1.0, 0.0, 0.0]
        temperature = fit_temperature_scaler(
            scores,
            labels,
            temperature_min=0.5,
            temperature_max=5.0,
            temperature_step=0.5,
            num_bins=5,
        )
        self.assertGreater(temperature, 1.0)

    def test_summarize_binary_calibration_outputs_expected_keys(self):
        summary = summarize_binary_calibration(
            [0.9, 0.8, 0.2, 0.1],
            [1.0, 1.0, 0.0, 0.0],
            num_bins=2,
            threshold_gap_min_count=1,
        )
        self.assertEqual(summary["sample_count"], 4)
        self.assertAlmostEqual(summary["accuracy"], 1.0, places=6)
        self.assertAlmostEqual(summary["positive_rate"], 0.5, places=6)
        self.assertAlmostEqual(summary["mean_confidence"], 0.5, places=6)
        self.assertAlmostEqual(summary["adaptive_ece"], 0.15, places=6)
        self.assertAlmostEqual(summary["mce"], 0.15, places=6)
        self.assertEqual(len(summary["reliability_bins"]), 2)
        self.assertEqual(len(summary["adaptive_reliability_bins"]), 2)


if __name__ == "__main__":
    unittest.main()
