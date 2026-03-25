import unittest

from keelnet.metrics import compute_stage1_metrics, exact_match_score, f1_score, normalize_answer


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
        self.assertAlmostEqual(metrics["unsupported_answer_rate"], 50.0)
        self.assertAlmostEqual(metrics["abstain_precision"], 50.0)
        self.assertAlmostEqual(metrics["abstain_recall"], 50.0)
        self.assertAlmostEqual(metrics["abstain_f1"], 50.0)


if __name__ == "__main__":
    unittest.main()
