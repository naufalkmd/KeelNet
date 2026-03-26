import unittest

from keelnet.control import ControlConfig, apply_fixed_controller, search_control_config


class ControlRuleTests(unittest.TestCase):
    def test_apply_fixed_controller_routes_answers_and_abstains(self):
        predictions = {
            "a": {"decision": "answer", "answer": "alpha", "scores": {}},
            "b": {"decision": "answer", "answer": "beta", "scores": {}},
            "c": {"decision": "answer", "answer": "gamma", "scores": {}},
            "d": {"decision": "answer", "answer": "delta", "scores": {}},
            "e": {"decision": "abstain", "answer": "", "scores": {}},
        }
        controller_scores = {
            "a": {"qa_confidence": 0.80, "support_confidence": 0.90},
            "b": {"qa_confidence": 0.85, "support_confidence": 0.45},
            "c": {"qa_confidence": 0.45, "support_confidence": 0.90},
            "d": {"qa_confidence": 0.62, "support_confidence": 0.62},
            "e": {"qa_confidence": 0.95, "support_confidence": 0.95},
        }

        controlled, summary = apply_fixed_controller(
            predictions,
            controller_scores,
            config=ControlConfig(
                support_threshold=0.60,
                qa_threshold=0.55,
                joint_threshold=0.65,
                alpha=0.70,
            ),
        )

        self.assertEqual(controlled["a"]["decision"], "answer")
        self.assertEqual(controlled["b"]["abstain_reason"], "support_gate")
        self.assertEqual(controlled["c"]["abstain_reason"], "qa_gate")
        self.assertEqual(controlled["d"]["abstain_reason"], "joint_gate")
        self.assertEqual(controlled["e"]["abstain_reason"], "qa_model")
        self.assertEqual(summary["answer_count"], 1)

    def test_search_control_config_prefers_constraint_satisfying_rule(self):
        predictions = {
            "supported": {"decision": "answer", "answer": "Mercury", "scores": {}},
            "unsupported": {"decision": "answer", "answer": "Venus", "scores": {}},
        }
        references = {
            "supported": {"is_answerable": True, "answers": ["Mercury"]},
            "unsupported": {"is_answerable": False, "answers": []},
        }
        controller_scores = {
            "supported": {"qa_confidence": 0.80, "support_confidence": 0.80},
            "unsupported": {"qa_confidence": 0.90, "support_confidence": 0.45},
        }

        best_config, best_entry, _, sweep = search_control_config(
            predictions,
            references,
            controller_scores,
            max_unsupported_answer_rate=0.0,
            support_threshold_min=0.40,
            support_threshold_max=0.60,
            support_threshold_step=0.20,
            qa_threshold_min=0.40,
            qa_threshold_max=0.40,
            qa_threshold_step=0.10,
            joint_threshold_min=0.40,
            joint_threshold_max=0.40,
            joint_threshold_step=0.10,
            alpha_min=0.50,
            alpha_max=0.50,
            alpha_step=0.10,
            match_f1_threshold=0.5,
        )

        self.assertTrue(any(entry["constraint_satisfied"] for entry in sweep))
        self.assertEqual(best_entry["unsupported_answer_rate"], 0.0)
        self.assertGreaterEqual(best_config.support_threshold, 0.60)


if __name__ == "__main__":
    unittest.main()
