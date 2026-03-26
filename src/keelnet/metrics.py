"""Pure-Python metrics for the Stage 1 and Stage 2 experiments."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping
from typing import Any


def normalize_answer(text: str) -> str:
    """Apply the standard SQuAD normalization."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        punctuation = set(string.punctuation)
        return "".join(ch for ch in value if ch not in punctuation)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    num_same = 0
    for token in truth_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, truth) for truth in ground_truths)


def _percentage(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def is_supported_answer(
    answer_text: str,
    gold_answers: list[str],
    *,
    match_f1_threshold: float,
) -> bool:
    """Return whether an answered prediction is supported by the references."""

    normalized_answer = normalize_answer(answer_text)
    if not normalized_answer or not gold_answers:
        return False
    return metric_max_over_ground_truths(f1_score, answer_text, gold_answers) >= match_f1_threshold


def compute_stage1_metrics(
    predictions: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Compute Stage 1 metrics as percentages.

    The overall metrics are decision-aware: a correct abstention on an
    unanswerable example receives full credit instead of being forced to zero.
    """

    answerable_total = 0
    unanswerable_total = 0
    answerable_em_sum = 0.0
    answerable_f1_sum = 0.0
    overall_em_sum = 0.0
    overall_f1_sum = 0.0
    unsupported_answers = 0
    abstain_tp = 0
    abstain_fp = 0
    abstain_fn = 0

    for example_id, reference in references.items():
        prediction = predictions.get(
            example_id,
            {"decision": "abstain", "answer": "", "scores": {}},
        )
        decision = str(prediction.get("decision", "abstain")).lower()
        answer_text = str(prediction.get("answer", ""))
        predicted_abstain = decision == "abstain"
        gold_answers = list(reference.get("answers", []))
        gold_abstain = not bool(reference.get("is_answerable", False))

        em = 0.0
        f1 = 0.0
        if predicted_abstain and gold_abstain:
            em = 1.0
            f1 = 1.0
        elif not predicted_abstain and gold_answers:
            em = metric_max_over_ground_truths(exact_match_score, answer_text, gold_answers)
            f1 = metric_max_over_ground_truths(f1_score, answer_text, gold_answers)

        overall_em_sum += em
        overall_f1_sum += f1

        if gold_abstain:
            unanswerable_total += 1
            if not predicted_abstain:
                unsupported_answers += 1
        else:
            answerable_total += 1
            answerable_em_sum += em
            answerable_f1_sum += f1

        if predicted_abstain and gold_abstain:
            abstain_tp += 1
        elif predicted_abstain and not gold_abstain:
            abstain_fp += 1
        elif (not predicted_abstain) and gold_abstain:
            abstain_fn += 1

    abstain_precision = 0.0
    if abstain_tp + abstain_fp > 0:
        abstain_precision = abstain_tp / (abstain_tp + abstain_fp)

    abstain_recall = 0.0
    if abstain_tp + abstain_fn > 0:
        abstain_recall = abstain_tp / (abstain_tp + abstain_fn)

    abstain_f1 = 0.0
    if abstain_precision + abstain_recall > 0:
        abstain_f1 = 2 * abstain_precision * abstain_recall / (abstain_precision + abstain_recall)

    total_examples = len(references)
    return {
        "answerable_em": _percentage(answerable_em_sum, answerable_total),
        "answerable_f1": _percentage(answerable_f1_sum, answerable_total),
        "overall_em": _percentage(overall_em_sum, total_examples),
        "overall_f1": _percentage(overall_f1_sum, total_examples),
        "unsupported_answer_rate": _percentage(unsupported_answers, unanswerable_total),
        "abstain_precision": 100.0 * abstain_precision,
        "abstain_recall": 100.0 * abstain_recall,
        "abstain_f1": 100.0 * abstain_f1,
        "answerable_count": float(answerable_total),
        "unanswerable_count": float(unanswerable_total),
    }


def compute_stage2_support_metrics(
    predictions: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    *,
    support_threshold: float,
    support_match_f1_threshold: float,
) -> dict[str, float]:
    """Compute Stage 2 support-verification metrics over answered predictions."""

    answered_total = 0
    gold_supported_total = 0
    predicted_supported_total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for example_id, reference in references.items():
        prediction = predictions.get(example_id, {"decision": "abstain", "answer": "", "support": {}})
        decision = str(prediction.get("decision", "abstain")).lower()
        if decision != "answer":
            continue

        answered_total += 1
        answer_text = str(prediction.get("answer", ""))
        gold_answers = list(reference.get("answers", []))
        gold_supported = bool(reference.get("is_answerable", False)) and is_supported_answer(
            answer_text,
            gold_answers,
            match_f1_threshold=support_match_f1_threshold,
        )
        if gold_supported:
            gold_supported_total += 1

        support_info = prediction.get("support", {})
        support_score = support_info.get("score")
        if support_score is None:
            predicted_supported = str(support_info.get("label", "unsupported")).lower() == "supported"
        else:
            predicted_supported = float(support_score) >= support_threshold
        if predicted_supported:
            predicted_supported_total += 1

        if predicted_supported and gold_supported:
            true_positive += 1
        elif predicted_supported and not gold_supported:
            false_positive += 1
        elif (not predicted_supported) and gold_supported:
            false_negative += 1
        else:
            true_negative += 1

    accuracy = 0.0
    if answered_total > 0:
        accuracy = (true_positive + true_negative) / answered_total

    precision = 0.0
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)

    recall = 0.0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)

    support_f1 = 0.0
    if precision + recall > 0:
        support_f1 = 2 * precision * recall / (precision + recall)

    return {
        "support_accuracy": 100.0 * accuracy,
        "support_precision": 100.0 * precision,
        "support_recall": 100.0 * recall,
        "support_f1": 100.0 * support_f1,
        "supported_answer_rate": _percentage(gold_supported_total, answered_total),
        "predicted_supported_rate": _percentage(predicted_supported_total, answered_total),
        "contradiction_rate": _percentage(false_positive, answered_total),
        "answered_count": float(answered_total),
        "gold_supported_count": float(gold_supported_total),
    }
