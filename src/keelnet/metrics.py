"""Pure-Python metrics for the Stage 1, Stage 2, and Stage 3 experiments."""

from __future__ import annotations

import math
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


def _clip_probability(value: float, *, eps: float = 1e-6) -> float:
    return min(1.0 - eps, max(eps, float(value)))


def brier_score(probabilities: list[float], labels: list[float]) -> float:
    if not probabilities:
        return 0.0
    total = 0.0
    for probability, label in zip(probabilities, labels, strict=False):
        total += (_clip_probability(probability) - float(label)) ** 2
    return total / len(probabilities)


def build_reliability_bins(
    probabilities: list[float],
    labels: list[float],
    *,
    num_bins: int = 10,
) -> list[dict[str, float]]:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    bins: list[dict[str, float]] = []
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        in_bin: list[int] = []
        for index, probability in enumerate(probabilities):
            clipped = _clip_probability(probability)
            if bin_index == num_bins - 1:
                matches = lower <= clipped <= upper
            else:
                matches = lower <= clipped < upper
            if matches:
                in_bin.append(index)

        if not in_bin:
            bins.append(
                {
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0.0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                }
            )
            continue

        count = float(len(in_bin))
        mean_confidence = sum(_clip_probability(probabilities[index]) for index in in_bin) / count
        accuracy = sum(float(labels[index]) for index in in_bin) / count
        bins.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return bins


def build_adaptive_reliability_bins(
    probabilities: list[float],
    labels: list[float],
    *,
    num_bins: int = 10,
) -> list[dict[str, float]]:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    if not probabilities:
        return []

    clipped_probabilities = [_clip_probability(probability) for probability in probabilities]
    sorted_pairs = sorted(
        zip(clipped_probabilities, labels, strict=False),
        key=lambda item: item[0],
    )
    adaptive_bin_count = min(num_bins, len(sorted_pairs))
    bins: list[dict[str, float]] = []
    for bin_index in range(adaptive_bin_count):
        start = (bin_index * len(sorted_pairs)) // adaptive_bin_count
        end = ((bin_index + 1) * len(sorted_pairs)) // adaptive_bin_count
        bucket = sorted_pairs[start:end]
        count = float(len(bucket))
        mean_confidence = sum(probability for probability, _ in bucket) / count
        accuracy = sum(float(label) for _, label in bucket) / count
        bins.append(
            {
                "bin_lower": bucket[0][0],
                "bin_upper": bucket[-1][0],
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return bins


def expected_calibration_error(
    probabilities: list[float],
    labels: list[float],
    *,
    num_bins: int = 10,
) -> float:
    if not probabilities:
        return 0.0

    bins = build_reliability_bins(probabilities, labels, num_bins=num_bins)
    total_count = float(len(probabilities))
    error = 0.0
    for bin_stats in bins:
        count = bin_stats["count"]
        if count == 0.0:
            continue
        error += (count / total_count) * abs(bin_stats["accuracy"] - bin_stats["mean_confidence"])
    return error


def adaptive_expected_calibration_error(
    probabilities: list[float],
    labels: list[float],
    *,
    num_bins: int = 10,
) -> float:
    if not probabilities:
        return 0.0

    bins = build_adaptive_reliability_bins(probabilities, labels, num_bins=num_bins)
    total_count = float(len(probabilities))
    error = 0.0
    for bin_stats in bins:
        count = bin_stats["count"]
        if count == 0.0:
            continue
        error += (count / total_count) * abs(bin_stats["accuracy"] - bin_stats["mean_confidence"])
    return error


def maximum_calibration_error(
    probabilities: list[float],
    labels: list[float],
    *,
    num_bins: int = 10,
) -> float:
    if not probabilities:
        return 0.0

    bins = build_reliability_bins(probabilities, labels, num_bins=num_bins)
    return max(
        (
            abs(bin_stats["accuracy"] - bin_stats["mean_confidence"])
            for bin_stats in bins
            if bin_stats["count"] > 0.0
        ),
        default=0.0,
    )


def pearson_correlation(probabilities: list[float], labels: list[float]) -> float:
    if not probabilities:
        return 0.0

    count = float(len(probabilities))
    mean_probability = sum(_clip_probability(probability) for probability in probabilities) / count
    mean_label = sum(float(label) for label in labels) / count

    numerator = 0.0
    probability_variance = 0.0
    label_variance = 0.0
    for probability, label in zip(probabilities, labels, strict=False):
        centered_probability = _clip_probability(probability) - mean_probability
        centered_label = float(label) - mean_label
        numerator += centered_probability * centered_label
        probability_variance += centered_probability**2
        label_variance += centered_label**2

    if probability_variance == 0.0 or label_variance == 0.0:
        return 0.0
    return numerator / math.sqrt(probability_variance * label_variance)


def threshold_calibration_gap(
    probabilities: list[float],
    labels: list[float],
    *,
    thresholds: list[float] | None = None,
    min_count: int = 25,
) -> float:
    if not probabilities:
        return 0.0

    if thresholds is None:
        thresholds = [index / 20.0 for index in range(10, 20)]

    gaps: list[float] = []
    for threshold in thresholds:
        selected_labels = [
            float(label)
            for probability, label in zip(probabilities, labels, strict=False)
            if _clip_probability(probability) >= threshold
        ]
        if len(selected_labels) < min_count:
            continue
        empirical_accuracy = sum(selected_labels) / len(selected_labels)
        gaps.append(abs(empirical_accuracy - float(threshold)))

    if not gaps:
        return 0.0
    return sum(gaps) / len(gaps)
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
