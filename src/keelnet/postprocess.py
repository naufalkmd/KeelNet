"""Prediction post-processing for extractive QA outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def postprocess_qa_predictions(
    examples: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    predictions: tuple[np.ndarray, np.ndarray],
    *,
    allow_abstain: bool,
    n_best_size: int,
    max_answer_length: int,
    null_score_diff_threshold: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Convert start/end logits into answer or abstain decisions."""

    all_start_logits, all_end_logits = predictions
    features_per_example: dict[str, list[int]] = defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_per_example[str(feature["example_id"])].append(feature_index)

    final_predictions: dict[str, dict[str, Any]] = {}
    for example in examples:
        example_id = str(example["id"])
        context = example["context"]
        feature_indices = features_per_example[example_id]

        best_span_text = ""
        best_span_score = float("-inf")
        best_null_score = float("-inf")

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            cls_index = int(features[feature_index].get("cls_index", 0))

            best_null_score = max(best_null_score, float(start_logits[cls_index] + end_logits[cls_index]))

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    if start_char is None or end_char is None:
                        continue

                    score = float(start_logits[start_index] + end_logits[end_index])
                    if score > best_span_score:
                        best_span_score = score
                        best_span_text = context[start_char:end_char]

        if allow_abstain and best_null_score - best_span_score > null_score_diff_threshold:
            final_predictions[example_id] = {
                "decision": "abstain",
                "answer": "",
                "scores": {
                    "best_span": best_span_score,
                    "null": best_null_score,
                    "margin": best_null_score - best_span_score,
                },
            }
            continue

        final_predictions[example_id] = {
            "decision": "answer",
            "answer": best_span_text,
            "scores": {
                "best_span": best_span_score,
                "null": best_null_score,
                "margin": best_null_score - best_span_score,
            },
        }

    return final_predictions
