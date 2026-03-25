# KeelNet Stage 3: Confidence Calibration

## Goal

Make the model's confidence scores reflect actual correctness and support strength instead of acting like raw uncalibrated logits.

## Scope

- input: Stage 1 or Stage 2 predictions
- output: calibrated confidence for answer / abstain decisions, and optionally support predictions

## Main Change

- expose confidence scores
- evaluate and, if needed, train for calibration

## Main Metrics

- `ECE`
- `Brier Score`
- correlation between confidence and supported correctness

## Notebook Policy

Use the same VS Code plus Colab kernel workflow as Stage 1.

Only add a notebook file when this stage becomes active.

## Handoff Condition

Do not move to the next stage until high-confidence predictions are meaningfully more reliable than low-confidence ones.
