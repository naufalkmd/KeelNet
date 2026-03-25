# KeelNet Stage 2: Evidence Support Verification

## Goal

Add a support-verification signal on top of Stage 1 so the system can distinguish supported answers from unsupported ones more reliably.

## Scope

- input: question, evidence passage, predicted answer
- output: support label or support score
- keep Stage 1 answer / abstain behavior as the base system

## Main Change

- add one support-verification head or equivalent verification module

## Main Metrics

- support classification F1
- supported-answer rate
- contradiction rate if defined

## Notebook Policy

Use the same VS Code plus Colab kernel workflow as Stage 1.

Only add a notebook file when this stage becomes active.

## Handoff Condition

Do not move to the next stage until support predictions are stable enough to be useful for training or analysis.
