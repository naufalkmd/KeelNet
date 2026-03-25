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

Use Google Colab as the default notebook kernel for this stage.

Place notebooks in:

- `notebooks/`

## Handoff Condition

Do not move to the next stage until support predictions are stable enough to be useful for training or analysis.
