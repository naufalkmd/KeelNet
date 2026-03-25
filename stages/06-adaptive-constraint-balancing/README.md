# KeelNet Stage 6: Adaptive Constraint Balancing

## Goal

Learn or control the trade-off among answer quality, support, abstention, and confidence more intelligently than fixed weights alone.

## Proof Status

This stage is not proof that hallucination is solved.

It is the full-pipeline validation stage:

- does the complete system beat simpler fixed baselines?
- does it improve the trade-off without hiding behind over-abstention or threshold tricks?

## Scope

- input: signals from the earlier stages
- output: adaptive balancing or decision control

## Main Change

- replace purely fixed balancing with an adaptive mechanism

## Main Metrics

- trade-off curve quality
- comparison against fixed-weight baselines
- robustness across operating points

## What This Stage Validates

- the complete KeelNet pipeline can manage the main constraints better than fixed balancing
- adaptive control adds real value beyond manual tuning

## What This Stage Does Not Validate

- that the method generalizes to arbitrary open-ended generation
- that hallucination is solved outside this task and evaluation setup

## Notebook Policy

Use the same VS Code plus Colab kernel workflow as Stage 1.

Only add a notebook file when this stage becomes active.

## Handoff Condition

This stage is only justified if the adaptive method clearly beats simpler fixed-weight or threshold baselines, and the gain is not just threshold gaming or over-abstention.
