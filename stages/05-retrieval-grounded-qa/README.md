# KeelNet Stage 5: Retrieval-Grounded QA

## Goal

Move from fixed evidence to retrieved evidence without losing grounded answer / abstain behavior.

## Proof Status

This stage is not the proof of the core idea.

It is the realism test:

- do the Stage 1-4 gains survive when evidence must be retrieved first?
- can retrieval failures be separated from answer-generation failures?

## Scope

- input: question
- intermediate step: retrieve top-k evidence
- output: answer or `ABSTAIN`

## Main Change

- add retrieval as a first-class part of the pipeline

## Main Metrics

- retrieval recall at k
- answer `F1`
- unsupported-answer rate after retrieval
- abstain quality

## What This Stage Validates

- the earlier grounded behavior still works when evidence is not handed to the model directly
- the pipeline is honest enough to attribute failure to retrieval vs answering

## What This Stage Does Not Validate

- that the core mechanism was already proven right
- that the system works outside retrieval-grounded QA

## Notebook Policy

Use the same VS Code plus Colab kernel workflow as Stage 1.

Only add a notebook file when this stage becomes active.

## Handoff Condition

Do not move to the next stage until retrieval failures and answer-generation failures can be separated cleanly in evaluation, and the end-to-end behavior keeps a meaningful share of the Stage 4 gains.
