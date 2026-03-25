# KeelNet Stage 4: Unsupported-Confidence Control

## Goal

Reduce confident unsupported answers without collapsing usefulness.

## Scope

- input: answer, support signal, confidence signal
- output: better control over unsupported confident answering

## Main Change

- add an explicit penalty or control mechanism for confident unsupported outputs

## Main Metrics

- unsupported-answer rate
- supported-answer rate
- answer `F1`
- abstain `F1`

## Notebook Policy

Use the same VS Code plus Colab kernel workflow as Stage 1.

Only add a notebook file when this stage becomes active.

## Handoff Condition

Do not move to the next stage until unsupported confident answers go down without answer quality collapsing.
