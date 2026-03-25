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

Use Google Colab as the default notebook kernel for this stage.

Place notebooks in:

- `notebooks/`

## Handoff Condition

Do not move to the next stage until unsupported confident answers go down without answer quality collapsing.
