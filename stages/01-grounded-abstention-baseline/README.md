# KeelNet Stage 1 Experiment Plan: Grounded Abstention Baseline

## Goal

Test whether explicit abstention training improves the trade-off between answer quality and unsupported answers on `SQuAD 2.0`.

Colab notebook:

- `notebooks/google-colab.ipynb`

Teammate setup and shared experiment workflow:

- `../../docs/experiment-guidelines.md`

This experiment does **not** try to solve hallucination in general.

It tests one narrow claim:

> A model trained to abstain on unsupported questions should produce fewer unsupported answers than an answer-only baseline, without collapsing answer quality on answerable questions.

## Exact Experimental Question

Given one question and one evidence passage:

- does explicit abstention training reduce unsupported answers on unanswerable examples?
- how much answer quality is lost, if any, on answerable examples?

## Fixed Decisions

Do not change these during Stage 1:

- dataset: `SQuAD 2.0` only
- input: one question and one passage
- output: answer span or `ABSTAIN`
- architecture family: one extractive QA backbone for both runs
- no retrieval
- no external verifier
- no custom scoring framework

## Recommended Implementation Choice

Use a standard **extractive question answering** setup, not a generative setup.

Why:

- `SQuAD 2.0` is already built for extractive QA with no-answer cases
- it gives you a clean answer-or-no-answer decision
- it keeps the first experiment simple and standard

Recommended default backbone:

- `distilbert-base-uncased`

Why this is a good default:

- small enough to train cheaply
- standard enough that results are easy to interpret

If you prefer `bert-base-uncased`, that is also fine.

What matters is:

- pick one backbone
- keep it fixed across all Stage 1 runs

## Experimental Runs

Run exactly two core models.

### Run A: Answer-Only Baseline

Train on:

- answerable `SQuAD 2.0` training examples only

Behavior:

- always predict an answer span
- no explicit abstention supervision

Purpose:

- establish what happens when the model is optimized only to answer

### Run B: Abstain-Aware Model

Train on:

- full `SQuAD 2.0` training data
- both answerable and unanswerable examples

Behavior:

- predict an answer span when supported
- predict `ABSTAIN` when the passage does not support an answer

Purpose:

- test whether explicit no-answer supervision improves grounded behavior

## Fair Comparison Rules

Keep these identical between Run A and Run B:

- backbone
- tokenizer
- preprocessing
- max sequence length
- stride
- optimizer
- learning rate
- batch size
- number of epochs
- random seed if possible

If these change, the comparison gets muddy.

## Data Split

Use `SQuAD 2.0` with three partitions:

### Training Split

- use the official train set
- optionally carve out a validation split from the train set for model selection

### Validation Split

Recommended:

- create a validation subset from the official train set
- keep the answerable / unanswerable mix roughly similar

Use validation for:

- early stopping
- null-threshold selection for the abstain-aware model

### Held-Out Evaluation Split

- use the official dev set for final comparison

For analysis, split the dev set into:

- answerable examples
- unanswerable examples

## Input And Output Format

### Input

```json
{
  "question": "Who discovered penicillin?",
  "context": "Alexander Fleming discovered penicillin in 1928. Penicillin became widely used in the 1940s."
}
```

### Output

Either:

```json
{
  "decision": "answer",
  "answer": "Alexander Fleming"
}
```

Or:

```json
{
  "decision": "abstain",
  "answer": null
}
```

## Training Setup

### Preprocessing

For each example:

- tokenize question and context
- apply truncation and stride if context is too long
- map answer spans to token positions for answerable cases
- mark no-answer cases for the abstain-aware model

Be careful with:

- offset mappings
- context truncation
- multiple gold answers in evaluation

### Objective For Run A

Use standard extractive QA loss:

- start-position loss
- end-position loss

No abstention target.

### Objective For Run B

Use standard extractive QA loss with no-answer handling:

- start-position loss
- end-position loss
- no-answer / null prediction through the standard SQuAD 2.0 setup

At inference time:

- predict `ABSTAIN` when the null answer is preferred over the best non-null span

Tune the null threshold on validation only.

## Evaluation Metrics

Do not report only one number.

### Primary Metric

- unsupported-answer rate on unanswerable dev examples

Definition:

```text
unsupported_answer_rate =
count(predicted non-abstain on unanswerable examples)
/ count(unanswerable examples)
```

This is the most important Stage 1 metric.

### Secondary Metrics

- answer `F1` on answerable dev examples
- answer `EM` on answerable dev examples
- abstain precision
- abstain recall
- abstain `F1`

### Optional Overall Metrics

- overall SQuAD `EM`
- overall SQuAD `F1`

These are useful, but they should not hide the answerable vs unanswerable breakdown.

## What Counts As A Win

The abstain-aware model wins if:

- unsupported-answer rate is lower than the answer-only baseline
- answer quality on answerable examples remains acceptable

The second condition must be decided before you run the final comparison.

Do not look at the results first and then redefine what "acceptable" means.

## What Counts As Failure

The abstain-aware model does **not** really help if:

- unsupported answers drop only because the model abstains too often on answerable questions
- answer `F1` collapses
- abstention behavior is unstable and depends heavily on threshold tricks

## Step-by-Step Procedure

1. Download `SQuAD 2.0`.
2. Build preprocessing for extractive QA.
3. Create a validation split from the official train set.
4. Implement metrics for:
   - answerable-only `EM/F1`
   - unanswerable-only unsupported-answer rate
   - abstain precision / recall / `F1`
5. Train Run A on answerable training examples only.
6. Evaluate Run A on the held-out dev set.
7. Train Run B on full training data.
8. Tune the null threshold for Run B on validation.
9. Evaluate Run B on the held-out dev set.
10. Compare the two models in one result table.
11. Manually inspect a sample of failures.

## Manual Error Analysis

Do not stop at metric tables.

Read at least:

- 20 answerable mistakes
- 20 unanswerable mistakes

Tag the failures into categories such as:

- unsupported but confident answer
- answerable but over-abstained
- wrong span despite supporting evidence
- ambiguous question
- context truncation problem
- thresholding problem

This matters because a small metric improvement can still hide bad behavior.

## Minimal Result Table

Use a table like this:

```text
| Model | Answerable EM | Answerable F1 | Unsupported Rate | Abstain Precision | Abstain Recall | Abstain F1 |
|-------|---------------|---------------|------------------|-------------------|----------------|------------|
| Run A |               |               |                  |                   |                |            |
| Run B |               |               |                  |                   |                |            |
```

## Minimal Deliverables

At the end of Stage 1, you should have:

- one fixed preprocessing pipeline
- one baseline training run
- one abstain-aware training run
- one evaluation script
- one comparison table
- one short error analysis note

If you do not have these, the experiment is not done.

## What Not To Add Yet

Do not add these during Stage 1:

- retrieval
- verification heads
- confidence calibration modules
- synthetic negatives
- multiple networks
- custom balancing mechanisms
- long-form generation

These may become later stages, but they will only make Stage 1 harder to interpret.

## Best Next Step After Stage 1

If Run B truly improves the trade-off, the next best step is:

- `Evidence Support Verification`

That is the clean next stage because it helps distinguish:

- unsupported answers
- supported answers
- over-abstention

Without jumping into retrieval too early.
