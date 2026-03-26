# KeelNet Stage 1 Results

## Result Table

```text
| Model | Answerable EM | Answerable F1 | Unsupported Rate | Abstain Precision | Abstain Recall | Abstain F1 |
|-------|---------------|---------------|------------------|-------------------|----------------|------------|
| Run A (baseline) | 76.18 | 84.82 | 100.00 | 0.00 | 0.00 | 0.00 |
| Run B (abstain-aware) | 61.15 | 66.76 | 27.97 | 73.97 | 72.03 | 72.98 |
```

## Threshold Notes

- Run A threshold: not applicable
- Run B validation-selected threshold: `-2.5`
- The selected threshold sits in a stable middle region of the dev sweep. Nearby thresholds `-3.0` and `-2.0` give very similar overall F1, so the result does not look like a single brittle operating point.

## Quick Verdict

- Unsupported-answer rate dropped sharply, from `100.00` to `27.97`.
- Answerable F1 stayed usable but fell materially, from `84.82` to `66.76`.
- The model did not collapse into trivial refusal, but it still trades too much answer quality for safer abstention.

## Error Analysis Tags

- unsupported but confident answer
- answerable but over-abstained
- wrong span despite supporting evidence
- ambiguous question
- context truncation problem
- thresholding problem
