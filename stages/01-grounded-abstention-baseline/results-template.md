# KeelNet Stage 1 Results Template

Fill this in only after both runs are complete.

## Result Table

```text
| Model | Answerable EM | Answerable F1 | Unsupported Rate | Abstain Precision | Abstain Recall | Abstain F1 |
|-------|---------------|---------------|------------------|-------------------|----------------|------------|
| Run A |               |               |                  |                   |                |            |
| Run B |               |               |                  |                   |                |            |
```

## Threshold Notes

- Run A threshold: not applicable
- Run B validation-selected threshold:
- Add the dev threshold-sweep figure and note whether the selected threshold sits on a sensible trade-off point.

## Quick Verdict

- Did unsupported-answer rate drop?
- Did answerable F1 stay acceptable?
- Did the model over-abstain?

## Error Analysis Tags

- unsupported but confident answer
- answerable but over-abstained
- wrong span despite supporting evidence
- ambiguous question
- context truncation problem
- thresholding problem
