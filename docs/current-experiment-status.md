# KeelNet Current Experiment Status

Generated: 2026-03-27

## At A Glance

- The canonical completed experiment chain currently lives in Drive-backed artifacts, not just local runtime storage.
- Best completed late-stage result on disk: `Stage 4 Fixed Control` with `overall_f1 = 69.41` and `unsupported_answer_rate = 27.59`.
- No saved comparable late-stage run currently meets the `20%` unsupported-answer budget.
- `Stage 8 Hybrid` now has a saved eval, but it did not beat `Stage 4`: `overall_f1 = 68.27` and `unsupported_answer_rate = 31.86`.
- `Stage 8 Hybrid` still does not have `RUN_COMPLETED.txt` or `collab-share-note.md` saved.

## Current Frontier

| Label | Overall F1 | Answerable F1 | Unsupported answer rate | Abstain F1 |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 Baseline | 42.35 | 84.82 | 100.00 | 0.00 |
| Stage 1 Abstain | 69.40 | 66.76 | 27.97 | 72.98 |
| Stage 4 Fixed Control | 69.41 | 66.41 | 27.59 | 72.96 |
| Stage 5 Learner | 68.28 | 71.40 | 34.84 | 71.17 |
| Stage 6 Adaptive Balance | 67.62 | 69.70 | 34.45 | 71.12 |
| Stage 7 Action Learner | 67.66 | 69.95 | 34.62 | 71.11 |
| Stage 8 Hybrid | 68.27 | 68.39 | 31.86 | 71.71 |

## Stage Status

| Stage | Canonical run | Storage | Status | Key outputs |
| --- | --- | --- | --- | --- |
| Stage 1 | `codex-stage1-live-20260326-014652` | Drive | complete | `baseline_eval.json`, `abstain_eval.json`, `baseline`, `abstain` |
| Stage 2 | `naufal-stage2-v2` | Drive | complete | `verifier_eval.json`, `verifier` |
| Stage 2.5 | `naufal-stage2-5-v1` | local | complete | `verifier_eval.json`, `verifier` |
| Stage 3 | `naufal-stage3-v1` | Drive | complete | `calibration_eval.json`, `qa_dev_reliability.png`, `support_dev_reliability.png` |
| Stage 4 | `naufal-stage4-v1` | Drive | complete | `control_eval.json` |
| Stage 5 | `naufal-stage5-v1` | Drive | complete | `learner_eval.json`, `learner` |
| Stage 6 | `naufal-stage6-v1` | Drive | complete | `balance_eval.json`, `balancer` |
| Stage 7 | `naufal-stage7-v1` | Drive | complete | `risk_action_eval.json`, `risk-action-learner` |
| Stage 8 Hybrid | `naufal-stage8-v1` | Drive | complete | `hybrid_eval.json`, `hybrid-controller` |
| Stage 8.2 | — | — | missing | — |
| Final Comparison | `naufal-final-comparison-v1` | Drive | complete | `comparison_metrics.csv`, `comparison_summary.json` |

## Missing Right Now

- `/mnt/g/My Drive/KeelNet/artifacts/stage1_colab/naufal-stage1-v1` is `partial`.
- `/mnt/g/My Drive/KeelNet/artifacts/stage1_colab/yourname-stage1-v1` is `partial`.
- `/content/KeelNet-local/artifacts/stage2_colab/naufal-stage2-v1` is `metadata-only`.
- `/content/KeelNet-local/artifacts/stage2_colab/yourname-stage2-v1` is `metadata-only`.
- `/content/KeelNet-local/artifacts/stage4_colab/naufal-stage4-v1` is `partial`.
- `/content/KeelNet-local/artifacts/stage4_colab/yourname-stage4-v1` is `metadata-only`.
- `/content/KeelNet-local/artifacts/stage4_colab/yourname-stage4-v2` is `metadata-only`.

## Ready Inputs For A Clean Stage 8.2 Run

- Stage 4 control eval: `/mnt/g/My Drive/KeelNet/artifacts/stage4_colab/naufal-stage4-v1/control_eval.json`
- Stage 5 learner dir: `/mnt/g/My Drive/KeelNet/artifacts/stage5_colab/naufal-stage5-v1/learner`
- Stage 5 eval: `/mnt/g/My Drive/KeelNet/artifacts/stage5_colab/naufal-stage5-v1/learner_eval.json`
- Optional Stage 6 balancer: `/mnt/g/My Drive/KeelNet/artifacts/stage6_colab/naufal-stage6-v1/balancer`
- Stage 7 comparison eval: `/mnt/g/My Drive/KeelNet/artifacts/stage7_colab/naufal-stage7-v1/risk_action_eval.json`

## Refresh

```bash
python analysis/generate_experiment_docs.py
```

Full audit: [current-experiment-audit.md](./current-experiment-audit.md)

