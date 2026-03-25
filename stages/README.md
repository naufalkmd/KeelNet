# KeelNet Stage Layout

This folder separates the roadmap into one working area per stage.

Each stage folder follows the same shape:

- `README.md`
  - stage objective, inputs, outputs, metrics, and stop condition
- `notebooks/`
  - Google Colab notebooks or notebook placeholders
- `notes/`
  - experiment notes, failure analysis, and design decisions
- `outputs/`
  - run artifacts, metrics dumps, and checkpoints

Stages:

1. `01-grounded-abstention-baseline`
2. `02-evidence-support-verification`
3. `03-confidence-calibration`
4. `04-unsupported-confidence-control`
5. `05-retrieval-grounded-qa`
6. `06-adaptive-constraint-balancing`

Interpretation:

- Stages `01` to `04` are the core proof path for the mechanism in a controlled setting.
- Stage `05` checks whether the mechanism survives realistic retrieval noise.
- Stage `06` checks whether the full pipeline beats simpler fixed balancing baselines.

The current active implementation stage is:

- `01-grounded-abstention-baseline`
