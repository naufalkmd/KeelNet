# KeelNet Stage Layout

This folder separates the roadmap into one working area per stage.

Keep each stage folder minimal.

Default rule:

- keep exactly one teammate-facing notebook template per stage at `notebooks/google-colab.ipynb`
- add result files only when they are actually used
- keep experiment artifacts in Google Drive, not inside the repo

Avoid creating extra placeholder folders such as `notes/` or `outputs/` when they are not being used.

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

Only Stage `01` currently has a full supporting implementation under `src/keelnet`.

Stages `02` to `06` already have teammate notebook templates, but the stage-specific Python code and commands for those stages still need to be defined during implementation.

Current active stage files:

- `results-template.md`
- `notebooks/google-colab.ipynb`

Other stages now also include a `notebooks/google-colab.ipynb` template so teammates can start each stage from the same Colab workflow.

## Collaboration Branching

Use Git branches for collaboration so each teammate can work in parallel without stepping on `main`.

Recommended shared branches:

- `collab/team-sync`
  - shared integration branch for teammate merges before promoting changes to `main`
- `stage/01-grounded-abstention-baseline`
- `stage/02-evidence-support-verification`
- `stage/03-confidence-calibration`
- `stage/04-unsupported-confidence-control`
- `stage/05-retrieval-grounded-qa`
- `stage/06-adaptive-constraint-balancing`

Suggested workflow:

- branch from `main` or the relevant `stage/*` branch
- open pull requests into the matching `stage/*` branch while the stage is active
- merge validated stage work into `collab/team-sync`
- merge stable work from `collab/team-sync` into `main`
