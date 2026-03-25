# KeelNet Stage Layout

This folder separates the roadmap into one working area per stage.

Keep each stage folder minimal.

Default rule:

- keep the stage `README.md`
- add a notebook only when that stage becomes active
- add result files only when they are actually used
- keep experiment artifacts in Google Drive, not inside the repo

Avoid creating empty placeholder folders such as `notes/`, `outputs/`, or `notebooks/` for inactive stages.

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

Current active stage files:

- `README.md`
- `results-template.md`
- `notebooks/google-colab.ipynb`

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
