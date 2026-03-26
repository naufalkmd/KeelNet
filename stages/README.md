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

Current execution priority:

- `02-evidence-support-verification`
- `04-unsupported-confidence-control`
- `06-adaptive-constraint-balancing`
- `05-retrieval-grounded-qa` after the controlled proof path is stable

Current implementation status:

- Stage `01` has the completed supporting implementation and reference results
- Stage `02` now has a supporting implementation under `src/keelnet`, but still needs a validated full run before it is considered complete
- Stages `03` to `06` have teammate notebooks and a shared workflow, but their full stage-specific Python code paths still need to be completed during implementation

Current active stage files:

- `results-template.md`
- `notebooks/google-colab.ipynb`

Other stages now also include a `notebooks/google-colab.ipynb` template so teammates can start each stage from the same Colab workflow.

Shared note template:

- `stage-note-template.md`

Use the shared template for lightweight stage planning and run notes in Stages `02` to `06`. Keep notes short and update them before implementation, after smoke test, and after any run worth sharing.

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
