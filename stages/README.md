# KeelNet Stage Layout

This folder separates the roadmap into one working area per stage.

Keep each stage folder minimal.

Default rule:

- keep exactly one teammate-facing notebook template per stage at
  `notebooks/stage-XX-<stage-name>-colab.ipynb`
- treat that repo notebook as the clean template, not the long-term executed archive
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
7. `07-risk-budgeted-action-learning`
8. `08-joint-optimization`

Interpretation:

- Stages `01` to `04` are the core proof path for the mechanism in a controlled setting.
- Stage `05` now hosts the support-constrained learning comparison, even though the folder name still reflects the older retrieval plan.
- Retrieval is now deferred as a later realism extension after the core proof path is stable.
- Stage `06` is the optional adaptive extension after the strongest fixed or constrained baseline is already clear.
- Stage `07` is the explicit utility-versus-risk action-learning extension if Stage `06` still behaves like a weak conservatizing wrapper.
- Stage `08` is the hybrid wrap-up stage that keeps Stage `05` as the answer engine and adds calibrated control back on top.

Current execution priority:

- `03-confidence-calibration`
- `04-unsupported-confidence-control`
- `05-retrieval-grounded-qa` as the support-constrained learning comparison after a strong Stage 4 baseline
- `06-adaptive-constraint-balancing` only if Stage 5 still leaves meaningful headroom
- `07-risk-budgeted-action-learning` only if Stage 6 still needs a stronger decision formulation
- `08-joint-optimization` only if the final answer is likely to be a hybrid instead of a pure modular or pure direct-learning system
- retrieval realism later, after the core Stage 4 versus Stage 5 comparison is stable

Current implementation status:

- Stage `01` has the completed supporting implementation and reference results
- Stage `02` now has the supporting implementation plus a completed reference run, which shows a learnable verifier signal but only a modest end-to-end gain
- Stages `03` to `08` now have teammate notebooks and runnable stage-specific Python paths in the repo

Current findings snapshot:

- Stage `01`: strong abstention gain, large answer-quality cost
- Stage `02`: support is learnable, but the current support gate is too permissive to change end-to-end behavior much
- next bottleneck: calibration and fixed control, not retrieval yet

Current active stage files:

- `notebooks/stage-XX-<stage-name>-colab.ipynb`

Examples:

- `notebooks/stage-01-grounded-abstention-baseline-colab.ipynb`
- `notebooks/stage-02-evidence-support-verification-colab.ipynb`
- `notebooks/stage-02-5-hard-negative-support-verification-colab.ipynb`
- `notebooks/stage-05-support-constrained-learning-colab.ipynb`
- `notebooks/stage-08-2-action-learner-calibrated-support-colab.ipynb`

Use the stage-specific `stage-XX-...-colab.ipynb` files as the canonical
notebook names for each stage.
For follow-up variants, keep the same pattern and insert the substage number,
for example `stage-02-5-...-colab.ipynb`.
If you want to preserve outputs from a meaningful run, export the executed
notebook into that run's artifact folder under `OUTPUT_ROOT/executed-notebook/`
instead of overwriting the canonical template file in the repo.

Shared note template:

- `stage-note-template.md`

Use the shared template for lightweight stage planning and run notes in Stages `02` to `06`. Keep notes short and update them before implementation, after smoke test, and after any run worth sharing.
Use the same pattern for Stage `07`.

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
- `stage/07-risk-budgeted-action-learning`
- `stage/08-joint-optimization`

Suggested workflow:

- branch from `main` or the relevant `stage/*` branch
- open pull requests into the matching `stage/*` branch while the stage is active
- merge validated stage work into `collab/team-sync`
- merge stable work from `collab/team-sync` into `main`
