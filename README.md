# KeelNet

KeelNet is a staged research scaffold for evidence-grounded question answering with controlled abstention.

The current active implementation scope is Stage 1:

- dataset: `SQuAD 2.0`
- input: one question and one evidence passage
- output: answer span or `ABSTAIN`
- goal: reduce unsupported answers without collapsing answer quality

## Current Focus

Stage 1 tests one narrow claim:

> A model trained to abstain on unsupported questions should produce fewer unsupported answers than an answer-only baseline, without sharply degrading answer quality on answerable questions.

The detailed execution plan lives in:

- `stages/01-grounded-abstention-baseline/README.md`

The broader roadmap lives in:

- `stages/README.md`
- `docs/hallucination-project-proposal.md`

## Repository Layout

- `src/keelnet/`
  - training, evaluation, data preparation, postprocessing, and metrics code
- `tests/`
  - unit tests for the current implementation
- `stages/`
  - one working area per research stage
- `docs/`
  - project framing and supporting writeups
- `archive/`
  - older notes and exploratory material

## Collaboration

Use Git branches for collaboration instead of pushing work directly to `main`.

Shared branches in this repo:

- `main`
  - stable branch
- `collab/team-sync`
  - shared integration branch for teammate merges
- `stage/01-grounded-abstention-baseline`
- `stage/02-evidence-support-verification`
- `stage/03-confidence-calibration`
- `stage/04-unsupported-confidence-control`
- `stage/05-retrieval-grounded-qa`
- `stage/06-adaptive-constraint-balancing`

Suggested workflow:

1. Start from the relevant `stage/*` branch.
2. Create your own feature branch for a task or experiment.
3. Open pull requests into the matching `stage/*` branch.
4. Merge validated stage work into `collab/team-sync`.
5. Merge stable work from `collab/team-sync` into `main`.

## Running The Stage 1 Scaffold

The package currently exposes two CLI entry points:

- `keelnet-train`
- `keelnet-eval`

Install the project in your environment, then run those commands against the Stage 1 setup as the implementation fills out.
