# KeelNet

KeelNet is a staged research project on evidence-grounded question answering
with controlled abstention.
The project studies a narrower and more operational version of hallucination:
when a system should answer, when it should abstain, and how to reduce
unsupported answers without collapsing usefulness.

## Project Status

The current implemented scope is Stage 1:

- dataset: `SQuAD 2.0`
- input: one question and one evidence passage
- output: extractive answer span or `ABSTAIN`
- current comparison: answer-only baseline versus abstain-aware training
- primary goal: reduce unsupported answers while preserving answer quality on
  answerable questions

Only Stage `01-grounded-abstention-baseline` is implemented end-to-end in
`src/keelnet` at the moment.
Stages `02` to `06` already have teammate notebooks, but their stage-specific
Python modules and commands are still part of the roadmap rather than completed
implementation.

## Current Research Claim

Stage 1 tests one narrow claim:

> A model trained to abstain on unsupported questions should produce fewer
> unsupported answers than an answer-only baseline, without sharply degrading
> answer quality on answerable questions.

This scope is intentionally narrow.
The goal is not to claim that hallucination is solved in general, but to
establish a controlled first result that later stages can extend.

## Repository Structure

- `src/keelnet/`
  - training, evaluation, data preparation, postprocessing, and metrics code
- `tests/`
  - unit tests for the active implementation
- `stages/`
  - one notebook-first working area per research stage
- `docs/`
  - project framing, workflow notes, and supporting documentation
- `paper/`
  - LaTeX paper scaffold and section drafts
- `archive/`
  - older notes and exploratory material

## Key Documents

Use these files as the main entry points into the project:

- `stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`
  - official Stage 1 execution notebook
- `docs/experiment-guidelines.md`
  - teammate setup, Colab workflow, and experiment loop
- `docs/local-gpu-runtime.md`
  - WSL2 and Colab local-runtime setup for using your own GPU
- `stages/README.md`
  - stage roadmap and implementation status
- `docs/hallucination-project-proposal.md`
  - project framing, scope, and research direction
- `docs/hallucination-penalty.md`
  - background note for the later paper and discussion sections
- `stages/01-grounded-abstention-baseline/results-template.md`
  - Stage 1 results and comparison template
- `paper/main.tex`
  - working LaTeX paper scaffold

## Collaboration Model

Use Git branches for collaboration instead of working directly on `main`.

Shared branches in this repository:

- `main`
  - stable branch
- `collab/team-sync`
  - shared integration branch before promoting work to `main`
- `stage/01-grounded-abstention-baseline`
- `stage/02-evidence-support-verification`
- `stage/03-confidence-calibration`
- `stage/04-unsupported-confidence-control`
- `stage/05-retrieval-grounded-qa`
- `stage/06-adaptive-constraint-balancing`

Suggested workflow:

1. Start from the relevant `stage/*` branch.
2. Create a teammate or task-specific feature branch.
3. Open pull requests into the matching `stage/*` branch.
4. Merge validated stage work into `collab/team-sync`.
5. Merge stable work from `collab/team-sync` into `main`.

## Running Stage 1

The official team workflow for Stage 1 is notebook-first:

1. Open the Stage 1 notebook in browser Google Colab from the GitHub branch you
   want to run.
2. Run the notebook in Colab.
3. Edit code locally in VS Code when changes are needed.
4. Commit and push code changes to GitHub.
5. Rerun the notebook setup cell so `/content/KeelNet` updates in Colab.
6. Save experiment artifacts to Google Drive.

The Stage 1 notebook and the Python package expose the current active
implementation.
For local commands, the package entry points are:

- `keelnet-train`
- `keelnet-eval`

If you are onboarding a teammate, start with:

- `docs/experiment-guidelines.md`
- `stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`

## References

- OpenAI, "Why language models hallucinate"
  - https://openai.com/index/why-language-models-hallucinate/
