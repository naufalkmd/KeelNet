# KeelNet Experiment Guidelines

Use this guide when onboarding a teammate, preparing a new environment, or running a new experiment.

This document is intentionally more detailed than the stage READMEs. The goal is to give every teammate one shared operating manual for setup, collaboration, execution, and reporting.

For the current active experiment, also read:

- [`stages/01-grounded-abstention-baseline/README.md`](../stages/01-grounded-abstention-baseline/README.md)
- [`stages/01-grounded-abstention-baseline/results-template.md`](../stages/01-grounded-abstention-baseline/results-template.md)

## Why This Guide Exists

Research projects usually break down for simple reasons before they fail for scientific reasons.

Common failure patterns include:

- different teammates running different code without realizing it
- notebooks saving outputs over each other
- code living only in Colab and never making it back into Git
- experiments being rerun without a record of what changed
- results being compared across runs that used different settings

This guide is here to reduce those problems.

The short version is:

- use GitHub for code
- use Google Drive for artifacts
- keep experiment settings controlled
- document what changed
- make each run reproducible enough that another teammate can understand it later

## Scope

These guidelines cover:

- prerequisites
- recommended local tools
- required accounts and tokens
- GitHub collaboration workflow
- Google Drive layout
- Google Colab workflow
- local CLI workflow
- experiment execution standards
- result logging
- troubleshooting

Stage-specific scientific decisions still belong in each stage folder.

## Core Principles

Before you touch the code, keep these principles in mind:

- GitHub is the source of truth for code and documentation.
- Google Drive is for persistent artifacts, not for the primary Git repository.
- A branch should communicate ownership and intent.
- A run name should communicate who ran it and what it was for.
- A result is only useful if the settings behind it are known.
- A notebook is a convenience layer, not a replacement for version control.

## What Lives Where

Use this mental model across the project:

- GitHub stores code, documentation, branch history, and reviewed changes.
- Google Drive stores checkpoints, evaluation JSON files, exported tables, and large run outputs.
- Colab is a temporary execution environment.
- The stage README defines the scientific question and fixed decisions for that stage.
- The results template captures the outcome after a run is complete.

If a teammate asks "where should this go?", the answer is usually:

- code change: GitHub
- artifact or checkpoint: Drive
- final summarized finding: repo docs or stage notes

## Prerequisites

Before you run anything, make sure you have:

- a GitHub account with access to this repository
- a Google account with Google Drive access
- a Hugging Face account
- Python `3.10+` if you plan to run locally
- Git installed locally if you plan to clone, branch, commit, or push
- a modern browser if you plan to use Colab

If you are the teammate onboarding others, verify access before the first meeting. Many "technical issues" are really permission issues.

## Recommended Local Tools

If you are using VS Code, install these extensions:

- `Python`
- `Pylance`
- `Jupyter`
- `GitLens`

Why these are recommended:

- `Python` and `Pylance` improve navigation, type hints, and error visibility.
- `Jupyter` makes local notebook review easier even if training happens in Colab.
- `GitLens` helps teammates see branch history and file changes more clearly.

Also recommended:

- Google Chrome or another modern browser for Colab
- a terminal that can run `git` and `python`
- a GPU-enabled Colab runtime for actual training runs

## Required Accounts And Tokens

### GitHub

Use GitHub as the source of truth for code.

Do not use Google Drive as the primary location for the Git repository. A synced Drive copy of a Git repo can create confusing conflicts, stale files, or broken `.git` state.

Make sure you can:

- clone the repository
- create branches
- push to your branch
- open pull requests

### Hugging Face

Create a Hugging Face account at `https://huggingface.co/`.

Then create a read token:

1. Open Settings.
2. Open Access Tokens.
3. Create a token with read access.
4. Copy the token somewhere safe.

Why this matters:

- higher download rate limits
- fewer failures when multiple teammates are downloading models
- faster and more reliable model and dataset access
- fewer anonymous-access warnings in Colab

This token does not need write access for normal experiment use.

### Google Drive

Use Google Drive for persistent experiment artifacts:

- checkpoints
- evaluation JSON files
- notebook-generated tables
- run-specific output folders
- notes or screenshots you want to keep outside the ephemeral Colab runtime

Do not use Drive as the main code collaboration system.

Drive is helpful because Colab runtimes disappear. If your artifacts only exist inside `/content`, they disappear with the runtime.

## Repository Structure Overview

The main folders you should know are:

- [`src/keelnet/`](../src/keelnet)
- [`tests/`](../tests)
- [`stages/`](../stages)
- [`docs/`](../docs)

How to think about them:

- `src/keelnet/` contains the actual implementation.
- `tests/` contains checks for current code behavior.
- `stages/` contains the staged research plan and stage-specific working areas.
- `docs/` contains cross-cutting documentation like this guide.

## Branching And Collaboration Workflow

Shared branches in this repository:

- `main`
- `collab/team-sync`
- `stage/01-grounded-abstention-baseline`
- `stage/02-evidence-support-verification`
- `stage/03-confidence-calibration`
- `stage/04-unsupported-confidence-control`
- `stage/05-retrieval-grounded-qa`
- `stage/06-adaptive-constraint-balancing`

### What Each Branch Is For

- `main` is the stable branch. Treat it as the clean summary of the project state.
- `collab/team-sync` is the shared integration branch where teammate work can be combined before being promoted to `main`.
- each `stage/*` branch is the active shared branch for a particular research stage

### Recommended Workflow

1. Start from the relevant `stage/*` branch.
2. Create your own feature branch.
3. Commit and push your work to your feature branch.
4. Open a pull request into the relevant `stage/*` branch.
5. Merge validated work into `collab/team-sync`.
6. Promote stable work from `collab/team-sync` into `main`.

This keeps work organized by stage and avoids having unrelated experiments collide on one branch.

### Suggested Branch Names

Suggested feature branch naming:

- `name/stage1-colab-fixes`
- `name/stage1-metrics-review`
- `name/stage1-error-analysis-notes`
- `name/stage2-verifier-prototype`

The branch name should help a teammate answer two questions:

- who is driving this work
- what kind of work it is

### Local Git Example

If you are starting Stage 1 work locally:

```bash
git clone git@github.com:naufalkmd/KeelNet.git
cd KeelNet
git checkout stage/01-grounded-abstention-baseline
git checkout -b yourname/stage1-smoke-test
```

Commit often enough that your progress is recoverable, but do not spam meaningless commits like `fix` or `update`.

## Drive Layout And Artifact Ownership

Create a folder like this in Drive:

```text
MyDrive/
  KeelNet/
    artifacts/
      stage1_colab/
```

You can use a shared drive instead if your team already has one.

Examples:

- `/content/drive/MyDrive/KeelNet`
- `/content/drive/MyDrive/Research/KeelNet`
- `/content/drive/Shareddrives/<TeamDrive>/KeelNet`

### Why This Matters

Keeping a stable folder structure helps the whole team find outputs later. It also reduces accidental overwrites when people are moving quickly.

### What Should Be Saved In Drive

Save these to Drive:

- model checkpoints
- `run_config.json`
- `baseline_eval.json`
- `abstain_eval.json`
- exported CSV or tables if you create them
- screenshots or ad hoc notes only if they are part of the experiment record

Do not treat Drive as the authoritative place for source code changes.

### Run Naming

Each run should have a unique name. Do not reuse generic names across teammates.

Suggested run naming:

- `naufal-stage1-20260325-a`
- `alice-stage1-smoke-test`
- `bob-stage2-threshold-sweep`

Good run names usually encode:

- teammate name
- stage
- date or intent
- optional suffix if there are multiple runs that day

## Colab Setup

The main Stage 1 notebook is:

- [`stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`](../stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb)

Use this workflow in Colab:

1. Open the notebook in Google Colab.
2. Change the runtime to GPU.
3. Open the Colab Secrets panel.
4. Add a secret named `HF_TOKEN`.
5. Paste your Hugging Face read token into that secret.
6. Edit `DRIVE_PROJECT_DIR` in the setup cell.
7. Edit `RUN_NAME` in the config cell so your outputs do not overwrite someone else's.
8. Run the notebook from top to bottom.

### What The Notebook Is Designed To Do

The notebook is set up to:

- mount Google Drive
- clone or update the repo under `/content/KeelNet`
- check out the Stage 1 branch
- install the project from the cloned repo
- save artifacts into Drive

This setup is intentional. Running code from `/content/KeelNet` is more reliable than trying to execute the whole repo from the Drive mount.

### What To Verify After The Config Cell

Check these printed values:

- `Repo dir` should point to `/content/KeelNet`
- `Artifacts root` should point to your Drive folder
- `Run output dir` should point to your unique run directory
- `CUDA available: True` is preferred for full training runs

If `CUDA available: False`, the run will likely still work, but it will be much slower and may be impractical for full experiments.

### Colab Secrets

The notebook is set up to load `HF_TOKEN` from Colab Secrets automatically.

That means the safest workflow is:

1. store the token in Colab Secrets
2. rerun the setup cell
3. confirm the setup cell says the token was loaded

Do not hardcode personal tokens into notebooks that may be committed later.

## Local Setup

If you want to run locally instead of Colab:

```bash
git clone git@github.com:naufalkmd/KeelNet.git
cd KeelNet
git checkout stage/01-grounded-abstention-baseline
python -m pip install -e .
python -m unittest discover -s tests
```

Project dependencies currently live in [`pyproject.toml`](../pyproject.toml).

### When Local Execution Makes Sense

Local execution is useful for:

- reading and editing code
- running unit tests
- smoke-testing CLI behavior
- debugging without waiting for Colab

Colab is usually the easier path for teammates who need GPU access but do not have a suitable local machine.

## Before You Start An Experiment

Before starting a run, check all of the following:

- you are on the correct branch
- you know whether the run is a smoke test or a full run
- your `RUN_NAME` is unique
- your Drive path is correct
- you know what change, if any, this run is testing
- you have read the stage README

If you cannot explain what changed from the last run, pause and write it down before starting.

## Conducting An Experiment

Start every experiment by reading the stage README first.

For Stage 1, the experiment plan is in:

- [`stages/01-grounded-abstention-baseline/README.md`](../stages/01-grounded-abstention-baseline/README.md)

Stage 1 requires exactly two core runs:

1. `baseline`
2. `abstain`

### Keep The Comparison Fair

Keep these fixed across both runs:

- model backbone
- tokenizer
- preprocessing
- max length
- stride
- optimizer settings
- batch size
- epochs
- seed when possible

Do not change multiple things at once if the goal is a clean comparison.

If you want to explore a new hypothesis, either:

- treat it as a clearly named separate experiment
- or document that the run is not directly comparable to the previous baseline

### Smoke Test Versus Full Run

Use a smoke test when you want to verify:

- the notebook setup works
- the code path completes end to end
- Drive outputs are being written correctly
- the branch and environment are correct

For smoke tests, use smaller sample limits in the notebook config:

- `MAX_TRAIN_SAMPLES`
- `MAX_EVAL_SAMPLES`

Use a full run only after the smoke test path is working.

### Stage 1 Notebook Workflow

Recommended Stage 1 run flow:

1. Run the setup cell.
2. Run the config cell.
3. Run the test cell.
4. Train the `baseline` run.
5. Train the `abstain` run.
6. Run evaluation for both.
7. Compare the final metrics table.
8. Save results and notes.

Expected Stage 1 output types:

- model checkpoints
- `run_config.json`
- `baseline_eval.json`
- `abstain_eval.json`
- comparison table in notebook output

### What To Look For In Stage 1

Stage 1 is not about maximizing one overall score. The main question is whether explicit abstention improves the tradeoff between answering and declining unsupported questions.

Pay special attention to:

- unsupported-answer rate on unanswerable examples
- answerable `EM`
- answerable `F1`
- abstain precision
- abstain recall
- abstain `F1`

A run is not automatically good just because unsupported answers decreased. If answerable performance collapses or the model over-abstains, the result may not support the stage claim.

## Local CLI Workflow

If you prefer the CLI, these are the main entry points:

- `keelnet-train`
- `keelnet-eval`

Equivalent module commands also work:

```bash
python -m keelnet.train --mode baseline --output-dir outputs/stage1/baseline
python -m keelnet.train --mode abstain --output-dir outputs/stage1/abstain
python -m keelnet.evaluate --mode baseline --model-path outputs/stage1/baseline --output-path outputs/stage1/baseline_eval.json
python -m keelnet.evaluate --mode abstain --model-path outputs/stage1/abstain --output-path outputs/stage1/abstain_eval.json
```

Useful optional flags include:

- `--model-name`
- `--num-train-epochs`
- `--train-batch-size`
- `--eval-batch-size`
- `--max-train-samples`
- `--max-eval-samples`
- `--seed`
- `--learning-rate`

Use the sample limits for smoke tests before full runs.

### Why The CLI Still Matters

Even if your team mostly uses notebooks, the CLI matters because:

- it exposes the real training and evaluation entry points
- it makes debugging easier
- it gives a more scriptable path later
- it helps confirm that notebook issues are not hiding code issues

## Recording Results

Do not stop at "the notebook finished."

For each completed experiment, record:

- branch name
- run name
- model name
- train and eval settings
- final metrics
- threshold notes when relevant
- what changed from the previous run
- whether the run was a smoke test or a full run

For Stage 1, fill in:

- [`stages/01-grounded-abstention-baseline/results-template.md`](../stages/01-grounded-abstention-baseline/results-template.md)

Also add short notes under the relevant stage `notes/` folder if you found:

- a failure mode
- a bug
- a surprising tradeoff
- a threshold sensitivity issue
- a notebook setup problem others may hit too

## Manual Error Analysis

Metrics are necessary, but they are not enough.

After a meaningful run, inspect failure cases manually. The Stage 1 plan already recommends reading both answerable and unanswerable mistakes.

When reading failures, tag cases like:

- unsupported but confident answer
- answerable but over-abstained
- wrong span despite supporting evidence
- ambiguous question
- context truncation problem
- thresholding problem

The point is not to create a perfect taxonomy immediately. The point is to avoid trusting a small metric gain without understanding the behavior behind it.

## Team Communication Expectations

To keep collaboration healthy, communicate these things clearly:

- what branch you are using
- what experiment you are running
- whether it is a smoke test or a full run
- where the outputs are saved
- what changed from the last comparable run

Good async updates are short but concrete. For example:

- "Running Stage 1 smoke test on `alice/stage1-colab-fix` with 32 train and eval samples."
- "Full abstain run finished. Outputs saved under `MyDrive/KeelNet/artifacts/stage1_colab/alice-stage1-20260325-a`."

## Experiment Hygiene

Use these rules across all stages:

- change one idea at a time when possible
- keep run names unique
- do not overwrite a teammate's artifacts
- do not push notebook output noise unless it is intentionally part of the record
- keep code changes in Git, not only inside Colab
- write down what you changed before you forget
- do not compare runs as if they are equivalent when their settings differ

## Common Mistakes To Avoid

Avoid these common collaboration mistakes:

- editing code in Colab and forgetting to push it back to Git
- rerunning a notebook on the wrong branch
- saving two different experiments into the same run directory
- comparing a smoke test to a full run as if they are equivalent
- changing both model settings and preprocessing at the same time without documenting it
- assuming the latest Colab runtime matches the local environment exactly

## Quick Troubleshooting

If a Colab run fails, check these first:

- did you rerun the setup cell after pulling new code?
- is the runtime actually on GPU?
- is `DRIVE_PROJECT_DIR` correct?
- is `RUN_NAME` unique?
- did the notebook clone the expected branch?
- is `HF_TOKEN` loaded from Colab Secrets?

Common notes:

- an anonymous Hugging Face warning is not fatal, but adding `HF_TOKEN` helps
- `CUDA available: False` means CPU-only execution
- the notebook should run code from `/content/KeelNet`, not directly from Drive
- a subprocess error in the notebook often means the real traceback is in the printed stderr above it

## Recommended Default Path For New Teammates

If you are unsure where to start, use this default path:

1. Read the root [`README.md`](../README.md).
2. Read the Stage 1 [`README.md`](../stages/01-grounded-abstention-baseline/README.md).
3. Open the Stage 1 Colab notebook.
4. Set up `HF_TOKEN` in Colab Secrets.
5. Point the notebook at your Drive folder.
6. Run a small smoke test first.
7. Run the full baseline and abstain experiments.
8. Fill in the results template.
9. Add a short note if you learned something the next teammate should know.

## Final Checklist

Before you call an experiment complete, make sure:

- the code changes are committed to Git if they matter
- the outputs are saved in Drive
- the run name is recorded
- the metrics are copied into the results template or notes
- the branch used for the run is recorded
- any unusual behavior is written down somewhere teammates can find it

If those boxes are checked, the experiment is much more likely to be useful later.
