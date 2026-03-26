# KeelNet Experiment Guidelines

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 55}, "themeVariables": {"fontSize": "20px"}}}%%
flowchart LR
    A[Make HF token] --> B[Open Colab<br/>from GitHub branch]
    B --> C[Add HF_TOKEN<br/>in Secrets]
    C --> D[Run setup,<br/>config, and test]
    D --> E[Edit in VS Code]
    E --> F[Commit and push]
    F --> G[Rerun setup]
    G --> H{Stage OK?}
    H -- No --> E
    H -- Yes --> I[Run full experiment<br/>and save to Drive]
```

Use this guide to set up the official team workflow:

1. open the notebook in browser Google Colab from the correct GitHub branch
2. run the notebook in Colab
3. edit code locally in VS Code when needed
4. commit and push changes to GitHub
5. rerun the setup cell in Colab
6. repeat until the stage succeeds

Before the first notebook run, do the key and Hugging Face setup in this order:

1. make a Hugging Face token
2. open the notebook in browser Google Colab
3. add the token to Colab Secrets as `HF_TOKEN`
4. then run the setup cell

Also read:

- [`stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`](../stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb)
- [`stages/01-grounded-abstention-baseline/results-template.md`](../stages/01-grounded-abstention-baseline/results-template.md)

## 1. Prerequisites

```mermaid
flowchart LR
    A[Install tools] --> B[Create accounts]
    B --> C[Make HF token]
    C --> D[Add HF_TOKEN in Colab Secrets]
    D --> E[Set up Drive folder]
    E --> F[Clone repo and branch locally]
```

### 1A. Install The Required Tools

Install:

- VS Code `Python` extension
- VS Code `Pylance` extension
- VS Code `Jupyter` extension
- Git

### 1B. Create The Required Accounts

Each teammate needs:

- GitHub account
- Google account with Drive access
- Hugging Face account

### 1C. Make A Hugging Face Token

1. Sign in at `https://huggingface.co/`.
2. Open Settings.
3. Open Access Tokens.
4. Create a new token.
5. Use a clear name such as `keelnet-colab`.
6. Choose a `Read` token.
7. Copy the token value right away.

![1774432510097](image/experiment-guidelines/1774432510097.png)

If Hugging Face shows the fine-grained token screen instead, keep it read-only and do not enable write or admin-style permissions.

### 1D. Put The Token In Google Colab

1. Open the stage notebook in browser Google Colab.
2. Open the left sidebar in Colab.
3. Click the key icon for `Secrets`.
4. Add a new secret named `HF_TOKEN`.
5. Paste your Hugging Face token as the value.
6. Rerun the notebook setup cell.

What you should see after rerunning the setup cell:

- `Loaded HF_TOKEN from Colab secrets.`

![1774432546823](image/experiment-guidelines/1774432546823.png)

If you do not see the key icon, you are probably not in a real Colab runtime yet.

### 1E. Prepare Local Editing

Clone the repo locally and switch to the correct branch:

```bash
git clone git@github.com:naufalkmd/KeelNet.git
cd KeelNet
git checkout stage/01-grounded-abstention-baseline
git checkout -b yourname/stage1-work
```

### 1F. Set Up Drive

Use this Drive path:

- `/content/drive/MyDrive/KeelNet`

Share the `KeelNet` folder in your Google Drive with your teammates so they can see the saved artifacts.

Each stage notebook saves outputs under its own stage-specific folder:

- `DRIVE_PROJECT_DIR / artifacts / stageN_colab / RUN_NAME`

Each notebook now builds `RUN_NAME` automatically from `AUTHOR_NAME` plus the stage and version number.

Example:

- `naufal-stage1-v1`
- `naufal-stage1-v2`

![1774439287595](image/experiment-guidelines/1774439287595.png)

### 1G. Understand The Three Places

Do not mix these up:

1. local VS Code repo: where you edit
2. `/content/KeelNet`: the Colab execution copy
3. `/content/drive/MyDrive/KeelNet`: the shared artifact folder

Important:

- local file edits do not automatically update `/content/KeelNet`
- Drive is for artifacts, not the repo

Important:

- Stage 1 has the completed reference run and report-ready baseline comparison
- Stage 2 now has a completed reference run with `verifier_eval.json`; the verifier signal is real, but the downstream gain is still small because the current gate is too permissive
- Stages 3 to 6 already have teammate notebooks and a shared workflow, but their stage-specific code paths still need to be implemented stage by stage

Current reference findings to keep in mind while planning later stages:

- Stage 1 changed unsupported-answer rate from `100.00%` to `27.97%`, but answerable `F1` fell from `84.82` to `66.76`
- Stage 2 reached support `F1 = 79.79`, but gated overall `F1` moved only from `69.40` to `69.42`
- a representative Stage 2 false positive is the unanswerable question `What is France a region of?`, where the system still answers `Normandy` and gives it a support score near `1.0`
- a representative Stage 2 false negative is the answerable question `If input size is equal to n, what can respectively be assumed is the function of n?`, where the system predicts `the time taken` correctly but the verifier suppresses it

## 2. Start Of A Stage

```mermaid
flowchart LR
    A[Open Colab] --> B[Pick GitHub repo and branch]
    B --> C[Open stage notebook]
    C --> D[Run setup]
    D --> E[Run config]
    E --> F[Run validate or test]
    F --> G[Optional smoke test]
    G --> H[Run main stage cells]
```

Open the stage notebook in browser Google Colab like this:

1. open `https://colab.google.com/`
2. click `GitHub`
3. choose `naufalkmd/KeelNet`
4. switch to the branch you want to run
5. open the stage notebook you want
6. example: `stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`
7. make sure the runtime uses GPU
8. do not run the setup cell yet if `HF_TOKEN` is not in Colab Secrets

<p align="center">
  <img src="image/experiment-guidelines/1774437932291.png" alt="Colab GitHub picker" width="48%" />
  <img src="image/experiment-guidelines/1774434899953.png" alt="Colab notebook open screen" width="48%" />
</p>

Important:

- use browser Colab for executing this notebook
- use VS Code for editing code between notebook runs
- hosted Colab uses `google.colab`, Drive mount, and Colab Secrets
- for Colab connected to a local runtime, use `docs/local-gpu-runtime.md`

Run the notebook in this order:

1. setup cell
2. config cell
3. validation or test cell
4. optional smoke test
5. main stage command cells
6. artifact or notes cells

For Stage 1 specifically, the main run order is:

1. train `baseline`
2. train `abstain`
3. evaluate both
4. compare the results

Before a long run, confirm:

1. `Repo dir` is `/content/KeelNet`
2. `Artifacts root` points to your Drive folder in hosted Colab or your local project folder in local runtime
3. `Run output dir` points to your unique run folder
4. `CUDA available: True` for full runs

Recommended execution priority after Stage 2:

1. do Stage 3 next so QA and verifier scores become calibrated enough for downstream control
2. do Stage 4 after that so you can test a fixed control rule against the current Stage 2 baseline
3. decide whether the fixed-control result is already strong enough for the paper
4. use Stage 6 only after the Stage 4 fixed controller is stable and worth trying to beat
5. use Stage 5 after the controlled proof path when you want a realism check under retrieval noise

Do not center the project claim on Stage 3 alone or on beating raw answer accuracy by itself. Treat Stage 3 as the calibration step that makes Stage 4 and Stage 6 defensible.

## 3. Editing Loop

```mermaid
flowchart LR
    A[Run current notebook cells] --> B{Need code changes?}
    B -- Yes --> C[Edit in VS Code]
    C --> D[Commit and push]
    D --> E[Rerun setup in Colab]
    E --> F[Rerun needed cells]
    F --> G{Stage successful?}
    G -- No --> B
    G -- Yes --> H[Push branch and share run]
    B -- No --> F
```

Use this same loop during the whole stage:

1. open the notebook in browser Colab from the correct GitHub branch
2. run the setup cell
3. run the config cell
4. run the validation or test cell
5. if something needs to change, edit the code locally in VS Code
6. commit your changes locally
7. push your branch to GitHub
8. if you are using hosted Colab, rerun the setup cell so `/content/KeelNet` updates; in local runtime, rerun setup after environment or dependency changes
9. rerun the cells you need
10. repeat until the stage succeeds

If you skip step 8 in hosted Colab, Colab may still run old code.

Before a full Stage 1 run, do a smoke test with smaller:

- `MAX_TRAIN_SAMPLES`
- `MAX_EVAL_SAMPLES`

Main rule:

1. push your branch whenever code changes
2. keep artifacts in Drive or your local project folder for every meaningful run
3. only write a short result note for runs worth sharing or comparing

If you want to leave a short result note, keep it minimal:

1. branch name
2. `RUN_NAME`
3. main metrics
4. artifact folder path

Each stage notebook now ends with a `Share This Run` cell that prints this summary, saves it into the current run folder, and marks the run complete so the next run becomes the next version automatically.

For Stage 1, you can use:

- [`stages/01-grounded-abstention-baseline/results-template.md`](../stages/01-grounded-abstention-baseline/results-template.md)

For Stages `02` to `06`, use:

- [`stages/stage-note-template.md`](../stages/stage-note-template.md)

Recommended note rhythm:

1. fill in the goal, success condition, and planned commands before implementation
2. update the note after smoke test with setup issues or command fixes
3. update it again after any meaningful full run with metrics, verdict, and next actions

## 4. Troubleshooting

If something fails, check:

1. did you push your latest code?
2. did you rerun the setup cell after pushing?
3. is `DRIVE_PROJECT_DIR` correct?
4. is `AUTHOR_NAME` correct, and does the generated `RUN_NAME` look right?
5. is the runtime on GPU?
6. is `HF_TOKEN` loaded?

## 5. One-Line Summary

Open the notebook in browser Colab from the correct GitHub branch, run it there, edit code in VS Code, push changes, rerun the setup cell, and repeat.
