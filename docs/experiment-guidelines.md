# KeelNet Experiment Guidelines

Use this guide to set up the official team workflow:

1. edit locally in VS Code
2. run the notebook with a Colab kernel
3. save artifacts to Google Drive
4. sync code through GitHub

Also read:

- [`stages/01-grounded-abstention-baseline/README.md`](../stages/01-grounded-abstention-baseline/README.md)
- [`stages/01-grounded-abstention-baseline/results-template.md`](../stages/01-grounded-abstention-baseline/results-template.md)

## 1. Install The Required Tools

Install:

- VS Code `Python` extension
- VS Code `Pylance` extension
- VS Code `Jupyter` extension
- Git

## 2. Create The Required Accounts

Each teammate needs:

- GitHub account
- Google account with Drive access
- Hugging Face account

Create a Hugging Face read token and store it as `HF_TOKEN` in Colab Secrets.

## 3. Get The Repo

Clone the repo locally and switch to the correct branch:

```bash
git clone git@github.com:naufalkmd/KeelNet.git
cd KeelNet
git checkout stage/01-grounded-abstention-baseline
git checkout -b yourname/stage1-work
```

## 4. Open The Notebook

Open this notebook in VS Code:

- [`stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`](../stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb)

Then:

1. connect it to a Colab kernel
2. make sure the runtime uses GPU

## 5. Set Up Drive

Use a Drive path like:

- `/content/drive/MyDrive/KeelNet`

The notebook saves outputs under:

- `DRIVE_PROJECT_DIR / artifacts / stage1_colab / RUN_NAME`

Use a unique `RUN_NAME` so teammates do not overwrite each other.

Example:

- `naufal-stage1-20260325-a`

## 6. Understand The Three Places

Do not mix these up:

1. local VS Code repo: where you edit
2. `/content/KeelNet`: the Colab execution copy
3. Google Drive: where artifacts are saved

Important:

- local file edits do not automatically update `/content/KeelNet`
- Drive is for artifacts, not the repo

## 7. Follow The Required Sync Loop

Every time you change code locally:

1. edit locally in VS Code
2. commit locally
3. push to GitHub
4. rerun the notebook setup cell
5. then run training or evaluation

If you skip step 4, Colab may still run old code.

## 8. Run The Notebook In This Order

For Stage 1, run:

1. setup cell
2. config cell
3. test cell
4. train `baseline`
5. train `abstain`
6. evaluate both
7. compare the results

Before a full run, do a smoke test with smaller:

- `MAX_TRAIN_SAMPLES`
- `MAX_EVAL_SAMPLES`

## 9. Check These Values Before Training

After the config cell, confirm:

1. `Repo dir` is `/content/KeelNet`
2. `Artifacts root` points to your Drive folder
3. `Run output dir` points to your unique run folder
4. `CUDA available: True` for full runs

## 10. Save And Report Results

For each completed run, record:

1. branch name
2. run name
3. what changed
4. main metrics
5. where the artifacts were saved

For Stage 1, fill in:

- [`stages/01-grounded-abstention-baseline/results-template.md`](../stages/01-grounded-abstention-baseline/results-template.md)

## 11. Quick Troubleshooting

If something fails, check:

1. did you push your latest code?
2. did you rerun the setup cell after pushing?
3. is `DRIVE_PROJECT_DIR` correct?
4. is `RUN_NAME` unique?
5. is the runtime on GPU?
6. is `HF_TOKEN` loaded?

## 12. One-Line Summary

Edit in VS Code, push to GitHub, rerun the setup cell, run on the Colab kernel, and save outputs to Drive.
