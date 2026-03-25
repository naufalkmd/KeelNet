# KeelNet Experiment Guidelines

Use this guide to set up the official team workflow:

1. edit locally in VS Code
2. run the notebook in browser Google Colab
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

### 2A. Make A Hugging Face Token

1. Sign in at `https://huggingface.co/`.
2. Open Settings.
3. Open Access Tokens.
4. Create a new token.
5. Use a clear name such as `keelnet-colab`.
6. Choose a `Read` token.
7. Copy the token value right away.

![1774432510097](image/experiment-guidelines/1774432510097.png)

If Hugging Face shows the fine-grained token screen instead, keep it read-only and do not enable write or admin-style permissions.

### 2B. Put The Token In Google Colab

1. Open the Stage 1 notebook in browser Google Colab.
2. Open the left sidebar in Colab.
3. Click the key icon for `Secrets`.
4. Add a new secret named `HF_TOKEN`.
5. Paste your Hugging Face token as the value.
6. Rerun the notebook setup cell.

What you should see after rerunning the setup cell:

- `Loaded HF_TOKEN from Colab secrets.`

![1774432546823](image/experiment-guidelines/1774432546823.png)

If you do not see the key icon, you are probably not in a real Colab runtime yet.

## 3. Get The Repo

Clone the repo locally and switch to the correct branch:

```bash
git clone git@github.com:naufalkmd/KeelNet.git
cd KeelNet
git checkout stage/01-grounded-abstention-baseline
git checkout -b yourname/stage1-work
```

## 4. Open The Notebook

Edit this notebook in VS Code if you want:

- [`stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`](../stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb)

Then:

1. open the same notebook in browser **Google Colab**
2. make sure the runtime uses GPU
3. run it there, not in a normal local Jupyter kernel

Important:

- use VS Code for editing
- use browser Colab for executing this notebook
- this notebook depends on `google.colab`, Drive mount, and Colab Secrets

![1774434899953](image/experiment-guidelines/1774434899953.png)

## 5. Set Up Drive

Use a shared Drive path like:

- `/content/drive/Shareddrives/YourTeamDrive/KeelNet`

If you are working alone temporarily, you can fall back to:

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
3. shared Google Drive: where artifacts are saved

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

Edit in VS Code, push to GitHub, open the notebook in browser Colab, rerun the setup cell, and save outputs to Drive.
