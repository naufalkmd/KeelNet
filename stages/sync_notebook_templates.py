from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]


def source_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


def intro_markdown(stage_label: str, stage_number: int) -> str:
    return dedent(
        f"""
        # KeelNet {stage_label} Template

        Use this notebook as the Stage {stage_number} working file for the official team workflow:

        1. edit locally in VS Code
        2. open this notebook in browser Google Colab
        3. rerun the setup cell after pushing code changes
        4. save artifacts to Google Drive or your local runtime project folder

        This notebook keeps the Stage {stage_number} notes, implementation hints, run commands, and shareable outputs in one place.
        """
    ).strip() + "\n"


SETUP_MARKDOWN = (
    dedent(
        """
        <h2 style="color: #1d4ed8;">1. Setup And Sync</h2>

        Run this cell in either hosted Google Colab or Google Colab connected to a local Jupyter runtime.

        What it does:

        - hosted Colab: mounts Drive, loads `HF_TOKEN` from Colab Secrets if available, clones or updates `/content/KeelNet`, checks out the stage branch, and installs the project
        - local runtime: reuses your local repo, uses a local project folder for artifacts, reads `HF_TOKEN` from the environment if available, and installs the project into the current kernel environment

        Path reminder:

        - hosted Colab defaults: repo `/content/KeelNet`, project folder `/content/drive/MyDrive/KeelNet`
        - local runtime defaults: repo `/content/KeelNet` if present, otherwise your current local checkout; project folder `/content/KeelNet-local`
        - optional overrides: `KEELNET_REPO_DIR` and `KEELNET_PROJECT_DIR`
        """
    ).strip()
    + "\n"
)

VALIDATE_MARKDOWN = (
    dedent(
        """
        <h2 style="color: #1d4ed8;">3. Validate The Environment</h2>

        Run the project tests before stage-specific work. This confirms the installed code path is at least minimally healthy inside the current runtime.
        """
    ).strip()
    + "\n"
)

STAGE_NOTE_TEMPLATE_MARKDOWN = (
    dedent(
        """
        ## Stage Note Template

        Keep your stage notes inside this notebook flow. Update them at three points:

        1. before implementation: fill in the goal, success condition, and planned commands
        2. after smoke test: record environment issues and command fixes
        3. after a meaningful run: record metrics, verdict, and next actions

        Use this structure for the generated run note:

        - Run info
        - Goal
        - Commands
        - Main metrics
        - What changed
        - What worked
        - What failed or looks risky
        - Error cases to review
        - Decision
        - Next actions
        """
    ).strip()
    + "\n"
)


def implementation_banner(stage_number: int, start_here: str, finish_here: str, out_of_scope: str) -> str:
    return (
        dedent(
            f"""
            <div style="border-left: 6px solid #c2410c; background: #fff7ed; padding: 12px 16px; border-radius: 8px;">
            <strong>Implementation Starts Here</strong><br/>
            Sections 1-4 are setup and validation. Section 5 onward is the main Stage {stage_number} work area.
            <ul>
              <li><strong>Start here:</strong> {start_here}</li>
              <li><strong>Finish here:</strong> {finish_here}</li>
              <li><strong>Out of scope:</strong> {out_of_scope}</li>
            </ul>
            </div>
            """
        ).strip()
        + "\n"
    )


def setup_code(branch: str) -> str:
    return (
        dedent(
            f"""
            from pathlib import Path
            import os
            import subprocess
            import sys


            GIT_REPO_URL = "https://github.com/naufalkmd/KeelNet.git"
            GIT_BRANCH = {branch!r}
            HOSTED_COLAB_PROJECT_DIR = Path("/content/drive/MyDrive/KeelNet")
            DEFAULT_LOCAL_PROJECT_DIR = Path("/content/KeelNet-local")
            DEFAULT_LOCAL_REPO_DIR = Path("/content/KeelNet")


            def detect_runtime_mode() -> str:
                try:
                    import google.colab  # noqa: F401
                except ImportError:
                    return "local-runtime"
                return "hosted-colab"


            RUNTIME_MODE = detect_runtime_mode()
            IS_HOSTED_COLAB = RUNTIME_MODE == "hosted-colab"
            PROJECT_STORAGE_LABEL = "Drive project dir" if IS_HOSTED_COLAB else "Local project dir"


            def run_setup(cmd, *, cwd: Path | None = None) -> None:
                rendered = [str(part) for part in cmd]
                print("$", " ".join(rendered))
                subprocess.run(rendered, check=True, cwd=str(cwd) if cwd else None)


            def configure_project_storage() -> Path:
                if IS_HOSTED_COLAB:
                    from google.colab import drive

                    project_dir = Path(os.environ.get("KEELNET_PROJECT_DIR", str(HOSTED_COLAB_PROJECT_DIR)))
                    drive.mount("/content/drive", force_remount=False)
                    if not str(project_dir).startswith("/content/drive/"):
                        raise ValueError(
                            f"KEELNET_PROJECT_DIR must point inside /content/drive in hosted Colab, got: {{project_dir}}"
                        )
                    project_dir.mkdir(parents=True, exist_ok=True)
                    print(f"{{PROJECT_STORAGE_LABEL}}: {{project_dir}}")
                    return project_dir

                project_dir = Path(os.environ.get("KEELNET_PROJECT_DIR", str(DEFAULT_LOCAL_PROJECT_DIR))).expanduser().resolve()
                project_dir.mkdir(parents=True, exist_ok=True)
                print(f"{{PROJECT_STORAGE_LABEL}}: {{project_dir}}")
                return project_dir


            def configure_hf_token() -> None:
                if os.environ.get("HF_TOKEN"):
                    print("HF_TOKEN already set in the environment.")
                    return

                if not IS_HOSTED_COLAB:
                    print("HF_TOKEN not set in the environment; continuing with anonymous HF access.")
                    return

                try:
                    from google.colab import userdata
                except ImportError:
                    print("Colab secrets are unavailable; continuing without HF_TOKEN.")
                    return

                try:
                    token = userdata.get("HF_TOKEN")
                except Exception:
                    token = None

                if token:
                    os.environ["HF_TOKEN"] = token
                    print("Loaded HF_TOKEN from Colab secrets.")
                else:
                    print("HF_TOKEN not found in Colab secrets; continuing with anonymous HF access.")


            def resolve_local_repo_dir() -> Path:
                candidates: list[Path] = []
                env_repo_dir = os.environ.get("KEELNET_REPO_DIR")
                if env_repo_dir:
                    candidates.append(Path(env_repo_dir).expanduser())
                candidates.append(DEFAULT_LOCAL_REPO_DIR)
                cwd = Path.cwd().resolve()
                candidates.append(cwd)
                candidates.extend(cwd.parents)

                seen: set[Path] = set()
                for candidate in candidates:
                    resolved = candidate.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    if (resolved / ".git").exists() and (resolved / "pyproject.toml").exists():
                        return resolved

                raise FileNotFoundError(
                    "Could not find the KeelNet repo. Set KEELNET_REPO_DIR to your local checkout before running this cell."
                )


            def ensure_repo() -> Path:
                if not IS_HOSTED_COLAB:
                    return resolve_local_repo_dir()

                local_repo_dir = Path(os.environ.get("KEELNET_REPO_DIR", str(DEFAULT_LOCAL_REPO_DIR)))
                if (local_repo_dir / ".git").exists():
                    run_setup(["git", "fetch", "origin"], cwd=local_repo_dir)
                else:
                    run_setup(["git", "clone", GIT_REPO_URL, str(local_repo_dir)])

                run_setup(["git", "checkout", GIT_BRANCH], cwd=local_repo_dir)
                run_setup(["git", "pull", "--ff-only", "origin", GIT_BRANCH], cwd=local_repo_dir)
                return local_repo_dir.resolve()


            PROJECT_STORAGE_DIR = configure_project_storage()
            DRIVE_PROJECT_DIR = PROJECT_STORAGE_DIR
            configure_hf_token()
            REPO_DIR = ensure_repo().resolve()
            os.chdir(REPO_DIR)
            print(f"Runtime mode: {{RUNTIME_MODE}}")
            print(f"Runtime repo dir: {{REPO_DIR}}")
            CURRENT_BRANCH = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
            ).stdout.strip()
            print(f"Git branch: {{CURRENT_BRANCH}}")
            run_setup([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)])
            """
        ).strip()
        + "\n"
    )


def generic_config_code(
    *,
    stage_label: str,
    stage_number: int,
    objective: str,
    metrics: list[str],
    hints: list[str],
    modules: list[str],
) -> str:
    return (
        dedent(
            f"""
            from pathlib import Path
            import json
            import re
            import subprocess
            import sys

            import torch

            REPO_DIR = Path(REPO_DIR).resolve()
            PROJECT_STORAGE_DIR = Path(PROJECT_STORAGE_DIR).resolve()
            DRIVE_PROJECT_DIR = PROJECT_STORAGE_DIR

            # Change only this for each teammate. The notebook builds the stage name and next version automatically.
            AUTHOR_NAME = "yourname"
            RUN_BASENAME = f"{{AUTHOR_NAME}}-stage{stage_number}"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "stage{stage_number}_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            STAGE_TITLE = {stage_label!r}
            STAGE_OBJECTIVE = {objective!r}
            TARGET_METRICS = {metrics!r}
            IMPLEMENTATION_HINTS = {hints!r}
            SUGGESTED_MODULES = {modules!r}

            # Fill these in when the stage code is ready.
            RUN_SMOKE_TEST = False
            SMOKE_TEST_COMMANDS = [
                # Example:
                # [sys.executable, "-m", "keelnet.some_module", "--help"],
            ]
            STAGE_COMMANDS = [
                # Example:
                # [sys.executable, "-m", "keelnet.some_module", "--output-dir", str(OUTPUT_ROOT / "trial-1")],
            ]


            def completed_versions(root: Path, run_basename: str) -> list[int]:
                versions: list[int] = []
                if not root.exists():
                    return versions

                pattern = re.compile(rf"^{{re.escape(run_basename)}}-v(\\d+)$")
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    match = pattern.match(child.name)
                    if match and (child / COMPLETION_MARKER_NAME).exists():
                        versions.append(int(match.group(1)))
                return sorted(versions)


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{{RUN_BASENAME}}-v{{RUN_VERSION}}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            RUN_MARKER_FILE.write_text(
                "\\n".join(
                    [
                        f"stage={{STAGE_TITLE}}",
                        f"run_name={{RUN_NAME}}",
                        f"run_version=v{{RUN_VERSION}}",
                        f"runtime_mode={{RUNTIME_MODE}}",
                        f"repo_dir={{REPO_DIR}}",
                        f"project_storage_dir={{PROJECT_STORAGE_DIR}}",
                        f"git_branch={{CURRENT_BRANCH}}",
                        "status=configured",
                        "note=This file is created when the config cell runs.",
                    ]
                )
                + "\\n",
                encoding="utf-8",
            )

            print(f"Runtime mode: {{RUNTIME_MODE}}")
            print(f"Repo dir: {{REPO_DIR}}")
            print(f"{{PROJECT_STORAGE_LABEL}}: {{PROJECT_STORAGE_DIR}}")
            print(f"Artifacts root: {{ARTIFACTS_ROOT}}")
            print(f"Run basename: {{RUN_BASENAME}}")
            print(f"Run version: v{{RUN_VERSION}}")
            print(f"Run output dir: {{OUTPUT_ROOT}}")
            print(f"Run marker file: {{RUN_MARKER_FILE}}")
            print(f"CUDA available: {{torch.cuda.is_available()}}")
            if torch.cuda.is_available():
                print(f"GPU: {{torch.cuda.get_device_name(0)}}")
            print("Target metrics:", ", ".join(TARGET_METRICS))
            print("Suggested modules:", ", ".join(SUGGESTED_MODULES))


            def run(cmd):
                rendered = [str(part) for part in cmd]
                print("$", " ".join(rendered))
                with subprocess.Popen(
                    rendered,
                    cwd=REPO_DIR,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                ) as process:
                    if process.stdout is not None:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                    return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, rendered)


            def run_many(commands, *, label: str) -> bool:
                if not commands:
                    print(f"No commands configured for {{label}} yet.")
                    return False

                for index, cmd in enumerate(commands, start=1):
                    print(f"\\n[{{label}} {{index}}/{{len(commands)}}]")
                    run(cmd)
                return True
            """
        ).strip()
        + "\n"
    )


def generic_save_code() -> str:
    return (
        dedent(
            """
            if not RUN_NOTES_FILE.exists():
                metric_lines = [f"- {metric}" for metric in TARGET_METRICS]
                RUN_NOTES_FILE.write_text(
                    "\\n".join(
                        [
                            f"# {STAGE_TITLE} Notes",
                            "",
                            "Update this note three times:",
                            "1. before implementation: goal, success condition, and commands",
                            "2. after smoke test: environment issues and command fixes",
                            "3. after a meaningful run: metrics, verdict, and next actions",
                            "",
                            "## Run Info",
                            f"- Branch: {CURRENT_BRANCH}",
                            f"- `RUN_NAME`: {RUN_NAME}",
                            f"- Output folder: {OUTPUT_ROOT}",
                            f"- Runtime mode: {RUNTIME_MODE}",
                            "",
                            "## Goal",
                            f"- One-sentence objective: {STAGE_OBJECTIVE}",
                            "- Success condition:",
                            "- Out of scope:",
                            "",
                            "## Commands",
                            f"- Smoke test command(s): {SMOKE_TEST_COMMANDS}",
                            f"- Main command(s): {STAGE_COMMANDS}",
                            "- Input artifacts or checkpoints:",
                            "- Output files to inspect:",
                            "",
                            "## Main Metrics",
                            *metric_lines,
                            "",
                            "## What Changed",
                            "- ",
                            "",
                            "## What Worked",
                            "- ",
                            "",
                            "## What Failed Or Looks Risky",
                            "- ",
                            "",
                            "## Error Cases To Review",
                            "- ",
                            "",
                            "## Decision",
                            "- Keep, revise, or stop:",
                            "- Reason:",
                            "",
                            "## Next Actions",
                            "1. ",
                            "2. ",
                            "3. ",
                        ]
                    )
                    + "\\n",
                    encoding="utf-8",
                )

            RUN_SUMMARY_FILE.write_text(
                json.dumps(
                    {
                        "stage": STAGE_TITLE,
                        "run_name": RUN_NAME,
                        "runtime_mode": RUNTIME_MODE,
                        "git_branch": CURRENT_BRANCH,
                        "project_storage_dir": str(PROJECT_STORAGE_DIR),
                        "output_root": str(OUTPUT_ROOT),
                        "target_metrics": TARGET_METRICS,
                        "suggested_modules": SUGGESTED_MODULES,
                    },
                    indent=2,
                )
                + "\\n",
                encoding="utf-8",
            )

            print(f"Notes template: {RUN_NOTES_FILE}")
            print(f"Run summary: {RUN_SUMMARY_FILE}")
            print("Current files under OUTPUT_ROOT:")
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                print(path)
            """
        ).strip()
        + "\n"
    )


def generic_share_code() -> str:
    return (
        dedent(
            """
            from datetime import datetime, timezone

            share_lines = [
                f"# {STAGE_TITLE} Share Note",
                "",
                f"- runtime mode: {RUNTIME_MODE}",
                f"- branch name: {CURRENT_BRANCH}",
                f"- RUN_NAME: {RUN_NAME}",
                *[f"- {metric}: <fill in after review>" for metric in TARGET_METRICS],
                f"- Output folder path: {OUTPUT_ROOT}",
            ]
            share_note = "\\n".join(share_lines) + "\\n"
            SHARE_NOTE_FILE = OUTPUT_ROOT / "collab-share-note.md"
            SHARE_NOTE_FILE.write_text(share_note, encoding="utf-8")
            COMPLETION_MARKER_FILE.write_text(
                "\\n".join(
                    [
                        f"run_name={RUN_NAME}",
                        f"completed_at={datetime.now(timezone.utc).isoformat()}",
                        f"share_note={SHARE_NOTE_FILE.name}",
                        "status=completed",
                    ]
                )
                + "\\n",
                encoding="utf-8",
            )
            print(share_note)
            print(f"Update the metric lines in: {SHARE_NOTE_FILE}")
            print(f"Saved completion marker: {COMPLETION_MARKER_FILE}")
            """
        ).strip()
        + "\n"
    )


STAGE_2 = {
    "path": REPO_ROOT / "stages/02-evidence-support-verification/notebooks/google-colab.ipynb",
    "stage_number": 2,
    "stage_label": "Stage 2: Evidence Support Verification",
    "config_markdown": (
        dedent(
            """
            <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

            Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage2-v1`, `yourname-stage2-v2`, `yourname-stage2-v3`, and so on based on completed runs.

            Set `BASE_QA_MODEL_DIR` to a finished Stage 1 model directory before running the main Stage 2 commands.

            This cell also prints the important values you should check before training.

            It creates a small `RUN_STARTED.txt` file in the current run folder immediately, so you can confirm the output path is correct before training finishes.
            """
        ).strip()
        + "\n"
    ),
    "smoke_markdown": (
        dedent(
            """
            <h2 style="color: #1d4ed8;">4. Optional Smoke Test</h2>

            Use this cell to run a tiny verifier-only training command before the full Stage 2 run. Keep it small so you can catch path, dependency, and runtime issues early.
            """
        ).strip()
        + "\n"
    ),
    "implementation_banner": implementation_banner(
        2,
        "point <code>BASE_QA_MODEL_DIR</code> at a finished Stage 1 model, then run the verifier train and evaluate commands below.",
        "question plus passage plus predicted answer produces a stable support label or score, and the verifier metrics are saved to <code>VERIFIER_EVAL</code>.",
        "retrieval, full calibration, adaptive balancing, large pipeline redesign.",
    ),
    "implementation_markdown": (
        dedent(
            """
            ## IMPLEMENTATION 1: Train And Evaluate The Stage 2 Verifier

            This is the main Stage 2 implementation and run section. Everything before this point is setup, sync, or validation.

            This section trains the verifier and, when `BASE_QA_MODEL_DIR` is set, evaluates it on top of the Stage 1 QA model.
            """
        ).strip()
        + "\n"
    ),
    "save_markdown": (
        dedent(
            """
            ## Save Notes And Review Artifacts

            This cell creates teammate-friendly note files inside the current run folder and lists the current artifacts. Update the generated notes as you learn what does and does not work in Stage 2: Evidence Support Verification.
            """
        ).strip()
        + "\n"
    ),
    "final_markdown": (
        dedent(
            """
            <h2 style="color: #15803d;">Final Check</h2>

            A useful Stage 2 result is not just a higher verifier score.

            Check all three:

            - support classification F1 is stable enough to trust
            - the gated QA metrics move in the right direction, not just the verifier-only metrics
            - failure cases are interpretable enough to inform the next stage

            After that, inspect `verifier_eval.json` and confirm the selected support threshold looks reasonable on the saved metrics.
            """
        ).strip()
        + "\n"
    ),
    "share_markdown": (
        dedent(
            """
            <h2 style="color: #15803d;">Share This Run</h2>

            This cell prints a minimal share-ready summary for teammates, saves it into the current run folder, and marks the run as completed so the next run becomes the next version.

            If `VERIFIER_EVAL` exists, this cell reads the saved metrics automatically.
            """
        ).strip()
        + "\n"
    ),
}


GENERIC_STAGES = [
    {
        "path": REPO_ROOT / "stages/03-confidence-calibration/notebooks/google-colab.ipynb",
        "branch": "stage/03-confidence-calibration",
        "stage_number": 3,
        "stage_label": "Stage 3: Confidence Calibration",
        "objective": "Make the model's confidence scores reflect actual correctness and support strength instead of acting like raw uncalibrated logits.",
        "metrics": ["ECE", "Brier Score", "correlation between confidence and supported correctness"],
        "hints": [
            "input: Stage 1 or Stage 2 predictions",
            "output: calibrated confidence for answer / abstain decisions, and optionally support predictions",
            "expose confidence scores and evaluate or train for calibration",
        ],
        "modules": ["keelnet.calibration", "keelnet.evaluate", "keelnet.metrics"],
        "config_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

                Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage3-v1`, `yourname-stage3-v2`, `yourname-stage3-v3`, and so on based on completed runs.

                Review `TARGET_METRICS`, `SUGGESTED_MODULES`, `SMOKE_TEST_COMMANDS`, and `STAGE_COMMANDS` before you move into implementation.

                This cell also prints the important values you should check before running stage commands.

                It creates a small `RUN_STARTED.txt` file in the current run folder immediately, so you can confirm the output path is correct before training or evaluation finishes.
                """
            ).strip()
            + "\n"
        ),
        "smoke_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">4. Optional Smoke Test</h2>

                Use this cell only after you fill in `SMOKE_TEST_COMMANDS` in the config cell. Keep those commands tiny so you can catch path, dependency, and runtime issues before a full Stage 3 run.
                """
            ).strip()
            + "\n"
        ),
        "implementation_banner": implementation_banner(
            3,
            "create <code>src/keelnet/calibration.py</code> or an equivalent calibration module, then extend <code>src/keelnet/evaluate.py</code> and <code>src/keelnet/metrics.py</code>.",
            "confidence better reflects correctness and support quality.",
            "retrieval, adaptive balancing, unrelated architecture changes.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Run The Stage 3 Calibration Commands

                This is the main Stage 3 implementation and run section. Everything before this point is setup, sync, or validation.

                Fill in `STAGE_COMMANDS` in the config cell with the actual commands for this stage. Start with one command, make sure the outputs land in `OUTPUT_ROOT`, then add the rest.
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                ## Save Notes And Review Artifacts

                This cell creates teammate-friendly note files inside the current run folder and lists the current artifacts. Update the generated notes as you learn what does and does not work in Stage 3: Confidence Calibration.
                """
            ).strip()
            + "\n"
        ),
        "final_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Final Check</h2>

                A useful Stage 3 result is not just a calibration plot.

                Check all three:

                - high-confidence predictions are more reliable than low-confidence predictions
                - calibration improves without just flattening all confidence values
                - the calibrated signals are practical enough to reuse in later stages

                After that, log the exact operating point you trust and keep the saved artifacts under `OUTPUT_ROOT` easy to compare against future stages.
                """
            ).strip()
            + "\n"
        ),
        "share_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Share This Run</h2>

                This cell prints a minimal share-ready summary for teammates, saves it into the current run folder, and marks the run as completed so the next run becomes the next version.

                Update the metric lines after you review the outputs for this stage.
                """
            ).strip()
            + "\n"
        ),
    },
    {
        "path": REPO_ROOT / "stages/04-unsupported-confidence-control/notebooks/google-colab.ipynb",
        "branch": "stage/04-unsupported-confidence-control",
        "stage_number": 4,
        "stage_label": "Stage 4: Unsupported-Confidence Control",
        "objective": "Reduce confident unsupported answers without collapsing usefulness.",
        "metrics": ["unsupported-answer rate", "supported-answer rate", "answer F1", "abstain F1"],
        "hints": [
            "input: answer, support signal, confidence signal",
            "output: better control over unsupported confident answering",
            "add an explicit penalty or control mechanism for confident unsupported outputs",
        ],
        "modules": ["keelnet.control", "keelnet.evaluate", "keelnet.metrics"],
        "config_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

                Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage4-v1`, `yourname-stage4-v2`, `yourname-stage4-v3`, and so on based on completed runs.

                Review `TARGET_METRICS`, `SUGGESTED_MODULES`, `SMOKE_TEST_COMMANDS`, and `STAGE_COMMANDS` before you move into implementation.

                This cell also prints the important values you should check before running stage commands.

                It creates a small `RUN_STARTED.txt` file in the current run folder immediately, so you can confirm the output path is correct before training or evaluation finishes.
                """
            ).strip()
            + "\n"
        ),
        "smoke_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">4. Optional Smoke Test</h2>

                Use this cell only after you fill in `SMOKE_TEST_COMMANDS` in the config cell. Keep those commands tiny so you can catch path, dependency, and runtime issues before a full Stage 4 run.
                """
            ).strip()
            + "\n"
        ),
        "implementation_banner": implementation_banner(
            4,
            "create <code>src/keelnet/control.py</code> or an equivalent control module, then connect answer, support, and confidence signals in evaluation.",
            "confident unsupported answers go down without useful supported answers collapsing.",
            "retrieval, adaptive balancing, unrelated model redesigns.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Run The Stage 4 Control Commands

                This is the main Stage 4 implementation and run section. Everything before this point is setup, sync, or validation.

                Fill in `STAGE_COMMANDS` in the config cell with the actual commands for this stage. Start with one command, make sure the outputs land in `OUTPUT_ROOT`, then add the rest.
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                ## Save Notes And Review Artifacts

                This cell creates teammate-friendly note files inside the current run folder and lists the current artifacts. Update the generated notes as you learn what does and does not work in Stage 4: Unsupported-Confidence Control.
                """
            ).strip()
            + "\n"
        ),
        "final_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Final Check</h2>

                A Stage 4 win is not just fewer unsupported answers.

                Check all three:

                - unsupported confident answers actually go down
                - supported-answer rate and answer quality remain useful
                - the behavior change comes from the intended control mechanism, not just over-abstention

                After that, record a few failure cases so you can tell whether Stage 5 needs better evidence or better answer control.
                """
            ).strip()
            + "\n"
        ),
        "share_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Share This Run</h2>

                This cell prints a minimal share-ready summary for teammates, saves it into the current run folder, and marks the run as completed so the next run becomes the next version.

                Update the metric lines after you review the outputs for this stage.
                """
            ).strip()
            + "\n"
        ),
    },
    {
        "path": REPO_ROOT / "stages/05-retrieval-grounded-qa/notebooks/google-colab.ipynb",
        "branch": "stage/05-retrieval-grounded-qa",
        "stage_number": 5,
        "stage_label": "Stage 5: Retrieval-Grounded QA",
        "objective": "Move from fixed evidence to retrieved evidence without losing grounded answer / abstain behavior.",
        "metrics": ["retrieval recall at k", "answer F1", "unsupported-answer rate after retrieval", "abstain quality"],
        "hints": [
            "input: question",
            "intermediate step: retrieve top-k evidence",
            "output: answer or ABSTAIN",
            "add retrieval as a first-class part of the pipeline",
        ],
        "modules": ["keelnet.retrieve", "keelnet.evaluate", "keelnet.metrics"],
        "config_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

                Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage5-v1`, `yourname-stage5-v2`, `yourname-stage5-v3`, and so on based on completed runs.

                Review `TARGET_METRICS`, `SUGGESTED_MODULES`, `SMOKE_TEST_COMMANDS`, and `STAGE_COMMANDS` before you move into implementation.

                This cell also prints the important values you should check before running stage commands.

                It creates a small `RUN_STARTED.txt` file in the current run folder immediately, so you can confirm the output path is correct before training or evaluation finishes.
                """
            ).strip()
            + "\n"
        ),
        "smoke_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">4. Optional Smoke Test</h2>

                Use this cell only after you fill in `SMOKE_TEST_COMMANDS` in the config cell. Keep those commands tiny so you can catch path, dependency, and runtime issues before a full Stage 5 run.
                """
            ).strip()
            + "\n"
        ),
        "implementation_banner": implementation_banner(
            5,
            "create <code>src/keelnet/retrieve.py</code> or an equivalent retrieval module, then extend <code>src/keelnet/data.py</code>, <code>src/keelnet/evaluate.py</code>, and <code>src/keelnet/metrics.py</code>.",
            "retrieval failures and answering failures can be separated cleanly.",
            "re-proving the core mechanism, adaptive balancing, open-ended generation claims.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Run The Stage 5 Retrieval Commands

                This is the main Stage 5 implementation and run section. Everything before this point is setup, sync, or validation.

                Fill in `STAGE_COMMANDS` in the config cell with the actual commands for this stage. Start with one command, make sure the outputs land in `OUTPUT_ROOT`, then add the rest.
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                ## Save Notes And Review Artifacts

                This cell creates teammate-friendly note files inside the current run folder and lists the current artifacts. Update the generated notes as you learn what does and does not work in Stage 5: Retrieval-Grounded QA.
                """
            ).strip()
            + "\n"
        ),
        "final_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Final Check</h2>

                Stage 5 is not successful just because retrieval runs.

                Check all three:

                - retrieval failures and answer-generation failures are separated clearly in evaluation
                - end-to-end grounded behavior still looks meaningfully better than a naive retrieval baseline
                - the retrieval stack is stable enough for later balancing work, not just one lucky run

                After that, document whether the next bottleneck is retrieval recall, evidence ranking, or downstream answer control.
                """
            ).strip()
            + "\n"
        ),
        "share_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Share This Run</h2>

                This cell prints a minimal share-ready summary for teammates, saves it into the current run folder, and marks the run as completed so the next run becomes the next version.

                Update the metric lines after you review the outputs for this stage.
                """
            ).strip()
            + "\n"
        ),
    },
    {
        "path": REPO_ROOT / "stages/06-adaptive-constraint-balancing/notebooks/google-colab.ipynb",
        "branch": "stage/06-adaptive-constraint-balancing",
        "stage_number": 6,
        "stage_label": "Stage 6: Adaptive Constraint Balancing",
        "objective": "Learn or control the trade-off among answer quality, support, abstention, and confidence more intelligently than fixed weights alone.",
        "metrics": ["trade-off curve quality", "comparison against fixed-weight baselines", "robustness across operating points"],
        "hints": [
            "input: signals from the earlier stages",
            "output: adaptive balancing or decision control",
            "replace purely fixed balancing with an adaptive mechanism",
        ],
        "modules": ["keelnet.balance", "keelnet.evaluate", "keelnet.metrics"],
        "config_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

                Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage6-v1`, `yourname-stage6-v2`, `yourname-stage6-v3`, and so on based on completed runs.

                Review `TARGET_METRICS`, `SUGGESTED_MODULES`, `SMOKE_TEST_COMMANDS`, and `STAGE_COMMANDS` before you move into implementation.

                This cell also prints the important values you should check before running stage commands.

                It creates a small `RUN_STARTED.txt` file in the current run folder immediately, so you can confirm the output path is correct before training or evaluation finishes.
                """
            ).strip()
            + "\n"
        ),
        "smoke_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">4. Optional Smoke Test</h2>

                Use this cell only after you fill in `SMOKE_TEST_COMMANDS` in the config cell. Keep those commands tiny so you can catch path, dependency, and runtime issues before a full Stage 6 run.
                """
            ).strip()
            + "\n"
        ),
        "implementation_banner": implementation_banner(
            6,
            "create <code>src/keelnet/balance.py</code> or an equivalent adaptive-control module, then compare against fixed baselines in evaluation.",
            "adaptive balancing beats simpler fixed baselines on the main trade-offs.",
            "restarting earlier stages, open-ended generation claims, general hallucination claims.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Run The Stage 6 Balancing Commands

                This is the main Stage 6 implementation and run section. Everything before this point is setup, sync, or validation.

                Fill in `STAGE_COMMANDS` in the config cell with the actual commands for this stage. Start with one command, make sure the outputs land in `OUTPUT_ROOT`, then add the rest.
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                ## Save Notes And Review Artifacts

                This cell creates teammate-friendly note files inside the current run folder and lists the current artifacts. Update the generated notes as you learn what does and does not work in Stage 6: Adaptive Constraint Balancing.
                """
            ).strip()
            + "\n"
        ),
        "final_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Final Check</h2>

                Stage 6 is only interesting if it really beats simpler controls.

                Check all three:

                - the adaptive method beats fixed-weight or fixed-threshold baselines
                - the gain is not just threshold gaming or over-abstention
                - the operating-point story is simple enough to explain to teammates and in the report

                After that, save the comparison artifacts that make the baseline-versus-adaptive trade-off easy to defend.
                """
            ).strip()
            + "\n"
        ),
        "share_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Share This Run</h2>

                This cell prints a minimal share-ready summary for teammates, saves it into the current run folder, and marks the run as completed so the next run becomes the next version.

                Update the metric lines after you review the outputs for this stage.
                """
            ).strip()
            + "\n"
        ),
    },
]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def sync_stage_2() -> None:
    notebook = load_notebook(STAGE_2["path"])
    notebook["cells"][0]["source"] = source_lines(intro_markdown(STAGE_2["stage_label"], STAGE_2["stage_number"]))
    notebook["cells"][2]["source"] = source_lines(SETUP_MARKDOWN)
    notebook["cells"][4]["source"] = source_lines(STAGE_2["config_markdown"])
    notebook["cells"][6]["source"] = source_lines(VALIDATE_MARKDOWN)
    notebook["cells"][8]["source"] = source_lines(STAGE_2["smoke_markdown"])
    notebook["cells"][10]["source"] = source_lines(STAGE_2["implementation_banner"])
    notebook["cells"][11]["source"] = source_lines(STAGE_2["implementation_markdown"])
    notebook["cells"][13]["source"] = source_lines(STAGE_NOTE_TEMPLATE_MARKDOWN)
    notebook["cells"][14]["source"] = source_lines(STAGE_2["save_markdown"])
    notebook["cells"][16]["source"] = source_lines(STAGE_2["final_markdown"])
    notebook["cells"][17]["source"] = source_lines(STAGE_2["share_markdown"])
    save_notebook(STAGE_2["path"], notebook)


def sync_generic_stage(stage: dict) -> None:
    notebook = load_notebook(stage["path"])
    notebook["cells"][0]["source"] = source_lines(intro_markdown(stage["stage_label"], stage["stage_number"]))
    notebook["cells"][2]["source"] = source_lines(SETUP_MARKDOWN)
    notebook["cells"][3]["source"] = source_lines(setup_code(stage["branch"]))
    notebook["cells"][4]["source"] = source_lines(stage["config_markdown"])
    notebook["cells"][5]["source"] = source_lines(
        generic_config_code(
            stage_label=stage["stage_label"],
            stage_number=stage["stage_number"],
            objective=stage["objective"],
            metrics=stage["metrics"],
            hints=stage["hints"],
            modules=stage["modules"],
        )
    )
    notebook["cells"][6]["source"] = source_lines(VALIDATE_MARKDOWN)
    notebook["cells"][8]["source"] = source_lines(stage["smoke_markdown"])
    notebook["cells"][10]["source"] = source_lines(stage["implementation_banner"])
    notebook["cells"][11]["source"] = source_lines(stage["implementation_markdown"])
    notebook["cells"][13]["source"] = source_lines(STAGE_NOTE_TEMPLATE_MARKDOWN)
    notebook["cells"][14]["source"] = source_lines(stage["save_markdown"])
    notebook["cells"][15]["source"] = source_lines(generic_save_code())
    notebook["cells"][16]["source"] = source_lines(stage["final_markdown"])
    notebook["cells"][17]["source"] = source_lines(stage["share_markdown"])
    notebook["cells"][18]["source"] = source_lines(generic_share_code())
    save_notebook(stage["path"], notebook)


def main() -> None:
    sync_stage_2()
    for stage in GENERIC_STAGES:
        sync_generic_stage(stage)


if __name__ == "__main__":
    main()
