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
        - optional overrides: `KEELNET_REPO_DIR`, `KEELNET_PROJECT_DIR`, and `KEELNET_DRIVE_SYNC_DIR`
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
            import shutil
            import subprocess
            import sys


            GIT_REPO_URL = "https://github.com/naufalkmd/KeelNet.git"
            DEFAULT_GIT_BRANCH = {branch!r}
            GIT_BRANCH = os.environ.get("KEELNET_GIT_BRANCH", DEFAULT_GIT_BRANCH)
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


            def configure_drive_project_dir(project_storage_dir: Path) -> Path | None:
                if IS_HOSTED_COLAB:
                    print(f"Drive project dir: {{project_storage_dir}}")
                    return project_storage_dir.resolve()

                env_drive_dir = os.environ.get("KEELNET_DRIVE_SYNC_DIR")
                if not env_drive_dir:
                    print(
                        "Drive project dir: disabled "
                        "(set KEELNET_DRIVE_SYNC_DIR to a local Google Drive sync folder to mirror artifacts there)."
                    )
                    return None

                drive_project_dir = Path(env_drive_dir).expanduser().resolve()
                drive_project_dir.mkdir(parents=True, exist_ok=True)
                print(f"Drive project dir: {{drive_project_dir}}")
                return drive_project_dir


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
            DRIVE_PROJECT_DIR = configure_drive_project_dir(PROJECT_STORAGE_DIR)
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


            def mirror_output_root(output_root: Path) -> Path | None:
                if DRIVE_PROJECT_DIR is None:
                    print("Drive artifact mirror is disabled for this runtime.")
                    return None

                output_root = Path(output_root).expanduser().resolve()
                if not output_root.exists():
                    print(f"Nothing to mirror yet: {{output_root}}")
                    return None

                drive_project_dir = DRIVE_PROJECT_DIR.expanduser().resolve()
                try:
                    relative_output = output_root.relative_to(PROJECT_STORAGE_DIR.expanduser().resolve())
                except ValueError:
                    relative_output = Path("artifacts") / output_root.name
                drive_output_root = drive_project_dir / relative_output

                if output_root == drive_output_root:
                    print(f"Artifacts already stored in Drive: {{output_root}}")
                    return output_root

                drive_output_root.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(output_root, drive_output_root, dirs_exist_ok=True)
                print(f"Mirrored artifacts to Drive: {{drive_output_root}}")
                return drive_output_root
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
            DRIVE_PROJECT_DIR = Path(DRIVE_PROJECT_DIR).resolve() if DRIVE_PROJECT_DIR is not None else None

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
            print(f"Drive project dir: {{DRIVE_PROJECT_DIR}}")
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


def generic_implementation_code() -> str:
    return (
        dedent(
            """
            ran_stage = run_many(STAGE_COMMANDS, label="stage command")
            if not ran_stage:
                print("Fill in STAGE_COMMANDS in the config cell before running this section.")
            """
        ).strip()
        + "\n"
    )


def generic_save_code() -> str:
    return (
        dedent(
            """
            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)

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
                        "drive_project_dir": str(DRIVE_PROJECT_DIR) if DRIVE_PROJECT_DIR is not None else None,
                        "output_root": str(OUTPUT_ROOT),
                        "mirrored_output_root": str(mirrored_output_root) if mirrored_output_root is not None else None,
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
            if mirrored_output_root is not None:
                print(f"Drive mirror: {mirrored_output_root}")
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

            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)
            share_lines = [
                f"# {STAGE_TITLE} Share Note",
                "",
                f"- runtime mode: {RUNTIME_MODE}",
                f"- branch name: {CURRENT_BRANCH}",
                f"- RUN_NAME: {RUN_NAME}",
                *[f"- {metric}: <fill in after review>" for metric in TARGET_METRICS],
                f"- Output folder path: {OUTPUT_ROOT}",
            ]
            if mirrored_output_root is not None:
                share_lines.append(f"- Drive mirror path: {mirrored_output_root}")
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
            if mirrored_output_root is not None:
                mirror_output_root(OUTPUT_ROOT)
            """
    ).strip()
        + "\n"
    )


def stage3_config_code() -> str:
    return (
        dedent(
            """
            from pathlib import Path
            import json
            import re
            import subprocess
            import sys

            import torch

            REPO_DIR = Path(REPO_DIR).resolve()
            PROJECT_STORAGE_DIR = Path(PROJECT_STORAGE_DIR).resolve()
            DRIVE_PROJECT_DIR = Path(DRIVE_PROJECT_DIR).resolve() if DRIVE_PROJECT_DIR is not None else None

            # Change only this for each teammate. The notebook builds the stage name and next version automatically.
            AUTHOR_NAME = "naufal"
            RUN_BASENAME = f"{AUTHOR_NAME}-stage3"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "stage3_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            STAGE_TITLE = "Stage 3: Confidence Calibration"
            STAGE_OBJECTIVE = "Calibrate QA and verifier scores so they become trustworthy control signals instead of raw uncalibrated logits."
            TARGET_METRICS = [
                "QA ECE",
                "QA Adaptive ECE",
                "QA MCE",
                "QA Brier Score",
                "support ECE",
                "support Adaptive ECE",
                "support MCE",
                "support Brier Score",
                "downstream threshold stability",
                "reliability plots",
            ]
            IMPLEMENTATION_HINTS = [
                "input: Stage 1 abstain predictions and Stage 2 verifier scores",
                "output: calibrated confidence for answer / abstain and support decisions",
                "compare raw-score behavior against calibrated downstream control behavior",
            ]
            SUGGESTED_MODULES = ["keelnet.calibration", "keelnet.evaluate", "keelnet.metrics"]


            def completed_versions(root: Path, run_basename: str) -> list[int]:
                versions: list[int] = []
                if not root.exists():
                    return versions

                pattern = re.compile(rf"^{re.escape(run_basename)}-v(\\d+)$")
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    match = pattern.match(child.name)
                    if match and (child / COMPLETION_MARKER_NAME).exists():
                        versions.append(int(match.group(1)))
                return sorted(versions)


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            DEFAULT_STAGE1_ABSTAIN_DIR = Path("/content/KeelNet-local/artifacts/stage1_colab/codex-stage1-live-20260326-014652/abstain")
            DEFAULT_STAGE2_VERIFIER_DIR = Path("/content/KeelNet-local/artifacts/stage2_colab/naufal-stage2-v2/verifier")
            BASE_QA_MODEL_DIR = str(DEFAULT_STAGE1_ABSTAIN_DIR) if DEFAULT_STAGE1_ABSTAIN_DIR.exists() else None
            BASE_QA_MODE = "abstain"
            VERIFIER_MODEL_DIR = str(DEFAULT_STAGE2_VERIFIER_DIR) if DEFAULT_STAGE2_VERIFIER_DIR.exists() else None
            EVAL_BATCH_SIZE = 16
            MAX_EVAL_SAMPLES = None
            CALIBRATION_BINS = 10

            RUN_SMOKE_TEST = False
            SMOKE_TEST_EVAL_SAMPLES = 64

            if BASE_QA_MODEL_DIR is not None:
                BASE_QA_MODEL_DIR = Path(BASE_QA_MODEL_DIR).expanduser().resolve()
                if not BASE_QA_MODEL_DIR.exists():
                    raise FileNotFoundError(f"Base QA model dir not found: {BASE_QA_MODEL_DIR}")

            if VERIFIER_MODEL_DIR is not None:
                VERIFIER_MODEL_DIR = Path(VERIFIER_MODEL_DIR).expanduser().resolve()
                if not VERIFIER_MODEL_DIR.exists():
                    raise FileNotFoundError(f"Verifier model dir not found: {VERIFIER_MODEL_DIR}")

            CALIBRATION_EVAL = OUTPUT_ROOT / "calibration_eval.json"


            def maybe_add_arg(cmd: list[str], flag: str, value: object | None) -> None:
                if value is None:
                    return
                cmd.extend([flag, str(value)])


            def build_calibration_command(
                output_path: Path,
                *,
                max_eval_samples: int | None,
                qa_threshold_step: float,
                support_threshold_step: float,
                qa_temperature_step: float,
                support_temperature_step: float,
            ) -> list[str] | None:
                if BASE_QA_MODEL_DIR is None or VERIFIER_MODEL_DIR is None:
                    return None

                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.calibration",
                    "--qa-model-path",
                    str(BASE_QA_MODEL_DIR),
                    "--qa-mode",
                    BASE_QA_MODE,
                    "--verifier-model-path",
                    str(VERIFIER_MODEL_DIR),
                    "--output-path",
                    str(output_path),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--calibration-bins",
                    str(CALIBRATION_BINS),
                    "--qa-threshold-step",
                    str(qa_threshold_step),
                    "--support-threshold-step",
                    str(support_threshold_step),
                    "--qa-temperature-step",
                    str(qa_temperature_step),
                    "--support-temperature-step",
                    str(support_temperature_step),
                ]
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                return cmd


            smoke_command = build_calibration_command(
                OUTPUT_ROOT / "smoke-calibration-eval.json",
                max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                qa_threshold_step=1.0,
                support_threshold_step=0.1,
                qa_temperature_step=0.25,
                support_temperature_step=0.25,
            )
            stage_command = build_calibration_command(
                CALIBRATION_EVAL,
                max_eval_samples=MAX_EVAL_SAMPLES,
                qa_threshold_step=0.5,
                support_threshold_step=0.05,
                qa_temperature_step=0.05,
                support_temperature_step=0.05,
            )

            SMOKE_TEST_COMMANDS = [smoke_command] if smoke_command is not None else []
            STAGE_COMMANDS = [stage_command] if stage_command is not None else []

            RUN_MARKER_FILE.write_text(
                "\\n".join(
                    [
                        f"stage={STAGE_TITLE}",
                        f"run_name={RUN_NAME}",
                        f"run_version=v{RUN_VERSION}",
                        f"runtime_mode={RUNTIME_MODE}",
                        f"repo_dir={REPO_DIR}",
                        f"project_storage_dir={PROJECT_STORAGE_DIR}",
                        f"git_branch={CURRENT_BRANCH}",
                        "status=configured",
                        "note=This file is created when the config cell runs.",
                    ]
                )
                + "\\n",
                encoding="utf-8",
            )

            print(f"Runtime mode: {RUNTIME_MODE}")
            print(f"Repo dir: {REPO_DIR}")
            print(f"{PROJECT_STORAGE_LABEL}: {PROJECT_STORAGE_DIR}")
            print(f"Drive project dir: {DRIVE_PROJECT_DIR}")
            print(f"Artifacts root: {ARTIFACTS_ROOT}")
            print(f"Run basename: {RUN_BASENAME}")
            print(f"Run version: v{RUN_VERSION}")
            print(f"Run output dir: {OUTPUT_ROOT}")
            print(f"Run marker file: {RUN_MARKER_FILE}")
            print(f"Base QA model dir: {BASE_QA_MODEL_DIR}")
            print(f"Base QA mode: {BASE_QA_MODE}")
            print(f"Verifier model dir: {VERIFIER_MODEL_DIR}")
            print(f"Calibration eval file: {CALIBRATION_EVAL}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("Target metrics:", ", ".join(TARGET_METRICS))
            print("Suggested modules:", ", ".join(SUGGESTED_MODULES))

            if BASE_QA_MODEL_DIR is None or VERIFIER_MODEL_DIR is None:
                print("Set BASE_QA_MODEL_DIR and VERIFIER_MODEL_DIR before running Stage 3.")


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
                    print(f"No commands configured for {label} yet.")
                    return False

                for index, cmd in enumerate(commands, start=1):
                    print(f"\\n[{label} {index}/{len(commands)}]")
                    run(cmd)
                return True
            """
        ).strip()
        + "\n"
    )


def stage3_implementation_code() -> str:
    return (
        dedent(
            """
            if BASE_QA_MODEL_DIR is None or VERIFIER_MODEL_DIR is None:
                print("Set BASE_QA_MODEL_DIR and VERIFIER_MODEL_DIR in the config cell before running Stage 3.")
            else:
                run_many(STAGE_COMMANDS, label="stage command")
            """
        ).strip()
        + "\n"
    )


def stage3_save_code() -> str:
    return (
        dedent(
            """
            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)

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
                            "- One-sentence objective:",
                            "- Success condition:",
                            "- Out of scope:",
                            "",
                            "## Commands",
                            f"- Smoke test command(s): {SMOKE_TEST_COMMANDS}",
                            f"- Main command(s): {STAGE_COMMANDS}",
                            f"- Input artifacts or checkpoints: Stage 1 QA model at {BASE_QA_MODEL_DIR}, Stage 2 verifier at {VERIFIER_MODEL_DIR}",
                            f"- Output files to inspect: {CALIBRATION_EVAL}",
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
                        "output_root": str(OUTPUT_ROOT),
                        "drive_project_dir": str(DRIVE_PROJECT_DIR) if DRIVE_PROJECT_DIR is not None else None,
                        "mirrored_output_root": str(mirrored_output_root) if mirrored_output_root is not None else None,
                        "base_qa_model_dir": str(BASE_QA_MODEL_DIR) if BASE_QA_MODEL_DIR is not None else None,
                        "base_qa_mode": BASE_QA_MODE,
                        "verifier_model_dir": str(VERIFIER_MODEL_DIR) if VERIFIER_MODEL_DIR is not None else None,
                        "calibration_eval": str(CALIBRATION_EVAL),
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
            if mirrored_output_root is not None:
                print(f"Drive mirror: {mirrored_output_root}")
            print("Current files under OUTPUT_ROOT:")
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                print(path)
            """
        ).strip()
        + "\n"
    )


def stage3_share_code() -> str:
    return (
        dedent(
            """
            from pathlib import Path
            from datetime import datetime, timezone

            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)
            metric_lines = [f"- {metric}: <fill in after review>" for metric in TARGET_METRICS]
            plot_paths = {}
            if CALIBRATION_EVAL.exists():
                calibration_results = json.loads(CALIBRATION_EVAL.read_text(encoding="utf-8"))
                qa_raw = calibration_results["dev_qa_raw_metrics"]
                qa_calibrated = calibration_results["dev_qa_calibrated_metrics"]
                support_raw = calibration_results["dev_support_raw_metrics"]
                support_calibrated = calibration_results["dev_support_calibrated_metrics"]
                plot_paths = calibration_results.get("reliability_plot_paths", {})
                metric_lines = [
                    f"- QA ECE (dev): {qa_raw['ece']:.4f} -> {qa_calibrated['ece']:.4f}",
                    f"- QA Adaptive ECE (dev): {qa_raw['adaptive_ece']:.4f} -> {qa_calibrated['adaptive_ece']:.4f}",
                    f"- QA MCE (dev): {qa_raw['mce']:.4f} -> {qa_calibrated['mce']:.4f}",
                    f"- QA Brier (dev): {qa_raw['brier_score']:.4f} -> {qa_calibrated['brier_score']:.4f}",
                    f"- QA threshold gap (dev): {qa_raw['threshold_gap']:.4f} -> {qa_calibrated['threshold_gap']:.4f}",
                    f"- Support ECE (dev): {support_raw['ece']:.4f} -> {support_calibrated['ece']:.4f}",
                    f"- Support Adaptive ECE (dev): {support_raw['adaptive_ece']:.4f} -> {support_calibrated['adaptive_ece']:.4f}",
                    f"- Support MCE (dev): {support_raw['mce']:.4f} -> {support_calibrated['mce']:.4f}",
                    f"- Support Brier (dev): {support_raw['brier_score']:.4f} -> {support_calibrated['brier_score']:.4f}",
                    f"- Support threshold gap (dev): {support_raw['threshold_gap']:.4f} -> {support_calibrated['threshold_gap']:.4f}",
                    f"- QA temperature: {calibration_results['qa_temperature']:.2f}",
                    f"- Support temperature: {calibration_results['support_temperature']:.2f}",
                    f"- QA threshold: {calibration_results['qa_selected_threshold']:.2f}",
                    f"- Support threshold: {calibration_results['support_selected_threshold']:.2f}",
                ]
                for plot_key, plot_path in plot_paths.items():
                    metric_lines.append(f"- Reliability plot ({plot_key}): {plot_path}")

            share_lines = [
                f"# {STAGE_TITLE} Share Note",
                "",
                f"- runtime mode: {RUNTIME_MODE}",
                f"- branch name: {CURRENT_BRANCH}",
                f"- RUN_NAME: {RUN_NAME}",
                *metric_lines,
                f"- Output folder path: {OUTPUT_ROOT}",
            ]
            if mirrored_output_root is not None:
                share_lines.append(f"- Drive mirror path: {mirrored_output_root}")
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
            print(f"Saved share note: {SHARE_NOTE_FILE}")
            print(f"Saved completion marker: {COMPLETION_MARKER_FILE}")
            if mirrored_output_root is not None:
                mirror_output_root(OUTPUT_ROOT)

            if plot_paths:
                try:
                    from IPython.display import Image, display

                    for plot_key in ("qa_dev", "support_dev"):
                        plot_path = plot_paths.get(plot_key)
                        if plot_path and Path(plot_path).exists():
                            print(f"Displaying {plot_key}: {plot_path}")
                            display(Image(filename=plot_path))
                except Exception as exc:
                    print(f"Could not display reliability plots inline: {exc}")
            """
        ).strip()
        + "\n"
    )


def stage4_config_code() -> str:
    return (
        dedent(
            """
            from pathlib import Path
            import json
            import re
            import subprocess
            import sys

            import torch

            REPO_DIR = Path(REPO_DIR).resolve()
            PROJECT_STORAGE_DIR = Path(PROJECT_STORAGE_DIR).resolve()
            DRIVE_PROJECT_DIR = Path(DRIVE_PROJECT_DIR).resolve() if DRIVE_PROJECT_DIR is not None else None

            # Change only this for each teammate. The notebook builds the stage name and next version automatically.
            AUTHOR_NAME = "naufal"
            RUN_BASENAME = f"{AUTHOR_NAME}-stage4"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "stage4_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            STAGE_TITLE = "Stage 4: Unsupported-Confidence Control"
            STAGE_OBJECTIVE = "Use calibrated QA and support signals to reduce confident unsupported answers without collapsing usefulness."
            TARGET_METRICS = [
                "unsupported-answer rate",
                "supported-answer rate",
                "answer F1",
                "abstain F1",
                "answer rate",
            ]
            IMPLEMENTATION_HINTS = [
                "input: Stage 3 calibrated QA and support confidence signals",
                "output: a fixed constrained controller with explicit support and QA gates",
                "beat the Stage 2 gated baseline under an unsupported-answer budget",
            ]
            SUGGESTED_MODULES = ["keelnet.control", "keelnet.calibration", "keelnet.metrics"]


            def completed_versions(root: Path, run_basename: str) -> list[int]:
                versions: list[int] = []
                if not root.exists():
                    return versions

                pattern = re.compile(rf"^{re.escape(run_basename)}-v(\\d+)$")
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    match = pattern.match(child.name)
                    if match and (child / COMPLETION_MARKER_NAME).exists():
                        versions.append(int(match.group(1)))
                return sorted(versions)


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            DEFAULT_STAGE3_CALIBRATION_EVAL = Path("/content/KeelNet-local/artifacts/stage3_colab/naufal-stage3-v1/calibration_eval.json")
            CALIBRATION_EVAL_PATH = str(DEFAULT_STAGE3_CALIBRATION_EVAL) if DEFAULT_STAGE3_CALIBRATION_EVAL.exists() else None
            EVAL_BATCH_SIZE = 16
            MAX_EVAL_SAMPLES = None
            MAX_UNSUPPORTED_ANSWER_RATE = 20.0
            SUPPORT_THRESHOLD_MIN = 0.40
            SUPPORT_THRESHOLD_MAX = 0.80
            SUPPORT_THRESHOLD_STEP = 0.05
            QA_THRESHOLD_MIN = 0.40
            QA_THRESHOLD_MAX = 0.80
            QA_THRESHOLD_STEP = 0.05
            JOINT_THRESHOLD_MIN = 0.45
            JOINT_THRESHOLD_MAX = 0.85
            JOINT_THRESHOLD_STEP = 0.05
            ALPHA_MIN = 0.50
            ALPHA_MAX = 0.90
            ALPHA_STEP = 0.10

            RUN_SMOKE_TEST = False
            SMOKE_TEST_EVAL_SAMPLES = 64

            if CALIBRATION_EVAL_PATH is not None:
                CALIBRATION_EVAL_PATH = Path(CALIBRATION_EVAL_PATH).expanduser().resolve()
                if not CALIBRATION_EVAL_PATH.exists():
                    raise FileNotFoundError(f"Calibration eval file not found: {CALIBRATION_EVAL_PATH}")

            CONTROL_EVAL = OUTPUT_ROOT / "control_eval.json"


            def maybe_add_arg(cmd: list[str], flag: str, value: object | None) -> None:
                if value is None:
                    return
                cmd.extend([flag, str(value)])


            def build_control_command(
                output_path: Path,
                *,
                max_eval_samples: int | None,
                support_threshold_step: float,
                qa_threshold_step: float,
                joint_threshold_step: float,
                alpha_step: float,
            ) -> list[str] | None:
                if CALIBRATION_EVAL_PATH is None:
                    return None

                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.control",
                    "--calibration-path",
                    str(CALIBRATION_EVAL_PATH),
                    "--output-path",
                    str(output_path),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                    "--support-threshold-min",
                    str(SUPPORT_THRESHOLD_MIN),
                    "--support-threshold-max",
                    str(SUPPORT_THRESHOLD_MAX),
                    "--support-threshold-step",
                    str(support_threshold_step),
                    "--qa-threshold-min",
                    str(QA_THRESHOLD_MIN),
                    "--qa-threshold-max",
                    str(QA_THRESHOLD_MAX),
                    "--qa-threshold-step",
                    str(qa_threshold_step),
                    "--joint-threshold-min",
                    str(JOINT_THRESHOLD_MIN),
                    "--joint-threshold-max",
                    str(JOINT_THRESHOLD_MAX),
                    "--joint-threshold-step",
                    str(joint_threshold_step),
                    "--alpha-min",
                    str(ALPHA_MIN),
                    "--alpha-max",
                    str(ALPHA_MAX),
                    "--alpha-step",
                    str(alpha_step),
                ]
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                return cmd


            smoke_command = build_control_command(
                OUTPUT_ROOT / "smoke-control-eval.json",
                max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                support_threshold_step=0.10,
                qa_threshold_step=0.10,
                joint_threshold_step=0.10,
                alpha_step=0.20,
            )
            stage_command = build_control_command(
                CONTROL_EVAL,
                max_eval_samples=MAX_EVAL_SAMPLES,
                support_threshold_step=SUPPORT_THRESHOLD_STEP,
                qa_threshold_step=QA_THRESHOLD_STEP,
                joint_threshold_step=JOINT_THRESHOLD_STEP,
                alpha_step=ALPHA_STEP,
            )

            SMOKE_TEST_COMMANDS = [smoke_command] if smoke_command is not None else []
            STAGE_COMMANDS = [stage_command] if stage_command is not None else []

            RUN_MARKER_FILE.write_text(
                "\\n".join(
                    [
                        f"stage={STAGE_TITLE}",
                        f"run_name={RUN_NAME}",
                        f"run_version=v{RUN_VERSION}",
                        f"runtime_mode={RUNTIME_MODE}",
                        f"repo_dir={REPO_DIR}",
                        f"project_storage_dir={PROJECT_STORAGE_DIR}",
                        f"git_branch={CURRENT_BRANCH}",
                        f"calibration_eval_path={CALIBRATION_EVAL_PATH}",
                        "status=configured",
                        "note=This file is created when the config cell runs.",
                    ]
                )
                + "\\n",
                encoding="utf-8",
            )

            print(f"Runtime mode: {RUNTIME_MODE}")
            print(f"Repo dir: {REPO_DIR}")
            print(f"{PROJECT_STORAGE_LABEL}: {PROJECT_STORAGE_DIR}")
            print(f"Drive project dir: {DRIVE_PROJECT_DIR}")
            print(f"Artifacts root: {ARTIFACTS_ROOT}")
            print(f"Run basename: {RUN_BASENAME}")
            print(f"Run version: v{RUN_VERSION}")
            print(f"Run output dir: {OUTPUT_ROOT}")
            print(f"Run marker file: {RUN_MARKER_FILE}")
            print(f"Calibration eval path: {CALIBRATION_EVAL_PATH}")
            print(f"Control eval file: {CONTROL_EVAL}")
            print(f"Max unsupported-answer rate: {MAX_UNSUPPORTED_ANSWER_RATE}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("Target metrics:", ", ".join(TARGET_METRICS))
            print("Suggested modules:", ", ".join(SUGGESTED_MODULES))

            if CALIBRATION_EVAL_PATH is None:
                print("Set CALIBRATION_EVAL_PATH before running Stage 4.")


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
                    print(f"No commands configured for {label} yet.")
                    return False

                for index, cmd in enumerate(commands, start=1):
                    print(f"\\n[{label} {index}/{len(commands)}]")
                    run(cmd)
                return True
            """
        ).strip()
        + "\n"
    )


def stage4_implementation_code() -> str:
    return (
        dedent(
            """
            if CALIBRATION_EVAL_PATH is None:
                print("Set CALIBRATION_EVAL_PATH in the config cell before running Stage 4.")
            else:
                run_many(STAGE_COMMANDS, label="stage command")
            """
        ).strip()
        + "\n"
    )


def stage4_save_code() -> str:
    return (
        dedent(
            """
            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)

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
                            "- One-sentence objective:",
                            "- Unsupported-answer budget:",
                            "- Out of scope:",
                            "",
                            "## Commands",
                            f"- Smoke test command(s): {SMOKE_TEST_COMMANDS}",
                            f"- Main command(s): {STAGE_COMMANDS}",
                            f"- Input artifacts or checkpoints: Stage 3 calibration eval at {CALIBRATION_EVAL_PATH}",
                            f"- Output files to inspect: {CONTROL_EVAL}",
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
                        "output_root": str(OUTPUT_ROOT),
                        "drive_project_dir": str(DRIVE_PROJECT_DIR) if DRIVE_PROJECT_DIR is not None else None,
                        "mirrored_output_root": str(mirrored_output_root) if mirrored_output_root is not None else None,
                        "calibration_eval_path": str(CALIBRATION_EVAL_PATH) if CALIBRATION_EVAL_PATH is not None else None,
                        "control_eval": str(CONTROL_EVAL),
                        "target_metrics": TARGET_METRICS,
                        "suggested_modules": SUGGESTED_MODULES,
                        "max_unsupported_answer_rate": MAX_UNSUPPORTED_ANSWER_RATE,
                    },
                    indent=2,
                )
                + "\\n",
                encoding="utf-8",
            )

            print(f"Notes template: {RUN_NOTES_FILE}")
            print(f"Run summary: {RUN_SUMMARY_FILE}")
            if mirrored_output_root is not None:
                print(f"Drive mirror: {mirrored_output_root}")
            print("Current files under OUTPUT_ROOT:")
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                print(path)
            """
        ).strip()
        + "\n"
    )


def stage4_share_code() -> str:
    return (
        dedent(
            """
            from datetime import datetime, timezone

            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)
            metric_lines = [f"- {metric}: <fill in after review>" for metric in TARGET_METRICS]
            if CONTROL_EVAL.exists():
                control_results = json.loads(CONTROL_EVAL.read_text(encoding="utf-8"))
                stage2_metrics = control_results["stage2_gated_dev_metrics"]
                stage4_metrics = control_results["control_dev_metrics"]
                stage2_mix = control_results["stage2_gated_dev_mix"]
                stage4_mix = control_results["control_dev_mix"]
                config = control_results["selected_config"]
                metric_lines = [
                    f"- Unsupported-answer rate (dev): {stage2_metrics['unsupported_answer_rate']:.2f} -> {stage4_metrics['unsupported_answer_rate']:.2f}",
                    f"- Answerable F1 (dev): {stage2_metrics['answerable_f1']:.2f} -> {stage4_metrics['answerable_f1']:.2f}",
                    f"- Overall F1 (dev): {stage2_metrics['overall_f1']:.2f} -> {stage4_metrics['overall_f1']:.2f}",
                    f"- Abstain F1 (dev): {stage2_metrics['abstain_f1']:.2f} -> {stage4_metrics['abstain_f1']:.2f}",
                    f"- Supported-answer rate (among answers, dev): {stage2_mix['supported_answer_rate']:.2f} -> {stage4_mix['supported_answer_rate']:.2f}",
                    f"- Answer rate (dev): {stage2_mix['answer_rate']:.2f} -> {stage4_mix['answer_rate']:.2f}",
                    f"- Selected support threshold: {config['support_threshold']:.2f}",
                    f"- Selected QA threshold: {config['qa_threshold']:.2f}",
                    f"- Selected joint threshold: {config['joint_threshold']:.2f}",
                    f"- Selected alpha: {config['alpha']:.2f}",
                    f"- Max unsupported-answer rate budget: {control_results['max_unsupported_answer_rate']:.2f}",
                ]

            share_lines = [
                f"# {STAGE_TITLE} Share Note",
                "",
                f"- runtime mode: {RUNTIME_MODE}",
                f"- branch name: {CURRENT_BRANCH}",
                f"- RUN_NAME: {RUN_NAME}",
                *metric_lines,
                f"- Output folder path: {OUTPUT_ROOT}",
            ]
            if mirrored_output_root is not None:
                share_lines.append(f"- Drive mirror path: {mirrored_output_root}")
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
            print(f"Saved share note: {SHARE_NOTE_FILE}")
            print(f"Saved completion marker: {COMPLETION_MARKER_FILE}")
            if mirrored_output_root is not None:
                mirror_output_root(OUTPUT_ROOT)
            """
        ).strip()
        + "\n"
    )


def stage5_config_code() -> str:
    return (
        dedent(
            """
            from pathlib import Path
            import json
            import re
            import subprocess
            import sys

            import torch

            REPO_DIR = Path(REPO_DIR).resolve()
            PROJECT_STORAGE_DIR = Path(PROJECT_STORAGE_DIR).resolve()
            DRIVE_PROJECT_DIR = Path(DRIVE_PROJECT_DIR).resolve() if DRIVE_PROJECT_DIR is not None else None

            AUTHOR_NAME = "naufal"
            RUN_BASENAME = f"{AUTHOR_NAME}-stage5"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "stage5_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            STAGE_TITLE = "Stage 5: Support-Constrained Learning Comparison"
            STAGE_OBJECTIVE = "Compare the best modular pipeline against a direct support-constrained learning objective under matched conditions."
            TARGET_METRICS = [
                "overall F1",
                "answerable F1",
                "unsupported-answer rate",
                "supported-answer rate",
                "abstain F1",
            ]
            IMPLEMENTATION_HINTS = [
                "input: the same grounded QA split used in Stages 1 to 4",
                "output: a jointly trained answer-support-abstain learner",
                "compare directly against the strongest modular Stage 4 baseline when available",
            ]
            SUGGESTED_MODULES = ["keelnet.learn", "keelnet.metrics", "keelnet.train", "keelnet.evaluate"]


            def completed_versions(root: Path, run_basename: str) -> list[int]:
                versions: list[int] = []
                if not root.exists():
                    return versions

                pattern = re.compile(rf"^{re.escape(run_basename)}-v(\\d+)$")
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    match = pattern.match(child.name)
                    if match and (child / COMPLETION_MARKER_NAME).exists():
                        versions.append(int(match.group(1)))
                return sorted(versions)


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            DEFAULT_STAGE4_CONTROL_EVAL = Path("/content/KeelNet-local/artifacts/stage4_colab/naufal-stage4-v1/control_eval.json")
            MODULAR_BASELINE_EVAL_PATH = str(DEFAULT_STAGE4_CONTROL_EVAL) if DEFAULT_STAGE4_CONTROL_EVAL.exists() else None

            MODEL_NAME = "distilbert-base-uncased"
            NUM_TRAIN_EPOCHS = 3.0
            TRAIN_BATCH_SIZE = 8
            EVAL_BATCH_SIZE = 8
            LEARNING_RATE = 2e-5
            WEIGHT_DECAY = 0.01
            WARMUP_RATIO = 0.0
            KEEP_LOSS_WEIGHT = 1.0
            SUPPORT_LOSS_WEIGHT = 1.0
            KEEP_POSITIVE_WEIGHT = 1.0
            KEEP_NEGATIVE_WEIGHT = 2.0
            MAX_TRAIN_SAMPLES = None
            MAX_EVAL_SAMPLES = None
            MAX_UNSUPPORTED_ANSWER_RATE = 20.0
            KEEP_THRESHOLD_MIN = 0.05
            KEEP_THRESHOLD_MAX = 0.95
            KEEP_THRESHOLD_STEP = 0.05

            RUN_SMOKE_TEST = False
            SMOKE_TEST_TRAIN_SAMPLES = 256
            SMOKE_TEST_EVAL_SAMPLES = 128
            SMOKE_TEST_EPOCHS = 1.0

            LEARNER_DIR = OUTPUT_ROOT / "learner"
            LEARNER_EVAL = OUTPUT_ROOT / "learner_eval.json"

            if MODULAR_BASELINE_EVAL_PATH is not None:
                MODULAR_BASELINE_EVAL_PATH = Path(MODULAR_BASELINE_EVAL_PATH).expanduser().resolve()
                if not MODULAR_BASELINE_EVAL_PATH.exists():
                    raise FileNotFoundError(f"Baseline eval file not found: {MODULAR_BASELINE_EVAL_PATH}")


            def maybe_add_arg(cmd: list[str], flag: str, value: object | None) -> None:
                if value is None:
                    return
                cmd.extend([flag, str(value)])


            def build_train_command(
                output_dir: Path,
                *,
                max_train_samples: int | None,
                max_eval_samples: int | None,
                num_train_epochs: float,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.learn",
                    "train",
                    "--output-dir",
                    str(output_dir),
                    "--model-name",
                    MODEL_NAME,
                    "--num-train-epochs",
                    str(num_train_epochs),
                    "--train-batch-size",
                    str(TRAIN_BATCH_SIZE),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--learning-rate",
                    str(LEARNING_RATE),
                    "--weight-decay",
                    str(WEIGHT_DECAY),
                    "--warmup-ratio",
                    str(WARMUP_RATIO),
                    "--keep-loss-weight",
                    str(KEEP_LOSS_WEIGHT),
                    "--support-loss-weight",
                    str(SUPPORT_LOSS_WEIGHT),
                    "--keep-positive-weight",
                    str(KEEP_POSITIVE_WEIGHT),
                    "--keep-negative-weight",
                    str(KEEP_NEGATIVE_WEIGHT),
                ]
                maybe_add_arg(cmd, "--max-train-samples", max_train_samples)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                return cmd


            def build_eval_command(
                model_path: Path,
                output_path: Path,
                *,
                max_eval_samples: int | None,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.learn",
                    "evaluate",
                    "--model-path",
                    str(model_path),
                    "--output-path",
                    str(output_path),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--keep-threshold-min",
                    str(KEEP_THRESHOLD_MIN),
                    "--keep-threshold-max",
                    str(KEEP_THRESHOLD_MAX),
                    "--keep-threshold-step",
                    str(KEEP_THRESHOLD_STEP),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                ]
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                return cmd


            smoke_model_dir = OUTPUT_ROOT / "smoke-learner"
            smoke_eval_path = OUTPUT_ROOT / "smoke-learner-eval.json"
            smoke_train_command = build_train_command(
                smoke_model_dir,
                max_train_samples=SMOKE_TEST_TRAIN_SAMPLES,
                max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                num_train_epochs=SMOKE_TEST_EPOCHS,
            )
            smoke_eval_command = build_eval_command(
                smoke_model_dir,
                smoke_eval_path,
                max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
            )
            stage_train_command = build_train_command(
                LEARNER_DIR,
                max_train_samples=MAX_TRAIN_SAMPLES,
                max_eval_samples=MAX_EVAL_SAMPLES,
                num_train_epochs=NUM_TRAIN_EPOCHS,
            )
            stage_eval_command = build_eval_command(
                LEARNER_DIR,
                LEARNER_EVAL,
                max_eval_samples=MAX_EVAL_SAMPLES,
            )

            SMOKE_TEST_COMMANDS = [smoke_train_command, smoke_eval_command]
            STAGE_COMMANDS = [stage_train_command, stage_eval_command]

            RUN_MARKER_FILE.write_text(
                "\\n".join(
                    [
                        f"stage={STAGE_TITLE}",
                        f"run_name={RUN_NAME}",
                        f"run_version=v{RUN_VERSION}",
                        f"runtime_mode={RUNTIME_MODE}",
                        f"repo_dir={REPO_DIR}",
                        f"project_storage_dir={PROJECT_STORAGE_DIR}",
                        f"git_branch={CURRENT_BRANCH}",
                        f"baseline_eval_path={MODULAR_BASELINE_EVAL_PATH}",
                        "status=configured",
                        "note=This file is created when the config cell runs.",
                    ]
                )
                + "\\n",
                encoding="utf-8",
            )

            print(f"Runtime mode: {RUNTIME_MODE}")
            print(f"Repo dir: {REPO_DIR}")
            print(f"{PROJECT_STORAGE_LABEL}: {PROJECT_STORAGE_DIR}")
            print(f"Drive project dir: {DRIVE_PROJECT_DIR}")
            print(f"Artifacts root: {ARTIFACTS_ROOT}")
            print(f"Run basename: {RUN_BASENAME}")
            print(f"Run version: v{RUN_VERSION}")
            print(f"Run output dir: {OUTPUT_ROOT}")
            print(f"Run marker file: {RUN_MARKER_FILE}")
            print(f"Baseline eval path: {MODULAR_BASELINE_EVAL_PATH}")
            print(f"Learner dir: {LEARNER_DIR}")
            print(f"Learner eval file: {LEARNER_EVAL}")
            print(f"Model name: {MODEL_NAME}")
            print(f"Max unsupported-answer rate: {MAX_UNSUPPORTED_ANSWER_RATE}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
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
                    print(f"No commands configured for {label} yet.")
                    return False

                for index, cmd in enumerate(commands, start=1):
                    print(f"\\n[{label} {index}/{len(commands)}]")
                    run(cmd)
                return True
            """
        ).strip()
        + "\n"
    )


def stage5_implementation_code() -> str:
    return (
        dedent(
            """
            run_many(STAGE_COMMANDS, label="stage command")
            """
        ).strip()
        + "\n"
    )


def stage5_save_code() -> str:
    return (
        dedent(
            """
            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)

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
                            f"- Modular baseline eval path: {MODULAR_BASELINE_EVAL_PATH}",
                            f"- Output files to inspect: {LEARNER_EVAL}",
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
                        "output_root": str(OUTPUT_ROOT),
                        "drive_project_dir": str(DRIVE_PROJECT_DIR) if DRIVE_PROJECT_DIR is not None else None,
                        "mirrored_output_root": str(mirrored_output_root) if mirrored_output_root is not None else None,
                        "learner_dir": str(LEARNER_DIR),
                        "learner_eval": str(LEARNER_EVAL),
                        "baseline_eval_path": str(MODULAR_BASELINE_EVAL_PATH) if MODULAR_BASELINE_EVAL_PATH is not None else None,
                        "target_metrics": TARGET_METRICS,
                        "suggested_modules": SUGGESTED_MODULES,
                        "max_unsupported_answer_rate": MAX_UNSUPPORTED_ANSWER_RATE,
                    },
                    indent=2,
                )
                + "\\n",
                encoding="utf-8",
            )

            print(f"Notes template: {RUN_NOTES_FILE}")
            print(f"Run summary: {RUN_SUMMARY_FILE}")
            if mirrored_output_root is not None:
                print(f"Drive mirror: {mirrored_output_root}")
            print("Current files under OUTPUT_ROOT:")
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                print(path)
            """
        ).strip()
        + "\n"
    )


def stage5_share_code() -> str:
    return (
        dedent(
            """
            from datetime import datetime, timezone

            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)
            metric_lines = [f"- {metric}: <fill in after review>" for metric in TARGET_METRICS]
            if LEARNER_EVAL.exists():
                learner_results = json.loads(LEARNER_EVAL.read_text(encoding="utf-8"))
                dev_metrics = learner_results["dev_metrics"]
                dev_mix = learner_results["dev_mix"]
                metric_lines = [
                    f"- Overall F1 (dev): {dev_metrics['overall_f1']:.2f}",
                    f"- Answerable F1 (dev): {dev_metrics['answerable_f1']:.2f}",
                    f"- Unsupported-answer rate (dev): {dev_metrics['unsupported_answer_rate']:.2f}",
                    f"- Abstain F1 (dev): {dev_metrics['abstain_f1']:.2f}",
                    f"- Supported-answer rate (among answers, dev): {dev_mix['supported_answer_rate']:.2f}",
                    f"- Answer rate (dev): {dev_mix['answer_rate']:.2f}",
                    f"- Selected keep threshold: {learner_results['selected_keep_threshold']:.2f}",
                    f"- Max unsupported-answer rate budget: {learner_results['max_unsupported_answer_rate']:.2f}",
                ]
                if MODULAR_BASELINE_EVAL_PATH is not None and MODULAR_BASELINE_EVAL_PATH.exists():
                    baseline_results = json.loads(MODULAR_BASELINE_EVAL_PATH.read_text(encoding="utf-8"))
                    baseline_metrics = baseline_results.get("control_dev_metrics")
                    baseline_mix = baseline_results.get("control_dev_mix")
                    if baseline_metrics is not None and baseline_mix is not None:
                        metric_lines = [
                            f"- Overall F1 (dev): {baseline_metrics['overall_f1']:.2f} -> {dev_metrics['overall_f1']:.2f}",
                            f"- Answerable F1 (dev): {baseline_metrics['answerable_f1']:.2f} -> {dev_metrics['answerable_f1']:.2f}",
                            f"- Unsupported-answer rate (dev): {baseline_metrics['unsupported_answer_rate']:.2f} -> {dev_metrics['unsupported_answer_rate']:.2f}",
                            f"- Abstain F1 (dev): {baseline_metrics['abstain_f1']:.2f} -> {dev_metrics['abstain_f1']:.2f}",
                            f"- Supported-answer rate (among answers, dev): {baseline_mix['supported_answer_rate']:.2f} -> {dev_mix['supported_answer_rate']:.2f}",
                            f"- Answer rate (dev): {baseline_mix['answer_rate']:.2f} -> {dev_mix['answer_rate']:.2f}",
                            f"- Selected keep threshold: {learner_results['selected_keep_threshold']:.2f}",
                            f"- Max unsupported-answer rate budget: {learner_results['max_unsupported_answer_rate']:.2f}",
                        ]

            share_lines = [
                f"# {STAGE_TITLE} Share Note",
                "",
                f"- runtime mode: {RUNTIME_MODE}",
                f"- branch name: {CURRENT_BRANCH}",
                f"- RUN_NAME: {RUN_NAME}",
                *metric_lines,
                f"- Output folder path: {OUTPUT_ROOT}",
            ]
            if mirrored_output_root is not None:
                share_lines.append(f"- Drive mirror path: {mirrored_output_root}")
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
            print(f"Saved share note: {SHARE_NOTE_FILE}")
            print(f"Saved completion marker: {COMPLETION_MARKER_FILE}")
            if mirrored_output_root is not None:
                mirror_output_root(OUTPUT_ROOT)
            """
        ).strip()
        + "\n"
    )


STAGE_2 = {
    "path": REPO_ROOT / "stages/02-evidence-support-verification/notebooks/stage-02-evidence-support-verification-colab.ipynb",
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
            <h2 style="color: #15803d;">Save Notes And Review Artifacts</h2>

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
        "path": REPO_ROOT / "stages/03-confidence-calibration/notebooks/stage-03-confidence-calibration-colab.ipynb",
        "branch": "main",
        "stage_number": 3,
        "stage_label": "Stage 3: Confidence Calibration",
        "objective": "Calibrate QA and verifier scores so they become trustworthy control signals instead of raw uncalibrated logits.",
        "metrics": ["ECE", "Brier Score", "correlation between confidence and supported correctness", "downstream threshold stability"],
        "hints": [
            "input: Stage 1 abstain predictions and Stage 2 verifier scores",
            "output: calibrated confidence for answer / abstain and support decisions",
            "compare raw-score behavior against calibrated downstream control behavior",
        ],
        "modules": ["keelnet.calibration", "keelnet.evaluate", "keelnet.metrics"],
        "config_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

                Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage3-v1`, `yourname-stage3-v2`, `yourname-stage3-v3`, and so on based on completed runs.

                Point `BASE_QA_MODEL_DIR` at a finished Stage 1 abstain model and `VERIFIER_MODEL_DIR` at a finished Stage 2 verifier before running the main Stage 3 commands.

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

                Use this cell to run a tiny Stage 3 calibration evaluation before the full pass. Keep it small so you can catch path, dependency, and runtime issues before a full Stage 3 run.
                """
            ).strip()
            + "\n"
        ),
        "implementation_banner": implementation_banner(
            3,
            "point <code>BASE_QA_MODEL_DIR</code> and <code>VERIFIER_MODEL_DIR</code> at finished Stage 1 and Stage 2 outputs, then run the calibration evaluation below.",
            "raw and calibrated QA and support metrics are saved to <code>CALIBRATION_EVAL</code> so the next stage can use them as real control signals.",
            "retrieval, adaptive balancing, unrelated architecture changes.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Run The Stage 3 Calibration Commands

                This is the main Stage 3 implementation and run section. Everything before this point is setup, sync, or validation.

                This section evaluates raw versus calibrated QA and verifier confidence on the saved Stage 1 and Stage 2 checkpoints.
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Save Notes And Review Artifacts</h2>

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
                - the calibrated signals make downstream gating or threshold selection less brittle than the raw scores

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

                If `CALIBRATION_EVAL` exists, this cell reads the saved metrics automatically.
                """
            ).strip()
            + "\n"
        ),
        "config_code": stage3_config_code(),
        "implementation_code": stage3_implementation_code(),
        "save_code": stage3_save_code(),
        "share_code": stage3_share_code(),
    },
    {
        "path": REPO_ROOT / "stages/04-unsupported-confidence-control/notebooks/stage-04-unsupported-confidence-control-colab.ipynb",
        "branch": "main",
        "stage_number": 4,
        "stage_label": "Stage 4: Unsupported-Confidence Control",
        "objective": "Use calibrated QA and support signals to reduce confident unsupported answers without collapsing usefulness.",
        "metrics": ["unsupported-answer rate", "supported-answer rate", "answer F1", "abstain F1"],
        "hints": [
            "input: answer, support signal, calibrated confidence signal",
            "output: a fixed control rule or penalty that beats the Stage 2 gated baseline",
            "add an explicit control mechanism for confident unsupported outputs",
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
            "create <code>src/keelnet/control.py</code> or an equivalent control module, then combine calibrated answer, support, and confidence signals in evaluation.",
            "confident unsupported answers go down and the result clearly beats the permissive Stage 2 gate.",
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
                <h2 style="color: #15803d;">Save Notes And Review Artifacts</h2>

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
                - the behavior change comes from the intended control mechanism, not just over-abstention or another tiny delta

                After that, record a few failure cases so Stage 5 can compare this controller against a direct support-constrained learner under matched conditions.
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
        "config_code": stage4_config_code(),
        "implementation_code": stage4_implementation_code(),
        "save_code": stage4_save_code(),
        "share_code": stage4_share_code(),
    },
    {
        "path": REPO_ROOT / "stages/05-retrieval-grounded-qa/notebooks/stage-05-support-constrained-learning-colab.ipynb",
        "branch": "main",
        "stage_number": 5,
        "stage_label": "Stage 5: Support-Constrained Learning Comparison",
        "objective": "Compare the best modular pipeline against a direct support-constrained learning objective under matched conditions.",
        "metrics": ["overall F1", "unsupported-answer rate", "supported-answer rate", "abstain F1", "comparison against modular baseline"],
        "hints": [
            "input: the same grounded QA examples and metrics used in Stages 1 to 4",
            "compare modular post-hoc control against a direct support-constrained objective",
            "output: a matched baseline-vs-new-learning comparison, not another isolated module",
            "treat retrieval as later realism work, not the proof step here",
        ],
        "modules": ["keelnet.learn", "keelnet.train", "keelnet.evaluate", "keelnet.metrics"],
        "notes_markdown": (
            dedent(
                """
                ## Stage Notes

                ### Goal

                Compare the strongest modular grounded-QA pipeline against a direct support-constrained learning objective under matched conditions.

                ### Scope

                - input: the same grounded QA examples used in Stages 1 to 4
                - output: answer or `ABSTAIN`
                - comparison: strongest modular baseline versus support-constrained learner

                ### Main Change

                - replace post-hoc-only control with a direct support-constrained learning objective

                ### Main Metrics

                - overall `F1`
                - answerable `F1`
                - unsupported-answer rate
                - supported-answer rate
                - abstain `F1`

                ### What This Stage Validates

                - changing the learning target adds value beyond modular tuning
                - the gain is real under matched data, backbone, and evaluation conditions

                ### Handoff Condition

                Do not move to the next stage until the support-constrained learner either clearly beats the strongest modular baseline or gives a crisp failure diagnosis you can explain.
                """
            ).strip()
            + "\n"
        ),
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
            "review the auto-built <code>keelnet.learn</code> train and evaluate commands, then compare the resulting learner against the strongest modular Stage 4 baseline.",
            "the new learning objective beats or clearly sharpens the best modular baseline under the same metrics and data split.",
            "retrieval realism, adaptive balancing, open-ended generation claims.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Run The Stage 5 Constrained-Learning Comparison

                This is the main Stage 5 implementation and run section. Everything before this point is setup, sync, or validation.

                The config cell now builds the default Stage 5 train and evaluate commands automatically. Review those commands, adjust the hyperparameters if needed, and then run this section.
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Save Notes And Review Artifacts</h2>

                This cell creates teammate-friendly note files inside the current run folder and lists the current artifacts. Update the generated notes as you learn what does and does not work in Stage 5: Support-Constrained Learning Comparison.
                """
            ).strip()
            + "\n"
        ),
        "final_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Final Check</h2>

                Stage 5 is not successful just because a new objective trains.

                Check all three:

                - the constrained-learning variant is compared against the strongest modular baseline under matched conditions
                - unsupported answers go down without collapsing answer quality through trivial refusal
                - the gain comes from the new objective itself, not just threshold changes hidden inside the comparison

                After that, document whether the next bottleneck is objective design, representation quality, or later adaptive balancing.
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
        "config_code": stage5_config_code(),
        "implementation_code": stage5_implementation_code(),
        "save_code": stage5_save_code(),
        "share_code": stage5_share_code(),
    },
    {
        "path": REPO_ROOT / "stages/06-adaptive-constraint-balancing/notebooks/stage-06-adaptive-constraint-balancing-colab.ipynb",
        "branch": "main",
        "stage_number": 6,
        "stage_label": "Stage 6: Adaptive Constraint Balancing",
        "objective": "Only if needed, beat the strongest earlier fixed or constrained baseline by adapting the trade-off among answer quality, support, abstention, and confidence.",
        "metrics": ["trade-off curve quality", "comparison against strongest earlier baselines", "robustness across operating points"],
        "hints": [
            "input: the calibrated and controlled signals from the earlier stages",
            "output: adaptive balancing or decision control",
            "replace the strongest earlier baseline, not a straw-man baseline",
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
            "create <code>src/keelnet/balance.py</code> or an equivalent adaptive-control module, then compare against the strongest earlier Stage 4 or Stage 5 baseline in evaluation.",
            "adaptive balancing beats the strongest earlier baseline on the main trade-offs.",
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
                <h2 style="color: #15803d;">Save Notes And Review Artifacts</h2>

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

                - the adaptive method beats the strongest fixed or constrained baseline you already trust
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
    if "notes_markdown" in stage:
        notebook["cells"][1]["source"] = source_lines(stage["notes_markdown"])
    notebook["cells"][2]["source"] = source_lines(SETUP_MARKDOWN)
    notebook["cells"][3]["source"] = source_lines(setup_code(stage["branch"]))
    notebook["cells"][4]["source"] = source_lines(stage["config_markdown"])
    notebook["cells"][5]["source"] = source_lines(
        stage.get(
            "config_code",
            generic_config_code(
                stage_label=stage["stage_label"],
                stage_number=stage["stage_number"],
                objective=stage["objective"],
                metrics=stage["metrics"],
                hints=stage["hints"],
                modules=stage["modules"],
            ),
        )
    )
    notebook["cells"][6]["source"] = source_lines(VALIDATE_MARKDOWN)
    notebook["cells"][8]["source"] = source_lines(stage["smoke_markdown"])
    notebook["cells"][10]["source"] = source_lines(stage["implementation_banner"])
    notebook["cells"][11]["source"] = source_lines(stage["implementation_markdown"])
    notebook["cells"][12]["source"] = source_lines(stage.get("implementation_code", generic_implementation_code()))
    notebook["cells"][13]["source"] = source_lines(STAGE_NOTE_TEMPLATE_MARKDOWN)
    notebook["cells"][14]["source"] = source_lines(stage["save_markdown"])
    notebook["cells"][15]["source"] = source_lines(stage.get("save_code", generic_save_code()))
    notebook["cells"][16]["source"] = source_lines(stage["final_markdown"])
    notebook["cells"][17]["source"] = source_lines(stage["share_markdown"])
    notebook["cells"][18]["source"] = source_lines(stage.get("share_code", generic_share_code()))
    save_notebook(stage["path"], notebook)


def main() -> None:
    sync_stage_2()
    for stage in GENERIC_STAGES:
        sync_generic_stage(stage)


if __name__ == "__main__":
    main()
