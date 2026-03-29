from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERIC_NOTEBOOK_TEMPLATE = (
    REPO_ROOT
    / "stages/06-adaptive-constraint-balancing/notebooks/stage-06-adaptive-constraint-balancing-colab.ipynb"
)


def source_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


def indented_block(text: str, prefix: str = "            ") -> str:
    return indent(text.rstrip(), prefix)


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

        Reliability reminder:

        - hosted Colab defaults to a fresh clone on each setup run to avoid stale or conflicted repo state
        - set `FORCE_FRESH_CLONE = False` in the setup cell or `KEELNET_FORCE_FRESH_CLONE=0` if you intentionally want to reuse `/content/KeelNet`
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

VALIDATE_CODE = (
    dedent(
        """
        from pathlib import Path
        import re


        CONFLICT_MARKER_PATTERN = re.compile(r"^(<<<<<<<|=======|>>>>>>>)", re.MULTILINE)


        def _problem_excerpt(path: Path, *, marker_line: int, context_lines: int = 2) -> str:
            lines = path.read_text(encoding="utf-8").splitlines()
            start = max(1, marker_line - context_lines)
            end = min(len(lines), marker_line + context_lines)
            excerpt_lines = ["--- Start of problematic section ---"]
            for line_number in range(start, end + 1):
                excerpt_lines.append(f"{line_number}: {lines[line_number - 1]}")
            excerpt_lines.append("--- End of problematic section ---")
            return "\\n".join(excerpt_lines)


        def assert_repo_has_no_conflict_markers(repo_dir: Path) -> None:
            candidate_paths: list[Path] = []
            for relative_path in ("pyproject.toml",):
                candidate = repo_dir / relative_path
                if candidate.exists():
                    candidate_paths.append(candidate)

            for subdir in ("src", "tests"):
                root = repo_dir / subdir
                if root.exists():
                    candidate_paths.extend(sorted(root.rglob("*.py")))

            for path in candidate_paths:
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue

                match = CONFLICT_MARKER_PATTERN.search(text)
                if match is None:
                    continue

                marker_line = text.count("\\n", 0, match.start()) + 1
                print(
                    f"ERROR: Found potential Git merge conflict markers in {path} around line {marker_line}:"
                )
                print(_problem_excerpt(path, marker_line=marker_line))
                raise RuntimeError(
                    "Please resolve these Git merge conflicts in the file directly, "
                    "or rerun the setup cell to refresh /content/KeelNet, then rerun this cell."
                )


        assert_repo_has_no_conflict_markers(REPO_DIR)
        run([sys.executable, "-m", "unittest", "discover", "-s", str(REPO_DIR / "tests")])
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
        - Executed notebook archive target
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

NOTEBOOK_ARCHIVE_CONFIG_CODE = (
    dedent(
        """
        NOTEBOOK_ARCHIVE_DIR = OUTPUT_ROOT / "executed-notebook"
        NOTEBOOK_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        EXECUTED_NOTEBOOK_PATH = NOTEBOOK_ARCHIVE_DIR / f"{RUN_NAME}-executed.ipynb"
        EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE = NOTEBOOK_ARCHIVE_DIR / "README-save-executed-notebook.txt"
        EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE.write_text(
            "\\n".join(
                [
                    "Keep the canonical stage notebook in the repo as a clean template.",
                    "",
                    "The save/share cells try to archive the live notebook automatically when Colab exposes it.",
                    "",
                    "If automatic capture is unavailable, preserve a meaningful run with outputs manually:",
                    "1. in Colab, use File -> Download .ipynb or File -> Save a copy in Drive",
                    f"2. save the exported file as: {EXECUTED_NOTEBOOK_PATH.name}",
                    f"3. place that file inside: {NOTEBOOK_ARCHIVE_DIR}",
                    "",
                    "Anything saved in this run folder stays outside the template-sync path and will mirror to Drive when KEELNET_DRIVE_SYNC_DIR is enabled.",
                ]
            )
            + "\\n",
            encoding="utf-8",
        )
        AUTO_SAVED_EXECUTED_NOTEBOOK_PATH = None


        def save_executed_notebook_snapshot() -> Path | None:
            global AUTO_SAVED_EXECUTED_NOTEBOOK_PATH

            try:
                from google.colab import _message
            except ImportError:
                print(
                    "Automatic executed-notebook capture is unavailable in this runtime. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
                return None

            try:
                response = _message.blocking_request("get_ipynb", timeout_sec=30)
            except TypeError:
                response = _message.blocking_request("get_ipynb", request="", timeout_sec=30)
            except Exception as exc:
                print(
                    "Automatic executed-notebook capture failed. "
                    f"Save the notebook manually to {EXECUTED_NOTEBOOK_PATH}. Error: {exc}"
                )
                return None

            notebook = response.get("ipynb") if isinstance(response, dict) else None
            if notebook is None:
                print(
                    "Automatic executed-notebook capture did not return notebook content. "
                    f"Save the notebook manually to {EXECUTED_NOTEBOOK_PATH}."
                )
                return None

            metadata = notebook.setdefault("metadata", {})
            keelnet_metadata = metadata.setdefault("keelnet", {})
            keelnet_metadata["run_name"] = RUN_NAME
            keelnet_metadata["executed_notebook_target"] = str(EXECUTED_NOTEBOOK_PATH)
            source_notebook_name = response.get("path") if isinstance(response, dict) else None
            if source_notebook_name:
                keelnet_metadata["source_notebook_name"] = source_notebook_name

            EXECUTED_NOTEBOOK_PATH.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\\n",
                encoding="utf-8",
            )
            AUTO_SAVED_EXECUTED_NOTEBOOK_PATH = EXECUTED_NOTEBOOK_PATH
            print(f"Saved executed notebook snapshot: {EXECUTED_NOTEBOOK_PATH}")
            return EXECUTED_NOTEBOOK_PATH
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
            FORCE_FRESH_CLONE = True

            env_force_fresh_clone = os.environ.get("KEELNET_FORCE_FRESH_CLONE")
            if env_force_fresh_clone is not None:
                FORCE_FRESH_CLONE = env_force_fresh_clone.strip().lower() not in {"0", "false", "no"}


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
                if FORCE_FRESH_CLONE and local_repo_dir.exists():
                    os.chdir("/content")
                    print(f"Removing existing hosted Colab repo for a fresh clone: {{local_repo_dir}}")
                    shutil.rmtree(local_repo_dir)

                if (local_repo_dir / ".git").exists():
                    run_setup(["git", "fetch", "origin"], cwd=local_repo_dir)
                    run_setup(["git", "checkout", GIT_BRANCH], cwd=local_repo_dir)
                    run_setup(["git", "pull", "--ff-only", "origin", GIT_BRANCH], cwd=local_repo_dir)
                else:
                    run_setup(["git", "clone", "--branch", GIT_BRANCH, GIT_REPO_URL, str(local_repo_dir)])
                return local_repo_dir.resolve()


            PROJECT_STORAGE_DIR = configure_project_storage()
            DRIVE_PROJECT_DIR = configure_drive_project_dir(PROJECT_STORAGE_DIR)
            configure_hf_token()
            REPO_DIR = ensure_repo().resolve()
            os.chdir(REPO_DIR)
            print(f"Runtime mode: {{RUNTIME_MODE}}")
            print(f"Runtime repo dir: {{REPO_DIR}}")
            print(f"Force fresh clone: {{FORCE_FRESH_CLONE}}")
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

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

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
            print(f"Executed notebook target: {{EXECUTED_NOTEBOOK_PATH}}")
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
    ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
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
            captured_notebook_path = save_executed_notebook_snapshot()
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
                            f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
                        "executed_notebook_dir": str(NOTEBOOK_ARCHIVE_DIR),
                        "executed_notebook_target": str(EXECUTED_NOTEBOOK_PATH),
                        "executed_notebook_saved": captured_notebook_path is not None,
                        "executed_notebook_instructions_file": str(EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE),
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
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

            captured_notebook_path = save_executed_notebook_snapshot()
            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)
            share_lines = [
                f"# {STAGE_TITLE} Share Note",
                "",
                f"- runtime mode: {RUNTIME_MODE}",
                f"- branch name: {CURRENT_BRANCH}",
                f"- RUN_NAME: {RUN_NAME}",
                *[f"- {metric}: <fill in after review>" for metric in TARGET_METRICS],
                f"- Output folder path: {OUTPUT_ROOT}",
                f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
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


            def completed_run_dirs(root: Path, *, run_prefix: str | None = None) -> list[Path]:
                if not root.exists():
                    return []

                pattern = re.compile(rf"^{re.escape(run_prefix)}-v(\\d+)$") if run_prefix is not None else None
                runs: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if not (child / COMPLETION_MARKER_NAME).exists():
                        continue
                    if pattern is not None and not pattern.match(child.name):
                        continue
                    runs.append(child)
                return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name))


            def default_upstream_path(stage_folder: str, relative_path: str, *, preferred_run_prefix: str | None = None) -> str | None:
                stage_root = PROJECT_STORAGE_DIR / "artifacts" / stage_folder
                ordered_runs: list[Path] = []

                if preferred_run_prefix is not None:
                    ordered_runs.extend(reversed(completed_run_dirs(stage_root, run_prefix=preferred_run_prefix)))

                for run_dir in reversed(completed_run_dirs(stage_root)):
                    if run_dir not in ordered_runs:
                        ordered_runs.append(run_dir)

                relative = Path(relative_path)
                for run_dir in ordered_runs:
                    candidate = run_dir / relative
                    if candidate.exists():
                        return str(candidate)
                return None


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

            DEFAULT_STAGE1_ABSTAIN_DIR = default_upstream_path(
                "stage1_colab",
                "abstain",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage1",
            )
            DEFAULT_STAGE2_VERIFIER_DIR = default_upstream_path(
                "stage2_colab",
                "verifier",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage2",
            )
            BASE_QA_MODEL_DIR = DEFAULT_STAGE1_ABSTAIN_DIR
            BASE_QA_MODE = "abstain"
            VERIFIER_MODEL_DIR = DEFAULT_STAGE2_VERIFIER_DIR
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
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
        ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
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
            captured_notebook_path = save_executed_notebook_snapshot()
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
                            f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
                        "executed_notebook_dir": str(NOTEBOOK_ARCHIVE_DIR),
                        "executed_notebook_target": str(EXECUTED_NOTEBOOK_PATH),
                        "executed_notebook_saved": captured_notebook_path is not None,
                        "executed_notebook_instructions_file": str(EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE),
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
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

            captured_notebook_path = save_executed_notebook_snapshot()
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
                f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
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


            def completed_run_dirs(root: Path, *, run_prefix: str | None = None) -> list[Path]:
                if not root.exists():
                    return []

                pattern = re.compile(rf"^{re.escape(run_prefix)}-v(\\d+)$") if run_prefix is not None else None
                runs: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if not (child / COMPLETION_MARKER_NAME).exists():
                        continue
                    if pattern is not None and not pattern.match(child.name):
                        continue
                    runs.append(child)
                return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name))


            def default_upstream_path(stage_folder: str, relative_path: str, *, preferred_run_prefix: str | None = None) -> str | None:
                stage_root = PROJECT_STORAGE_DIR / "artifacts" / stage_folder
                ordered_runs: list[Path] = []

                if preferred_run_prefix is not None:
                    ordered_runs.extend(reversed(completed_run_dirs(stage_root, run_prefix=preferred_run_prefix)))

                for run_dir in reversed(completed_run_dirs(stage_root)):
                    if run_dir not in ordered_runs:
                        ordered_runs.append(run_dir)

                relative = Path(relative_path)
                for run_dir in ordered_runs:
                    candidate = run_dir / relative
                    if candidate.exists():
                        return str(candidate)
                return None


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

            DEFAULT_STAGE3_CALIBRATION_EVAL = default_upstream_path(
                "stage3_colab",
                "calibration_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage3",
            )
            CALIBRATION_EVAL_PATH = DEFAULT_STAGE3_CALIBRATION_EVAL
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
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
        ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
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
            captured_notebook_path = save_executed_notebook_snapshot()
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
                            f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
                        "executed_notebook_dir": str(NOTEBOOK_ARCHIVE_DIR),
                        "executed_notebook_target": str(EXECUTED_NOTEBOOK_PATH),
                        "executed_notebook_saved": captured_notebook_path is not None,
                        "executed_notebook_instructions_file": str(EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE),
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
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

            captured_notebook_path = save_executed_notebook_snapshot()
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
                f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
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


            def completed_run_dirs(root: Path, *, run_prefix: str | None = None) -> list[Path]:
                if not root.exists():
                    return []

                pattern = re.compile(rf"^{re.escape(run_prefix)}-v(\\d+)$") if run_prefix is not None else None
                runs: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if not (child / COMPLETION_MARKER_NAME).exists():
                        continue
                    if pattern is not None and not pattern.match(child.name):
                        continue
                    runs.append(child)
                return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name))


            def default_upstream_path(stage_folder: str, relative_path: str, *, preferred_run_prefix: str | None = None) -> str | None:
                stage_root = PROJECT_STORAGE_DIR / "artifacts" / stage_folder
                ordered_runs: list[Path] = []

                if preferred_run_prefix is not None:
                    ordered_runs.extend(reversed(completed_run_dirs(stage_root, run_prefix=preferred_run_prefix)))

                for run_dir in reversed(completed_run_dirs(stage_root)):
                    if run_dir not in ordered_runs:
                        ordered_runs.append(run_dir)

                relative = Path(relative_path)
                for run_dir in ordered_runs:
                    candidate = run_dir / relative
                    if candidate.exists():
                        return str(candidate)
                return None


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

            DEFAULT_STAGE4_CONTROL_EVAL = default_upstream_path(
                "stage4_colab",
                "control_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage4",
            )
            MODULAR_BASELINE_EVAL_PATH = DEFAULT_STAGE4_CONTROL_EVAL

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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
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
        ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
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
            captured_notebook_path = save_executed_notebook_snapshot()
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
                            f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
                        "executed_notebook_dir": str(NOTEBOOK_ARCHIVE_DIR),
                        "executed_notebook_target": str(EXECUTED_NOTEBOOK_PATH),
                        "executed_notebook_saved": captured_notebook_path is not None,
                        "executed_notebook_instructions_file": str(EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE),
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
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

            captured_notebook_path = save_executed_notebook_snapshot()
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
                f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
            print(f"Saved share note: {SHARE_NOTE_FILE}")
            print(f"Saved completion marker: {COMPLETION_MARKER_FILE}")
            if mirrored_output_root is not None:
                mirror_output_root(OUTPUT_ROOT)
            """
        ).strip()
        + "\n"
    )


def stage6_config_code() -> str:
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
            RUN_BASENAME = f"{AUTHOR_NAME}-stage6"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "stage6_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            STAGE_TITLE = "Stage 6: Adaptive Constraint Balancing"
            STAGE_OBJECTIVE = "Only if needed, beat the strongest earlier fixed or constrained baseline by adapting Stage 5 candidate selection under the unsupported-answer budget, while keeping the final SQuAD validation split untouched for final testing."
            TARGET_METRICS = [
                "overall F1",
                "answerable F1",
                "unsupported-answer rate",
                "supported-answer rate",
                "abstain F1",
            ]
            IMPLEMENTATION_HINTS = [
                "input: candidate spans and keep/support signals from the Stage 5 learner",
                "output: a learned adaptive controller that can abstain under budget",
                "compare against the strongest available Stage 4 or Stage 5 baseline",
            ]
            SUGGESTED_MODULES = ["keelnet.balance", "keelnet.learn", "keelnet.metrics"]


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


            def completed_run_dirs(root: Path, *, run_prefix: str | None = None) -> list[Path]:
                if not root.exists():
                    return []

                pattern = re.compile(rf"^{re.escape(run_prefix)}-v(\\d+)$") if run_prefix is not None else None
                runs: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if not (child / COMPLETION_MARKER_NAME).exists():
                        continue
                    if pattern is not None and not pattern.match(child.name):
                        continue
                    runs.append(child)
                return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name))


            def default_upstream_path(stage_folder: str, relative_path: str, *, preferred_run_prefix: str | None = None) -> str | None:
                stage_root = PROJECT_STORAGE_DIR / "artifacts" / stage_folder
                ordered_runs: list[Path] = []

                if preferred_run_prefix is not None:
                    ordered_runs.extend(reversed(completed_run_dirs(stage_root, run_prefix=preferred_run_prefix)))

                for run_dir in reversed(completed_run_dirs(stage_root)):
                    if run_dir not in ordered_runs:
                        ordered_runs.append(run_dir)

                relative = Path(relative_path)
                for run_dir in ordered_runs:
                    candidate = run_dir / relative
                    if candidate.exists():
                        return str(candidate)
                return None


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

            DEFAULT_STAGE5_MODEL_DIR = default_upstream_path(
                "stage5_colab",
                "learner",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage5",
            )
            DEFAULT_STAGE5_EVAL = default_upstream_path(
                "stage5_colab",
                "learner_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage5",
            )
            DEFAULT_STAGE4_CONTROL_EVAL = default_upstream_path(
                "stage4_colab",
                "control_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage4",
            )
            STAGE5_MODEL_DIR = DEFAULT_STAGE5_MODEL_DIR
            COMPARISON_BASELINE_EVAL_PATH = DEFAULT_STAGE4_CONTROL_EVAL or DEFAULT_STAGE5_EVAL

            TRAIN_BATCH_SIZE = 64
            EVAL_BATCH_SIZE = 8
            CONTROLLER_EPOCHS = 10
            CONTROLLER_LR = 1e-3
            CONTROLLER_WEIGHT_DECAY = 0.01
            HIDDEN_SIZE = 32
            DROPOUT = 0.10
            MAX_CANDIDATES_PER_EXAMPLE = 6
            MAX_CANDIDATES_PER_FEATURE = 3
            POSITIVE_WEIGHT = 1.0
            NEGATIVE_WEIGHT_INIT = 1.0
            NEGATIVE_WEIGHT_MIN = 0.5
            NEGATIVE_WEIGHT_MAX = 6.0
            ADAPTIVE_WEIGHT_STEP = 0.5
            HARD_NEGATIVE_WEIGHT = 1.5
            MAX_TRAIN_SAMPLES = None
            MAX_EVAL_SAMPLES = None
            CLEAN_SPLITTING = True
            MAX_UNSUPPORTED_ANSWER_RATE = 20.0
            CANDIDATE_THRESHOLD_MIN = 0.05
            CANDIDATE_THRESHOLD_MAX = 0.95
            CANDIDATE_THRESHOLD_STEP = 0.05

            RUN_SMOKE_TEST = False
            SMOKE_TEST_TRAIN_SAMPLES = 256
            SMOKE_TEST_EVAL_SAMPLES = 128
            SMOKE_TEST_EPOCHS = 2

            BALANCER_DIR = OUTPUT_ROOT / "balancer"
            BALANCER_EVAL = OUTPUT_ROOT / "balance_eval.json"

            if STAGE5_MODEL_DIR is not None:
                STAGE5_MODEL_DIR = Path(STAGE5_MODEL_DIR).expanduser().resolve()
                if not STAGE5_MODEL_DIR.exists():
                    raise FileNotFoundError(f"Stage 5 model directory not found: {STAGE5_MODEL_DIR}")

            if COMPARISON_BASELINE_EVAL_PATH is not None:
                COMPARISON_BASELINE_EVAL_PATH = Path(COMPARISON_BASELINE_EVAL_PATH).expanduser().resolve()
                if not COMPARISON_BASELINE_EVAL_PATH.exists():
                    raise FileNotFoundError(f"Baseline eval file not found: {COMPARISON_BASELINE_EVAL_PATH}")


            def maybe_add_arg(cmd: list[str], flag: str, value: object | None) -> None:
                if value is None:
                    return
                cmd.extend([flag, str(value)])


            def maybe_add_flag(cmd: list[str], flag: str, enabled: bool) -> None:
                if enabled:
                    cmd.append(flag)


            def build_train_command(
                stage5_model_dir: Path,
                output_dir: Path,
                *,
                max_train_samples: int | None,
                max_eval_samples: int | None,
                num_train_epochs: int,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.balance",
                    "train",
                    "--stage5-model-path",
                    str(stage5_model_dir),
                    "--output-dir",
                    str(output_dir),
                    "--train-batch-size",
                    str(TRAIN_BATCH_SIZE),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--learning-rate",
                    str(CONTROLLER_LR),
                    "--weight-decay",
                    str(CONTROLLER_WEIGHT_DECAY),
                    "--num-train-epochs",
                    str(num_train_epochs),
                    "--hidden-size",
                    str(HIDDEN_SIZE),
                    "--dropout",
                    str(DROPOUT),
                    "--max-candidates-per-example",
                    str(MAX_CANDIDATES_PER_EXAMPLE),
                    "--max-candidates-per-feature",
                    str(MAX_CANDIDATES_PER_FEATURE),
                    "--positive-weight",
                    str(POSITIVE_WEIGHT),
                    "--negative-weight-init",
                    str(NEGATIVE_WEIGHT_INIT),
                    "--negative-weight-min",
                    str(NEGATIVE_WEIGHT_MIN),
                    "--negative-weight-max",
                    str(NEGATIVE_WEIGHT_MAX),
                    "--adaptive-weight-step",
                    str(ADAPTIVE_WEIGHT_STEP),
                    "--hard-negative-weight",
                    str(HARD_NEGATIVE_WEIGHT),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                ]
                maybe_add_flag(cmd, "--clean-splitting", CLEAN_SPLITTING)
                maybe_add_arg(cmd, "--max-train-samples", max_train_samples)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                maybe_add_arg(cmd, "--max-test-samples", max_eval_samples if CLEAN_SPLITTING else None)
                return cmd


            def build_eval_command(
                controller_path: Path,
                stage5_model_dir: Path,
                output_path: Path,
                *,
                max_eval_samples: int | None,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.balance",
                    "evaluate",
                    "--controller-path",
                    str(controller_path),
                    "--stage5-model-path",
                    str(stage5_model_dir),
                    "--output-path",
                    str(output_path),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--candidate-threshold-min",
                    str(CANDIDATE_THRESHOLD_MIN),
                    "--candidate-threshold-max",
                    str(CANDIDATE_THRESHOLD_MAX),
                    "--candidate-threshold-step",
                    str(CANDIDATE_THRESHOLD_STEP),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                ]
                maybe_add_flag(cmd, "--clean-splitting", CLEAN_SPLITTING)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                maybe_add_arg(cmd, "--max-test-samples", max_eval_samples if CLEAN_SPLITTING else None)
                return cmd


            if STAGE5_MODEL_DIR is None:
                SMOKE_TEST_COMMANDS = []
                STAGE_COMMANDS = []
            else:
                smoke_model_dir = OUTPUT_ROOT / "smoke-balancer"
                smoke_eval_path = OUTPUT_ROOT / "smoke-balance-eval.json"
                smoke_train_command = build_train_command(
                    STAGE5_MODEL_DIR,
                    smoke_model_dir,
                    max_train_samples=SMOKE_TEST_TRAIN_SAMPLES,
                    max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                    num_train_epochs=SMOKE_TEST_EPOCHS,
                )
                smoke_eval_command = build_eval_command(
                    smoke_model_dir,
                    STAGE5_MODEL_DIR,
                    smoke_eval_path,
                    max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                )
                stage_train_command = build_train_command(
                    STAGE5_MODEL_DIR,
                    BALANCER_DIR,
                    max_train_samples=MAX_TRAIN_SAMPLES,
                    max_eval_samples=MAX_EVAL_SAMPLES,
                    num_train_epochs=CONTROLLER_EPOCHS,
                )
                stage_eval_command = build_eval_command(
                    BALANCER_DIR,
                    STAGE5_MODEL_DIR,
                    BALANCER_EVAL,
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
                        f"stage5_model_dir={STAGE5_MODEL_DIR}",
                        f"comparison_baseline_eval_path={COMPARISON_BASELINE_EVAL_PATH}",
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            print(f"Run marker file: {RUN_MARKER_FILE}")
            print(f"Stage 5 model dir: {STAGE5_MODEL_DIR}")
            print(f"Comparison baseline eval path: {COMPARISON_BASELINE_EVAL_PATH}")
            print(f"Balancer dir: {BALANCER_DIR}")
            print(f"Balancer eval file: {BALANCER_EVAL}")
            print(f"Max unsupported-answer rate: {MAX_UNSUPPORTED_ANSWER_RATE}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("Target metrics:", ", ".join(TARGET_METRICS))
            print("Suggested modules:", ", ".join(SUGGESTED_MODULES))

            if STAGE5_MODEL_DIR is None:
                print("Set STAGE5_MODEL_DIR before running Stage 6.")


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
        ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
        + "\n"
    )


def stage6_implementation_code() -> str:
    return (
        dedent(
            """
            if STAGE5_MODEL_DIR is None:
                print("Set STAGE5_MODEL_DIR in the config cell before running Stage 6.")
            else:
                run_many(STAGE_COMMANDS, label="stage command")
            """
        ).strip()
        + "\n"
    )


def stage6_save_code() -> str:
    return (
        dedent(
            """
            captured_notebook_path = save_executed_notebook_snapshot()
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
                            f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
                            f"- Stage 5 model dir: {STAGE5_MODEL_DIR}",
                            f"- Comparison baseline eval path: {COMPARISON_BASELINE_EVAL_PATH}",
                            f"- Output files to inspect: {BALANCER_EVAL}",
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
                        "executed_notebook_dir": str(NOTEBOOK_ARCHIVE_DIR),
                        "executed_notebook_target": str(EXECUTED_NOTEBOOK_PATH),
                        "executed_notebook_saved": captured_notebook_path is not None,
                        "executed_notebook_instructions_file": str(EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE),
                        "stage5_model_dir": str(STAGE5_MODEL_DIR) if STAGE5_MODEL_DIR is not None else None,
                        "balancer_dir": str(BALANCER_DIR),
                        "balancer_eval": str(BALANCER_EVAL),
                        "comparison_baseline_eval_path": str(COMPARISON_BASELINE_EVAL_PATH) if COMPARISON_BASELINE_EVAL_PATH is not None else None,
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
            if mirrored_output_root is not None:
                print(f"Drive mirror: {mirrored_output_root}")
            print("Current files under OUTPUT_ROOT:")
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                print(path)
            """
        ).strip()
        + "\n"
    )


def stage6_share_code() -> str:
    return (
        dedent(
            """
            from datetime import datetime, timezone

            captured_notebook_path = save_executed_notebook_snapshot()
            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)
            metric_lines = [f"- {metric}: <fill in after review>" for metric in TARGET_METRICS]
            if BALANCER_EVAL.exists():
                balance_results = json.loads(BALANCER_EVAL.read_text(encoding="utf-8"))
                eval_split_label = balance_results.get("final_eval_split", "dev")
                result_metrics = (
                    balance_results.get("final_metrics")
                    or balance_results.get(f"{eval_split_label}_metrics")
                    or balance_results["dev_metrics"]
                )
                result_mix = (
                    balance_results.get("final_mix")
                    or balance_results.get(f"{eval_split_label}_mix")
                    or balance_results["dev_mix"]
                )
                metric_lines = [
                    f"- Overall F1 ({eval_split_label}): {result_metrics['overall_f1']:.2f}",
                    f"- Answerable F1 ({eval_split_label}): {result_metrics['answerable_f1']:.2f}",
                    f"- Unsupported-answer rate ({eval_split_label}): {result_metrics['unsupported_answer_rate']:.2f}",
                    f"- Abstain F1 ({eval_split_label}): {result_metrics['abstain_f1']:.2f}",
                    f"- Supported-answer rate (among answers, {eval_split_label}): {result_mix['supported_answer_rate']:.2f}",
                    f"- Answer rate ({eval_split_label}): {result_mix['answer_rate']:.2f}",
                    f"- Selected candidate threshold: {balance_results['selected_candidate_threshold']:.2f}",
                    f"- Max unsupported-answer rate budget: {balance_results['max_unsupported_answer_rate']:.2f}",
                ]
                if COMPARISON_BASELINE_EVAL_PATH is not None and COMPARISON_BASELINE_EVAL_PATH.exists():
                    baseline_results = json.loads(COMPARISON_BASELINE_EVAL_PATH.read_text(encoding="utf-8"))
                    baseline_metrics = None
                    baseline_mix = None
                    baseline_split_label = None
                    if "final_metrics" in baseline_results and "final_mix" in baseline_results:
                        baseline_split_label = baseline_results.get("final_eval_split", "dev")
                        baseline_metrics = baseline_results["final_metrics"]
                        baseline_mix = baseline_results["final_mix"]
                    elif "control_dev_metrics" in baseline_results:
                        baseline_split_label = "dev"
                        baseline_metrics = baseline_results["control_dev_metrics"]
                        baseline_mix = baseline_results["control_dev_mix"]
                    elif "dev_metrics" in baseline_results:
                        baseline_split_label = "dev"
                        baseline_metrics = baseline_results["dev_metrics"]
                        baseline_mix = baseline_results["dev_mix"]

                    if (
                        baseline_metrics is not None
                        and baseline_mix is not None
                        and baseline_split_label == eval_split_label
                    ):
                        metric_lines = [
                            f"- Overall F1 ({eval_split_label}): {baseline_metrics['overall_f1']:.2f} -> {result_metrics['overall_f1']:.2f}",
                            f"- Answerable F1 ({eval_split_label}): {baseline_metrics['answerable_f1']:.2f} -> {result_metrics['answerable_f1']:.2f}",
                            f"- Unsupported-answer rate ({eval_split_label}): {baseline_metrics['unsupported_answer_rate']:.2f} -> {result_metrics['unsupported_answer_rate']:.2f}",
                            f"- Abstain F1 ({eval_split_label}): {baseline_metrics['abstain_f1']:.2f} -> {result_metrics['abstain_f1']:.2f}",
                            f"- Supported-answer rate (among answers, {eval_split_label}): {baseline_mix['supported_answer_rate']:.2f} -> {result_mix['supported_answer_rate']:.2f}",
                            f"- Answer rate ({eval_split_label}): {baseline_mix['answer_rate']:.2f} -> {result_mix['answer_rate']:.2f}",
                            f"- Selected candidate threshold: {balance_results['selected_candidate_threshold']:.2f}",
                            f"- Max unsupported-answer rate budget: {balance_results['max_unsupported_answer_rate']:.2f}",
                        ]
                    elif baseline_metrics is not None and baseline_mix is not None:
                        metric_lines.append(
                            f"- Baseline comparison skipped: baseline reports {baseline_split_label}, current run reports {eval_split_label}."
                        )

            share_lines = [
                f"# {STAGE_TITLE} Share Note",
                "",
                f"- runtime mode: {RUNTIME_MODE}",
                f"- branch name: {CURRENT_BRANCH}",
                f"- RUN_NAME: {RUN_NAME}",
                *metric_lines,
                f"- Output folder path: {OUTPUT_ROOT}",
                f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
            print(f"Saved share note: {SHARE_NOTE_FILE}")
            print(f"Saved completion marker: {COMPLETION_MARKER_FILE}")
            if mirrored_output_root is not None:
                mirror_output_root(OUTPUT_ROOT)
            """
        ).strip()
        + "\n"
    )


def stage7_config_code() -> str:
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
            RUN_BASENAME = f"{AUTHOR_NAME}-stage7"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "stage7_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            STAGE_TITLE = "Stage 7: Risk-Budgeted Action Learning"
            STAGE_OBJECTIVE = "Turn Stage 6 into an explicit utility-versus-risk action learner that chooses among answer candidates and abstain under an unsupported-answer budget, while keeping the final SQuAD validation split untouched for final testing."
            TARGET_METRICS = [
                "overall F1",
                "answerable F1",
                "unsupported-answer rate",
                "supported-answer rate",
                "answer rate",
                "abstain F1",
            ]
            IMPLEMENTATION_HINTS = [
                "input: Stage 5 candidate spans and the strongest available Stage 6 or Stage 4 baseline signals",
                "output: a risk-budgeted action learner over {candidate_1 ... candidate_k, abstain}",
                "success means a better utility-versus-groundedness trade-off, not only lower answer rate",
            ]
            SUGGESTED_MODULES = ["keelnet.action", "keelnet.balance", "keelnet.learn", "keelnet.metrics"]


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


            def completed_run_dirs(root: Path, *, run_prefix: str | None = None) -> list[Path]:
                if not root.exists():
                    return []

                pattern = re.compile(rf"^{re.escape(run_prefix)}-v(\\d+)$") if run_prefix is not None else None
                runs: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if not (child / COMPLETION_MARKER_NAME).exists():
                        continue
                    if pattern is not None and not pattern.match(child.name):
                        continue
                    runs.append(child)
                return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name))


            def default_upstream_path(stage_folder: str, relative_path: str, *, preferred_run_prefix: str | None = None) -> str | None:
                stage_root = PROJECT_STORAGE_DIR / "artifacts" / stage_folder
                ordered_runs: list[Path] = []

                if preferred_run_prefix is not None:
                    ordered_runs.extend(reversed(completed_run_dirs(stage_root, run_prefix=preferred_run_prefix)))

                for run_dir in reversed(completed_run_dirs(stage_root)):
                    if run_dir not in ordered_runs:
                        ordered_runs.append(run_dir)

                relative = Path(relative_path)
                for run_dir in ordered_runs:
                    candidate = run_dir / relative
                    if candidate.exists():
                        return str(candidate)
                return None


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

            DEFAULT_STAGE5_MODEL_DIR = default_upstream_path(
                "stage5_colab",
                "learner",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage5",
            )
            DEFAULT_STAGE6_BALANCER_DIR = default_upstream_path(
                "stage6_colab",
                "balancer",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage6",
            )
            DEFAULT_STAGE6_EVAL = default_upstream_path(
                "stage6_colab",
                "balance_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage6",
            )
            DEFAULT_STAGE4_CONTROL_EVAL = default_upstream_path(
                "stage4_colab",
                "control_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage4",
            )
            STAGE5_MODEL_DIR = DEFAULT_STAGE5_MODEL_DIR
            STAGE6_BALANCER_DIR = DEFAULT_STAGE6_BALANCER_DIR
            COMPARISON_BASELINE_EVAL_PATH = DEFAULT_STAGE4_CONTROL_EVAL or DEFAULT_STAGE6_EVAL

            MAX_UNSUPPORTED_ANSWER_RATE = 20.0
            MAX_OVERABSTAIN_RATE = 20.0
            MAX_CANDIDATES_PER_EXAMPLE = 6
            TAIL_RISK_WEIGHT = 2.0
            INITIAL_UNSAFE_DUAL = 1.0
            INITIAL_OVERABSTAIN_DUAL = 1.0
            HARD_RISK_SHIELD = 0.35
            TRAIN_BATCH_SIZE = 16
            EVAL_BATCH_SIZE = 32
            ACTION_LR = 2e-3
            ACTION_WEIGHT_DECAY = 0.01
            ACTION_EPOCHS = 12
            HIDDEN_SIZE = 64
            DROPOUT = 0.10
            ACTION_LOSS_WEIGHT = 1.0
            UTILITY_LOSS_WEIGHT = 0.5
            RISK_LOSS_WEIGHT = 1.0
            UNSAFE_DUAL_LR = 0.25
            OVERABSTAIN_DUAL_LR = 0.25
            MAX_TRAIN_SAMPLES = None
            MAX_EVAL_SAMPLES = None
            CLEAN_SPLITTING = True
            SMOKE_TEST_TRAIN_SAMPLES = 256
            SMOKE_TEST_EVAL_SAMPLES = 128
            SMOKE_TEST_EPOCHS = 2

            ACTION_LEARNER_DIR = OUTPUT_ROOT / "risk-action-learner"
            ACTION_EVAL = OUTPUT_ROOT / "risk_action_eval.json"

            RUN_SMOKE_TEST = False

            if STAGE5_MODEL_DIR is not None:
                STAGE5_MODEL_DIR = Path(STAGE5_MODEL_DIR).expanduser().resolve()
                if not STAGE5_MODEL_DIR.exists():
                    raise FileNotFoundError(f"Stage 5 model directory not found: {STAGE5_MODEL_DIR}")

            if STAGE6_BALANCER_DIR is not None:
                STAGE6_BALANCER_DIR = Path(STAGE6_BALANCER_DIR).expanduser().resolve()
                if not STAGE6_BALANCER_DIR.exists():
                    raise FileNotFoundError(f"Stage 6 balancer directory not found: {STAGE6_BALANCER_DIR}")

            if COMPARISON_BASELINE_EVAL_PATH is not None:
                COMPARISON_BASELINE_EVAL_PATH = Path(COMPARISON_BASELINE_EVAL_PATH).expanduser().resolve()
                if not COMPARISON_BASELINE_EVAL_PATH.exists():
                    raise FileNotFoundError(f"Baseline eval file not found: {COMPARISON_BASELINE_EVAL_PATH}")

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
                        f"stage5_model_dir={STAGE5_MODEL_DIR}",
                        f"stage6_balancer_dir={STAGE6_BALANCER_DIR}",
                        f"comparison_baseline_eval_path={COMPARISON_BASELINE_EVAL_PATH}",
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            print(f"Run marker file: {RUN_MARKER_FILE}")
            print(f"Stage 5 model dir: {STAGE5_MODEL_DIR}")
            print(f"Stage 6 balancer dir: {STAGE6_BALANCER_DIR}")
            print(f"Comparison baseline eval path: {COMPARISON_BASELINE_EVAL_PATH}")
            print(f"Planned action-learner dir: {ACTION_LEARNER_DIR}")
            print(f"Planned action-eval file: {ACTION_EVAL}")
            print(f"Max unsupported-answer rate: {MAX_UNSUPPORTED_ANSWER_RATE}")
            print(f"Max over-abstain rate: {MAX_OVERABSTAIN_RATE}")
            print(f"Hard risk shield: {HARD_RISK_SHIELD}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("Target metrics:", ", ".join(TARGET_METRICS))
            print("Suggested modules:", ", ".join(SUGGESTED_MODULES))

            if STAGE5_MODEL_DIR is None:
                print("Set STAGE5_MODEL_DIR before implementing Stage 7.")
            if STAGE6_BALANCER_DIR is None:
                print("Stage 6 balancer not found. Stage 7 will still run, but without the Stage 6 prior feature.")


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


            def maybe_add_arg(cmd: list[str], flag: str, value) -> None:
                if value is None:
                    return
                cmd.extend([flag, str(value)])


            def maybe_add_flag(cmd: list[str], flag: str, enabled: bool) -> None:
                if enabled:
                    cmd.append(flag)


            def build_train_command(
                stage5_model_dir: Path,
                output_dir: Path,
                *,
                stage6_controller_dir: Path | None,
                max_train_samples: int | None,
                max_eval_samples: int | None,
                num_train_epochs: int,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.action",
                    "train",
                    "--stage5-model-path",
                    str(stage5_model_dir),
                    "--output-dir",
                    str(output_dir),
                    "--train-batch-size",
                    str(TRAIN_BATCH_SIZE),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--learning-rate",
                    str(ACTION_LR),
                    "--weight-decay",
                    str(ACTION_WEIGHT_DECAY),
                    "--num-train-epochs",
                    str(num_train_epochs),
                    "--hidden-size",
                    str(HIDDEN_SIZE),
                    "--dropout",
                    str(DROPOUT),
                    "--max-candidates-per-example",
                    str(MAX_CANDIDATES_PER_EXAMPLE),
                    "--action-loss-weight",
                    str(ACTION_LOSS_WEIGHT),
                    "--utility-loss-weight",
                    str(UTILITY_LOSS_WEIGHT),
                    "--risk-loss-weight",
                    str(RISK_LOSS_WEIGHT),
                    "--tail-risk-weight",
                    str(TAIL_RISK_WEIGHT),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                    "--max-overabstain-rate",
                    str(MAX_OVERABSTAIN_RATE),
                    "--unsafe-dual-init",
                    str(INITIAL_UNSAFE_DUAL),
                    "--overabstain-dual-init",
                    str(INITIAL_OVERABSTAIN_DUAL),
                    "--unsafe-dual-lr",
                    str(UNSAFE_DUAL_LR),
                    "--overabstain-dual-lr",
                    str(OVERABSTAIN_DUAL_LR),
                    "--hard-risk-threshold",
                    str(HARD_RISK_SHIELD),
                ]
                if stage6_controller_dir is not None:
                    cmd.extend(["--stage6-controller-path", str(stage6_controller_dir)])
                maybe_add_flag(cmd, "--clean-splitting", CLEAN_SPLITTING)
                maybe_add_arg(cmd, "--max-train-samples", max_train_samples)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                maybe_add_arg(cmd, "--max-test-samples", max_eval_samples if CLEAN_SPLITTING else None)
                return cmd


            def build_eval_command(
                model_dir: Path,
                stage5_model_dir: Path,
                output_path: Path,
                *,
                stage6_controller_dir: Path | None,
                max_eval_samples: int | None,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.action",
                    "evaluate",
                    "--model-path",
                    str(model_dir),
                    "--stage5-model-path",
                    str(stage5_model_dir),
                    "--output-path",
                    str(output_path),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--risk-threshold-min",
                    "0.10",
                    "--risk-threshold-max",
                    "0.90",
                    "--risk-threshold-step",
                    "0.05",
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                    "--max-overabstain-rate",
                    str(MAX_OVERABSTAIN_RATE),
                ]
                if stage6_controller_dir is not None:
                    cmd.extend(["--stage6-controller-path", str(stage6_controller_dir)])
                maybe_add_flag(cmd, "--clean-splitting", CLEAN_SPLITTING)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                maybe_add_arg(cmd, "--max-test-samples", max_eval_samples if CLEAN_SPLITTING else None)
                return cmd


            if STAGE5_MODEL_DIR is None:
                SMOKE_TEST_COMMANDS = []
                STAGE_COMMANDS = []
            else:
                smoke_model_dir = OUTPUT_ROOT / "smoke-risk-action-learner"
                smoke_eval_path = OUTPUT_ROOT / "smoke-risk-action-eval.json"
                smoke_train_command = build_train_command(
                    STAGE5_MODEL_DIR,
                    smoke_model_dir,
                    stage6_controller_dir=STAGE6_BALANCER_DIR,
                    max_train_samples=SMOKE_TEST_TRAIN_SAMPLES,
                    max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                    num_train_epochs=SMOKE_TEST_EPOCHS,
                )
                smoke_eval_command = build_eval_command(
                    smoke_model_dir,
                    STAGE5_MODEL_DIR,
                    smoke_eval_path,
                    stage6_controller_dir=STAGE6_BALANCER_DIR,
                    max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                )
                stage_train_command = build_train_command(
                    STAGE5_MODEL_DIR,
                    ACTION_LEARNER_DIR,
                    stage6_controller_dir=STAGE6_BALANCER_DIR,
                    max_train_samples=MAX_TRAIN_SAMPLES,
                    max_eval_samples=MAX_EVAL_SAMPLES,
                    num_train_epochs=ACTION_EPOCHS,
                )
                stage_eval_command = build_eval_command(
                    ACTION_LEARNER_DIR,
                    STAGE5_MODEL_DIR,
                    ACTION_EVAL,
                    stage6_controller_dir=STAGE6_BALANCER_DIR,
                    max_eval_samples=MAX_EVAL_SAMPLES,
                )
                SMOKE_TEST_COMMANDS = [smoke_train_command, smoke_eval_command]
                STAGE_COMMANDS = [stage_train_command, stage_eval_command]
            """
        ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
        + "\n"
    )


def stage7_save_code() -> str:
    return (
        dedent(
            """
            captured_notebook_path = save_executed_notebook_snapshot()
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
                            f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
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
                            f"- Stage 5 model dir: {STAGE5_MODEL_DIR}",
                            f"- Stage 6 balancer dir: {STAGE6_BALANCER_DIR}",
                            f"- Comparison baseline eval path: {COMPARISON_BASELINE_EVAL_PATH}",
                            f"- Planned output files to inspect: {ACTION_EVAL}",
                            "",
                            "## Planned Design",
                            f"- Max unsupported-answer rate: {MAX_UNSUPPORTED_ANSWER_RATE}",
                            f"- Max over-abstain rate: {MAX_OVERABSTAIN_RATE}",
                            f"- Tail-risk weight: {TAIL_RISK_WEIGHT}",
                            f"- Initial unsafe dual: {INITIAL_UNSAFE_DUAL}",
                            f"- Initial over-abstain dual: {INITIAL_OVERABSTAIN_DUAL}",
                            f"- Hard risk shield: {HARD_RISK_SHIELD}",
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
                        "executed_notebook_dir": str(NOTEBOOK_ARCHIVE_DIR),
                        "executed_notebook_target": str(EXECUTED_NOTEBOOK_PATH),
                        "executed_notebook_saved": captured_notebook_path is not None,
                        "executed_notebook_instructions_file": str(EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE),
                        "stage5_model_dir": str(STAGE5_MODEL_DIR) if STAGE5_MODEL_DIR is not None else None,
                        "stage6_balancer_dir": str(STAGE6_BALANCER_DIR) if STAGE6_BALANCER_DIR is not None else None,
                        "comparison_baseline_eval_path": str(COMPARISON_BASELINE_EVAL_PATH) if COMPARISON_BASELINE_EVAL_PATH is not None else None,
                        "action_learner_dir": str(ACTION_LEARNER_DIR),
                        "action_eval": str(ACTION_EVAL),
                        "target_metrics": TARGET_METRICS,
                        "suggested_modules": SUGGESTED_MODULES,
                        "max_unsupported_answer_rate": MAX_UNSUPPORTED_ANSWER_RATE,
                        "max_overabstain_rate": MAX_OVERABSTAIN_RATE,
                        "tail_risk_weight": TAIL_RISK_WEIGHT,
                        "initial_unsafe_dual": INITIAL_UNSAFE_DUAL,
                        "initial_overabstain_dual": INITIAL_OVERABSTAIN_DUAL,
                        "hard_risk_shield": HARD_RISK_SHIELD,
                    },
                    indent=2,
                )
                + "\\n",
                encoding="utf-8",
            )

            print(f"Notes template: {RUN_NOTES_FILE}")
            print(f"Run summary: {RUN_SUMMARY_FILE}")
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
            if mirrored_output_root is not None:
                print(f"Drive mirror: {mirrored_output_root}")
            print("Current files under OUTPUT_ROOT:")
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                print(path)
            """
        ).strip()
        + "\n"
    )


STAGE_2 = {
    "path": REPO_ROOT / "stages/02-evidence-support-verification/notebooks/stage-02-evidence-support-verification-colab.ipynb",
    "branch": "main",
    "stage_number": 2,
    "stage_label": "Stage 2: Evidence Support Verification",
    "config_markdown": (
        dedent(
            """
            <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

            Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage2-v1`, `yourname-stage2-v2`, `yourname-stage2-v3`, and so on based on completed runs.

            This notebook first tries to auto-fill `BASE_QA_MODEL_DIR` from the latest completed Stage 1 run under the current artifact root. Override it if you want to compare against a different checkpoint.

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


def stage8_config_code() -> str:
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
            RUN_BASENAME = f"{AUTHOR_NAME}-stage8-2"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "stage8_2_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            STAGE_TITLE = "Stage 8.2: Action Learner + Calibrated Support"
            STAGE_OBJECTIVE = "Keep Stage 5 as the answer engine, inject calibrated Stage 4 support into the Stage 7 action learner, optionally retain the Stage 6 prior, and report final metrics on a truly untouched test split."
            TARGET_METRICS = [
                "overall F1",
                "answerable F1",
                "unsupported-answer rate",
                "supported-answer rate",
                "answer rate",
                "comparison against Stage 5 and Stage 7",
            ]
            IMPLEMENTATION_HINTS = [
                "input: frozen Stage 5 candidate answers, calibrated Stage 4 support, and optionally the Stage 6 prior",
                "output: a decision-aware action learner over {candidate_1 ... candidate_k, abstain}",
                "success means a better answer-quality versus groundedness balance than raw Stage 5 or plain Stage 7",
            ]
            SUGGESTED_MODULES = ["keelnet.action", "keelnet.control", "keelnet.learn", "keelnet.verify", "keelnet.metrics"]


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


            def completed_run_dirs(root: Path, *, run_prefix: str | None = None) -> list[Path]:
                if not root.exists():
                    return []

                pattern = re.compile(rf"^{re.escape(run_prefix)}-v(\\d+)$") if run_prefix is not None else None
                runs: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if not (child / COMPLETION_MARKER_NAME).exists():
                        continue
                    if pattern is not None and not pattern.match(child.name):
                        continue
                    runs.append(child)
                return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name))


            def default_upstream_path(stage_folder: str, relative_path: str, *, preferred_run_prefix: str | None = None) -> str | None:
                stage_root = PROJECT_STORAGE_DIR / "artifacts" / stage_folder
                ordered_runs: list[Path] = []

                if preferred_run_prefix is not None:
                    ordered_runs.extend(reversed(completed_run_dirs(stage_root, run_prefix=preferred_run_prefix)))

                for run_dir in reversed(completed_run_dirs(stage_root)):
                    if run_dir not in ordered_runs:
                        ordered_runs.append(run_dir)

                relative = Path(relative_path)
                for run_dir in ordered_runs:
                    candidate = run_dir / relative
                    if candidate.exists():
                        return str(candidate)
                return None


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

            DEFAULT_STAGE5_MODEL_DIR = default_upstream_path(
                "stage5_colab",
                "learner",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage5",
            )
            DEFAULT_STAGE5_EVAL = default_upstream_path(
                "stage5_colab",
                "learner_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage5",
            )
            DEFAULT_STAGE6_BALANCER_DIR = default_upstream_path(
                "stage6_colab",
                "balancer",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage6",
            )
            DEFAULT_STAGE4_CONTROL_EVAL = default_upstream_path(
                "stage4_colab",
                "control_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage4",
            )
            DEFAULT_STAGE7_EVAL = default_upstream_path(
                "stage7_colab",
                "risk_action_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage7",
            )
            STAGE5_MODEL_DIR = DEFAULT_STAGE5_MODEL_DIR
            STAGE6_BALANCER_DIR = DEFAULT_STAGE6_BALANCER_DIR
            CONTROL_EVAL_PATH = DEFAULT_STAGE4_CONTROL_EVAL
            COMPARISON_BASELINE_EVAL_PATH = DEFAULT_STAGE7_EVAL or DEFAULT_STAGE5_EVAL

            TRAIN_BATCH_SIZE = 16
            EVAL_BATCH_SIZE = 32
            ACTION_EPOCHS = 12
            ACTION_LR = 2e-3
            ACTION_WEIGHT_DECAY = 0.01
            HIDDEN_SIZE = 64
            DROPOUT = 0.10
            MAX_CANDIDATES_PER_EXAMPLE = 6
            MAX_CANDIDATES_PER_FEATURE = 3
            MAX_UNSUPPORTED_ANSWER_RATE = 20.0
            MAX_OVERABSTAIN_RATE = 20.0
            RISK_THRESHOLD_MIN = 0.10
            RISK_THRESHOLD_MAX = 0.90
            RISK_THRESHOLD_STEP = 0.05
            ACTION_LOSS_WEIGHT = 1.0
            UTILITY_LOSS_WEIGHT = 0.5
            RISK_LOSS_WEIGHT = 1.0
            TAIL_RISK_WEIGHT = 2.0
            INITIAL_UNSAFE_DUAL = 1.0
            INITIAL_OVERABSTAIN_DUAL = 1.0
            UNSAFE_DUAL_LR = 0.25
            OVERABSTAIN_DUAL_LR = 0.25
            HARD_RISK_SHIELD = 0.35
            CLEAN_SPLITTING = True
            MAX_TRAIN_SAMPLES = None
            MAX_EVAL_SAMPLES = None

            RUN_SMOKE_TEST = False
            SMOKE_TEST_TRAIN_SAMPLES = 256
            SMOKE_TEST_EVAL_SAMPLES = 128
            SMOKE_TEST_EPOCHS = 2

            HYBRID_DIR = OUTPUT_ROOT / "stage8-2-action-learner"
            HYBRID_EVAL = OUTPUT_ROOT / "hybrid_eval.json"

            if STAGE5_MODEL_DIR is not None:
                STAGE5_MODEL_DIR = Path(STAGE5_MODEL_DIR).expanduser().resolve()
                if not STAGE5_MODEL_DIR.exists():
                    raise FileNotFoundError(f"Stage 5 model directory not found: {STAGE5_MODEL_DIR}")

            if CONTROL_EVAL_PATH is not None:
                CONTROL_EVAL_PATH = Path(CONTROL_EVAL_PATH).expanduser().resolve()
                if not CONTROL_EVAL_PATH.exists():
                    raise FileNotFoundError(f"Stage 4 control eval not found: {CONTROL_EVAL_PATH}")

            if STAGE6_BALANCER_DIR is not None:
                STAGE6_BALANCER_DIR = Path(STAGE6_BALANCER_DIR).expanduser().resolve()
                if not STAGE6_BALANCER_DIR.exists():
                    raise FileNotFoundError(f"Stage 6 balancer directory not found: {STAGE6_BALANCER_DIR}")

            if COMPARISON_BASELINE_EVAL_PATH is not None:
                COMPARISON_BASELINE_EVAL_PATH = Path(COMPARISON_BASELINE_EVAL_PATH).expanduser().resolve()
                if not COMPARISON_BASELINE_EVAL_PATH.exists():
                    raise FileNotFoundError(f"Comparison baseline eval not found: {COMPARISON_BASELINE_EVAL_PATH}")


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


            def maybe_add_arg(cmd: list[str], flag: str, value) -> None:
                if value is None:
                    return
                cmd.extend([flag, str(value)])


            def maybe_add_flag(cmd: list[str], flag: str, enabled: bool) -> None:
                if enabled:
                    cmd.append(flag)


            def build_train_command(
                stage5_model_dir: Path,
                control_eval_path: Path,
                output_dir: Path,
                *,
                stage6_controller_dir: Path | None,
                max_train_samples: int | None,
                max_eval_samples: int | None,
                num_train_epochs: int,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.action",
                    "train",
                    "--stage5-model-path",
                    str(stage5_model_dir),
                    "--control-path",
                    str(control_eval_path),
                    "--output-dir",
                    str(output_dir),
                    "--train-batch-size",
                    str(TRAIN_BATCH_SIZE),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--learning-rate",
                    str(ACTION_LR),
                    "--weight-decay",
                    str(ACTION_WEIGHT_DECAY),
                    "--num-train-epochs",
                    str(num_train_epochs),
                    "--hidden-size",
                    str(HIDDEN_SIZE),
                    "--dropout",
                    str(DROPOUT),
                    "--max-candidates-per-example",
                    str(MAX_CANDIDATES_PER_EXAMPLE),
                    "--max-candidates-per-feature",
                    str(MAX_CANDIDATES_PER_FEATURE),
                    "--action-loss-weight",
                    str(ACTION_LOSS_WEIGHT),
                    "--utility-loss-weight",
                    str(UTILITY_LOSS_WEIGHT),
                    "--risk-loss-weight",
                    str(RISK_LOSS_WEIGHT),
                    "--tail-risk-weight",
                    str(TAIL_RISK_WEIGHT),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                    "--max-overabstain-rate",
                    str(MAX_OVERABSTAIN_RATE),
                    "--unsafe-dual-init",
                    str(INITIAL_UNSAFE_DUAL),
                    "--overabstain-dual-init",
                    str(INITIAL_OVERABSTAIN_DUAL),
                    "--unsafe-dual-lr",
                    str(UNSAFE_DUAL_LR),
                    "--overabstain-dual-lr",
                    str(OVERABSTAIN_DUAL_LR),
                    "--hard-risk-threshold",
                    str(HARD_RISK_SHIELD),
                ]
                if stage6_controller_dir is not None:
                    cmd.extend(["--stage6-controller-path", str(stage6_controller_dir)])
                maybe_add_flag(cmd, "--clean-splitting", CLEAN_SPLITTING)
                maybe_add_arg(cmd, "--max-train-samples", max_train_samples)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                maybe_add_arg(cmd, "--max-test-samples", max_eval_samples if CLEAN_SPLITTING else None)
                return cmd


            def build_eval_command(
                model_dir: Path,
                stage5_model_dir: Path,
                control_eval_path: Path,
                output_path: Path,
                *,
                stage6_controller_dir: Path | None,
                max_eval_samples: int | None,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.action",
                    "evaluate",
                    "--model-path",
                    str(model_dir),
                    "--stage5-model-path",
                    str(stage5_model_dir),
                    "--control-path",
                    str(control_eval_path),
                    "--output-path",
                    str(output_path),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--risk-threshold-min",
                    str(RISK_THRESHOLD_MIN),
                    "--risk-threshold-max",
                    str(RISK_THRESHOLD_MAX),
                    "--risk-threshold-step",
                    str(RISK_THRESHOLD_STEP),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                    "--max-overabstain-rate",
                    str(MAX_OVERABSTAIN_RATE),
                ]
                if stage6_controller_dir is not None:
                    cmd.extend(["--stage6-controller-path", str(stage6_controller_dir)])
                maybe_add_flag(cmd, "--clean-splitting", CLEAN_SPLITTING)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                maybe_add_arg(cmd, "--max-test-samples", max_eval_samples if CLEAN_SPLITTING else None)
                return cmd


            if STAGE5_MODEL_DIR is None or CONTROL_EVAL_PATH is None:
                SMOKE_TEST_COMMANDS = []
                STAGE_COMMANDS = []
            else:
                smoke_model_dir = OUTPUT_ROOT / "smoke-hybrid-controller"
                smoke_eval_path = OUTPUT_ROOT / "smoke-hybrid-eval.json"
                smoke_train_command = build_train_command(
                    STAGE5_MODEL_DIR,
                    CONTROL_EVAL_PATH,
                    smoke_model_dir,
                    stage6_controller_dir=STAGE6_BALANCER_DIR,
                    max_train_samples=SMOKE_TEST_TRAIN_SAMPLES,
                    max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                    num_train_epochs=SMOKE_TEST_EPOCHS,
                )
                smoke_eval_command = build_eval_command(
                    smoke_model_dir,
                    STAGE5_MODEL_DIR,
                    CONTROL_EVAL_PATH,
                    smoke_eval_path,
                    stage6_controller_dir=STAGE6_BALANCER_DIR,
                    max_eval_samples=SMOKE_TEST_EVAL_SAMPLES,
                )
                stage_train_command = build_train_command(
                    STAGE5_MODEL_DIR,
                    CONTROL_EVAL_PATH,
                    HYBRID_DIR,
                    stage6_controller_dir=STAGE6_BALANCER_DIR,
                    max_train_samples=MAX_TRAIN_SAMPLES,
                    max_eval_samples=MAX_EVAL_SAMPLES,
                    num_train_epochs=ACTION_EPOCHS,
                )
                stage_eval_command = build_eval_command(
                    HYBRID_DIR,
                    STAGE5_MODEL_DIR,
                    CONTROL_EVAL_PATH,
                    HYBRID_EVAL,
                    stage6_controller_dir=STAGE6_BALANCER_DIR,
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
                        f"stage5_model_dir={STAGE5_MODEL_DIR}",
                        f"stage6_balancer_dir={STAGE6_BALANCER_DIR}",
                        f"control_eval_path={CONTROL_EVAL_PATH}",
                        f"comparison_baseline_eval_path={COMPARISON_BASELINE_EVAL_PATH}",
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            print(f"Run marker file: {RUN_MARKER_FILE}")
            print(f"Stage 5 model dir: {STAGE5_MODEL_DIR}")
            print(f"Stage 6 balancer dir: {STAGE6_BALANCER_DIR}")
            print(f"Stage 4 control eval path: {CONTROL_EVAL_PATH}")
            print(f"Comparison baseline eval path: {COMPARISON_BASELINE_EVAL_PATH}")
            print(f"Planned Stage 8.2 model dir: {HYBRID_DIR}")
            print(f"Planned Stage 8.2 eval file: {HYBRID_EVAL}")
            print(f"Max unsupported-answer rate: {MAX_UNSUPPORTED_ANSWER_RATE}")
            print(f"Max over-abstain rate: {MAX_OVERABSTAIN_RATE}")
            print(f"Hard risk shield: {HARD_RISK_SHIELD}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("Target metrics:", ", ".join(TARGET_METRICS))
            print("Suggested modules:", ", ".join(SUGGESTED_MODULES))

            if STAGE5_MODEL_DIR is None:
                print("Set STAGE5_MODEL_DIR before running Stage 8.")
            if STAGE6_BALANCER_DIR is None:
                print("Stage 6 balancer not found. Stage 8.2 will still run, but without the Stage 6 prior feature.")
            if CONTROL_EVAL_PATH is None:
                print("Set CONTROL_EVAL_PATH before running Stage 8.")
            """
        ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
        + "\n"
    )


def stage9_config_code() -> str:
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
            RUN_BASENAME = f"{AUTHOR_NAME}-stage9"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "stage9_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            STAGE_TITLE = "Stage 9: Risk Generalization And Tail-Risk Control"
            STAGE_OBJECTIVE = "Run the exact isolated Stage 9 architecture as a self-contained pipeline: train the proposal model and support scorer inside Stage 9 itself, then learn the risk-aware decision layer on top."
            TARGET_METRICS = [
                "overall F1",
                "answerable F1",
                "unsupported-answer rate",
                "supported-answer rate",
                "answer rate",
                "over-abstain rate",
                "comparison against Stage 8.2 and the Stage 4 anchor",
            ]
            IMPLEMENTATION_HINTS = [
                "inference path: Stage 9 trained proposal model -> Stage 9 trained support scorer -> candidate plus uncertainty features -> utility, risk, and abstain heads -> post-hoc risk calibration -> hard support shield -> better abstain boundary -> answer or abstain",
                "training path: train proposal and support modules inside Stage 9, then optimize joint decision loss plus utility, risk, and abstain supervision with stronger tail-risk pressure",
                "success means a smaller held-out safety gap than Stage 8.2 without collapsing answerable utility toward the Stage 4 anchor",
            ]
            SUGGESTED_MODULES = ["keelnet.stage9"]


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


            def completed_run_dirs(root: Path, *, run_prefix: str | None = None) -> list[Path]:
                if not root.exists():
                    return []

                pattern = re.compile(rf"^{re.escape(run_prefix)}-v(\\d+)$") if run_prefix is not None else None
                runs: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if not (child / COMPLETION_MARKER_NAME).exists():
                        continue
                    if pattern is not None and not pattern.match(child.name):
                        continue
                    runs.append(child)
                return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name))


            def default_upstream_path(stage_folder: str, relative_path: str, *, preferred_run_prefix: str | None = None) -> str | None:
                stage_root = PROJECT_STORAGE_DIR / "artifacts" / stage_folder
                ordered_runs: list[Path] = []

                if preferred_run_prefix is not None:
                    ordered_runs.extend(reversed(completed_run_dirs(stage_root, run_prefix=preferred_run_prefix)))

                for run_dir in reversed(completed_run_dirs(stage_root)):
                    if run_dir not in ordered_runs:
                        ordered_runs.append(run_dir)

                relative = Path(relative_path)
                for run_dir in ordered_runs:
                    candidate = run_dir / relative
                    if candidate.exists():
                        return str(candidate)
                return None


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

            DEFAULT_STAGE4_CONTROL_EVAL = default_upstream_path(
                "stage4_colab",
                "control_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage4",
            )
            DEFAULT_STAGE5_EVAL = default_upstream_path(
                "stage5_colab",
                "learner_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage5",
            )
            DEFAULT_STAGE7_EVAL = default_upstream_path(
                "stage7_colab",
                "risk_action_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage7",
            )
            DEFAULT_STAGE8_2_EVAL = default_upstream_path(
                "stage8_2_colab",
                "hybrid_eval.json",
                preferred_run_prefix=f"{AUTHOR_NAME}-stage8-2",
            )
            COMPARISON_BASELINE_EVAL_PATH = DEFAULT_STAGE8_2_EVAL or DEFAULT_STAGE7_EVAL or DEFAULT_STAGE5_EVAL
            SAFETY_ANCHOR_EVAL_PATH = DEFAULT_STAGE4_CONTROL_EVAL
            BASE_MODEL_NAME = "distilbert-base-uncased"

            TRAIN_BATCH_SIZE = 16
            EVAL_BATCH_SIZE = 32
            PROPOSAL_TRAIN_BATCH_SIZE = 8
            SUPPORT_TRAIN_BATCH_SIZE = 16
            PROPOSAL_EPOCHS = 2
            SUPPORT_EPOCHS = 2
            PROPOSAL_LR = 2e-5
            SUPPORT_LR = 2e-5
            PROPOSAL_WEIGHT_DECAY = 0.01
            SUPPORT_WEIGHT_DECAY = 0.01
            ACTION_EPOCHS = 16
            ACTION_LR = 2e-3
            ACTION_WEIGHT_DECAY = 0.01
            HIDDEN_SIZE = 96
            DROPOUT = 0.15
            MAX_CANDIDATES_PER_EXAMPLE = 6
            MAX_CANDIDATES_PER_FEATURE = 3
            MAX_UNSUPPORTED_ANSWER_RATE = 20.0
            MAX_OVERABSTAIN_RATE = 20.0
            RISK_THRESHOLD_MIN = 0.05
            RISK_THRESHOLD_MAX = 0.85
            RISK_THRESHOLD_STEP = 0.02
            ACTION_LOSS_WEIGHT = 1.0
            UTILITY_LOSS_WEIGHT = 0.5
            RISK_LOSS_WEIGHT = 1.25
            ABSTAIN_LOSS_WEIGHT = 0.5
            RISK_PENALTY = 1.0
            TAIL_RISK_WEIGHT = 3.0
            RANDOMIZATION_SCALE = 0.10
            HARD_SUPPORT_THRESHOLD_OVERRIDE = None
            CLEAN_SPLITTING = True
            SEED = 42
            MAX_TRAIN_SAMPLES = None
            MAX_EVAL_SAMPLES = None

            RUN_SMOKE_TEST = False
            SMOKE_TEST_TRAIN_SAMPLES = 256
            SMOKE_TEST_EVAL_SAMPLES = 128
            SMOKE_TEST_EPOCHS = 2

            STAGE9_DIR = OUTPUT_ROOT / "stage9-risk-generalization"
            STAGE9_EVAL = OUTPUT_ROOT / "stage9_eval.json"

            if COMPARISON_BASELINE_EVAL_PATH is not None:
                COMPARISON_BASELINE_EVAL_PATH = Path(COMPARISON_BASELINE_EVAL_PATH).expanduser().resolve()
                if not COMPARISON_BASELINE_EVAL_PATH.exists():
                    raise FileNotFoundError(f"Comparison baseline eval not found: {COMPARISON_BASELINE_EVAL_PATH}")

            if SAFETY_ANCHOR_EVAL_PATH is not None:
                SAFETY_ANCHOR_EVAL_PATH = Path(SAFETY_ANCHOR_EVAL_PATH).expanduser().resolve()
                if not SAFETY_ANCHOR_EVAL_PATH.exists():
                    raise FileNotFoundError(f"Safety anchor eval not found: {SAFETY_ANCHOR_EVAL_PATH}")


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


            def maybe_add_arg(cmd: list[str], flag: str, value) -> None:
                if value is None:
                    return
                cmd.extend([flag, str(value)])


            def maybe_add_flag(cmd: list[str], flag: str, enabled: bool) -> None:
                if enabled:
                    cmd.append(flag)


            def build_train_command(
                output_dir: Path,
                *,
                max_train_samples: int | None,
                max_eval_samples: int | None,
                num_train_epochs: int,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.stage9",
                    "train",
                    "--base-model-name",
                    str(BASE_MODEL_NAME),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    str(SEED),
                    "--proposal-train-batch-size",
                    str(PROPOSAL_TRAIN_BATCH_SIZE),
                    "--support-train-batch-size",
                    str(SUPPORT_TRAIN_BATCH_SIZE),
                    "--proposal-learning-rate",
                    str(PROPOSAL_LR),
                    "--support-learning-rate",
                    str(SUPPORT_LR),
                    "--proposal-weight-decay",
                    str(PROPOSAL_WEIGHT_DECAY),
                    "--support-weight-decay",
                    str(SUPPORT_WEIGHT_DECAY),
                    "--proposal-num-train-epochs",
                    str(PROPOSAL_EPOCHS),
                    "--support-num-train-epochs",
                    str(SUPPORT_EPOCHS),
                    "--train-batch-size",
                    str(TRAIN_BATCH_SIZE),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--learning-rate",
                    str(ACTION_LR),
                    "--weight-decay",
                    str(ACTION_WEIGHT_DECAY),
                    "--num-train-epochs",
                    str(num_train_epochs),
                    "--hidden-size",
                    str(HIDDEN_SIZE),
                    "--dropout",
                    str(DROPOUT),
                    "--max-candidates-per-example",
                    str(MAX_CANDIDATES_PER_EXAMPLE),
                    "--max-candidates-per-feature",
                    str(MAX_CANDIDATES_PER_FEATURE),
                    "--utility-loss-weight",
                    str(UTILITY_LOSS_WEIGHT),
                    "--risk-loss-weight",
                    str(RISK_LOSS_WEIGHT),
                    "--tail-risk-weight",
                    str(TAIL_RISK_WEIGHT),
                    "--randomization-scale",
                    str(RANDOMIZATION_SCALE),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                    "--max-overabstain-rate",
                    str(MAX_OVERABSTAIN_RATE),
                    "--joint-loss-weight",
                    str(ACTION_LOSS_WEIGHT),
                    "--abstain-loss-weight",
                    str(ABSTAIN_LOSS_WEIGHT),
                    "--risk-penalty",
                    str(RISK_PENALTY),
                    "--abstain-margin-min",
                    "0.0",
                    "--abstain-margin-max",
                    "0.50",
                    "--abstain-margin-step",
                    "0.05",
                ]
                maybe_add_flag(cmd, "--clean-splitting", CLEAN_SPLITTING)
                maybe_add_arg(cmd, "--hard-support-threshold", HARD_SUPPORT_THRESHOLD_OVERRIDE)
                maybe_add_arg(cmd, "--max-train-samples", max_train_samples)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                maybe_add_arg(cmd, "--max-test-samples", max_eval_samples if CLEAN_SPLITTING else None)
                return cmd


            def build_eval_command(
                model_dir: Path,
                output_path: Path,
                *,
                max_eval_samples: int | None,
            ) -> list[str]:
                cmd = [
                    sys.executable,
                    "-m",
                    "keelnet.stage9",
                    "evaluate",
                    "--model-path",
                    str(model_dir),
                    "--output-path",
                    str(output_path),
                    "--seed",
                    str(SEED),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--risk-threshold-min",
                    str(RISK_THRESHOLD_MIN),
                    "--risk-threshold-max",
                    str(RISK_THRESHOLD_MAX),
                    "--risk-threshold-step",
                    str(RISK_THRESHOLD_STEP),
                    "--max-unsupported-answer-rate",
                    str(MAX_UNSUPPORTED_ANSWER_RATE),
                    "--max-overabstain-rate",
                    str(MAX_OVERABSTAIN_RATE),
                    "--abstain-margin-min",
                    "0.0",
                    "--abstain-margin-max",
                    "0.50",
                    "--abstain-margin-step",
                    "0.05",
                ]
                maybe_add_flag(cmd, "--clean-splitting", CLEAN_SPLITTING)
                maybe_add_arg(cmd, "--hard-support-threshold", HARD_SUPPORT_THRESHOLD_OVERRIDE)
                maybe_add_arg(cmd, "--max-eval-samples", max_eval_samples)
                maybe_add_arg(cmd, "--max-test-samples", max_eval_samples if CLEAN_SPLITTING else None)
                return cmd


            smoke_model_dir = OUTPUT_ROOT / "smoke-stage9-controller"
            smoke_eval_path = OUTPUT_ROOT / "smoke-stage9-eval.json"
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
                STAGE9_DIR,
                max_train_samples=MAX_TRAIN_SAMPLES,
                max_eval_samples=MAX_EVAL_SAMPLES,
                num_train_epochs=ACTION_EPOCHS,
            )
            stage_eval_command = build_eval_command(
                STAGE9_DIR,
                STAGE9_EVAL,
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
                        f"seed={SEED}",
                        f"base_model_name={BASE_MODEL_NAME}",
                        f"comparison_baseline_eval_path={COMPARISON_BASELINE_EVAL_PATH}",
                        f"safety_anchor_eval_path={SAFETY_ANCHOR_EVAL_PATH}",
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
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            print(f"Run marker file: {RUN_MARKER_FILE}")
            print(f"Seed: {SEED}")
            print(f"Base model name: {BASE_MODEL_NAME}")
            print(f"Comparison baseline eval path: {COMPARISON_BASELINE_EVAL_PATH}")
            print(f"Safety anchor eval path: {SAFETY_ANCHOR_EVAL_PATH}")
            print(f"Planned Stage 9 model dir: {STAGE9_DIR}")
            print(f"Planned Stage 9 eval file: {STAGE9_EVAL}")
            print(f"Max unsupported-answer rate: {MAX_UNSUPPORTED_ANSWER_RATE}")
            print(f"Max over-abstain rate: {MAX_OVERABSTAIN_RATE}")
            print(f"Hard support threshold override: {HARD_SUPPORT_THRESHOLD_OVERRIDE}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("Target metrics:", ", ".join(TARGET_METRICS))
            print("Suggested modules:", ", ".join(SUGGESTED_MODULES))

            if COMPARISON_BASELINE_EVAL_PATH is None:
                print("No Stage 8.2 or Stage 7 baseline eval was found automatically. Add one if you want a direct comparison anchor.")
            if SAFETY_ANCHOR_EVAL_PATH is None:
                print("No optional Stage 4 safety-anchor eval was found automatically. Add one only if you want that modular comparison in the same run notes.")
            """
        ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
        + "\n"
    )


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

                This notebook first tries to auto-fill `BASE_QA_MODEL_DIR` and `VERIFIER_MODEL_DIR` from the latest completed Stage 1 and Stage 2 runs under the current artifact root. Override them if you want to compare against different checkpoints.

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

                This notebook first tries to auto-fill `CALIBRATION_EVAL_PATH` from the latest completed Stage 3 run under the current artifact root. Override it if you want to compare against a different calibration pass.

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

                This notebook first tries to auto-fill `MODULAR_BASELINE_EVAL_PATH` from the latest completed Stage 4 run under the current artifact root. Override it if you want to compare against a different controller baseline.

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

                This notebook first tries to auto-fill `STAGE5_MODEL_DIR` from the latest completed Stage 5 run under the current artifact root. It also tries to auto-fill `COMPARISON_BASELINE_EVAL_PATH` from the latest available Stage 4 controller eval, falling back to Stage 5 eval if needed.

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

                Use this cell to run a tiny Stage 6 controller train-and-evaluate pass before the full run. Keep it small so you can catch path, dependency, and runtime issues early.
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

                The config cell now builds the default Stage 6 train and evaluate commands automatically. Review those commands, adjust the controller hyperparameters if needed, and then run this section.
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
        "config_code": stage6_config_code(),
        "implementation_code": stage6_implementation_code(),
        "save_code": stage6_save_code(),
        "share_code": stage6_share_code(),
    },
    {
        "path": REPO_ROOT / "stages/07-risk-budgeted-action-learning/notebooks/stage-07-risk-budgeted-action-learning-colab.ipynb",
        "branch": "main",
        "stage_number": 7,
        "stage_label": "Stage 7: Risk-Budgeted Action Learning",
        "objective": "Turn grounded QA into an explicit utility-versus-risk action problem over candidate answers and abstention under hard budgets.",
        "metrics": [
            "overall F1",
            "answerable F1",
            "unsupported-answer rate",
            "supported-answer rate",
            "answer rate",
            "abstain F1",
        ],
        "hints": [
            "input: Stage 5 candidate spans plus the strongest Stage 6 or Stage 4 baseline signals",
            "output: a learned action policy over {candidate_1 ... candidate_k, abstain}",
            "keep the method honest: better trade-off, not just lower answer rate",
        ],
        "modules": ["keelnet.action", "keelnet.balance", "keelnet.learn", "keelnet.metrics"],
        "template_path": DEFAULT_GENERIC_NOTEBOOK_TEMPLATE,
        "notes_markdown": (
            dedent(
                """
                ## Stage Notes

                ### Goal

                Turn grounded QA into an explicit utility-versus-risk action problem that chooses among answer candidates and `ABSTAIN`.

                ### Proof Status

                This stage is not proof that hallucination is solved. It is the first stage that treats grounded answering as an action-selection problem under a risk budget.

                ### Scope

                - input: Stage 5 candidate spans plus the strongest available Stage 6 or Stage 4 control signals
                - output: one action from `{candidate_1 ... candidate_k, ABSTAIN}`

                ### Main Change

                - replace thresholded candidate selection with explicit utility, explicit risk, and explicit abstain action choice

                ### Main Metrics

                - decision-aware overall `F1`
                - answerable `F1`
                - unsupported-answer rate
                - supported-answer rate
                - answer rate
                - abstain `F1`

                ### What This Stage Validates

                - unsupported-answer pressure can be built into training itself instead of only threshold search
                - explicit abstain choice works better than late gating if the risk model is strong enough
                - utility and safety costs can be balanced under explicit budgets

                ### Handoff Condition

                Do not call Stage 7 successful unless it improves the trade-off beyond Stage 6 without winning only by answering much less.
                """
            ).strip()
            + "\n"
        ),
        "config_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

                Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage7-v1`, `yourname-stage7-v2`, `yourname-stage7-v3`, and so on based on completed runs.

                This notebook tries to auto-fill `STAGE5_MODEL_DIR` from the latest completed Stage 5 run and `STAGE6_BALANCER_DIR` from the latest completed Stage 6 run under the current artifact root. It also tries to auto-fill `COMPARISON_BASELINE_EVAL_PATH` from Stage 4 if available, falling back to Stage 6.

                The config cell now prepares the upstream paths and default Stage 7 train/evaluate commands automatically. Review the printed paths, budgets, and command lists before launching a long run.
                """
            ).strip()
            + "\n"
        ),
        "smoke_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">4. Optional Smoke Test</h2>

                Use this section to run a small Stage 7 train/evaluate cycle before the full run. Keep it short so you can catch path, dependency, or runtime issues before spending a full Colab session.
                """
            ).strip()
            + "\n"
        ),
        "implementation_banner": implementation_banner(
            7,
            "run the new <code>keelnet.action</code> Stage 7 path and compare it against the strongest earlier direct-learning result.",
            "the action learner improves decision-aware performance under the unsupported-answer budget without winning only by over-abstaining.",
            "renaming Stage 6 outputs as Stage 7, calling threshold-only changes a new method, general hallucination claims.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Train And Evaluate The Stage 7 Action Learner

                This is the main Stage 7 run section. The default commands now use the real risk-budgeted action learner, which combines:

                - explicit abstain as an action
                - a candidate-conditioned risk head
                - dual budget updates during training
                - a hard risk shield at inference

                Use the smoke test first if you want a quick wiring check, then run the full train and evaluate commands here.
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Save Notes And Review Artifacts</h2>

                This cell creates teammate-friendly notes and a run summary for the Stage 7 design or run. Use it even before code exists so the design assumptions, artifact paths, and next implementation steps stay attached to the stage folder.
                """
            ).strip()
            + "\n"
        ),
        "final_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Final Check</h2>

                Stage 7 is only worth keeping if it does more than slightly conservatize Stage 6.

                Check all four:

                - abstain is treated as a real action, not a late threshold
                - unsupported-answer pressure is part of training, not only threshold search
                - the risk signal is candidate-conditioned
                - the gain is a better trade-off, not just fewer answered questions

                If those are not true yet, Stage 7 is still design work, not a completed method.
                """
            ).strip()
            + "\n"
        ),
        "share_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Share This Run</h2>

                This cell prints a minimal share-ready summary for teammates, saves it into the current run folder, and marks the run as completed so the next run becomes the next version.

                If Stage 7 is still design-only, use this cell to share the planned upstream artifacts, budgets, and implementation decisions.
                """
            ).strip()
            + "\n"
        ),
        "config_code": stage7_config_code(),
        "save_code": stage7_save_code(),
    },
    {
        "path": REPO_ROOT / "stages/08-joint-optimization/notebooks/stage-08-2-action-learner-calibrated-support-colab.ipynb",
        "branch": "main",
        "stage_number": 8,
        "stage_label": "Stage 8.2: Action Learner + Calibrated Support",
        "objective": "Use Stage 5 as the answer engine, inject calibrated Stage 4 support into a Stage 7-style action learner, and test the cleaned joined decision policy on an untouched final split.",
        "metrics": [
            "overall F1",
            "answerable F1",
            "unsupported-answer rate",
            "supported-answer rate",
            "answer rate",
            "comparison against Stage 5 and Stage 7",
        ],
        "hints": [
            "input: question and evidence only; Stage 9 trains its own proposal and support stack internally",
            "output: a risk-budgeted action learner with a hard calibrated support shield",
            "success means a better answer-quality versus groundedness balance than raw Stage 5 and plain Stage 7",
        ],
        "modules": ["keelnet.action", "keelnet.learn", "keelnet.control", "keelnet.verify", "keelnet.metrics"],
        "template_path": DEFAULT_GENERIC_NOTEBOOK_TEMPLATE,
        "notes_markdown": (
            dedent(
                """
                ## Stage Notes

                ### Goal

                Treat Stage 5 as the strong answer proposer, then let a calibrated action learner decide among answer candidates and abstain.

                ### Scope

                - input: frozen Stage 5 candidate answers plus the Stage 4 calibrated control artifact and optional Stage 6 prior
                - output: answer or `ABSTAIN`
                - comparison: raw Stage 5 learner versus Stage 7 versus joined Stage 8.2

                ### Main Change

                - keep Stage 5 fixed for answer capability
                - replace the lightweight final controller with a Stage 7-style action learner
                - inject calibrated Stage 4 support directly into the decision learner
                - keep a hard calibrated support shield at inference

                ### Main Metrics

                - overall `F1`
                - answerable `F1`
                - unsupported-answer rate
                - supported-answer rate
                - answer rate
                - abstain `F1`

                ### What This Stage Validates

                - the best current learner benefits from calibrated support inside the decision learner, not only after it
                - calibrated support information still adds value when Stage 5 already proposes strong answers
                - the joined action path is a better practical frontier than raw direct learning alone

                ### Handoff Condition

                Do not call Stage 8.2 successful unless it clearly improves the Stage 5 and Stage 7 trade-off without collapsing into Stage 4-style over-conservatism.
                """
            ).strip()
            + "\n"
        ),
        "config_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

                Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage8-2-v1`, `yourname-stage8-2-v2`, `yourname-stage8-2-v3`, and so on based on completed runs.

                This notebook tries to auto-fill `STAGE5_MODEL_DIR` from the latest completed Stage 5 run, `STAGE6_BALANCER_DIR` from the latest completed Stage 6 run, and `CONTROL_EVAL_PATH` from the latest completed Stage 4 run under the current artifact root. It also tries to auto-fill `COMPARISON_BASELINE_EVAL_PATH` from Stage 7 first, then Stage 5.

                The config cell prepares the upstream paths and default Stage 8 train/evaluate commands automatically. Review the printed paths, thresholds, and command lists before launching a full Colab run.
                """
            ).strip()
            + "\n"
        ),
        "smoke_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">4. Optional Smoke Test</h2>

                Use this section to run a small Stage 8 train/evaluate cycle before the full run. Keep it short so you can catch path, dependency, or runtime issues before spending a full hosted-Colab session.
                """
            ).strip()
            + "\n"
        ),
        "implementation_banner": implementation_banner(
            8,
            "run the new <code>keelnet.action</code> path to freeze Stage 5, inject calibrated verifier support from Stage 4, and train the decision learner on top.",
            "the joined action system beats raw Stage 5 and improves on plain Stage 7 under the cleaned split protocol.",
            "retraining Stage 5 end to end again, renaming Stage 5 thresholds as a new method, broad hallucination claims.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Train And Evaluate The Stage 8.2 Action Learner

                This is the main Stage 8 run section. The default commands now use the joined v2 path, which combines:

                - a frozen Stage 5 answer engine
                - calibrated Stage 4 verifier support as the learner's groundedness signal
                - the Stage 7 action learner over answer candidates and abstain
                - an optional Stage 6 prior feature
                - a hard calibrated support shield at inference

                Use the smoke test first if you want a quick wiring check, then run the full train and evaluate commands here.
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Save Notes And Review Artifacts</h2>

                This cell creates teammate-friendly note files inside the current run folder and lists the current artifacts. Update the generated notes as you learn what does and does not work in Stage 8.2: Action Learner + Calibrated Support.
                """
            ).strip()
            + "\n"
        ),
        "final_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Final Check</h2>

                Stage 8.2 only matters if the joined action learner actually combines the strengths of both paths.

                Check all four:

                - Stage 5 answer quality remains meaningfully strong
                - unsupported answers go down compared with raw Stage 5
                - the gain is not just over-abstention or a tiny threshold trick
                - the joined action frontier is easier to defend than either path alone

                If those are not true yet, Stage 8.2 is still an experiment, not a convincing wrap-up stage.
                """
            ).strip()
            + "\n"
        ),
        "share_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Share This Run</h2>

                This cell prints a minimal share-ready summary for teammates, saves it into the current run folder, and marks the run as completed so the next run becomes the next version.

                Use it to capture whether the joined action learner is actually stronger than raw Stage 5 and plain Stage 7, and whether the safety shield is doing useful work.
                """
            ).strip()
            + "\n"
        ),
        "config_code": stage8_config_code(),
    },
    {
        "path": REPO_ROOT / "stages/09-risk-generalization/notebooks/stage-09-risk-generalization-colab.ipynb",
        "branch": "main",
        "stage_number": 9,
        "stage_label": "Stage 9: Risk Generalization",
        "objective": "Run the exact isolated Stage 9 Mermaid logic in a dedicated module and test whether stronger tail-risk control improves held-out safety transfer without collapsing answerable utility.",
        "metrics": [
            "overall F1",
            "answerable F1",
            "unsupported-answer rate",
            "supported-answer rate",
            "answer rate",
            "over-abstain rate",
            "comparison against Stage 8.2 and Stage 4",
        ],
        "hints": [
            "input: question and evidence only; Stage 9 trains its own proposal and support stack internally",
            "logic: candidate plus uncertainty features feed utility, risk, and abstain heads before support shielding and the final answer-versus-abstain boundary",
            "success means a smaller validation-to-held-out safety gap than Stage 8.2 while keeping answerable F1 meaningfully above the Stage 4 anchor",
        ],
        "modules": ["keelnet.stage9"],
        "template_path": DEFAULT_GENERIC_NOTEBOOK_TEMPLATE,
        "notes_markdown": (
            dedent(
                """
                ## Stage Notes

                ### Goal

                Treat Stage 9 as a clean isolated architecture: keep the same answer-versus-abstain problem, but implement the Mermaid logic directly instead of routing through the old Stage 7 or Stage 8 action code.

                ### Scope

                - input: question and evidence only; Stage 9 trains its own proposal and support stack internally
                - output: answer or `ABSTAIN`
                - comparison: Stage 9 run versus Stage 8.2 first, with Stage 4 as an optional safety anchor

                ### Current Mermaid Logic

                - `Frozen: Base QA Model` -> `Modular: Top-K Candidate Generator`
                - `Frozen: Support / Evidence Scorer` -> `Post-hoc: Support Calibration`
                - `Modular: Candidate + Uncertainty Feature Builder` -> `Utility Head`, `Risk Head`, `Abstain Head`
                - `Candidate-Set Interaction Signals` strengthen the abstain path and `Optional Domain Signal or Domain Features` can support the risk path
                - `Post-hoc: Risk Calibration` plus `Post-hoc: Hard Support Shield` feed `Risk-Adjusted Action Scoring`
                - `Post-hoc: Better Abstain Boundary` -> final answer or `ABSTAIN`

                ### Current Training Logic

                - `Utility Supervision` supports the utility head
                - `Risk Supervision` plus `Tail-Risk Training` support the risk head
                - `Abstain Supervision` supports the abstain head
                - `Joint Decision Loss` should align the final answer-versus-abstain behavior

                ### What This Notebook Applies Today

                - a dedicated `keelnet.stage9` training and evaluation path
                - an internally trained proposal model that is frozen for Stage 9 candidate generation
                - an internally trained support scorer plus Stage 9 support calibration
                - optional comparison hooks for Stage 8.2 and the Stage 4 anchor

                ### What This Notebook Still Does Not Add

                - online adaptation, fallback pipelines, or human-in-the-loop feedback
                - broad domain-shift claims beyond the current controlled benchmark

                ### Handoff Condition

                Do not call Stage 9 successful unless it reduces the held-out safety gap that Stage 8.2 still shows while preserving a meaningful answer-quality advantage over the modular Stage 4 anchor.
                """
            ).strip()
            + "\n"
        ),
        "config_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">2. Configure The Run</h2>

                Set `AUTHOR_NAME` to your name. This notebook builds the stage-specific `RUN_NAME` automatically as `yourname-stage9-v1`, `yourname-stage9-v2`, `yourname-stage9-v3`, and so on based on completed runs.

                This notebook does not require any external Stage 5 or Stage 4 artifact to run. It only tries to auto-fill optional comparison paths from Stage 8.2 first, then Stage 7, then Stage 5 so you still have matched anchors when they exist.

                The config cell now points to the isolated `keelnet.stage9` path as a self-contained pipeline: Stage 9 trains its own proposal model, trains its own support scorer, freezes both inside the run, then learns the candidate-level utility/risk/abstain decision layer with explicit risk calibration and a tighter abstain boundary. Review the printed base-model choice, hyperparameters, and command lists before launching a full run.
                """
            ).strip()
            + "\n"
        ),
        "smoke_markdown": (
            dedent(
                """
                <h2 style="color: #1d4ed8;">4. Optional Smoke Test</h2>

                Use this section to run a small Stage 9 train/evaluate cycle before the full run. Keep it short so you can catch path, dependency, or runtime issues before spending a full hosted-Colab session on a longer tail-risk experiment.
                """
            ).strip()
            + "\n"
        ),
        "implementation_banner": implementation_banner(
            9,
            "run the dedicated <code>keelnet.stage9</code> experiment that implements the current Stage 9 Mermaid architecture directly and compares it cleanly against Stage 8.2 and Stage 4.",
            "the held-out unsupported-answer behavior is safer than Stage 8.2 without giving back all of the learned answer-quality gains.",
            "broad domain-shift claims, online adaptation, fallback deployment systems.",
        ),
        "implementation_markdown": (
            dedent(
                """
                ## IMPLEMENTATION 1: Train And Evaluate The Stage 9 Risk-Generalization Run

                This is the main Stage 9 run section. It now uses the dedicated `keelnet.stage9` implementation rather than reusing the old action learner shell, and it follows the current Stage 9 diagram directly:

                - Stage 9 trains and freezes its own answer proposal model
                - Stage 9 trains and calibrates its own support scorer as the groundedness path
                - candidate plus uncertainty features feeding utility, risk, and abstain heads
                - candidate-set interaction cues strengthening abstention and optional domain cues strengthening risk sensitivity
                - post-hoc risk calibration plus hard support shielding before final risk-adjusted action scoring
                - a more conservative answer-versus-abstain decision boundary

                The isolated implementation applies that logic through:

                - internal proposal-model training, then frozen Top-K candidate generation inside Stage 9
                - internal support-scorer training plus support calibration inside Stage 9
                - candidate plus uncertainty features, candidate-set interaction features, and optional domain/context features
                - explicit utility, risk, and abstain heads
                - post-hoc risk calibration plus a margin-based abstain boundary
                - optional comparison against the latest Stage 8.2 run and the Stage 4 safety anchor
                """
            ).strip()
            + "\n"
        ),
        "save_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Save Notes And Review Artifacts</h2>

                This cell creates teammate-friendly note files inside the current run folder and lists the current artifacts. Update the generated notes as you learn whether Stage 9 actually tightens held-out risk control relative to Stage 8.2.
                """
            ).strip()
            + "\n"
        ),
        "final_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Final Check</h2>

                Stage 9 only matters if the same learned path becomes safer in a way that transfers.

                Check all four:

                - unsupported answers are lower than Stage 8.2 on held-out data
                - the gain is not explained only by lower answer rate or much higher over-abstention
                - answerable `F1` stays meaningfully above the Stage 4 anchor
                - the result is easier to defend as real risk generalization rather than threshold luck

                If those are not true yet, Stage 9 is still a setup notebook for the next experiment block, not a solved stage.
                """
            ).strip()
            + "\n"
        ),
        "share_markdown": (
            dedent(
                """
                <h2 style="color: #15803d;">Share This Run</h2>

                This cell prints a minimal share-ready summary for teammates, saves it into the current run folder, and marks the run as completed so the next run becomes the next version.

                Use it to capture whether Stage 9 genuinely closes the held-out safety gap versus Stage 8.2 and how close it gets to the Stage 4 anchor without losing the learned answer-quality gains.
                """
            ).strip()
            + "\n"
        ),
        "config_code": stage9_config_code(),
    },
]


FINAL_COMPARISON_NOTEBOOK = {
    "path": REPO_ROOT / "analysis/notebooks/final-comparison-colab.ipynb",
    "branch": "main",
}

FINAL_COMPARISON_NOTES_MARKDOWN = (
    dedent(
        """
        ## Comparison Notes

        Use this notebook after meaningful stage runs already exist in the artifact store.

        ### Goal

        Build one trustworthy comparison across the strongest completed runs without turning this into another method stage.

        ### What This Notebook Should Produce

        - one comparison table across Stage 1 baseline, Stage 1 abstain, Stage 4, Stage 5, Stage 6, Stage 7, and Stage 8 when artifacts exist
        - paper-ready figures saved into the run folder
        - a short written takeaway that says which trade-off looks strongest and which stages are still missing

        ### Interpretation Rules

        - do not overclaim from missing rows
        - prefer trade-off plots over single-metric winners
        - treat unsupported-answer rate as a primary constraint, not a side metric
        """
    ).strip()
    + "\n"
)

FINAL_COMPARISON_CONFIG_MARKDOWN = (
    dedent(
        """
        <h2 style="color: #1d4ed8;">2. Configure The Comparison Run</h2>

        Set `AUTHOR_NAME` to your name if you want the notebook to prefer your own completed runs first.

        This notebook auto-finds the latest completed artifact for:

        - Stage 1 baseline and abstain
        - Stage 4 fixed control
        - Stage 5 support-constrained learner
        - Stage 6 adaptive balancer
        - Stage 7 risk-budgeted action learner
        - Stage 8 hybrid Stage 5 plus calibrated control

        You can override any discovered path directly in the config cell before building the comparison.
        """
    ).strip()
    + "\n"
)

FINAL_COMPARISON_DISCOVERY_MARKDOWN = (
    dedent(
        """
        <h2 style="color: #1d4ed8;">4. Discover And Build The Comparison</h2>

        Run this section after you review the printed artifact paths. It loads each available eval file, normalizes the metrics into one table, and saves comparison figures into the current run folder.
        """
    ).strip()
    + "\n"
)

FINAL_COMPARISON_IMPLEMENTATION_BANNER = (
    dedent(
        """
        <div style="border-left: 6px solid #c2410c; background: #fff7ed; padding: 12px 16px; border-radius: 8px;">
        <strong>Comparison Starts Here</strong><br/>
        Sections 1-4 prepare the runtime and artifact paths. This section turns the completed stage artifacts into one comparison table and graph set.
        <ul>
          <li><strong>Start here:</strong> confirm the auto-filled eval paths, then run the comparison cell below.</li>
          <li><strong>Finish here:</strong> you have a saved metrics table, saved figures, and a short textual takeaway under the comparison run folder.</li>
          <li><strong>Out of scope:</strong> retraining models, silently changing earlier metrics, or inventing missing stage results.</li>
        </ul>
        </div>
        """
    ).strip()
    + "\n"
)

FINAL_COMPARISON_IMPLEMENTATION_MARKDOWN = (
    dedent(
        """
        ## IMPLEMENTATION 1: Load Stage Evals And Generate Comparison Figures

        This is the main comparison section. It does four things:

        1. load each available stage eval JSON
        2. normalize the metrics into one table
        3. save comparison figures into `FIGURES_DIR`
        4. save a machine-readable comparison summary into `OUTPUT_ROOT`

        Missing artifact paths are handled gracefully and reported in the notebook output.
        """
    ).strip()
    + "\n"
)

FINAL_COMPARISON_NOTE_TEMPLATE_MARKDOWN = (
    dedent(
        """
        ## Comparison Note Template

        Update the notes for the final analysis rather than a single training run.

        Use this structure:

        - Coverage
        - Missing stages or missing artifacts
        - Strongest result by overall F1
        - Strongest result under the unsupported-answer constraint
        - Best trade-off plot to cite in the paper
        - Main takeaway
        - Next paper edits
        """
    ).strip()
    + "\n"
)

FINAL_COMPARISON_SAVE_MARKDOWN = (
    dedent(
        """
        <h2 style="color: #15803d;">Save Notes And Review Comparison Artifacts</h2>

        This cell saves the comparison table, figure list, and a note template for the final write-up. Use it after the comparison cell succeeds.
        """
    ).strip()
    + "\n"
)

FINAL_COMPARISON_FINAL_MARKDOWN = (
    dedent(
        """
        <h2 style="color: #15803d;">Final Check</h2>

        A useful wrap-up comparison is not just a pretty graph.

        Check all four:

        - the rows come from real saved artifacts, not manual copying
        - the stage labels match the eval files they came from
        - the graphs reflect the same metrics the paper discusses
        - the takeaway is honest about missing stages or failed runs
        """
    ).strip()
    + "\n"
)

FINAL_COMPARISON_SHARE_MARKDOWN = (
    dedent(
        """
        <h2 style="color: #15803d;">Share This Comparison</h2>

        This cell saves a small share note and marks the comparison run as completed. Use it after you review the figures and decide which ones belong in the paper or slides.
        """
    ).strip()
    + "\n"
)


def final_comparison_intro_markdown() -> str:
    return (
        dedent(
            """
            # KeelNet Final Comparison Template

            Use this notebook after the method stages are done to compare the strongest completed artifacts in one place.

            Recommended flow:

            1. run setup and validation
            2. confirm the discovered eval paths
            3. generate the comparison table and figures
            4. save the summary artifacts for the paper
            """
        ).strip()
        + "\n"
    )


def final_comparison_config_code() -> str:
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
            RUN_BASENAME = f"{AUTHOR_NAME}-final-comparison"
            ARTIFACTS_ROOT = PROJECT_STORAGE_DIR / "artifacts" / "final_comparison_colab"
            COMPLETION_MARKER_NAME = "RUN_COMPLETED.txt"

            NOTEBOOK_TITLE = "KeelNet Final Comparison"
            NOTEBOOK_OBJECTIVE = "Aggregate the strongest completed stage artifacts into one comparison table and one reusable figure set."
            TARGET_METRICS = [
                "overall F1",
                "answerable F1",
                "unsupported-answer rate",
                "abstain F1",
                "answer rate",
                "supported-answer rate",
            ]
            SUGGESTED_FIGURES = [
                "overall F1 vs unsupported-answer rate",
                "answerable F1 vs unsupported-answer rate",
                "summary bar chart across key metrics",
                "answer rate vs supported-answer rate for support-aware stages",
            ]


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


            def completed_run_dirs(root: Path, *, run_prefix: str | None = None) -> list[Path]:
                if not root.exists():
                    return []

                pattern = re.compile(rf"^{re.escape(run_prefix)}-v(\\d+)$") if run_prefix is not None else None
                runs: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if not (child / COMPLETION_MARKER_NAME).exists():
                        continue
                    if pattern is not None and not pattern.match(child.name):
                        continue
                    runs.append(child)
                return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name))


            def default_upstream_path(stage_folder: str, relative_path: str, *, preferred_run_prefix: str | None = None) -> str | None:
                stage_root = PROJECT_STORAGE_DIR / "artifacts" / stage_folder
                ordered_runs: list[Path] = []

                if preferred_run_prefix is not None:
                    ordered_runs.extend(reversed(completed_run_dirs(stage_root, run_prefix=preferred_run_prefix)))

                for run_dir in reversed(completed_run_dirs(stage_root)):
                    if run_dir not in ordered_runs:
                        ordered_runs.append(run_dir)

                relative = Path(relative_path)
                for run_dir in ordered_runs:
                    candidate = run_dir / relative
                    if candidate.exists():
                        return str(candidate)
                return None


            def normalize_optional_path(value: str | Path | None) -> Path | None:
                if value is None:
                    return None
                candidate = Path(value).expanduser().resolve()
                if not candidate.exists():
                    print(f"Missing artifact path: {candidate}")
                    return None
                return candidate


            RUN_VERSION = max(completed_versions(ARTIFACTS_ROOT, RUN_BASENAME), default=0) + 1
            RUN_NAME = f"{RUN_BASENAME}-v{RUN_VERSION}"
            OUTPUT_ROOT = ARTIFACTS_ROOT / RUN_NAME
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUN_MARKER_FILE = OUTPUT_ROOT / "RUN_STARTED.txt"
            RUN_NOTES_FILE = OUTPUT_ROOT / "run-notes-template.md"
            RUN_SUMMARY_FILE = OUTPUT_ROOT / "run-summary.json"
            COMPLETION_MARKER_FILE = OUTPUT_ROOT / COMPLETION_MARKER_NAME
            FIGURES_DIR = OUTPUT_ROOT / "figures"
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            COMPARISON_TABLE_CSV = OUTPUT_ROOT / "comparison_metrics.csv"
            COMPARISON_TABLE_JSON = OUTPUT_ROOT / "comparison_metrics.json"
            COMPARISON_SUMMARY_JSON = OUTPUT_ROOT / "comparison_summary.json"

            __NOTEBOOK_ARCHIVE_CONFIG_CODE__

            STAGE1_BASELINE_EVAL_PATH = normalize_optional_path(
                default_upstream_path("stage1_colab", "baseline_eval.json", preferred_run_prefix=f"{AUTHOR_NAME}-stage1")
            )
            STAGE1_ABSTAIN_EVAL_PATH = normalize_optional_path(
                default_upstream_path("stage1_colab", "abstain_eval.json", preferred_run_prefix=f"{AUTHOR_NAME}-stage1")
            )
            STAGE4_CONTROL_EVAL_PATH = normalize_optional_path(
                default_upstream_path("stage4_colab", "control_eval.json", preferred_run_prefix=f"{AUTHOR_NAME}-stage4")
            )
            STAGE5_LEARNER_EVAL_PATH = normalize_optional_path(
                default_upstream_path("stage5_colab", "learner_eval.json", preferred_run_prefix=f"{AUTHOR_NAME}-stage5")
            )
            STAGE6_BALANCE_EVAL_PATH = normalize_optional_path(
                default_upstream_path("stage6_colab", "balance_eval.json", preferred_run_prefix=f"{AUTHOR_NAME}-stage6")
            )
            STAGE7_ACTION_EVAL_PATH = normalize_optional_path(
                default_upstream_path("stage7_colab", "risk_action_eval.json", preferred_run_prefix=f"{AUTHOR_NAME}-stage7")
            )
            STAGE8_HYBRID_EVAL_PATH = normalize_optional_path(
                default_upstream_path("stage8_colab", "hybrid_eval.json", preferred_run_prefix=f"{AUTHOR_NAME}-stage8")
            )
            STAGE8_2_ACTION_EVAL_PATH = normalize_optional_path(
                default_upstream_path("stage8_2_colab", "hybrid_eval.json", preferred_run_prefix=f"{AUTHOR_NAME}-stage8-2")
            )

            COMPARISON_STAGE_SPECS = [
                {
                    "label": "Stage 1 Baseline",
                    "family": "stage1",
                    "eval_path": STAGE1_BASELINE_EVAL_PATH,
                },
                {
                    "label": "Stage 1 Abstain",
                    "family": "stage1",
                    "eval_path": STAGE1_ABSTAIN_EVAL_PATH,
                },
                {
                    "label": "Stage 4 Fixed Control",
                    "family": "stage4",
                    "eval_path": STAGE4_CONTROL_EVAL_PATH,
                },
                {
                    "label": "Stage 5 Learner",
                    "family": "stage5",
                    "eval_path": STAGE5_LEARNER_EVAL_PATH,
                },
                {
                    "label": "Stage 6 Adaptive Balance",
                    "family": "stage6",
                    "eval_path": STAGE6_BALANCE_EVAL_PATH,
                },
                {
                    "label": "Stage 7 Action Learner",
                    "family": "stage7",
                    "eval_path": STAGE7_ACTION_EVAL_PATH,
                },
                {
                    "label": "Stage 8 Hybrid",
                    "family": "stage8",
                    "eval_path": STAGE8_HYBRID_EVAL_PATH,
                },
                {
                    "label": "Stage 8.2 Action + Calibrated Support",
                    "family": "stage8_2",
                    "eval_path": STAGE8_2_ACTION_EVAL_PATH,
                },
            ]

            AVAILABLE_STAGE_LABELS = [
                spec["label"]
                for spec in COMPARISON_STAGE_SPECS
                if spec["eval_path"] is not None
            ]
            MISSING_STAGE_LABELS = [
                spec["label"]
                for spec in COMPARISON_STAGE_SPECS
                if spec["eval_path"] is None
            ]

            RUN_MARKER_FILE.write_text(
                "\\n".join(
                    [
                        f"notebook={NOTEBOOK_TITLE}",
                        f"run_name={RUN_NAME}",
                        f"run_version=v{RUN_VERSION}",
                        f"runtime_mode={RUNTIME_MODE}",
                        f"repo_dir={REPO_DIR}",
                        f"project_storage_dir={PROJECT_STORAGE_DIR}",
                        f"git_branch={CURRENT_BRANCH}",
                        f"stage1_baseline_eval_path={STAGE1_BASELINE_EVAL_PATH}",
                        f"stage1_abstain_eval_path={STAGE1_ABSTAIN_EVAL_PATH}",
                        f"stage4_control_eval_path={STAGE4_CONTROL_EVAL_PATH}",
                        f"stage5_learner_eval_path={STAGE5_LEARNER_EVAL_PATH}",
                        f"stage6_balance_eval_path={STAGE6_BALANCE_EVAL_PATH}",
                        f"stage7_action_eval_path={STAGE7_ACTION_EVAL_PATH}",
                        f"stage8_hybrid_eval_path={STAGE8_HYBRID_EVAL_PATH}",
                        f"stage8_2_action_eval_path={STAGE8_2_ACTION_EVAL_PATH}",
                        "status=configured",
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
            print(f"Figures dir: {FIGURES_DIR}")
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            print(f"Run marker file: {RUN_MARKER_FILE}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("Target metrics:", ", ".join(TARGET_METRICS))
            print("Suggested figures:", ", ".join(SUGGESTED_FIGURES))
            print("")
            print("Discovered eval paths:")
            for spec in COMPARISON_STAGE_SPECS:
                print(f"- {spec['label']}: {spec['eval_path']}")
            print("")
            print("Available rows:", ", ".join(AVAILABLE_STAGE_LABELS) if AVAILABLE_STAGE_LABELS else "none")
            print("Missing rows:", ", ".join(MISSING_STAGE_LABELS) if MISSING_STAGE_LABELS else "none")


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
            """
        ).replace("__NOTEBOOK_ARCHIVE_CONFIG_CODE__", NOTEBOOK_ARCHIVE_CONFIG_CODE.rstrip()).strip()
        + "\n"
    )


def final_comparison_implementation_code() -> str:
    return (
        dedent(
            """
            import json
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            try:
                plt.style.use("seaborn-v0_8-whitegrid")
            except OSError:
                pass


            def load_eval_payload(path: Path) -> dict:
                return json.loads(path.read_text(encoding="utf-8"))


            def _stage1_answer_rate(payload: dict, metrics: dict) -> float | None:
                predictions = payload.get("dev_predictions")
                if isinstance(predictions, dict) and predictions:
                    total = len(predictions)
                    answered = sum(
                        1
                        for prediction in predictions.values()
                        if str(prediction.get("decision", "abstain")).lower() == "answer"
                    )
                    return 100.0 * answered / total if total > 0 else None

                total = float(metrics.get("answerable_count", 0.0)) + float(metrics.get("unanswerable_count", 0.0))
                return None if total <= 0 else 100.0 - float(metrics.get("abstain_recall", 0.0))


            def extract_comparison_row(spec: dict, payload: dict) -> dict:
                family = spec["family"]

                if family == "stage1":
                    metrics = payload.get("dev_metrics", {}) or {}
                    mix = {
                        "answer_rate": _stage1_answer_rate(payload, metrics),
                        "supported_answer_rate": None,
                        "unsupported_among_answers_rate": None,
                    }
                    selected_operating_point = payload.get("selected_threshold")
                elif family == "stage4":
                    metrics = payload.get("control_dev_metrics", {}) or {}
                    mix = payload.get("control_dev_mix", {}) or {}
                    selected_operating_point = payload.get("selected_config")
                elif family == "stage5":
                    metrics = payload.get("dev_metrics", {}) or {}
                    mix = payload.get("dev_mix", {}) or {}
                    selected_operating_point = payload.get("selected_keep_threshold")
                elif family == "stage6":
                    metrics = payload.get("final_metrics", {}) or payload.get("dev_metrics", {}) or payload.get("test_metrics", {}) or {}
                    mix = payload.get("final_mix", {}) or payload.get("dev_mix", {}) or payload.get("test_mix", {}) or {}
                    selected_operating_point = payload.get("selected_candidate_threshold") or payload.get("selected_threshold")
                elif family == "stage7":
                    metrics = payload.get("final_metrics", {}) or payload.get("dev_metrics", {}) or payload.get("test_metrics", {}) or {}
                    mix = payload.get("final_mix", {}) or payload.get("dev_mix", {}) or payload.get("test_mix", {}) or {}
                    overabstain = (
                        payload.get("final_overabstain", {})
                        or payload.get("dev_overabstain", {})
                        or payload.get("test_overabstain", {})
                        or {}
                    )
                    selected_operating_point = payload.get("selected_risk_threshold")
                elif family in {"stage8", "stage8_2"}:
                    metrics = payload.get("final_metrics", {}) or payload.get("dev_metrics", {}) or payload.get("test_metrics", {}) or {}
                    mix = payload.get("final_mix", {}) or payload.get("dev_mix", {}) or payload.get("test_mix", {}) or {}
                    selected_operating_point = payload.get("selected_risk_threshold") or payload.get("selected_candidate_threshold")
                else:
                    raise ValueError(f"Unsupported comparison family: {family}")

                row = {
                    "label": spec["label"],
                    "family": family,
                    "source_path": str(spec["eval_path"]),
                    "evaluation_split": payload.get("final_eval_split", "dev"),
                    "selected_operating_point": selected_operating_point,
                    "overall_f1": metrics.get("overall_f1"),
                    "answerable_f1": metrics.get("answerable_f1"),
                    "unsupported_answer_rate": metrics.get("unsupported_answer_rate"),
                    "abstain_f1": metrics.get("abstain_f1"),
                    "answer_rate": mix.get("answer_rate"),
                    "supported_answer_rate": mix.get("supported_answer_rate"),
                    "unsupported_among_answers_rate": mix.get("unsupported_among_answers_rate"),
                    "max_unsupported_answer_rate": payload.get("max_unsupported_answer_rate"),
                }
                if family == "stage7":
                    row["overabstain_rate"] = overabstain.get("overabstain_rate")
                    row["max_overabstain_rate"] = payload.get("max_overabstain_rate")
                else:
                    row["overabstain_rate"] = None
                    row["max_overabstain_rate"] = None
                return row


            comparison_records = []
            missing_specs = []
            for spec in COMPARISON_STAGE_SPECS:
                eval_path = spec["eval_path"]
                if eval_path is None:
                    missing_specs.append(spec)
                    continue
                payload = load_eval_payload(eval_path)
                comparison_records.append(extract_comparison_row(spec, payload))

            comparison_df = pd.DataFrame(comparison_records)
            if comparison_df.empty:
                raise RuntimeError(
                    "No comparison rows were loaded. Confirm that at least one completed stage eval file exists."
                )

            stage_order = [spec["label"] for spec in COMPARISON_STAGE_SPECS]
            comparison_df["label"] = pd.Categorical(comparison_df["label"], categories=stage_order, ordered=True)
            comparison_df = comparison_df.sort_values("label").reset_index(drop=True)

            rounded_columns = [
                "overall_f1",
                "answerable_f1",
                "unsupported_answer_rate",
                "abstain_f1",
                "answer_rate",
                "supported_answer_rate",
                "unsupported_among_answers_rate",
                "overabstain_rate",
            ]
            display_df = comparison_df.copy()
            for column in rounded_columns:
                if column in display_df:
                    display_df[column] = display_df[column].map(lambda value: round(float(value), 2) if pd.notna(value) else value)

            print("Loaded comparison rows:")
            print(", ".join(str(label) for label in comparison_df["label"].tolist()))
            if missing_specs:
                print("Missing rows:")
                for spec in missing_specs:
                    print(f"- {spec['label']}")
            display(display_df)

            COMPARISON_TABLE_CSV.write_text(display_df.to_csv(index=False), encoding="utf-8")
            COMPARISON_TABLE_JSON.write_text(json.dumps(comparison_records, indent=2) + "\\n", encoding="utf-8")

            colors = {
                "stage1": "#475569",
                "stage4": "#0284c7",
                "stage5": "#dc2626",
                "stage6": "#7c3aed",
                "stage7": "#059669",
                "stage8": "#ea580c",
                "stage8_2": "#f59e0b",
            }
            family_display_names = {
                "stage1": "Stage 1 variants",
                "stage4": "Stage 4 fixed control",
                "stage5": "Stage 5 learner",
                "stage6": "Stage 6 adaptive balance",
                "stage7": "Stage 7 action learner",
                "stage8": "Stage 8 hybrid",
                "stage8_2": "Stage 8.2 joined action",
            }
            markers = {
                "stage1": "o",
                "stage4": "s",
                "stage5": "^",
                "stage6": "D",
                "stage7": "P",
                "stage8": "X",
                "stage8_2": "*",
            }
            short_labels = {
                "Stage 1 Baseline": "S1 base",
                "Stage 1 Abstain": "S1 abs",
                "Stage 4 Fixed Control": "S4",
                "Stage 5 Learner": "S5",
                "Stage 6 Adaptive Balance": "S6",
                "Stage 7 Action Learner": "S7",
                "Stage 8 Hybrid": "S8",
                "Stage 8.2 Action + Calibrated Support": "S8.2",
            }
            budget_color = "#ef4444"
            budget_fill = "#fee2e2"


            def _short_label(label: str) -> str:
                return short_labels.get(str(label), str(label))


            def _family_legend_handles(*, include_budget_line: bool = False, include_budget_wall: bool = False) -> list:
                handles = [
                    Line2D(
                        [0],
                        [0],
                        marker=markers.get(family, "o"),
                        linestyle="",
                        markerfacecolor=colors.get(family, "#111827"),
                        markeredgecolor="#111827",
                        markeredgewidth=0.8,
                        markersize=8,
                        label=family_display_names.get(family, family),
                    )
                    for family in colors
                ]
                if include_budget_line:
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=budget_color,
                            linestyle="--",
                            linewidth=1.5,
                            label="20% unsupported-answer budget",
                        )
                    )
                if include_budget_wall:
                    handles.append(
                        Patch(
                            facecolor=budget_fill,
                            edgecolor=budget_color,
                            alpha=0.18,
                            label="20% budget wall",
                        )
                    )
                return handles


            def _scatter_by_family(ax, frame: pd.DataFrame, *, x: str, y: str, z: str | None = None, size: int = 120) -> None:
                for family, family_df in frame.groupby("family", sort=False):
                    scatter_kwargs = {
                        "s": size,
                        "c": colors.get(family, "#111827"),
                        "marker": markers.get(family, "o"),
                        "edgecolors": "#111827",
                        "linewidths": 0.8,
                        "alpha": 0.95,
                    }
                    if z is None:
                        ax.scatter(family_df[x], family_df[y], **scatter_kwargs)
                    else:
                        ax.scatter(family_df[x], family_df[y], family_df[z], depthshade=False, **scatter_kwargs)


            def _annotate_points(ax, frame: pd.DataFrame, *, x: str, y: str, z: str | None = None) -> None:
                for _, row in frame.iterrows():
                    if pd.isna(row[x]) or pd.isna(row[y]) or (z is not None and pd.isna(row[z])):
                        continue
                    if z is None:
                        ax.annotate(
                            _short_label(str(row["label"])),
                            (row[x], row[y]),
                            textcoords="offset points",
                            xytext=(6, 6),
                            fontsize=8.5,
                            bbox={
                                "boxstyle": "round,pad=0.18",
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.75,
                            },
                        )
                    else:
                        ax.text(
                            float(row[x]) + 0.15,
                            float(row[y]) + 0.05,
                            float(row[z]) + 0.05,
                            _short_label(str(row["label"])),
                            fontsize=8,
                        )


            overall_vs_unsupported_path = FIGURES_DIR / "overall-f1-vs-unsupported-answer-rate.png"
            answerable_vs_unsupported_path = FIGURES_DIR / "answerable-f1-vs-unsupported-answer-rate.png"
            summary_bar_path = FIGURES_DIR / "summary-metrics-bar-chart.png"
            support_mix_path = FIGURES_DIR / "answer-rate-vs-supported-answer-rate.png"
            three_metric_frontier_path = FIGURES_DIR / "three-metric-frontier-3d.png"

            scatter_df = comparison_df.dropna(subset=["overall_f1", "unsupported_answer_rate"]).copy()
            fig, ax = plt.subplots(figsize=(10.5, 6.5))
            if not scatter_df.empty:
                unsafe_limit = float(scatter_df["unsupported_answer_rate"].max()) + 1.5
                ax.axvspan(20.0, unsafe_limit, color=budget_fill, alpha=0.35, zorder=0)
            _scatter_by_family(ax, scatter_df, x="unsupported_answer_rate", y="overall_f1")
            _annotate_points(ax, scatter_df, x="unsupported_answer_rate", y="overall_f1")
            ax.set_title("Overall F1 vs Unsupported-Answer Rate")
            ax.set_xlabel("Unsupported-Answer Rate (%)")
            ax.set_ylabel("Overall F1")
            ax.axvline(20.0, color=budget_color, linestyle="--", linewidth=1.4)
            ax.legend(
                handles=_family_legend_handles(include_budget_line=True),
                title="System family",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
            fig.savefig(overall_vs_unsupported_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            scatter_df = comparison_df.dropna(subset=["answerable_f1", "unsupported_answer_rate"]).copy()
            fig, ax = plt.subplots(figsize=(10.5, 6.5))
            if not scatter_df.empty:
                unsafe_limit = float(scatter_df["unsupported_answer_rate"].max()) + 1.5
                ax.axvspan(20.0, unsafe_limit, color=budget_fill, alpha=0.35, zorder=0)
            _scatter_by_family(ax, scatter_df, x="unsupported_answer_rate", y="answerable_f1")
            _annotate_points(ax, scatter_df, x="unsupported_answer_rate", y="answerable_f1")
            ax.set_title("Answerable F1 vs Unsupported-Answer Rate")
            ax.set_xlabel("Unsupported-Answer Rate (%)")
            ax.set_ylabel("Answerable F1")
            ax.axvline(20.0, color=budget_color, linestyle="--", linewidth=1.4)
            ax.legend(
                handles=_family_legend_handles(include_budget_line=True),
                title="System family",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
            fig.savefig(answerable_vs_unsupported_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            bar_metrics = ["overall_f1", "answerable_f1", "unsupported_answer_rate", "abstain_f1"]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for axis, metric in zip(axes.flat, bar_metrics, strict=False):
                metric_df = comparison_df.dropna(subset=[metric]).copy()
                bars = axis.bar(
                    metric_df["label"].astype(str),
                    metric_df[metric],
                    color=[colors.get(family, "#111827") for family in metric_df["family"]],
                )
                axis.set_title(metric.replace("_", " ").title())
                axis.tick_params(axis="x", rotation=28)
                axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
                if metric == "unsupported_answer_rate":
                    axis.axhline(20.0, color=budget_color, linestyle="--", linewidth=1.4)
            fig.tight_layout()
            fig.savefig(summary_bar_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            support_df = comparison_df.dropna(subset=["answer_rate", "supported_answer_rate"]).copy()
            if not support_df.empty:
                fig, ax = plt.subplots(figsize=(10.5, 6.5))
                _scatter_by_family(ax, support_df, x="answer_rate", y="supported_answer_rate")
                _annotate_points(ax, support_df, x="answer_rate", y="supported_answer_rate")
                ax.set_title("Answer Rate vs Supported-Answer Rate")
                ax.set_xlabel("Answer Rate (%)")
                ax.set_ylabel("Supported-Answer Rate (%)")
                ax.legend(
                    handles=_family_legend_handles(),
                    title="System family",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                )
                fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
                fig.savefig(support_mix_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
            else:
                support_mix_path = None
                print("Support-aware mix plot skipped because no loaded rows expose answer-rate and supported-answer-rate together.")

            three_metric_df = comparison_df.dropna(
                subset=["unsupported_answer_rate", "overall_f1", "answerable_f1"]
            ).copy()
            if not three_metric_df.empty:
                fig = plt.figure(figsize=(11.5, 8.5))
                ax = fig.add_subplot(111, projection="3d")
                _scatter_by_family(
                    ax,
                    three_metric_df,
                    x="unsupported_answer_rate",
                    y="overall_f1",
                    z="answerable_f1",
                    size=130,
                )
                y_values = np.linspace(
                    float(three_metric_df["overall_f1"].min()) - 0.4,
                    float(three_metric_df["overall_f1"].max()) + 0.4,
                    2,
                )
                z_values = np.linspace(
                    float(three_metric_df["answerable_f1"].min()) - 0.4,
                    float(three_metric_df["answerable_f1"].max()) + 0.4,
                    2,
                )
                yy, zz = np.meshgrid(y_values, z_values)
                xx = np.full_like(yy, 20.0)
                ax.plot_surface(xx, yy, zz, color=budget_fill, alpha=0.18, linewidth=0.0, shade=False)
                _annotate_points(
                    ax,
                    three_metric_df,
                    x="unsupported_answer_rate",
                    y="overall_f1",
                    z="answerable_f1",
                )
                ax.set_title("Three-Metric Frontier")
                ax.set_xlabel("Unsupported-Answer Rate (%)")
                ax.set_ylabel("Overall F1")
                ax.set_zlabel("Answerable F1")
                ax.view_init(elev=22, azim=-58)
                ax.legend(
                    handles=_family_legend_handles(include_budget_wall=True),
                    title="System family",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                )
                fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
                fig.savefig(three_metric_frontier_path, dpi=220, bbox_inches="tight")
                plt.close(fig)
            else:
                three_metric_frontier_path = None
                print("3D frontier plot skipped because the loaded rows do not share all three core metrics.")

            available_figure_paths = [
                overall_vs_unsupported_path,
                answerable_vs_unsupported_path,
                summary_bar_path,
            ]
            if support_mix_path is not None:
                available_figure_paths.append(support_mix_path)
            if three_metric_frontier_path is not None:
                available_figure_paths.append(three_metric_frontier_path)

            best_overall_row = comparison_df.loc[comparison_df["overall_f1"].astype(float).idxmax()].to_dict()
            safest_row = comparison_df.loc[comparison_df["unsupported_answer_rate"].astype(float).idxmin()].to_dict()
            constrained_df = comparison_df[
                comparison_df["unsupported_answer_rate"].notna()
                & (comparison_df["unsupported_answer_rate"] <= 20.0)
            ].copy()
            best_under_budget_row = None
            if not constrained_df.empty:
                best_under_budget_row = constrained_df.loc[constrained_df["overall_f1"].astype(float).idxmax()].to_dict()

            comparison_summary = {
                "available_rows": [str(label) for label in comparison_df["label"].tolist()],
                "missing_rows": [spec["label"] for spec in missing_specs],
                "best_overall_f1_row": best_overall_row,
                "lowest_unsupported_answer_rate_row": safest_row,
                "best_overall_f1_under_20pct_budget": best_under_budget_row,
                "figure_paths": [str(path) for path in available_figure_paths],
            }
            COMPARISON_SUMMARY_JSON.write_text(json.dumps(comparison_summary, indent=2) + "\\n", encoding="utf-8")

            print("")
            print(f"Saved comparison table: {COMPARISON_TABLE_CSV}")
            print(f"Saved comparison records: {COMPARISON_TABLE_JSON}")
            print(f"Saved comparison summary: {COMPARISON_SUMMARY_JSON}")
            print("Saved figures:")
            for figure_path in available_figure_paths:
                print(f"- {figure_path}")
            print("")
            print("Best overall F1 row:", best_overall_row["label"])
            print("Lowest unsupported-answer rate row:", safest_row["label"])
            if best_under_budget_row is None:
                print("No loaded row satisfies the 20% unsupported-answer budget.")
            else:
                print("Best overall F1 row under the 20% unsupported-answer budget:", best_under_budget_row["label"])
            """
        ).strip()
        + "\n"
    )


def final_comparison_save_code() -> str:
    return (
        dedent(
            """
            captured_notebook_path = save_executed_notebook_snapshot()
            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)

            if "comparison_df" not in globals():
                print("Run the comparison cell first so the metrics table and figure list exist.")
            else:
                if not RUN_NOTES_FILE.exists():
                    available_rows = comparison_df["label"].astype(str).tolist()
                    figure_paths = []
                    if COMPARISON_SUMMARY_JSON.exists():
                        summary_payload = json.loads(COMPARISON_SUMMARY_JSON.read_text(encoding="utf-8"))
                        figure_paths = summary_payload.get("figure_paths", [])

                    RUN_NOTES_FILE.write_text(
                        "\\n".join(
                            [
                                "# KeelNet Final Comparison Notes",
                                "",
                                "## Coverage",
                                *[f"- {label}" for label in available_rows],
                                "",
                                "## Missing Stages Or Missing Artifacts",
                                *([f"- {label}" for label in MISSING_STAGE_LABELS] if MISSING_STAGE_LABELS else ["- none"]),
                                "",
                                "## Output Paths",
                                f"- Output folder: {OUTPUT_ROOT}",
                                f"- Figures dir: {FIGURES_DIR}",
                                f"- Comparison CSV: {COMPARISON_TABLE_CSV}",
                                f"- Comparison JSON: {COMPARISON_TABLE_JSON}",
                                f"- Comparison summary JSON: {COMPARISON_SUMMARY_JSON}",
                                f"- Executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
                                "",
                                "## Figures To Review",
                                *([f"- {path}" for path in figure_paths] if figure_paths else ["- run the comparison cell first"]),
                                "",
                                "## Main Takeaway",
                                "- ",
                                "",
                                "## Paper Follow-Ups",
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
                            "notebook": NOTEBOOK_TITLE,
                            "run_name": RUN_NAME,
                            "runtime_mode": RUNTIME_MODE,
                            "git_branch": CURRENT_BRANCH,
                            "output_root": str(OUTPUT_ROOT),
                            "figures_dir": str(FIGURES_DIR),
                            "comparison_csv": str(COMPARISON_TABLE_CSV),
                            "comparison_json": str(COMPARISON_TABLE_JSON),
                            "comparison_summary_json": str(COMPARISON_SUMMARY_JSON),
                            "available_rows": comparison_df["label"].astype(str).tolist(),
                            "missing_rows": MISSING_STAGE_LABELS,
                            "executed_notebook_dir": str(NOTEBOOK_ARCHIVE_DIR),
                            "executed_notebook_target": str(EXECUTED_NOTEBOOK_PATH),
                            "executed_notebook_saved": captured_notebook_path is not None,
                            "executed_notebook_instructions_file": str(EXECUTED_NOTEBOOK_INSTRUCTIONS_FILE),
                        },
                        indent=2,
                    )
                    + "\\n",
                    encoding="utf-8",
                )

            print(f"Notes template: {RUN_NOTES_FILE}")
            print(f"Run summary: {RUN_SUMMARY_FILE}")
            print(f"Executed notebook target: {EXECUTED_NOTEBOOK_PATH}")
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
            if mirrored_output_root is not None:
                print(f"Drive mirror: {mirrored_output_root}")
            print("Current files under OUTPUT_ROOT:")
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                print(path)
            """
        ).strip()
        + "\n"
    )


def final_comparison_share_code() -> str:
    return (
        dedent(
            """
            from datetime import datetime, timezone

            captured_notebook_path = save_executed_notebook_snapshot()
            mirrored_output_root = mirror_output_root(OUTPUT_ROOT)

            available_rows = comparison_df["label"].astype(str).tolist() if "comparison_df" in globals() else []
            best_overall_label = "<run comparison cell first>"
            lowest_unsupported_label = "<run comparison cell first>"
            figure_paths = []
            if COMPARISON_SUMMARY_JSON.exists():
                summary_payload = json.loads(COMPARISON_SUMMARY_JSON.read_text(encoding="utf-8"))
                best_overall = summary_payload.get("best_overall_f1_row")
                safest = summary_payload.get("lowest_unsupported_answer_rate_row")
                if isinstance(best_overall, dict):
                    best_overall_label = str(best_overall.get("label"))
                if isinstance(safest, dict):
                    lowest_unsupported_label = str(safest.get("label"))
                figure_paths = summary_payload.get("figure_paths", [])

            share_lines = [
                "# KeelNet Final Comparison Share Note",
                "",
                f"- runtime mode: {RUNTIME_MODE}",
                f"- branch name: {CURRENT_BRANCH}",
                f"- RUN_NAME: {RUN_NAME}",
                f"- available rows: {', '.join(available_rows) if available_rows else 'run comparison cell first'}",
                f"- best overall F1 row: {best_overall_label}",
                f"- lowest unsupported-answer-rate row: {lowest_unsupported_label}",
                f"- comparison CSV: {COMPARISON_TABLE_CSV}",
                f"- comparison summary JSON: {COMPARISON_SUMMARY_JSON}",
                f"- executed notebook archive target: {EXECUTED_NOTEBOOK_PATH}",
                "",
                "## Figure Paths",
                *([f"- {path}" for path in figure_paths] if figure_paths else ["- run comparison cell first"]),
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
            if captured_notebook_path is None:
                print(
                    "Automatic notebook capture was unavailable. "
                    f"If you want the notebook archive, save it manually to {EXECUTED_NOTEBOOK_PATH}."
                )
            print(f"Saved share note: {SHARE_NOTE_FILE}")
            print(f"Saved completion marker: {COMPLETION_MARKER_FILE}")
            if mirrored_output_root is not None:
                mirror_output_root(OUTPUT_ROOT)
            """
        ).strip()
        + "\n"
    )


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def scaffold_notebook(path: Path, *, template_path: Path = DEFAULT_GENERIC_NOTEBOOK_TEMPLATE) -> None:
    notebook = load_notebook(template_path)
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            metadata = cell.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata.pop("executionInfo", None)
                metadata.pop("outputId", None)
                colab_meta = metadata.get("colab")
                if isinstance(colab_meta, dict):
                    colab_meta.pop("base_uri", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    save_notebook(path, notebook)


def sync_stage_2() -> None:
    notebook = load_notebook(STAGE_2["path"])
    notebook["cells"][0]["source"] = source_lines(intro_markdown(STAGE_2["stage_label"], STAGE_2["stage_number"]))
    notebook["cells"][2]["source"] = source_lines(SETUP_MARKDOWN)
    notebook["cells"][3]["source"] = source_lines(setup_code(STAGE_2["branch"]))
    notebook["cells"][4]["source"] = source_lines(STAGE_2["config_markdown"])
    notebook["cells"][6]["source"] = source_lines(VALIDATE_MARKDOWN)
    notebook["cells"][7]["source"] = source_lines(VALIDATE_CODE)
    notebook["cells"][8]["source"] = source_lines(STAGE_2["smoke_markdown"])
    notebook["cells"][10]["source"] = source_lines(STAGE_2["implementation_banner"])
    notebook["cells"][11]["source"] = source_lines(STAGE_2["implementation_markdown"])
    notebook["cells"][13]["source"] = source_lines(STAGE_NOTE_TEMPLATE_MARKDOWN)
    notebook["cells"][14]["source"] = source_lines(STAGE_2["save_markdown"])
    notebook["cells"][16]["source"] = source_lines(STAGE_2["final_markdown"])
    notebook["cells"][17]["source"] = source_lines(STAGE_2["share_markdown"])
    save_notebook(STAGE_2["path"], notebook)


def sync_generic_stage(stage: dict) -> None:
    template_path = stage.get("template_path", DEFAULT_GENERIC_NOTEBOOK_TEMPLATE)
    if not stage["path"].exists():
        scaffold_notebook(stage["path"], template_path=template_path)
    notebook = load_notebook(stage["path"])
    if len(notebook.get("cells", [])) < 19:
        scaffold_notebook(stage["path"], template_path=template_path)
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
    notebook["cells"][7]["source"] = source_lines(VALIDATE_CODE)
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


def sync_final_comparison_notebook() -> None:
    spec = FINAL_COMPARISON_NOTEBOOK
    if not spec["path"].exists():
        scaffold_notebook(spec["path"], template_path=DEFAULT_GENERIC_NOTEBOOK_TEMPLATE)
    notebook = load_notebook(spec["path"])
    notebook["cells"][0]["source"] = source_lines(final_comparison_intro_markdown())
    notebook["cells"][1]["source"] = source_lines(FINAL_COMPARISON_NOTES_MARKDOWN)
    notebook["cells"][2]["source"] = source_lines(SETUP_MARKDOWN)
    notebook["cells"][3]["source"] = source_lines(setup_code(spec["branch"]))
    notebook["cells"][4]["source"] = source_lines(FINAL_COMPARISON_CONFIG_MARKDOWN)
    notebook["cells"][5]["source"] = source_lines(final_comparison_config_code())
    notebook["cells"][6]["source"] = source_lines(VALIDATE_MARKDOWN)
    notebook["cells"][7]["source"] = source_lines(VALIDATE_CODE)
    notebook["cells"][8]["source"] = source_lines(FINAL_COMPARISON_DISCOVERY_MARKDOWN)
    notebook["cells"][9]["source"] = source_lines("print(\"Comparison build starts in the next cell.\")\n")
    notebook["cells"][10]["source"] = source_lines(FINAL_COMPARISON_IMPLEMENTATION_BANNER)
    notebook["cells"][11]["source"] = source_lines(FINAL_COMPARISON_IMPLEMENTATION_MARKDOWN)
    notebook["cells"][12]["source"] = source_lines(final_comparison_implementation_code())
    notebook["cells"][13]["source"] = source_lines(FINAL_COMPARISON_NOTE_TEMPLATE_MARKDOWN)
    notebook["cells"][14]["source"] = source_lines(FINAL_COMPARISON_SAVE_MARKDOWN)
    notebook["cells"][15]["source"] = source_lines(final_comparison_save_code())
    notebook["cells"][16]["source"] = source_lines(FINAL_COMPARISON_FINAL_MARKDOWN)
    notebook["cells"][17]["source"] = source_lines(FINAL_COMPARISON_SHARE_MARKDOWN)
    notebook["cells"][18]["source"] = source_lines(final_comparison_share_code())
    save_notebook(spec["path"], notebook)


def main() -> None:
    sync_stage_2()
    for stage in GENERIC_STAGES:
        sync_generic_stage(stage)
    sync_final_comparison_notebook()


if __name__ == "__main__":
    main()
