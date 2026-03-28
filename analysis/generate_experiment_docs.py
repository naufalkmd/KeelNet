#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


GENERIC_METADATA_FILES = {
    "RUN_STARTED.txt",
    "RUN_COMPLETED.txt",
    "run-summary.json",
    "run-notes-template.md",
    "collab-share-note.md",
    "desktop.ini",
}

GENERIC_METADATA_DIRS = {
    "executed-notebook",
}


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    artifact_dir: str
    preferred_prefixes: tuple[str, ...]
    required_paths: tuple[str, ...]
    key_output_paths: tuple[str, ...]
    partial_markers: tuple[str, ...] = ()


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        key="stage1",
        label="Stage 1",
        artifact_dir="stage1_colab",
        preferred_prefixes=("codex-stage1-live", "naufal-stage1", "YOUR_NAME_HERE-stage1", "yourname-stage1"),
        required_paths=("baseline_eval.json", "abstain_eval.json"),
        key_output_paths=("baseline_eval.json", "abstain_eval.json", "baseline", "abstain"),
    ),
    StageSpec(
        key="stage2",
        label="Stage 2",
        artifact_dir="stage2_colab",
        preferred_prefixes=("naufal-stage2", "yourname-stage2"),
        required_paths=("verifier_eval.json", "verifier"),
        key_output_paths=("verifier_eval.json", "verifier"),
    ),
    StageSpec(
        key="stage2_5",
        label="Stage 2.5",
        artifact_dir="stage2_5_colab",
        preferred_prefixes=("naufal-stage2-5",),
        required_paths=("verifier_eval.json", "verifier"),
        key_output_paths=("verifier_eval.json", "verifier"),
    ),
    StageSpec(
        key="stage3",
        label="Stage 3",
        artifact_dir="stage3_colab",
        preferred_prefixes=("naufal-stage3",),
        required_paths=("calibration_eval.json",),
        key_output_paths=("calibration_eval.json", "qa_dev_reliability.png", "support_dev_reliability.png"),
    ),
    StageSpec(
        key="stage4",
        label="Stage 4",
        artifact_dir="stage4_colab",
        preferred_prefixes=("naufal-stage4", "yourname-stage4"),
        required_paths=("control_eval.json",),
        key_output_paths=("control_eval.json",),
    ),
    StageSpec(
        key="stage5",
        label="Stage 5",
        artifact_dir="stage5_colab",
        preferred_prefixes=("naufal-stage5",),
        required_paths=("learner_eval.json", "learner"),
        key_output_paths=("learner_eval.json", "learner"),
    ),
    StageSpec(
        key="stage6",
        label="Stage 6",
        artifact_dir="stage6_colab",
        preferred_prefixes=("naufal-stage6",),
        required_paths=("balance_eval.json", "balancer"),
        key_output_paths=("balance_eval.json", "balancer"),
    ),
    StageSpec(
        key="stage7",
        label="Stage 7",
        artifact_dir="stage7_colab",
        preferred_prefixes=("naufal-stage7",),
        required_paths=("risk_action_eval.json", "risk-action-learner"),
        key_output_paths=("risk_action_eval.json", "risk-action-learner"),
    ),
    StageSpec(
        key="stage8",
        label="Stage 8 Hybrid",
        artifact_dir="stage8_colab",
        preferred_prefixes=("naufal-stage8",),
        required_paths=("hybrid_eval.json",),
        key_output_paths=("hybrid_eval.json", "hybrid-controller"),
        partial_markers=("hybrid-controller/stage8_hybrid_controller.pt", "hybrid-controller"),
    ),
    StageSpec(
        key="stage8_2",
        label="Stage 8.2",
        artifact_dir="stage8_2_colab",
        preferred_prefixes=("naufal-stage8-2",),
        required_paths=("hybrid_eval.json",),
        key_output_paths=("hybrid_eval.json", "stage8-2-action-learner"),
        partial_markers=("stage8-2-action-learner",),
    ),
    StageSpec(
        key="final_comparison",
        label="Final Comparison",
        artifact_dir="final_comparison_colab",
        preferred_prefixes=("naufal-final-comparison",),
        required_paths=("comparison_metrics.csv", "comparison_summary.json"),
        key_output_paths=("comparison_metrics.csv", "comparison_summary.json"),
    ),
)


LABELS = {
    "stage1_baseline": "Stage 1 Baseline",
    "stage1_abstain": "Stage 1 Abstain",
    "stage4": "Stage 4 Fixed Control",
    "stage5": "Stage 5 Learner",
    "stage6": "Stage 6 Adaptive Balance",
    "stage7": "Stage 7 Action Learner",
    "stage8": "Stage 8 Hybrid",
    "stage8_2": "Stage 8.2 Action + Calibrated Support",
}


NOTEBOOK_TEMPLATE_PATHS: dict[str, str] = {
    "stage1": "stages/01-grounded-abstention-baseline/notebooks/stage-01-grounded-abstention-baseline-colab.ipynb",
    "stage2": "stages/02-evidence-support-verification/notebooks/stage-02-evidence-support-verification-colab.ipynb",
    "stage2_5": "stages/02-evidence-support-verification/notebooks/stage-02-5-hard-negative-support-verification-colab.ipynb",
    "stage3": "stages/03-confidence-calibration/notebooks/stage-03-confidence-calibration-colab.ipynb",
    "stage4": "stages/04-unsupported-confidence-control/notebooks/stage-04-unsupported-confidence-control-colab.ipynb",
    "stage5": "stages/05-retrieval-grounded-qa/notebooks/stage-05-support-constrained-learning-colab.ipynb",
    "stage6": "stages/06-adaptive-constraint-balancing/notebooks/stage-06-adaptive-constraint-balancing-colab.ipynb",
    "stage7": "stages/07-risk-budgeted-action-learning/notebooks/stage-07-risk-budgeted-action-learning-colab.ipynb",
    "stage8": "stages/08-joint-optimization/notebooks/stage-08-joint-optimization-colab.ipynb",
    "stage8_2": "stages/08-joint-optimization/notebooks/stage-08-2-action-learner-calibrated-support-colab.ipynb",
    "final_comparison": "analysis/notebooks/final-comparison-colab.ipynb",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def notebook_template_path(repo_root: Path, stage_key: str) -> Path | None:
    relative_path = NOTEBOOK_TEMPLATE_PATHS.get(stage_key)
    if relative_path is None:
        return None
    return repo_root / relative_path


def executed_notebook_path(run_dir: Path) -> Path | None:
    archive_dir = run_dir / "executed-notebook"
    if not archive_dir.exists():
        return None

    summary_path = run_dir / "run-summary.json"
    if summary_path.exists():
        try:
            summary = load_json(summary_path)
        except Exception:
            summary = {}
        target = summary.get("executed_notebook_target")
        if isinstance(target, str) and target:
            target_name = Path(target).name
            candidate = archive_dir / target_name
            if candidate.exists():
                return candidate

    ipynb_files = sorted(archive_dir.glob("*.ipynb"))
    if not ipynb_files:
        return None

    run_name = run_dir.name
    preferred = sorted(path for path in ipynb_files if run_name in path.name)
    if preferred:
        return preferred[0]
    return ipynb_files[0]


def preferred_rank(name: str, preferred_prefixes: tuple[str, ...]) -> int:
    for index, prefix in enumerate(preferred_prefixes):
        if name.startswith(prefix):
            return index
    return len(preferred_prefixes)


def storage_priority(storage: str) -> int:
    return 0 if storage == "Drive" else 1


def stage_path_exists(run_dir: Path, relative_path: str) -> bool:
    return (run_dir / relative_path).exists()


def run_contents(run_dir: Path) -> list[str]:
    items: list[str] = []
    for child in sorted(run_dir.iterdir(), key=lambda path: path.name):
        items.append(child.name)
    return items


def classify_run(run_dir: Path, spec: StageSpec) -> str:
    if all(stage_path_exists(run_dir, rel) for rel in spec.required_paths):
        return "complete"
    if any(stage_path_exists(run_dir, rel) for rel in spec.partial_markers):
        return "partial"

    contents = run_contents(run_dir)
    non_generic = [
        name
        for name in contents
        if name not in GENERIC_METADATA_FILES and name not in GENERIC_METADATA_DIRS
    ]
    if non_generic:
        return "partial"
    if contents:
        return "metadata-only"
    return "empty"


def scan_stage_runs(root: Path, storage: str, spec: StageSpec) -> list[dict[str, Any]]:
    stage_root = root / spec.artifact_dir
    if not stage_root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for child in sorted(stage_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        status = classify_run(child, spec)
        runs.append(
            {
                "name": child.name,
                "path": child,
                "storage": storage,
                "status": status,
                "preferred_rank": preferred_rank(child.name, spec.preferred_prefixes),
                "mtime": child.stat().st_mtime,
            }
        )
    return runs


def choose_canonical_run(spec: StageSpec, runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None

    status_rank = {"complete": 0, "partial": 1, "metadata-only": 2, "empty": 3}
    return sorted(
        runs,
        key=lambda item: (
            status_rank[item["status"]],
            item["preferred_rank"],
            storage_priority(item["storage"]),
            -item["mtime"],
            item["name"],
        ),
    )[0]


def maybe_metric_dict(data: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return None


def maybe_answer_rate_from_predictions(data: dict[str, Any], *keys: str) -> float | None:
    if not keys:
        keys = ("dev_predictions",)

    predictions = None
    for key in keys:
        candidate = data.get(key)
        if isinstance(candidate, dict) and candidate:
            predictions = candidate
            break
    if predictions is None:
        return None
    answered = 0
    for value in predictions.values():
        if isinstance(value, str):
            if value.strip():
                answered += 1
            continue
        if isinstance(value, dict):
            decision = value.get("decision")
            if decision == "answer":
                answered += 1
                continue
            answer = value.get("answer")
            if isinstance(answer, str) and answer.strip():
                answered += 1
    return 100.0 * answered / len(predictions)


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metric_row(label: str, family: str, source_path: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    split_name = "dev"
    metrics = maybe_metric_dict(data, "final_metrics")
    mix = maybe_metric_dict(data, "final_mix")
    overabstain = maybe_metric_dict(data, "final_overabstain")
    prediction_keys = ("final_predictions",)

    if metrics is not None:
        split_name = str(data.get("final_eval_split") or "final")
    else:
        metrics = maybe_metric_dict(data, "test_metrics")
        mix = maybe_metric_dict(data, "test_mix")
        overabstain = maybe_metric_dict(data, "test_overabstain")
        prediction_keys = ("test_predictions",)
        if metrics is not None:
            split_name = "test"
        else:
            metrics = maybe_metric_dict(data, "dev_metrics", "control_dev_metrics", "gated_dev_metrics")
            mix = maybe_metric_dict(data, "dev_mix", "control_dev_mix")
            overabstain = maybe_metric_dict(data, "dev_overabstain")
            prediction_keys = ("dev_predictions",)
            if metrics is None:
                return None

    answer_rate = float_or_none(mix.get("answer_rate")) if mix else maybe_answer_rate_from_predictions(data, *prediction_keys)
    supported_answer_rate = float_or_none(mix.get("supported_answer_rate")) if mix else None
    unsupported_among_answers_rate = float_or_none(mix.get("unsupported_among_answers_rate")) if mix else None

    return {
        "label": label,
        "family": family,
        "source_path": str(source_path),
        "reported_split": split_name,
        "overall_f1": float_or_none(metrics.get("overall_f1")),
        "answerable_f1": float_or_none(metrics.get("answerable_f1")),
        "unsupported_answer_rate": float_or_none(metrics.get("unsupported_answer_rate")),
        "abstain_f1": float_or_none(metrics.get("abstain_f1")),
        "answer_rate": answer_rate,
        "supported_answer_rate": supported_answer_rate,
        "unsupported_among_answers_rate": unsupported_among_answers_rate,
        "overabstain_rate": float_or_none(overabstain.get("overabstain_rate")) if overabstain else None,
    }


def metric_row_from_stage1(run_dir: Path, mode: str) -> dict[str, Any] | None:
    eval_path = run_dir / f"{mode}_eval.json"
    if not eval_path.exists():
        return None
    data = load_json(eval_path)
    return extract_metric_row(
        LABELS[f"stage1_{mode}"],
        "stage1",
        eval_path,
        data,
    )


def metric_row_from_eval_file(label_key: str, family: str, eval_path: Path) -> dict[str, Any] | None:
    if not eval_path.exists():
        return None
    data = load_json(eval_path)
    return extract_metric_row(LABELS[label_key], family, eval_path, data)


def collect_stage_rows(canonical_runs: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    stage1 = canonical_runs.get("stage1")
    if stage1 is not None:
        for mode in ("baseline", "abstain"):
            row = metric_row_from_stage1(stage1["path"], mode)
            if row is not None:
                rows.append(row)

    file_rows = [
        ("stage4", "stage4", "control_eval.json"),
        ("stage5", "stage5", "learner_eval.json"),
        ("stage6", "stage6", "balance_eval.json"),
        ("stage7", "stage7", "risk_action_eval.json"),
        ("stage8", "stage8", "hybrid_eval.json"),
        ("stage8_2", "stage8_2", "hybrid_eval.json"),
    ]
    for key, family, filename in file_rows:
        run = canonical_runs.get(key)
        if run is None:
            continue
        row = metric_row_from_eval_file(key, family, run["path"] / filename)
        if row is not None:
            rows.append(row)

    return rows


def completed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("overall_f1") is not None and row.get("unsupported_answer_rate") is not None
    ]


def best_row(rows: list[dict[str, Any]], key: str, reverse: bool = True) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row[key], reverse=reverse)[0]


def under_budget_best(rows: list[dict[str, Any]], budget: float = 20.0) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if row.get("overall_f1") is not None
        and row.get("unsupported_answer_rate") is not None
        and row["unsupported_answer_rate"] <= budget
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row["overall_f1"], reverse=True)[0]


def fmt(value: Any, digits: int = 2) -> str:
    number = float_or_none(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def bullet_paths(title: str, paths: list[Path]) -> list[str]:
    lines = [f"- {title}:"]
    for path in paths:
        lines.append(f"  - `{path}`")
    return lines


def find_ready_path(canonical_runs: dict[str, dict[str, Any] | None], stage_key: str, relative_path: str) -> Path | None:
    run = canonical_runs.get(stage_key)
    if run is None:
        return None
    candidate = run["path"] / relative_path
    return candidate if candidate.exists() else None


def complete_runs_missing_executed_notebook(
    canonical_runs: dict[str, dict[str, Any] | None],
) -> list[tuple[str, dict[str, Any]]]:
    missing: list[tuple[str, dict[str, Any]]] = []
    for spec in STAGE_SPECS:
        if spec.key == "stage2_5":
            continue
        run = canonical_runs.get(spec.key)
        if run is None or run["status"] != "complete":
            continue
        if spec.key not in NOTEBOOK_TEMPLATE_PATHS:
            continue
        if executed_notebook_path(run["path"]) is None:
            missing.append((spec.label, run))
    return missing


def load_comparison_csv(run: dict[str, Any] | None) -> list[dict[str, str]]:
    if run is None:
        return []
    path = run["path"] / "comparison_metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_worktree_state(repo_root: Path) -> str:
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return "clean" if not result.stdout.strip() else "dirty"


def stage_summary_line(stage_key: str, canonical_runs: dict[str, dict[str, Any] | None], rows: list[dict[str, Any]]) -> str:
    run = canonical_runs.get(stage_key)
    if run is None:
        return f"- {STAGE_LABEL_FROM_KEY[stage_key]}: missing"

    row = next((item for item in rows if item["family"] == stage_key), None)
    if row is not None:
        return (
            f"- {STAGE_LABEL_FROM_KEY[stage_key]}: {run['status']} in {run['storage']} "
            f"as `{run['name']}` with `overall_f1 = {fmt(row['overall_f1'])}` and "
            f"`unsupported_answer_rate = {fmt(row['unsupported_answer_rate'])}`"
        )
    return f"- {STAGE_LABEL_FROM_KEY[stage_key]}: {run['status']} in {run['storage']} as `{run['name']}`"


STAGE_LABEL_FROM_KEY = {
    spec.key: spec.label for spec in STAGE_SPECS
}


def generate_status_board(
    generated_on: str,
    canonical_runs: dict[str, dict[str, Any] | None],
    rows: list[dict[str, Any]],
    incomplete_runs: list[dict[str, Any]],
    audit_reference: str,
) -> str:
    complete_rows = completed_rows(rows)
    best_overall = best_row([row for row in complete_rows if row["family"] != "stage1"], "overall_f1")
    best_budget = under_budget_best([row for row in complete_rows if row["family"] != "stage1"])
    stage8_run = canonical_runs.get("stage8")
    stage8_row = next((row for row in rows if row["family"] == "stage8"), None)
    stage8_complete = bool(stage8_run and stage8_run["status"] == "complete" and stage8_row is not None)
    stage8_completion_marker_missing = bool(
        stage8_run and stage8_run["status"] == "complete" and not (stage8_run["path"] / "RUN_COMPLETED.txt").exists()
    )
    stage8_2_run = canonical_runs.get("stage8_2")
    stage8_2_row = next((row for row in rows if row["family"] == "stage8_2"), None)
    stage8_2_complete = bool(stage8_2_run and stage8_2_run["status"] == "complete" and stage8_2_row is not None)

    lines: list[str] = []
    lines.append("# KeelNet Current Experiment Status")
    lines.append("")
    lines.append(f"Generated: {generated_on}")
    lines.append("")
    lines.append("## At A Glance")
    lines.append("")
    lines.append("- The canonical completed experiment chain currently lives in Drive-backed artifacts, not just local runtime storage.")
    if best_overall is not None:
        lines.append(
            f"- Best completed late-stage result on disk: `{best_overall['label']}` "
            f"with `overall_f1 = {fmt(best_overall['overall_f1'])}` and "
            f"`unsupported_answer_rate = {fmt(best_overall['unsupported_answer_rate'])}`."
        )
    if best_budget is None:
        lines.append("- No saved comparable late-stage run currently meets the `20%` unsupported-answer budget.")
    else:
        lines.append(
            f"- Best saved late-stage run under the `20%` unsupported-answer budget: "
            f"`{best_budget['label']}` with `overall_f1 = {fmt(best_budget['overall_f1'])}`."
        )
    if stage8_complete and stage8_row is not None:
        lines.append(
            f"- `Stage 8 Hybrid` now has a saved eval, but it did not beat `Stage 4`: "
            f"`overall_f1 = {fmt(stage8_row['overall_f1'])}` and "
            f"`unsupported_answer_rate = {fmt(stage8_row['unsupported_answer_rate'])}`."
        )
        if stage8_completion_marker_missing:
            lines.append("- `Stage 8 Hybrid` still does not have `RUN_COMPLETED.txt` or `collab-share-note.md` saved.")
    elif not stage8_2_complete:
        lines.append("- `Stage 8 Hybrid` is only partially saved right now, and `Stage 8.2` has no saved run yet.")
    if stage8_2_complete and stage8_2_row is not None:
        lines.append(
            f"- `Stage 8.2` now has a clean-split saved run on `{stage8_2_row['reported_split']}`: "
            f"`overall_f1 = {fmt(stage8_2_row['overall_f1'])}` and "
            f"`unsupported_answer_rate = {fmt(stage8_2_row['unsupported_answer_rate'])}`."
        )
    lines.append("")
    lines.append("## Current Frontier")
    lines.append("")
    lines.append("| Label | Split | Overall F1 | Answerable F1 | Unsupported answer rate | Abstain F1 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for family in ("stage1", "stage4", "stage5", "stage6", "stage7", "stage8", "stage8_2"):
        for row in rows:
            if row["family"] == family:
                lines.append(
                    f"| {row['label']} | {row['reported_split']} | {fmt(row['overall_f1'])} | {fmt(row['answerable_f1'])} | "
                    f"{fmt(row['unsupported_answer_rate'])} | {fmt(row['abstain_f1'])} |"
                )
    lines.append("")
    lines.append("## Stage Status")
    lines.append("")
    lines.append("| Stage | Canonical run | Storage | Status | Key outputs |")
    lines.append("| --- | --- | --- | --- | --- |")
    for spec in STAGE_SPECS:
        run = canonical_runs.get(spec.key)
        if run is None:
            lines.append(f"| {spec.label} | — | — | missing | — |")
            continue
        outputs = ", ".join(f"`{path}`" for path in spec.key_output_paths)
        lines.append(
            f"| {spec.label} | `{run['name']}` | {run['storage']} | {run['status']} | {outputs} |"
        )
    lines.append("")
    lines.append("## Missing Right Now")
    lines.append("")
    if not incomplete_runs:
        lines.append("- No incomplete or metadata-only runs were detected.")
    else:
        for item in incomplete_runs:
            lines.append(
                f"- `{item['path']}` is `{item['status']}`."
            )
    lines.append("")
    lines.append("## Ready Inputs For A Clean Stage 8.2 Run")
    lines.append("")
    ready_paths = [
        ("Stage 4 control eval", find_ready_path(canonical_runs, "stage4", "control_eval.json")),
        ("Stage 5 learner dir", find_ready_path(canonical_runs, "stage5", "learner")),
        ("Stage 5 eval", find_ready_path(canonical_runs, "stage5", "learner_eval.json")),
        ("Optional Stage 6 balancer", find_ready_path(canonical_runs, "stage6", "balancer")),
        ("Stage 7 comparison eval", find_ready_path(canonical_runs, "stage7", "risk_action_eval.json")),
    ]
    for title, path in ready_paths:
        lines.append(f"- {title}: `{path}`" if path is not None else f"- {title}: missing")
    lines.append("")
    lines.append("## Refresh")
    lines.append("")
    lines.append("```bash")
    lines.append("python analysis/generate_experiment_docs.py")
    lines.append("```")
    lines.append("")
    lines.append(f"Full audit: {audit_reference}")
    lines.append("")
    return "\n".join(lines) + "\n"


def generate_full_audit(
    generated_on: str,
    repo_root: Path,
    roots: list[tuple[str, Path]],
    canonical_runs: dict[str, dict[str, Any] | None],
    rows: list[dict[str, Any]],
    incomplete_runs: list[dict[str, Any]],
) -> str:
    complete_rows = completed_rows(rows)
    best_overall = best_row([row for row in complete_rows if row["family"] != "stage1"], "overall_f1")
    best_budget = under_budget_best([row for row in complete_rows if row["family"] != "stage1"])
    lowest_unsupported = best_row([row for row in complete_rows if row["family"] != "stage1"], "unsupported_answer_rate", reverse=False)
    comparison_run = canonical_runs.get("final_comparison")
    comparison_csv_rows = load_comparison_csv(comparison_run)
    repo_status = git_worktree_state(repo_root)
    missing_executed_notebooks = complete_runs_missing_executed_notebook(canonical_runs)
    stage8_run = canonical_runs.get("stage8")
    stage8_row = next((row for row in rows if row["family"] == "stage8"), None)
    stage8_complete = bool(stage8_run and stage8_run["status"] == "complete" and stage8_row is not None)
    stage8_completion_marker_missing = bool(
        stage8_run and stage8_run["status"] == "complete" and not (stage8_run["path"] / "RUN_COMPLETED.txt").exists()
    )
    stage8_2_run = canonical_runs.get("stage8_2")
    stage8_2_row = next((row for row in rows if row["family"] == "stage8_2"), None)
    stage8_2_complete = bool(stage8_2_run and stage8_2_run["status"] == "complete" and stage8_2_row is not None)

    lines: list[str] = []
    lines.append("# KeelNet Current Experiment Audit")
    lines.append("")
    lines.append(f"Generated: {generated_on}")
    lines.append("")
    lines.append("This file is generated by `analysis/generate_experiment_docs.py`.")
    lines.append("It is the canonical merged experiment doc and replaces the older standalone status board and dated snapshot.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    for label, path in roots:
        lines.append(f"- {label}: `{path}`")
    lines.append(f"- Repo root: `{repo_root}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("- The canonical completed experiment chain currently lives in the Drive-backed artifact store, not in local runtime storage.")
    if best_overall is not None:
        lines.append(
            f"- The strongest completed late-stage result on disk is still `{best_overall['label']}`: "
            f"`overall_f1 = {fmt(best_overall['overall_f1'])}`, "
            f"`answerable_f1 = {fmt(best_overall['answerable_f1'])}`, "
            f"`unsupported_answer_rate = {fmt(best_overall['unsupported_answer_rate'])}`."
        )
    if lowest_unsupported is not None:
        lines.append(
            f"- The lowest unsupported-answer rate among the saved comparable late-stage runs is "
            f"`{fmt(lowest_unsupported['unsupported_answer_rate'])}` from `{lowest_unsupported['label']}`."
        )
    if best_budget is None:
        lines.append("- The current saved late-stage runs do not meet the `20%` unsupported-answer budget on the reported comparison split.")
    else:
        lines.append(
            f"- The best late-stage run under the `20%` unsupported-answer budget is `{best_budget['label']}` "
            f"with `overall_f1 = {fmt(best_budget['overall_f1'])}`."
        )
    if stage8_complete and stage8_row is not None:
        lines.append(
            f"- `Stage 8 Hybrid` now has a saved eval, but it still trails `Stage 4`: "
            f"`overall_f1 = {fmt(stage8_row['overall_f1'])}` and "
            f"`unsupported_answer_rate = {fmt(stage8_row['unsupported_answer_rate'])}`."
        )
        if stage8_completion_marker_missing:
            lines.append("- `Stage 8 Hybrid` still lacks notebook completion bookkeeping: `RUN_COMPLETED.txt` and `collab-share-note.md` are missing.")
    elif not stage8_2_complete:
        lines.append("- `Stage 8 Hybrid` is incomplete on disk, and `Stage 8.2` has not been saved yet.")
    if stage8_2_complete and stage8_2_row is not None:
        lines.append(
            f"- `Stage 8.2` now has a clean-split `{stage8_2_row['reported_split']}` result on disk: "
            f"`overall_f1 = {fmt(stage8_2_row['overall_f1'])}`, "
            f"`answerable_f1 = {fmt(stage8_2_row['answerable_f1'])}`, "
            f"`unsupported_answer_rate = {fmt(stage8_2_row['unsupported_answer_rate'])}`."
        )
    lines.append("")
    lines.append("## Canonical Artifact Inventory")
    lines.append("")
    lines.append("| Stage | Canonical run | Storage | Status | Raw notebook | Executed notebook | Key outputs |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for spec in STAGE_SPECS:
        run = canonical_runs.get(spec.key)
        raw_notebook = notebook_template_path(repo_root, spec.key)
        raw_notebook_text = f"`{raw_notebook}`" if raw_notebook is not None else "—"
        if run is None:
            lines.append(f"| {spec.label} | — | — | missing | {raw_notebook_text} | — | — |")
            continue
        outputs = ", ".join(f"`{path}`" for path in spec.key_output_paths)
        executed_notebook = executed_notebook_path(run["path"])
        executed_notebook_text = f"`{executed_notebook}`" if executed_notebook is not None else "missing"
        lines.append(
            f"| {spec.label} | `{run['name']}` | {run['storage']} | {run['status']} | "
            f"{raw_notebook_text} | {executed_notebook_text} | {outputs} |"
        )
    lines.append("")
    lines.append("## Headline Comparison Metrics")
    lines.append("")
    lines.append("| Label | Split | Overall F1 | Answerable F1 | Unsupported answer rate | Abstain F1 | Answer rate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['reported_split']} | {fmt(row['overall_f1'])} | {fmt(row['answerable_f1'])} | "
            f"{fmt(row['unsupported_answer_rate'])} | {fmt(row['abstain_f1'])} | {fmt(row['answer_rate'])} |"
        )
    lines.append("")
    if comparison_csv_rows:
        lines.append("Source: `final_comparison_colab` exists and provides a saved comparison snapshot.")
        lines.append("")
    lines.append("## Stage Notes")
    lines.append("")
    for stage_key in ("stage1", "stage2", "stage2_5", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8", "stage8_2"):
        lines.append(f"### {STAGE_LABEL_FROM_KEY[stage_key]}")
        lines.append("")
        run = canonical_runs.get(stage_key)
        if run is None:
            lines.append("- No saved run was found during this audit.")
            lines.append("")
            continue
        lines.append(f"- Canonical run: `{run['path']}`")
        lines.append(f"- Storage: {run['storage']}")
        lines.append(f"- Status: {run['status']}")
        raw_notebook = notebook_template_path(repo_root, stage_key)
        executed_notebook = executed_notebook_path(run["path"])
        if raw_notebook is not None:
            lines.append(f"- Raw repo notebook: `{raw_notebook}`")
        lines.append(
            f"- Executed notebook archive: `{executed_notebook}`."
            if executed_notebook is not None
            else "- Executed notebook archive: missing."
        )

        if stage_key == "stage1":
            baseline = next((row for row in rows if row["label"] == LABELS["stage1_baseline"]), None)
            abstain = next((row for row in rows if row["label"] == LABELS["stage1_abstain"]), None)
            if baseline is not None:
                lines.append(
                    f"- Baseline dev: `overall_f1 = {fmt(baseline['overall_f1'])}`, "
                    f"`answerable_f1 = {fmt(baseline['answerable_f1'])}`, "
                    f"`unsupported_answer_rate = {fmt(baseline['unsupported_answer_rate'])}`."
                )
            if abstain is not None:
                lines.append(
                    f"- Abstain dev: `overall_f1 = {fmt(abstain['overall_f1'])}`, "
                    f"`answerable_f1 = {fmt(abstain['answerable_f1'])}`, "
                    f"`unsupported_answer_rate = {fmt(abstain['unsupported_answer_rate'])}`."
                )
        elif stage_key == "stage2":
            path = run["path"] / "verifier_eval.json"
            if path.exists():
                data = load_json(path)
                gated = maybe_metric_dict(data, "gated_dev_metrics")
                support = maybe_metric_dict(data, "dev_support_metrics")
                if gated is not None:
                    lines.append(
                        f"- Gated dev: `overall_f1 = {fmt(gated.get('overall_f1'))}`, "
                        f"`answerable_f1 = {fmt(gated.get('answerable_f1'))}`, "
                        f"`unsupported_answer_rate = {fmt(gated.get('unsupported_answer_rate'))}`."
                    )
                if support is not None:
                    lines.append(
                        f"- Dev support verifier: `support_f1 = {fmt(support.get('support_f1'))}`, "
                        f"`support_accuracy = {fmt(support.get('support_accuracy'))}`."
                    )
        elif stage_key == "stage2_5":
            path = run["path"] / "verifier_eval.json"
            if path.exists():
                data = load_json(path)
                gated = maybe_metric_dict(data, "gated_dev_metrics")
                support = maybe_metric_dict(data, "dev_support_metrics")
                if gated is not None:
                    lines.append(
                        f"- Gated dev: `overall_f1 = {fmt(gated.get('overall_f1'))}`, "
                        f"`answerable_f1 = {fmt(gated.get('answerable_f1'))}`, "
                        f"`unsupported_answer_rate = {fmt(gated.get('unsupported_answer_rate'))}`."
                    )
                if support is not None:
                    lines.append(
                        f"- Dev support verifier: `support_f1 = {fmt(support.get('support_f1'))}`, "
                        f"`support_accuracy = {fmt(support.get('support_accuracy'))}`."
                    )
        elif stage_key == "stage3":
            path = run["path"] / "calibration_eval.json"
            if path.exists():
                data = load_json(path)
                qa_raw = maybe_metric_dict(data, "dev_qa_raw_metrics")
                qa_cal = maybe_metric_dict(data, "dev_qa_calibrated_metrics")
                support_raw = maybe_metric_dict(data, "dev_support_raw_metrics")
                support_cal = maybe_metric_dict(data, "dev_support_calibrated_metrics")
                if qa_raw and qa_cal:
                    lines.append(
                        f"- Dev QA ECE improved from `{fmt(qa_raw.get('ece'), 4)}` raw "
                        f"to `{fmt(qa_cal.get('ece'), 4)}` calibrated."
                    )
                if support_raw and support_cal:
                    lines.append(
                        f"- Dev support ECE improved from `{fmt(support_raw.get('ece'), 4)}` raw "
                        f"to `{fmt(support_cal.get('ece'), 4)}` calibrated."
                    )
        else:
            row = next((item for item in rows if item["family"] == stage_key), None)
            if row is not None:
                lines.append(
                    f"- {str(row.get('reported_split', 'dev')).title()}: `overall_f1 = {fmt(row['overall_f1'])}`, "
                    f"`answerable_f1 = {fmt(row['answerable_f1'])}`, "
                    f"`unsupported_answer_rate = {fmt(row['unsupported_answer_rate'])}`."
                )
                if row.get("overabstain_rate") is not None:
                    lines.append(f"- {str(row.get('reported_split', 'dev')).title()} over-abstain rate: `{fmt(row['overabstain_rate'])}`.")

        if stage_key == "stage4":
            path = run["path"] / "control_eval.json"
            if path.exists():
                selected = maybe_metric_dict(load_json(path), "selected_config")
                if selected is not None:
                    pieces = ", ".join(f"`{key} = {value}`" for key, value in selected.items())
                    lines.append(f"- Selected config: {pieces}.")

        if stage_key == "stage8":
            if run["status"] == "complete":
                if not (run["path"] / "RUN_COMPLETED.txt").exists():
                    lines.append("- `hybrid_eval.json` exists, but the notebook completion files were not written yet.")
            else:
                lines.append("- This run should not be treated as a comparable result until `hybrid_eval.json` and `RUN_COMPLETED.txt` both exist.")
        if stage_key == "stage8_2":
            path = run["path"] / "hybrid_eval.json"
            if path.exists():
                data = load_json(path)
                risk_threshold = data.get("selected_risk_threshold")
                final_split = data.get("final_eval_split")
                if risk_threshold is not None:
                    lines.append(f"- Selected risk threshold: `{fmt(risk_threshold)}`.")
                if final_split is not None:
                    lines.append(f"- Final reported split: `{final_split}`.")
        lines.append("")

    lines.append("## Local Runtime Vs Drive Storage")
    lines.append("")
    local_root = next((path for label, path in roots if label.startswith("Local")), None)
    drive_root = next((path for label, path in roots if label.startswith("Drive")), None)
    if local_root is not None:
        lines.append(f"- Local runtime artifacts root: `{local_root}`")
    if drive_root is not None:
        lines.append(f"- Drive-backed artifacts root: `{drive_root}`")
    lines.append("- Local storage currently contains the strongest early-stage trail.")
    lines.append("- Drive storage currently contains the strongest late-stage trail and the saved final-comparison outputs.")
    lines.append("")
    lines.append("## Missing Or Incomplete Items")
    lines.append("")
    if incomplete_runs:
        for item in incomplete_runs:
            lines.append(
                f"- `{item['path']}` is `{item['status']}` under `{item['storage']}`."
            )
    else:
        lines.append("- No incomplete runs were detected.")
    if missing_executed_notebooks:
        lines.append("- Complete runs still missing an archived executed notebook snapshot:")
        for label, run in missing_executed_notebooks:
            lines.append(f"  - {label}: `{run['path'] / 'executed-notebook'}`")
    lines.append("")
    lines.append("## Ready-To-Use Paths For The Next Clean Run")
    lines.append("")
    ready_items = [
        ("Stage 3 calibration", find_ready_path(canonical_runs, "stage3", "calibration_eval.json")),
        ("Stage 4 control eval", find_ready_path(canonical_runs, "stage4", "control_eval.json")),
        ("Stage 5 learner dir", find_ready_path(canonical_runs, "stage5", "learner")),
        ("Stage 5 eval", find_ready_path(canonical_runs, "stage5", "learner_eval.json")),
        ("Optional Stage 6 balancer", find_ready_path(canonical_runs, "stage6", "balancer")),
        ("Stage 7 eval", find_ready_path(canonical_runs, "stage7", "risk_action_eval.json")),
    ]
    for title, path in ready_items:
        lines.append(f"- {title}: `{path}`" if path is not None else f"- {title}: missing")
    lines.append("")
    lines.append("## Repo State Notes")
    lines.append("")
    lines.append(f"- Repo root: `{repo_root}`")
    lines.append(f"- Working tree state at generation time: `{repo_status}`")
    lines.append("- The repo now has distinct notebook identities for `Stage 8 Hybrid` and `Stage 8.2 Action Learner + Calibrated Support`.")
    lines.append("- The final-comparison notebook already distinguishes those variants and includes the newer plotting improvements.")
    lines.append("")
    lines.append("## Refresh")
    lines.append("")
    lines.append("```bash")
    lines.append("python analysis/generate_experiment_docs.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    docs_dir = repo_root / "docs"

    parser = argparse.ArgumentParser(description="Generate KeelNet experiment markdown docs.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--local-artifacts-root", type=Path, default=Path("/content/KeelNet-local/artifacts"))
    parser.add_argument("--drive-artifacts-root", type=Path, default=Path("/mnt/g/My Drive/KeelNet/artifacts"))
    parser.add_argument("--audit-output", type=Path, default=docs_dir / "current-experiment-audit.md")
    parser.add_argument(
        "--board-output",
        type=Path,
        default=None,
        help="Optional legacy status-board output path. Omitted by default because the audit is now the canonical merged doc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    roots: list[tuple[str, Path]] = []
    if args.local_artifacts_root.exists():
        roots.append(("Local runtime artifacts", args.local_artifacts_root.resolve()))
    if args.drive_artifacts_root.exists():
        roots.append(("Drive-backed artifacts", args.drive_artifacts_root.resolve()))

    scanned: dict[str, list[dict[str, Any]]] = {spec.key: [] for spec in STAGE_SPECS}
    for storage_label, root in roots:
        storage = "Drive" if "Drive" in storage_label else "local"
        storage = "Drive" if storage == "Drive" else "local"
        for spec in STAGE_SPECS:
            scanned[spec.key].extend(scan_stage_runs(root, storage, spec))

    canonical_runs = {spec.key: choose_canonical_run(spec, scanned[spec.key]) for spec in STAGE_SPECS}
    rows = collect_stage_rows(canonical_runs)

    incomplete_runs: list[dict[str, Any]] = []
    for spec in STAGE_SPECS:
        for item in scanned[spec.key]:
            if item["status"] in {"partial", "metadata-only"}:
                incomplete_runs.append(item)
    incomplete_runs.sort(key=lambda item: (storage_priority(item["storage"]), str(item["path"])))

    generated_on = date.today().isoformat()
    audit_text = generate_full_audit(
        generated_on,
        repo_root,
        roots,
        canonical_runs,
        rows,
        incomplete_runs,
    )

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(audit_text, encoding="utf-8")

    if args.board_output is not None:
        audit_reference = (
            f"[{args.audit_output.name}](./{args.audit_output.name})"
            if args.audit_output.parent.resolve() == args.board_output.parent.resolve()
            else f"`{args.audit_output.resolve()}`"
        )
        status_text = generate_status_board(
            generated_on,
            canonical_runs,
            rows,
            incomplete_runs,
            audit_reference,
        )
        args.board_output.parent.mkdir(parents=True, exist_ok=True)
        args.board_output.write_text(status_text, encoding="utf-8")
        print(f"Wrote legacy status board: {args.board_output}")
    print(f"Wrote audit: {args.audit_output}")


if __name__ == "__main__":
    main()
