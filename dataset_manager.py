from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from preprocess import load_raw_issues, parse_issue, run_pipeline

DATA_ROOT = Path("data/user_datasets")
RAW_ROOT = DATA_ROOT / "raw"
SLICE_ROOT = DATA_ROOT / "slices"
INDEX_PATH = DATA_ROOT / "index.json"


def _ensure_dirs() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    SLICE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _load_index() -> dict:
    _ensure_dirs()
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return {"datasets": []}


def _save_index(index: dict) -> None:
    _ensure_dirs()
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _find_dataset(index: dict, dataset_id: str) -> dict:
    for dataset in index["datasets"]:
        if dataset["dataset_id"] == dataset_id:
            return dataset
    raise KeyError(f"Dataset {dataset_id} not found")


def discover_projects(input_path: Path) -> list[dict]:
    counts: dict[str, int] = {}
    total_valid = 0
    for raw in load_raw_issues(input_path):
        parsed = parse_issue(raw)
        if not parsed:
            continue
        total_valid += 1
        project = (parsed.get("project") or "UNKNOWN").upper()
        counts[project] = counts.get(project, 0) + 1
    projects = [{"key": key, "issue_count": count} for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return projects


def save_uploaded_dataset(temp_path: Path, original_name: str) -> dict:
    _ensure_dirs()
    dataset_id = f"ds_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    ext = Path(original_name).suffix.lower() or temp_path.suffix.lower() or ".bin"
    raw_path = RAW_ROOT / f"{dataset_id}{ext}"
    shutil.move(str(temp_path), raw_path)

    projects = discover_projects(raw_path)
    record = {
        "dataset_id": dataset_id,
        "original_name": original_name,
        "raw_path": str(raw_path),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "projects": projects,
        "slices": {},
    }
    index = _load_index()
    index["datasets"].append(record)
    _save_index(index)
    return record


def list_datasets() -> list[dict]:
    index = _load_index()
    datasets = []
    for dataset in index["datasets"]:
        datasets.append({
            "dataset_id": dataset["dataset_id"],
            "original_name": dataset["original_name"],
            "uploaded_at": dataset["uploaded_at"],
            "projects": dataset.get("projects", []),
            "slice_count": len(dataset.get("slices", {})),
        })
    return datasets


def get_dataset(dataset_id: str) -> dict:
    index = _load_index()
    return _find_dataset(index, dataset_id)


def prepare_slice(
    dataset_id: str,
    project_key: str,
    max_issues: int = 1500,
    include_subtasks: bool = False,
    augment_soft_deps: bool = True,
) -> dict:
    index = _load_index()
    dataset = _find_dataset(index, dataset_id)
    project_key = project_key.upper()
    output_dir = SLICE_ROOT / dataset_id / project_key.lower()
    run_pipeline(
        input_path=Path(dataset["raw_path"]),
        output_dir=output_dir,
        max_issues=max_issues,
        project_filter=project_key,
        synthetic=False,
        include_subtasks=include_subtasks,
        augment_soft_deps=augment_soft_deps,
    )
    stats_path = output_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    dataset.setdefault("slices", {})[project_key] = {
        "project": project_key,
        "output_dir": str(output_dir),
        "issues_csv": str(output_dir / "issues.csv"),
        "deps_csv": str(output_dir / "dependencies.csv"),
        "stats_json": str(stats_path),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "max_issues": max_issues,
        "include_subtasks": include_subtasks,
        "augment_soft_deps": augment_soft_deps,
        "stats": stats,
    }
    _save_index(index)
    return dataset["slices"][project_key]

