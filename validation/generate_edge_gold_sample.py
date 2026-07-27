from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def label_edge(row: dict) -> tuple[int, str]:
    confidence = float(row.get("confidence") or 0.0)
    link_type = row.get("link_type", "")
    source_project = row["source"].split("-", 1)[0]
    target_project = row["target"].split("-", 1)[0]

    if not is_truthy(row.get("inferred", "")):
        return 1, "positive: explicit Jira dependency link"
    if confidence >= 0.75 and source_project == target_project:
        return 1, (
            "heuristic positive: inferred edge with confidence >= 0.75 "
            "and same Hadoop sub-project"
        )
    return 0, (
        "heuristic negative: inferred edge lacks high confidence and/or "
        "same-sub-project support"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a stratified 100-edge validation sample.")
    parser.add_argument("--deps", type=Path, default=Path("data/real_hadoop/dependencies.csv"))
    parser.add_argument("--output", type=Path, default=Path("validation/edge_gold_template.csv"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.deps, newline="", encoding="utf-8") as f:
        inferred = [row for row in csv.DictReader(f) if is_truthy(row.get("inferred", ""))]

    strata: dict[str, list[dict]] = defaultdict(list)
    for row in inferred:
        confidence = float(row.get("confidence") or 0.0)
        confidence_bucket = "high" if confidence >= 0.75 else "lower"
        strata[f"{row['source'].split('-', 1)[0]}:{confidence_bucket}"].append(row)

    rng = random.Random(args.seed)
    projects = sorted(strata)
    base_quota = max(args.sample_size // max(len(projects), 1), 1)
    sample = []
    for project in projects:
        rows = strata[project]
        take = min(base_quota, len(rows))
        sample.extend(rng.sample(rows, take))

    remaining = args.sample_size - len(sample)
    if remaining > 0:
        chosen = {(row["source"], row["target"]) for row in sample}
        pool = [row for row in inferred if (row["source"], row["target"]) not in chosen]
        sample.extend(rng.sample(pool, min(remaining, len(pool))))

    sample = sample[:args.sample_size]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "label", "notes"])
        writer.writeheader()
        for row in sample:
            label, note = label_edge(row)
            writer.writerow({
                "source": row["source"],
                "target": row["target"],
                "label": label,
                "notes": note,
            })

    print(
        f"Wrote {len(sample)} heuristic validation labels to {args.output}. "
        "These labels are documented heuristics, not hand-labeled ground truth."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
