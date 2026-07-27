from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def count_csv_records(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail if stats.json total_issues drifts from parser-visible issues.csv records.")
    parser.add_argument("--dataset", type=Path, default=Path("data/real_hadoop"))
    args = parser.parse_args()

    issues_path = args.dataset / "issues.csv"
    stats_path = args.dataset / "stats.json"
    actual = count_csv_records(issues_path)
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    reported = int(stats.get("total_issues", -1))

    if reported != actual:
        print(
            f"FAIL: {stats_path} total_issues={reported}, "
            f"but {issues_path} has {actual} parser-visible issue records."
        )
        return 1

    print(f"PASS: {stats_path} total_issues matches {issues_path} issue records ({actual}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
