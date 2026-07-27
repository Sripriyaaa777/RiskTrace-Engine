"""
Simple validation scaffold for inferred dependency edges.

Usage:
    .venv/bin/python validate_dependencies.py \
        --pred data/real_hadoop/dependencies.csv \
        --gold validation/edge_gold_template.csv

The gold file is expected to contain:
    source,target,label,notes

Where label is 1 for a real dependency and 0 for not-a-dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_predicted(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row.get("source", "").strip()
            target = row.get("target", "").strip()
            if source and target:
                pairs.add((source, target))
    return pairs


def load_gold(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row.get("source", "").strip()
            target = row.get("target", "").strip()
            label = int((row.get("label") or "0").strip())
            if source and target:
                rows.append((source, target, label))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Validate inferred dependency edges against a gold set.")
    parser.add_argument("--pred", type=Path, default=Path("data/real_hadoop/dependencies.csv"))
    parser.add_argument("--gold", type=Path, default=Path("validation/edge_gold_template.csv"))
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for precision/recall/F1 results.")
    args = parser.parse_args()

    predicted = load_predicted(args.pred)
    gold = load_gold(args.gold)

    tp = fp = fn = tn = 0
    for source, target, label in gold:
        pred = (source, target) in predicted
        if pred and label == 1:
            tp += 1
        elif pred and label == 0:
            fp += 1
        elif not pred and label == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print("Dependency Validation")
    print("---------------------")
    print(f"Gold pairs: {len(gold)}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "predicted_path": str(args.pred),
            "gold_path": str(args.gold),
            "gold_pairs": len(gold),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "labeling_note": (
                "Labels in validation/edge_gold_template.csv are heuristic, not manually adjudicated ground truth. "
                "Explicit Jira links are labeled positive; inferred edges are labeled positive when they have "
                "confidence >= 0.75 and source/target project agreement, and negative otherwise."
            ),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON results to {args.output}")


if __name__ == "__main__":
    main()
