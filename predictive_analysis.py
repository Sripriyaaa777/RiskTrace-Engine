from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from predictive_model import (
    actual_delay_days,
    actually_delayed,
    build_snapshot_feature_row,
    load_processed_rows,
    predict_with_trained_model,
    prediction_time,
)
from risk_engine import IssueNode, build_reverse_adjacency, find_downstream_nodes, run_propagation

log = logging.getLogger(__name__)


@dataclass
class PredictiveRecord:
    issue_id: str
    risk_score: float
    predicted: bool
    actually_delayed: bool
    downstream_impact_count: int
    evaluation_time: str
    actual_delay_days: float
    actionable_insight: str


def load_processed_data(
    issues_path: Path,
    deps_path: Path,
    project: Optional[str] = None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    log.info("[stage:load] Loading processed graph data from %s and %s", issues_path, deps_path)
    return load_processed_rows(str(issues_path), str(deps_path), project=project)


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def snapshot_issue(issue: dict, as_of: datetime) -> Optional[IssueNode]:
    created = parse_datetime(issue.get("created"))
    if created and created > as_of:
        return None

    updated = parse_datetime(issue.get("updated"))
    resolved = parse_datetime(issue.get("resolved"))
    due_date = parse_datetime(issue.get("due_date"))

    status = issue.get("status", "Open") or "Open"
    if resolved and resolved <= as_of:
        status = "Done"
        effective_updated = resolved
    else:
        if status == "Done":
            status = "In Progress"
        effective_updated = min(dt for dt in [updated, as_of] if dt is not None) if updated else as_of

    delay_days = None
    if due_date and status != "Done":
        delay_days = round((as_of - due_date).total_seconds() / 86400.0, 2)

    is_delayed = bool(delay_days is not None and delay_days > 0)
    if status == "Blocked":
        is_delayed = True

    return IssueNode(
        issue_id=issue["issue_id"],
        project=issue.get("project", ""),
        summary=issue.get("summary", ""),
        status=status,
        priority=issue.get("priority", "Medium") or "Medium",
        assignee=issue.get("assignee", ""),
        due_date=due_date,
        updated=effective_updated,
        delay_days=delay_days,
        is_delayed=is_delayed,
    )


def select_experiment_issues(
    issues: list[dict],
    edges: list[tuple[str, str]],
    positive_target: int = 20,
    total_target: int = 30,
) -> list[dict]:
    reverse_adj = build_reverse_adjacency(edges)

    scored = []
    for issue in issues:
        t = prediction_time(issue)
        if t is None:
            continue
        impacted = find_downstream_nodes(issue["issue_id"], reverse_adj, 10)
        scored.append(
            (
                actually_delayed(issue),
                len([nid for nid, depth in impacted.items() if nid != issue["issue_id"] and depth > 0]),
                actual_delay_days(issue),
                issue,
            )
        )

    positives = [item[3] for item in sorted(scored, key=lambda x: (x[1], x[2]), reverse=True) if item[0]]
    negatives = [item[3] for item in sorted(scored, key=lambda x: (x[1], x[2]), reverse=True) if not item[0]]

    chosen = positives[:positive_target]
    remaining = max(total_target - len(chosen), 0)
    chosen.extend(negatives[:remaining])
    return chosen


def precision_recall_accuracy(records: list[PredictiveRecord]) -> dict:
    tp = sum(1 for r in records if r.predicted and r.actually_delayed)
    fp = sum(1 for r in records if r.predicted and not r.actually_delayed)
    tn = sum(1 for r in records if not r.predicted and not r.actually_delayed)
    fn = sum(1 for r in records if not r.predicted and r.actually_delayed)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(records) if records else 0.0
    early_flag_accuracy = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "flagged_before_delay_accuracy": round(early_flag_accuracy, 3),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_predictive_experiment(
    issues_path: str = "data/processed/issues.csv",
    deps_path: str = "data/processed/dependencies.csv",
    project: Optional[str] = None,
    threshold: float = 0.35,
    positive_target: int = 20,
    total_target: int = 30,
    output_path: Optional[str] = None,
    model_path: str = "data/models/predictive_model.joblib",
    use_trained_model: bool = True,
) -> dict:
    issues, edges = load_processed_data(Path(issues_path), Path(deps_path), project=project)
    selected = select_experiment_issues(
        issues,
        edges,
        positive_target=positive_target,
        total_target=total_target,
    )
    reverse_adj = build_reverse_adjacency(edges)
    issues_by_id = {row["issue_id"]: row for row in issues}
    forward_adj: dict[str, list[str]] = {}
    for src, tgt in edges:
        forward_adj.setdefault(src, []).append(tgt)

    results: list[PredictiveRecord] = []
    log.info("[stage:risk_propagation] Running predictive experiment on %d evaluation issues", len(selected))
    model_feature_rows = []
    model_issue_ids = []

    for issue in selected:
        as_of = prediction_time(issue)
        if as_of is None:
            continue

        snapshot_nodes = [node for row in issues if (node := snapshot_issue(row, as_of)) is not None]
        valid_ids = {node.issue_id for node in snapshot_nodes}
        snapshot_edges = [(src, tgt) for src, tgt in edges if src in valid_ids and tgt in valid_ids]
        risk_results = run_propagation(snapshot_nodes, snapshot_edges)
        risk = risk_results.get(issue["issue_id"])
        heuristic_score = round(risk.risk_score if risk else 0.0, 3)
        model_row = build_snapshot_feature_row(issue, issues_by_id, forward_adj, reverse_adj, as_of)
        if model_row is not None:
            model_feature_rows.append(model_row)
            model_issue_ids.append(issue["issue_id"])
        predicted = heuristic_score >= threshold
        impacted = find_downstream_nodes(issue["issue_id"], reverse_adj, 10)
        downstream_count = len([nid for nid, depth in impacted.items() if nid != issue["issue_id"] and depth > 0])

        results.append(
            PredictiveRecord(
                issue_id=issue["issue_id"],
                risk_score=heuristic_score,
                predicted=predicted,
                actually_delayed=actually_delayed(issue),
                downstream_impact_count=downstream_count,
                evaluation_time=as_of.isoformat(),
                actual_delay_days=round(actual_delay_days(issue), 3),
                actionable_insight=f"Fix Issue {issue['issue_id']} -> reduces risk in {downstream_count} dependent issues",
            )
        )

    if use_trained_model and model_feature_rows and Path(model_path).exists():
        scores = predict_with_trained_model(model_feature_rows, model_path=model_path)
        by_issue = {iid: score for iid, score in zip(model_issue_ids, scores)}
        for record in results:
            if record.issue_id in by_issue:
                record.risk_score = round(float(by_issue[record.issue_id]), 3)
                record.predicted = record.risk_score >= threshold

    payload = {
        "project": project,
        "threshold": threshold,
        "model_used": use_trained_model and Path(model_path).exists(),
        "evaluated_issues": [asdict(record) for record in results],
        "metrics": precision_recall_accuracy(results),
    }

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("Predictive analysis written to %s", output)

    return payload


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="RiskTrace predictive validation over processed Jira data")
    parser.add_argument("--issues", default="data/processed/issues.csv")
    parser.add_argument("--deps", default="data/processed/dependencies.csv")
    parser.add_argument("--project", default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--positive-target", type=int, default=20)
    parser.add_argument("--total-target", type=int, default=30)
    parser.add_argument("--output", default="data/processed/predictive_analysis.json")
    parser.add_argument("--model-path", default="data/models/predictive_model.joblib")
    parser.add_argument("--no-trained-model", action="store_true")
    args = parser.parse_args()

    payload = run_predictive_experiment(
        issues_path=args.issues,
        deps_path=args.deps,
        project=args.project,
        threshold=args.threshold,
        positive_target=args.positive_target,
        total_target=args.total_target,
        output_path=args.output,
        model_path=args.model_path,
        use_trained_model=not args.no_trained_model,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
