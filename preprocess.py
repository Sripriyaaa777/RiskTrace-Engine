"""
IssueGraphAgent++ — Phase 1: Data Preprocessing Pipeline
=========================================================
Ingests the Apache Jira dataset (Zenodo) and produces a clean,
normalized subset suitable for temporal dependency graph construction.

Input  : Raw JSON/CSV export from the Apache Jira dataset
Output : data/processed/issues.csv
         data/processed/dependencies.csv
         data/processed/stats.json          ← for the paper

Usage:
    python scripts/preprocess.py --input data/raw/issues.json \
                                  --max-issues 300 \
                                  --project HADOOP

Apache Jira dataset: https://zenodo.org/records/7740379
"""

import json
import csv
import argparse
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Canonical status buckets used throughout the system.
# The raw dataset may use different labels — we normalise here once.
STATUS_MAP = {
    "open":        "Open",
    "in progress": "In Progress",
    "in-progress": "In Progress",
    "inprogress":  "In Progress",
    "resolved":    "Done",
    "closed":      "Done",
    "done":        "Done",
    "won't fix":   "Done",
    "wontfix":     "Done",
    "duplicate":   "Done",
    "invalid":     "Done",
    "reopened":    "Open",
    "blocked":     "Blocked",
    # Anything not listed → treated as "Unknown" (filtered out later)
}

# Link types in Jira that represent a true task dependency.
# "relates to" and "duplicates" are intentionally excluded —
# they do not represent a blocking/ordering relationship.
DEPENDENCY_LINK_TYPES = {
    "blocks",
    "is blocked by",
    "depends on",
    "is depended on by",
    "cloners",          # cloned issue often has inherited dependency
}

# ISO 8601 date formats we may encounter in the dataset
DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


# ─────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────

def parse_date(raw: Optional[str]) -> Optional[datetime]:
    """Attempt to parse a date string into a timezone-aware datetime."""
    if not raw:
        return None
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    log.debug("Could not parse date: %s", raw)
    return None


def normalise_status(raw: Optional[str]) -> str:
    """Map raw Jira status to our canonical set."""
    if not raw:
        return "Unknown"
    return STATUS_MAP.get(raw.strip().lower(), "Unknown")


def compute_delay_days(
    deadline: Optional[datetime],
    last_updated: Optional[datetime],
    status: str,
) -> Optional[float]:
    """
    Return how many days past the deadline an issue is.
    Positive  → overdue.
    Negative  → still has time.
    None      → no deadline to evaluate against.
    """
    if deadline is None:
        return None
    reference = datetime.now(timezone.utc) if status != "Done" else last_updated
    if reference is None:
        reference = datetime.now(timezone.utc)
    delta = (reference - deadline).total_seconds() / 86400.0
    return round(delta, 2)


# ─────────────────────────────────────────────
# Core parsing
# ─────────────────────────────────────────────

def parse_issue(raw: dict) -> Optional[dict]:
    """
    Extract and normalise the fields we need from a raw Jira JSON issue.

    Returns None if the issue is missing required fields (id, summary).
    The Apache dataset wraps most fields inside a 'fields' key.
    """
    fields = raw.get("fields", raw)   # handle both flat and nested layouts

    issue_id = (
        raw.get("key")
        or raw.get("id")
        or fields.get("key")
        or fields.get("id")
    )
    if not issue_id:
        return None

    summary = fields.get("summary") or fields.get("title") or ""
    if not summary.strip():
        return None

    # Status
    status_raw = None
    s = fields.get("status")
    if isinstance(s, dict):
        status_raw = s.get("name")
    elif isinstance(s, str):
        status_raw = s
    status = normalise_status(status_raw)

    # Skip issues we cannot reason about
    if status == "Unknown":
        return None

    # Priority
    priority_raw = fields.get("priority")
    priority = "Medium"
    if isinstance(priority_raw, dict):
        priority = priority_raw.get("name", "Medium")
    elif isinstance(priority_raw, str):
        priority = priority_raw

    # Timestamps
    created     = parse_date(fields.get("created"))
    updated     = parse_date(fields.get("updated"))
    resolved    = parse_date(fields.get("resolutiondate") or fields.get("resolved"))
    due_date    = parse_date(fields.get("duedate") or fields.get("due_date"))

    # Project key
    project_raw = fields.get("project")
    project = ""
    if isinstance(project_raw, dict):
        project = project_raw.get("key", "")
    elif isinstance(project_raw, str):
        project = project_raw
    if not project and "-" in str(issue_id):
        project = str(issue_id).split("-")[0]

    # Assignee
    assignee_raw = fields.get("assignee")
    assignee = ""
    if isinstance(assignee_raw, dict):
        assignee = assignee_raw.get("displayName") or assignee_raw.get("name", "")
    elif isinstance(assignee_raw, str):
        assignee = assignee_raw

    delay_days = compute_delay_days(due_date, updated, status)

    # Is this issue at risk?
    is_delayed = (
        (delay_days is not None and delay_days > 0 and status != "Done")
        or status == "Blocked"
    )

    return {
        "issue_id":    str(issue_id),
        "project":     project,
        "summary":     summary[:200],          # cap length for storage
        "status":      status,
        "priority":    priority,
        "assignee":    assignee,
        "created":     created.isoformat() if created else "",
        "updated":     updated.isoformat() if updated else "",
        "resolved":    resolved.isoformat() if resolved else "",
        "due_date":    due_date.isoformat() if due_date else "",
        "delay_days":  delay_days if delay_days is not None else "",
        "is_delayed":  is_delayed,
    }


def parse_dependencies(raw: dict) -> list[dict]:
    """
    Extract dependency edges from a raw Jira issue's issuelinks field.

    Returns a list of {source, target, link_type} dicts.
    We keep only links in DEPENDENCY_LINK_TYPES (blocking relationships).
    """
    fields = raw.get("fields", raw)
    issue_id = raw.get("key") or raw.get("id") or fields.get("key") or fields.get("id")
    if not issue_id:
        return []

    links = fields.get("issuelinks") or fields.get("issue_links") or []
    edges = []

    for link in links:
        if not isinstance(link, dict):
            continue

        link_type_raw = link.get("type", {})
        if isinstance(link_type_raw, dict):
            link_name = (
                link_type_raw.get("name", "")
                or link_type_raw.get("outward", "")
            ).lower().strip()
        else:
            link_name = str(link_type_raw).lower().strip()

        if link_name not in DEPENDENCY_LINK_TYPES:
            continue

        # Outward link: current issue → other (e.g. "blocks" another issue)
        outward = link.get("outwardIssue") or link.get("outward_issue")
        if outward:
            target_id = outward.get("key") or outward.get("id")
            if target_id:
                edges.append({
                    "source":    str(issue_id),
                    "target":    str(target_id),
                    "link_type": link_name,
                    "direction": "outward",
                })

        # Inward link: other issue → current (e.g. "is blocked by")
        inward = link.get("inwardIssue") or link.get("inward_issue")
        if inward:
            source_id = inward.get("key") or inward.get("id")
            if source_id:
                edges.append({
                    "source":    str(source_id),
                    "target":    str(issue_id),
                    "link_type": link_name,
                    "direction": "inward",
                })

    return edges


# ─────────────────────────────────────────────
# Dataset loader (handles multiple raw formats)
# ─────────────────────────────────────────────

def load_raw_issues(path: Path) -> list[dict]:
    """
    Load raw issues from a file.
    Handles:
      - JSON array:             [ {issue}, {issue}, ... ]
      - JSON lines (JSONL):     one JSON object per line
      - Wrapped JSON object:    { "issues": [ ... ] }
    """
    log.info("Loading raw data from %s", path)
    suffix = path.suffix.lower()

    if suffix in (".json", ".jsonl"):
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()

        # JSONL — one object per line
        if content.startswith("{") and "\n" in content:
            issues = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        issues.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return issues

        # Standard JSON
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Try common wrapper keys
            for key in ("issues", "data", "items", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Single issue wrapped in a dict
            return [data]

    elif suffix == ".csv":
        issues = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                issues.append(dict(row))
        return issues

    raise ValueError(f"Unsupported file format: {suffix}")


# ─────────────────────────────────────────────
# Synthetic dataset generator
# (used for testing and when no real data is available)
# ─────────────────────────────────────────────

def generate_synthetic_dataset(n_issues: int = 200) -> tuple[list[dict], list[dict]]:
    """
    Generates a realistic synthetic project dataset for testing.
    Creates a dependency graph with:
      - Linear chains  (A → B → C)
      - Fan-out nodes  (A → B, A → C, A → D)
      - Some delayed / blocked tasks

    Returns (issues, dependencies) in the same format as the real pipeline.
    """
    import random
    random.seed(42)

    now = datetime.now(timezone.utc)
    statuses = ["Open", "In Progress", "Done", "Blocked"]
    priorities = ["Critical", "High", "Medium", "Low"]
    projects = ["HADOOP", "SPARK", "KAFKA"]

    issues = []
    for i in range(1, n_issues + 1):
        project = random.choice(projects)
        issue_id = f"{project}-{i}"
        status = random.choices(
            statuses, weights=[30, 25, 35, 10], k=1
        )[0]

        # Randomly assign a deadline: 60% of issues have one
        due_date = None
        delay_days_val = ""
        is_delayed = False
        if random.random() < 0.6:
            offset = random.randint(-30, 30)  # days from now
            due_date = datetime(
                now.year, now.month, now.day, tzinfo=timezone.utc
            )
            from datetime import timedelta
            due_date = due_date + timedelta(days=offset)
            delay_days_val = compute_delay_days(due_date, now, status)
            is_delayed = (
                (delay_days_val is not None and delay_days_val > 0 and status != "Done")
                or status == "Blocked"
            )

        created = now
        from datetime import timedelta
        created = now - timedelta(days=random.randint(10, 180))
        updated = created + timedelta(days=random.randint(1, 30))

        issues.append({
            "issue_id":   issue_id,
            "project":    project,
            "summary":    f"Task {i}: implement {random.choice(['feature','fix','refactor','test','deploy'])} module {i}",
            "status":     status,
            "priority":   random.choice(priorities),
            "assignee":   f"user_{random.randint(1, 15)}",
            "created":    created.isoformat(),
            "updated":    updated.isoformat(),
            "resolved":   updated.isoformat() if status == "Done" else "",
            "due_date":   due_date.isoformat() if due_date else "",
            "delay_days": delay_days_val if delay_days_val != "" else "",
            "is_delayed": is_delayed,
        })

    # Build a dependency graph with chains and fan-outs
    ids = [iss["issue_id"] for iss in issues]
    deps = []
    seen_edges = set()

    # Chain: issue i depends on issue i-1 (for first 40 issues)
    for i in range(1, min(40, n_issues)):
        src, tgt = ids[i], ids[i - 1]
        if (src, tgt) not in seen_edges:
            deps.append({"source": src, "target": tgt, "link_type": "blocks", "direction": "outward"})
            seen_edges.add((src, tgt))

    # Fan-out: some central nodes block many others
    hubs = random.sample(ids[:50], k=min(10, len(ids[:50])))
    for hub in hubs:
        targets = random.sample([x for x in ids if x != hub], k=random.randint(2, 5))
        for t in targets:
            if (hub, t) not in seen_edges:
                deps.append({"source": hub, "target": t, "link_type": "blocks", "direction": "outward"})
                seen_edges.add((hub, t))

    # Random additional edges
    for _ in range(min(50, n_issues // 4)):
        src, tgt = random.sample(ids, 2)
        if (src, tgt) not in seen_edges:
            deps.append({"source": src, "target": tgt, "link_type": "depends on", "direction": "outward"})
            seen_edges.add((src, tgt))

    return issues, deps


# ─────────────────────────────────────────────
# Statistics reporter (for paper)
# ─────────────────────────────────────────────

def compute_stats(issues: list[dict], dependencies: list[dict]) -> dict:
    """Compute dataset statistics to report in the paper's Evaluation section."""
    statuses = [i["status"] for i in issues]
    delayed  = [i for i in issues if i["is_delayed"]]
    with_deadline = [i for i in issues if i["due_date"]]
    delay_vals = [
        float(i["delay_days"]) for i in issues
        if i["delay_days"] != "" and i["delay_days"] is not None
    ]

    # Connectivity stats
    node_ids = {i["issue_id"] for i in issues}
    dep_node_ids = {d["source"] for d in dependencies} | {d["target"] for d in dependencies}
    isolated_nodes = node_ids - dep_node_ids

    # In-degree distribution (how many things depend on this node)
    in_degree: dict[str, int] = {}
    for d in dependencies:
        in_degree[d["target"]] = in_degree.get(d["target"], 0) + 1
    in_degrees = list(in_degree.values())

    return {
        "total_issues":             len(issues),
        "total_dependencies":       len(dependencies),
        "status_distribution":      {s: statuses.count(s) for s in set(statuses)},
        "delayed_issues":           len(delayed),
        "delayed_percentage":       round(len(delayed) / max(len(issues), 1) * 100, 1),
        "issues_with_deadline":     len(with_deadline),
        "avg_delay_days":           round(statistics.mean(delay_vals), 2) if delay_vals else 0,
        "max_delay_days":           round(max(delay_vals), 2) if delay_vals else 0,
        "isolated_nodes":           len(isolated_nodes),
        "nodes_in_dependency_graph": len(dep_node_ids & node_ids),
        "avg_in_degree":            round(statistics.mean(in_degrees), 2) if in_degrees else 0,
        "max_in_degree":            max(in_degrees) if in_degrees else 0,
    }


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    input_path: Optional[Path],
    output_dir: Path,
    max_issues: int,
    project_filter: Optional[str],
    synthetic: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    issues_out  = output_dir / "issues.csv"
    deps_out    = output_dir / "dependencies.csv"
    stats_out   = output_dir / "stats.json"

    # ── Step 1: Load ──────────────────────────────────────────────────
    if synthetic or input_path is None:
        log.info("Generating synthetic dataset with %d issues …", max_issues)
        issues, dependencies = generate_synthetic_dataset(max_issues)
    else:
        raw_issues = load_raw_issues(input_path)
        log.info("Loaded %d raw records", len(raw_issues))

        # ── Step 2: Parse & normalise ─────────────────────────────────
        issues = []
        dependencies = []
        skipped = 0

        for raw in raw_issues:
            parsed = parse_issue(raw)
            if parsed is None:
                skipped += 1
                continue

            # Optional project filter
            if project_filter and parsed["project"].upper() != project_filter.upper():
                continue

            issues.append(parsed)
            dependencies.extend(parse_dependencies(raw))

        log.info(
            "Parsed %d valid issues, %d skipped, %d dependency edges found",
            len(issues), skipped, len(dependencies),
        )

        # ── Step 3: Subset ────────────────────────────────────────────
        # Prioritise delayed/blocked issues so the graph has interesting risk signals.
        # Then fill up with connected issues, then the rest.
        if len(issues) > max_issues:
            delayed_first = [i for i in issues if i["is_delayed"]]
            rest          = [i for i in issues if not i["is_delayed"]]
            issues        = (delayed_first + rest)[:max_issues]
            log.info("Subsetted to %d issues (delayed issues prioritised)", len(issues))

        # Prune dependencies to only include edges where both nodes are in our subset
        valid_ids = {i["issue_id"] for i in issues}
        before = len(dependencies)
        dependencies = [
            d for d in dependencies
            if d["source"] in valid_ids and d["target"] in valid_ids
        ]
        log.info(
            "Pruned dependencies: %d → %d (kept only intra-subset edges)",
            before, len(dependencies),
        )

    # ── Step 4: De-duplicate dependencies ─────────────────────────────
    seen = set()
    unique_deps = []
    for d in dependencies:
        key = (d["source"], d["target"])
        if key not in seen:
            seen.add(key)
            unique_deps.append(d)
    dependencies = unique_deps

    # ── Step 5: Write CSVs ────────────────────────────────────────────
    issue_fields = [
        "issue_id", "project", "summary", "status", "priority",
        "assignee", "created", "updated", "resolved",
        "due_date", "delay_days", "is_delayed",
    ]
    with open(issues_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=issue_fields)
        writer.writeheader()
        writer.writerows(issues)

    dep_fields = ["source", "target", "link_type", "direction"]
    with open(deps_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dep_fields)
        writer.writeheader()
        writer.writerows(dependencies)

    # ── Step 6: Stats ─────────────────────────────────────────────────
    stats = compute_stats(issues, dependencies)
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log.info("Output written to %s", output_dir)
    log.info("─" * 50)
    log.info("DATASET SUMMARY (for paper)")
    log.info("─" * 50)
    for k, v in stats.items():
        log.info("  %-35s %s", k, v)
    log.info("─" * 50)
    log.info("✓ Preprocessing complete.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IssueGraphAgent++ — Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to raw dataset file (.json, .jsonl, or .csv). "
             "Omit to generate a synthetic dataset.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--max-issues", "-n",
        type=int,
        default=300,
        help="Maximum number of issues to include in the subset (default: 300)",
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="Filter to a single Jira project key, e.g. HADOOP",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ignore --input and generate a synthetic dataset instead",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        max_issues=args.max_issues,
        project_filter=args.project,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
