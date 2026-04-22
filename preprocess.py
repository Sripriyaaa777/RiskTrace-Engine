"""
IssueGraphAgent++ — Phase 1: Data Preprocessing Pipeline
=========================================================
Ingests the Apache Jira dataset (Zenodo) and produces a clean,
normalized subset suitable for temporal dependency graph construction.

Input  : Raw JSON/CSV/BSON export from the Apache Jira dataset
Output : data/processed/issues.csv
         data/processed/dependencies.csv
         data/processed/stats.json          ← for the paper

Usage:
    python preprocess.py --input data/raw/issues.bson.gz \
                         --max-issues 300 \
                         --project HADOOP

Apache Jira dataset: https://zenodo.org/records/7740379
"""

import json
import csv
import argparse
import gzip
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import statistics
from collections.abc import Iterable, Iterator

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

DEPENDENCY_CUE_PATTERNS = [
    re.compile(r"\bdepends?\s+on\b", re.IGNORECASE),
    re.compile(r"\bblocked\s+by\b", re.IGNORECASE),
    re.compile(r"\brequires?\b", re.IGNORECASE),
    re.compile(r"\bwaiting\s+for\b", re.IGNORECASE),
    re.compile(r"\bafter\b", re.IGNORECASE),
]

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
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    raw = str(raw)
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


def _extract_names(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("name", "value", "key"):
            if value.get(key):
                return [str(value[key])]
        return []
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            names.extend(_extract_names(item))
        return names
    return [str(value)]


def _normalise_list_field(values: list[str]) -> str:
    return "|".join(sorted({v.strip() for v in values if v and str(v).strip()}))


def _issue_number(issue_id: str) -> int:
    match = re.search(r"-(\d+)$", issue_id)
    return int(match.group(1)) if match else 0


def _summary_tokens(summary: str) -> set[str]:
    stop_words = {
        "the", "and", "for", "with", "from", "that", "this", "into", "when",
        "what", "have", "will", "does", "your", "task", "issue", "jira",
        "apache", "project", "module", "open", "close", "closed", "fix",
        "fails", "failure", "error", "support", "update", "implement",
    }
    return {
        token for token in re.findall(r"[a-z0-9]{3,}", summary.lower())
        if token not in stop_words
    }


def build_candidate_issue_pool(
    issues: list[dict],
    dependencies: list[dict],
    max_issues: int,
) -> tuple[list[dict], list[dict]]:
    """
    Trim very large corpora to a graph-worthy candidate pool before running
    expensive heuristic augmentation.

    This keeps preprocessing fast without compromising the final graph because
    we preserve unresolved, risky, and explicitly connected neighborhoods first.
    """
    if len(issues) <= max(max_issues * 4, 12000):
        return issues, dependencies

    issue_map = {issue["issue_id"]: issue for issue in issues}
    degree: dict[str, int] = {iid: 0 for iid in issue_map}
    adjacency: dict[str, set[str]] = {iid: set() for iid in issue_map}
    for dep in dependencies:
        src = dep["source"]
        tgt = dep["target"]
        if src in issue_map and tgt in issue_map:
            degree[src] += 1
            degree[tgt] += 1
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)

    target_pool_size = min(max(len(issue_map), 0), max(max_issues * 5, 15000))
    unresolved_ids = [iid for iid, issue in issue_map.items() if issue["status"] != "Done"]
    done_ids = [iid for iid, issue in issue_map.items() if issue["status"] == "Done"]

    def rank(iid: str) -> tuple:
        issue = issue_map[iid]
        return (
            issue["status"] == "Blocked",
            issue["is_delayed"],
            degree[iid],
            issue["priority"] in {"Critical", "High"},
            bool(issue.get("due_date")),
        )

    selected: list[str] = []
    selected_set: set[str] = set()

    def add(iid: str):
        if iid in issue_map and iid not in selected_set and len(selected) < target_pool_size:
            selected.append(iid)
            selected_set.add(iid)

    unresolved_quota = min(len(unresolved_ids), max(int(target_pool_size * 0.82), min(len(unresolved_ids), max_issues * 2)))
    done_quota = max(target_pool_size - unresolved_quota, 0)

    for iid in sorted(unresolved_ids, key=rank, reverse=True):
        if len(selected) >= unresolved_quota:
            break
        add(iid)

    frontier = list(selected)
    seen_frontier = set(frontier)
    while frontier and len(selected) < unresolved_quota:
        current = frontier.pop(0)
        for neighbor in sorted(adjacency.get(current, set()), key=lambda nid: (issue_map[nid]["status"] != "Done", degree[nid]), reverse=True):
            if issue_map[neighbor]["status"] == "Done":
                continue
            before = len(selected)
            add(neighbor)
            if len(selected) > before and neighbor not in seen_frontier:
                frontier.append(neighbor)
                seen_frontier.add(neighbor)
            if len(selected) >= unresolved_quota:
                break

    for iid in sorted(done_ids, key=rank, reverse=True):
        if len(selected) >= unresolved_quota + done_quota:
            break
        add(iid)

    candidate_ids = set(selected)
    filtered_issues = [issue_map[iid] for iid in selected]
    filtered_dependencies = [
        dep for dep in dependencies
        if dep["source"] in candidate_ids and dep["target"] in candidate_ids
    ]
    return filtered_issues, filtered_dependencies


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

    components = _normalise_list_field(_extract_names(fields.get("components")))
    labels = _normalise_list_field(_extract_names(fields.get("labels")))
    fix_versions = _normalise_list_field(
        _extract_names(fields.get("fixVersions") or fields.get("fix_versions"))
    )
    issue_type = _normalise_list_field(_extract_names(fields.get("issuetype") or fields.get("issue_type")))

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
        "components":  components,
        "labels":      labels,
        "fix_versions": fix_versions,
        "issue_type":  issue_type,
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


def _extract_issue_id(raw: dict) -> Optional[str]:
    fields = raw.get("fields", raw)
    issue_id = raw.get("key") or raw.get("id") or fields.get("key") or fields.get("id")
    return str(issue_id) if issue_id else None


def _textify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key in ("text", "body", "value", "content", "name"):
            if key in value:
                parts.append(_textify(value[key]))
        return " ".join(p for p in parts if p)
    if isinstance(value, list):
        return " ".join(_textify(v) for v in value)
    return str(value)


def parse_augmented_dependencies(raw: dict) -> list[dict]:
    """
    Augment explicit Jira dependency edges with lightweight inferred edges.

    Sources:
      - explicit issuelinks
      - parent/sub-task hierarchy
      - dependency-like references to other issue IDs in summary/description/comments
    """
    issue_id = _extract_issue_id(raw)
    if not issue_id:
        return []

    fields = raw.get("fields", raw)
    edges = list(parse_dependencies(raw))

    def add_edge(source: str, target: str, link_type: str, direction: str, edge_source: str, confidence: float):
        if not source or not target or source == target:
            return
        edges.append({
            "source": str(source),
            "target": str(target),
            "link_type": link_type,
            "direction": direction,
            "edge_source": edge_source,
            "confidence": round(confidence, 2),
        })

    # Parent/sub-task hierarchy: sub-task depends on parent.
    parent = fields.get("parent")
    if isinstance(parent, dict):
        parent_id = parent.get("key") or parent.get("id")
        if parent_id:
            add_edge(issue_id, str(parent_id), "parent_of", "outward", "parent_child", 0.9)

    # Text-based references with dependency cues.
    text_parts = [
        _textify(fields.get("summary")),
        _textify(fields.get("description")),
        _textify(fields.get("comment")),
        _textify(fields.get("comments")),
    ]
    combined_text = " ".join(part for part in text_parts if part)[:12000]
    if combined_text:
        referenced_ids = {m.group(0).upper() for m in re.finditer(r"\b[A-Z][A-Z0-9]+-\d+\b", combined_text)}
        cue_present = any(pattern.search(combined_text) for pattern in DEPENDENCY_CUE_PATTERNS)
        if cue_present:
            for ref_id in referenced_ids:
                if ref_id != issue_id:
                    add_edge(issue_id, ref_id, "mentions_dependency", "outward", "text_reference", 0.55)

    # Ensure all explicit issuelink edges carry metadata too.
    for edge in edges:
        edge.setdefault("edge_source", "explicit_link")
        edge.setdefault("confidence", 1.0)

    return edges


def augment_dependency_graph(issues: list[dict], dependencies: list[dict]) -> list[dict]:
    """
    Add a denser but still explainable layer of inferred edges.

    The real Jira dump is sparse in explicit dependency links, so we enrich it
    using lightweight project-management heuristics and keep every inferred edge
    labelled with a source + confidence.
    """
    existing: dict[tuple[str, str], dict] = {}
    for dep in dependencies:
        key = (dep["source"], dep["target"])
        prev = existing.get(key)
        if prev is None or float(dep.get("confidence", 0) or 0) > float(prev.get("confidence", 0) or 0):
            existing[key] = dep

    issues_by_project: dict[str, list[dict]] = {}
    for issue in issues:
        issues_by_project.setdefault(issue["project"], []).append(issue)

    for project_issues in issues_by_project.values():
        project_issues.sort(key=lambda item: (_issue_number(item["issue_id"]), item["issue_id"]))
        token_cache = {issue["issue_id"]: _summary_tokens(issue["summary"]) for issue in project_issues}
        component_cache = {issue["issue_id"]: set(filter(None, issue.get("components", "").split("|"))) for issue in project_issues}
        label_cache = {issue["issue_id"]: set(filter(None, issue.get("labels", "").split("|"))) for issue in project_issues}
        version_cache = {issue["issue_id"]: set(filter(None, issue.get("fix_versions", "").split("|"))) for issue in project_issues}
        for source in project_issues:
            if source["status"] == "Done":
                continue

            source_tokens = token_cache[source["issue_id"]]
            source_components = component_cache[source["issue_id"]]
            source_labels = label_cache[source["issue_id"]]
            source_versions = version_cache[source["issue_id"]]

            candidates: list[dict] = []
            for target in project_issues:
                if target["issue_id"] == source["issue_id"]:
                    continue
                if _issue_number(target["issue_id"]) >= _issue_number(source["issue_id"]):
                    continue

                target_components = component_cache[target["issue_id"]]
                target_labels = label_cache[target["issue_id"]]
                target_versions = version_cache[target["issue_id"]]
                shared_tokens = source_tokens & token_cache[target["issue_id"]]

                confidence = 0.0
                edge_type = "semantic_flow"
                issue_gap = max(_issue_number(source["issue_id"]) - _issue_number(target["issue_id"]), 1)

                strong_anchor = False
                if source_components and target_components and source_components & target_components:
                    confidence += 0.35
                    edge_type = "shared_component"
                    strong_anchor = True
                if source_versions and target_versions and source_versions & target_versions:
                    confidence += 0.28
                    edge_type = "shared_fix_version"
                    strong_anchor = True
                if source_labels and target_labels and source_labels & target_labels:
                    confidence += 0.18
                if len(shared_tokens) >= 2:
                    confidence += min(0.30, 0.09 * len(shared_tokens))
                if source.get("assignee") and source.get("assignee") == target.get("assignee"):
                    confidence += 0.14
                if target.get("is_delayed") or target.get("status") == "Blocked":
                    confidence += 0.16
                if source.get("priority") in {"Critical", "High"} and target.get("priority") in {"Critical", "High"}:
                    confidence += 0.10
                if issue_gap <= 3:
                    confidence += 0.18
                    edge_type = "project_flow"
                elif issue_gap <= 8:
                    confidence += 0.10
                if source.get("issue_type") and target.get("issue_type") and source.get("issue_type") == target.get("issue_type"):
                    confidence += 0.08
                if target.get("status") in {"Blocked", "In Progress"}:
                    confidence += 0.08

                # Keep inferred edges believable: require a strong anchor or a
                # very high confidence score before adding a heuristic edge.
                if confidence < 0.74:
                    continue
                if not strong_anchor and confidence < 0.88:
                    continue

                candidates.append({
                    "source": source["issue_id"],
                    "target": target["issue_id"],
                    "link_type": edge_type,
                    "direction": "outward",
                    "edge_source": "heuristic_inference",
                    "confidence": round(min(confidence, 0.95), 2),
                })

            candidates.sort(key=lambda dep: dep["confidence"], reverse=True)
            for dep in candidates[:1]:
                key = (dep["source"], dep["target"])
                prev = existing.get(key)
                if prev is None or dep["confidence"] > float(prev.get("confidence", 0) or 0):
                    existing[key] = dep

        # Add local project-flow edges between nearby unresolved issues so the
        # resulting subgraph behaves more like an execution plan than a set of
        # isolated tickets.
        unresolved = [issue for issue in project_issues if issue["status"] != "Done"]
        for idx, source in enumerate(unresolved):
            for target in unresolved[max(0, idx - 1):idx]:
                if target["issue_id"] == source["issue_id"]:
                    continue
                key = (source["issue_id"], target["issue_id"])
                if key in existing:
                    continue

                shared_component = source.get("components") and source.get("components") == target.get("components")
                shared_fix_version = source.get("fix_versions") and source.get("fix_versions") == target.get("fix_versions")
                if not shared_component or not shared_fix_version:
                    continue

                base_conf = 0.62
                if shared_component:
                    base_conf += 0.14
                if shared_fix_version:
                    base_conf += 0.12
                if not (target.get("status") == "Blocked" or target.get("is_delayed")):
                    continue
                base_conf += 0.10
                if source.get("priority") in {"Critical", "High"}:
                    base_conf += 0.06

                existing[key] = {
                    "source": source["issue_id"],
                    "target": target["issue_id"],
                    "link_type": "project_flow",
                    "direction": "outward",
                    "edge_source": "heuristic_inference",
                    "confidence": round(min(base_conf, 0.82), 2),
                }

    return sorted(
        existing.values(),
        key=lambda dep: (
            dep["source"],
            -float(dep.get("confidence", 0) or 0),
            dep["target"],
        ),
    )


# ─────────────────────────────────────────────
# Dataset loader (handles multiple raw formats)
# ─────────────────────────────────────────────

def load_raw_issues(path: Path) -> Iterable[dict]:
    """
    Load raw issues from a file.
    Handles:
      - JSON array:             [ {issue}, {issue}, ... ]
      - JSON lines (JSONL):     one JSON object per line
      - Wrapped JSON object:    { "issues": [ ... ] }
    """
    log.info("Loading raw data from %s", path)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    is_gz = suffixes[-1:] == [".gz"]
    data_suffix = suffixes[-2] if is_gz and len(suffixes) >= 2 else (suffixes[-1] if suffixes else "")

    opener = gzip.open if is_gz else open

    if data_suffix in (".json", ".jsonl"):
        with opener(path, "rt", encoding="utf-8") as f:
            content = f.read().strip()

        # JSONL — one object per line
        if content.startswith("{") and "\n" in content:
            def iter_jsonl() -> Iterator[dict]:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            return iter_jsonl()

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

    elif data_suffix == ".csv":
        def iter_csv() -> Iterator[dict]:
            with opener(path, "rt", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield dict(row)
        return iter_csv()

    elif data_suffix == ".bson":
        try:
            from bson import decode_file_iter
        except ImportError as exc:
            raise ImportError(
                "Reading Zenodo .bson/.bson.gz files requires the 'pymongo' package. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        def iter_bson() -> Iterator[dict]:
            with opener(path, "rb") as f:
                yield from decode_file_iter(f)
        return iter_bson()

    raise ValueError(f"Unsupported file format: {''.join(suffixes) or path.suffix}")


def load_raw_issues_from_mongodb(
    uri: str,
    database: str,
    collection: str,
    project_filter: Optional[str] = None,
) -> Iterable[dict]:
    """Stream raw issues directly from MongoDB."""
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise ImportError(
            "Reading from MongoDB requires the 'pymongo' package. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    log.info("Streaming raw data from MongoDB %s/%s", database, collection)
    client = MongoClient(uri)
    coll = client[database][collection]

    query = {}
    if project_filter:
        query = {
            "$or": [
                {"fields.project.key": project_filter},
                {"project.key": project_filter},
                {"project": project_filter},
                {"key": {"$regex": f"^{re.escape(project_filter)}-"}} ,
            ]
        }

    def iter_mongo() -> Iterator[dict]:
        try:
            cursor = coll.find(query, no_cursor_timeout=True)
            for doc in cursor:
                yield doc
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            client.close()

    return iter_mongo()


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
            deps.append({
                "source": src,
                "target": tgt,
                "link_type": "blocks",
                "direction": "outward",
                "edge_source": "explicit_link",
                "confidence": 1.0,
            })
            seen_edges.add((src, tgt))

    # Fan-out: some central nodes block many others
    hubs = random.sample(ids[:50], k=min(10, len(ids[:50])))
    for hub in hubs:
        targets = random.sample([x for x in ids if x != hub], k=random.randint(2, 5))
        for t in targets:
            if (hub, t) not in seen_edges:
                deps.append({
                    "source": hub,
                    "target": t,
                    "link_type": "blocks",
                    "direction": "outward",
                    "edge_source": "explicit_link",
                    "confidence": 1.0,
                })
                seen_edges.add((hub, t))

    # Random additional edges
    for _ in range(min(50, n_issues // 4)):
        src, tgt = random.sample(ids, 2)
        if (src, tgt) not in seen_edges:
            deps.append({
                "source": src,
                "target": tgt,
                "link_type": "depends on",
                "direction": "outward",
                "edge_source": "explicit_link",
                "confidence": 0.95,
            })
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
    edge_source_counts: dict[str, int] = {}
    for d in dependencies:
        source = d.get("edge_source", "explicit_link")
        edge_source_counts[source] = edge_source_counts.get(source, 0) + 1

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
        "edge_source_counts":       edge_source_counts,
    }


# ─────────────────────────────────────────────
# Dependency density control
# ─────────────────────────────────────────────

def cap_dependency_density(dependencies: list[dict], max_issues: int) -> list[dict]:
    """
    Preserve stronger edges first and cap weak heuristic density.
    """
    if not dependencies:
        return dependencies

    trustworthy = [
        d for d in dependencies
        if d.get("edge_source") in {"explicit_link", "parent_child", "text_reference"}
    ]
    heuristic = [d for d in dependencies if d.get("edge_source") == "heuristic_inference"]
    target_total = max(int(max_issues * 0.85), len(trustworthy))
    remaining = max(target_total - len(trustworthy), 0)
    heuristic.sort(
        key=lambda d: (
            float(d.get("confidence", 0) or 0),
            d.get("link_type") in {"shared_component", "shared_fix_version"},
        ),
        reverse=True,
    )
    return trustworthy + heuristic[:remaining]


# ─────────────────────────────────────────────
# Subset selection
# ─────────────────────────────────────────────

def select_connected_subset(
    issues: list[dict],
    dependencies: list[dict],
    max_issues: int,
) -> list[dict]:
    """Prefer connected and risky issues so the final graph stays useful."""
    if len(issues) <= max_issues:
        return issues

    issue_map = {i["issue_id"]: i for i in issues}
    degree: dict[str, float] = {iid: 0.0 for iid in issue_map}
    adjacency: dict[str, set[str]] = {iid: set() for iid in issue_map}
    for dep in dependencies:
        src = dep["source"]
        tgt = dep["target"]
        if src in issue_map and tgt in issue_map:
            weight = float(dep.get("confidence", 1.0) or 1.0)
            degree[src] += weight
            degree[tgt] += weight
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)

    selected: list[str] = []
    selected_set: set[str] = set()

    def add(issue_id: str):
        if issue_id in issue_map and issue_id not in selected_set and len(selected) < max_issues:
            selected.append(issue_id)
            selected_set.add(issue_id)

    def activity_score(issue: dict) -> tuple:
        return (
            issue["status"] != "Done",
            issue["status"] == "Blocked",
            issue["is_delayed"],
            issue["priority"] in {"Critical", "High"},
            degree[issue["issue_id"]],
        )

    unresolved_ids = [iid for iid, issue in issue_map.items() if issue["status"] != "Done"]
    done_ids = [iid for iid, issue in issue_map.items() if issue["status"] == "Done"]

    # Reserve most of the subset for active work so the live risk engine has
    # enough unresolved structure to propagate over.
    unresolved_quota = min(
        len(unresolved_ids),
        max(int(max_issues * 0.78), min(len(unresolved_ids), 250)),
    )
    done_quota = max_issues - unresolved_quota

    connected_unresolved = sorted(
        unresolved_ids,
        key=lambda iid: (
            issue_map[iid]["status"] == "Blocked",
            issue_map[iid]["is_delayed"],
            degree[iid],
            issue_map[iid]["priority"] in {"Critical", "High"},
        ),
        reverse=True,
    )

    for iid in connected_unresolved:
        if len(selected) >= unresolved_quota:
            break
        if degree[iid] > 0:
            add(iid)

    frontier = list(selected)
    seen_frontier = set(frontier)
    while frontier and len(selected) < unresolved_quota:
        current = frontier.pop(0)
        for neighbor in sorted(adjacency.get(current, set()), key=lambda iid: degree[iid], reverse=True):
            if issue_map[neighbor]["status"] == "Done":
                continue
            before = len(selected)
            add(neighbor)
            if len(selected) > before and neighbor not in seen_frontier:
                frontier.append(neighbor)
                seen_frontier.add(neighbor)
            if len(selected) >= unresolved_quota:
                break

    unresolved_remaining = sorted(
        [iid for iid in unresolved_ids if iid not in selected_set],
        key=lambda iid: activity_score(issue_map[iid]),
        reverse=True,
    )
    for iid in unresolved_remaining:
        if len(selected) >= unresolved_quota:
            break
        add(iid)

    connected_done = sorted(
        done_ids,
        key=lambda iid: (
            degree[iid],
            issue_map[iid]["priority"] in {"Critical", "High"},
        ),
        reverse=True,
    )
    for iid in connected_done:
        if len(selected) >= unresolved_quota + done_quota:
            break
        add(iid)

    for iid in issue_map:
        if len(selected) >= max_issues:
            break
        add(iid)

    return [issue_map[iid] for iid in selected]


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    input_path: Optional[Path],
    output_dir: Path,
    max_issues: int,
    project_filter: Optional[str],
    synthetic: bool,
    mongodb_uri: Optional[str] = None,
    mongodb_database: Optional[str] = None,
    mongodb_collection: Optional[str] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    issues_out  = output_dir / "issues.csv"
    deps_out    = output_dir / "dependencies.csv"
    stats_out   = output_dir / "stats.json"

    # ── Step 1: Load ──────────────────────────────────────────────────
    if synthetic:
        log.info("Generating synthetic dataset with %d issues …", max_issues)
        issues, dependencies = generate_synthetic_dataset(max_issues)
    else:
        if mongodb_uri:
            if not mongodb_database or not mongodb_collection:
                raise ValueError("--mongodb-db and --mongodb-collection are required with --mongodb-uri")
            raw_issues = load_raw_issues_from_mongodb(
                mongodb_uri, mongodb_database, mongodb_collection, project_filter
            )
        elif input_path is not None:
            raw_issues = load_raw_issues(input_path)
        else:
            raise ValueError("Provide --input, --synthetic, or --mongodb-uri")

        # ── Step 2: Parse & normalise ─────────────────────────────────
        issues = []
        dependencies = []
        skipped = 0
        loaded = 0

        for raw in raw_issues:
            loaded += 1
            if loaded % 50000 == 0:
                log.info("Scanned %d raw records so far…", loaded)
            parsed = parse_issue(raw)
            if parsed is None:
                skipped += 1
                continue

            # Optional project filter
            if project_filter and parsed["project"].upper() != project_filter.upper():
                continue

            issues.append(parsed)
            dependencies.extend(parse_augmented_dependencies(raw))

        pre_pool_issue_count = len(issues)
        pre_pool_dep_count = len(dependencies)
        issues, dependencies = build_candidate_issue_pool(issues, dependencies, max_issues)
        log.info(
            "Candidate pool reduced from %d issues / %d explicit edges to %d issues / %d explicit edges before augmentation",
            pre_pool_issue_count, pre_pool_dep_count, len(issues), len(dependencies),
        )

        explicit_edges = len(dependencies)
        dependencies = augment_dependency_graph(issues, dependencies)
        log.info(
            "Scanned %d raw records; parsed %d valid issues, %d skipped, %d dependency edges found (%d explicit, %d augmented)",
            loaded, len(issues), skipped, len(dependencies), explicit_edges, max(len(dependencies) - explicit_edges, 0),
        )

        # ── Step 3: Subset ────────────────────────────────────────────
        # Preserve connected structure first so the graph remains useful.
        if len(issues) > max_issues:
            issues = select_connected_subset(issues, dependencies, max_issues)
            log.info("Subsetted to %d issues (connectivity-aware selection)", len(issues))

        # Prune dependencies to only include edges where both nodes are in our subset
        valid_ids = {i["issue_id"] for i in issues}
        before = len(dependencies)
        dependencies = [
            d for d in dependencies
            if d["source"] in valid_ids and d["target"] in valid_ids
        ]
        dependencies = cap_dependency_density(dependencies, max_issues)
        log.info(
            "Pruned dependencies: %d → %d (kept only intra-subset edges, then capped density)",
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
        "components", "labels", "fix_versions", "issue_type",
    ]
    with open(issues_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=issue_fields)
        writer.writeheader()
        writer.writerows(issues)

    dep_fields = ["source", "target", "link_type", "direction", "edge_source", "confidence"]
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
        help="Path to raw dataset file (.json, .jsonl, .csv, .bson, or .bson.gz). "
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
    parser.add_argument(
        "--mongodb-uri",
        type=str,
        default=None,
        help="Optional MongoDB connection string. Use this instead of --input to stream records from MongoDB.",
    )
    parser.add_argument(
        "--mongodb-db",
        type=str,
        default=None,
        help="MongoDB database name used with --mongodb-uri.",
    )
    parser.add_argument(
        "--mongodb-collection",
        type=str,
        default=None,
        help="MongoDB collection name used with --mongodb-uri.",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        max_issues=args.max_issues,
        project_filter=args.project,
        synthetic=args.synthetic,
        mongodb_uri=args.mongodb_uri,
        mongodb_database=args.mongodb_db,
        mongodb_collection=args.mongodb_collection,
    )


if __name__ == "__main__":
    main()
