from __future__ import annotations
"""
IssueGraphAgent++ — CSV-backed In-Memory Graph DB
===================================================
A drop-in replacement for the Neo4j GraphDB class that reads directly
from the processed CSVs. This lets you run the FULL API pipeline
(all agents, risk propagation, LLM) without Neo4j installed.

Switch back to Neo4j any time by setting USE_NEO4J=true in your .env.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class CsvGraphDB:
    """
    Mimics the Neo4j GraphDB interface using in-memory data loaded from
    data/processed/issues.csv and data/processed/dependencies.csv.

    Supported query patterns (matching what agents.py calls):
      - MATCH (n:Issue) ... RETURN n {.*} AS props
      - MATCH (src)-[:DEPENDS_ON]->(tgt) RETURN src.issue_id, tgt.issue_id
      - MATCH (n:Issue {issue_id: $id}) RETURN n {.*} AS props
      - MATCH path = (...)-[:DEPENDS_ON*]->(...) RETURN upstream {.*}
      - MATCH (delayed:Issue) WHERE delayed.is_delayed = true ...
      - MATCH (n:Issue) WHERE n.status <> 'Done' ...
    """

    def __init__(
        self,
        issues_path: str = "data/processed/issues.csv",
        deps_path: str   = "data/processed/dependencies.csv",
    ):
        self._issues: "dict" = {}   # issue_id → row dict
        self._deps: "list" = []  # (src, tgt, link_type)

        self._load_issues(Path(issues_path))
        self._load_deps(Path(deps_path))
        log.info(
            "CsvGraphDB loaded: %d issues, %d edges",
            len(self._issues), len(self._deps),
        )

    # ── Loaders ───────────────────────────────────────────────────────

    def _load_issues(self, path: Path):
        if not path.exists():
            log.warning("Issues CSV not found at %s — run preprocess.py first", path)
            return
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["is_delayed"] = row.get("is_delayed", "false").lower() in ("true", "1")
                try:
                    row["delay_days"] = float(row["delay_days"]) if row.get("delay_days") else None
                except ValueError:
                    row["delay_days"] = None
                self._issues[row["issue_id"]] = row

    def _load_deps(self, path: Path):
        if not path.exists():
            log.warning("Dependencies CSV not found at %s", path)
            return
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._deps.append((row["source"], row["target"], row.get("link_type", "depends on")))

    # ── Query interface (mimics neo4j driver .run()) ──────────────────

    def run(self, query: str, parameters: Optional[dict] = None) -> "_ResultSet":
        """
        Parse and execute a simplified subset of Cypher.
        Returns an iterable of record dicts.
        """
        params = parameters or {}
        q = query.strip().lower()

        # Pattern: count delayed issues
        if "is_delayed = true" in q and "count(n)" in q:
            n = sum(1 for v in self._issues.values() if v["is_delayed"])
            return _ResultSet([{"c": n}])

        # Pattern: fetch ALL issue nodes including Done (for counterfactual)
        if "match (n:issue" in q and "n.status <> 'done'" not in q and "return n {.*}" in q:
            project = params.get("project")
            rows = [
                {"props": dict(v)}
                for v in self._issues.values()
                if (not project or v["project"] == project)
            ]
            return _ResultSet(rows)

        # Pattern: fetch all issue nodes (not Done)
        if "match (n:issue" in q and "n.status <> 'done'" in q and "return n {.*}" in q:
            project = params.get("project")
            rows = [
                {"props": dict(v)}
                for v in self._issues.values()
                if v["status"] != "Done"
                and (not project or v["project"] == project)
            ]
            return _ResultSet(rows)

        # Pattern: fetch all issue nodes including Done (for snapshot)
        if "match (n:issue" in q and "return n {.*}" not in q and "return n.issue_id" in q:
            rows = [
                {"id": v["issue_id"], "status": v["status"], "delay_days": v["delay_days"]}
                for v in self._issues.values()
                if v["status"] != "Done"
            ]
            return _ResultSet(rows)

        # Pattern: snapshot (status + delay per open issue)
        if "match (n:issue" in q and "n.status <> 'done'" in q and "return n.issue_id" in q:
            rows = [
                {"id": v["issue_id"], "status": v["status"], "delay_days": v["delay_days"]}
                for v in self._issues.values()
                if v["status"] != "Done"
            ]
            return _ResultSet(rows)

        # Pattern: fetch single issue by id
        if "match (n:issue {issue_id:" in q and "return n {.*}" in q:
            issue_id = params.get("issue_id", "")
            node = self._issues.get(issue_id)
            if node:
                return _ResultSet([{"props": dict(node)}])
            return _ResultSet([])

        # Pattern: fetch all DEPENDS_ON edges
        if "match (src:issue)-[:depends_on]->(tgt:issue)" in q and "return src.issue_id" in q:
            rows = [
                {"source": src, "target": tgt}
                for src, tgt, _ in self._deps
                if src in self._issues and tgt in self._issues
            ]
            return _ResultSet(rows)

        # Pattern: multi-hop upstream traversal
        if "depends_on*" in q and "return distinct upstream" in q:
            issue_id = params.get("issue_id", "")
            max_hops = params.get("max_hops", 8)
            upstream_ids = self._multi_hop_upstream(issue_id, max_hops)
            rows = [
                {"props": dict(self._issues[uid])}
                for uid in upstream_ids
                if uid in self._issues
            ]
            return _ResultSet(rows)

        # Pattern: edges between a set of nodes (for /graph endpoint)
        if "src.issue_id in $ids" in q:
            ids = set(params.get("ids", []))
            rows = [
                {"source": src, "target": tgt, "link_type": lt}
                for src, tgt, lt in self._deps
                if src in ids and tgt in ids
            ]
            return _ResultSet(rows)

        # Pattern: blockers of delayed issues
        if "match (blocker:issue)-[:depends_on]->(blocked:issue)" in q and "blocked.is_delayed = true" in q:
            blocker_ids = {
                src for src, tgt, _ in self._deps
                if tgt in self._issues and self._issues[tgt]["is_delayed"] and src in self._issues
            }
            rows = [{"props": dict(self._issues[iid])} for iid in blocker_ids]
            return _ResultSet(rows)

        # Pattern: blocked/delayed issues with dependents (for CypherBaseline)
        if "is_delayed = true" in q:
            delayed_ids = {iid for iid, v in self._issues.items() if v["is_delayed"]}
            # find immediate dependents
            dependent_ids = {
                src for src, tgt, _ in self._deps if tgt in delayed_ids
            }
            all_ids = list(delayed_ids | dependent_ids)[:params.get("k", 10)]
            return _ResultSet([{"id": iid} for iid in all_ids])

        # Pattern: status distribution (for validate)
        if "return n.status as status, count" in q:
            from collections import Counter
            counts = Counter(v["status"] for v in self._issues.values())
            rows = [{"status": s, "count": c} for s, c in counts.items()]
            return _ResultSet(rows)

        # Pattern: count queries
        if "return count(n)" in q:
            return _ResultSet([{"c": len(self._issues)}])
        if "return count(r)" in q:
            return _ResultSet([{"c": len(self._deps)}])

        # Pattern: max chain depth (simplified)
        if "length(path)" in q:
            depth = self._estimate_max_depth()
            return _ResultSet([{"depth": depth}])

        # Default: return empty (unknown query pattern)
        log.debug("CsvGraphDB: unrecognised query pattern, returning empty. Query: %s", query[:80])
        return _ResultSet([])

    def run_write(self, query: str, parameters: Optional[dict] = None):
        """Write operations are no-ops for the CSV backend."""
        return []

    def close(self):
        pass

    # ── Internal helpers ──────────────────────────────────────────────

    def _multi_hop_upstream(self, start_id: str, max_hops: int) -> "list":
        """BFS upstream: find all nodes that start_id depends on."""
        # forward adjacency: src depends on tgt  → tgt is upstream of src
        forward: "dict" = {}
        for src, tgt, _ in self._deps:
            forward.setdefault(src, []).append(tgt)

        visited = set()
        queue = [(start_id, 0)]
        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth >= max_hops:
                continue
            visited.add(node)
            for upstream in forward.get(node, []):
                if upstream not in visited:
                    queue.append((upstream, depth + 1))

        visited.discard(start_id)
        return list(visited)

    def _estimate_max_depth(self) -> int:
        """Estimate the longest chain length via BFS from all sources."""
        forward: "dict" = {}
        for src, tgt, _ in self._deps:
            forward.setdefault(src, []).append(tgt)

        max_d = 0
        for start in list(forward.keys())[:20]:   # sample to keep it fast
            visited: dict[str, int] = {}
            queue = [(start, 0)]
            while queue:
                node, d = queue.pop(0)
                if node in visited:
                    continue
                visited[node] = d
                max_d = max(max_d, d)
                for nxt in forward.get(node, []):
                    if nxt not in visited:
                        queue.append((nxt, d + 1))
        return max_d

    def single(self):
        """Compatibility shim — not used directly."""
        return None


class _ResultSet:
    """Iterable result container that also supports .single()."""

    def __init__(self, rows: "list"):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None
