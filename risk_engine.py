from __future__ import annotations
"""
IssueGraphAgent++ — Phase 2: Temporal Risk Propagation Engine
=============================================================
This is the core algorithmic contribution of the paper.

The engine performs multi-hop traversal over the temporal dependency
graph starting from at-risk (delayed / blocked) nodes and propagates
risk scores downstream using a decay-weighted formula.

Risk Score Formula (per node)
──────────────────────────────
  R(v) = Σ [ delay_severity(u) × depth_weight(d) × temporal_weight(u) ]
           for each upstream risky ancestor u at depth d

Where:
  delay_severity(u)  = min(delay_days / 30.0, 1.0)   for delayed nodes
                       1.0                             for blocked nodes
  depth_weight(d)    = γ^(d-1)   γ = 0.8 (configurable)
                       Risk attenuates with distance but never vanishes.
  temporal_weight(u) = freshness of the signal (recently updated = 1.0,
                       stale = decays toward 0.5 over 30 days)

Final risk level thresholds:
  score ≥ 0.7  → High
  score ≥ 0.35 → Medium
  otherwise    → Low

This file is self-contained and Neo4j-agnostic: it can operate on
a plain Python dict representation of the graph (for unit testing)
or be called by GraphReasoningAgent with live Neo4j data.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class IssueNode:
    """In-memory representation of one Issue node."""
    issue_id:    str
    project:     str
    summary:     str
    status:      str                    # Open | In Progress | Done | Blocked
    priority:    str                    # Critical | High | Medium | Low
    assignee:    str
    due_date:    Optional[datetime]
    updated:     Optional[datetime]
    delay_days:  Optional[float]        # positive → overdue
    is_delayed:  bool

    def is_at_risk(self) -> bool:
        """True if this node is a risk origin (seed for propagation)."""
        return self.is_delayed or self.status == "Blocked"

    def priority_multiplier(self) -> float:
        """Higher-priority issues amplify propagated risk."""
        return {"Critical": 1.4, "High": 1.2, "Medium": 1.0, "Low": 0.8}.get(
            self.priority, 1.0
        )


@dataclass
class RiskResult:
    """Output of the risk propagation engine for one issue."""
    issue_id:         str
    summary:          str
    status:           str
    risk_score:       float              # 0.0 – 1.0
    risk_level:       str               # High | Medium | Low
    is_origin:        bool              # True if this node is itself at risk
    delay_days:       Optional[float]
    affected_by:      list[str]         # issue_ids of upstream risky ancestors
    chain:            list[str]         # full dependency path to riskiest ancestor
    explanation:      str               # human-readable, used by Decision Agent
    depth_from_origin: int             # hops from nearest risky ancestor


@dataclass
class PropagationConfig:
    """Tunable parameters for the risk propagation algorithm."""
    depth_decay:         float = 0.88   # γ — gentler attenuation for demo visibility
    temporal_half_life:  float = 45.0   # slightly slower decay for long-running work
    max_depth:           int   = 12     # stop traversal beyond this depth
    high_threshold:      float = 0.58
    medium_threshold:    float = 0.22


# ─────────────────────────────────────────────
# Temporal weighting
# ─────────────────────────────────────────────

def temporal_weight(updated: Optional[datetime], half_life_days: float) -> float:
    """
    Returns a weight in [0.5, 1.0] based on how recently the issue was updated.
    A fresh signal (updated today) returns 1.0.
    A stale signal (updated half_life_days ago) returns ~0.75.
    This prevents very old delays from dominating the risk score.
    """
    if updated is None:
        return 0.75  # conservative: treat unknown as moderately fresh
    now = datetime.now(timezone.utc)
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_days = max((now - updated).total_seconds() / 86400.0, 0.0)
    # Exponential decay: w = 0.5 + 0.5 × e^(-age / half_life)
    return 0.5 + 0.5 * math.exp(-age_days / half_life_days)


def delay_severity(node: IssueNode) -> float:
    """
    Maps a node's delay magnitude to a severity score in [0.0, 1.0].
    Blocked nodes are treated as maximum severity (1.0).
    """
    if node.status == "Blocked":
        return 0.82
    if node.delay_days is not None and node.delay_days > 0:
        # Smoother curve so overdue work does not instantly saturate the graph.
        return min(0.25 + math.log1p(node.delay_days) / 6.0, 0.92)
    return 0.0


# ─────────────────────────────────────────────
# Graph traversal helpers
# ─────────────────────────────────────────────

def _normalise_dependencies(
    dependencies,
) -> list[tuple[str, str, float, str]]:
    normalised: list[tuple[str, str, float, str]] = []
    for dep in dependencies:
        if isinstance(dep, dict):
            src = dep.get("source")
            tgt = dep.get("target")
            confidence = float(dep.get("confidence", 1.0) or 1.0)
            edge_source = dep.get("edge_source", "explicit_link")
        else:
            if len(dep) >= 4:
                src, tgt, confidence, edge_source = dep[:4]
            elif len(dep) >= 3:
                src, tgt, confidence = dep[:3]
                edge_source = "explicit_link"
            else:
                src, tgt = dep[:2]
                confidence = 1.0
                edge_source = "explicit_link"
            confidence = float(confidence or 1.0)
        if src and tgt and src != tgt:
            normalised.append((str(src), str(tgt), max(0.15, min(confidence, 1.0)), str(edge_source)))
    return normalised


def build_reverse_adjacency(
    dependencies: list[tuple[str, str, float, str]]
) -> dict[str, list[tuple[str, float]]]:
    """
    Given a list of (source, target) pairs where source DEPENDS_ON target,
    build a reverse adjacency: for each node, who depends on it?

    This lets us answer: "if X is at risk, which nodes are downstream
    (i.e. depend on X, directly or transitively)?"

    reverse_adj[X] = [A, B, C]  means A, B, C all depend on X.
    """
    rev: dict[str, list[tuple[str, float]]] = {}
    for src, tgt, confidence, _ in dependencies:
        rev.setdefault(tgt, []).append((src, confidence))
    return rev


def find_downstream_nodes(
    origin_id: str,
    reverse_adj: dict[str, list[tuple[str, float]]],
    max_depth: int,
) -> dict[str, tuple[int, float]]:
    """
    BFS from origin_id following reverse edges.
    Returns {issue_id: depth_from_origin} for all reachable dependents.
    """
    visited: dict[str, tuple[int, float]] = {}
    queue = [(origin_id, 0, 1.0)]
    while queue:
        node_id, depth, confidence = queue.pop(0)
        best = visited.get(node_id)
        if depth > max_depth:
            continue
        if best is not None and best[0] <= depth and best[1] >= confidence:
            continue
        visited[node_id] = (depth, confidence)
        for dependent, edge_confidence in reverse_adj.get(node_id, []):
            next_confidence = confidence * edge_confidence
            queue.append((dependent, depth + 1, next_confidence))
    return visited


def find_dependency_chain(
    start_id: str,
    end_id: str,
    forward_adj: dict[str, list[tuple[str, float]]],
    max_depth: int = 10,
) -> list[str]:
    """
    BFS to find the shortest path from start_id to end_id following
    DEPENDS_ON edges. Used to build the explanatory chain.
    Returns the path as a list of issue_ids, or empty list if not found.
    """
    if start_id == end_id:
        return [start_id]
    queue: list[tuple[list[str], float]] = [([start_id], 1.0)]
    best_score = {start_id: 1.0}
    while queue:
        queue.sort(key=lambda item: (len(item[0]), -item[1]))
        path, score = queue.pop(0)
        if len(path) > max_depth:
            continue
        current = path[-1]
        for neighbor, edge_confidence in forward_adj.get(current, []):
            new_score = score * edge_confidence
            if best_score.get(neighbor, -1.0) >= new_score:
                continue
            new_path = path + [neighbor]
            if neighbor == end_id:
                return new_path
            best_score[neighbor] = new_score
            queue.append((new_path, new_score))
    return []


# ─────────────────────────────────────────────
# Main propagation engine
# ─────────────────────────────────────────────

class RiskPropagationEngine:
    """
    Multi-hop temporal risk propagation over a dependency graph.

    Works with plain Python dicts — no Neo4j dependency at this layer.
    The GraphReasoningAgent fetches data from Neo4j and passes it here.

    Usage:
        engine = RiskPropagationEngine(config)
        results = engine.propagate(nodes, dependencies)
    """

    def __init__(self, config: Optional[PropagationConfig] = None):
        self.config = config or PropagationConfig()

    def propagate(
        self,
        nodes: list[IssueNode],
        dependencies,
    ) -> dict[str, RiskResult]:
        """
        Run the full propagation algorithm.

        Returns a dict mapping issue_id → RiskResult for every node
        that received a non-zero risk score.
        """
        cfg = self.config

        # Index nodes for O(1) lookup
        node_map: dict[str, IssueNode] = {n.issue_id: n for n in nodes}

        # Build adjacency structures
        # forward_adj[A] = [B, C]  means A depends on B and C
        normalised_deps = _normalise_dependencies(dependencies)
        forward_adj: dict[str, list[tuple[str, float]]] = {}
        for src, tgt, confidence, _ in normalised_deps:
            forward_adj.setdefault(src, []).append((tgt, confidence))

        # reverse_adj[X] = [A, B]  means A and B depend on X
        reverse_adj = build_reverse_adjacency(normalised_deps)

        # Identify risk origins (delayed / blocked nodes)
        origins = [n for n in nodes if n.is_at_risk() and n.status != "Done"]
        log.info(
            "Propagating risk from %d origin nodes across %d total nodes",
            len(origins), len(nodes),
        )

        # Accumulate risk contributions per node
        # risk_accum[node_id] = list of (score_contribution, origin_id, depth, chain)
        risk_accum: dict[str, list[tuple[float, str, int, list[str]]]] = {}

        for origin in origins:
            sev  = delay_severity(origin)
            tw   = temporal_weight(origin.updated, cfg.temporal_half_life)
            base = sev * tw * origin.priority_multiplier()

            if base <= 0.0:
                continue

            # BFS downstream: find all nodes that depend on this origin
            downstream = find_downstream_nodes(
                origin.issue_id, reverse_adj, cfg.max_depth
            )

            for dep_id, (depth, path_confidence) in downstream.items():
                if dep_id == origin.issue_id:
                    continue  # don't score the origin against itself here
                dep_node = node_map.get(dep_id)
                if dep_node is None or dep_node.status == "Done":
                    continue

                # Contribution from this origin to dep_id at given depth
                contribution = base * (cfg.depth_decay ** max(depth - 1, 0)) * max(path_confidence, 0.45)

                # Find the explanatory chain
                chain = find_dependency_chain(
                    dep_id, origin.issue_id, forward_adj, cfg.max_depth
                )

                risk_accum.setdefault(dep_id, []).append(
                    (contribution, origin.issue_id, depth, chain)
                )

        # Build RiskResult objects
        results: dict[str, RiskResult] = {}

        # First, record all origin nodes
        for origin in origins:
            sev = delay_severity(origin)
            tw = temporal_weight(origin.updated, cfg.temporal_half_life)
            score = min(0.18 + sev * tw * origin.priority_multiplier() * 0.72, 0.96)
            level = self._score_to_level(score)
            summary = self._origin_explanation(origin)
            results[origin.issue_id] = RiskResult(
                issue_id=origin.issue_id,
                summary=origin.summary,
                status=origin.status,
                risk_score=round(score, 3),
                risk_level=level,
                is_origin=True,
                delay_days=origin.delay_days,
                affected_by=[],
                chain=[origin.issue_id],
                explanation=summary,
                depth_from_origin=0,
            )

        # Then, record all downstream nodes that accumulated risk
        for dep_id, contributions in risk_accum.items():
            raw_total = sum(c[0] for c in contributions)
            strongest = max(c[0] for c in contributions)
            avg_depth = sum(c[2] for c in contributions) / max(len(contributions), 1)
            fanin = len(contributions)
            normalised_total = raw_total / (1 + 0.55 * max(fanin - 1, 0))
            breadth_bonus = min(0.10, 0.02 * max(fanin - 1, 0))
            depth_penalty = min(0.18, 0.025 * max(avg_depth - 1, 0))
            total_score = min(
                0.42 * strongest
                + 0.36 * (1 - math.exp(-0.95 * normalised_total))
                + breadth_bonus
                - depth_penalty,
                0.98,
            )
            level = self._score_to_level(total_score)

            # Find the strongest contributing origin
            contributions.sort(key=lambda x: x[0], reverse=True)
            top_origin_id = contributions[0][1]
            top_chain     = contributions[0][3]
            top_depth     = contributions[0][2]

            affected_by = list({c[1] for c in contributions})
            dep_node    = node_map.get(dep_id)
            summary_txt = dep_node.summary if dep_node else dep_id

            explanation = self._propagation_explanation(
                dep_node, contributions, node_map
            )

            results[dep_id] = RiskResult(
                issue_id=dep_id,
                summary=summary_txt,
                status=dep_node.status if dep_node else "Unknown",
                risk_score=round(total_score, 3),
                risk_level=level,
                is_origin=False,
                delay_days=dep_node.delay_days if dep_node else None,
                affected_by=affected_by,
                chain=top_chain,
                explanation=explanation,
                depth_from_origin=top_depth,
            )

        log.info(
            "Propagation complete: %d at-risk nodes identified "
            "(%d High, %d Medium, %d Low)",
            len(results),
            sum(1 for r in results.values() if r.risk_level == "High"),
            sum(1 for r in results.values() if r.risk_level == "Medium"),
            sum(1 for r in results.values() if r.risk_level == "Low"),
        )
        return results

    # ── Helpers ───────────────────────────────────────────────────────

    def _score_to_level(self, score: float) -> str:
        if score >= self.config.high_threshold:
            return "High"
        if score >= self.config.medium_threshold:
            return "Medium"
        return "Low"

    def _origin_explanation(self, node: IssueNode) -> str:
        if node.status == "Blocked":
            return (
                f"{node.issue_id} is blocked and cannot progress. "
                f"All downstream tasks that depend on it are at risk."
            )
        days = node.delay_days or 0
        return (
            f"{node.issue_id} is {days:.1f} days overdue "
            f"(priority: {node.priority}). "
            f"This delay may cascade to dependent tasks."
        )

    def _propagation_explanation(
        self,
        node: Optional[IssueNode],
        contributions: list[tuple[float, str, int, list[str]]],
        node_map: dict[str, IssueNode],
    ) -> str:
        if node is None:
            return "Risk propagated from upstream delays."

        # Sort by contribution descending
        top = contributions[:3]
        origin_parts = []
        for score, origin_id, depth, chain in top:
            origin_node = node_map.get(origin_id)
            if origin_node:
                if origin_node.status == "Blocked":
                    reason = "blocked"
                else:
                    days = origin_node.delay_days or 0
                    reason = f"{days:.1f} days overdue"
                chain_str = " → ".join(chain) if chain else origin_id
                origin_parts.append(
                    f"{origin_id} ({reason}, {depth} hop{'s' if depth!=1 else ''} away) "
                    f"via chain: {chain_str}"
                )

        origins_text = "; ".join(origin_parts) if origin_parts else "upstream delays"
        return (
            f"{node.issue_id} has inherited risk from: {origins_text}. "
            f"Risk score: {min(sum(c[0] for c in contributions), 1.0):.2f}."
        )

    
# ─────────────────────────────────────────────
# Convenience function for agents
# ─────────────────────────────────────────────

def run_propagation(
    nodes: list[IssueNode],
    dependencies: list[tuple[str, str]],
    config: Optional[PropagationConfig] = None,
) -> dict[str, RiskResult]:
    """One-call entry point used by GraphReasoningAgent."""
    engine = RiskPropagationEngine(config)
    return engine.propagate(nodes, dependencies)
