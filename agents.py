from __future__ import annotations
"""
IssueGraphAgent++ — Phase 3: Multi-Agent Layer
===============================================
Six agents implementing the closed-loop agentic pipeline:

  Perception Agent    — ingests graph state from Neo4j / CSV
  Graph Reasoning     — runs risk propagation queries
  Planning Agent      — decomposes the mitigation goal into sub-tasks
  Decision Agent      — Gemini 1.5 Flash powered LLM explanation + recommendations
  Monitoring Agent    — continuous background watcher
  Critic Agent        — validates and filters LLM outputs
"""

import json
import logging
import re
import threading
from datetime import datetime, timezone
from typing import Optional

from core.risk_engine import (
    IssueNode,
    PropagationConfig,
    RiskResult,
    run_propagation,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────

def parse_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def neo4j_record_to_issue_node(record: dict) -> IssueNode:
    return IssueNode(
        issue_id   = record.get("issue_id", ""),
        project    = record.get("project", ""),
        summary    = record.get("summary", ""),
        status     = record.get("status", "Open"),
        priority   = record.get("priority", "Medium"),
        assignee   = record.get("assignee", ""),
        due_date   = parse_datetime(record.get("due_date")),
        updated    = parse_datetime(record.get("updated")),
        delay_days = float(record["delay_days"]) if record.get("delay_days") not in (None, "") else None,
        is_delayed = bool(record.get("is_delayed", False)),
    )


# ─────────────────────────────────────────────
# Perception Agent
# ─────────────────────────────────────────────

class PerceptionAgent:
    def __init__(self, db):
        self.db = db
        self._last_snapshot: Optional[dict] = None

    def fetch_all_nodes(self, project: Optional[str] = None) -> list[IssueNode]:
        if project:
            query = "MATCH (n:Issue {project: $project}) WHERE n.status <> 'Done' RETURN n {.*} AS props"
            records = self.db.run(query, {"project": project})
        else:
            query = "MATCH (n:Issue) WHERE n.status <> 'Done' RETURN n {.*} AS props"
            records = self.db.run(query)
        nodes = [neo4j_record_to_issue_node(r["props"]) for r in records]
        log.debug("PerceptionAgent: fetched %d nodes", len(nodes))
        return nodes

    def fetch_all_edges(self) -> list[tuple[str, str]]:
        query = "MATCH (src:Issue)-[:DEPENDS_ON]->(tgt:Issue) RETURN src.issue_id AS source, tgt.issue_id AS target"
        records = self.db.run(query)
        edges = [(r["source"], r["target"]) for r in records]
        log.debug("PerceptionAgent: fetched %d edges", len(edges))
        return edges

    def fetch_node(self, issue_id: str) -> Optional[IssueNode]:
        query = "MATCH (n:Issue {issue_id: $issue_id}) RETURN n {.*} AS props"
        record = self.db.run(query, {"issue_id": issue_id}).single()
        if record:
            return neo4j_record_to_issue_node(record["props"])
        return None

    def fetch_upstream_chain(self, issue_id: str, max_hops: int = 8) -> list[IssueNode]:
        query = """
        MATCH path = (start:Issue {issue_id: $issue_id})-[:DEPENDS_ON*1..$max_hops]->(upstream:Issue)
        RETURN DISTINCT upstream {.*} AS props
        """
        records = self.db.run(query, {"issue_id": issue_id, "max_hops": max_hops})
        return [neo4j_record_to_issue_node(r["props"]) for r in records]

    def snapshot(self) -> dict:
        query = "MATCH (n:Issue) WHERE n.status <> 'Done' RETURN n.issue_id AS id, n.status AS status, n.delay_days AS delay_days"
        snap = {}
        for r in self.db.run(query):
            snap[r["id"]] = (r["status"], r["delay_days"])
        return snap

    def detect_changes(self) -> list[dict]:
        current = self.snapshot()
        if self._last_snapshot is None:
            self._last_snapshot = current
            return []
        changes = []
        for issue_id, (cur_status, cur_delay) in current.items():
            prev = self._last_snapshot.get(issue_id)
            if prev is None:
                changes.append({"type": "new_issue", "issue_id": issue_id, "status": cur_status})
                continue
            prev_status, prev_delay = prev
            if cur_status != prev_status:
                changes.append({"type": "status_change", "issue_id": issue_id,
                                 "from_status": prev_status, "to_status": cur_status})
            if (cur_delay or 0) > (prev_delay or 0) + 1.0:
                changes.append({"type": "delay_increased", "issue_id": issue_id,
                                 "prev_delay": prev_delay, "cur_delay": cur_delay})
        self._last_snapshot = current
        return changes


# ─────────────────────────────────────────────
# Graph Reasoning Agent
# ─────────────────────────────────────────────

class GraphReasoningAgent:
    def __init__(self, perception: PerceptionAgent, config: Optional[PropagationConfig] = None):
        self.perception = perception
        self.config = config or PropagationConfig()

    def global_risk_analysis(self, project: Optional[str] = None) -> dict[str, RiskResult]:
        nodes = self.perception.fetch_all_nodes(project)
        edges = self.perception.fetch_all_edges()
        return run_propagation(nodes, edges, self.config)

    def issue_risk(self, issue_id: str) -> Optional[RiskResult]:
        results = self.global_risk_analysis()
        return results.get(issue_id)

    def dependency_chain(self, issue_id: str) -> list[IssueNode]:
        return self.perception.fetch_upstream_chain(issue_id)

    def find_root_causes(self, issue_id: str) -> list[IssueNode]:
        chain = self.dependency_chain(issue_id)
        return [n for n in chain if n.is_at_risk()]

    def top_risky_issues(self, k: int = 10, project: Optional[str] = None) -> list[RiskResult]:
        results = self.global_risk_analysis(project)
        sorted_results = sorted(results.values(), key=lambda r: r.risk_score, reverse=True)
        return sorted_results[:k]

    def blocking_issues(self) -> list[IssueNode]:
        query = """
        MATCH (blocker:Issue)-[:DEPENDS_ON]->(blocked:Issue)
        WHERE blocker.status <> 'Done' AND blocked.is_delayed = true
        RETURN DISTINCT blocker {.*} AS props
        """
        records = self.perception.db.run(query)
        return [neo4j_record_to_issue_node(r["props"]) for r in records]


# ─────────────────────────────────────────────
# Planning Agent
# ─────────────────────────────────────────────

class PlanningAgent:
    PRIORITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

    def create_mitigation_plan(
        self,
        risk_results: dict[str, RiskResult],
        node_map: dict[str, IssueNode],
    ) -> list[dict]:
        plan = []
        origins = {iid: r for iid, r in risk_results.items() if r.is_origin}

        for issue_id, result in origins.items():
            node = node_map.get(issue_id)
            downstream_count = sum(1 for r in risk_results.values() if issue_id in r.affected_by)
            priority_rank = self.PRIORITY_RANK.get(node.priority if node else "Medium", 2)
            priority_score = result.risk_score * (1 + downstream_count * 0.1) * priority_rank

            if node and node.status == "Blocked":
                action = "unblock"
                rationale = f"{issue_id} is blocked. Unblocking it will de-risk {downstream_count} downstream task(s)."
            elif result.delay_days and result.delay_days > 7:
                action = "escalate"
                rationale = f"{issue_id} is {result.delay_days:.0f} days overdue — escalation needed. Affects {downstream_count} downstream task(s)."
            elif result.delay_days and result.delay_days > 0:
                action = "fix_delay"
                rationale = f"{issue_id} is {result.delay_days:.0f} day(s) overdue. Resolving it would de-risk {downstream_count} downstream task(s)."
            else:
                action = "monitor"
                rationale = f"{issue_id} shows risk signals. Monitor closely."

            plan.append({
                "issue_id":         issue_id,
                "summary":          result.summary,
                "action":           action,
                "rationale":        rationale,
                "downstream_count": downstream_count,
                "risk_score":       result.risk_score,
                "priority_score":   round(priority_score, 3),
            })

        plan.sort(key=lambda x: x["priority_score"], reverse=True)
        return plan


# ─────────────────────────────────────────────
# Decision Agent — Gemini 1.5 Flash
# ─────────────────────────────────────────────

class DecisionAgent:
    """
    LLM-powered explanation and recommendation layer.

    Uses a graph-aware system prompt + structured task-specific prompts
    (Steps 1, 2, 4, 5 of the prompt engineering plan).
    The LLM never does risk computation — it only explains and recommends
    based on data already computed by the deterministic engine.
    """

    # ── STEP 5: Reusable graph-aware system prompt ────────────────────
    SYSTEM_PROMPT = """You are a graph-aware AI system specialised in software project risk analysis.

You reason using:
- Dependency graphs (which issue blocks which)
- Risk propagation scores (computed by a deterministic engine)
- Temporal delays (how many days overdue each issue is)

You NEVER:
- Invent dependencies not present in the provided data
- Assume missing issue IDs or delay values
- Give generic answers that ignore the actual issue IDs

You ALWAYS:
- Reference issue IDs explicitly (e.g., HADOOP-16, KAFKA-23)
- Explain risk using graph relationships (X is blocked because Y depends on Z)
- Base all reasoning ONLY on the structured data provided

Respond ONLY with a valid JSON object. No markdown, no preamble, no explanation outside the JSON.
Required keys: summary (string), root_causes (list of strings), recommendations (list of strings), confidence (float 0-1).
"""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.model_name = model
        self.api_key    = api_key
        self._client    = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            log.info("DecisionAgent: Groq client ready (model: %s)", self.model_name)
            return self._client
        except ImportError:
            raise ImportError("groq not installed. Run: pip install groq")

    # ── STEP 1: Structured project risk summary prompt ────────────────
    def summarise_project_risks(self, top_risks, plan):
        """
        Upgraded prompt: identifies root causes, explains propagation,
        and gives top 3 actionable mitigations. (Step 1)
        """
        risk_lines = []
        for r in top_risks[:10]:
            line = (
                f"  - {r.issue_id}: risk={r.risk_score:.2f} level={r.risk_level}"
                f" is_root_cause={r.is_origin}"
            )
            if r.delay_days:
                line += f" delay={r.delay_days:.1f}d"
            if r.affected_by:
                line += f" blocked_by={r.affected_by[:3]}"
            if r.chain:
                line += f" chain={' → '.join(r.chain[:5])}"
            risk_lines.append(line)

        plan_lines = []
        for p in plan[:5]:
            plan_lines.append(
                f"  - {p['issue_id']}: action={p['action']} "
                f"downstream_impact={p['downstream_count']} rationale={p['rationale']}"
            )

        prompt = f"""You are a software project risk analyst.

You are given:
1. A list of high-risk issues with their dependency chains
2. A computed mitigation plan

Your tasks:
1. Identify the root cause issues (those marked is_root_cause=True — these have no upstream blockers)
2. Explain how risk propagates across the dependency graph (trace the chains)
3. Suggest the top 3 most effective mitigation actions based on downstream impact

STRICT RULES:
- Use ONLY the provided data
- Mention issue IDs explicitly (e.g., HADOOP-16, KAFKA-23)
- Keep explanations concise and technical
- Do NOT hallucinate missing dependencies

DATA:
Top Risk Issues:
{chr(10).join(risk_lines)}

Mitigation Plan:
{chr(10).join(plan_lines) if plan_lines else "  No plan generated."}

Respond with a JSON object: summary, root_causes, recommendations, confidence."""
        return self._call_llm(prompt)

    # ── STEP 1 variant: focused summary for specific query types ──────
    def summarise_project_risks_focused(self, top_risks, plan, focus_instruction: str):
        """Focused variant that adjusts the analysis based on query intent."""
        risk_lines = []
        for r in top_risks[:8]:
            line = (
                f"  - {r.issue_id}: risk={r.risk_score:.2f} level={r.risk_level}"
                f" is_root_cause={r.is_origin} status={r.status}"
            )
            if r.delay_days:
                line += f" delay={r.delay_days:.1f}d"
            if r.affected_by:
                line += f" blocked_by={r.affected_by[:3]}"
            if r.chain:
                line += f" dep_chain={' → '.join(r.chain[:4])}"
            if r.explanation:
                line += f" context: {r.explanation[:100]}"
            risk_lines.append(line)

        prompt = f"""You are a software project risk analyst.

FOCUS INSTRUCTION: {focus_instruction}

Project risk data:
{chr(10).join(risk_lines)}

Action plan top items:
{chr(10).join(f"  - {p['issue_id']}: {p['action']} — {p['rationale']}" for p in plan[:3])}

Answer the user's specific question using the data above.
Reference issue IDs. Explain using dependency relationships.
Respond with a JSON object: summary, root_causes, recommendations, confidence."""
        return self._call_llm(prompt)

    # ── STEP 2: Context-aware issue-specific prompt ───────────────────
    def explain_issue_risk(self, issue_id, risk_result, chain, plan):
        """
        Upgraded prompt: uses full dependency context and query-aware
        instructions to give a precise, non-generic explanation. (Step 2)
        """
        upstream_lines = []
        for n in chain[:10]:
            line = f"  - {n.issue_id}: status={n.status}"
            if n.is_delayed:
                line += f" OVERDUE by {n.delay_days:.1f}d"
            upstream_lines.append(line)

        dep_chain_str = " → ".join(risk_result.chain) if risk_result.chain else "no chain"
        affected_str  = ", ".join(risk_result.affected_by[:5]) if risk_result.affected_by else "none"

        prompt = f"""You are an intelligent assistant for software project risk analysis.

Context:
- Issue ID: {issue_id}
- Risk Score: {risk_result.risk_score:.2f} ({risk_result.risk_level})
- Status: {risk_result.status}
- Is Root Cause: {risk_result.is_origin}
- Delay: {f"{risk_result.delay_days:.1f} days overdue" if risk_result.delay_days else "no deadline data"}
- Dependency chain: {dep_chain_str}
- Blocked by (upstream at-risk issues): {affected_str}

Upstream dependency nodes:
{chr(10).join(upstream_lines) if upstream_lines else "  No upstream dependencies found."}

Recommended actions for this issue:
{chr(10).join(f"  - {p['action']}: {p['rationale']}" for p in plan[:3]) if plan else "  No specific plan."}

Instructions:
- Answer specifically about {issue_id}, not generically
- If the question involves risk, explain using the dependency chain above
- Identify which upstream issue is the root blocker
- Be precise and avoid repeating the raw data — synthesise it into insight

Respond with a JSON object: summary, root_causes, recommendations, confidence."""
        return self._call_llm(prompt)

    # ── STEP 4: Counterfactual / what-if explanation ──────────────────
    def explain_counterfactual(self, issue_id: str, diff: dict) -> dict:
        """
        Explains what would happen if issue_id were resolved.
        Called by GraphReasoningAgent.simulate_resolution(). (Step 4)
        """
        improved = [
            f"  - {iid}: {d['before']:.2f} → {d['after']:.2f} (Δ {d['delta']:.2f})"
            for iid, d in diff.items()
            if d.get("delta", 0) > 0.05
        ]
        unchanged = sum(1 for d in diff.values() if d.get("delta", 0) <= 0.05)

        prompt = f"""You are analyzing a what-if simulation in a project dependency graph.

Scenario: Issue {issue_id} was resolved.

Risk changes in downstream nodes (before → after):
{chr(10).join(improved) if improved else "  No significant downstream risk reduction."}
Unchanged nodes: {unchanged}

Tasks:
1. Explain WHY risk decreased in those downstream nodes (trace the dependency relationship)
2. Identify which downstream nodes benefited most
3. Summarize the overall impact of resolving {issue_id}

Rules:
- Use issue IDs explicitly
- Base reasoning ONLY on the dependency relationships shown
- Keep it short and analytical (under 200 words)

Respond with a JSON object: summary, root_causes (what was unblocked), recommendations, confidence."""
        return self._call_llm(prompt)

    def _call_llm(self, user_prompt: str) -> dict:
        """Call Groq API with the graph-aware system prompt prepended."""
        import traceback as _tb
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model    = self.model_name,
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature     = 0.2,
                max_tokens      = 800,
                response_format = {"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            log.error("DecisionAgent LLM call failed: %s\n%s", e, _tb.format_exc())
            return {
                "summary":         "LLM unavailable. See structured data.",
                "root_causes":     [],
                "recommendations": [],
                "confidence":      0.0,
                "error":           str(e),
            }


class MonitoringAgent:
    def __init__(
        self,
        perception: PerceptionAgent,
        reasoning: GraphReasoningAgent,
        poll_interval_secs: int = 30,
    ):
        self.perception    = perception
        self.reasoning     = reasoning
        self.poll_interval = poll_interval_secs
        self._alerts: list[dict] = []
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MonitoringAgent")
        self._thread.start()
        log.info("MonitoringAgent started (interval: %ds)", self.poll_interval)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("MonitoringAgent stopped.")

    def get_alerts(self, clear: bool = True) -> list[dict]:
        with self._lock:
            alerts = list(self._alerts)
            if clear:
                self._alerts.clear()
        return alerts

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                log.error("MonitoringAgent poll error: %s", e)
            self._stop_event.wait(self.poll_interval)

    def _poll_cycle(self):
        changes = self.perception.detect_changes()
        if not changes:
            return
        log.info("MonitoringAgent: %d change(s) detected", len(changes))
        results   = self.reasoning.global_risk_analysis()
        high_risk = [r for r in results.values() if r.risk_level == "High"]
        new_alerts = []
        for change in changes:
            new_alerts.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type":      change["type"],
                "issue_id":  change["issue_id"],
                "message":   self._format_change_alert(change, results),
                "severity":  "high" if change["issue_id"] in {r.issue_id for r in high_risk} else "medium",
            })
        if high_risk:
            new_alerts.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type":      "risk_summary",
                "issue_id":  None,
                "message":   (
                    f"{len(high_risk)} task(s) are currently at HIGH risk. "
                    f"Top concern: {high_risk[0].issue_id} "
                    f"(score: {high_risk[0].risk_score:.2f}). "
                    f"Root cause: {high_risk[0].explanation[:120]}"
                ),
                "severity": "high",
            })
        with self._lock:
            self._alerts.extend(new_alerts)

    def _format_change_alert(self, change: dict, results: dict) -> str:
        issue_id = change["issue_id"]
        result   = results.get(issue_id)
        risk_tag = f"[{result.risk_level} RISK]" if result else ""
        if change["type"] == "status_change":
            return f"{risk_tag} {issue_id} changed status: {change['from_status']} → {change['to_status']}."
        if change["type"] == "delay_increased":
            return (f"{risk_tag} {issue_id} delay increased from "
                    f"{change.get('prev_delay', 0):.1f} to {change.get('cur_delay', 0):.1f} days.")
        if change["type"] == "new_issue":
            return f"{risk_tag} New issue detected: {issue_id}."
        return f"{issue_id}: {change['type']}"


# ─────────────────────────────────────────────
# Critic Agent
# ─────────────────────────────────────────────

class CriticAgent:
    """Validates LLM outputs — strips hallucinated IDs, checks confidence."""

    def validate(
        self,
        llm_output: dict,
        risk_result: Optional[RiskResult],
        known_issue_ids: set[str],
    ) -> dict:
        issues_found = []
        cleaned = dict(llm_output)

        # Check 1: hallucinated issue IDs
        text_blob = json.dumps(llm_output)
        mentioned_ids = set(re.findall(r'\b[A-Z]+-\d+\b', text_blob))
        hallucinated  = mentioned_ids - known_issue_ids
        if hallucinated:
            issues_found.append(f"Hallucinated issue IDs detected and flagged: {hallucinated}")
            if "recommendations" in cleaned:
                cleaned["recommendations"] = [
                    r for r in cleaned["recommendations"]
                    if not any(hid in r for hid in hallucinated)
                ]

        # Check 2: confidence vs risk score mismatch
        conf = llm_output.get("confidence", 0.5)
        if risk_result and risk_result.risk_score > 0.7 and conf < 0.4:
            issues_found.append("Confidence mismatch: high computed risk but LLM reported low confidence.")
            cleaned["confidence"] = 0.6

        # Check 3: empty recommendations
        if not cleaned.get("recommendations"):
            issues_found.append("No recommendations provided by LLM.")
            if risk_result:
                cleaned["recommendations"] = [
                    f"Address {iid} immediately."
                    for iid in (risk_result.affected_by or [risk_result.issue_id])[:3]
                ]

        cleaned["critique"] = {
            "passed":       len(issues_found) == 0,
            "issues_found": issues_found,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        return cleaned


# ─────────────────────────────────────────────
# AgentPipeline — orchestrates all agents
# ─────────────────────────────────────────────

class AgentPipeline:
    """
    Top-level orchestrator.
    Observe → RAG filter → Analyze → Plan → Act (LLM) → Evaluate → Return
    """

    def __init__(self, db, openai_api_key: str, poll_interval: int = 30):
        self.perception = PerceptionAgent(db)
        self.reasoning  = GraphReasoningAgent(self.perception)
        self.planning   = PlanningAgent()
        self.decision   = DecisionAgent(api_key=openai_api_key)
        self.monitoring = MonitoringAgent(self.perception, self.reasoning, poll_interval)
        self.critic     = CriticAgent()

    def start_monitoring(self):
        self.monitoring.start()

    def stop_monitoring(self):
        self.monitoring.stop()

    # ── STEP 3: Lightweight RAG — keyword-filtered context ────────────
    def get_relevant_context(
        self,
        query: str,
        nodes: list[IssueNode],
        risk_results: dict,
    ) -> list[IssueNode]:
        """
        Filters the full node list to only those relevant to the query.
        Avoids dumping the entire graph into the LLM context window.
        Combines keyword matching with risk score to pick the best nodes.
        """
        keywords = [k for k in query.lower().split() if len(k) > 2]

        scored = []
        for n in nodes:
            text = (n.issue_id + " " + n.summary).lower()
            kw_score  = sum(1 for k in keywords if k in text)
            risk_score = risk_results.get(n.issue_id, None)
            risk_val   = risk_score.risk_score if risk_score else 0.0
            is_at_risk = 1 if n.is_at_risk() else 0
            total = kw_score * 2 + risk_val + is_at_risk
            if total > 0:
                scored.append((total, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:10]]

    def run_query(self, user_query: str, project: Optional[str] = None) -> dict:
        log.info("AgentPipeline.run_query: %r", user_query)

        # Step 1 — Observe
        nodes     = self.perception.fetch_all_nodes(project)
        edges     = self.perception.fetch_all_edges()
        node_map  = {n.issue_id: n for n in nodes}
        known_ids = set(node_map.keys())

        # Step 2 — Analyse
        specific_id = self._extract_issue_id(user_query, known_ids)

        if specific_id:
            risk_results = self.reasoning.global_risk_analysis(project)
            risk_result  = risk_results.get(specific_id)
            chain        = self.reasoning.dependency_chain(specific_id)
            plan         = self.planning.create_mitigation_plan(risk_results, node_map)
            issue_plan   = [p for p in plan if p["issue_id"] == specific_id]

            if risk_result is None:
                return {
                    "issue_id":   specific_id,
                    "message":    f"{specific_id} has no computed risk — it may be Done or not at risk.",
                    "risk_level": "None",
                    "risk_score": 0.0,
                }

            # Step 3 — Act with structured issue-specific prompt (Step 2 of plan)
            llm_output = self.decision.explain_issue_risk(
                specific_id, risk_result, chain, issue_plan
            )
            validated = self.critic.validate(llm_output, risk_result, known_ids)

            return {
                "issue_id":         specific_id,
                "risk_score":       risk_result.risk_score,
                "risk_level":       risk_result.risk_level,
                "is_origin":        risk_result.is_origin,
                "delay_days":       risk_result.delay_days,
                "affected_by":      risk_result.affected_by,
                "dependency_chain": risk_result.chain,
                "explanation":      risk_result.explanation,
                "llm_analysis":     validated,
            }

        else:
            # General query — use RAG + query-routing
            risk_results = self.reasoning.global_risk_analysis(project)
            top_risks    = self.reasoning.top_risky_issues(k=10, project=project)
            plan         = self.planning.create_mitigation_plan(
                {r.issue_id: r for r in top_risks}, node_map
            )

            # STEP 3: RAG filter — only send relevant context to LLM
            relevant_nodes = self.get_relevant_context(user_query, nodes, risk_results)
            log.info("RAG: filtered to %d relevant nodes from %d total", len(relevant_nodes), len(nodes))

            # Route to appropriate focused prompt based on query intent
            query_lower = user_query.lower()

            if any(w in query_lower for w in ["block", "blocked", "blocking"]):
                focus     = "Focus on blocked issues and what is blocking them. Trace the blocking chain."
                subset    = [r for r in top_risks if r.status == "Blocked"][:6] or top_risks[:5]
            elif any(w in query_lower for w in ["delay", "overdue", "late", "behind", "miss"]):
                focus     = "Focus on overdue issues and their delay severity. Prioritise by days overdue."
                subset    = [r for r in top_risks if (r.delay_days or 0) > 0][:6] or top_risks[:5]
            elif any(w in query_lower for w in ["root", "cause", "origin", "source", "why"]):
                focus     = "Focus on root cause issues (is_root_cause=True) causing cascading risk."
                subset    = [r for r in top_risks if r.is_origin][:6] or top_risks[:5]
            elif any(w in query_lower for w in ["fix", "solve", "mitigate", "action", "recommend", "should"]):
                focus     = "Focus on actionable recommendations. Rank by downstream impact."
                subset    = top_risks[:6]
            elif any(w in query_lower for w in ["summary", "overview", "status", "state", "report"]):
                focus     = "Give an executive summary of the current project risk state across all issues."
                subset    = top_risks[:8]
            elif any(w in query_lower for w in ["high", "critical", "urgent", "most"]):
                focus     = "Focus on the highest risk issues and explain why they are critical."
                subset    = [r for r in top_risks if r.risk_level == "High"][:6] or top_risks[:5]
            else:
                focus     = f'Answer this specific question: "{user_query}". Use the dependency graph data to give a precise answer.'
                # Use RAG-filtered nodes to build focused subset
                rag_ids   = {n.issue_id for n in relevant_nodes}
                subset    = [r for r in top_risks if r.issue_id in rag_ids] or top_risks[:6]

            llm_output = self.decision.summarise_project_risks_focused(subset, plan, focus)
            validated  = self.critic.validate(llm_output, None, known_ids)

            return {
                "top_risks": [
                    {
                        "issue_id":    r.issue_id,
                        "risk_level":  r.risk_level,
                        "risk_score":  r.risk_score,
                        "is_origin":   r.is_origin,
                        "explanation": r.explanation,
                    }
                    for r in top_risks
                ],
                "action_plan":      plan,
                "llm_analysis":     validated,
                "rag_context_size": len(relevant_nodes),
            }

    def get_alerts(self) -> list[dict]:
        return self.monitoring.get_alerts(clear=True)

    @staticmethod
    def _extract_issue_id(query: str, known_ids: set[str]) -> Optional[str]:
        # Exact Jira-style match: HADOOP-42
        matches = re.findall(r'\b([A-Z]+-\d+)\b', query)
        for m in matches:
            if m in known_ids:
                return m
        # Case-insensitive full match
        query_upper = query.upper()
        for iid in known_ids:
            if iid in query_upper:
                return iid
        # Natural language: "issue 8", "task 42", "ticket 16"
        num_match = re.search(r'\b(?:issue|task|ticket|item|bug)\s+(\d+)\b', query, re.IGNORECASE)
        if num_match:
            num = num_match.group(1)
            for iid in known_ids:
                if iid.endswith(f"-{num}"):
                    return iid
        return None
