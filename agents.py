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
    Uses Groq (llama-3.3-70b) for LLM explanations and recommendations.
    Groq is free, fast (500 tokens/sec), and has 30 RPM on free tier.
    Get a free key at: https://console.groq.com

    The LLM only runs AFTER the deterministic risk engine has computed
    scores and chains. Its job is explanation, not reasoning.
    """

    SYSTEM_PROMPT = (
        "You are IssueGraphAgent++, an expert AI assistant specialised in "
        "software project risk management. You receive structured JSON data "
        "from a deterministic risk propagation engine that has already computed "
        "dependency chains and risk scores. Your job is to:\n"
        "1. Explain the risk situation clearly and concisely to a project manager.\n"
        "2. Recommend specific, actionable interventions.\n"
        "3. Prioritise the interventions by impact.\n\n"
        "Rules:\n"
        "- Be direct and specific. Name the actual issue IDs.\n"
        "- Explain dependency chains in plain English.\n"
        "- Never fabricate issue IDs, delays, or scores not in the input.\n"
        "- Keep responses under 300 words.\n"
        "- Respond ONLY with a valid JSON object — no markdown, no extra text.\n"
        "- Required JSON keys: summary (string), root_causes (list of strings), "
        "recommendations (list of strings), confidence (float 0-1)."
    )

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

    def explain_issue_risk(self, issue_id, risk_result, chain, plan):
        context = {
            "query_issue":      issue_id,
            "risk_score":       risk_result.risk_score,
            "risk_level":       risk_result.risk_level,
            "status":           risk_result.status,
            "delay_days":       risk_result.delay_days,
            "affected_by":      risk_result.affected_by,
            "dependency_chain": risk_result.chain,
            "explanation":      risk_result.explanation,
            "upstream_issues": [
                {"issue_id": n.issue_id, "status": n.status,
                 "is_delayed": n.is_delayed, "delay_days": n.delay_days}
                for n in chain
            ],
            "recommended_actions": plan[:3],
        }
        prompt = (
            f"Analyse the risk for issue {issue_id} using this data:\n\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Respond ONLY with a JSON object with keys: "
            "summary, root_causes, recommendations, confidence."
        )
        return self._call_llm(prompt)

    def summarise_project_risks(self, top_risks, plan):
        context = {
            "top_risks": [
                {"issue_id": r.issue_id, "risk_level": r.risk_level,
                 "risk_score": r.risk_score, "is_origin": r.is_origin,
                 "delay_days": r.delay_days, "affected_by": r.affected_by[:5]}
                for r in top_risks[:10]
            ],
            "action_plan": plan[:5],
        }
        prompt = (
            "Summarise the current project risk state based on this data:\n\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Respond ONLY with a JSON object with keys: "
            "summary, root_causes, recommendations, confidence."
        )
        return self._call_llm(prompt)

    def _call_llm(self, user_prompt: str) -> dict:
        """Call Groq API and return parsed JSON."""
        import traceback as _tb
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model       = self.model_name,
                messages    = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = 0.2,
                max_tokens  = 800,
                response_format = {"type": "json_object"},  # forces clean JSON
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences just in case
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
    Observe → Analyze → Plan → Act (LLM) → Evaluate → Return
    """

    def __init__(self, db, openai_api_key: str, poll_interval: int = 30):
        self.perception = PerceptionAgent(db)
        self.reasoning  = GraphReasoningAgent(self.perception)
        self.planning   = PlanningAgent()
        self.decision   = DecisionAgent(api_key=openai_api_key)  # api_key now holds Gemini key
        self.monitoring = MonitoringAgent(self.perception, self.reasoning, poll_interval)
        self.critic     = CriticAgent()

    def start_monitoring(self):
        self.monitoring.start()

    def stop_monitoring(self):
        self.monitoring.stop()

    def run_query(self, user_query: str, project: Optional[str] = None) -> dict:
        log.info("AgentPipeline.run_query: %r", user_query)

        nodes    = self.perception.fetch_all_nodes(project)
        edges    = self.perception.fetch_all_edges()
        node_map = {n.issue_id: n for n in nodes}
        known_ids = set(node_map.keys())

        specific_id = self._extract_issue_id(user_query, known_ids)

        if specific_id:
            risk_results = self.reasoning.global_risk_analysis(project)
            risk_result  = risk_results.get(specific_id)
            chain        = self.reasoning.dependency_chain(specific_id)
            plan         = self.planning.create_mitigation_plan(risk_results, node_map)
            issue_plan   = [p for p in plan if p["issue_id"] == specific_id]

            if risk_result is None:
                return {"issue_id": specific_id,
                        "message": f"{specific_id} has no computed risk — it may be Done or not at risk.",
                        "risk_level": "None", "risk_score": 0.0}

            llm_output = self.decision.explain_issue_risk(specific_id, risk_result, chain, issue_plan)
            validated  = self.critic.validate(llm_output, risk_result, known_ids)

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
            top_risks    = self.reasoning.top_risky_issues(k=10, project=project)
            risk_results = {r.issue_id: r for r in top_risks}
            plan         = self.planning.create_mitigation_plan(risk_results, node_map)
            llm_output   = self.decision.summarise_project_risks(top_risks, plan)
            validated    = self.critic.validate(llm_output, None, known_ids)

            return {
                "top_risks": [
                    {"issue_id": r.issue_id, "risk_level": r.risk_level,
                     "risk_score": r.risk_score, "is_origin": r.is_origin,
                     "explanation": r.explanation}
                    for r in top_risks
                ],
                "action_plan":  plan,
                "llm_analysis": validated,
            }

    def get_alerts(self) -> list[dict]:
        return self.monitoring.get_alerts(clear=True)

    @staticmethod
    def _extract_issue_id(query: str, known_ids: set[str]) -> Optional[str]:
        matches = re.findall(r'\b([A-Z]+-\d+)\b', query)
        for m in matches:
            if m in known_ids:
                return m
        query_upper = query.upper()
        for iid in known_ids:
            if iid in query_upper:
                return iid
        return None
