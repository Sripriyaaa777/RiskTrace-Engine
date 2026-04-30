"""
agentic_pipeline.py — Real Agentic AI with LangGraph
=====================================================
Replaces AgentPipeline.run_query() with a true agentic loop.

Architecture:
  perception_node  →  rag_retrieval_node  →  planning_node
                                                    ↓
                              [conditional: needs_issue_detail?]
                                   ↙ yes              ↘ no
                           issue_detail_node      decision_node
                                    ↘                  ↙
                                      critic_node
                                          ↓
                           [conditional: critic passes?]
                               ↙ pass        ↘ fail (max 2 retries)
                              END          planning_node  (retry loop)

The LLM (Groq llama-3.3-70b-versatile) decides at the planning node:
  - which tool to call next
  - whether it needs more detail on a specific issue
  - whether the current context is sufficient to answer

Key difference from old AgentPipeline:
  OLD: fixed Python call sequence — always does Perception→Plan→Decision
  NEW: LangGraph StateGraph with conditional edges — the LLM drives the path

All existing agents (PerceptionAgent, GraphReasoningAgent, etc.) are
REUSED as tool-providers, not replaced. The LangGraph wraps them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Literal, Optional, TypedDict

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Shared State — typed dict passed between nodes
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The shared state object that flows through every node in the graph.
    Each node reads from it and returns a partial update dict.
    """
    # Input
    user_query:        str
    project:           Optional[str]

    # Perception outputs
    nodes:             list
    edges:             list
    node_map:          dict
    known_ids:         set

    # RAG outputs
    rag_results:       list[dict]
    rag_context_str:   str

    # Risk analysis outputs
    risk_results:      dict
    top_risks:         list
    specific_issue_id: Optional[str]
    issue_risk_result: Any
    dependency_chain:  list

    # Planning outputs
    action_plan:       list[dict]
    llm_routing:       dict    # LLM's routing decision

    # Decision outputs
    llm_raw_output:    dict

    # Critic outputs
    final_output:      dict
    critic_passed:     bool
    retry_count:       int

    # Tracing
    execution_path:    list[str]   # records which nodes fired in order


# ─────────────────────────────────────────────
# RiskTraceAgenticPipeline — the LangGraph graph
# ─────────────────────────────────────────────

class RiskTraceAgenticPipeline:
    """
    Drop-in replacement for AgentPipeline that uses a real LangGraph StateGraph.

    Usage (identical interface to old AgentPipeline):
        pipeline = RiskTraceAgenticPipeline(db=db, groq_api_key=key)
        pipeline.start_monitoring()
        result = pipeline.run_query("Why is HADOOP-14 high risk?")
        alerts  = pipeline.get_alerts()
        pipeline.stop_monitoring()
    """

    def __init__(
        self,
        db,
        openai_api_key: str,          # kept same param name for drop-in compat
        poll_interval: int = 30,
        rag_engine=None,
    ):
        # ── Reuse existing agents as tool-providers ──
        from agents import (
            PerceptionAgent, GraphReasoningAgent, PlanningAgent,
            DecisionAgent, MonitoringAgent, CriticAgent,
        )
        self.perception = PerceptionAgent(db)
        self.reasoning  = GraphReasoningAgent(self.perception)
        self.planning   = PlanningAgent()
        self.decision   = DecisionAgent(api_key=openai_api_key)
        self.monitoring = MonitoringAgent(self.perception, self.reasoning, poll_interval)
        self.critic_agent = CriticAgent()
        # Backward-compatibility for older dashboard/API code paths that still
        # expect the classic AgentPipeline attribute name.
        self.critic = self.critic_agent

        # ── RAG engine ──
        self.rag = rag_engine  # injected from main.py startup

        # ── Build the LangGraph graph ──
        self._graph = self._build_graph()

    # ─────────────────────────────────────────
    # Public interface (same as AgentPipeline)
    # ─────────────────────────────────────────

    def start_monitoring(self):
        self.monitoring.start()

    def stop_monitoring(self):
        self.monitoring.stop()

    def get_alerts(self) -> list[dict]:
        return self.monitoring.get_alerts(clear=True)

    def run_query(self, user_query: str, project: Optional[str] = None) -> dict:
        """
        Entry point — identical signature to old AgentPipeline.run_query().
        Internally runs the LangGraph execution loop.
        """
        log.info("RiskTraceAgenticPipeline: running query %r", user_query)

        initial_state: AgentState = {
            "user_query":        user_query,
            "project":           project,
            "nodes":             [],
            "edges":             [],
            "node_map":          {},
            "known_ids":         set(),
            "rag_results":       [],
            "rag_context_str":   "",
            "risk_results":      {},
            "top_risks":         [],
            "specific_issue_id": None,
            "issue_risk_result": None,
            "dependency_chain":  [],
            "action_plan":       [],
            "llm_routing":       {},
            "llm_raw_output":    {},
            "final_output":      {},
            "critic_passed":     False,
            "retry_count":       0,
            "execution_path":    [],
        }

        try:
            final_state = self._graph.invoke(initial_state)
            result = final_state.get("final_output", {})
            # Attach execution path for transparency / demo
            result["_agent_execution_path"] = final_state.get("execution_path", [])
            result["_rag_context_size"] = len(final_state.get("rag_results", []))
            result["_rag_used"] = bool(final_state.get("rag_results"))
            return result
        except Exception as e:
            log.exception("Agentic pipeline failed")
            return {
                "error": str(e),
                "message": "Agentic pipeline encountered an error.",
                "_agent_execution_path": ["error"],
            }

    # ─────────────────────────────────────────
    # LangGraph graph construction
    # ─────────────────────────────────────────

    def _build_graph(self):
        """
        Constructs and compiles the LangGraph StateGraph.
        Each node is a method on this class.
        Conditional edges give the graph its branching logic.
        """
        from langgraph.graph import StateGraph, END

        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("perception",     self._perception_node)
        builder.add_node("rag_retrieval",  self._rag_retrieval_node)
        builder.add_node("planning",       self._planning_node)
        builder.add_node("issue_detail",   self._issue_detail_node)
        builder.add_node("decision",       self._decision_node)
        builder.add_node("critic",         self._critic_node)

        # Entry point
        builder.set_entry_point("perception")

        # Fixed edges
        builder.add_edge("perception", "rag_retrieval")
        builder.add_edge("rag_retrieval", "planning")

        # Conditional edge after planning: does the LLM need issue detail?
        builder.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {
                "issue_detail": "issue_detail",
                "decision":     "decision",
            },
        )

        # issue_detail always proceeds to decision
        builder.add_edge("issue_detail", "decision")

        # decision always goes to critic
        builder.add_edge("decision", "critic")

        # Conditional edge after critic: pass or retry
        builder.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "end":     END,
                "retry":   "planning",
            },
        )

        return builder.compile()

    # ─────────────────────────────────────────
    # Node implementations
    # ─────────────────────────────────────────

    def _perception_node(self, state: AgentState) -> dict:
        """
        Node 1: Fetches graph data from the DB.
        Reuses PerceptionAgent + GraphReasoningAgent (unchanged code).
        """
        log.info("[node:perception] fetching graph data")
        project = state["project"]

        nodes        = self.perception.fetch_all_nodes(project)
        edges        = self.perception.fetch_all_edges()
        node_map     = {n.issue_id: n for n in nodes}
        risk_results = self.reasoning.global_risk_analysis(project)
        top_risks    = self.reasoning.top_risky_issues(k=10, project=project)

        # Extract specific issue ID from query if present
        specific_id = self._extract_issue_id(state["user_query"], set(node_map.keys()))

        return {
            "nodes":             nodes,
            "edges":             edges,
            "node_map":          node_map,
            "known_ids":         set(node_map.keys()),
            "risk_results":      risk_results,
            "top_risks":         top_risks,
            "specific_issue_id": specific_id,
            "execution_path":    state["execution_path"] + ["perception"],
        }

    def _rag_retrieval_node(self, state: AgentState) -> dict:
        """
        Node 2: Retrieves relevant historical context from ChromaDB.
        If RAG is not ready, returns empty context (graceful degradation).

        This is what makes the LLM answers GROUNDED — it retrieves
        actual past issues similar to the current query and injects
        them into the LLM context window.
        """
        log.info("[node:rag_retrieval] retrieving context")

        rag_results    = []
        rag_context_str = ""

        if self.rag and self.rag.is_ready:
            # Build a rich retrieval query combining user query + risk data
            query = state["user_query"]
            specific_id = state.get("specific_issue_id")
            if specific_id:
                query = f"{query} {specific_id}"

            rag_results = self.rag.retrieve(query, k=5)
            rag_context_str = self.rag.format_context_for_llm(rag_results)
            log.info("[node:rag_retrieval] retrieved %d documents", len(rag_results))
        else:
            log.info("[node:rag_retrieval] RAG not ready, skipping")

        return {
            "rag_results":     rag_results,
            "rag_context_str": rag_context_str,
            "execution_path":  state["execution_path"] + ["rag_retrieval"],
        }

    def _planning_node(self, state: AgentState) -> dict:
        """
        Node 3: Runs PlanningAgent + asks the LLM to decide routing.

        THIS IS THE KEY AGENTIC NODE.
        The LLM receives the current state and decides:
          - Does it need more detail on a specific issue? → issue_detail
          - Or can it answer directly? → decision
          - What focus should the answer have?

        This is different from the old code where routing was hard-coded
        keyword matching in Python. Here the LLM reasons about it.
        """
        log.info("[node:planning] building plan + LLM routing decision")

        risk_results = state["risk_results"]
        node_map     = state["node_map"]
        top_risks    = state["top_risks"]
        risk_map     = {r.issue_id: r for r in top_risks}

        # Run PlanningAgent (reused unchanged)
        action_plan = self.planning.create_mitigation_plan(risk_map, node_map)

        # Ask the LLM to decide how to route
        routing = self._llm_routing_decision(state, action_plan)

        return {
            "action_plan":    action_plan,
            "llm_routing":    routing,
            "execution_path": state["execution_path"] + ["planning"],
        }

    def _issue_detail_node(self, state: AgentState) -> dict:
        """
        Node 4 (conditional): Fetches deep detail for a specific issue.
        Only executed if the planning node's LLM decided it was needed.
        """
        issue_id = state.get("specific_issue_id") or state["llm_routing"].get("target_issue")
        log.info("[node:issue_detail] fetching detail for %s", issue_id)

        chain = []
        risk_result = None

        if issue_id:
            chain = self.reasoning.dependency_chain(issue_id)
            risk_result = state["risk_results"].get(issue_id)

        return {
            "dependency_chain":  chain,
            "issue_risk_result": risk_result,
            "execution_path":    state["execution_path"] + ["issue_detail"],
        }

    def _decision_node(self, state: AgentState) -> dict:
        """
        Node 5: Calls the LLM (Groq) with:
          - Structured risk data (from PerceptionAgent)
          - RAG-retrieved historical context (from ChromaDB)
          - Routing focus (from planning LLM decision)
          - Issue detail if available

        This is where the RAG context gets injected into the prompt.
        The LLM sees: "here is what you asked about" + "here is relevant history"
        and produces grounded, auditable explanations.
        """
        log.info("[node:decision] calling LLM with RAG context")

        routing      = state["llm_routing"]
        rag_context  = state["rag_context_str"]
        specific_id  = state.get("specific_issue_id")
        risk_result  = state.get("issue_risk_result")
        chain        = state.get("dependency_chain", [])
        action_plan  = state["action_plan"]
        known_ids    = state["known_ids"]
        top_risks    = state["top_risks"]

        # Inject RAG context into the decision agent's prompts
        # We temporarily patch the system prompt to include RAG context
        original_system = self.decision.SYSTEM_PROMPT
        if rag_context and rag_context != "No relevant historical context found.":
            self.decision.SYSTEM_PROMPT = (
                original_system
                + f"\n\n{rag_context}\n\n"
                + "Use the above retrieved historical context to enrich your analysis. "
                + "Cite specific historical issue IDs from the context when relevant."
            )

        try:
            if specific_id and risk_result:
                issue_plan = [p for p in action_plan if p["issue_id"] == specific_id]
                llm_output = self.decision.explain_issue_risk(
                    specific_id, risk_result, chain, issue_plan
                )
            else:
                focus = routing.get("focus", "Give a comprehensive project risk analysis.")
                llm_output = self.decision.summarise_project_risks_focused(
                    top_risks[:8], action_plan[:5], focus
                )
        finally:
            # Always restore original system prompt
            self.decision.SYSTEM_PROMPT = original_system

        return {
            "llm_raw_output": llm_output,
            "execution_path": state["execution_path"] + ["decision"],
        }

    def _critic_node(self, state: AgentState) -> dict:
        """
        Node 6: Validates the LLM output.
        If it fails, sets critic_passed=False so the conditional edge
        routes back to planning_node for a retry (max 2 retries).

        This is a real feedback loop — the old code just applied the
        critic and returned regardless of the result.
        """
        log.info("[node:critic] validating LLM output")

        llm_output  = state["llm_raw_output"]
        known_ids   = state["known_ids"]
        risk_result = state.get("issue_risk_result")
        specific_id = state.get("specific_issue_id")
        top_risks   = state["top_risks"]
        action_plan = state["action_plan"]
        rag_results = state["rag_results"]

        validated = self.critic_agent.validate(llm_output, risk_result, known_ids)
        passed = validated.get("critique", {}).get("passed", True)

        # Build the final structured response
        if specific_id and risk_result:
            final_output = {
                "issue_id":         specific_id,
                "risk_score":       risk_result.risk_score,
                "risk_level":       risk_result.risk_level,
                "is_origin":        risk_result.is_origin,
                "delay_days":       risk_result.delay_days,
                "affected_by":      risk_result.affected_by,
                "dependency_chain": risk_result.chain,
                "explanation":      risk_result.explanation,
                "llm_analysis":     validated,
                "rag_sources":      [
                    {"issue_id": r["metadata"].get("issue_id"),
                     "relevance": r["relevance_score"],
                     "snippet": r["text"][:150]}
                    for r in rag_results if r["metadata"].get("doc_type") == "issue"
                ],
            }
        else:
            final_output = {
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
                "action_plan":  action_plan,
                "llm_analysis": validated,
                "rag_sources":  [
                    {"issue_id": r["metadata"].get("issue_id"),
                     "relevance": r["relevance_score"],
                     "snippet": r["text"][:150]}
                    for r in rag_results if r["metadata"].get("doc_type") == "issue"
                ],
                "rag_context_size": len(rag_results),
            }

        return {
            "final_output":   final_output,
            "critic_passed":  passed,
            "retry_count":    state["retry_count"] + (0 if passed else 1),
            "execution_path": state["execution_path"] + ["critic"],
        }

    # ─────────────────────────────────────────
    # Conditional edge routers
    # ─────────────────────────────────────────

    def _route_after_planning(
        self, state: AgentState
    ) -> Literal["issue_detail", "decision"]:
        """
        Conditional edge after planning_node.
        Routes to issue_detail if the LLM decided a specific issue needs deep analysis.
        Otherwise routes directly to decision.
        """
        routing = state.get("llm_routing", {})
        needs_detail = routing.get("needs_issue_detail", False)
        target = routing.get("target_issue")
        specific = state.get("specific_issue_id")

        if needs_detail and (target or specific):
            log.info("[router] → issue_detail for %s", target or specific)
            return "issue_detail"
        log.info("[router] → decision (no detail needed)")
        return "decision"

    def _route_after_critic(
        self, state: AgentState
    ) -> Literal["end", "retry"]:
        """
        Conditional edge after critic_node.
        If the critic passed, or we've retried twice, end.
        Otherwise loop back to planning for a retry.
        """
        passed      = state.get("critic_passed", True)
        retry_count = state.get("retry_count", 0)

        if passed:
            log.info("[router] critic passed → END")
            return "end"
        if retry_count >= 2:
            log.warning("[router] critic failed but max retries reached → END")
            return "end"
        log.info("[router] critic failed (retry %d/2) → retry planning", retry_count)
        return "retry"

    # ─────────────────────────────────────────
    # LLM routing decision
    # ─────────────────────────────────────────

    def _llm_routing_decision(self, state: AgentState, action_plan: list[dict]) -> dict:
        """
        Calls the LLM at the planning stage to decide:
          1. Does this query need deep detail on a specific issue?
          2. What is the focus/intent of the answer?
          3. Which issue (if any) to focus on?

        Returns a routing dict used by the conditional edge.
        This is the core of "real" agency — the LLM controls execution flow.
        """
        query       = state["user_query"]
        specific_id = state.get("specific_issue_id")
        top_risks   = state["top_risks"]
        retry_count = state.get("retry_count", 0)

        risk_summary = "\n".join([
            f"  - {r.issue_id}: score={r.risk_score:.2f} level={r.risk_level} "
            f"is_root={r.is_origin} status={r.status}"
            for r in top_risks[:6]
        ])

        plan_summary = "\n".join([
            f"  - {p['issue_id']}: action={p['action']} downstream={p['downstream_count']}"
            for p in action_plan[:4]
        ])

        retry_instruction = ""
        if retry_count > 0:
            retry_instruction = (
                f"\nNote: this is retry attempt {retry_count}. "
                "The previous answer was rejected by validation. "
                "Please adjust your focus and provide a more precise answer."
            )

        routing_prompt = f"""You are a routing agent for a project risk analysis system.

User query: "{query}"
Specific issue mentioned: {specific_id or "none"}

Available top-risk issues:
{risk_summary}

Recommended action plan:
{plan_summary}
{retry_instruction}

Decide the following and respond ONLY with a JSON object:
{{
  "needs_issue_detail": true/false,   // true if query is about a specific issue and needs deep analysis
  "target_issue": "ISSUE-ID or null", // the specific issue to analyze in depth (null if general query)
  "focus": "one sentence describing what the answer should focus on",
  "intent": "one of: specific_issue | blocked_issues | delay_analysis | root_cause | recommendations | summary | general"
}}

Rules:
- Set needs_issue_detail=true only if the user is asking about ONE specific issue
- If the query is general (project health, top risks, recommendations), set needs_issue_detail=false
- focus should directly address the user's question
- Do not include any text outside the JSON object"""

        try:
            routing = self.decision._call_llm(routing_prompt)
            # Validate the routing response has required keys
            if not isinstance(routing, dict):
                raise ValueError("routing response is not a dict")
            routing.setdefault("needs_issue_detail", bool(specific_id))
            routing.setdefault("target_issue", specific_id)
            routing.setdefault("focus", f"Answer the user's question: {query}")
            routing.setdefault("intent", "general")
            log.info("[routing] LLM decided: intent=%s needs_detail=%s",
                     routing["intent"], routing["needs_issue_detail"])
            return routing
        except Exception as e:
            log.warning("LLM routing failed, using defaults: %s", e)
            return {
                "needs_issue_detail": bool(specific_id),
                "target_issue": specific_id,
                "focus": f"Answer: {query}",
                "intent": "general",
            }

    # ─────────────────────────────────────────
    # Utility (same as old AgentPipeline)
    # ─────────────────────────────────────────

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
        num_match = re.search(
            r'\b(?:issue|task|ticket|item|bug)\s+(\d+)\b', query, re.IGNORECASE
        )
        if num_match:
            num = num_match.group(1)
            for iid in known_ids:
                if iid.endswith(f"-{num}"):
                    return iid
        return None
