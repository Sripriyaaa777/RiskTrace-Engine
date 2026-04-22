from __future__ import annotations
"""
IssueGraphAgent++ — FastAPI Backend (with CSV fallback)
========================================================
Runs in two modes:

  Mode A — CSV mode  (default, no Neo4j needed)
    Set USE_NEO4J=false in .env (or just don't set it).
    Reads directly from data/processed/issues.csv and dependencies.csv.
    All agents, risk propagation, and LLM work identically.

  Mode B — Neo4j mode
    Set USE_NEO4J=true and NEO4J_* vars in .env.
    Full graph database with Cypher queries.

Usage:
    cd RiskTrace-Engine-main/
    uvicorn main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────
# DB factory — Neo4j or CSV fallback
# ─────────────────────────────────────────────

def create_db():
    use_neo4j = os.getenv("USE_NEO4J", "false").lower() == "true"
    if use_neo4j:
        try:
            from build_graph import GraphDB
            db = GraphDB(
                uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
                user     = os.getenv("NEO4J_USER",     "neo4j"),
                password = os.getenv("NEO4J_PASSWORD",  "issuegraph123"),
            )
            log.info("✓ Connected to Neo4j")
            return db, "neo4j"
        except Exception as e:
            log.warning("Neo4j failed (%s) — falling back to CSV mode", e)

    from csv_db import CsvGraphDB
    db = CsvGraphDB(
        issues_path = os.getenv("ISSUES_CSV", "data/processed/issues.csv"),
        deps_path   = os.getenv("DEPS_CSV",   "data/processed/dependencies.csv"),
    )
    log.info("✓ Running in CSV mode")
    return db, "csv"


_db = None
_pipeline = None
_db_mode = "uninitialised"
_edge_profile = "minimal"


def build_pipeline(db):
    from agents import AgentPipeline
    api_key = os.getenv("GROQ_API_KEY", "")
    pipeline = AgentPipeline(
        db             = db,
        openai_api_key = api_key,
        poll_interval  = int(os.getenv("MONITOR_INTERVAL", "60")),
    )
    pipeline.perception.set_edge_profile(_edge_profile)
    return pipeline


def swap_runtime(db, mode: str):
    global _db, _pipeline, _db_mode
    if _pipeline:
        _pipeline.stop_monitoring()
    if _db:
        _db.close()
    _db = db
    _db_mode = mode
    _pipeline = build_pipeline(_db)
    _pipeline.start_monitoring()
    log.info("IssueGraphAgent++ ready  [mode: %s]", _db_mode)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db, _pipeline, _db_mode
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    try:
        db, mode = create_db()
        swap_runtime(db, mode)
    except Exception as e:
        log.error("Startup failed: %s", e, exc_info=True)
    yield
    if _pipeline:
        _pipeline.stop_monitoring()
    if _db:
        _db.close()


app = FastAPI(
    title       = "IssueGraphAgent++",
    description = "Proactive risk propagation over temporal dependency graphs.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query:   str
    project: Optional[str] = None


class CounterfactualRequest(BaseModel):
    resolve_as: Optional[str] = "Done"


class SyntheticDemoRequest(BaseModel):
    max_issues: int = 250


class EdgeProfileRequest(BaseModel):
    profile: str = "full"


def require_pipeline():
    if _pipeline is None:
        raise HTTPException(
            503,
            "Pipeline not ready. Run preprocess.py to create data/processed/*.csv first."
        )
    return _pipeline


@app.get("/health", tags=["system"])
def health():
    return {
        "status":      "ok" if _pipeline else "degraded",
        "db_mode":     _db_mode,
        "edge_profile": _edge_profile,
        "pipeline":    _pipeline is not None,
        "data_loaded": Path("data/processed/issues.csv").exists(),
    }


@app.post("/graph-profile", tags=["system"])
def set_graph_profile(req: EdgeProfileRequest):
    global _edge_profile
    profile = "minimal" if req.profile.lower() == "minimal" else "full"
    _edge_profile = profile
    pipeline = require_pipeline()
    pipeline.perception.set_edge_profile(profile)
    return {
        "status": "ok",
        "edge_profile": _edge_profile,
        "message": f"Switched to {profile} dependency profile.",
    }


@app.post("/demo/load-synthetic", tags=["system"])
def load_synthetic_demo(req: SyntheticDemoRequest = SyntheticDemoRequest()):
    """Generate a synthetic demo dataset and keep Neo4j mode when configured."""
    try:
        from preprocess import run_pipeline

        issue_count = max(50, min(req.max_issues, 2000))
        run_pipeline(
            input_path=None,
            output_dir=Path("data/processed"),
            max_issues=issue_count,
            project_filter=None,
            synthetic=True,
        )

        current_mode = (_db_mode or "").lower()
        use_neo4j = (
            os.getenv("USE_NEO4J", "false").lower() == "true"
            or current_mode.startswith("neo4j")
        )
        if use_neo4j:
            from build_graph import GraphDB, clear_db, ensure_schema, load_issues, load_dependencies
            db = GraphDB(
                uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
                user     = os.getenv("NEO4J_USER",     "neo4j"),
                password = os.getenv("NEO4J_PASSWORD", "issuegraph123"),
            )
            clear_db(db)
            ensure_schema(db)
            load_issues(db)
            load_dependencies(db)
            swap_runtime(db, "neo4j-demo")
            return {
                "status": "ok",
                "message": f"Synthetic demo dataset loaded into Neo4j with {issue_count} issues.",
                "db_mode": _db_mode,
            }

        from csv_db import CsvGraphDB
        db = CsvGraphDB(
            issues_path=os.getenv("ISSUES_CSV", "data/processed/issues.csv"),
            deps_path=os.getenv("DEPS_CSV", "data/processed/dependencies.csv"),
        )
        swap_runtime(db, "csv-demo")
        return {
            "status": "ok",
            "message": f"Synthetic demo dataset loaded in CSV mode with {issue_count} issues.",
            "db_mode": _db_mode,
        }
    except Exception as e:
        log.exception("Synthetic demo load failed")
        raise HTTPException(500, str(e))


@app.post("/query", tags=["agents"])
def query_endpoint(req: QueryRequest):
    """Natural language query — ask anything about your project."""
    pipeline = require_pipeline()
    try:
        result = pipeline.run_query(req.query, project=req.project)
        return {"status": "ok", "data": result}
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(500, str(e))


@app.get("/risk/{issue_id}", tags=["risk"])
def get_risk(issue_id: str):
    """Risk score + explanation for one issue."""
    pipeline = require_pipeline()
    try:
        risk = pipeline.reasoning.issue_risk(issue_id)
        if risk is None:
            return {"issue_id": issue_id, "risk_level": "None", "risk_score": 0.0,
                    "message": f"{issue_id} not found or not at risk."}
        return {
            "issue_id":         risk.issue_id,
            "summary":          risk.summary,
            "status":           risk.status,
            "risk_score":       risk.risk_score,
            "risk_level":       risk.risk_level,
            "is_origin":        risk.is_origin,
            "delay_days":       risk.delay_days,
            "affected_by":      risk.affected_by,
            "dependency_chain": risk.chain,
            "explanation":      risk.explanation,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/graph/{issue_id}", tags=["risk"])
def get_graph(issue_id: str):
    """Return a focused local neighborhood around an issue for visualisation."""
    pipeline = require_pipeline()
    try:
        risk_results = pipeline.reasoning.global_risk_analysis()
        nodes = pipeline.perception.fetch_all_nodes_including_done()
        node_map = {n.issue_id: n for n in nodes}
        edge_records = list(pipeline.perception.db.run(
            "MATCH (src:Issue)-[r:DEPENDS_ON]->(tgt:Issue) "
            "RETURN src.issue_id AS source, tgt.issue_id AS target, "
            "coalesce(r.link_type, 'depends_on') AS link_type, "
            "coalesce(r.edge_source, 'explicit_link') AS edge_source, "
            "coalesce(r.confidence, 1.0) AS confidence"
        ))

        adjacency: dict[str, set[str]] = {}
        for r in edge_records:
            adjacency.setdefault(r["source"], set()).add(r["target"])
            adjacency.setdefault(r["target"], set()).add(r["source"])

        seen_ids = {issue_id}
        frontier = {issue_id}
        for _ in range(3):
            next_frontier = set()
            for current in frontier:
                next_frontier.update(adjacency.get(current, set()))
            next_frontier -= seen_ids
            seen_ids |= next_frontier
            frontier = next_frontier

        if len(seen_ids) < 14:
            scored_neighbors = sorted(
                (
                    node_id for node_id in adjacency.keys()
                    if node_id not in seen_ids and adjacency.get(node_id)
                ),
                key=lambda node_id: (
                    risk_results.get(node_id).risk_score if risk_results.get(node_id) else 0.0,
                    len(adjacency.get(node_id, set())),
                ),
                reverse=True,
            )
            for node_id in scored_neighbors[: max(0, 18 - len(seen_ids))]:
                seen_ids.add(node_id)

        unique_nodes = []
        for node_id in seen_ids:
            n = node_map.get(node_id)
            if not n:
                continue
            risk = risk_results.get(node_id)
            unique_nodes.append({
                "id":         n.issue_id,
                "label":      n.issue_id,
                "summary":    n.summary[:80],
                "status":     n.status,
                "priority":   n.priority,
                "is_delayed": n.is_delayed,
                "delay_days": n.delay_days,
                "risk_level": risk.risk_level if risk else "None",
                "risk_score": risk.risk_score if risk else 0.0,
            })

        edges = [
            {
                "source": r["source"],
                "target": r["target"],
                "link_type": r.get("link_type", "depends_on"),
                "edge_source": r.get("edge_source", "explicit_link"),
                "confidence": float(r.get("confidence", 1.0) or 1.0),
            }
            for r in edge_records
            if r["source"] in seen_ids and r["target"] in seen_ids
        ]
        return {"issue_id": issue_id, "nodes": unique_nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/alerts", tags=["monitoring"])
def get_alerts():
    """Proactive alerts from the Monitoring Agent. Poll every 30–60s."""
    pipeline = require_pipeline()
    try:
        alerts = pipeline.get_alerts()
        return {"count": len(alerts), "alerts": alerts}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/dashboard", tags=["risk"])
def get_dashboard(project: Optional[str] = Query(None)):
    """Full dashboard: summary cards + top risks + action plan + LLM analysis."""
    pipeline = require_pipeline()
    try:
        top_risks = pipeline.reasoning.top_risky_issues(k=10, project=project)
        nodes     = pipeline.perception.fetch_all_nodes(project)
        node_map  = {n.issue_id: n for n in nodes}
        risk_map  = {r.issue_id: r for r in top_risks}
        plan      = pipeline.planning.create_mitigation_plan(risk_map, node_map)
        known_ids = set(node_map.keys())

        if os.getenv("GROQ_API_KEY", "").strip():
            llm_out   = pipeline.decision.summarise_project_risks(top_risks, plan)
            validated = pipeline.critic.validate(llm_out, None, known_ids)
        else:
            validated = {
                "summary":         "Add GROQ_API_KEY to .env to enable LLM explanations.",
                "root_causes":     [r.issue_id for r in top_risks[:3] if r.is_origin],
                "recommendations": [p["rationale"] for p in plan[:3]],
                "confidence":      0.0,
                "critique":        {"passed": True, "issues_found": [], "validated_at": ""},
            }

        statuses    = [n.status for n in nodes]
        high_risk   = [r for r in top_risks if r.risk_level == "High"]
        medium_risk = [r for r in top_risks if r.risk_level == "Medium"]
        whatif_candidates = pipeline.reasoning.whatif_candidates(k=8, project=project)

        return {
            "db_mode": _db_mode,
            "edge_profile": _edge_profile,
            "summary": {
                "total_open":    len(nodes),
                "high_risk":     len(high_risk),
                "medium_risk":   len(medium_risk),
                "status_counts": {s: statuses.count(s) for s in set(statuses)},
            },
            "top_risks": [
                {
                    "issue_id":    r.issue_id,
                    "summary":     r.summary[:100],
                    "risk_level":  r.risk_level,
                    "risk_score":  r.risk_score,
                    "is_origin":   r.is_origin,
                    "delay_days":  r.delay_days,
                    "explanation": r.explanation[:200],
                    "chain":       r.chain,
                }
                for r in top_risks
            ],
            "action_plan":  plan[:5],
            "whatif_candidates": whatif_candidates,
            "llm_analysis": validated,
        }
    except Exception as e:
        log.exception("Dashboard failed")
        raise HTTPException(500, str(e))


@app.post("/counterfactual/{issue_id}", tags=["counterfactual"])
def counterfactual(issue_id: str, req: CounterfactualRequest = CounterfactualRequest()):
    """
    Counterfactual reasoning: simulate resolving an issue and show
    before/after risk scores for all downstream nodes.
    No DB writes — pure in-memory clone + propagation.
    """
    pipeline = require_pipeline()
    try:
        result = pipeline.reasoning.counterfactual_resolve(
            issue_id   = issue_id.upper(),
            resolve_as = req.resolve_as or "Done",
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        log.exception("Counterfactual analysis failed")
        raise HTTPException(500, str(e))
