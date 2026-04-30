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
    cd project/
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

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
_rag = None
_db_mode = "uninitialised"
_rag_status = {
    "ready": False,
    "initializing": False,
    "last_error": None,
    "dataset": None,
}
_active_dataset = {
    "dataset_id": None,
    "dataset_name": "Dataset",
    "project": None,
    "issues_path": os.getenv("ISSUES_CSV", "data/processed/issues.csv"),
    "deps_path": os.getenv("DEPS_CSV", "data/processed/dependencies.csv"),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db, _pipeline, _rag, _db_mode
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    try:
        _refresh_pipeline()
        log.info("RiskTrace runtime ready  [mode: %s]", _db_mode)
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


class SliceActivationRequest(BaseModel):
    project_key: str
    max_issues: int = 1500
    include_subtasks: bool = False
    augment_soft_deps: bool = True
    activate: bool = True
    use_neo4j: bool = True


def _build_active_dataset_label(dataset_id: Optional[str], original_name: str, project_key: Optional[str]) -> str:
    if project_key:
        return f"{original_name} [{project_key}]"
    return original_name


def _set_runtime_state() -> None:
    app.state.pipeline = _pipeline
    app.state.rag = _rag
    app.state.rag_status = dict(_rag_status)


def _current_dataset_signature() -> dict:
    return {
        "issues_csv": os.getenv("ISSUES_CSV", "data/processed/issues.csv"),
        "deps_csv": os.getenv("DEPS_CSV", "data/processed/dependencies.csv"),
        "db_mode": _db_mode,
    }


def _start_rag_background_build(force_rebuild: bool = False) -> None:
    global _rag, _rag_status

    if _rag_status.get("initializing"):
        return

    dataset_signature = _current_dataset_signature()
    _rag_status = {
        "ready": False,
        "initializing": True,
        "last_error": None,
        "dataset": dataset_signature,
    }
    _set_runtime_state()

    def _worker():
        global _rag, _rag_status
        try:
            from rag_engine import RagEngine

            rag = RagEngine()
            built = rag.build_or_load(
                issues_csv=dataset_signature["issues_csv"],
                deps_csv=dataset_signature["deps_csv"],
                force_rebuild=force_rebuild,
            )
            if built and rag.is_ready:
                _rag = rag
                if _pipeline is not None and hasattr(_pipeline, "rag"):
                    _pipeline.rag = rag
                _rag_status = {
                    "ready": True,
                    "initializing": False,
                    "last_error": None,
                    "dataset": dataset_signature,
                }
                log.info("RAG engine attached in background: %s", rag.get_stats())
            else:
                _rag = None
                if _pipeline is not None and hasattr(_pipeline, "rag"):
                    _pipeline.rag = None
                _rag_status = {
                    "ready": False,
                    "initializing": False,
                    "last_error": "RAG build returned not ready",
                    "dataset": dataset_signature,
                }
        except Exception as e:
            _rag = None
            if _pipeline is not None and hasattr(_pipeline, "rag"):
                _pipeline.rag = None
            _rag_status = {
                "ready": False,
                "initializing": False,
                "last_error": str(e),
                "dataset": dataset_signature,
            }
            log.warning("Background RAG initialisation failed: %s", e)
        finally:
            _set_runtime_state()

    threading.Thread(target=_worker, daemon=True, name="risktrace-rag-init").start()


def _refresh_pipeline() -> None:
    global _db, _pipeline, _rag, _db_mode, _rag_status
    if _pipeline:
        _pipeline.stop_monitoring()
    if _db:
        _db.close()

    _rag = None
    _rag_status = {
        "ready": False,
        "initializing": False,
        "last_error": None,
        "dataset": None,
    }
    _db, _db_mode = create_db()

    api_key = os.getenv("GROQ_API_KEY", "")

    try:
        from agentic_pipeline import RiskTraceAgenticPipeline

        _pipeline = RiskTraceAgenticPipeline(
            db=_db,
            openai_api_key=api_key,
            poll_interval=int(os.getenv("MONITOR_INTERVAL", "60")),
            rag_engine=_rag,
        )
        log.info("Agentic pipeline enabled")
    except Exception as e:
        log.warning("Agentic pipeline unavailable, falling back to classic pipeline: %s", e)
        from agents import AgentPipeline

        _pipeline = AgentPipeline(
            db=_db,
            openai_api_key=api_key,
            poll_interval=int(os.getenv("MONITOR_INTERVAL", "60")),
        )

    _pipeline.start_monitoring()
    _set_runtime_state()
    _start_rag_background_build(force_rebuild=False)


def _activate_slice(issues_csv: str, deps_csv: str, *, use_neo4j: bool, dataset_id: Optional[str], dataset_name: str, project_key: Optional[str]) -> dict:
    desired_mode = "neo4j" if use_neo4j else "csv"
    if (
        _active_dataset.get("dataset_id") == dataset_id
        and _active_dataset.get("project") == project_key
        and _active_dataset.get("issues_path") == issues_csv
        and _active_dataset.get("deps_path") == deps_csv
        and _db_mode == desired_mode
        and _pipeline is not None
    ):
        return {
            "status": "ok",
            "db_mode": _db_mode,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "project": project_key,
            "issues_path": issues_csv,
            "deps_path": deps_csv,
            "reused_runtime": True,
        }

    os.environ["ISSUES_CSV"] = issues_csv
    os.environ["DEPS_CSV"] = deps_csv
    os.environ["USE_NEO4J"] = "true" if use_neo4j else "false"

    if use_neo4j:
        from build_graph import GraphDB, rebuild_graph
        db = GraphDB(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "issuegraph123"),
        )
        try:
            rebuild_graph(db, issues_csv=issues_csv, deps_csv=deps_csv)
        finally:
            db.close()

    _refresh_pipeline()
    _active_dataset.update({
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "project": project_key,
        "issues_path": issues_csv,
        "deps_path": deps_csv,
    })
    return {
        "status": "ok",
        "db_mode": _db_mode,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "project": project_key,
        "issues_path": issues_csv,
        "deps_path": deps_csv,
    }


def require_pipeline():
    # Try app.state first (survives uvicorn reload), fall back to global
    pipeline = getattr(app.state, "pipeline", None) or _pipeline
    if pipeline is None:
        raise HTTPException(
            503,
            "Pipeline not ready yet. Start the backend and wait for RAG / graph initialisation."
        )
    return pipeline


@app.get("/health", tags=["system"])
def health():
    issues_path = Path(os.getenv("ISSUES_CSV", "data/processed/issues.csv"))
    deps_path = Path(os.getenv("DEPS_CSV", "data/processed/dependencies.csv"))
    pipeline = getattr(app.state, "pipeline", None) or _pipeline
    return {
        "status":      "ok" if pipeline else "degraded",
        "db_mode":     _db_mode,
        "pipeline":    pipeline is not None,
        "pipeline_type": type(pipeline).__name__ if pipeline else None,
        "rag_ready": bool((getattr(app.state, "rag", None) or _rag) and (getattr(app.state, "rag", None) or _rag).is_ready),
        "data_loaded": issues_path.exists() and deps_path.exists(),
        "issues_path": str(issues_path),
        "deps_path": str(deps_path),
        "active_dataset": dict(_active_dataset),
    }


@app.get("/rag-status", tags=["system"])
def rag_status():
    """Shows RAG knowledge base status — proves ChromaDB is live."""
    # Always read from app.state — survives uvicorn --reload process boundary
    rag = getattr(app.state, "rag", None) or _rag
    if rag is not None:
        stats = rag.get_stats()
        stats["is_ready"] = rag.is_ready
        stats["initializing"] = False
        return stats
    status = dict(getattr(app.state, "rag_status", _rag_status))
    status.setdefault("ready", False)
    return status


@app.get("/agent-trace", tags=["agents"])
def agent_trace(query: str = Query("What are the top blocked issues and why?")):
    """Returns LangGraph execution path — demonstrates real agentic routing."""
    pipeline = require_pipeline()
    try:
        result = pipeline.run_query(query)
        return {
            "query":              query,
            "execution_path":     result.get("_agent_execution_path", []),
            "rag_used":           result.get("_rag_used", False),
            "rag_docs_retrieved": result.get("_rag_context_size", 0),
            "answer_summary":     (result.get("llm_analysis") or {}).get("summary", ""),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/datasets", tags=["datasets"])
def list_available_datasets():
    from dataset_manager import list_datasets

    return {
        "active_dataset": dict(_active_dataset),
        "datasets": list_datasets(),
    }


@app.get("/datasets/{dataset_id}", tags=["datasets"])
def get_dataset_details(dataset_id: str):
    from dataset_manager import get_dataset

    try:
        dataset = get_dataset(dataset_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return dataset


@app.post("/datasets/upload", tags=["datasets"])
async def upload_dataset(file: UploadFile = File(...)):
    from dataset_manager import save_uploaded_dataset

    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        temp_path = Path(tmp.name)
    try:
        dataset = save_uploaded_dataset(temp_path, file.filename or temp_path.name)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        log.exception("Dataset upload failed")
        raise HTTPException(500, str(e))
    return dataset


@app.post("/datasets/{dataset_id}/activate", tags=["datasets"])
def prepare_and_activate_dataset_slice(dataset_id: str, req: SliceActivationRequest):
    from dataset_manager import get_dataset, prepare_slice

    try:
        dataset = get_dataset(dataset_id)
        slice_info = prepare_slice(
            dataset_id=dataset_id,
            project_key=req.project_key,
            max_issues=req.max_issues,
            include_subtasks=req.include_subtasks,
            augment_soft_deps=req.augment_soft_deps,
        )
        activation = None
        if req.activate:
            activation = _activate_slice(
                slice_info["issues_csv"],
                slice_info["deps_csv"],
                use_neo4j=req.use_neo4j,
                dataset_id=dataset_id,
                dataset_name=_build_active_dataset_label(dataset_id, dataset["original_name"], req.project_key.upper()),
                project_key=req.project_key.upper(),
            )
        return {
            "dataset_id": dataset_id,
            "project": req.project_key.upper(),
            "slice": slice_info,
            "activation": activation,
        }
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        log.exception("Slice preparation failed")
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
    """Dependency chain as nodes + edges for frontend visualisation."""
    pipeline = require_pipeline()
    try:
        chain_nodes = pipeline.reasoning.dependency_chain(issue_id)
        root_node   = pipeline.perception.fetch_node(issue_id)
        if root_node is None:
            raise HTTPException(404, f"Issue {issue_id} not found")
        all_nodes   = ([root_node] if root_node else []) + chain_nodes
        seen_ids: set[str] = set()
        unique_nodes = []
        risk_results = pipeline.reasoning.global_risk_analysis()
        for n in all_nodes:
            if n and n.issue_id not in seen_ids:
                seen_ids.add(n.issue_id)
                risk = risk_results.get(n.issue_id)
                downstream = pipeline.reasoning.downstream_impact(n.issue_id)
                unique_nodes.append({
                    "id":         n.issue_id,
                    "label":      n.issue_id,
                    "summary":    (n.summary or "")[:80],
                    "status":     n.status,
                    "priority":   n.priority,
                    "is_delayed": n.is_delayed,
                    "delay_days": n.delay_days,
                    "risk_level": risk.risk_level if risk else "None",
                    "risk_score": risk.risk_score if risk else 0.0,
                    "downstream_impact_count": len(downstream),
                    "highlight_high_risk": bool(risk and risk.risk_level == "High"),
                })
        edge_records = pipeline.perception.db.run(
            "MATCH (src:Issue)-[r:DEPENDS_ON]->(tgt:Issue) "
            "WHERE src.issue_id IN $ids AND tgt.issue_id IN $ids "
            "RETURN src.issue_id AS source, tgt.issue_id AS target, r.link_type AS link_type",
            {"ids": list(seen_ids)},
        )
        edges = [{"source": r["source"], "target": r["target"],
                  "link_type": r.get("link_type", "depends_on")} for r in edge_records]
        return {"issue_id": issue_id, "nodes": unique_nodes, "edges": edges}
    except HTTPException:
        raise
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
        effective_project = project or _active_dataset.get("project")
        top_risks = pipeline.reasoning.top_risky_issues(k=10, project=effective_project)
        nodes     = pipeline.perception.fetch_all_nodes(effective_project)
        node_map  = {n.issue_id: n for n in nodes}
        risk_map  = {r.issue_id: r for r in top_risks}
        plan      = pipeline.planning.create_mitigation_plan(risk_map, node_map)
        known_ids = set(node_map.keys())

        if os.getenv("GROQ_API_KEY", ""):
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

        return {
            "db_mode": _db_mode,
            "active_dataset": dict(_active_dataset),
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


@app.get("/predictive-analysis", tags=["predictive"])
def predictive_analysis(
    project: Optional[str] = Query(None),
    threshold: float = Query(0.35, ge=0.0, le=1.0),
    positive_target: int = Query(20, ge=1, le=30),
    total_target: int = Query(30, ge=1, le=60),
    use_trained_model: bool = Query(True),
):
    """Run predictive validation using only information available before each issue's delay window."""
    try:
        from predictive_analysis import run_predictive_experiment

        payload = run_predictive_experiment(
            issues_path=os.getenv("ISSUES_CSV", "data/processed/issues.csv"),
            deps_path=os.getenv("DEPS_CSV", "data/processed/dependencies.csv"),
            project=project or _active_dataset.get("project"),
            threshold=threshold,
            positive_target=positive_target,
            total_target=total_target,
            output_path="data/processed/predictive_analysis.json",
            model_path="data/models/predictive_model.joblib",
            use_trained_model=use_trained_model,
        )
        return payload
    except Exception as e:
        log.exception("Predictive analysis failed")
        raise HTTPException(500, str(e))


@app.post("/train-predictive-model", tags=["predictive"])
def train_predictive_model(
    project: Optional[str] = Query(None),
    folds: int = Query(5, ge=2, le=10),
    text_encoder_model: str = Query("distilroberta-base"),
):
    """Train the predictive model using historical Jira snapshots and evaluate with CV + temporal holdout."""
    try:
        from predictive_model import train_predictive_model as train_model

        payload = train_model(
            issues_path=os.getenv("ISSUES_CSV", "data/processed/issues.csv"),
            deps_path=os.getenv("DEPS_CSV", "data/processed/dependencies.csv"),
            project=project or _active_dataset.get("project"),
            model_path="data/models/predictive_model.joblib",
            n_splits=folds,
            text_encoder_model=text_encoder_model,
        )
        return payload
    except Exception as e:
        log.exception("Predictive model training failed")
        raise HTTPException(500, str(e))


@app.get("/predictive-model-info", tags=["predictive"])
def predictive_model_info():
    """Expose the currently installed predictive model artifact for the UI."""
    try:
        from predictive_model import load_trained_model

        artifact = load_trained_model("data/models/predictive_model.joblib")
        return {
            "available": True,
            "model_kind": artifact.get("model_kind", "unknown"),
            "text_encoder_model": artifact.get("text_encoder_model", "unknown"),
            "trained_at": artifact.get("trained_at"),
            "project": artifact.get("project"),
            "active_dataset": dict(_active_dataset),
        }
    except FileNotFoundError:
        return {"available": False}
    except Exception as e:
        log.exception("Predictive model info failed")
        raise HTTPException(500, str(e))
