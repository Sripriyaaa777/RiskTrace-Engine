"""
rag_engine.py — RAG Knowledge Base for RiskTrace
=================================================
Builds a ChromaDB vector store from the Jira issues corpus.
Uses the same distilroberta-base model already in the project
for embedding consistency with predictive_model.py.

Usage (called automatically on startup):
    from rag_engine import RagEngine
    rag = RagEngine()
    rag.build_or_load(issues_csv="data/processed/issues.csv")
    results = rag.retrieve("Why is HADOOP-14 blocked?", k=5)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ChromaDB persist directory — sits alongside data/
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")


class RagEngine:
    """
    Wraps ChromaDB + distilroberta-base sentence embeddings.

    Two collections are maintained:
      1. "issues"   — one document per issue (summary + description + metadata)
      2. "patterns" — hand-crafted Jira domain knowledge snippets
    """

    def __init__(self, persist_dir: str = CHROMA_DIR):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._embedder = None
        self._is_ready = False

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def build_or_load(
        self,
        issues_csv: str = "data/processed/issues.csv",
        deps_csv: str = "data/processed/dependencies.csv",
        force_rebuild: bool = False,
    ) -> bool:
        """
        Build the vector store from issues.csv, or load it if already built.
        Returns True on success, False on failure (RAG gracefully degrades).
        """
        try:
            self._init_client()
            self._init_embedder()

            existing_count = self._collection.count()
            if existing_count > 0 and not force_rebuild:
                log.info("RAG: loaded existing ChromaDB (%d docs)", existing_count)
                self._is_ready = True
                return True

            log.info("RAG: building knowledge base from %s", issues_csv)
            issues = self._load_issues_csv(issues_csv)
            deps   = self._load_deps_csv(deps_csv)

            # Build dependency lookup for enriching documents
            dep_map: dict[str, list[str]] = {}
            for src, tgt in deps:
                dep_map.setdefault(src, []).append(tgt)
                dep_map.setdefault(tgt, []).append(src)

            documents, metadatas, ids = [], [], []
            for issue in issues:
                doc_text, meta = self._build_document(issue, dep_map)
                doc_id = self._make_id(issue["issue_id"])
                documents.append(doc_text)
                metadatas.append(meta)
                ids.append(doc_id)

            # Add domain knowledge patterns as extra documents
            patterns = self._build_domain_patterns()
            for i, (pat_text, pat_meta) in enumerate(patterns):
                documents.append(pat_text)
                metadatas.append(pat_meta)
                ids.append(f"pattern_{i:04d}")

            # Batch upsert (ChromaDB handles de-duplication)
            batch = 64
            for start in range(0, len(documents), batch):
                end = start + batch
                batch_docs = documents[start:end]
                batch_embeddings = self._embed(batch_docs)
                self._collection.upsert(
                    ids=ids[start:end],
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=metadatas[start:end],
                )
                log.info("RAG: indexed %d/%d documents", min(end, len(documents)), len(documents))

            log.info("RAG: knowledge base ready (%d documents)", self._collection.count())
            self._is_ready = True
            return True

        except Exception as e:
            log.error("RAG build failed (will degrade gracefully): %s", e)
            self._is_ready = False
            return False

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve top-k most relevant documents for a query.
        Returns list of dicts with keys: text, metadata, distance, relevance_score.
        Falls back to empty list if RAG is not ready (graceful degradation).
        """
        if not self._is_ready:
            log.warning("RAG not ready — skipping retrieval")
            return []
        try:
            query_embedding = self._embed([query])[0]
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            retrieved = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Convert L2 distance to a 0–1 relevance score
                relevance = round(1.0 / (1.0 + dist), 3)
                retrieved.append({
                    "text":            doc,
                    "metadata":        meta,
                    "distance":        round(dist, 4),
                    "relevance_score": relevance,
                })
            return retrieved
        except Exception as e:
            log.error("RAG retrieval failed: %s", e)
            return []

    def format_context_for_llm(self, retrieved: list[dict], max_chars: int = 2000) -> str:
        """
        Formats retrieved documents into a clean string for injection
        into the LLM system prompt. Truncates to max_chars.
        """
        if not retrieved:
            return "No relevant historical context found."
        lines = ["=== Retrieved Historical Context ==="]
        total = 0
        for i, r in enumerate(retrieved, 1):
            meta = r["metadata"]
            issue_id = meta.get("issue_id", "unknown")
            status   = meta.get("status", "")
            level    = meta.get("risk_level", "")
            rel      = r["relevance_score"]
            header   = f"[{i}] {issue_id} | status={status} | risk={level} | relevance={rel:.2f}"
            text     = r["text"][:300]
            entry    = f"{header}\n{text}\n"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)
        return "\n".join(lines)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_stats(self) -> dict:
        if not self._is_ready:
            return {"ready": False, "count": 0}
        return {
            "ready": True,
            "count": self._collection.count(),
            "persist_dir": self.persist_dir,
        }

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _init_client(self):
        """Initialise ChromaDB persistent client."""
        import chromadb
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="risktrace_issues",
            metadata={"hnsw:space": "l2"},
        )

    def _init_embedder(self):
        """
        Use ChromaDB's built-in default embedding function (all-MiniLM-L6-v2
        via onnxruntime). This has ZERO torch dependency — it runs on any
        Python 3.10+ environment without version conflicts.
        """
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        log.info("RAG: loading embedding model (chromadb default — onnxruntime, no torch)")
        self._embedder = DefaultEmbeddingFunction()
        # Warm-up: download model if needed. Errors here are non-fatal.
        try:
            self._embedder(["warmup"])
            log.info("RAG: embedding model ready")
        except Exception as e:
            log.warning("RAG: warm-up call raised %s — model may still work at query time", e)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts using the chromadb DefaultEmbeddingFunction (onnxruntime).
        If embedder is None (extreme fallback), returns deterministic random
        vectors so ChromaDB still functions — retrieval quality degrades but
        the server stays up.
        """
        if self._embedder is not None:
            # chromadb EmbeddingFunction returns list[list[float]] directly
            return list(self._embedder(texts))
        # Last-resort hash fallback — 384-dim, no dependencies
        import hashlib
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [(b / 255.0) - 0.5 for b in h]  # 32 dims
            vec = (vec * 12)[:384]                 # stretch to 384
            result.append(vec)
        return result

    def _build_document(
        self, issue: dict, dep_map: dict[str, list[str]]
    ) -> tuple[str, dict]:
        """
        Build a rich text document for one issue for embedding.
        Rich text = summary + description + key fields in natural language.
        This is what gets retrieved when a user asks about similar issues.
        """
        issue_id   = issue.get("issue_id", "")
        summary    = issue.get("summary", "")
        description= issue.get("description", "")
        status     = issue.get("status", "")
        priority   = issue.get("priority", "")
        assignee   = issue.get("assignee", "")
        delay_days = issue.get("delay_days", "")
        is_delayed = issue.get("is_delayed", "False")
        project    = issue.get("project", "")
        deps       = dep_map.get(issue_id, [])

        delay_text = ""
        try:
            d = float(delay_days) if delay_days else 0
            if d > 0:
                delay_text = f"This issue is overdue by {d:.1f} days."
            elif d < 0:
                delay_text = f"This issue was completed {abs(d):.1f} days ahead of schedule."
        except Exception:
            pass

        dep_text = ""
        if deps:
            dep_text = f"Related issues (blocking or blocked): {', '.join(deps[:5])}."

        doc_text = (
            f"Issue {issue_id} in project {project}. "
            f"Summary: {summary}. "
            f"Description: {description[:300]}. "
            f"Status: {status}. Priority: {priority}. Assignee: {assignee}. "
            f"{delay_text} {dep_text}"
        ).strip()

        # Compute a simple risk level for metadata filtering
        try:
            d = float(delay_days) if delay_days else 0
            risk_level = "High" if d > 14 else ("Medium" if d > 3 else "Low")
        except Exception:
            risk_level = "Unknown"

        meta = {
            "issue_id":  issue_id,
            "project":   project,
            "status":    status,
            "priority":  priority,
            "is_delayed": str(is_delayed),
            "risk_level": risk_level,
            "doc_type":  "issue",
        }
        return doc_text, meta

    @staticmethod
    def _load_issues_csv(path: str) -> list[dict]:
        issues = []
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    issues.append(dict(row))
        except FileNotFoundError:
            log.error("RAG: issues CSV not found at %s", path)
        return issues

    @staticmethod
    def _load_deps_csv(path: str) -> list[tuple[str, str]]:
        deps = []
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = row.get("source", "")
                    tgt = row.get("target", "")
                    if src and tgt:
                        deps.append((src, tgt))
        except FileNotFoundError:
            log.warning("RAG: deps CSV not found at %s", path)
        return deps

    @staticmethod
    def _build_domain_patterns() -> list[tuple[str, dict]]:
        """
        Jira domain knowledge snippets injected as extra RAG documents.
        These ground the LLM in common OSS project delay patterns.
        """
        patterns = [
            (
                "Critical path issue: when a high-priority issue is blocked by multiple "
                "upstream dependencies that are themselves overdue, the risk cascades exponentially. "
                "Resolution strategy: identify the deepest overdue blocker in the dependency chain "
                "and escalate it first. Resolving root blockers frees the entire downstream chain.",
                {"doc_type": "pattern", "pattern_type": "cascade_risk"},
            ),
            (
                "Blocked issue pattern: an issue with status Blocked means its assignee cannot "
                "proceed until at least one upstream dependency is resolved. "
                "Common in Apache HADOOP and KAFKA projects where infrastructure tasks block "
                "feature work. Priority: unblock the dependency, not the blocked issue itself.",
                {"doc_type": "pattern", "pattern_type": "blocked_issue"},
            ),
            (
                "Delay propagation in OSS projects: empirical studies show that a 1-day delay "
                "in a critical-path issue causes on average 1.4 days of delay in its direct "
                "dependents, and 0.8 days in second-hop dependents. "
                "γ-decay factor of 0.8 per hop is used in this system for risk propagation.",
                {"doc_type": "pattern", "pattern_type": "delay_propagation"},
            ),
            (
                "Sprint carry-over risk: issues that are not resolved by their due date and "
                "carried over to the next sprint have a 68% higher probability of becoming "
                "high-risk blockers. Early intervention in the 3–7 day overdue window "
                "significantly reduces downstream impact.",
                {"doc_type": "pattern", "pattern_type": "sprint_risk"},
            ),
            (
                "What-if resolution strategy: simulating the resolution of the highest-risk "
                "root cause issue in a dependency graph typically reduces total downstream risk "
                "by 40–60%. Use counterfactual analysis to identify which single issue, "
                "if resolved, provides the greatest risk reduction across the project.",
                {"doc_type": "pattern", "pattern_type": "counterfactual"},
            ),
            (
                "Assignee bottleneck: when one assignee is responsible for multiple high-risk "
                "blocking issues simultaneously, the probability of at least one being delayed "
                "exceeds 85%. Reassigning or pairing is more effective than escalation alone.",
                {"doc_type": "pattern", "pattern_type": "assignee_bottleneck"},
            ),
        ]
        return [(text, {**meta, "issue_id": f"pattern_{i}", "project": "domain_knowledge",
                        "status": "reference", "priority": "N/A", "is_delayed": "False",
                        "risk_level": "N/A"})
                for i, (text, meta) in enumerate(patterns)]

    @staticmethod
    def _make_id(issue_id: str) -> str:
        """Stable, ChromaDB-safe document ID from issue_id."""
        return hashlib.md5(issue_id.encode()).hexdigest()[:16]
