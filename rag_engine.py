"""
rag_engine.py — Local RAG Knowledge Base for RiskTrace
======================================================
Builds a fully local TF-IDF retrieval index from the Jira issues corpus.
This avoids any model downloads or external embedding runtime at startup.

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
import pickle
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

INDEX_DIR = Path("data/rag_index")
MANIFEST_NAME = "rag_manifest.json"
DOCS_NAME = "docs.pkl"
MATRIX_NAME = "matrix.pkl"
VECTORIZER_NAME = "vectorizer.pkl"


class RagEngine:
    """
    Fully local lexical-semantic retrieval using TF-IDF + cosine similarity.

    It is much lighter than the Chroma + ONNX path:
      - no network download
      - fast warm startup
      - persistent cached index on disk
    """

    def __init__(self, persist_dir: str | Path = INDEX_DIR):
        self.persist_dir = Path(persist_dir)
        self._is_ready = False
        self._docs: list[dict] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    def build_or_load(
        self,
        issues_csv: str = "data/processed/issues.csv",
        deps_csv: str = "data/processed/dependencies.csv",
        force_rebuild: bool = False,
    ) -> bool:
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

            desired_manifest = self._build_manifest(issues_csv, deps_csv)
            existing_manifest = self._load_manifest()

            if not force_rebuild and existing_manifest == desired_manifest and self._artifacts_exist():
                self._load_index()
                self._is_ready = True
                log.info("RAG: loaded cached local TF-IDF index (%d docs)", len(self._docs))
                return True

            issues = self._load_issues_csv(issues_csv)
            deps = self._load_deps_csv(deps_csv)
            dep_map: dict[str, list[str]] = {}
            for src, tgt in deps:
                dep_map.setdefault(src, []).append(tgt)
                dep_map.setdefault(tgt, []).append(src)

            docs: list[dict] = []
            for issue in issues:
                text, meta = self._build_document(issue, dep_map)
                docs.append({"text": text, "metadata": meta})

            for i, (text, meta) in enumerate(self._build_domain_patterns()):
                docs.append({"text": text, "metadata": meta, "id": f"pattern_{i:04d}"})

            corpus = [doc["text"] for doc in docs]
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=12000,
            )
            matrix = vectorizer.fit_transform(corpus)

            self._docs = docs
            self._vectorizer = vectorizer
            self._matrix = matrix
            self._save_index(desired_manifest)
            self._is_ready = True
            log.info("RAG: built local TF-IDF index (%d docs)", len(self._docs))
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
        if not self._is_ready or self._vectorizer is None or self._matrix is None:
            return []

        try:
            query_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._matrix).ravel()
            ranked_indices = scores.argsort()[::-1]

            retrieved: list[dict] = []
            for idx in ranked_indices:
                score = float(scores[idx])
                if score <= 0:
                    continue
                doc = self._docs[idx]
                meta = dict(doc["metadata"])
                if where and any(meta.get(key) != value for key, value in where.items()):
                    continue
                retrieved.append(
                    {
                        "text": doc["text"],
                        "metadata": meta,
                        "distance": round(1.0 - score, 4),
                        "relevance_score": round(score, 3),
                    }
                )
                if len(retrieved) >= k:
                    break
            return retrieved
        except Exception as e:
            log.error("RAG retrieval failed: %s", e)
            return []

    def format_context_for_llm(self, retrieved: list[dict], max_chars: int = 2000) -> str:
        if not retrieved:
            return "No relevant historical context found."
        lines = ["=== Retrieved Historical Context ==="]
        total = 0
        for i, r in enumerate(retrieved, 1):
            meta = r["metadata"]
            issue_id = meta.get("issue_id", "unknown")
            status = meta.get("status", "")
            level = meta.get("risk_level", "")
            rel = r["relevance_score"]
            header = f"[{i}] {issue_id} | status={status} | risk={level} | relevance={rel:.2f}"
            text = r["text"][:300]
            entry = f"{header}\n{text}\n"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)
        return "\n".join(lines)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_stats(self) -> dict:
        return {
            "ready": self._is_ready,
            "count": len(self._docs),
            "persist_dir": str(self.persist_dir),
            "backend": "local_tfidf",
        }

    def _artifacts_exist(self) -> bool:
        return all(
            (self.persist_dir / name).exists()
            for name in (MANIFEST_NAME, DOCS_NAME, MATRIX_NAME, VECTORIZER_NAME)
        )

    def _build_manifest(self, issues_csv: str, deps_csv: str) -> dict:
        def stat_info(path_str: str) -> dict:
            path = Path(path_str)
            if not path.exists():
                return {"path": str(path), "exists": False}
            stat = path.stat()
            payload = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
            return {
                "path": str(path),
                "exists": True,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "fingerprint": hashlib.sha256(payload.encode()).hexdigest(),
            }

        return {
            "issues_csv": stat_info(issues_csv),
            "deps_csv": stat_info(deps_csv),
        }

    def _load_manifest(self) -> Optional[dict]:
        path = self.persist_dir / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _save_index(self, manifest: dict) -> None:
        (self.persist_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        with open(self.persist_dir / DOCS_NAME, "wb") as f:
            pickle.dump(self._docs, f)
        with open(self.persist_dir / VECTORIZER_NAME, "wb") as f:
            pickle.dump(self._vectorizer, f)
        with open(self.persist_dir / MATRIX_NAME, "wb") as f:
            pickle.dump(self._matrix, f)

    def _load_index(self) -> None:
        with open(self.persist_dir / DOCS_NAME, "rb") as f:
            self._docs = pickle.load(f)
        with open(self.persist_dir / VECTORIZER_NAME, "rb") as f:
            self._vectorizer = pickle.load(f)
        with open(self.persist_dir / MATRIX_NAME, "rb") as f:
            self._matrix = pickle.load(f)

    def _build_document(self, issue: dict, dep_map: dict[str, list[str]]) -> tuple[str, dict]:
        issue_id = issue.get("issue_id", "")
        summary = issue.get("summary", "")
        description = issue.get("description", "")
        status = issue.get("status", "")
        priority = issue.get("priority", "")
        assignee = issue.get("assignee", "")
        delay_days = issue.get("delay_days", "")
        is_delayed = issue.get("is_delayed", "False")
        project = issue.get("project", "")
        deps = dep_map.get(issue_id, [])

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

        try:
            d = float(delay_days) if delay_days else 0
            risk_level = "High" if d > 14 else ("Medium" if d > 3 else "Low")
        except Exception:
            risk_level = "Unknown"

        meta = {
            "issue_id": issue_id,
            "project": project,
            "status": status,
            "priority": priority,
            "is_delayed": str(is_delayed),
            "risk_level": risk_level,
            "doc_type": "issue",
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
        patterns = [
            (
                "Critical path issue: when a high-priority issue is blocked by multiple upstream dependencies that are themselves overdue, the risk cascades exponentially. Resolution strategy: identify the deepest overdue blocker in the dependency chain and escalate it first. Resolving root blockers frees the entire downstream chain.",
                {"doc_type": "pattern", "pattern_type": "cascade_risk"},
            ),
            (
                "Blocked issue pattern: an issue with status Blocked means its assignee cannot proceed until at least one upstream dependency is resolved. Priority: unblock the dependency, not the blocked issue itself.",
                {"doc_type": "pattern", "pattern_type": "blocked_issue"},
            ),
            (
                "Delay propagation in OSS projects: a 1-day delay in a critical-path issue can amplify downstream delay. This system uses dependency-aware propagation with decay over hops.",
                {"doc_type": "pattern", "pattern_type": "delay_propagation"},
            ),
            (
                "What-if resolution strategy: simulating the resolution of a root cause issue helps identify whether it reduces downstream project risk or only affects the issue locally.",
                {"doc_type": "pattern", "pattern_type": "counterfactual"},
            ),
        ]
        return [
            (
                text,
                {
                    **meta,
                    "issue_id": f"pattern_{i}",
                    "project": "domain_knowledge",
                    "status": "reference",
                    "priority": "N/A",
                    "is_delayed": "False",
                    "risk_level": "N/A",
                },
            )
            for i, (text, meta) in enumerate(patterns)
        ]
