from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from risk_engine import build_reverse_adjacency

log = logging.getLogger(__name__)

DEFAULT_TRANSFORMER_MODEL = "distilroberta-base"


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def actual_delay_days(issue: dict) -> float:
    raw = issue.get("delay_days")
    try:
        if raw not in ("", None):
            return max(float(raw), 0.0)
    except ValueError:
        pass

    due_date = parse_datetime(issue.get("due_date"))
    resolved = parse_datetime(issue.get("resolved")) or parse_datetime(issue.get("updated"))
    if due_date and resolved and resolved > due_date:
        return round((resolved - due_date).total_seconds() / 86400.0, 2)
    return 0.0


def actually_delayed(issue: dict) -> bool:
    return parse_bool(issue.get("is_delayed")) or actual_delay_days(issue) > 0


def prediction_time(issue: dict) -> Optional[datetime]:
    created = parse_datetime(issue.get("created"))
    due_date = parse_datetime(issue.get("due_date"))
    resolved = parse_datetime(issue.get("resolved"))
    updated = parse_datetime(issue.get("updated"))

    if due_date:
        t = due_date - timedelta(seconds=1)
    elif resolved:
        t = resolved - timedelta(days=1)
    elif updated:
        t = updated - timedelta(days=1)
    else:
        return None

    if created and t < created:
        return created
    return t


def load_processed_rows(
    issues_path: str = "data/processed/issues.csv",
    deps_path: str = "data/processed/dependencies.csv",
    project: Optional[str] = None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    with open(issues_path, newline="", encoding="utf-8") as f:
        issues = [dict(row) for row in csv.DictReader(f)]
    if project:
        issues = [row for row in issues if row.get("project", "").upper() == project.upper()]
    valid_ids = {row["issue_id"] for row in issues}
    with open(deps_path, newline="", encoding="utf-8") as f:
        edges = [
            (row["source"], row["target"])
            for row in csv.DictReader(f)
            if row["source"] in valid_ids and row["target"] in valid_ids
        ]
    return issues, edges


def _build_adjacency(edges: list[tuple[str, str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    forward: dict[str, list[str]] = {}
    reverse = build_reverse_adjacency(edges)
    for src, tgt in edges:
        forward.setdefault(src, []).append(tgt)
    return forward, reverse


def _status_one_hot(status: str) -> dict[str, float]:
    return {
        "status_open": 1.0 if status == "Open" else 0.0,
        "status_in_progress": 1.0 if status == "In Progress" else 0.0,
        "status_blocked": 1.0 if status == "Blocked" else 0.0,
        "status_done": 1.0 if status == "Done" else 0.0,
    }


def _priority_score(priority: str) -> float:
    return {"Critical": 4.0, "High": 3.0, "Medium": 2.0, "Low": 1.0, "Blocker": 5.0, "Major": 3.0}.get(priority, 2.0)


def build_snapshot_feature_row(
    issue: dict,
    issues_by_id: dict[str, dict],
    forward: dict[str, list[str]],
    reverse: dict[str, list[str]],
    as_of: datetime,
) -> Optional[dict]:
    created = parse_datetime(issue.get("created"))
    if created and created > as_of:
        return None

    updated = parse_datetime(issue.get("updated"))
    resolved = parse_datetime(issue.get("resolved"))
    due_date = parse_datetime(issue.get("due_date"))
    status = issue.get("status", "Open") or "Open"
    if resolved and resolved <= as_of:
        status = "Done"
    elif status == "Done":
        status = "In Progress"

    upstream_ids = [nid for nid in forward.get(issue["issue_id"], []) if nid in issues_by_id]
    downstream_ids = [nid for nid in reverse.get(issue["issue_id"], []) if nid in issues_by_id]

    upstream_delayed = 0
    upstream_blocked = 0
    for uid in upstream_ids:
        upstream_issue = issues_by_id[uid]
        u_due = parse_datetime(upstream_issue.get("due_date"))
        u_resolved = parse_datetime(upstream_issue.get("resolved"))
        u_status = upstream_issue.get("status", "Open") or "Open"
        if u_resolved and u_resolved <= as_of:
            u_status = "Done"
        elif u_status == "Done":
            u_status = "In Progress"
        if u_status == "Blocked":
            upstream_blocked += 1
        if u_due and (as_of - u_due).total_seconds() > 0 and u_status != "Done":
            upstream_delayed += 1

    age_days = 0.0
    if created:
        age_days = max((as_of - created).total_seconds() / 86400.0, 0.0)

    days_until_due = 0.0
    has_due_date = 0.0
    if due_date:
        has_due_date = 1.0
        days_until_due = (due_date - as_of).total_seconds() / 86400.0

    days_since_update = 0.0
    if updated:
        effective_updated = min(updated, as_of)
        days_since_update = max((as_of - effective_updated).total_seconds() / 86400.0, 0.0)

    text = " ".join(
        p for p in [issue.get("summary", ""), issue.get("description", ""), issue.get("assignee", "")]
        if p
    )

    row = {
        "issue_id": issue["issue_id"],
        "project": issue.get("project", ""),
        "as_of": as_of.isoformat(),
        "text": text,
        "priority_score": _priority_score(issue.get("priority", "Medium") or "Medium"),
        "age_days": round(age_days, 3),
        "days_until_due": round(days_until_due, 3),
        "days_since_update": round(days_since_update, 3),
        "has_due_date": has_due_date,
        "in_degree": float(len(upstream_ids)),
        "out_degree": float(len(downstream_ids)),
        "upstream_delayed_count": float(upstream_delayed),
        "upstream_blocked_count": float(upstream_blocked),
        "downstream_impact_count": float(len(downstream_ids)),
        "assignee_known": 1.0 if issue.get("assignee") else 0.0,
    }
    row.update(_status_one_hot(status))
    return row


def build_training_samples(
    issues: list[dict],
    edges: list[tuple[str, str]],
) -> list[dict]:
    issues_by_id = {row["issue_id"]: row for row in issues}
    forward, reverse = _build_adjacency(edges)
    samples = []
    for issue in issues:
        as_of = prediction_time(issue)
        if as_of is None:
            continue
        feature_row = build_snapshot_feature_row(issue, issues_by_id, forward, reverse, as_of)
        if feature_row is None:
            continue
        feature_row["target"] = 1 if actually_delayed(issue) else 0
        feature_row["actual_delay_days"] = actual_delay_days(issue)
        samples.append(feature_row)
    return samples


class TransformerTextEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_TRANSFORMER_MODEL,
        batch_size: int = 16,
        max_length: int = 128,
        cache_dir: str = "data/models/embedding_cache",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def _cache_key(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model_name}::{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.joblib"

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * input_mask_expanded).sum(dim=1)
        counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        torch = self._torch
        uncached = []
        uncached_index = []
        vectors: list[Optional[list[float]]] = [None] * len(texts)

        for idx, text in enumerate(texts):
            cache_path = self._cache_key(text)
            if cache_path.exists():
                vectors[idx] = joblib.load(cache_path)
            else:
                uncached.append(text)
                uncached_index.append(idx)

        for start in range(0, len(uncached), self.batch_size):
            batch = uncached[start:start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self._model(**encoded)
                pooled = self._mean_pool(output, encoded["attention_mask"]).cpu().numpy()
            for local_idx, vector in enumerate(pooled):
                global_idx = uncached_index[start + local_idx]
                vector_list = vector.astype("float32").tolist()
                vectors[global_idx] = vector_list
                joblib.dump(vector_list, self._cache_key(batch[local_idx]))

        return [v or [] for v in vectors]


def _structured_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            k: v for k, v in row.items()
            if k not in {"text", "issue_id", "as_of", "target", "actual_delay_days"}
        }
        for row in rows
    ]


def add_transformer_embeddings(
    rows: list[dict],
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    cache_dir: str = "data/models/embedding_cache",
) -> list[dict]:
    encoder = TransformerTextEncoder(model_name=model_name, cache_dir=cache_dir)
    embeddings = encoder.encode([row.get("text", "") for row in rows])
    enriched = []
    for row, vector in zip(rows, embeddings):
        updated = dict(row)
        for idx, value in enumerate(vector):
            updated[f"emb_{idx:03d}"] = float(value)
        enriched.append(updated)
    return enriched


def build_classifier_pipeline() -> Pipeline:
    return Pipeline([
        ("vectorize", DictVectorizer(sparse=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def _metric_dict(y_true: list[int], scores: list[float], preds: list[int]) -> dict:
    metrics = {
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 3),
        "accuracy": round(float(accuracy_score(y_true, preds)), 3),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 3),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 3)
    else:
        metrics["roc_auc"] = None
    return metrics


def train_predictive_model(
    issues_path: str = "data/processed/issues.csv",
    deps_path: str = "data/processed/dependencies.csv",
    project: Optional[str] = None,
    model_path: str = "data/models/predictive_model.joblib",
    n_splits: int = 5,
    text_encoder_model: str = DEFAULT_TRANSFORMER_MODEL,
) -> dict:
    issues, edges = load_processed_rows(issues_path, deps_path, project=project)
    base_samples = build_training_samples(issues, edges)
    if len(base_samples) < max(n_splits, 10):
        raise ValueError("Not enough training samples to run cross-validation")

    log.info("Encoding %d Jira issue texts with transformer model %s", len(base_samples), text_encoder_model)
    samples = add_transformer_embeddings(base_samples, model_name=text_encoder_model)

    X = _structured_rows(samples)
    y = [row["target"] for row in samples]
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    effective_splits = min(n_splits, positive_count or 1, negative_count or 1)
    if effective_splits < 2:
        raise ValueError("Need at least two positive and two negative samples for cross-validation")
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)

    cv_rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        train_rows = [X[i] for i in train_idx]
        test_rows = [X[i] for i in test_idx]
        model = build_classifier_pipeline()
        model.fit(train_rows, [y[i] for i in train_idx])
        scores = model.predict_proba(test_rows)[:, 1].tolist()
        preds = [1 if score >= 0.5 else 0 for score in scores]
        fold_metrics = _metric_dict([y[i] for i in test_idx], scores, preds)
        fold_metrics["fold"] = fold
        cv_rows.append(fold_metrics)

    ordered_idx = sorted(range(len(samples)), key=lambda idx: samples[idx]["as_of"])
    split_at = max(int(len(ordered_idx) * 0.8), 1)
    temporal_train_idx = ordered_idx[:split_at]
    temporal_test_idx = ordered_idx[split_at:]
    temporal_metrics = None
    if temporal_test_idx:
        temporal_model = build_classifier_pipeline()
        temporal_model.fit([X[i] for i in temporal_train_idx], [y[i] for i in temporal_train_idx])
        temporal_scores = temporal_model.predict_proba([X[i] for i in temporal_test_idx])[:, 1].tolist()
        temporal_preds = [1 if score >= 0.5 else 0 for score in temporal_scores]
        temporal_metrics = _metric_dict(
            [y[i] for i in temporal_test_idx],
            temporal_scores,
            temporal_preds,
        )

    final_model = build_classifier_pipeline()
    final_model.fit(X, y)
    artifact = {
        "model": final_model,
        "project": project,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "text_encoder_model": text_encoder_model,
        "embedding_cache_dir": "data/models/embedding_cache",
        "model_kind": "roberta_embedding_logistic_regression",
    }
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    averages = {
        key: round(sum(row[key] for row in cv_rows if row[key] is not None) / len([r for r in cv_rows if r[key] is not None]), 3)
        for key in ("precision", "recall", "accuracy", "f1")
    }
    roc_rows = [row["roc_auc"] for row in cv_rows if row["roc_auc"] is not None]
    averages["roc_auc"] = round(sum(roc_rows) / len(roc_rows), 3) if roc_rows else None

    return {
        "project": project,
        "model_path": str(output),
        "sample_count": len(samples),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "model_kind": artifact["model_kind"],
        "text_encoder_model": text_encoder_model,
        "cross_validation": {
            "requested_folds": n_splits,
            "used_folds": effective_splits,
            "folds": cv_rows,
            "average": averages,
        },
        "temporal_holdout": temporal_metrics,
    }


def load_trained_model(model_path: str = "data/models/predictive_model.joblib") -> dict:
    return joblib.load(model_path)


def predict_with_trained_model(rows: list[dict], model_path: str = "data/models/predictive_model.joblib") -> list[float]:
    artifact = load_trained_model(model_path)
    text_encoder_model = artifact.get("text_encoder_model", DEFAULT_TRANSFORMER_MODEL)
    cache_dir = artifact.get("embedding_cache_dir", "data/models/embedding_cache")
    model = artifact["model"]
    enriched_rows = add_transformer_embeddings(rows, model_name=text_encoder_model, cache_dir=cache_dir)
    X = _structured_rows(enriched_rows)
    return model.predict_proba(X)[:, 1].tolist()


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train predictive Jira risk model with RoBERTa/BERT embeddings, 5-fold CV, and temporal holdout")
    parser.add_argument("--issues", default="data/processed/issues.csv")
    parser.add_argument("--deps", default="data/processed/dependencies.csv")
    parser.add_argument("--project", default=None)
    parser.add_argument("--model-path", default="data/models/predictive_model.joblib")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--text-encoder-model", default=DEFAULT_TRANSFORMER_MODEL)
    args = parser.parse_args()

    result = train_predictive_model(
        issues_path=args.issues,
        deps_path=args.deps,
        project=args.project,
        model_path=args.model_path,
        n_splits=args.folds,
        text_encoder_model=args.text_encoder_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
