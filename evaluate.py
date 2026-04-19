"""
IssueGraphAgent++ — Phase 4: Evaluation Framework
==================================================
Implements the benchmark evaluation described in Section 9 of the paper.

Tasks evaluated:
  1. Risk identification accuracy
  2. Root cause correctness
  3. Blocking detection
  4. Delay explanation quality

Baselines compared against:
  A. Keyword-based retrieval (TF-IDF over issue summaries)
  B. Direct Cypher graph query (no risk scoring, no LLM)
  C. IssueGraphAgent++ (our system)

Metrics:
  - Precision / Recall / F1     (for binary classification tasks)
  - Mean Reciprocal Rank (MRR)  (for ranking tasks)
  - Explanation BLEU score      (proxy for explanation quality)
  - Time-to-detection (seconds) (latency)

Usage:
    python evaluation/evaluate.py \
        --benchmark evaluation/benchmark_queries.json \
        --output    evaluation/results.json
"""

import json
import logging
import time
import re
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────
# Benchmark query format
# ─────────────────────────────────────────────
# Each query in benchmark_queries.json looks like:
#
# {
#   "id":           "Q001",
#   "type":         "risk_identification",   <- one of the 4 task types
#   "query":        "Which tasks are at risk due to HADOOP-5 being delayed?",
#   "expected_ids": ["HADOOP-1", "HADOOP-3", "HADOOP-7"],   <- ground truth
#   "expected_root_cause": "HADOOP-5",                       <- optional
#   "expected_risk_level": "High"                            <- optional
# }


@dataclass
class EvalResult:
    query_id:       str
    query_type:     str
    query:          str
    system:         str            # "ours" | "keyword" | "cypher"
    predicted_ids:  list[str]
    expected_ids:   list[str]
    precision:      float
    recall:         float
    f1:             float
    mrr:            float
    root_cause_correct: Optional[bool]
    risk_level_correct: Optional[bool]
    latency_secs:   float
    explanation_bleu: Optional[float]


# ─────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────

def precision_recall_f1(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    if not expected:
        return (1.0, 1.0, 1.0) if not predicted else (0.0, 0.0, 0.0)
    if not predicted:
        return 0.0, 0.0, 0.0
    pred_set  = set(predicted)
    exp_set   = set(expected)
    tp        = len(pred_set & exp_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(exp_set) if exp_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def mean_reciprocal_rank(predicted: list[str], expected: list[str]) -> float:
    """MRR: 1/rank of the first relevant item in the predicted list."""
    for i, pid in enumerate(predicted):
        if pid in set(expected):
            return round(1.0 / (i + 1), 3)
    return 0.0


def simple_bleu(hypothesis: str, reference: str) -> float:
    """
    Simplified 1-gram BLEU for explanation quality.
    We use this as a fast proxy — a real evaluation would use human judges.
    """
    if not hypothesis or not reference:
        return 0.0
    hyp_tokens = set(re.findall(r'\b\w+\b', hypothesis.lower()))
    ref_tokens = set(re.findall(r'\b\w+\b', reference.lower()))
    if not ref_tokens:
        return 0.0
    overlap = len(hyp_tokens & ref_tokens)
    # Brevity penalty
    bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)) if len(hyp_tokens) < len(ref_tokens) else 1.0
    return round(bp * overlap / len(ref_tokens), 3)


# ─────────────────────────────────────────────
# Baseline: Keyword retrieval
# ─────────────────────────────────────────────

class KeywordBaseline:
    """
    TF-IDF keyword matching over issue summaries.
    Represents the simplest non-graph baseline.
    """

    def __init__(self, issues: list[dict]):
        self.issues = issues
        self._build_index()

    def _build_index(self):
        from collections import Counter
        import math
        self._doc_counts = []
        self._idf: dict[str, float] = {}
        N = len(self.issues)

        for issue in self.issues:
            tokens = re.findall(r'\b\w+\b', (issue.get("summary", "") + " " + issue.get("status", "")).lower())
            self._doc_counts.append(Counter(tokens))

        # IDF
        all_tokens: set[str] = set()
        for c in self._doc_counts:
            all_tokens.update(c.keys())
        for tok in all_tokens:
            df = sum(1 for c in self._doc_counts if tok in c)
            self._idf[tok] = math.log((N + 1) / (df + 1)) + 1

    def query(self, query_text: str, top_k: int = 10) -> list[str]:
        q_tokens = re.findall(r'\b\w+\b', query_text.lower())
        scores = []
        for i, (issue, counts) in enumerate(zip(self.issues, self._doc_counts)):
            score = sum(counts.get(t, 0) * self._idf.get(t, 1.0) for t in q_tokens)
            scores.append((issue["issue_id"], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:top_k] if s[1] > 0]


# ─────────────────────────────────────────────
# Baseline: Direct Cypher query
# ─────────────────────────────────────────────

class CypherBaseline:
    """
    Uses Cypher to find directly blocked/delayed tasks without risk scoring.
    Represents "graph database without AI" baseline.
    """

    def __init__(self, db):
        self.db = db

    def query(self, query_text: str, top_k: int = 10) -> list[str]:
        """
        Simple approach: find all delayed issues and their immediate dependents.
        No propagation, no scoring.
        """
        cypher = """
        MATCH (delayed:Issue)
        WHERE delayed.is_delayed = true
        OPTIONAL MATCH (dependent:Issue)-[:DEPENDS_ON]->(delayed)
        WITH collect(DISTINCT delayed.issue_id) + collect(DISTINCT dependent.issue_id) AS ids
        UNWIND ids AS id
        RETURN DISTINCT id
        LIMIT $k
        """
        records = self.db.run(cypher, {"k": top_k})
        return [r["id"] for r in records if r["id"]]


# ─────────────────────────────────────────────
# Our system adapter
# ─────────────────────────────────────────────

class OurSystemAdapter:
    """Wraps AgentPipeline for the evaluator interface."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def query(self, query_text: str, top_k: int = 10) -> tuple[list[str], dict]:
        """Returns (predicted_ids, full_result)."""
        result = self.pipeline.run_query(query_text)
        if "top_risks" in result:
            ids = [r["issue_id"] for r in result["top_risks"][:top_k]]
        elif "issue_id" in result:
            ids = [result["issue_id"]]
            # Include affected_by for chain queries
            ids += result.get("affected_by", [])
            ids = ids[:top_k]
        else:
            ids = []
        return ids, result


# ─────────────────────────────────────────────
# Benchmark generator
# ─────────────────────────────────────────────

def generate_benchmark_from_graph(db, n_queries: int = 60) -> list[dict]:
    """
    Auto-generate benchmark queries from the actual graph data.
    Used when you don't have manually curated ground truth.

    Strategy:
      - Identify delayed issues as ground truth risk origins
      - Compute their downstream dependents as expected_ids
      - Generate natural language query templates
    """
    import random
    random.seed(2024)

    # Fetch delayed issues
    delayed_query = """
    MATCH (n:Issue)
    WHERE n.is_delayed = true
    RETURN n.issue_id AS id, n.delay_days AS delay_days
    ORDER BY n.delay_days DESC
    LIMIT 20
    """
    delayed = [{"id": r["id"], "delay_days": r["delay_days"]}
               for r in db.run(delayed_query)]

    if not delayed:
        log.warning("No delayed issues found for benchmark generation.")
        return []

    # For each delayed issue, find its downstream dependents
    benchmark = []
    q_id = 1

    query_templates = {
        "risk_identification": [
            "Which tasks are at risk because {} is delayed?",
            "What will be affected if {} is not resolved?",
            "Show me all tasks blocked by {}",
        ],
        "root_cause_analysis": [
            "Why is there a delay originating from {}?",
            "What is causing the risk cascade from {}?",
            "Explain the root cause of delays related to {}",
        ],
        "blocking_detection": [
            "Is {} blocking any other tasks?",
            "What does {} block?",
            "Find all tasks that cannot proceed because of {}",
        ],
        "delay_explanation": [
            "Why is {} delayed?",
            "Explain the delay in {}",
            "What caused {} to fall behind schedule?",
        ],
    }

    for delayed_issue in delayed[:min(len(delayed), n_queries // 4)]:
        issue_id = delayed_issue["id"]

        # Find downstream dependents
        dep_query = """
        MATCH path = (dependent:Issue)-[:DEPENDS_ON*1..5]->(origin:Issue {issue_id: $id})
        WHERE origin.is_delayed = true
        RETURN DISTINCT dependent.issue_id AS dep_id
        """
        dependents = [r["dep_id"] for r in db.run(dep_query, {"id": issue_id})]
        expected_ids = [issue_id] + dependents

        for task_type, templates in query_templates.items():
            template = random.choice(templates)
            benchmark.append({
                "id":                  f"Q{q_id:03d}",
                "type":                task_type,
                "query":               template.format(issue_id),
                "expected_ids":        expected_ids,
                "expected_root_cause": issue_id,
                "expected_risk_level": "High" if (delayed_issue["delay_days"] or 0) > 7 else "Medium",
            })
            q_id += 1
            if q_id > n_queries:
                break
        if q_id > n_queries:
            break

    log.info("Generated %d benchmark queries", len(benchmark))
    return benchmark


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class Evaluator:

    def __init__(self, our_system, keyword_baseline, cypher_baseline):
        self.systems = {
            "ours":    our_system,
            "keyword": keyword_baseline,
            "cypher":  cypher_baseline,
        }

    def run(self, benchmark: list[dict]) -> list[EvalResult]:
        results = []
        for q in benchmark:
            log.info("Evaluating %s: %s", q["id"], q["query"][:60])
            for system_name, system in self.systems.items():
                t0 = time.time()
                if system_name == "ours":
                    pred_ids, full_result = system.query(q["query"])
                else:
                    pred_ids  = system.query(q["query"])
                    full_result = {}
                latency = round(time.time() - t0, 3)

                p, r, f1 = precision_recall_f1(pred_ids, q["expected_ids"])
                mrr      = mean_reciprocal_rank(pred_ids, q["expected_ids"])

                # Root cause correctness
                root_cause_correct = None
                if "expected_root_cause" in q and system_name == "ours":
                    llm_text = json.dumps(full_result.get("llm_analysis", {}))
                    root_cause_correct = q["expected_root_cause"] in llm_text

                # Risk level correctness
                risk_level_correct = None
                if "expected_risk_level" in q and system_name == "ours":
                    risk_level_correct = (
                        full_result.get("risk_level") == q["expected_risk_level"]
                        or any(
                            r.get("risk_level") == q["expected_risk_level"]
                            for r in full_result.get("top_risks", [])
                        )
                    )

                # Explanation BLEU (our system only)
                bleu = None
                if system_name == "ours":
                    explanation = (
                        full_result.get("explanation", "")
                        or json.dumps(full_result.get("llm_analysis", {}).get("summary", ""))
                    )
                    ref = q["query"] + " " + " ".join(q["expected_ids"])
                    bleu = simple_bleu(explanation, ref)

                results.append(EvalResult(
                    query_id=q["id"],
                    query_type=q["type"],
                    query=q["query"],
                    system=system_name,
                    predicted_ids=pred_ids,
                    expected_ids=q["expected_ids"],
                    precision=p,
                    recall=r,
                    f1=f1,
                    mrr=mrr,
                    root_cause_correct=root_cause_correct,
                    risk_level_correct=risk_level_correct,
                    latency_secs=latency,
                    explanation_bleu=bleu,
                ))
        return results

    @staticmethod
    def aggregate(results: list[EvalResult]) -> dict:
        """Compute aggregate metrics per system and per query type for the paper."""
        from collections import defaultdict
        aggregated: dict[str, dict] = defaultdict(lambda: defaultdict(list))

        for r in results:
            key = (r.system, r.query_type)
            aggregated[key]["precision"].append(r.precision)
            aggregated[key]["recall"].append(r.recall)
            aggregated[key]["f1"].append(r.f1)
            aggregated[key]["mrr"].append(r.mrr)
            aggregated[key]["latency"].append(r.latency_secs)
            if r.explanation_bleu is not None:
                aggregated[key]["bleu"].append(r.explanation_bleu)
            if r.root_cause_correct is not None:
                aggregated[key]["root_cause_acc"].append(int(r.root_cause_correct))

        summary = {}
        for (system, qtype), metrics in aggregated.items():
            summary[f"{system}/{qtype}"] = {
                metric: round(sum(vals) / len(vals), 3)
                for metric, vals in metrics.items()
                if vals
            }

        # Overall per system
        for system in ["ours", "keyword", "cypher"]:
            sys_results = [r for r in results if r.system == system]
            if sys_results:
                summary[f"{system}/OVERALL"] = {
                    "precision": round(sum(r.precision for r in sys_results) / len(sys_results), 3),
                    "recall":    round(sum(r.recall for r in sys_results) / len(sys_results), 3),
                    "f1":        round(sum(r.f1 for r in sys_results) / len(sys_results), 3),
                    "mrr":       round(sum(r.mrr for r in sys_results) / len(sys_results), 3),
                    "latency":   round(sum(r.latency_secs for r in sys_results) / len(sys_results), 3),
                }

        return summary


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    import argparse, os, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="IssueGraphAgent++ — Evaluation")
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Path to benchmark JSON. If omitted, auto-generates from graph.")
    parser.add_argument("--output",    type=Path, default=Path("evaluation/results.json"))
    parser.add_argument("--n-queries", type=int, default=60)
    args = parser.parse_args()

    # Connect
    from scripts.build_graph import GraphDB
    from agents.agents import AgentPipeline

    db = GraphDB(
        uri      = os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user     = os.getenv("NEO4J_USER", "neo4j"),
        password = os.getenv("NEO4J_PASSWORD", "issuegraph123"),
    )
    pipeline = AgentPipeline(db, openai_api_key=os.getenv("OPENAI_API_KEY", ""))

    # Load / generate benchmark
    if args.benchmark and args.benchmark.exists():
        with open(args.benchmark) as f:
            benchmark = json.load(f)
        log.info("Loaded %d benchmark queries from %s", len(benchmark), args.benchmark)
    else:
        log.info("Auto-generating benchmark from graph …")
        benchmark = generate_benchmark_from_graph(db, n_queries=args.n_queries)
        if args.benchmark:
            args.benchmark.parent.mkdir(parents=True, exist_ok=True)
            with open(args.benchmark, "w") as f:
                json.dump(benchmark, f, indent=2)
            log.info("Saved benchmark to %s", args.benchmark)

    # Load issues for keyword baseline
    issues_path = Path("data/processed/issues.csv")
    import csv
    issues_raw = []
    if issues_path.exists():
        with open(issues_path, newline="", encoding="utf-8") as f:
            issues_raw = list(csv.DictReader(f))

    # Build systems
    our_system       = OurSystemAdapter(pipeline)
    keyword_baseline = KeywordBaseline(issues_raw)
    cypher_baseline  = CypherBaseline(db)

    evaluator = Evaluator(our_system, keyword_baseline, cypher_baseline)

    # Run
    log.info("Running evaluation on %d queries …", len(benchmark))
    results = evaluator.run(benchmark)
    summary = evaluator.aggregate(results)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "summary":  summary,
        "per_query": [asdict(r) for r in results],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    log.info("\n" + "═" * 70)
    log.info("EVALUATION RESULTS (for paper Table 2)")
    log.info("═" * 70)
    for key, metrics in sorted(summary.items()):
        log.info("%-40s %s", key, metrics)
    log.info("═" * 70)
    log.info("Full results saved to %s", args.output)

    db.close()


if __name__ == "__main__":
    main()
