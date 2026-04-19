# RiskTrace-Engine
# IssueGraphAgent++

> A Multi-Agent Generative AI Framework for Proactive Risk Propagation  
> over Temporal Dependency Graphs in Software Project Management

---

## Project Structure

```
issuegraphagent/
├── scripts/
│   ├── preprocess.py       Phase 1 — Data ingestion & normalisation
│   └── build_graph.py      Phase 2 — Neo4j graph construction
├── core/
│   └── risk_engine.py      Phase 2 — Temporal risk propagation algorithm
├── agents/
│   └── agents.py           Phase 3 — All 6 agents + AgentPipeline
├── api/
│   └── main.py             Phase 4 — FastAPI REST backend
├── evaluation/
│   └── evaluate.py         Phase 4 — Benchmark evaluation framework
├── data/
│   ├── raw/                Put downloaded dataset here
│   └── processed/          Auto-generated CSVs go here
├── requirements.txt
└── .env.example
```

---

## Quickstart (step by step)

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Set up environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=issuegraph123
MONITOR_INTERVAL=60
```

### Step 3 — Start Neo4j (Docker, no installation needed)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/issuegraph123 \
  neo4j:5.18-community
```

Neo4j Browser will be available at http://localhost:7474  
Use credentials: `neo4j` / `issuegraph123`

### Step 4 — Download the Apache Jira Dataset

Download from: https://zenodo.org/records/7740379  
Place the JSON/CSV file in `data/raw/`.

### Step 5 — Preprocess the dataset

**Option A — Real dataset:**
```bash
python scripts/preprocess.py \
    --input data/raw/issues.json \
    --max-issues 300 \
    --project HADOOP \
    --output data/processed
```

**Option B — Synthetic dataset (for testing without downloading):**
```bash
python scripts/preprocess.py --synthetic --max-issues 200
```

This produces:
- `data/processed/issues.csv`       — normalised issue nodes
- `data/processed/dependencies.csv` — dependency edges
- `data/processed/stats.json`       — dataset statistics for the paper

### Step 6 — Build the graph in Neo4j

```bash
python scripts/build_graph.py \
    --issues   data/processed/issues.csv \
    --deps     data/processed/dependencies.csv
```

Verify by opening http://localhost:7474 and running:
```cypher
MATCH (n:Issue) RETURN count(n)
MATCH ()-[r:DEPENDS_ON]->() RETURN count(r)
```

### Step 7 — Start the API

```bash
cd issuegraphagent
uvicorn api.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### Step 8 — Test the system

```bash
# Health check
curl http://localhost:8000/health

# Natural language query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the riskiest tasks?"}'

# Issue-specific risk
curl http://localhost:8000/risk/HADOOP-42

# Dependency graph for an issue
curl http://localhost:8000/graph/HADOOP-42

# Dashboard data
curl http://localhost:8000/dashboard

# Proactive alerts
curl http://localhost:8000/alerts
```

### Step 9 — Run the evaluation (for paper)

```bash
python evaluation/evaluate.py \
    --n-queries 60 \
    --output evaluation/results.json
```

This auto-generates benchmark queries from the graph, runs all three
systems (ours, keyword baseline, Cypher baseline), computes metrics,
and prints a summary table for Table 2 in the paper.

---

## Architecture

```
Dataset (Apache Jira)
        ↓
Perception Agent        → Ingests graph state from Neo4j
        ↓
Graph Reasoning Agent   → Multi-hop traversal + risk propagation
        ↓                 (core algorithm: temporal decay formula)
Planning Agent          → Decomposes mitigation goals
        ↓
Decision Agent (GPT-4o) → Natural language explanation + recommendations
        ↓
Critic Agent            → Validates LLM output, strips hallucinations
        ↓
FastAPI / Next.js       → REST API + Frontend

Monitoring Agent        → Background thread, continuous proactive alerts
```

## Risk Score Formula

```
R(v) = Σ [ delay_severity(u) × depth_weight(d) × temporal_weight(u) ]

Where:
  delay_severity(u)  = min(delay_days / 30, 1.0)     for delayed nodes
                       1.0                             for blocked nodes
  depth_weight(d)    = 0.8^(d-1)                     attenuates with hops
  temporal_weight(u) = 0.5 + 0.5 × e^(-age_days/30)  recent = fresher signal

Thresholds: High ≥ 0.70 · Medium ≥ 0.35 · Low < 0.35
```

---

## For the Paper

Key files to reference in your implementation section:
- `core/risk_engine.py`     — Algorithm 1 (risk propagation)
- `agents/agents.py`        — Figure 2 (agent pipeline)
- `evaluation/evaluate.py`  — Section 9 (evaluation framework)
- `data/processed/stats.json` — Table 1 (dataset statistics)
- `evaluation/results.json` — Table 2 (experimental results)
