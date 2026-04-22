# IssueGraphAgent++ 🔍

**Proactive risk propagation and counterfactual reasoning over software project dependency graphs.**

RiskTrace Engine ingests issue-tracker data (JIRA-style tickets), builds a temporal dependency graph, and continuously propagates risk scores from delayed or blocked tasks downstream — alerting teams *before* cascades become critical.

The centerpiece novel contribution is **Counterfactual What-If Analysis**: a manager can hypothetically resolve any issue and instantly see how downstream risk scores change, enabling principled prioritisation decisions backed by quantitative impact estimates.

---

## Key Features

- **Multi-hop temporal risk propagation** — Risk decays with graph distance (γ = 0.8 per hop) and issue staleness, producing a principled score in [0, 1] for every node
- **Counterfactual reasoning** — In-memory graph clone + re-propagation; no database writes. Shows before/after risk scores and a visual graph diff
- **Multi-agent pipeline** — Six agents: Perception, Graph Reasoning, Planning, Decision (LLM), Monitoring, Critic
- **Two backend modes** — CSV mode (no Neo4j needed) or full Neo4j mode; switch with one env variable
- **Dark-themed dashboard UI** — Risk board, action plan, alerts, and What-If tab; pure HTML, no build step

---

## How It Works

```
Issues CSV + Dependencies CSV  (or Neo4j graph)
          │
          ▼
    PerceptionAgent          fetches nodes + edges
          │
          ▼
  GraphReasoningAgent        runs RiskPropagationEngine
          │
          ▼
   RiskPropagationEngine     R(v) = Σ [ severity × γ^depth × temporal_weight ]
          │
          ├──▶ PlanningAgent      ranked mitigation plan
          ├──▶ DecisionAgent      LLM explanation (optional)
          ├──▶ MonitoringAgent    background watcher, proactive alerts
          └──▶ CriticAgent        validates LLM outputs

What-If:
  Manager selects issue → POST /counterfactual/{id}
  → clone graph in memory → patch node to Done → re-propagate
  → return before/after diff + graph_nodes for visual rendering
```

**Risk Score Formula:**
```
R(v) = Σ [ delay_severity(u) × γ^(depth-1) × temporal_weight(u) ]
       for each upstream risky ancestor u at depth d

delay_severity = min(delay_days / 30, 1.0)   for delayed nodes
               = 1.0                          for blocked nodes
γ (depth decay) = 0.80
temporal_weight = 0.5 + 0.5 × e^(−age_days / 30)
```

---

## Project Structure

```
RiskTrace-Engine-main/
├── main.py            FastAPI app — all routes including /counterfactual
├── agents.py          All six agents + AgentPipeline orchestrator
├── risk_engine.py     Core propagation algorithm (Neo4j-agnostic)
├── csv_db.py          In-memory CSV graph DB (drop-in Neo4j replacement)
├── build_graph.py     Neo4j graph builder (used in Neo4j mode)
├── preprocess.py      Synthetic data generator
├── evaluate.py        Evaluation + benchmarking utilities
├── issuegraph_ui.html Single-file dashboard (open directly in browser)
└── requirements.txt   Python dependencies
```

---

## Quickstart — Mode A: CSV (No Neo4j Required)

This is the recommended way to get started. Everything runs locally with no database setup.

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/RiskTrace-Engine.git
cd RiskTrace-Engine-main
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file
Create a file named `.env` in the project root:
```env
USE_NEO4J=false
MONITOR_INTERVAL=60

# Optional — add for LLM explanations in the dashboard
# GROQ_API_KEY=your_groq_key_here
```

### 4. Generate synthetic data
```bash
python preprocess.py --synthetic
```
This creates `data/processed/issues.csv` and `data/processed/dependencies.csv` — around 200 synthetic issues with realistic dependency chains, delays, and blocked tasks.

### 5. Start the API server
```bash
uvicorn main:app --reload --port 8000
```
Expected output:
```
✓ Running in CSV mode
IssueGraphAgent++ ready  [mode: csv]
```

### 6. Open the dashboard
Open `issuegraph_ui.html` directly in your browser (double-click the file or drag it into your browser). When prompted for the API URL, enter:
```
http://localhost:8000
```

---

## Quickstart — Mode B: Neo4j

Use this mode if you want a real graph database backend, or when working with actual JIRA data exports.

### 1. Install Neo4j Desktop
Download and install from [neo4j.com/download](https://neo4j.com/download).
- Open Neo4j Desktop → Create a new Project → Add a Database → Start it
- Set a password when prompted (you'll use this in `.env`)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file
```env
USE_NEO4J=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
MONITOR_INTERVAL=60

# Optional — add for LLM explanations
# GROQ_API_KEY=your_groq_key_here
```

### 4. Generate synthetic data and load into Neo4j
```bash
# First generate the CSV data
python preprocess.py --synthetic

# Then load it into Neo4j (creates Issue nodes + DEPENDS_ON relationships)
python build_graph.py
```

### 5. Start the API server
```bash
uvicorn main:app --reload --port 8000
```
Expected output:
```
✓ Connected to Neo4j
IssueGraphAgent++ ready  [mode: neo4j]
```

> **Note:** If Neo4j is not reachable at startup, the server automatically falls back to CSV mode with a warning. This means the server will never crash due to a database connection issue.

### 6. Open the dashboard
Open `issuegraph_ui.html` in your browser and enter `http://localhost:8000` as the API URL.

---

## Switching Between Modes

Switching is a one-line change in `.env`:

```env
# CSV mode (no Neo4j)
USE_NEO4J=false

# Neo4j mode
USE_NEO4J=true
```

All agents, the risk engine, and the What-If counterfactual feature work **identically in both modes**. The only difference is where the graph data comes from.

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Server status and active DB mode |
| GET | `/dashboard` | Full dashboard: summary + top risks + action plan |
| GET | `/risk/{issue_id}` | Risk score + explanation for one issue |
| GET | `/graph/{issue_id}` | Dependency chain as nodes + edges |
| GET | `/alerts` | Proactive monitoring alerts |
| POST | `/query` | Natural language query to AI agent |
| POST | `/counterfactual/{issue_id}` | **What-If simulation** — before/after risk diff |

Interactive API docs (Swagger UI) available at:
```
http://localhost:8000/docs
```

### Counterfactual endpoint example
```bash
curl -X POST http://localhost:8000/counterfactual/HADOOP-121 \
  -H "Content-Type: application/json" \
  -d '{"resolve_as": "Done"}'
```

Response includes:
- `impact_summary` — nodes improved, high-risk reduction, estimated delay-days saved
- `diff[]` — per-node before/after risk scores sorted by most improved
- `graph_nodes[]` + `graph_edges[]` — full graph state for visual rendering

---

## What-If Analysis (Counterfactual Reasoning)

This is the novel research contribution of this project. To use it:

1. Open the dashboard and click **What-If** in the left sidebar
2. Type any issue ID (e.g. `HADOOP-121`, `KAFKA-23`, `HADOOP-16`)
3. Click **Run Simulation** or press Enter

The system will:
- Clone the current dependency graph entirely in memory (zero DB writes)
- Mark the selected issue as resolved (Done, delay = 0)
- Rerun the full temporal risk propagation algorithm on the clone
- Display a **side-by-side graph diff** (BEFORE / AFTER) with colour-coded nodes
- Show impact cards: nodes improved, high-risk reduction, delay-days saved
- List every node whose risk score changed, sorted by biggest improvement

**Why it's novel:** Counterfactual reasoning has not previously been applied to software project dependency graphs. The simulation uses the same temporal decay formula as the live engine, making the estimates principled rather than heuristic. This is formally equivalent to interventional reasoning in causal inference literature.

**Good issues to test with synthetic data:**
- `HADOOP-121` — direct blocker of the KAFKA-23 chain
- `HADOOP-16` — blocked, cascades into 7+ high-risk downstream nodes
- `KAFKA-33` — blocked, feeds SPARK-34 → SPARK-35 → HADOOP-36 chain
- `KAFKA-23` — high centrality node; resolving it affects a large portion of the graph

---

## Optional: LLM Explanations

Add a Groq API key to `.env` to enable natural language explanations:

```env
GROQ_API_KEY=your_groq_key_here
```

This powers:
- The **LLM Analysis** panel in the dashboard (root cause summary + recommendations)
- The **AI Agent chat** (ask natural language questions about your project)

The risk engine, What-If feature, and all other functionality work fully without it.

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---
