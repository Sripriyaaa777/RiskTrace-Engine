# RiskTrace Engine

> Hybrid GenAI-based predictive risk detection over temporal Jira dependency graphs

RiskTrace Engine ingests Jira-style issue data, builds a dependency graph, propagates downstream risk, runs counterfactual "what-if" analysis, and validates predictive delay detection on historical issue snapshots.

The project now supports:
- real Apache Jira BSON preprocessing
- synthetic demo data for visualization
- transformer-based prediction using `distilroberta-base`
- optional Groq-powered LLM explanations
- CSV-backed runtime and Neo4j graph loading support

## What The System Does

RiskTrace has four main layers:

1. Data preprocessing
- Parse raw Jira-like issue data from BSON, JSON, or CSV
- Normalize dates, statuses, priorities, and descriptions
- Extract explicit dependency links and infer some soft dependencies from text

2. Graph reasoning
- Build a graph where:
  - node = issue
  - edge = dependency
- Propagate delay/blocker risk through downstream dependencies

3. Predictive modeling
- Reconstruct issue state at historical time `T`
- Encode issue text with a transformer embedding model
- Combine embeddings with graph and temporal features
- Predict whether an issue would later become delayed

4. Decision support
- Run counterfactual simulations like "what happens if I fix this issue now?"
- Optionally explain findings using a Groq-hosted LLM

## Current Models Used

### Core graph model
- `RiskPropagationEngine`
- Type: deterministic graph propagation
- Role: compute cascading risk over dependency edges

### Core predictive model
- `roberta_embedding_logistic_regression`
- Text encoder: `distilroberta-base`
- Role: use Jira issue text + structured graph/time features to predict future delay risk

### Optional LLM explanation model
- Groq model: `llama-3.3-70b-versatile`
- Role: summaries, explanations, recommendations, chat

This means the project is best described as:

`a hybrid GenAI + graph analytics + predictive ML system`

## End-to-End Workflow

```text
Raw Jira Data (BSON / JSON / CSV)
    ->
Preprocessing
    ->
issues.csv + dependencies.csv
    ->
Graph backend (CSV or Neo4j)
    ->
RiskPropagationEngine
    ->
Counterfactual analysis / dashboard risk views
    ->
Snapshot feature builder at time T
    ->
distilroberta-base text embeddings
    ->
LogisticRegression classifier
    ->
Predictive validation metrics
    ->
Optional Groq LLM explanation layer
```

## Repository Structure

```text
RiskTrace-Engine-main/
├── main.py                   FastAPI app and all API routes
├── preprocess.py             Raw Jira preprocessing and synthetic data generation
├── predictive_model.py       Transformer-based predictive training pipeline
├── predictive_analysis.py    Historical replay and predictive validation
├── risk_engine.py            Core dependency risk propagation
├── agents.py                 Agent orchestration and explanation logic
├── csv_db.py                 CSV graph backend
├── build_graph.py            Neo4j graph loader
├── issuegraph_ui.html        Single-file dashboard UI
├── requirements.txt          Python dependencies
└── data/
    ├── processed/            Synthetic/demo processed data
    ├── real_hadoop/          Real Apache Jira processed HADOOP slice
    └── models/               Saved predictive model artifacts + embedding cache
```

## Key Features

- Time-aware Jira preprocessing
- BSON support for real Apache Jira exports
- Safe handling of malformed records
- Explicit + soft dependency extraction
- Temporal downstream risk propagation
- Counterfactual graph simulation
- Transformer-enhanced predictive delay detection
- Cross-validation and temporal holdout evaluation
- UI support for both synthetic and real Jira backends

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running With Synthetic Data

Synthetic mode is best for demos because it has a denser dependency graph and more visually obvious cascades.

### 1. Generate synthetic processed data

```bash
.venv/bin/python preprocess.py --synthetic
```

This creates:
- `data/processed/issues.csv`
- `data/processed/dependencies.csv`

### 2. Start the synthetic backend

```bash
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8005
```

### 3. Open the UI

Serve the static UI:

```bash
python3 -m http.server 9000
```

Then open:

- UI: [http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8005](http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8005)

## Recommended Run: Real Apache Jira + Neo4j + UI

This is the current presentation-ready setup. It uses the real Apache Jira HADOOP slice, imports the graph into Neo4j, runs the FastAPI backend in Neo4j mode, and serves the dashboard UI.

### 1. Preprocess the raw BSON data

Example:

```bash
.venv/bin/python preprocess.py \
  --input ../Downloads/issues.bson \
  --project HADOOP \
  --max-issues 1500 \
  --output-dir data/real_hadoop \
  --augment-soft-deps
```

Expected outputs:
- `data/real_hadoop/issues.csv`
- `data/real_hadoop/dependencies.csv`
- `data/real_hadoop/stats.json`

The CSV files here are a staging format: they normalize raw Jira records into a graph schema before loading Neo4j.

### 2. Start Neo4j

If the local Neo4j container already exists:

```bash
docker start risktrace-neo4j
```

If you need to create it:

```bash
docker run --name risktrace-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/risktrace123 \
  -d neo4j:5
```

Neo4j Browser:

- [http://127.0.0.1:7474](http://127.0.0.1:7474)
- username: `neo4j`
- password: `risktrace123`

### 3. Load the Jira graph into Neo4j

```bash
NEO4J_URI=bolt://127.0.0.1:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=risktrace123 \
ISSUES_CSV=data/real_hadoop/issues.csv \
DEPS_CSV=data/real_hadoop/dependencies.csv \
.venv/bin/python build_graph.py
```

### 4. Start the real Jira backend in Neo4j mode

```bash
USE_NEO4J=true \
NEO4J_URI=bolt://127.0.0.1:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=risktrace123 \
ISSUES_CSV=data/real_hadoop/issues.csv \
DEPS_CSV=data/real_hadoop/dependencies.csv \
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8004
```

Verify:

```bash
curl http://127.0.0.1:8004/health
```

Expected:

```json
{
  "status": "ok",
  "db_mode": "neo4j",
  "data_loaded": true
}
```

### 5. Serve and open the UI

In another terminal:

```bash
python3 -m http.server 9000
```

Open:

- UI: [http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004](http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004)
- Docs: [http://127.0.0.1:8004/docs](http://127.0.0.1:8004/docs)

The UI should show:

```text
Connected
backend: neo4j
CSV staging imported to graph
```

That means Neo4j is the active backend. CSV is only the import/staging format produced by preprocessing.

## Current Real Jira Slice

Current verified HADOOP slice:
- `1500` issues
- `4353` dependency edges after soft-dependency augmentation
- Neo4j runtime mode

Important caveat: most Jira projects have sparse explicit dependency links. The enhanced graph includes inferred soft dependencies from issue text/project context, so downstream impact should be interpreted as a decision-support signal, not guaranteed causality.

## Training The Predictive Model

Train the current transformer-based predictor:

```bash
.venv/bin/python predictive_model.py \
  --issues data/real_hadoop/issues.csv \
  --deps data/real_hadoop/dependencies.csv \
  --project HADOOP \
  --folds 5 \
  --model-path data/models/predictive_model.joblib \
  --text-encoder-model distilroberta-base
```

This:
- encodes issue text with `distilroberta-base`
- combines embeddings with graph/time features
- trains a logistic regression classifier
- reports cross-validation and temporal holdout metrics
- saves the model artifact

## Predictive API Endpoints

### Train predictive model

`POST /train-predictive-model`

Query params:
- `project`
- `folds`
- `text_encoder_model`

Example:

```bash
curl -X POST "http://127.0.0.1:8004/train-predictive-model?project=HADOOP&folds=5&text_encoder_model=distilroberta-base"
```

### Predictive analysis

`GET /predictive-analysis`

Query params:
- `project`
- `threshold`
- `positive_target`
- `total_target`
- `use_trained_model`

Example:

```bash
curl "http://127.0.0.1:8004/predictive-analysis?project=HADOOP&positive_target=5&total_target=8&use_trained_model=true"
```

### Predictive model metadata

`GET /predictive-model-info`

Returns:
- model kind
- encoder model
- trained time
- project

## Main API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check and active DB mode |
| GET | `/dashboard` | Dashboard summary, top risks, action plan |
| GET | `/risk/{issue_id}` | Risk details for a single issue |
| GET | `/graph/{issue_id}` | Issue-centered dependency graph |
| GET | `/alerts` | Monitoring alerts |
| POST | `/query` | Natural language agent query |
| POST | `/counterfactual/{issue_id}` | What-if graph simulation |
| POST | `/train-predictive-model` | Train predictive delay model |
| GET | `/predictive-analysis` | Historical predictive validation |
| GET | `/predictive-model-info` | Active predictive model metadata |

## UI Features

The dashboard in [issuegraph_ui.html](/Users/sreejith/RiskTrace-Engine-main%202/issuegraph_ui.html) includes:

- Dashboard summary
- Risk board
- Action plan
- Alerts
- What-If analysis
- Predictive model summary panel
- Dataset switching between:
  - real Jira backend
  - synthetic backend

## Counterfactual What-If Analysis

The What-If tab lets a manager test:

`What happens if this issue is resolved right now?`

Workflow:
- clone the current graph in memory
- mark the selected issue as resolved
- rerun risk propagation
- compare before vs after

Output includes:
- nodes improved
- high-risk reduction
- delay-days saved
- before/after graph diff
- node-level deltas

## Current Verified Results

### Real Jira predictive model

On the current HADOOP slice:
- model kind: `roberta_embedding_logistic_regression`
- encoder: `distilroberta-base`

5-fold cross-validation average:
- precision: `0.910`
- recall: `0.933`
- accuracy: `0.998`
- f1: `0.909`
- roc_auc: `0.967`

Temporal holdout:
- precision: `0.857`
- recall: `0.667`
- accuracy: `0.987`
- f1: `0.750`
- roc_auc: `0.954`

Example predictive-analysis run:
- precision: `1.000`
- recall: `1.000`
- accuracy: `1.000`

These numbers are on a small real-data slice and should be interpreted cautiously.

## Groq / LLM Support

To enable LLM explanations, add:

```env
GROQ_API_KEY=your_key_here
```

This powers:
- dashboard LLM summary
- recommendations
- natural language chat

Without Groq, the core graph engine, predictive model, and counterfactual simulation still work.

## Neo4j Support

Neo4j graph loading is implemented in `build_graph.py`. The current verified live Jira run uses Neo4j as the active graph backend.

To use Neo4j, provide:

```env
USE_NEO4J=true
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

Load data with:

```bash
.venv/bin/python build_graph.py
```

## Known Limitation: Real Jira Dependency Sparsity

The main limitation is the dataset, not the pipeline.

Problems with the real Apache Jira dataset:
- relatively few explicit dependency links
- inconsistent workflow/status semantics
- incomplete due-date coverage
- many hidden dependencies only implied in text/comments
- weaker graph density than synthetic demo data

Effects:
- real graphs look sparser
- counterfactual improvements are often smaller
- what-if views can be less visually dramatic

## How To Improve The Real Jira Pipeline

Recommended next steps:

1. Improve dependency extraction
- normalize Jira link types more carefully
- infer stronger soft dependencies from text/comments

2. Improve labels
- define cleaner delay labels
- add blocker and downstream-impact labels

3. Improve temporal reconstruction
- ensure every feature uses only information available at time `T`

4. Explore richer project slices
- compare HADOOP with KAFKA or SPARK

5. Upgrade graph learning
- add a graph neural network or stronger learned graph component

## Project Positioning

This project now qualifies as:

`a hybrid GenAI project`

Why:
- transformer embeddings are used in the core prediction pipeline
- Groq LLM can be used for explanations
- graph reasoning remains central for explainability and counterfactual analysis

Most accurate description:

`Hybrid GenAI-based predictive risk detection over temporal Jira dependency graphs`
