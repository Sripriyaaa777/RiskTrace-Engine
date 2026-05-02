# RiskTrace Engine

> Hybrid GenAI-based predictive risk detection over temporal Jira dependency graphs

RiskTrace Engine ingests Jira-style issue data, builds a dependency graph, propagates downstream risk, runs counterfactual "what-if" analysis, and validates predictive delay detection on historical issue snapshots.

The project now supports:
- real Apache Jira BSON preprocessing
- user-uploaded Jira-like datasets with project-slice selection
- transformer-based prediction using `distilroberta-base`
- local RAG retrieval with a persisted TF-IDF index
- agentic multi-step routing with planning / decision / critic stages
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
- Retrieve related issue context via local RAG
- Dynamically route through an agentic reasoning loop
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

### Current RAG backend
- Backend: `local_tfidf`
- Role: retrieve relevant issue / pattern documents without external model download

### Current agentic orchestration
- Runtime: `RiskTraceAgenticPipeline`
- Stages: `perception -> rag_retrieval -> planning -> issue_detail/decision -> critic`
- Role: dynamically route reasoning instead of using one fixed call chain

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
Local RAG retrieval (TF-IDF)
    ->
Agentic planning / decision / critic routing
    ->
Optional Groq LLM explanation layer
```

## Repository Structure

```text
RiskTrace-Engine-main/
├── main.py                   FastAPI app and all API routes
├── preprocess.py             Raw Jira preprocessing and synthetic data generation
├── dataset_manager.py        Uploaded dataset registry and slice preparation
├── predictive_model.py       Transformer-based predictive training pipeline
├── predictive_analysis.py    Historical replay and predictive validation
├── risk_engine.py            Core dependency risk propagation
├── agents.py                 Base agents and classic orchestration
├── agentic_pipeline.py       Agentic LangGraph-style orchestration layer
├── rag_engine.py             Local persisted TF-IDF RAG backend
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
- User-uploaded dataset ingestion
- Automatic project-slice discovery (e.g. HADOOP, SPARK, HIVE)
- Temporal downstream risk propagation
- Counterfactual graph simulation
- Transformer-enhanced predictive delay detection
- Cross-validation and temporal holdout evaluation
- Dataset-driven UI workflow for upload → slice selection → activation
- Cached slice reuse for faster repeat activation
- Background RAG refresh so API startup is not blocked

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

## Quick Start

If you just want the app running locally with the default processed dataset:

### Terminal 1 — start the backend

```bash
cd /path/to/RiskTrace-Engine-main
source .venv/bin/activate
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8004
```

### Terminal 2 — serve the UI

```bash
cd /path/to/RiskTrace-Engine-main
python3 -m http.server 9000 --bind 127.0.0.1
```

### Open in browser

- UI: [http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004](http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004)
- Health: [http://127.0.0.1:8004/health](http://127.0.0.1:8004/health)
- API docs: [http://127.0.0.1:8004/docs](http://127.0.0.1:8004/docs)

Expected health response:

```json
{
  "status": "ok",
  "pipeline_type": "RiskTraceAgenticPipeline",
  "rag_ready": true
}
```

This quick start uses:
- CSV mode
- local TF-IDF RAG
- agentic pipeline
- optional Groq LLM if `GROQ_API_KEY` is present in `.env`

## Detailed Run Guide

This section explains the exact commands for the three main ways to run the project:

1. local default CSV mode
2. real Jira + Neo4j mode
3. upload-your-own-dataset product flow

---

## Run Mode 1: Local Default CSV Mode

This is the fastest way to run the project for development and demos.

### What it uses

- `data/processed/issues.csv`
- `data/processed/dependencies.csv`
- local TF-IDF RAG index in `data/rag_index`
- `RiskTraceAgenticPipeline`

### Start backend

```bash
cd /path/to/RiskTrace-Engine-main
source .venv/bin/activate
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8004
```

### Start UI

```bash
cd /path/to/RiskTrace-Engine-main
python3 -m http.server 9000 --bind 127.0.0.1
```

### Open

- [http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004](http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004)

### Verify backend

```bash
curl http://127.0.0.1:8004/health
curl http://127.0.0.1:8004/rag-status
curl "http://127.0.0.1:8004/agent-trace?query=Why%20is%20HADOOP-8%20high%20risk%3F"
```

What you should see:

- `/health`
  - `pipeline_type: RiskTraceAgenticPipeline`
  - `rag_ready: true`
- `/rag-status`
  - `backend: local_tfidf`
- `/agent-trace`
  - `execution_path` includes `perception`, `rag_retrieval`, `planning`, `decision`, `critic`

---

## Run Mode 2: Real Apache Jira + Neo4j

This is the heavier, presentation-ready graph database setup.

### Step 1 — preprocess the raw BSON into a slice

Example for `HADOOP`:

```bash
cd /path/to/RiskTrace-Engine-main
source .venv/bin/activate

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

### Step 2 — start Neo4j

If the container already exists:

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

### Step 3 — import the graph into Neo4j

```bash
cd /path/to/RiskTrace-Engine-main

NEO4J_URI=bolt://127.0.0.1:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=risktrace123 \
ISSUES_CSV=data/real_hadoop/issues.csv \
DEPS_CSV=data/real_hadoop/dependencies.csv \
.venv/bin/python build_graph.py
```

### Step 4 — start backend in Neo4j mode

```bash
cd /path/to/RiskTrace-Engine-main

USE_NEO4J=true \
NEO4J_URI=bolt://127.0.0.1:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=risktrace123 \
ISSUES_CSV=data/real_hadoop/issues.csv \
DEPS_CSV=data/real_hadoop/dependencies.csv \
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8004
```

### Step 5 — start UI

```bash
cd /path/to/RiskTrace-Engine-main
python3 -m http.server 9000 --bind 127.0.0.1
```

### Step 6 — verify

```bash
curl http://127.0.0.1:8004/health
```

Expected:

```json
{
  "status": "ok",
  "db_mode": "neo4j"
}
```

In the UI you should see:

```text
Connected
backend: neo4j
CSV staging imported to graph
```

That means:
- raw Jira was normalized into CSV
- CSV was loaded into Neo4j
- backend is now querying Neo4j

---

## Run Mode 3: Upload Your Own Dataset

This is the product-style workflow inside the UI.

### Step 1 — start backend

```bash
cd /path/to/RiskTrace-Engine-main
source .venv/bin/activate
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8004
```

### Step 2 — start UI

```bash
cd /path/to/RiskTrace-Engine-main
python3 -m http.server 9000 --bind 127.0.0.1
```

### Step 3 — open UI

- [http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004](http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004)

### Step 4 — use Settings

1. open `Settings`
2. upload a dataset file:
   - BSON
   - JSON
   - CSV
3. choose the uploaded dataset
4. choose a project slice such as:
   - `HADOOP`
   - `SPARK`
   - `HIVE`
   - `HBASE`
5. choose max issues per slice
6. click `Prepare Slice & Activate`

### What happens internally

```text
raw uploaded dataset
-> discover projects
-> prepare selected slice
-> generate issues.csv + dependencies.csv
-> activate slice in backend
-> refresh dashboard
-> rebuild or reuse local RAG index
```

### Performance notes

- first activation of a brand-new slice is slower
- activating an already prepared slice is faster
- selecting the exact same active slice reuses the running runtime when possible

---

## Optional Groq LLM Setup

To enable full natural-language explanations in the dashboard and agent chat:

Create `.env` in the project root:

```env
GROQ_API_KEY=your_key_here
```

Then restart the backend:

```bash
pkill -f "uvicorn main:app --host 127.0.0.1 --port 8004"
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8004
```

Without Groq:
- graph logic still works
- predictive model still works
- counterfactual analysis still works
- RAG still works
- agentic routing still works
- only the natural-language LLM summary falls back to structured text

---

## Stop / Restart Commands

### Stop backend

```bash
pkill -f "uvicorn main:app --host 127.0.0.1 --port 8004"
```

### Stop UI server

```bash
pkill -f "http.server 9000"
```

### Restart backend

```bash
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8004
```

### Restart UI server

```bash
python3 -m http.server 9000 --bind 127.0.0.1
```

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

## Product Workflow: Upload Your Own Dataset

The UI now supports user-provided Jira-like datasets, not just a preloaded HADOOP slice.

Supported input formats:
- BSON
- JSON
- CSV

### What the product flow does

```text
Upload dataset
    ->
Discover available project slices
    ->
Choose one slice (e.g. HADOOP / SPARK / HIVE)
    ->
Prepare issues.csv + dependencies.csv for that slice
    ->
Activate the slice in the live backend
    ->
Refresh dashboard / what-if / predictive views
```

### UI flow

1. Open the served dashboard:

- [http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004](http://127.0.0.1:9000/issuegraph_ui.html?api=http://127.0.0.1:8004)

2. Open `Settings`
3. Upload a BSON / JSON / CSV Jira-like dataset
4. Select an uploaded dataset
5. Select a discovered project slice like `HADOOP`, `SPARK`, or `HIVE`
6. Click `Prepare Slice & Activate`

The Settings modal now supports:
- `Uploading...` and `Preparing slice...` progress states
- close button
- click-outside-to-close
- `Back to Dashboard`
- automatic return to the dashboard after successful activation

### Slice activation performance

To keep repeated activations fast, the runtime now:
- reuses already prepared slice CSVs when the same dataset/project/options are selected
- skips a full runtime rebuild when the exact same active slice is selected again
- refreshes RAG in the background instead of blocking server startup

This means:
- first activation of a brand-new slice is still heavier
- reactivating an already prepared slice is much faster than before

### New dataset endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/datasets` | List uploaded datasets and current active dataset |
| GET | `/datasets/{dataset_id}` | Show discovered project slices for one uploaded dataset |
| POST | `/datasets/upload` | Upload a raw dataset file and discover projects |
| POST | `/datasets/{dataset_id}/activate` | Prepare and activate a chosen slice |

### What “CSV staging imported to graph” means

When the UI status shows:

```text
backend: neo4j
CSV staging imported to graph
```

it means:

```text
raw Jira dataset
-> normalized slice CSV files
-> graph loaded into Neo4j
-> backend queries Neo4j
```

So CSV is not the final database. It is only the normalized staging/import format.

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
- active dataset

## Main API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check and active DB mode |
| GET | `/rag-status` | Live RAG backend status |
| GET | `/agent-trace` | Agentic execution path and RAG usage proof |
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
- Dataset upload + slice activation
- Backend settings and connection status

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

### Uploaded-slice runtime example

Verified uploaded slices in this workspace include real project slices such as:
- `HADOOP`
- `SPARK`
- `HIVE`

Example activated runtime slice:
- dataset label: `issues.bson [SPARK]`
- slice size: `1500` issues
- dependencies: `4368`

Important note: different project slices can have very different metadata quality. Some slices may have good dependency structure but weak delay/deadline coverage, which affects dashboard ranking quality.

## Groq / LLM Support

To enable LLM explanations, add:

```env
GROQ_API_KEY=your_key_here
```

This powers:
- dashboard LLM summary
- recommendations
- natural language chat

Without Groq, the core graph engine, predictive model, counterfactual simulation, and RAG-backed retrieval still work. The system falls back to structured outputs instead of natural-language summaries.

## RAG and Agentic Verification

The final working runtime can be verified directly from the API:

### Health

```bash
curl http://127.0.0.1:8004/health
```

Look for:

```json
{
  "pipeline_type": "RiskTraceAgenticPipeline",
  "rag_ready": true
}
```

### RAG status

```bash
curl http://127.0.0.1:8004/rag-status
```

Expected shape:

```json
{
  "ready": true,
  "backend": "local_tfidf",
  "count": 304
}
```

### Agentic execution trace

```bash
curl "http://127.0.0.1:8004/agent-trace?query=Why%20is%20HADOOP-8%20high%20risk%3F"
```

Expected shape:

```json
{
  "execution_path": [
    "perception",
    "rag_retrieval",
    "planning",
    "issue_detail",
    "decision",
    "critic"
  ],
  "rag_used": true,
  "rag_docs_retrieved": 5
}
```

This proves:
- the agentic routing path is active
- RAG retrieval is active
- the critic stage is active
- the app is not falling back to the old fixed pipeline

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
- some slices may load correctly but still produce weak top-risk ranking if deadline or delay metadata is missing

## Known Limitation: Slice Quality Varies

Not every uploaded project slice is equally informative.

Examples of slice-specific issues:
- some slices have many dependencies but no usable due dates
- some slices have zero explicit delay labels
- some slices rely heavily on inferred soft dependencies

This means a slice can still be:
- graph-valid
- loaded successfully
- useful for what-if analysis

while still being weaker for:
- top-risk ranking
- delay-days interpretation
- predictive validation quality

In these cases, the system should be interpreted as a graph-based decision-support tool rather than an exact delay forecaster.

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
