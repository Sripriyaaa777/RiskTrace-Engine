# RiskTrace Engine

RiskTrace Engine is a FastAPI-based risk analysis service for issue dependency graphs. It can run in:

- `CSV mode`: reads `data/processed/issues.csv` and `data/processed/dependencies.csv`
- `Neo4j mode`: reads the same processed data after loading it into Neo4j

The project also includes a standalone frontend dashboard in [issuegraph_ui.html](/Users/sreejith/Downloads/RiskTrace-Engine-main/issuegraph_ui.html) that talks to the backend over HTTP.

## Current Status

The current codebase has been updated so that:

- the broken local imports are fixed
- the backend starts correctly in this repo layout
- the preprocessor accepts Zenodo `issues.bson` and `issues.bson.gz` directly
- BSON preprocessing now streams records instead of loading the full dump into memory first
- you can optionally stream raw issues from MongoDB with `--mongodb-uri`
- the Neo4j loader preserves issue fields and dependency metadata
- the frontend endpoint wiring matches the backend routes
- the dashboard LLM gate now checks `GROQ_API_KEY` consistently

Verified locally in this workspace:

- `python3 -m compileall` passes for the main Python files
- `preprocess.py --synthetic` runs successfully
- a fresh backend instance on `http://127.0.0.1:8001` returns valid responses for:
  - `/health`
  - `/dashboard`
  - `/counterfactual/{issue_id}`

Not fully verified here:

- Neo4j mode, because that requires your local Neo4j instance and credentials
- clicking through the HTML manually in a browser, although the frontend fetch targets match the working backend endpoints

## Project Layout

```text
RiskTrace-Engine-main/
├── agents.py
├── build_graph.py
├── csv_db.py
├── data/
│   └── processed/
├── evaluate.py
├── issuegraph_ui.html
├── main.py
├── preprocess.py
├── requirements.txt
└── risk_engine.py
```

## What Works Together

Frontend requests in [issuegraph_ui.html](/Users/sreejith/Downloads/RiskTrace-Engine-main/issuegraph_ui.html) call these backend endpoints:

- `GET /health`
- `GET /dashboard`
- `GET /alerts`
- `GET /risk/{issue_id}`
- `POST /query`
- `POST /counterfactual/{issue_id}`

Those routes exist in [main.py](/Users/sreejith/Downloads/RiskTrace-Engine-main/main.py), so the frontend and backend are integrated at the API-contract level.

## Dataset Support

This repo supports two input paths:

1. Synthetic demo data
2. Real Apache Jira data from Zenodo

Zenodo record:

- [Apache Jira Issue Tracking Dataset](https://zenodo.org/records/7740379)

Recommended download from that record:

- `issues.bson.gz` or `issues.bson`

The preprocessor now supports:

- `.json`
- `.jsonl`
- `.csv`
- `.bson`
- `.bson.gz`

## Quick Start

### Option A: Run the app right now with the included processed CSV data

This is the easiest path and does not require Neo4j.

1. Create a local virtual environment:

```bash
python3 -m venv .venv
```

2. Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

3. Start the backend:

```bash
.venv/bin/uvicorn main:app --reload --port 8000
```

4. Open the frontend:

- open [issuegraph_ui.html](/Users/sreejith/Downloads/RiskTrace-Engine-main/issuegraph_ui.html) in your browser
- when prompted for the API base URL, enter:

```text
http://localhost:8000
```

### Option B: Regenerate processed CSV data from the real Zenodo dataset

1. Download `issues.bson.gz` or `issues.bson` from the Zenodo record.

2. Run preprocessing:

```bash
.venv/bin/python preprocess.py --input /absolute/path/to/issues.bson --project HADOOP --max-issues 300
```

This writes:

- `data/processed/issues.csv`
- `data/processed/dependencies.csv`
- `data/processed/stats.json`

3. Start the backend:

```bash
.venv/bin/uvicorn main:app --reload --port 8000
```

4. Open [issuegraph_ui.html](/Users/sreejith/Downloads/RiskTrace-Engine-main/issuegraph_ui.html) and connect to `http://localhost:8000`.

### Option C: Run with Neo4j

1. Start your local Neo4j database.

2. Create a `.env` file in the project root:

```env
USE_NEO4J=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
MONITOR_INTERVAL=60

# Optional
# GROQ_API_KEY=your_groq_api_key_here
```

3. Prepare processed CSVs from either:

- the real Zenodo BSON file
- or synthetic data

Real data:

```bash
.venv/bin/python preprocess.py --input /absolute/path/to/issues.bson --project HADOOP --max-issues 300
```

Synthetic fallback:

```bash
.venv/bin/python preprocess.py --synthetic --max-issues 300
```

4. Load the processed graph into Neo4j:

```bash
.venv/bin/python build_graph.py
```

5. Start the backend:

```bash
.venv/bin/uvicorn main:app --reload --port 8000
```

6. Open [issuegraph_ui.html](/Users/sreejith/Downloads/RiskTrace-Engine-main/issuegraph_ui.html) and connect to `http://localhost:8000`.

## Recommended `.env` Files

### CSV mode

```env
USE_NEO4J=false
MONITOR_INTERVAL=60

# Optional
# GROQ_API_KEY=your_groq_api_key_here
```

### Neo4j mode

```env
USE_NEO4J=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
MONITOR_INTERVAL=60

# Optional
# GROQ_API_KEY=your_groq_api_key_here
```

## End-to-End Run Guide

If you want the shortest successful path, do exactly this from the project root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cat > .env <<'EOF'
USE_NEO4J=false
MONITOR_INTERVAL=60
EOF
.venv/bin/uvicorn main:app --reload --port 8000
```

Then:

- open [issuegraph_ui.html](/Users/sreejith/Downloads/RiskTrace-Engine-main/issuegraph_ui.html)
- connect it to `http://localhost:8000`
- click `Refresh Data`
- use `Risk Board`, `Action Plan`, and `What-If`

## API Endpoints

### Health

```bash
curl http://127.0.0.1:8000/health
```

Expected shape:

```json
{
  "status": "ok",
  "db_mode": "csv",
  "pipeline": true,
  "data_loaded": true
}
```

### Dashboard

```bash
curl http://127.0.0.1:8000/dashboard
```

Returns:

- summary cards
- top risky issues
- action plan
- LLM analysis if `GROQ_API_KEY` is set

### Counterfactual

```bash
curl -X POST http://127.0.0.1:8000/counterfactual/HADOOP-16 \
  -H "Content-Type: application/json" \
  -d '{"resolve_as":"Done"}'
```

Returns:

- `diff`
- `graph_nodes`
- `graph_edges`
- `impact_summary`

### Swagger docs

```text
http://localhost:8000/docs
```

## Frontend Usage

The frontend is a single HTML file and does not require a build step.

Features available in the UI:

- dashboard summary
- risk board
- action plan
- alerts fetch
- AI query drawer
- counterfactual what-if analysis

The frontend stores the API base URL in browser `localStorage` under `igraph_api`.

## MongoDB Option

If you restore the BSON dump into MongoDB, you can preprocess directly from MongoDB instead of repeatedly reading the raw BSON file from disk.

Example:

```bash
.venv/bin/python preprocess.py \
  --mongodb-uri mongodb://localhost:27017 \
  --mongodb-db apache_jira \
  --mongodb-collection issues \
  --project HADOOP \
  --max-issues 300
```

This is useful if:

- you already loaded the dump with `mongorestore`
- you want repeated preprocessing runs to be faster

## How to Use the Real Dataset

Example with the Hadoop slice:

```bash
.venv/bin/python preprocess.py \
  --input /absolute/path/to/issues.bson \
  --project HADOOP \
  --max-issues 300
```

Notes:

- `--project HADOOP` keeps the graph smaller and easier to inspect
- `--max-issues 300` is a good starting point for responsiveness
- you can swap `HADOOP` for another project key present in the dataset

## Synthetic Data Fallback

If you only want to check the app quickly:

```bash
.venv/bin/python preprocess.py --synthetic --max-issues 300
```

This generates a dependency graph with delays and blockers so the dashboard has meaningful risk signals.

## Troubleshooting

### `Pipeline not ready`

Cause:

- processed CSVs do not exist yet

Fix:

```bash
.venv/bin/python preprocess.py --synthetic
```

or run the real BSON preprocessing command.

### Port `8000` is already in use

Start on a different port:

```bash
.venv/bin/uvicorn main:app --reload --port 8001
```

Then connect the HTML dashboard to:

```text
http://localhost:8001
```

### Neo4j mode falls back to CSV mode

Cause:

- Neo4j is not running
- wrong credentials
- wrong `NEO4J_URI`

Fix:

- confirm Neo4j is started
- verify `.env`
- rerun `build_graph.py`

### `bson` import error during preprocessing

Cause:

- dependencies were not installed in the local venv

Fix:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### No LLM explanations in dashboard

Cause:

- `GROQ_API_KEY` not set

Fix:

Add this to `.env`:

```env
GROQ_API_KEY=your_key_here
```

The risk engine still works without it.

## Known Notes

- The app can run fully in CSV mode without Neo4j.
- Neo4j mode uses the same processed CSVs as the source for graph loading.
- The existing process already running on your `localhost:8000` during verification returned `500`, so if you see that behavior, stop that older process and restart using the commands in this README.

## Useful Commands

Rebuild synthetic data:

```bash
.venv/bin/python preprocess.py --synthetic --max-issues 300
```

Build from real BSON:

```bash
.venv/bin/python preprocess.py --input /absolute/path/to/issues.bson --project HADOOP --max-issues 300
```

Load into Neo4j:

```bash
.venv/bin/python build_graph.py
```

Run backend:

```bash
.venv/bin/uvicorn main:app --reload --port 8000
```

## Summary

Yes, the current updated backend is now starting correctly, and the frontend is wired to the right endpoints. The version I verified works in CSV mode. Neo4j mode should work once your local Neo4j instance is running and loaded with `build_graph.py`.
