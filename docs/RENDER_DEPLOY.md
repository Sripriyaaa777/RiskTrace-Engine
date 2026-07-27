# Render Deployment

RiskTrace can run as a single Render web service. FastAPI serves both the API and `issuegraph_ui.html`.

## Settings

- Repository: `git@github.com:Sripriyaaa777/RiskTrace-Engine.git`
- Branch: `graph-visualization`
- Build command: `pip install -r requirements-render.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## Environment Variables

- `USE_NEO4J=false`
- `MONITOR_INTERVAL=120`
- `RISKTRACE_ALLOW_MODEL_DOWNLOAD=false`
- `GROQ_API_KEY=<your Groq API key>`

The Render deploy uses `requirements-render.txt` to keep the build light. Semantic RAG dependencies are intentionally omitted there, so hosted retrieval falls back to deterministic keyword matching. The full local environment can still use `requirements.txt` for ChromaDB and sentence-transformer retrieval.

## URLs After Deploy

- Dashboard: `https://<render-service>.onrender.com/`
- API health: `https://<render-service>.onrender.com/health`
- Swagger docs: `https://<render-service>.onrender.com/docs`
