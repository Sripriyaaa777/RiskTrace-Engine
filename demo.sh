#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
UVICORN_BIN="${UVICORN_BIN:-uvicorn}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [ -x ".venv/bin/uvicorn" ]; then
  UVICORN_BIN=".venv/bin/uvicorn"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python3 is required to run the demo."
  exit 1
fi

if ! command -v "${UVICORN_BIN}" >/dev/null 2>&1; then
  echo "uvicorn is not installed. Install dependencies with: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"
SERVER_PID=""

cleanup() {
  if [ -n "${SERVER_PID}" ]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

health_check() {
  "${PYTHON_BIN}" - "$BASE_URL" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1]
try:
    with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
    raise SystemExit(0 if payload.get("data_loaded") else 1)
except Exception:
    raise SystemExit(1)
PY
}

if health_check; then
  echo "Using existing RiskTrace server at ${BASE_URL}"
else
  echo "Starting RiskTrace server at ${BASE_URL} in CSV mode"
  USE_NEO4J="${USE_NEO4J:-false}" "${UVICORN_BIN}" main:app --host "${HOST}" --port "${PORT}" > /tmp/risktrace-demo.log 2>&1 &
  SERVER_PID="$!"

  for _ in $(seq 1 30); do
    if health_check; then
      break
    fi
    sleep 1
  done

  if ! health_check; then
    echo "Server did not become healthy. Last log lines:"
    tail -40 /tmp/risktrace-demo.log || true
    exit 1
  fi
fi

"${PYTHON_BIN}" - "$BASE_URL" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base_url = sys.argv[1]

def request(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))

print("\n=== Health ===")
print(json.dumps(request("GET", "/health"), indent=2))

queries = [
    "Which blocked tasks are causing the most downstream risk?",
    "What caused HADOOP-16 to become risky?",
    "Show me tasks behind schedule",
    "Which issue should we resolve first?",
]

for query in queries:
    print(f"\n=== Query: {query} ===")
    try:
        payload = request("POST", "/query", {"query": query})
        data = payload.get("data", {})
        llm = data.get("llm_analysis", {}) if isinstance(data.get("llm_analysis"), dict) else {}
        summary = llm.get("summary") or data.get("summary") or data.get("message") or data.get("explanation")
        print("Summary:", summary)
        for item in data.get("top_risks", [])[:3]:
            print(f"- {item.get('issue_id')}: {item.get('risk_level')} ({item.get('risk_score')})")
        for item in data.get("action_plan", [])[:2]:
            print(f"* Action: {item.get('issue_id')} -> {item.get('action')} ({item.get('rationale')})")
        for item in llm.get("recommendations", [])[:2]:
            print(f"* Recommendation: {item}")
    except urllib.error.HTTPError as exc:
        print(f"Query failed gracefully with HTTP {exc.code}: {exc.read().decode('utf-8')[:300]}")

print("\n=== Counterfactual: resolve KAFKA-23 ===")
try:
    payload = request("POST", "/counterfactual/KAFKA-23", {"resolve_as": "Done"})
    data = payload.get("data", {})
    print(json.dumps(data.get("impact_summary", data), indent=2))
    for item in data.get("diff", [])[:3]:
        print(f"- {item.get('issue_id')}: {item.get('before_score')} -> {item.get('after_score')} (delta {item.get('delta')})")
except urllib.error.HTTPError as exc:
    print(f"Counterfactual failed gracefully with HTTP {exc.code}: {exc.read().decode('utf-8')[:300]}")

print("\nDemo complete.")
PY
