from agents import RAGRetriever, _RAG_AVAILABLE
from risk_engine import IssueNode, RiskResult

print("RAG available:", _RAG_AVAILABLE)

# Use realistic risk results too — this is what happens in production
from dataclasses import dataclass

nodes = [
    IssueNode(
        issue_id="HADOOP-1", project="proj",
        summary="MapReduce job is overdue and blocking release",
        status="Blocked", priority="Critical", assignee="alice",
        due_date=None, updated=None, delay_days=14.0, is_delayed=True
    ),
    IssueNode(
        issue_id="HADOOP-2", project="proj",
        summary="Documentation update",
        status="Open", priority="Low", assignee="bob",
        due_date=None, updated=None, delay_days=0.0, is_delayed=False
    ),
    IssueNode(
        issue_id="HADOOP-3", project="proj",
        summary="Unit tests failing on CI pipeline, build is late",
        status="In Progress", priority="High", assignee="carol",
        due_date=None, updated=None, delay_days=5.0, is_delayed=True
    ),
    IssueNode(
        issue_id="HADOOP-4", project="proj",
        summary="Upgrade dependency versions",
        status="Open", priority="Medium", assignee="dave",
        due_date=None, updated=None, delay_days=0.0, is_delayed=False
    ),
    IssueNode(
        issue_id="HADOOP-5", project="proj",
        summary="Deployment blocked, release deadline missed by 3 weeks",
        status="Blocked", priority="Critical", assignee="eve",
        due_date=None, updated=None, delay_days=21.0, is_delayed=True
    ),
]

rag = RAGRetriever()
print(f"Running in {'SEMANTIC' if getattr(rag, '_ready', False) else 'KEYWORD FALLBACK'} mode")
rag.build_index(nodes, {})

node_map = {n.issue_id: n for n in nodes}

print("\n--- Test 1: 'tasks behind schedule' ---")
results = rag.retrieve("tasks behind schedule", node_map, k=3)
for i, n in enumerate(results):
    print(f"  #{i+1}: {n.issue_id} | at_risk={n.is_at_risk()} | {n.summary}")
top = results[0].issue_id if results else None
print("PASS" if top in ("HADOOP-1","HADOOP-3","HADOOP-5") else f"FAIL — got {top}, expected a delayed issue")

print("\n--- Test 2: 'blocked critical issues' ---")
results = rag.retrieve("blocked critical issues", node_map, k=3)
for i, n in enumerate(results):
    print(f"  #{i+1}: {n.issue_id} | status={n.status} | {n.summary}")
top = results[0].issue_id if results else None
print("PASS" if top in ("HADOOP-1","HADOOP-5") else f"FAIL — got {top}, expected a Blocked issue")

print("\n--- Test 3: 'routine low priority tasks' ---")
results = rag.retrieve("routine low priority tasks", node_map, k=3)
for i, n in enumerate(results):
    print(f"  #{i+1}: {n.issue_id} | priority={n.priority} | {n.summary}")
top = results[0].issue_id if results else None
print("PASS" if top in ("HADOOP-2","HADOOP-4") else f"FAIL — got {top}, expected a low-risk issue")
