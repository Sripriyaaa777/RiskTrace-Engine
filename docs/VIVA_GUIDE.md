# RiskTrace Engine Viva Guide

## 30-Second Explanation

RiskTrace Engine is a GenAI-assisted risk intelligence system for software issue trackers. It builds a dependency graph from Jira-style issues, propagates delay and blocker risk downstream, retrieves relevant issues with semantic RAG, and uses an LLM layer to explain root causes and mitigations. Its novel feature is counterfactual what-if analysis: resolving one issue is simulated in memory and the system recomputes downstream risk to show what would improve.

## Why GenAI Is Justified

A normal dashboard can show sorted risk scores, but it cannot easily answer natural-language project questions such as "which blocked tasks are causing downstream risk?" or "what caused HADOOP-16 to become risky?" RiskTrace keeps risk scoring deterministic, then uses GenAI for the language-heavy layer: semantic retrieval, graph-aware explanations, and recommendations. This makes GenAI useful without letting the LLM invent scores or dependencies.

## What Is Novel

The strongest originality claim is the counterfactual/interventional layer over a software dependency graph. The system does not only predict or rank risk; it asks what happens if a specific upstream issue is resolved, clones the graph in memory, reruns propagation, and reports downstream before/after risk deltas.

The risk formula, RAG stack, and multi-agent structure are not claimed as original research by themselves. They are established techniques applied carefully to this domain.

## Important Technical Choices

- Risk propagation is deterministic and graph-based, so scores are explainable and reproducible.
- RAG uses ChromaDB and `all-MiniLM-L6-v2` when available, with keyword fallback for offline demos.
- Groq LLaMA 3.3 70B is used only for explanation/recommendation, not for computing risk.
- CSV mode works without Neo4j, so the demo remains functional on a basic laptop.
- Counterfactual simulation makes no database writes; it operates on an in-memory clone.

## Known Limitations To State Honestly

- The included real-Hadoop CSV loads as 1,500 parser-visible issue records in this project copy.
- Most real-Hadoop dependency edges are inferred, not explicit Jira links.
- The 100-edge dependency validation sample uses documented heuristic labels, not fully hand-adjudicated ground truth.
- Full LLM answers require a working Groq API key and network access; otherwise the system returns structured fallback output.
- Neo4j mode is supported by code, but CSV mode is the safest live-demo path unless Neo4j is already configured.

## Demo Path

1. Run `./demo.sh`.
2. Show `/health` returning CSV mode with data loaded.
3. Show one `/query` result with top risks and action plan.
4. Show `/counterfactual/KAFKA-23`, which reports improved downstream nodes and one high-risk reduction.
5. Mention graceful fallback if Groq or Neo4j is unavailable.

## Likely Questions

**Why not just use an LLM directly?**  
Because the LLM should not invent risk scores. RiskTrace computes scores deterministically from graph structure and delays, then uses the LLM only to explain and recommend.

**What is the main innovation?**  
Counterfactual what-if simulation over software dependency graphs.

**What happens if RAG dependencies or model files are unavailable?**  
`RAGRetriever` falls back to deterministic keyword matching, so the system still works.

**What happens if Groq is unavailable?**  
The API still returns structured risk data, action plans, and fallback summaries.

**Which part should each teammate explain?**  
One member should cover preprocessing/graph/risk propagation, one should cover RAG/LLM/agents, and one should cover API/dashboard/evaluation/demo.
