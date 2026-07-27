# Team Contributions

Team Member 1 owned the graph and risk engine: preprocessing Jira-style issue data, building dependency edges, implementing the temporal risk propagation algorithm in `risk_engine.py`, and integrating graph traversal through `GraphReasoningAgent`.

Team Member 2 owned the GenAI pipeline: semantic RAG retrieval with ChromaDB and `all-MiniLM-L6-v2`, Groq LLaMA 3.3 70B prompt design in `DecisionAgent`, the critic layer for hallucination checks, and graceful keyword fallback when RAG dependencies are unavailable.

Team Member 3 owned evaluation and presentation: the FastAPI endpoints, dashboard workflow, evaluation harness and baselines, counterfactual demo flow, validation artifacts, and documentation needed for the viva and academic submission.
