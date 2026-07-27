# Novelty Statement

The novel part of RiskTrace Engine is the counterfactual/interventional reasoning layer over software dependency graphs. The system does not merely identify risky issues; it asks what would happen if a selected upstream issue were resolved, clones the graph in memory, recomputes propagated risk, and returns downstream before/after deltas. Based on our literature scan, this explicit what-if framing is uncommon in issue-tracker risk tools, which usually focus on prediction, prioritization, or dashboarding rather than intervention analysis.

What is not claimed as novel is just as important. The risk propagation formula uses established graph-propagation ideas with temporal decay. The RAG pipeline uses known components: ChromaDB, `all-MiniLM-L6-v2` sentence embeddings, and LLM summarization. The multi-agent architecture is also a well-executed composition of known roles: perception, graph reasoning, planning, decision, monitoring, and critique.

The research contribution is therefore the combination of deterministic temporal risk propagation with counterfactual simulation for software project dependency graphs. The engineering contribution is making that idea usable through graceful CSV/Neo4j backends, semantic retrieval with keyword fallback, LLM explanations with structured prompts, and a dashboard/demo flow suitable for viva presentation.
