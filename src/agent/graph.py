import time
from typing import Dict, Any
from .state import AgentState
from .nodes import AgentNodes
from ..retrieval.hybrid import HybridSearchEngine
from ..config import config

class EnterpriseRAGWorkflow:
    """
    Self-Corrective Agentic State Graph (CRAG / Self-RAG).
    Orchestrates cyclic retrieval, grading, query rewrites, and hallucination checks.
    """

    def __init__(self, search_engine: HybridSearchEngine):
        self.nodes = AgentNodes(search_engine)

    def run(self, query: str) -> AgentState:
        # Initialize state
        state: AgentState = {
            "original_query": query,
            "current_query": query,
            "retrieved_chunks": [],
            "relevance_score": 0.0,
            "is_relevant": False,
            "generation": "",
            "citations": [],
            "hallucination_score": 0.0,
            "is_grounded": False,
            "iterations": 0,
            "execution_trace": []
        }

        # 1. First Retrieval
        state = self.nodes.retrieve_node(state)

        # 2. Grade retrieved documents
        state = self.nodes.grade_documents_node(state)

        # 3. Corrective loop if documents are not relevant and within max iterations
        while not state["is_relevant"] and state["iterations"] < config.max_agent_iterations:
            state = self.nodes.rewrite_query_node(state)
            state = self.nodes.retrieve_node(state)
            state = self.nodes.grade_documents_node(state)

        # 4. Generate grounded answer
        state = self.nodes.generate_node(state)

        # 5. Hallucination Guardrail Check
        if config.enable_hallucination_guard:
            state = self.nodes.hallucination_guard_node(state)
        else:
            state["is_grounded"] = True
            state["hallucination_score"] = 1.0

        return state
