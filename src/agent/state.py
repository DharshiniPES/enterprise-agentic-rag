from typing import List, Dict, Any, TypedDict, Optional
from ..retrieval.hybrid import ScoredChunk

class AgentExecutionEvent(TypedDict):
    node_name: str
    status: str
    message: str
    details: Dict[str, Any]

class AgentState(TypedDict):
    original_query: str
    current_query: str
    retrieved_chunks: List[ScoredChunk]
    relevance_score: float
    is_relevant: bool
    generation: str
    citations: List[Dict[str, Any]]
    hallucination_score: float
    is_grounded: bool
    iterations: int
    execution_trace: List[AgentExecutionEvent]
