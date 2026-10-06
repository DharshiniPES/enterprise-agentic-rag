import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import config, RAW_DATA_DIR
from ..ingestion.parser import DocumentParser, ParsedDocument
from ..ingestion.chunker import SemanticTableChunker, DocumentChunk
from ..retrieval.hybrid import HybridSearchEngine
from ..agent.graph import EnterpriseRAGWorkflow
from ..evaluation.ragas_bench import RagasBenchmarkSuite

app = FastAPI(
    title="Enterprise Agentic RAG Platform API",
    description="Production-grade Self-Corrective Hybrid RAG with Ragas Evaluation & Hallucination Guardrails",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory state
search_engine = HybridSearchEngine(rrf_k=config.rrf_k, reranker_top_k=config.reranker_top_k)
workflow: Optional[EnterpriseRAGWorkflow] = None
chunker = SemanticTableChunker(target_chunk_size=config.chunk_size, overlap_size=config.chunk_overlap)
benchmark_suite: Optional[RagasBenchmarkSuite] = None
docs_metadata: List[Dict[str, Any]] = []

def initialize_knowledge_base():
    global workflow, benchmark_suite, docs_metadata
    parsed_docs = DocumentParser.parse_directory(RAW_DATA_DIR)
    all_chunks: List[DocumentChunk] = []

    docs_metadata = []
    for doc in parsed_docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        docs_metadata.append({
            "source_name": doc.source_name,
            "title": doc.metadata.get("title", doc.source_name),
            "chunks_count": len(chunks),
            "char_count": doc.metadata.get("character_count", 0)
        })

    search_engine.index(all_chunks)
    workflow = EnterpriseRAGWorkflow(search_engine)
    benchmark_suite = RagasBenchmarkSuite(search_engine)

# Pre-initialize knowledge base on import
initialize_knowledge_base()

@app.on_event("startup")
def startup_event():
    if not workflow:
        initialize_knowledge_base()

# Request & Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., example="What was Apple's total quarterly revenue in Q3 FY2024?")
    use_agentic_workflow: bool = Field(default=True, description="Enable Self-RAG agent loop")

class QueryResponse(BaseModel):
    query: str
    generation: str
    relevance_score: float
    hallucination_score: float
    is_grounded: bool
    iterations: int
    citations: List[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]
    latency_ms: float

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "indexed_chunks": len(search_engine.all_chunks),
        "documents": docs_metadata,
        "config": {
            "dense_top_k": config.dense_top_k,
            "sparse_top_k": config.sparse_top_k,
            "reranker_top_k": config.reranker_top_k,
            "relevance_threshold": config.relevance_threshold,
            "provider": config.llm_provider
        }
    }

@app.post("/api/v1/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not workflow:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    start_t = time.perf_counter()

    if req.use_agentic_workflow:
        state = workflow.run(req.query)
    else:
        # Naive single step
        scored_chunks = search_engine.search(req.query, dense_top_k=3, sparse_top_k=0, apply_reranker=False)
        context = "\n".join([sc.chunk.text for sc in scored_chunks])
        gen = f"Direct Naive Retrieval:\n{context[:300]}..."
        state = {
            "original_query": req.query,
            "generation": gen,
            "relevance_score": 0.5,
            "hallucination_score": 0.7,
            "is_grounded": True,
            "iterations": 0,
            "citations": [],
            "execution_trace": []
        }

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    return QueryResponse(
        query=req.query,
        generation=state["generation"],
        relevance_score=state["relevance_score"],
        hallucination_score=state["hallucination_score"],
        is_grounded=state["is_grounded"],
        iterations=state["iterations"],
        citations=state["citations"],
        execution_trace=state["execution_trace"],
        latency_ms=elapsed_ms
    )

@app.get("/api/v1/benchmark")
def run_benchmark():
    if not benchmark_suite:
        raise HTTPException(status_code=503, detail="Benchmark suite not initialized")
    return benchmark_suite.run_benchmark_comparison()
