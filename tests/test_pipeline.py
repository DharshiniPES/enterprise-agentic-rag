import pytest
from pathlib import Path

from src.config import RAW_DATA_DIR, BENCHMARK_DIR
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import SemanticTableChunker
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseVectorRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid import HybridSearchEngine
from src.agent.graph import EnterpriseRAGWorkflow
from src.evaluation.ragas_bench import RagasBenchmarkSuite

def test_ingestion_and_chunking():
    docs = DocumentParser.parse_directory(RAW_DATA_DIR)
    assert len(docs) >= 2, "Should parse at least Apple and Tesla sample files"

    chunker = SemanticTableChunker(target_chunk_size=400, overlap_size=50)
    chunks = chunker.chunk_document(docs[0])
    assert len(chunks) > 0
    # Ensure tables are preserved
    has_table = any(c.metadata.get("is_table") for c in chunks)
    assert has_table, "Table-aware chunker should identify markdown tables"

def test_sparse_bm25_retrieval():
    docs = DocumentParser.parse_directory(RAW_DATA_DIR)
    chunker = SemanticTableChunker()
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk_document(d))

    retriever = BM25Retriever(all_chunks)
    results = retriever.search("Megapack Lathrop 40 GWh", top_k=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert score > 0.0
    assert "megapack" in top_chunk.text.lower() or "lathrop" in top_chunk.text.lower()

def test_dense_vector_retrieval():
    docs = DocumentParser.parse_directory(RAW_DATA_DIR)
    chunker = SemanticTableChunker()
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk_document(d))

    retriever = DenseVectorRetriever()
    retriever.index(all_chunks)
    results = retriever.search("Apple quarterly net revenue growth", top_k=3)
    assert len(results) == 3
    top_chunk, score = results[0]
    assert score >= 0.0

def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker(top_k=2)
    query = "Apple Services gross margin"
    good_doc = "Services gross margin reached 74.0%, compared to 70.5% in the prior year period."
    bad_doc = "Automotive regulatory credits recognized reached an all-time high of $890 million."

    good_score = reranker.score_pair(query, good_doc)
    bad_score = reranker.score_pair(query, bad_doc)
    assert good_score > bad_score, "Cross-encoder should score relevant doc significantly higher"

def test_agentic_workflow_end_to_end():
    docs = DocumentParser.parse_directory(RAW_DATA_DIR)
    chunker = SemanticTableChunker()
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk_document(d))

    engine = HybridSearchEngine()
    engine.index(all_chunks)

    workflow = EnterpriseRAGWorkflow(engine)
    state = workflow.run("What was Apple's total quarterly revenue in Q3 FY2024?")

    assert state["generation"] != ""
    assert len(state["citations"]) > 0
    assert state["hallucination_score"] >= 0.80
    assert len(state["execution_trace"]) >= 4

def test_benchmark_suite():
    docs = DocumentParser.parse_directory(RAW_DATA_DIR)
    chunker = SemanticTableChunker()
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk_document(d))

    engine = HybridSearchEngine()
    engine.index(all_chunks)

    bench = RagasBenchmarkSuite(engine)
    comp = bench.run_benchmark_comparison()

    assert "naive_rag" in comp
    assert "agentic_rag" in comp
    assert comp["agentic_rag"]["faithfulness"] >= comp["naive_rag"]["faithfulness"]
