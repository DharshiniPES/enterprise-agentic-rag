from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ..ingestion.chunker import DocumentChunk
from .dense import DenseVectorRetriever
from .sparse import BM25Retriever
from .reranker import CrossEncoderReranker

class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    dense_rank: int = -1
    sparse_rank: int = -1
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HybridSearchEngine:
    """
    Two-stage Hybrid Information Retrieval Engine:
    Stage 1: Multi-index candidate retrieval (Dense Semantic + Sparse BM25) fused via Reciprocal Rank Fusion (RRF).
    Stage 2: Precision Cross-Encoder Reranking to eliminate out-of-context documents.
    """

    def __init__(self, rrf_k: int = 60, reranker_top_k: int = 4):
        self.dense_retriever = DenseVectorRetriever()
        self.sparse_retriever = BM25Retriever()
        self.reranker = CrossEncoderReranker(top_k=reranker_top_k)
        self.rrf_k = rrf_k
        self.all_chunks: List[DocumentChunk] = []

    def index(self, chunks: List[DocumentChunk]):
        self.all_chunks = chunks
        self.dense_retriever.index(chunks)
        self.sparse_retriever.index(chunks)

    def search(
        self,
        query: str,
        dense_top_k: int = 12,
        sparse_top_k: int = 12,
        apply_reranker: bool = True
    ) -> List[ScoredChunk]:
        if not self.all_chunks:
            return []

        # 1. Retrieve candidates from both systems
        dense_results = self.dense_retriever.search(query, top_k=dense_top_k)
        sparse_results = self.sparse_retriever.search(query, top_k=sparse_top_k)

        # 2. Reciprocal Rank Fusion (RRF)
        chunk_map: Dict[str, DocumentChunk] = {}
        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        for rank, (chunk, _score) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, (chunk, _score) in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            sparse_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort candidates by combined RRF score
        sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_chunks = [chunk_map[cid] for cid, _ in sorted_by_rrf[:max(dense_top_k, sparse_top_k)]]

        if not apply_reranker:
            return [
                ScoredChunk(
                    chunk=chunk_map[cid],
                    dense_rank=dense_ranks.get(cid, -1),
                    sparse_rank=sparse_ranks.get(cid, -1),
                    rrf_score=round(score, 6),
                    rerank_score=round(score * 100, 4)
                )
                for cid, score in sorted_by_rrf[:self.reranker.top_k]
            ]

        # 3. Stage 2: Cross-Encoder Reranking
        reranked_pairs = self.reranker.rerank(query, candidate_chunks)

        scored_results: List[ScoredChunk] = []
        for chunk, rerank_score in reranked_pairs:
            cid = chunk.chunk_id
            scored_results.append(
                ScoredChunk(
                    chunk=chunk,
                    dense_rank=dense_ranks.get(cid, -1),
                    sparse_rank=sparse_ranks.get(cid, -1),
                    rrf_score=round(rrf_scores.get(cid, 0.0), 6),
                    rerank_score=round(rerank_score, 4)
                )
            )

        return scored_results
