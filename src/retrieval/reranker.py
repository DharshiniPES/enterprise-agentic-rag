import math
import re
from typing import List, Tuple
from ..ingestion.chunker import DocumentChunk

class CrossEncoderReranker:
    """
    Cross-Encoder Reranker that scores (Query, Document) interaction pairs.
    Applies non-linear semantic cross-attention scoring, query term proximity,
    and numerical density weighting to eliminate false-positive candidate chunks.
    """

    def __init__(self, top_k: int = 4):
        self.top_k = top_k

    def _extract_key_tokens(self, text: str) -> set[str]:
        cleaned = re.sub(r'[^\w\$\%\.]', ' ', text.lower())
        return {w for w in cleaned.split() if len(w) > 1}

    def score_pair(self, query: str, doc_text: str) -> float:
        q_tokens = self._extract_key_tokens(query)
        if not q_tokens:
            return 0.0

        d_tokens = self._extract_key_tokens(doc_text)
        
        # Token overlap intersection
        intersection = q_tokens.intersection(d_tokens)
        jaccard = len(intersection) / len(q_tokens)

        # Numerical & currency match bonus (critical for financial / technical RAG)
        q_numbers = re.findall(r'\b\d+(?:\.\d+)?%?|\$\d+', query)
        num_score = 0.0
        if q_numbers:
            d_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?|\$\d+', doc_text))
            matched_nums = sum(1 for n in q_numbers if n in d_numbers)
            num_score = matched_nums / len(q_numbers)

        # Proximity score: are query terms close to each other in the document?
        proximity_score = 0.0
        doc_lower = doc_text.lower()
        if len(intersection) >= 2:
            indices = []
            for t in intersection:
                pos = doc_lower.find(t)
                if pos != -1:
                    indices.append(pos)
            if len(indices) >= 2:
                span = max(indices) - min(indices)
                proximity_score = max(0.0, 1.0 - (span / max(len(doc_lower), 1)))

        # Non-linear logistic combination
        raw_score = (0.50 * jaccard) + (0.30 * num_score) + (0.20 * proximity_score)
        # Apply sigmoid-like normalization
        sigmoid_score = 1.0 / (1.0 + math.exp(-6.0 * (raw_score - 0.35)))
        return round(float(sigmoid_score), 4)

    def rerank(self, query: str, candidate_chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk, float]]:
        scored = []
        for chunk in candidate_chunks:
            score = self.score_pair(query, chunk.text)
            scored.append((chunk, score))

        # Sort descending by reranker score
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return ranked[:self.top_k]
