import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from ..ingestion.chunker import DocumentChunk

class BM25Retriever:
    """
    Lexical BM25 retriever for exact keyword matching, tickers, and acronyms.
    """

    def __init__(self, chunks: List[DocumentChunk] = None):
        self.chunks: List[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: List[List[str]] = []
        if chunks:
            self.index(chunks)

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, retain alphanumeric tokens and key financial symbols
        clean_text = re.sub(r'[^\w\s\$\%\.]', ' ', text.lower())
        tokens = [t for t in clean_text.split() if len(t) > 1 or t in ['$', '%']]
        return tokens

    def index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        scored_pairs = list(zip(self.chunks, doc_scores))
        
        # Sort by BM25 score descending
        ranked = sorted(scored_pairs, key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
