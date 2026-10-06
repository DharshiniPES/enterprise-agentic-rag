import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from ..ingestion.chunker import DocumentChunk

class DenseVectorRetriever:
    """
    Dense semantic retriever utilizing latent semantic indexing (LSI / Dense subspace embeddings)
    with cosine similarity for continuous vector search.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
            sublinear_tf=True
        )
        self.svd: TruncatedSVD | None = None
        self.chunk_embeddings: np.ndarray | None = None

    def index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not chunks:
            return

        corpus = [c.text for c in chunks]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        n_samples, n_features = tfidf_matrix.shape
        n_components = min(self.embedding_dim, n_samples - 1, n_features - 1)

        if n_components > 2:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            dense_vectors = self.svd.fit_transform(tfidf_matrix)
        else:
            dense_vectors = tfidf_matrix.toarray()

        # L2-normalize vectors for fast cosine similarity via dot product
        norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.chunk_embeddings = dense_vectors / norms

    def embed_query(self, query: str) -> np.ndarray:
        if self.svd is None or self.chunk_embeddings is None:
            tfidf_q = self.vectorizer.transform([query]).toarray()
        else:
            tfidf_q = self.vectorizer.transform([query])
            tfidf_q = self.svd.transform(tfidf_q)

        norm = np.linalg.norm(tfidf_q)
        if norm > 0:
            tfidf_q = tfidf_q / norm
        return tfidf_q

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if self.chunk_embeddings is None or not self.chunks:
            return []

        q_vec = self.embed_query(query)
        # Cosine similarity is the dot product of normalized vectors
        scores = np.dot(self.chunk_embeddings, q_vec.T).flatten()

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            results.append((self.chunks[idx], max(0.0, score)))

        return results
