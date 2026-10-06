import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

class PipelineConfig(BaseModel):
    # LLM Settings
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "offline"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 80

    # Retrieval
    dense_top_k: int = 12
    sparse_top_k: int = 12
    reranker_top_k: int = 4
    rrf_k: int = 60  # Constant for Reciprocal Rank Fusion

    # Agent Loop
    relevance_threshold: float = 0.60
    max_agent_iterations: int = 2
    enable_hallucination_guard: bool = True

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

config = PipelineConfig()
