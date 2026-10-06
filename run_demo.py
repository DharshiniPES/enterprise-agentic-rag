#!/usr/bin/env python3
"""
Enterprise Agentic RAG Platform - Interactive Runner
Usage:
  python run_demo.py          # Interactive CLI demo with benchmark & query execution
  python run_demo.py --ui     # Launches Streamlit Web Dashboard
  python run_demo.py --api    # Launches FastAPI REST API Server
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

# Configure utf-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, config
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import SemanticTableChunker
from src.retrieval.hybrid import HybridSearchEngine
from src.agent.graph import EnterpriseRAGWorkflow
from src.evaluation.ragas_bench import RagasBenchmarkSuite

def print_banner():
    banner = """
========================================================================
    [AI] ENTERPRISE AGENTIC RAG PLATFORM (CRAG & SELF-RAG)
    Hybrid Retrieval (Dense + BM25) | RRF | Cross-Encoder Reranking
========================================================================
"""
    print(banner)

def run_cli_demo():
    print_banner()
    print("[*] [1/3] Ingesting enterprise filings (Apple Q3 & Tesla Q2)...")
    docs = DocumentParser.parse_directory(RAW_DATA_DIR)
    chunker = SemanticTableChunker(target_chunk_size=config.chunk_size, overlap_size=config.chunk_overlap)
    
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk_document(d))
    print(f"[OK] Ingestion complete: {len(docs)} documents parsed into {len(all_chunks)} table-aware chunks.")

    print("\n[*] [2/3] Initializing Multi-Index Hybrid Search (Dense + BM25 + Cross-Encoder)...")
    engine = HybridSearchEngine(rrf_k=config.rrf_k, reranker_top_k=config.reranker_top_k)
    engine.index(all_chunks)
    workflow = EnterpriseRAGWorkflow(engine)
    print("[OK] Hybrid Index ready.")

    sample_query = "What was Apple's total quarterly revenue in Q3 FY2024 and what was the year-over-year percentage increase?"
    print(f"\n[>] [3/3] Executing Live Agentic Workflow on Sample Query:")
    print(f"    Query: '{sample_query}'\n")

    t0 = time.perf_counter()
    state = workflow.run(sample_query)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    print("--- [Agent State Graph Execution Trace] ---")
    for idx, event in enumerate(state["execution_trace"], start=1):
        status_symbol = "[OK]" if event["status"] in ["completed", "verified"] else "[!]"
        print(f"  {status_symbol} Step {idx} [{event['node_name']}]: {event['message']}")

    print("\n--- [Grounded Synthesized Output] ---")
    print(state["generation"])

    print("\n--- [Quality & Hallucination Guardrails] ---")
    print(f"  • Grounding Confidence (NLI): {round(state['hallucination_score']*100, 1)}%")
    print(f"  • Total Pipeline Latency:      {duration_ms} ms")
    print(f"  • Citations Attached:          {len(state['citations'])} sources verified")

    print("\n" + "="*72)
    print(">> Launching interactive web dashboard? Run:")
    print("   streamlit run ui/app.py")
    print("="*72)

def main():
    parser = argparse.ArgumentParser(description="Enterprise Agentic RAG Platform")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit Web UI")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI server")
    args = parser.parse_args()

    if args.ui:
        print("🚀 Starting Streamlit Web Dashboard at http://localhost:8501 ...")
        subprocess.run(["streamlit", "run", "ui/app.py"])
    elif args.api:
        print("🚀 Starting FastAPI Server at http://localhost:8000 ...")
        subprocess.run(["uvicorn", "src.api.server:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])
    else:
        run_cli_demo()

if __name__ == "__main__":
    main()
