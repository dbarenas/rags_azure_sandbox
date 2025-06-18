from .openai_client import get_embedding, get_completion
from .search_client import search_similar_documents # Assuming AZURE_SEARCH_KEY is set for it to load
from .memory_cache import search_cache, save_to_cache
from .rag_pipeline import construct_prompt # The newly public function
from .ragas_logging import log_ragas_entry
import os # For testing environment variable checks

def run_cag_workflow(user_query: str, query_embedding: list[float]) -> str | None:
    """
    Checks cache for a response.
    Returns cached response if found, otherwise None.
    """
    cached_response = search_cache(query_embedding)
    if cached_response:
        return f"[CAG - cached]\n{cached_response}"
    return None

def run_rag_core_workflow(user_query: str, query_embedding: list[float]) -> tuple[str, str]:
    """
    Runs the core RAG steps: search, prompt construction, LLM call.
    Returns the generated response and the context used.
    """
    retrieved_docs = search_similar_documents(query_embedding)
    ragas_context = "\n---\n".join(retrieved_docs)

    prompt = construct_prompt(user_query, retrieved_docs) # Use the public function
    response = get_completion(prompt)
    return response, ragas_context

def execute_full_rag_cag_pipeline(user_query: str) -> str:
    """
    Orchestrates the full RAG+CAG pipeline.
    """
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        return "[ERROR - Workflow] Failed to generate query embedding."

    # Try CAG first
    cached_result = run_cag_workflow(user_query, query_embedding)
    if cached_result:
        return cached_result

    # If not cached, run RAG
    response, ragas_context = run_rag_core_workflow(user_query, query_embedding)

    # Save to cache
    save_to_cache(user_query, query_embedding, response)

    # Log for RAGAS
    log_ragas_entry(user_query, ragas_context, response)

    return f"[RAG - generated]\n{response}"

if __name__ == "__main__":
    print("Testing workflow orchestrator (requires configured .env and indexed data)...")

    # Check for necessary environment variables to run tests
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
    SEARCH_API_KEY = os.getenv("AZURE_SEARCH_KEY")

    if not OPENAI_API_KEY or not SEARCH_API_KEY:
        print("Azure OpenAI or Search keys not configured in .env. Skipping orchestrator tests.")
    else:
        print("\n--- Testing execute_full_rag_cag_pipeline ---")
        test_query_orchestrator = "What is Azure Cognitive Search?"

        print(f"\nRunning full pipeline for query: '{test_query_orchestrator}' (1st time - expect RAG)")
        response_orch_1 = execute_full_rag_cag_pipeline(test_query_orchestrator)
        print(f"Response 1:\n{response_orch_1}")
        assert "[RAG - generated]" in response_orch_1

        print(f"\nRunning full pipeline for query: '{test_query_orchestrator}' (2nd time - expect CAG)")
        response_orch_2 = execute_full_rag_cag_pipeline(test_query_orchestrator)
        print(f"Response 2:\n{response_orch_2}")
        assert "[CAG - cached]" in response_orch_2

        print("\nOrchestrator tests complete.")
        print(f"Check '.cache/ragas_log.jsonl' for logged entries.")
