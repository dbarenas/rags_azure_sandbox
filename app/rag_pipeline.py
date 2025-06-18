from .openai_client import get_embedding, get_completion
from .search_client import search_similar_documents
from .memory_cache import search_cache, save_to_cache
from .ragas_logging import log_ragas_entry

def run_rag_pipeline(user_query: str):
    """
    Runs the full RAG pipeline:
    1. Tries to find a response in the cache.
    2. If not cached, generates an embedding for the user query.
    3. Searches for similar documents in Azure Cognitive Search.
    4. Constructs a prompt with the retrieved context and user query.
    5. Gets a completion (answer) from Azure OpenAI.
    6. Saves the new query-response pair to the cache.
    7. Logs the interaction for RAGAS evaluation.
    Returns the response, prefixed with [CAG - cached] or [RAG - generated].
    """
    # For RAG, we first generate an embedding for the user query to search cache and vector store
    query_embedding = get_embedding(user_query)

    if not query_embedding:
        # This can happen if the OpenAI client failed to generate an embedding
        # Handle this gracefully, perhaps by falling back to a non-RAG response or error message
        print("Error: Could not generate embedding for the user query. Cannot proceed with RAG pipeline.")
        # You might want to return a specific error message to the bot/user
        return "[ERROR - RAG Pipeline] Failed to generate query embedding."


    # 1. Try to find a response in the cache using the query embedding
    # The search_cache function should handle similarity checks
    cached_response = search_cache(query_embedding) # Pass embedding to cache search
    if cached_response:
        # Optional: Log cache hit for RAGAS if desired, though RAGAS typically evaluates the RAG part.
        # log_ragas_entry(user_query, "Context from cache", cached_response) # Example if you want to log this
        return f"[CAG - cached]\n{cached_response}"

    # 2. If not cached, proceed with RAG
    # (Embedding for user_query was already generated)

    # 3. Search for similar documents in Azure Cognitive Search
    retrieved_docs = search_similar_documents(query_embedding) # Pass embedding to search

    # The search_similar_documents function now returns a list of strings,
    # where each string is "Source: ..., Content: ..., Metadata: ..."
    # We need to extract just the content for the prompt, but also keep context for RAGAS

    # For the LLM prompt, we typically just want the textual content of the documents
    context_for_prompt = "\n".join(retrieved_docs) # This now includes Source and Metadata in the context.
                                                   # You might want to refine this to extract only 'Content' part for the LLM prompt.
                                                   # For example:
                                                   # context_for_llm = "\n".join([doc.split("Content: ", 1)[1].split(", Metadata:",1)[0] if "Content: " in doc else doc for doc in retrieved_docs])

    # For RAGAS logging, the `retrieved_docs` (which includes source/metadata) is good as "context"
    ragas_context = "\n---\n".join(retrieved_docs)


    # 4. Construct a prompt with the retrieved context and user query
    # Using a refined context for LLM if you chose to extract only content:
    # prompt = f"Answer the question based on the context below:\n{context_for_llm}\n\nQuestion: {user_query}"
    # Using the direct output from search_similar_documents:
    prompt = f"Based on the following information, please answer the question.\n\nContext:\n{context_for_prompt}\n\nQuestion: {user_query}"


    # 5. Get a completion (answer) from Azure OpenAI
    response = get_completion(prompt)

    # 6. Save the new query-response pair to the cache
    # Pass the original user_query (text) and its embedding
    save_to_cache(user_query, query_embedding, response)

    # 7. Logs the interaction for RAGAS evaluation
    # log_ragas_entry expects question (str), context (str), answer (str)
    log_ragas_entry(user_query, ragas_context, response) # Use the more detailed ragas_context

    return f"[RAG - generated]\n{response}"

if __name__ == "__main__":
    # This is for testing purposes.
    # Ensure your .env file has the necessary Azure OpenAI and Search credentials.
    # Also, ensure you have ingested some data into your Azure Search index.

    print("Testing RAG pipeline (requires configured .env and indexed data)...")

    # Configure necessary clients if not already done by their respective modules
    from .openai_client import OPENAI_API_KEY
    from .search_client import AZURE_SEARCH_KEY as SEARCH_API_KEY

    if not OPENAI_API_KEY or not SEARCH_API_KEY:
        print("Azure OpenAI or Search keys not configured. Skipping RAG pipeline test.")
    else:
        # First test: Cache miss, then RAG, then cache hit
        test_query_1 = "What are the benefits of using a vector database for similarity search?"
        print(f"\nRunning RAG for query 1: '{test_query_1}'")
        response_1 = run_rag_pipeline(test_query_1)
        print(f"Response 1:\n{response_1}")

        print(f"\nRunning RAG for query 1 again (should be a cache hit if similarity is high enough): '{test_query_1}'")
        response_1_cached = run_rag_pipeline(test_query_1)
        print(f"Response 1 (cached):\n{response_1_cached}")
        assert "[CAG - cached]" in response_1_cached, "Second call for query 1 was not cached!"

        # Second test: A different query
        test_query_2 = "How does Azure OpenAI differ from standard OpenAI services?"
        print(f"\nRunning RAG for query 2: '{test_query_2}'")
        response_2 = run_rag_pipeline(test_query_2)
        print(f"Response 2:\n{response_2}")
        assert "[RAG - generated]" in response_2, "Query 2 should be RAG generated if cache is working correctly."

        print("\nRAG pipeline tests complete.")
        print(f"Check '.cache/ragas_log.jsonl' for logged entries.")
