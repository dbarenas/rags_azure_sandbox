import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# In-memory cache (dictionary)
# For a production system, consider Redis, SQLite, or another persistent store.
_CACHE = {} # Stores embedding as key (tuple for hashability) and response as value.
            # More robust: query string as key, and store {'embedding': [], 'response': ''}

# For more advanced caching, we might store {'query_text': query, 'embedding': embedding, 'response': response, 'timestamp': time.time()}
_ADVANCED_CACHE_LIST = [] # Stores dicts like the one above
SIMILARITY_THRESHOLD = 0.95 # Example threshold for considering a cache hit

def _calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    # Ensure inputs are numpy arrays for cosine_similarity
    vec1_np = np.array(vec1).reshape(1, -1)
    vec2_np = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1_np, vec2_np)[0][0]

def search_cache(query_embedding: list[float]) -> str | None:
    """
    Searches the cache for a similar query embedding.
    Returns the cached response if a sufficiently similar embedding is found, else None.
    """
    if not query_embedding:
        return None

    # Simple exact match cache (less useful for varying queries)
    # query_embedding_tuple = tuple(query_embedding)
    # return _CACHE.get(query_embedding_tuple)

    # Advanced similarity based cache
    best_match_response = None
    highest_similarity = 0.0

    # Iterate through _ADVANCED_CACHE_LIST (or a more optimized structure)
    # This is a naive linear scan, not suitable for large caches.
    # For larger caches, use vector DBs or approximate nearest neighbor search libraries.
    for item in _ADVANCED_CACHE_LIST:
        cached_embedding = item.get('embedding')
        if cached_embedding:
            similarity = _calculate_cosine_similarity(query_embedding, cached_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                if similarity >= SIMILARITY_THRESHOLD:
                    best_match_response = item.get('response')

    if best_match_response:
        print(f"Cache hit with similarity: {highest_similarity:.4f}")
        return best_match_response

    print("Cache miss.")
    return None


def save_to_cache(query_text: str, query_embedding: list[float], response: str):
    """
    Saves a query, its embedding, and its response to the cache.
    """
    if not query_embedding:
        return

    # Simple exact match cache
    # query_embedding_tuple = tuple(query_embedding)
    # _CACHE[query_embedding_tuple] = response
    # print(f"Saved to simple cache. Cache size: {len(_CACHE)}")

    # Advanced cache entry
    cache_entry = {
        "query_text": query_text,
        "embedding": query_embedding,
        "response": response,
        # "timestamp": time.time() # Optional: for TTL or LRU eviction policies
    }
    _ADVANCED_CACHE_LIST.append(cache_entry)
    # Optional: Implement cache eviction strategy if it grows too large
    print(f"Saved to advanced cache. Cache size: {len(_ADVANCED_CACHE_LIST)}")


if __name__ == "__main__":
    # Test cosine similarity
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.707, 0.707, 0.0]
    vec_c = [0.0, 1.0, 0.0]
    print(f"Similarity A-B: {_calculate_cosine_similarity(vec_a, vec_b):.4f}") # Should be around 0.707
    print(f"Similarity A-C: {_calculate_cosine_similarity(vec_a, vec_c):.4f}") # Should be 0.0
    print(f"Similarity A-A: {_calculate_cosine_similarity(vec_a, vec_a):.4f}") # Should be 1.0

    # Test cache functionality
    print("\n--- Testing Cache ---")
    test_query1 = "What is AI?"
    test_embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5] # Dummy embedding
    test_response1 = "AI is artificial intelligence."

    test_query2 = "Tell me about machine learning." # Different query
    test_embedding2 = [0.6, 0.7, 0.8, 0.9, 1.0] # Different dummy embedding
    test_response2 = "Machine learning is a subset of AI."

    test_query3_similar = "What is Artificial Intelligence?" # Similar query to query1
    test_embedding3_similar = [0.11, 0.21, 0.31, 0.41, 0.51] # Slightly different embedding

    # 1. Save first query
    save_to_cache(test_query1, test_embedding1, test_response1)

    # 2. Search for the first query (should be a hit if threshold is okay, or if using exact match)
    cached_res1 = search_cache(test_embedding1)
    print(f"Search for embedding1: {cached_res1}")
    assert cached_res1 == test_response1, "Cache hit failed for identical embedding"

    # 3. Search for a similar query (test_embedding3_similar)
    #    This will be a hit if SIMILARITY_THRESHOLD is met by cosine similarity
    #    For this dummy data, it will likely be high.
    print(f"Similarity between embedding1 and embedding3_similar: {_calculate_cosine_similarity(test_embedding1, test_embedding3_similar):.4f}")
    cached_res3 = search_cache(test_embedding3_similar)
    print(f"Search for embedding3_similar: {cached_res3}")
    if cached_res3:
        assert cached_res3 == test_response1, "Cache hit for similar embedding returned wrong response"
        print("Cache hit for similar embedding was successful (as expected with high similarity).")
    else:
        print("Cache miss for similar embedding (this may be OK depending on threshold and similarity).")


    # 4. Search for a different query (should be a miss)
    cached_res2 = search_cache(test_embedding2)
    print(f"Search for embedding2: {cached_res2}")
    assert cached_res2 is None, "Cache should have missed for a different embedding"

    # 5. Save second query
    save_to_cache(test_query2, test_embedding2, test_response2)
    cached_res2_after_save = search_cache(test_embedding2)
    print(f"Search for embedding2 after saving: {cached_res2_after_save}")
    assert cached_res2_after_save == test_response2, "Cache hit failed for embedding2 after saving"

    print("\nCache test complete.")
    print(f"Current advanced cache size: {len(_ADVANCED_CACHE_LIST)}")
    # print(f"Current simple cache: {_CACHE}") # If using _CACHE
