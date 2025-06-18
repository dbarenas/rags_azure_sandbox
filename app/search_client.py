import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)

# Load environment variables from .env file
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX") or "rag-vector-index" # Default if not set

# Global clients, initialize if credentials are available
search_admin_client = None
search_client = None

if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY:
    try:
        # Client for managing indexes (creating, updating, deleting)
        search_admin_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        # Client for querying and managing documents in a specific index
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        print(f"Azure Search clients initialized for index '{AZURE_SEARCH_INDEX_NAME}'.")
    except Exception as e:
        print(f"Error initializing Azure Search clients: {e}")
else:
    print("Warning: AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_KEY is not set. Azure Search clients will not be functional.")

def create_vector_index_if_not_exists(index_name: str = AZURE_SEARCH_INDEX_NAME, vector_dimensions: int = 1536):
    """
    Creates a vector search index in Azure Cognitive Search if it doesn't already exist.
    The index schema is predefined with common fields for RAG.
    `vector_dimensions` should match the output of your embedding model (e.g., 1536 for text-embedding-ada-002).
    """
    if not search_admin_client:
        print("Error: Search admin client not configured. Cannot create index.")
        return

    try:
        search_admin_client.get_index(index_name)
        print(f"Index '{index_name}' already exists.")
    except Exception: # If get_index throws an error, index does not exist
        print(f"Index '{index_name}' not found. Creating new index...")
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=vector_dimensions,
                        vector_search_configuration="default-hnsw"),
            SearchableField(name="source_document_id", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True, default_value=""),
            SearchableField(name="metadata", type=SearchFieldDataType.String, searchable=True, default_value="{}") # JSON string for other metadata
        ]

        vector_search = VectorSearch(
            algorithm_configurations=[
                VectorSearchAlgorithmConfiguration(
                    name="default-hnsw",
                    kind="hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ]
        )

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
        try:
            search_admin_client.create_index(index)
            print(f"Index '{index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating index '{index_name}': {e}")


def upload_documents(documents: list[dict], index_name: str = AZURE_SEARCH_INDEX_NAME):
    """
    Uploads a list of documents to the specified Azure Cognitive Search index.
    Each document in the list should be a dictionary.
    Example document:
    {
        "id": "unique_doc_id_1",
        "content": "Text content of the document.",
        "content_vector": [0.1, 0.2, ..., N.N], (embedding vector)
        "source_document_id": "original_pdf_filename.pdf",
        "metadata": "{ \"page_number\": 1 }"
    }
    """
    if not search_client:
        print("Error: Search client not configured for the target index. Cannot upload documents.")
        # Update search_client to point to the correct index if different from default
        if index_name != AZURE_SEARCH_INDEX_NAME:
             current_search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=index_name, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
        else:
            print("Search client is not configured due to missing credentials.")
            return
    else:
        current_search_client = search_client
        if index_name != AZURE_SEARCH_INDEX_NAME: # If target index is different from default
             current_search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=index_name, credential=AzureKeyCredential(AZURE_SEARCH_KEY))


    if not documents:
        print("No documents provided to upload.")
        return

    try:
        result = current_search_client.upload_documents(documents=documents)
        succeeded_count = sum(1 for r in result if r.succeeded)
        print(f"Uploaded {succeeded_count} of {len(documents)} documents to index '{index_name}'.")
        for res in result:
            if not res.succeeded:
                print(f"Failed to upload document {res.key}: {res.error_message}")
    except Exception as e:
        print(f"Error uploading documents to index '{index_name}': {e}")


def search_similar_documents(embedding: list[float], top_k: int = 5, index_name: str = AZURE_SEARCH_INDEX_NAME) -> list[str]:
    """
    Searches for documents with similar content_vector using vector search.
    Returns a list of document contents.
    """
    if not search_client:
        print("Error: Search client not configured. Cannot perform search.")
        # Update search_client to point to the correct index if different from default
        if index_name != AZURE_SEARCH_INDEX_NAME:
             current_search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=index_name, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
        else:
            print("Search client is not configured due to missing credentials.")
            return []
    else:
        current_search_client = search_client
        if index_name != AZURE_SEARCH_INDEX_NAME: # If target index is different from default
             current_search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=index_name, credential=AzureKeyCredential(AZURE_SEARCH_KEY))

    try:
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="content_vector")
        results = current_search_client.search(
            search_text=None,  # No keyword search text, pure vector search
            vector_queries=[vector_query],
            select=["id", "content", "source_document_id", "metadata"], # Specify fields to retrieve
            top=top_k
        )

        docs = []
        for result in results:
            # You might want to format this string or return more structured data
            doc_info = f"Source: {result.get('source_document_id', 'N/A')}, Content: {result['content']}"
            if result.get('metadata'):
                doc_info += f", Metadata: {result['metadata']}"
            docs.append(doc_info)
        return docs
    except Exception as e:
        print(f"Error during vector search in index '{index_name}': {e}")
        return []

if __name__ == "__main__":
    # This is for testing purposes.
    # Ensure your .env file has the necessary Azure Search credentials.
    print(f"Using Azure Search endpoint: {AZURE_SEARCH_ENDPOINT}")
    print(f"Using Azure Search index: {AZURE_SEARCH_INDEX_NAME}")

    if search_admin_client and search_client:
        print("\n--- Testing Index Creation ---")
        create_vector_index_if_not_exists() # Use default index name and dimensions

        print("\n--- Testing Document Upload ---")
        # Dummy documents for testing (replace with actual embeddings)
        # Ensure the vector dimensions match your index configuration (e.g., 1536)
        dummy_embedding_dim = 1536
        sample_docs_to_upload = [
            {
                "id": "test_doc_1", "content": "This is a test document about cats.",
                "content_vector": [0.1] * dummy_embedding_dim, # Simplified dummy vector
                "source_document_id": "cats.txt", "metadata": "{ \"topic\": \"animals\" }"
            },
            {
                "id": "test_doc_2", "content": "Another test document, this one is about dogs.",
                "content_vector": [0.2] * dummy_embedding_dim, # Simplified dummy vector
                "source_document_id": "dogs.txt", "metadata": "{ \"topic\": \"animals\" }"
            }
        ]
        upload_documents(sample_docs_to_upload)

        print("\n--- Testing Vector Search ---")
        # Dummy query vector (should be generated by your embedding model)
        query_vector = [0.15] * dummy_embedding_dim # Simplified dummy query vector

        # Wait a bit for indexing to complete if running immediately after upload
        import time
        print("Waiting 5 seconds for indexing to catch up...")
        time.sleep(5)

        similar_docs = search_similar_documents(query_vector, top_k=2)
        if similar_docs:
            print("Found similar documents:")
            for doc_content in similar_docs:
                print(f"- {doc_content}")
        else:
            print("No similar documents found or search failed.")
    else:
        print("\nSkipping live Azure Search tests as credentials are not fully set.")
