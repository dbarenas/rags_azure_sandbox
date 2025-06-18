import os
import fitz # PyMuPDF
import json # Added for metadata in ingest_pdf
from pathlib import Path
from .openai_client import get_embedding
from .search_client import upload_documents, create_vector_index_if_not_exists, AZURE_SEARCH_INDEX_NAME

# Define the default embedding model's expected dimensions
# This should match the model used in openai_client.py (e.g., text-embedding-ada-002 outputs 1536)
# And also the dimensions set when creating the search index.
EMBEDDING_DIMENSIONS = 1536 # Default for text-embedding-ada-002

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits text into overlapping chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks

def ingest_pdf(file_path: str | Path,
               index_name: str = AZURE_SEARCH_INDEX_NAME,
               chunk_size: int = 1000,
               overlap: int = 100):
    """
    Ingests a PDF file:
    1. Extracts text from the PDF.
    2. Splits text into manageable chunks.
    3. Generates embeddings for each chunk.
    4. Prepares documents in the format expected by Azure Cognitive Search.
    5. Uploads the documents to the specified search index.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists() or not file_path.is_file():
        print(f"Error: File not found or is not a file: {file_path}")
        return

    if file_path.suffix.lower() != ".pdf":
        print(f"Error: File is not a PDF: {file_path}")
        return

    print(f"Starting ingestion for PDF: {file_path.name} into index '{index_name}'...")

    # 0. Ensure index exists, create if not (uses default dimensions or you can pass it)
    print(f"Ensuring index '{index_name}' exists with vector dimensions {EMBEDDING_DIMENSIONS}...")
    create_vector_index_if_not_exists(index_name=index_name, vector_dimensions=EMBEDDING_DIMENSIONS)

    # 1. Extract text from PDF
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") + "\n\n" # Add double newline between pages
        doc.close()
        print(f"Successfully extracted text from {file_path.name}. Total characters: {len(full_text)}")
    except Exception as e:
        print(f"Error extracting text from PDF {file_path.name}: {e}")
        return

    if not full_text.strip():
        print(f"No text extracted from {file_path.name}. Skipping.")
        return

    # 2. Split text into chunks
    text_chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    if not text_chunks:
        print(f"Text could not be split into chunks for {file_path.name}. Skipping.")
        return
    print(f"Split text into {len(text_chunks)} chunks.")

    # 3. Generate embeddings and prepare documents
    documents_to_upload = []
    for i, chunk in enumerate(text_chunks):
        try:
            embedding = get_embedding(chunk)
            if not embedding:
                print(f"Warning: Failed to generate embedding for chunk {i} from {file_path.name}. Skipping chunk.")
                continue

            # Ensure embedding dimension matches index expectation
            if len(embedding) != EMBEDDING_DIMENSIONS:
                print(f"Warning: Embedding for chunk {i} has dimension {len(embedding)}, expected {EMBEDDING_DIMENSIONS}. Skipping chunk.")
                # This could be due to an error or misconfiguration of the embedding model.
                continue

            doc_id = f"{file_path.stem}_chunk_{i}"
            document = {
                "id": doc_id,
                "content": f"Chunk {i+1}: {chunk}", # Adding chunk index for context, as per user prompt
                "content_vector": embedding,
                "source_document_id": file_path.name,
                "metadata": json.dumps({"original_file": file_path.name, "chunk_index": i, "text_length": len(chunk)})
            }
            documents_to_upload.append(document)
            # print(f"Prepared document ID: {doc_id} for upload.")

        except Exception as e:
            print(f"Error processing chunk {i} from {file_path.name}: {e}")
            continue

    # 4. Upload documents
    if documents_to_upload:
        print(f"Attempting to upload {len(documents_to_upload)} documents to index '{index_name}'...")
        upload_documents(documents_to_upload, index_name=index_name)
    else:
        print(f"No documents prepared for upload from {file_path.name}.")

    print(f"Finished ingestion process for PDF: {file_path.name}")


if __name__ == "__main__":
    import argparse
    import json # Added for metadata in ingest_pdf

    parser = argparse.ArgumentParser(description="Ingest a PDF document into Azure Cognitive Search.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to ingest.")
    parser.add_argument("--index", type=str, default=AZURE_SEARCH_INDEX_NAME, help=f"Target Azure Search index name (default: {AZURE_SEARCH_INDEX_NAME}).")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of text chunks (default: 1000).")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between text chunks (default: 100).")

    # This requires environment variables for Azure Search and OpenAI to be set.
    # Example usage:
    # python -m app.ingest "path/to/your/document.pdf"
    #
    # Before running, create a dummy PDF for testing if you don't have one.
    # You also need to have your .env file configured.

    args = parser.parse_args()

    # Check if OpenAI and Search clients are configured (basic check)
    from .openai_client import OPENAI_API_KEY
    from .search_client import AZURE_SEARCH_KEY as SEARCH_API_KEY

    if not OPENAI_API_KEY or not SEARCH_API_KEY:
        print("Error: Azure OpenAI API key or Azure Search API key is not set in environment variables.")
        print("Please ensure your .env file is correctly configured and loaded.")
        print("Skipping ingestion example.")
    else:
        print(f"Attempting to ingest '{args.pdf_path}' into index '{args.index}'...")
        # Create a dummy PDF for testing if it doesn't exist
        dummy_pdf_path = Path("dummy_test_document.pdf")
        if args.pdf_path == str(dummy_pdf_path) and not dummy_pdf_path.exists():
            try:
                from .search_client import search_admin_client # check if client is there
                if search_admin_client: # only create if client is available
                    doc = fitz.open() # new empty PDF
                    page = doc.new_page()
                    page.insert_text((72, 72), "This is a simple test PDF for ingestion.")
                    page.insert_text((72, 92), "It contains some sample text to be chunked and embedded.")
                    page.insert_text((72, 112), "RAG systems retrieve relevant information to augment generation.")
                    doc.save(dummy_pdf_path)
                    doc.close()
                    print(f"Created dummy PDF: {dummy_pdf_path}")
                    args.pdf_path = str(dummy_pdf_path) # use this for the test
                else:
                    print("Search admin client not available, cannot create dummy PDF safely.")
            except Exception as e:
                print(f"Could not create dummy PDF: {e}")


        if Path(args.pdf_path).exists():
             ingest_pdf(args.pdf_path, index_name=args.index, chunk_size=args.chunk_size, overlap=args.overlap)
        else:
            print(f"Test PDF file '{args.pdf_path}' not found. Please provide a valid PDF path or run with '{dummy_pdf_path}' to auto-create a dummy.")
