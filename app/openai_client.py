import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI client for Azure
OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE") or "azure"
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT") # Renamed from AZURE_OPENAI_BASE for consistency
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL") or "text-embedding-ada-002" # Default if not set
COMPLETION_MODEL_NAME = os.getenv("AZURE_OPENAI_COMPLETION_MODEL") or "gpt-35-turbo" # Default if not set for GPT 3.5 Turbo

# Set up OpenAI client
if OPENAI_API_KEY: # Proceed only if the key is available
    openai.api_type = OPENAI_API_TYPE
    openai.api_key = OPENAI_API_KEY
    openai.api_base = OPENAI_API_BASE
    openai.api_version = OPENAI_API_VERSION
else:
    print("Warning: AZURE_OPENAI_KEY is not set. OpenAI client will not be functional.")
    # Optionally, raise an error or handle this case as needed
    # raise ValueError("AZURE_OPENAI_KEY must be set in environment variables.")

def get_embedding(text: str, engine: str = EMBEDDING_MODEL_NAME) -> list[float]:
    """
    Generates an embedding for the given text using Azure OpenAI.
    """
    if not openai.api_key:
        print("Error: OpenAI client not configured. Cannot generate embedding.")
        return [] # Or raise an exception

    try:
        response = openai.Embedding.create(
            input=text,
            engine=engine
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [] # Or raise an exception

def get_completion(prompt: str, engine: str = COMPLETION_MODEL_NAME) -> str:
    """
    Generates a completion for the given prompt using Azure OpenAI.
    """
    if not openai.api_key:
        print("Error: OpenAI client not configured. Cannot generate completion.")
        return "Error: OpenAI client not configured." # Or raise an exception

    try:
        # For chat models like gpt-3.5-turbo or gpt-4, the API expects a list of messages
        messages = [{"role": "user", "content": prompt}]
        completion = openai.ChatCompletion.create(
            engine=engine,
            messages=messages
        )
        return completion.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error generating completion: {e}")
        return f"Error generating completion: {e}" # Or raise an exception

if __name__ == "__main__":
    # This is for testing purposes.
    # Ensure your .env file has the necessary Azure OpenAI credentials.
    print(f"Using Azure OpenAI endpoint: {OPENAI_API_BASE}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Completion model: {COMPLETION_MODEL_NAME}")

    if OPENAI_API_KEY:
        sample_text = "This is a sample document for testing embeddings."
        embedding_vector = get_embedding(sample_text)
        if embedding_vector:
            print(f"Sample embedding (first 5 dimensions): {embedding_vector[:5]}")
            print(f"Embedding vector dimension: {len(embedding_vector)}")
        else:
            print("Failed to generate embedding.")

        sample_prompt = "What is the capital of France?"
        completion_response = get_completion(sample_prompt)
        print(f"Sample completion for '{sample_prompt}': {completion_response}")
    else:
        print("Skipping live tests as AZURE_OPENAI_KEY is not set.")
