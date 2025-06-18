import os
import json
from pathlib import Path

# Define the directory for RAGAS logs
RAGAS_LOG_DIR = Path(__file__).parent / ".cache"
RAGAS_LOG_PATH = RAGAS_LOG_DIR / "ragas_log.jsonl"

def log_ragas_entry(question: str, context: str, answer: str):
    """
    Logs a RAGAS entry (question, context, answer) to a JSONL file.
    Creates the log directory if it doesn't exist.
    """
    try:
        RAGAS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "question": question,
            "context": context,
            "answer": answer
        }
        with open(RAGAS_LOG_PATH, "a", encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Error logging RAGAS entry: {e}")

# Example usage (optional, can be removed or commented out)
if __name__ == "__main__":
    # Create dummy data for testing
    log_ragas_entry(
        question="What is Azure Cognitive Search?",
        context="Azure Cognitive Search is a cloud search service...",
        answer="Azure Cognitive Search is a search-as-a-service cloud solution..."
    )
    log_ragas_entry(
        question="What is OpenAI?",
        context="OpenAI is an AI research and deployment company...",
        answer="OpenAI is known for its advanced language models like GPT."
    )
    print(f"RAGAS log created at: {RAGAS_LOG_PATH}")
