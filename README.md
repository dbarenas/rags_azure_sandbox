# Azure RAG Bot with Caching and RAGAS Logging

This project implements a Retrieval-Augmented Generation (RAG) bot using Python and various Azure services. It features a caching mechanism (Cache-Augmented Generation - CAG) to improve response times and reduce costs, as well as logging capabilities compatible with the RAGAS framework for performance evaluation.

**Key Technologies:**
- Python 3.8+
- Azure OpenAI Service (for embeddings and text generation)
- Azure Cognitive Search (for vector storage and retrieval)
- Azure Bot Service (for bot interaction, though runnable locally with Bot Framework Emulator)
- `aiohttp` for the bot's web server
- `botbuilder-sdk` for Bot Framework integration
- `scikit-learn` for cache similarity
- `PyMuPDF` for PDF text extraction


## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Python:** Version 3.8 or higher. You can download Python from [python.org](https://www.python.org/downloads/).
- **Git:** For cloning the repository. You can download Git from [git-scm.com](https://git-scm.com/downloads/).
- **Azure Account:** An active Azure subscription is required to provision and use Azure OpenAI, Azure Cognitive Search, and (optionally) Azure Bot Service.
  - If you don't have one, you can create a [free Azure account](https://azure.microsoft.com/free/).
- **Bot Framework Emulator (Recommended for local testing):** For testing the bot locally. Download it from the [Bot Framework Emulator releases page](https://github.com/Microsoft/BotFramework-Emulator/releases).


## Setup Instructions

Follow these steps to set up the project locally:

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory_name> # Replace <repository_directory_name>
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```
    Activate the virtual environment:
    -   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables (`.env` file):**
    Copy the template file `.env.template` to a new file named `.env` in the project root:
    ```bash
    # On Windows (Command Prompt)
    copy .env.template .env
    # On Windows (PowerShell)
    Copy-Item .env.template .env
    # On macOS/Linux
    cp .env.template .env
    ```
    Now, open the `.env` file and fill in the required values. Here's where to find them:

    *   **Azure OpenAI Service:**
        -   `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI service endpoint URL (e.g., `https://your-openai-resource.openai.azure.com/`).
        -   `AZURE_OPENAI_KEY`: An API key for your Azure OpenAI service.
        -   `AZURE_OPENAI_API_VERSION`: The API version your Azure OpenAI service uses (e.g., `2023-07-01-preview`). Check the Azure portal for the version compatible with your models.
        -   `AZURE_OPENAI_EMBEDDING_MODEL`: The deployment name of your text embedding model (e.g., `text-embedding-ada-002`).
        -   `AZURE_OPENAI_COMPLETION_MODEL`: The deployment name of your chat/completion model (e.g., `gpt-35-turbo` or `gpt-4`).
        *Finding these values:*
            1.  Navigate to your Azure OpenAI resource in the [Azure portal](https://portal.azure.com/).
            2.  Under "Keys and Endpoint," you'll find the endpoint and keys.
            3.  Model deployment names are found under "Model deployments" within your Azure OpenAI Studio.

    *   **Azure Cognitive Search Service:**
        -   `AZURE_SEARCH_ENDPOINT`: Your Azure Cognitive Search service endpoint URL (e.g., `https://your-search-service.search.windows.net`).
        -   `AZURE_SEARCH_KEY`: An admin or query key for your Azure Cognitive Search service. For ingestion (`ingest.py`), an admin key is typically required.
        -   `AZURE_SEARCH_INDEX` (Optional): The name for your search index. Defaults to `rag-vector-index` if not set.
        *Finding these values:*
            1.  Navigate to your Azure Cognitive Search resource in the [Azure portal](https://portal.azure.com/).
            2.  The endpoint is on the "Overview" page.
            3.  Keys are found under "Keys."

    *   **Azure Bot Service (Optional - for cloud deployment or specific emulator features):**
        -   `MICROSOFT_APP_ID`: The Microsoft App ID for your bot registration.
        -   `MICROSOFT_APP_PASSWORD`: The Microsoft App Password (client secret) for your bot registration.
        *Finding these values:*
            1.  If you register your bot with Azure Bot Service, these values are provided during the registration process or can be found in the bot's "Configuration" blade in the Azure portal.
            2.  For local testing with the Bot Framework Emulator, these can often be left blank unless you are testing features that specifically require them. The `run_bot.py` script has defaults that allow running without these if they are not set.


## Ingesting Data

Before the RAG bot can answer questions effectively, you need to populate its knowledge base (Azure Cognitive Search index) with your documents. The project includes a script for ingesting PDF files.

1.  **Prepare your PDF documents:**
    Ensure the PDF files you want to ingest are accessible from your local machine.

2.  **Ensure Environment Variables are Set:**
    The ingestion script (`app/ingest.py`) requires Azure OpenAI and Azure Cognitive Search credentials to be correctly set in your `.env` file (see "Set Up Environment Variables" above).

3.  **Run the Ingestion Script:**
    Open your terminal (with the virtual environment activated) and run the following command for each PDF you want to ingest:
    ```bash
    python -m app.ingest "path/to/your/document.pdf"
    ```
    Replace `"path/to/your/document.pdf"` with the actual path to your PDF file.

    *Optional arguments for the ingestion script:*
    -   `--index <index_name>`: Specify a custom Azure Search index name. Defaults to the value of `AZURE_SEARCH_INDEX` in your `.env` file or `rag-vector-index`.
    -   `--chunk_size <size>`: Set the size of text chunks (default: 1000 characters).
    -   `--overlap <size>`: Set the overlap between text chunks (default: 100 characters).

    Example with optional arguments:
    ```bash
    python -m app.ingest "my_report.pdf" --index "my-custom-index" --chunk_size 1500
    ```

    The script will:
    -   Extract text from the PDF.
    -   Split the text into manageable, overlapping chunks.
    -   Generate vector embeddings for each chunk using Azure OpenAI.
    -   Upload the chunks and their embeddings to your Azure Cognitive Search index.
    -   It will also create the search index if it doesn't already exist, using the schema defined in `app/search_client.py`.


## Running the Bot

Once you have set up your environment variables and (optionally) ingested data, you can run the bot server:

1.  **Ensure Environment Variables are Set:**
    The bot (`run_bot.py`) requires Azure OpenAI credentials. If you plan to connect it via Azure Bot Service or use features requiring authentication in the Bot Framework Emulator, `MICROSOFT_APP_ID` and `MICROSOFT_APP_PASSWORD` should also be set in your `.env` file.

2.  **Start the Bot Server:**
    Open your terminal (with the virtual environment activated) and run:
    ```bash
    python run_bot.py
    ```
    You should see output indicating the server has started, typically:
    ```
    Bot server starting on http://localhost:3978
    Messaging endpoint available at http://localhost:3978/api/messages
    Health check available at http://localhost:3978/health
    ```

3.  **Connect to the Bot (Local Testing):**
    You can use the [Bot Framework Emulator](https://github.com/Microsoft/BotFramework-Emulator/releases) for local testing:
    -   Launch the Bot Framework Emulator.
    -   Click "Open Bot."
    -   For "Bot URL," enter the messaging endpoint: `http://localhost:3978/api/messages`.
    -   If you have `MICROSOFT_APP_ID` and `MICROSOFT_APP_PASSWORD` set in your `.env` file and want to test with them, enter them in the Emulator's configuration. Otherwise, they can often be left blank for local connections to `localhost`.
    -   Click "Connect."

    You should now be able to send messages to your bot and receive responses. The bot will use the RAG pipeline to retrieve information from your Azure Cognitive Search index and generate answers with Azure OpenAI.


## Project Structure

A brief overview of the key files and directories:

-   `app/`: Contains the core application logic.
    -   `__init__.py`: Makes `app` a Python package.
    -   `bot.py`: Defines the `AzureRAGBot` class (Bot Framework `ActivityHandler`).
    -   `ingest.py`: Script for ingesting PDF documents into Azure Cognitive Search.
    -   `memory_cache.py`: Implements in-memory caching for query-response pairs.
    -   `openai_client.py`: Handles interactions with Azure OpenAI Service (embeddings, completions).
    -   `rag_pipeline.py`: Orchestrates the RAG + Caching (CAG) pipeline.
    -   `ragas_logging.py`: Logs interactions in a RAGAS-compatible format.
    -   `search_client.py`: Manages interactions with Azure Cognitive Search (indexing, querying).
    -   `.cache/`: Directory (created automatically) to store `ragas_log.jsonl`.
-   `run_bot.py`: The main entry point to start the bot's web server using `aiohttp`.
-   `requirements.txt`: Lists all Python dependencies for the project.
-   `.env.template`: A template for the `.env` file, listing required environment variables.
-   `README.md`: This file – providing documentation for the project.


## RAGAS Logging

The application is configured to log interactions in a format compatible with the [RAGAS](https://github.com/explodinggradients/ragas) framework, which is designed for evaluating RAG pipelines.

-   **Log File Location:** Each time the RAG pipeline generates a new response (i.e., not a cache hit), an entry containing the `question`, retrieved `context`, and generated `answer` is logged to:
    `app/.cache/ragas_log.jsonl`

-   **Usage:** This JSONL file can be used as a dataset for evaluation with RAGAS. You would typically use the RAGAS library to load this dataset and compute metrics like faithfulness, answer relevancy, context precision, and context recall.

    Refer to the [RAGAS documentation](https://docs.ragas.io/en/latest/getstarted/evaluation.html) for more details on how to perform evaluations.
