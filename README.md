# Azure RAG Infrastructure and Credentials Setup

This project provides Terraform scripts to provision Azure resources for a Retrieval Augmented Generation (RAG) application and a Python script to fetch the necessary credentials and generate a `.env` file.

## Overview

The Terraform scripts will create the following Azure resources:
*   Azure Resource Group
*   Azure Cognitive Search
*   Azure Cognitive Services (for OpenAI)
*   Azure Bot Channels Registration

The Python script will then connect to your Azure subscription to retrieve API keys and endpoints for these services and store them in a `.env` file for easy use in your RAG application.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Azure Account and Subscription**: You'll need an active Azure subscription. If you don't have one, create a [free Azure account](https://azure.microsoft.com/free/).
2.  **Azure CLI**: Install the [Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli) and log in to your account:
    ```bash
    az login
    ```
3.  **Terraform**: Install [Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli).
4.  **Python**: Install [Python 3.x](https://www.python.org/downloads/).
5.  **Microsoft App Registration**: You need to create a Microsoft App Registration for the Azure Bot.
    *   Go to the [Azure portal](https://portal.azure.com/) and navigate to **Microsoft Entra ID**.
    *   Select **App registrations** and click **+ New registration**.
    *   Give it a name (e.g., `MyRAGBotApp`).
    *   Supported account types: Choose what's appropriate, often "Accounts in this organizational directory only" or "Accounts in any organizational directory".
    *   Redirect URI: You can leave this blank or set it to a placeholder like `http://localhost` (Web).
    *   Click **Register**.
    *   Once registered, note down the **Application (client) ID** – this will be your `bot_app_id`.
    *   Go to **Certificates & secrets**, click **+ New client secret**. Add a description, choose an expiry, and click **Add**. Copy the **Value** of the secret immediately – this will be your `bot_app_secret`. It won't be visible again.
    *   For more details, see [Register an app with the Microsoft identity platform](https://docs.microsoft.com/azure/active-directory/develop/quickstart-register-app).

## Setup Instructions

### 1. Clone the Repository (Optional)

If you've received these files as part of a project, you might already have them. If it's a Git repository, clone it:
```bash
# git clone <repository-url>
# cd <repository-directory>
```
Otherwise, ensure all project files (`main.tf`, `variables.tf`, `outputs.tf`, `terraform.tfvars`, `generate_env.py`, `requirements.txt`) are in the same directory.

### 2. Configure Terraform Variables

Open the `terraform.tfvars` file and replace the placeholder values with your Bot's Application (client) ID and Client Secret obtained from the Microsoft App Registration step:

```hcl
bot_app_id     = "YOUR-ACTUAL-BOT-APP-ID"
bot_app_secret = "YOUR-ACTUAL-BOT-APP-SECRET"
```

You can also customize other variables in `variables.tf` (like resource names and location) if needed, or by creating a `terraform.tfvars` file or overriding them at the command line.

### 3. Provision Azure Infrastructure with Terraform

Navigate to the directory containing the Terraform files in your terminal.

*   **Initialize Terraform**: This downloads the necessary provider plugins.
    ```bash
    terraform init
    ```
*   **Apply Terraform Configuration**: This creates the Azure resources.
    ```bash
    terraform apply
    ```
    Terraform will show you a plan of the resources to be created. Review it and type `yes` when prompted to proceed. This might take a few minutes.

### 4. Generate the `.env` File

Once the Terraform deployment is complete, you can generate the `.env` file.

*   **Set Azure Subscription ID**: The Python script needs your Azure Subscription ID. Set it as an environment variable. Replace `YOUR_SUBSCRIPTION_ID` with your actual Subscription ID.
    *   For Linux/macOS:
        ```bash
        export AZURE_SUBSCRIPTION_ID="YOUR_SUBSCRIPTION_ID"
        ```
    *   For Windows (Command Prompt):
        ```bash
        set AZURE_SUBSCRIPTION_ID="YOUR_SUBSCRIPTION_ID"
        ```
    *   For Windows (PowerShell):
        ```bash
        $env:AZURE_SUBSCRIPTION_ID="YOUR_SUBSCRIPTION_ID"
        ```
    You can find your Subscription ID in the [Azure portal](https://portal.azure.com/#blade/Microsoft_Azure_Billing/SubscriptionsBlade).

*   **Install Python Dependencies**: Install the required Python libraries using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

*   **Run the Python Script**: Execute the `generate_env.py` script.
    ```bash
    python generate_env.py
    ```
    This script will use your Azure CLI credentials (from `az login`) to fetch the outputs from your Terraform deployment (like endpoints and keys) and create a `.env` file in the current directory.

## `.env` File Contents

The generated `.env` file will look something like this:

```env
AZURE_OPENAI_ENDPOINT=https://<your-openai-name>.openai.azure.com/
AZURE_OPENAI_KEY=<your-openai-key1>
AZURE_SEARCH_ENDPOINT=https://<your-search-name>.search.windows.net
AZURE_SEARCH_KEY=<your-search-primary-key>
```

It will also include any Bot-related environment variables if you choose to add them to the `generate_env.py` script in the future (e.g., `MICROSOFT_APP_ID`, `MICROSOFT_APP_PASSWORD` if needed by your bot application directly from the .env).

## Using the `.env` File

This `.env` file can now be used by your RAG application (e.g., built with FastAPI, LangChain, Semantic Kernel, etc.) to connect to the Azure services. Most application frameworks and libraries have native support or simple integrations for loading variables from a `.env` file (e.g., using the `python-dotenv` library in Python).

## Cleaning Up

To remove the resources created by Terraform, run:
```bash
terraform destroy
```
Review the plan and type `yes` when prompted. Remember to also delete the Microsoft App Registration in Azure AD if you no longer need it.

---

This `README.md` provides a comprehensive guide for setting up the Azure infrastructure and generating the necessary credentials.
