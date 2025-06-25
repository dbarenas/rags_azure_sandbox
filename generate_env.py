import os
from dotenv import set_key
from azure.identity import DefaultAzureCredential
from azure.mgmt.search import SearchManagementClient
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

# Load values from Terraform outputs or manually
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = "rag-rg"
search_name = "ragsearchsvc"
openai_account_name = "rag-openai"

credential = DefaultAzureCredential()

# Cognitive OpenAI
cognitive_client = CognitiveServicesManagementClient(credential, subscription_id)
openai_account = cognitive_client.accounts.get(resource_group, openai_account_name)
openai_keys = cognitive_client.accounts.list_keys(resource_group, openai_account_name)

# Cognitive Search
search_client = SearchManagementClient(credential, subscription_id)
search_keys = search_client.admin_keys.get(resource_group, search_name)

# Save to .env
env_file = ".env"
set_key(env_file, "AZURE_OPENAI_ENDPOINT", openai_account.properties.endpoint)
set_key(env_file, "AZURE_OPENAI_KEY", openai_keys.key1)
set_key(env_file, "AZURE_SEARCH_ENDPOINT", f"https://{search_name}.search.windows.net")
set_key(env_file, "AZURE_SEARCH_KEY", search_keys.primary_key)

print(".env file created with OpenAI and Search credentials.")
