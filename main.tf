provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rag" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_cognitive_account" "openai" {
  name                = var.openai_name
  location            = azurerm_resource_group.rag.location
  resource_group_name = azurerm_resource_group.rag.name
  kind                = "OpenAI"
  sku_name            = "S0"
  custom_subdomain_name = var.openai_name
  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_search_service" "search" {
  name                = var.search_name
  location            = azurerm_resource_group.rag.location
  resource_group_name = azurerm_resource_group.rag.name
  sku                 = "standard"
  replica_count       = 1
  partition_count     = 1
}

resource "azurerm_bot_channels_registration" "bot" {
  name                    = var.bot_name
  location                = azurerm_resource_group.rag.location
  resource_group_name     = azurerm_resource_group.rag.name
  sku                     = "F0"
  microsoft_app_id        = var.bot_app_id
  microsoft_app_password  = var.bot_app_secret
}
