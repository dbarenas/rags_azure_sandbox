output "openai_endpoint" {
  value = azurerm_cognitive_account.openai.endpoint
}

output "search_name" {
  value = azurerm_search_service.search.name
}

output "bot_endpoint" {
  value = "https://${azurerm_bot_channels_registration.bot.name}.azurewebsites.net"
}

output "resource_group" {
  value = azurerm_resource_group.rag.name
}
