variable "resource_group_name" { default = "rag-rg" }
variable "location" { default = "eastus" }
variable "openai_name" { default = "rag-openai" }
variable "search_name" { default = "ragsearchsvc" }
variable "bot_name" { default = "ragbot" }

variable "bot_app_id" {
  type        = string
  description = "Bot Microsoft App ID"
}

variable "bot_app_secret" {
  type        = string
  sensitive   = true
  description = "Bot Microsoft App Secret"
}
