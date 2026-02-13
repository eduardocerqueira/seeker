#date: 2026-02-13T17:21:30Z
#url: https://api.github.com/gists/e7f99716eaa13ef2e66304c82c9e9b99
#owner: https://api.github.com/users/santhoshbhandari3008

# Data Disk
resource "azurerm_managed_disk" "data" {
  count = var.data_disk_size_gb != null ? 1 : 0

  name                 = "${var.hostname}-data-01"
  location             = var.location
  resource_group_name  = var.resource_group_name
  storage_account_type = var.data_disk_type
  create_option        = "Empty"
  disk_size_gb         = var.data_disk_size_gb
}

# Attach Data Disk to VM
resource "azurerm_virtual_machine_data_disk_attachment" "data" {
  count = var.data_disk_size_gb != null ? 1 : 0

  managed_disk_id    = azurerm_managed_disk.data[0].id
  virtual_machine_id = azurerm_windows_virtual_machine.vm.id
  lun                = 0
  caching            = var.data_disk_caching
}

# Azure Disk Encryption (BitLocker)
resource "azurerm_virtual_machine_extension" "ade" {
  name                       = "AzureDiskEncryption"
  virtual_machine_id         = azurerm_windows_virtual_machine.vm.id
  publisher                  = "Microsoft.Azure.Security"
  type                       = "AzureDiskEncryption"
  type_handler_version       = "2.2"
  auto_upgrade_minor_version = true

  settings = jsonencode({
    EncryptionOperation    = "EnableEncryption"
    KeyVaultURL            = var.key_vault_url
    KeyVaultResourceId     = var.key_vault_id
    KeyEncryptionAlgorithm = "RSA-OAEP"
    VolumeType             = "All"  # OS, Data, or All
  })

  depends_on = [azurerm_virtual_machine_data_disk_attachment.data]
}


variable "key_vault_url" {
  type = string
}

variable "key_vault_id" {
  type = string
}

variable "data_disk_size_gb" {
  type    = number
  default = null
}

variable "data_disk_type" {
  type    = string
  default = "Standard_LRS"
}

variable "data_disk_caching" {
  type    = string
  default = "ReadWrite"
}