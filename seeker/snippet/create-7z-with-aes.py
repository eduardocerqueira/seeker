#date: 2023-09-05T17:00:25Z
#url: https://api.github.com/gists/cb2452ffc29e67c365a4eca8d65ecf28
#owner: https://api.github.com/users/aspose-com-gists

import aspose.zip as az

# Create archive settings and set password
archive_settings = az.saving.SevenZipEntrySettings(None, az.saving.SevenZipAESEncryptionSettings("pass", az.saving.EncryptionMethod.AES128))

# Create and save archive with multiple files
with az.sevenzip.SevenZipArchive(archive_settings) as archive:        
    # Add files or folder to 7z
    archive.create_entries("files")

    # Create and save 7z archive
    archive.save('protected_archive.7z')