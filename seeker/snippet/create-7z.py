#date: 2023-09-05T17:00:25Z
#url: https://api.github.com/gists/cb2452ffc29e67c365a4eca8d65ecf28
#owner: https://api.github.com/users/aspose-com-gists

import aspose.zip as az

# Create and save archive with single file
with az.sevenzip.SevenZipArchive() as archive:
    # Add file into archive
    archive.create_entry("file", "file.txt")

    # Create and save 7z archive
    archive.save('my_archive.7z')