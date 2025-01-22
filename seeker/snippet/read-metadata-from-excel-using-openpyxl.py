#date: 2025-01-22T16:56:19Z
#url: https://api.github.com/gists/fab5431047636fa12d871c8c00185621
#owner: https://api.github.com/users/documentprocessing

from openpyxl import load_workbook

# Load the Excel workbook
file_path = "example.xlsx"
workbook = load_workbook(file_path)

# Access the workbook's metadata
properties = workbook.properties

# Print metadata information
print("Excel Metadata:")
print(f"Title: {properties.title}")
print(f"Author: {properties.author}")
print(f"Subject: {properties.subject}")
print(f"Keywords: {properties.keywords}")
print(f"Created Date: {properties.created}")
print(f"Last Modified By: {properties.lastModifiedBy}")
print(f"Modified Date: {properties.modified}")
