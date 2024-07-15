#date: 2024-07-15T16:46:46Z
#url: https://api.github.com/gists/2c1f39e2f487c29c0e4a51822e4282df
#owner: https://api.github.com/users/summerofgeorge

import openpyxl
import random
from faker import Faker

# Set the random seed
random.seed(1234)

# Initialize Faker for generating fake data
fake = Faker()

# Create a new workbook
wb = openpyxl.Workbook()

# Get the active sheet
ws = wb.active

# Add headers to the sheet
ws.append(["Name", "Address"])

# Generate 100 fake customer records
for _ in range(100):
    name = fake.name()
    address = fake.address().replace("\n", ", ")
    ws.append([name, address])

# Save the workbook
wb.save("FakeCustomers.xlsx")
print("Excel workbook 'FakeCustomers.xlsx' created successfully!")
