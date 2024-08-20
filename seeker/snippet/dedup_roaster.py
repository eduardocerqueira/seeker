#date: 2024-08-20T16:46:29Z
#url: https://api.github.com/gists/5ad11d4d71544e6eeb18901f262328ca
#owner: https://api.github.com/users/datageek19

import pandas as pd
import glob
import os

# Step 1: List all relevant CSV files
files = glob.glob("UHCCS-1-z-*.csv")  # Adjust the pattern to match your files

# Step 2: Create a dictionary to group files by month
file_groups = {}
for file in files:
    # Extract the date part from the filename
    date_part = os.path.basename(file).split('-')[-3:]  # Extract the '01-03-2024' part
    month_year = '-'.join(date_part[1:])  # Extract '03-2024'
    
    # Group files by month-year
    if month_year not in file_groups:
        file_groups[month_year] = []
    file_groups[month_year].append(file)

# Step 3: Process each group of files
for month_year, file_list in file_groups.items():
    # Read all files for the given month into a DataFrame
    month_df_list = [pd.read_csv(f) for f in file_list]
    combined_df = pd.concat(month_df_list, ignore_index=True)
    
    # Step 4: Check for duplicates based on First_Name, Last_Name, Birthdate
    duplicates = combined_df.duplicated(subset=['First_Name', 'Last_Name', 'Birthdate'], keep=False)
    duplicates_df = combined_df[duplicates]
    
    # Print or save the duplicates found for this month
    print(f"Duplicates for {month_year}:")
    print(duplicates_df)

    # Optional: Save duplicates to a CSV file
    # duplicates_df.to_csv(f'duplicates_{month_year}.csv', index=False)
# +++++++++++++++++++++++++++
import dask.dataframe as dd

combined_df = dd.concat([dd.read_csv(f) for f in file_list])
duplicates_df = combined_df[combined_df.duplicated(subset=['First_Name', 'Last_Name', 'Birthdate'], keep=False)].compute()

import dask.dataframe as dd

combined_df = dd.concat([dd.read_csv(f) for f in file_list])
duplicates_df = combined_df[combined_df.duplicated(subset=['First_Name', 'Last_Name', 'Birthdate'], keep=False)].compute()


writer = pd.ExcelWriter('duplicate_report.xlsx', engine='xlsxwriter')
for month_year, duplicates_df in all_duplicates.items():
    duplicates_df.to_excel(writer, sheet_name=month_year, index=False)
writer.save()
