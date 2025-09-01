#date: 2025-09-01T17:03:11Z
#url: https://api.github.com/gists/063b8d4dc01a7c8f30061ec8841b4256
#owner: https://api.github.com/users/Fernigithub

import pandas as pd

def generate_insert_queries_for_data_layers_v2(df, num_rows, column_mappings):
    '''
    Generate SQL insert queries from a DataFrame for the DataLayersInfo table.
    It carries forward layer_category and layer_subcategory when they are not explicitly mentioned in every row.

    :param df: DataFrame containing the data.
    :param num_rows: Number of rows to generate queries for.
    :param column_mappings: Dictionary mapping excel column names to table attribute names.
    :return: List of SQL insert queries.
    '''
    def format_value(value, is_boolean=False):
        '''Format the value for the SQL query. If the value is NaN, return NULL, handle boolean conversion if needed.'''
        if pd.isna(value):
            return 'NULL'
        if is_boolean:
            return 'TRUE' if value.lower() == 'vector' else 'FALSE' if value.lower() == 'raster' else 'NULL'
        return f"'{value}'"

    # Initialize variables to hold the last seen category and subcategory
    last_category = None
    last_subcategory = None

    # Generate SQL insert queries
    queries = []
    sub_region_id = 8  # Sub region ID 

    for index, row in df.iterrows():
        if index >= num_rows:  # Limit the number of queries
            break
        # Update last_category and last_subcategory if current row has values
        current_category = row['Layer Category (short)']
        current_subcategory = row['Layer Subcategory (short)']
        last_category = current_category if pd.notna(current_category) else last_category
        last_subcategory = current_subcategory if pd.notna(current_subcategory) else last_subcategory

        # Prepare query using either current row's value or the last seen value
        query = 'INSERT INTO data_layers_info ('
        query += 'sub_region_id, is_active,'
        query += ', '.join(column_mappings.values())
        query += ') VALUES ('
        query += f'{sub_region_id}, '  # Sub region ID
        query += 'TRUE, '  # Is active
        query += ', '.join([format_value(row[column] if pd.notna(row[column]) else last_category if column == 'Layer Category (short)' else last_subcategory if column == 'Layer Subcategory' else None, column == 'Geotype') for column in column_mappings.keys()])
        query += ');'
        queries.append(query)

    return queries

# Usage
file_path = '/Users/fer/Downloads/USA_Iowa - MASTER FILE - Geodatalayers & Exclusion_Assement Criteria.xlsx'  # Replace with the path to your Excel file
sheet_name = 'GEODATALAYERS'

num_rows = 80  # Number of rows to generate queries for
header_row = 2  # Header row in the Excel sheet

# Reading the DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)

# replace special characters ' " % in the content of the columns with empty string using regex
df = df.replace(to_replace=r'["\'%$*`~/]', value='', regex=True)

# remove line breaks and extra spaces
df = df.replace(to_replace=r'\s+', value=' ', regex=True)
df = df.replace(to_replace=r'\n', value=' ', regex=True)
df = df.replace(to_replace=r'\r', value=' ', regex=True)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
# Remove leading and trailing spaces from the DataFrame
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Column mappings based on the provided information
column_mappings_v2 = {
    'Layer Category (short)': 'layer_category',
    'Layer Subcategory (short)': 'layer_subcategory',
    'Layer Description': 'layer_description',
    'Geotype': 'is_vector',
    'Source': 'source_url',
    'Feature Type': 'feature_type',
    'Comments': 'comments',
    'File/Table Name': 'layer_table_name'
}

# Generating the queries
data_layer_insert_queries_v2 = generate_insert_queries_for_data_layers_v2(df, num_rows, column_mappings_v2)

# Printing the queries
for query in data_layer_insert_queries_v2:
    print(query)