#date: 2023-03-29T17:52:45Z
#url: https://api.github.com/gists/2f0f9d4002f181815da17a7cce253fb1
#owner: https://api.github.com/users/zackbunch

import pandas as pd
import json

class CSVParser:
    def __init__(self, filename):
        self.filename = filename
        self.allowed_departments = ['Department A', 'Department B', 'Department C']
        self.df = None
        self.grouped = None
        self.output = None
    
    def parse_csv(self):
        self.df = pd.read_csv(self.filename)
    
    def clean_data(self):
        # Replace NaN values with None
        self.df = self.df.fillna(value=None)

        # Replace Release date with None for rows where Status is 'In Work'
        self.df.loc[self.df['Status'] == 'In Work', 'Release date'] = None

        # Filter the DataFrame to exclude any departments that are not in the list of allowed departments
        self.df = self.df[self.df['Department'].isin(self.allowed_departments)]
    
    def group_data(self):
        # Group by Department and create a dictionary of documents for each department
        self.grouped = self.df.groupby('Department').apply(lambda x: x[['ID', 'Revision', 'Document', 'Status', 'Release date']].to_dict(orient='records')).to_dict()
        
        # Create a new DataFrame with the grouped data
        new_df = pd.DataFrame({'Department': [], 'Documents': []})

        for department, documents in self.grouped.items():
            new_row = {'Department': department, 'Documents': documents}
            new_df = new_df.append(new_row, ignore_index=True)

        # Convert the DataFrame to a list of dictionaries
        self.output = new_df.to_dict(orient='records')
    
    def write_output(self, output_filename):
        # Output to JSON file
        with open(output_filename, 'w') as f:
            json.dump(self.output, f, indent=4)
