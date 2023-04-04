#date: 2023-04-04T16:41:56Z
#url: https://api.github.com/gists/04eaf703dea9b13481a5a6aa313c313e
#owner: https://api.github.com/users/zackbunch

import pandas as pd

# Sample data frame
df = pd.DataFrame({
    'Department': ['A', 'B', 'C', 'A', 'B', 'C'],
    'ID': ['001', '002', '003', '004', '005', '006'],
    'Revision': [1, 2, 3, 1, 2, 3],
    'Name': ['Project A', 'Project B', 'Project C', 'Project D', 'Project E', 'Project F'],
    'WF Status': ['In Progress', 'Complete', 'In Progress', 'In Progress', 'Complete', 'In Progress'],
    'Date Released': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06']
})

# Group the data frame by department and then have the other attributes below
df_grouped = df.groupby('Department').apply(lambda x: pd.DataFrame({
    'ID': x['ID'].tolist(),
    'Revision': x['Revision'].tolist(),
    'Name': x['Name'].tolist(),
    'WF Status': x['WF Status'].tolist(),
    'Date Released': x['Date Released'].tolist()
})).reset_index(level=1, drop=True)

# Display the grouped data frame
print(df_grouped)
