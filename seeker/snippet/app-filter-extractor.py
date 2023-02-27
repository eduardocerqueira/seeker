#date: 2023-02-27T17:04:42Z
#url: https://api.github.com/gists/19338bbbb43eb8bd88a5c1f21bb99648
#owner: https://api.github.com/users/charleseiq

import json
import pandas as pd

view_filters = pd.read_gbq("""
SELECT o.name, JSON_QUERY(PARSE_JSON(o.view_json), '$.filters') AS filters FROM eiq-production.sedgwick_app.claim_views AS o
INNER JOIN (
  SELECT name, MAX(time_created) AS time_created 
  FROM eiq-production.sedgwick_app.claim_views
  WHERE module LIKE 'wc'
  GROUP BY name  
) AS l
ON o.name = l.name
AND o.time_created = l.time_created
WHERE 1 = 1
AND o.name NOT IN ('Parking Lot', 'My Assigned Claims', 'Waiting on Info', 'Pending CE Review', 'Pending Feedback')
AND user_id IS NULL
""")

def convert_metadata_to_string(field, id, operator, type, value, values):
    if type in ['BOOLEAN', 'STRICT_BOOLEAN']:
        result = [field]
        if not int(value):
            result = ['NOT'] + result
    else:
        result = [field, operator, value if value else str(values).replace('[', '(').replace(']', ')')]
    
    return ' '.join(result)

view_filters.filters = view_filters.filters.apply(json.loads).apply(lambda x: [convert_metadata_to_string(**f) for f in x])

for name, row in view_filters.set_index('name').iterrows():
    print(name)
    filters = row.iloc[0]
    for filter in filters:
        print(f'\t{filter}')