#date: 2025-04-02T16:48:49Z
#url: https://api.github.com/gists/2576980948af17803de7d61e8ecb131b
#owner: https://api.github.com/users/ianmcook

import pandas as pd
import pyarrow as pa

file_path = 'fruit.arrow'
stream_path = 'fruit.arrows'

df = pd.DataFrame(data={'fruit': ['apple', 'apple', 'apple', 'orange', 'orange', 'orange'],
                        'variety': ['gala', 'honeycrisp', 'fuji', 'navel', 'valencia', 'cara cara'],
                        'weight': [134.2 , 158.6, None, 142.1, 96.7, None]})

table = pa.Table.from_pandas(df, preserve_index=False)
table = table.replace_schema_metadata(None)

# write file in Arrow IPC file format
with pa.ipc.new_file(file_path, table.schema) as writer:
  writer.write_table(table)

# write file in Arrow IPC stream format
with pa.ipc.new_stream(stream_path, table.schema) as writer:
  writer.write_table(table)
