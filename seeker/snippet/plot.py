#date: 2025-08-15T16:56:35Z
#url: https://api.github.com/gists/a69dfcb7bdbefe7e384fbbe2668ce4d1
#owner: https://api.github.com/users/farseerfc

#!/usr/bin/python
import pandas as pd
import plotly.graph_objects as go
import re

df = pd.read_csv('linux-firmware.csv', header=None, names=['timestamp_str', 'package_name', 'filesize_str'])
df['timestamp'] = pd.to_datetime(df['timestamp_str'], utc=True)

def parse_filesize(size_str):
    match = re.match(r'([\d.]+)\s*([KM]iB)', size_str)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)
    if unit == 'KiB':
        return value * 1024
    elif unit == 'MiB':
        return value * 1024 * 1024
    else:
        return None

df['filesize'] = df['filesize_str'].apply(parse_filesize)

df.dropna(subset=['filesize'], inplace=True)

def extract_prefix(package_name):
    match = re.match(r'([a-zA-Z0-9-]+?)-[0-9]+.*', package_name.strip())
    if match:
        return match.group(1)
    return package_name.strip()
df['package_prefix'] = df['package_name'].apply(extract_prefix)


df_stacked = df.pivot_table(
    index=df['timestamp'].dt.date,
    columns='package_prefix',
    values=['filesize', 'filesize_str'],
    aggfunc={'filesize': 'sum', 'filesize_str': lambda x: '<br>'.join(x)}
).fillna('')
df_stacked.columns = [f'{col[0]}_{col[1]}' for col in df_stacked.columns]
df_stacked.reset_index(inplace=True)

fig = go.Figure()

unique_prefixes = df['package_prefix'].unique()

for prefix in unique_prefixes:
    fig.add_trace(go.Bar(
        x=df_stacked['timestamp'],
        y=df_stacked[f'filesize_{prefix}'],
        name=prefix,
        marker_line_width=0,
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                      '<b>Package:</b> ' + prefix + '<br>' +
                      '<b>Size:</b> %{customdata}',
        customdata=df_stacked[f'filesize_str_{prefix}'],
        width=[24 * 60 * 60 * 1000 * 5] * len(df_stacked)
    ))

fig.update_layout(
    barmode='stack',
    title={
        'text': 'File Size of Packages Over Time (Stacked)',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='Timestamp',
    yaxis_title='File Size (MiB)',
    font=dict(family="Arial", size=12, color="Black"),
    legend_title_text='Package Prefix'
)

fig.write_html("plot.html")
fig.show()