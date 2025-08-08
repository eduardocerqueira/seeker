#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px

def visualize_user_activity_by_date(data, grouping='date', chart_type='line', output_file='user_activity.png', **kwargs):
    if 'date_joined' not in data.columns:
        raise ValueError("Column 'date_joined' not found in data.")
    
    data['date_joined'] = pd.to_datetime(data['date_joined'], errors='coerce')
    
    if grouping == 'date':
        grouped = data['date_joined'].dt.date.value_counts().sort_index()
    elif grouping == 'month':
        grouped = data['date_joined'].dt.to_period('M').value_counts().sort_index()
    elif grouping == 'year':
        grouped = data['date_joined'].dt.to_period('Y').value_counts().sort_index()
    else:
        raise ValueError(f"Unsupported grouping '{grouping}'.")
    
    if chart_type == 'line':
        fig = go.Figure(data=go.Scatter(x=grouped.index, y=grouped.values, mode='lines+markers'))
        fig.update_layout(title=f'User Activity Over Time (by {grouping})',
                          xaxis_title=f'{grouping.capitalize()}',
                          yaxis_title='Number of Users Joined')
    elif chart_type == 'bar':
        fig = go.Figure(data=go.Bar(x=grouped.index, y=grouped.values))
        fig.update_layout(title=f'User Activity Over Time (by {grouping})',
                          xaxis_title=f'{grouping.capitalize()}',
                          yaxis_title='Number of Users Joined')
    else:
        raise ValueError("Unsupported chart type. Use 'line' or 'bar'.")
    
    # Apply custom styles
    if 'color' in kwargs:
        fig.update_traces(marker_color=kwargs['color'])
    
    # Save as HTML for interactivity
    fig.write_html(output_file.replace('.png', '.html'))
    
    # Optionally save as static image
    if output_file.endswith('.png'):
        fig.write_image(output_file)

def visualize_chat_frequency(data, output_file='chat_frequency.png', **kwargs):
    if 'chat' not in data.columns or 'name' not in data.columns:
        raise ValueError("Required columns 'chat' or 'name' not found in data.")
    
    data['chat_count'] = data['chat'].str.split().str.len().fillna(0)
    chat_freq = data.groupby('name')['chat_count'].sum().sort_values(ascending=False)
    
    fig = go.Figure(data=[go.Bar(x=chat_freq.index, y=chat_freq.values)])
    fig.update_layout(title='Chat Frequency by User',
                      xaxis_title='User',
                      yaxis_title='Chat Count',
                      xaxis_tickangle=-45)
    
    # Apply custom styles
    if 'color' in kwargs:
        fig.update_traces(marker_color=kwargs['color'])
    
    fig.write_html(output_file.replace('.png', '.html'))
    if output_file.endswith('.png'):
        fig.write_image(output_file)

def visualize_online_status(data, output_file='online_status.png', **kwargs):
    if 'online' not in data.columns:
        raise ValueError("Column 'online' not found in data.")
    
    online_counts = data['online'].value_counts()
    
    fig = go.Figure(data=[go.Pie(labels=['Online' if status else 'Offline' for status in online_counts.index], 
                                 values=online_counts.values)])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title='User Online Status')
    
    # Apply custom styles
    if 'colors' in kwargs:
        fig.update_traces(marker=dict(colors=kwargs['colors']))
    
    fig.write_html(output_file.replace('.png', '.html'))
    if output_file.endswith('.png'):
        fig.write_image(output_file)

def visualize_activity_heatmap(data, output_file='activity_heatmap.png', **kwargs):
    if 'date_joined' not in data.columns:
        raise ValueError("Column 'date_joined' not found in data.")
    
    data['date_joined'] = pd.to_datetime(data['date_joined'], errors='coerce')
    
    simulated_data = []
    for _, row in data.iterrows():
        for _ in range(random.randint(0, 10)):  # Random number of actions
            action_time = row['date_joined'] + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            simulated_data.append({'name': row['name'], 'action_time': action_time})
    
    activity_data = pd.DataFrame(simulated_data)
    activity_data['hour'] = activity_data['action_time'].dt.hour
    activity_data['day'] = activity_data['action_time'].dt.day_name()
    
    heatmap_data = activity_data.pivot_table(index='day', columns='hour', values='name', aggfunc='count', fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis'
    ))
    fig.update_layout(title='User Activity Heatmap',
                      xaxis_title='Hour of Day',
                      yaxis_title='Day of Week')
    
    # Apply custom styles
    if 'colorscale' in kwargs:
        fig.update_traces(colorscale=kwargs['colorscale'])
    
    fig.write_html(output_file.replace('.png', '.html'))
    if output_file.endswith('.png'):
        fig.write_image(output_file)

# Performance optimization could involve chunking data for processing if datasets are extremely large, 
# but this would require more extensive refactoring based on actual data size and processing needs.