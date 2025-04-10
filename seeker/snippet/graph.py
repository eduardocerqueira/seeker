#date: 2025-04-10T16:59:03Z
#url: https://api.github.com/gists/22af75936b8c5ca647421cf8b1686dba
#owner: https://api.github.com/users/timzolleis

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def frequency_line_plot(data_frame):
    """
    Creates a seaborn line plot to show start frequencies by class over time.
    
    Parameters:
    -----------
    data_frame : pandas.DataFrame
        DataFrame containing solar flare data with columns:
        - start_datetime: datetime object for when the flare started
        - start_freq_khz: frequency in kHz
        - x_ray_flare_imp: flare class and magnitude (e.g., 'X1.2', 'M5.7')
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    # Make a copy to avoid modifying original
    df = data_frame.copy()
    
    # Ensure datetime column is properly formatted
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    
    # Extract flare class from x_ray_flare_imp
    df['flare_class'] = df['x_ray_flare_imp'].str[0]
    
    # Create a year-month column for aggregation
    df['year_month'] = df['start_datetime'].dt.strftime('%Y-%m')
    
    # Group by year-month and flare class, calculate mean frequency
    grouped_df = df.groupby(['year_month', 'flare_class'])['start_freq_khz'].mean().reset_index()
    
    # Convert year_month back to datetime for proper plotting
    grouped_df['date'] = pd.to_datetime(grouped_df['year_month'] + '-01')
    grouped_df = grouped_df.sort_values('date')
    
    # Define a custom color palette for flare classes
    class_colors = {
        'A': '#86C5E3',  # Light Blue
        'B': '#4682B4',  # Steel Blue
        'C': '#FFD700',  # Gold
        'M': '#FF8C00',  # Dark Orange
        'X': '#B22222',   # Fire Brick (deep red)
        'E': '#86C5E3',
        '-': '#4682B4'
    }
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the data using seaborn
    sns.lineplot(
        data=grouped_df,
        x='date',
        y='start_freq_khz',
        hue='flare_class',
        palette=class_colors,
        marker='o',
        linewidth=2.5,
        markersize=6,
        ax=ax
    )
    
    # Add labels and title
    plt.title('Solar Flare Start Frequencies by Class Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Start Frequency (kHz)', fontsize=14)
    plt.legend(title='Flare Class', fontsize=12)
    
    # Format x-axis for dates
    plt.gcf().autofmt_xdate()
    
    # Improve y-axis (log scale often works better for frequency data)
    plt.yscale('log')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
