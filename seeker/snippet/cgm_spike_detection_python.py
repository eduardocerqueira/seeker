#date: 2025-02-26T16:49:40Z
#url: https://api.github.com/gists/57afc16db57ef8127fa3d4458353ef41
#owner: https://api.github.com/users/bigsnarfdude

"""
CGM Spike Detection Algorithm

This module provides functions to analyze Continuous Glucose Monitoring (CGM) data
and detect various patterns including rapid rises, sustained high periods, 
meal responses, and hypoglycemic events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta


def read_cgm_data(file_path):
    """
    Read CGM data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Processed CGM data
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time
    df = df.sort_values(by='time')
    
    return df


def detect_rapid_rises(data, threshold=30, window_minutes=30, 
                      hyperglycemia_threshold=180, min_rate=1.5):
    """
    Detect rapid rises in glucose levels.
    
    Parameters:
    data (pandas.DataFrame): Glucose data for a single subject
    threshold (int): Minimum mg/dL increase to be considered a rise
    window_minutes (int): Time window in minutes to detect rise
    hyperglycemia_threshold (int): Threshold for hyperglycemia (mg/dL)
    min_rate (float): Minimum rate of change (mg/dL/min)
    
    Returns:
    list: Detected rapid rise events
    """
    rapid_rises = []
    
    for i in range(len(data) - 1):
        current = data.iloc[i]
        
        # Define window end time
        window_end = current['time'] + pd.Timedelta(minutes=window_minutes)
        
        # Get all readings within the window
        window_data = data[(data['time'] > current['time']) & 
                           (data['time'] <= window_end)]
        
        if len(window_data) == 0:
            continue
        
        # Find the highest reading in the window
        highest = window_data.loc[window_data['gl'].idxmax()]
        
        # Calculate glucose change
        glucose_change = highest['gl'] - current['gl']
        
        # Check if it meets criteria
        if (glucose_change >= threshold and 
            highest['gl'] >= hyperglycemia_threshold):
            
            # Calculate time difference in minutes
            time_diff = (highest['time'] - current['time']).total_seconds() / 60
            
            # Calculate rate of change
            rate_of_change = glucose_change / time_diff
            
            if rate_of_change >= min_rate:
                rapid_rises.append({
                    'start_time': current['time'],
                    'end_time': highest['time'],
                    'start_glucose': current['gl'],
                    'end_glucose': highest['gl'],
                    'glucose_change': glucose_change,
                    'time_change_minutes': time_diff,
                    'rate_of_change': rate_of_change
                })
                
                # Skip ahead to avoid counting the same rise multiple times
                i = data.index.get_loc(highest.name)
    
    return rapid_rises


def detect_sustained_periods(data, threshold=180, min_duration_minutes=30):
    """
    Detect sustained periods where glucose exceeds a threshold.
    
    Parameters:
    data (pandas.DataFrame): Glucose data for a single subject
    threshold (int): Glucose threshold in mg/dL
    min_duration_minutes (int): Minimum duration to consider
    
    Returns:
    list: Sustained periods
    """
    sustained_periods = []
    
    # Find readings above threshold
    high_readings = data[data['gl'] >= threshold]
    
    if len(high_readings) == 0:
        return sustained_periods
    
    # Initialize tracking variables
    in_period = False
    period_start = None
    period_readings = []
    
    # Check for gaps to segment into periods
    for i, row in data.iterrows():
        if row['gl'] >= threshold:
            if not in_period:
                in_period = True
                period_start = row
            period_readings.append(row)
        else:
            if in_period:
                # End of a period, check duration
                period_end = period_readings[-1]
                duration_minutes = (period_end['time'] - period_start['time']).total_seconds() / 60
                
                if duration_minutes >= min_duration_minutes:
                    sustained_periods.append({
                        'start_time': period_start['time'],
                        'end_time': period_end['time'],
                        'duration_minutes': duration_minutes,
                        'avg_glucose': np.mean([r['gl'] for r in period_readings]),
                        'max_glucose': max([r['gl'] for r in period_readings]),
                        'readings_count': len(period_readings)
                    })
                
                # Reset tracking
                in_period = False
                period_readings = []
    
    # Handle case where dataset ends during a high period
    if in_period and len(period_readings) > 0:
        period_end = period_readings[-1]
        duration_minutes = (period_end['time'] - period_start['time']).total_seconds() / 60
        
        if duration_minutes >= min_duration_minutes:
            sustained_periods.append({
                'start_time': period_start['time'],
                'end_time': period_end['time'],
                'duration_minutes': duration_minutes,
                'avg_glucose': np.mean([r['gl'] for r in period_readings]),
                'max_glucose': max([r['gl'] for r in period_readings]),
                'readings_count': len(period_readings)
            })
    
    return sustained_periods


def detect_meal_responses(data, threshold=40, window_minutes=120):
    """
    Detect potential meal response patterns.
    
    Parameters:
    data (pandas.DataFrame): Glucose data for a single subject
    threshold (int): Minimum glucose increase to consider as meal response
    window_minutes (int): Maximum window in minutes for meal response
    
    Returns:
    list: Detected meal response events
    """
    meal_responses = []
    
    for i in range(len(data) - 1):
        current = data.iloc[i]
        
        # Define window end time
        window_end = current['time'] + pd.Timedelta(minutes=window_minutes)
        
        # Get all readings within the window
        window_data = data[(data['time'] > current['time']) & 
                           (data['time'] <= window_end)]
        
        if len(window_data) == 0:
            continue
        
        # Find the highest reading in the window
        highest = window_data.loc[window_data['gl'].idxmax()]
        
        # Calculate glucose change
        glucose_change = highest['gl'] - current['gl']
        
        # Check if it meets criteria for meal response
        if glucose_change >= threshold:
            time_diff = (highest['time'] - current['time']).total_seconds() / 60
            
            # Typical meal responses happen between 30-120 minutes
            if 30 <= time_diff <= window_minutes:
                meal_responses.append({
                    'start_time': current['time'],
                    'peak_time': highest['time'],
                    'baseline_glucose': current['gl'],
                    'peak_glucose': highest['gl'],
                    'rise': glucose_change,
                    'time_to_peak_minutes': time_diff
                })
                
                # Skip ahead to avoid multiple counting
                i = data.index.get_loc(highest.name)
    
    return meal_responses


def detect_hypoglycemic_events(data, threshold=70, recovery_threshold=80):
    """
    Detect hypoglycemic events.
    
    Parameters:
    data (pandas.DataFrame): Glucose data for a single subject
    threshold (int): Hypoglycemia threshold in mg/dL
    recovery_threshold (int): Glucose level considered recovery from hypoglycemia
    
    Returns:
    list: Detected hypoglycemic events
    """
    hypo_events = []
    
    # Find readings below threshold
    low_readings = data[data['gl'] < threshold]
    
    if len(low_readings) == 0:
        return hypo_events
    
    # Initialize tracking variables
    in_event = False
    event_start = None
    event_readings = []
    
    # Check for gaps to segment into events
    for i, row in data.iterrows():
        if row['gl'] < threshold:
            if not in_event:
                in_event = True
                event_start = row
            event_readings.append(row)
        elif in_event and row['gl'] >= recovery_threshold:
            # End of an event due to recovery
            event_end = event_readings[-1]
            duration_minutes = (event_end['time'] - event_start['time']).total_seconds() / 60
            
            hypo_events.append({
                'start_time': event_start['time'],
                'end_time': event_end['time'],
                'duration_minutes': duration_minutes,
                'min_glucose': min([r['gl'] for r in event_readings]),
                'avg_glucose': np.mean([r['gl'] for r in event_readings]),
                'readings_count': len(event_readings)
            })
            
            # Reset tracking
            in_event = False
            event_readings = []
        elif in_event:
            # Still in recovery zone but not below threshold
            event_readings.append(row)
    
    # Handle case where dataset ends during a hypo event
    if in_event and len(event_readings) > 0:
        event_end = event_readings[-1]
        duration_minutes = (event_end['time'] - event_start['time']).total_seconds() / 60
        
        hypo_events.append({
            'start_time': event_start['time'],
            'end_time': event_end['time'],
            'duration_minutes': duration_minutes,
            'min_glucose': min([r['gl'] for r in event_readings]),
            'avg_glucose': np.mean([r['gl'] for r in event_readings]),
            'readings_count': len(event_readings)
        })
    
    return hypo_events


def detect_glucose_patterns(df, params=None):
    """
    Detect various glucose patterns in CGM data.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns 'id', 'time', and 'gl'
    params (dict, optional): Parameter dictionary for customization
    
    Returns:
    dict: Dictionary containing detected patterns
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'rapid_rise_threshold': 30,      # mg/dL
            'rapid_rise_window': 30,         # minutes
            'hyperglycemia_threshold': 180,  # mg/dL
            'hypoglycemia_threshold': 70,    # mg/dL
            'meal_response_window': 120,     # minutes
            'meal_response_threshold': 40,   # mg/dL
            'min_rate_of_change': 1.5        # mg/dL/min
        }
    
    # Group by subject ID
    grouped = df.groupby('id')
    
    # Initialize results dictionary
    results = {}
    
    # Process each subject
    for subject_id, subject_data in grouped:
        # Sort by time to ensure chronological order
        subject_data = subject_data.sort_values(by='time')
        
        # Initialize pattern containers for this subject
        subject_results = {
            'rapid_rises': [],
            'sustained_high': [],
            'meal_responses': [],
            'hypoglycemic_events': []
        }
        
        # 1. Detect rapid rises
        rapid_rises = detect_rapid_rises(
            subject_data, 
            threshold=params['rapid_rise_threshold'],
            window_minutes=params['rapid_rise_window'],
            hyperglycemia_threshold=params['hyperglycemia_threshold'],
            min_rate=params['min_rate_of_change']
        )
        subject_results['rapid_rises'] = rapid_rises
        
        # 2. Detect sustained high periods
        sustained_high = detect_sustained_periods(
            subject_data, 
            threshold=params['hyperglycemia_threshold'],
            min_duration_minutes=30
        )
        subject_results['sustained_high'] = sustained_high
        
        # 3. Detect meal responses
        meal_responses = detect_meal_responses(
            subject_data,
            threshold=params['meal_response_threshold'],
            window_minutes=params['meal_response_window']
        )
        subject_results['meal_responses'] = meal_responses
        
        # 4. Detect hypoglycemic events
        hypo_events = detect_hypoglycemic_events(
            subject_data,
            threshold=params['hypoglycemia_threshold']
        )
        subject_results['hypoglycemic_events'] = hypo_events
        
        # Store results for this subject
        results[subject_id] = subject_results
    
    return results


def calculate_statistics(results):
    """
    Calculate overall statistics from the detected patterns.
    
    Parameters:
    results (dict): Results from detect_glucose_patterns
    
    Returns:
    dict: Statistics summary
    """
    stats = {
        'total_rapid_rises': 0,
        'total_sustained_high': 0,
        'total_meal_responses': 0,
        'total_hypo_events': 0,
        'avg_rise_magnitude': 0,
        'avg_hyper_duration': 0,
        'highest_glucose': 0,
        'lowest_glucose': 1000,  # Start high, will find minimum
        'subjects_with_hypo': 0
    }
    
    total_rises = []
    total_sustained = []
    
    for subject_id, subject_results in results.items():
        # Count events
        stats['total_rapid_rises'] += len(subject_results['rapid_rises'])
        stats['total_sustained_high'] += len(subject_results['sustained_high'])
        stats['total_meal_responses'] += len(subject_results['meal_responses'])
        stats['total_hypo_events'] += len(subject_results['hypoglycemic_events'])
        
        # Track if subject had hypo events
        if len(subject_results['hypoglycemic_events']) > 0:
            stats['subjects_with_hypo'] += 1
        
        # Collect all rises for averaging
        total_rises.extend(subject_results['rapid_rises'])
        total_sustained.extend(subject_results['sustained_high'])
        
        # Track highest/lowest glucose
        for rise in subject_results['rapid_rises']:
            if rise['end_glucose'] > stats['highest_glucose']:
                stats['highest_glucose'] = rise['end_glucose']
        
        for hypo in subject_results['hypoglycemic_events']:
            if hypo['min_glucose'] < stats['lowest_glucose']:
                stats['lowest_glucose'] = hypo['min_glucose']
    
    # Calculate averages
    if len(total_rises) > 0:
        stats['avg_rise_magnitude'] = sum(rise['glucose_change'] for rise in total_rises) / len(total_rises)
    
    if len(total_sustained) > 0:
        stats['avg_hyper_duration'] = sum(period['duration_minutes'] for period in total_sustained) / len(total_sustained)
    
    return stats


def visualize_glucose_patterns(df, results, subject_id=None):
    """
    Visualize glucose patterns and detected events.
    
    Parameters:
    df (pandas.DataFrame): Original glucose data
    results (dict): Results from detect_glucose_patterns
    subject_id (str, optional): Specific subject to visualize
    
    Returns:
    matplotlib.Figure: Figure with visualizations
    """
    if subject_id is None and len(results) > 0:
        # Use the first subject if none specified
        subject_id = list(results.keys())[0]
    
    if subject_id not in results:
        print(f"Subject {subject_id} not found in results")
        return None
    
    # Filter data for this subject
    subject_data = df[df['id'] == subject_id].sort_values(by='time')
    subject_results = results[subject_id]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot glucose profile
    axes[0].plot(subject_data['time'], subject_data['gl'], 'b-', label='Glucose')
    axes[0].set_title(f'Glucose Profile for {subject_id}')
    axes[0].set_ylabel('Glucose (mg/dL)')
    axes[0].grid(True)
    
    # Add horizontal lines for thresholds
    axes[0].axhline(y=180, color='r', linestyle='--', label='Hyperglycemia')
    axes[0].axhline(y=70, color='orange', linestyle='--', label='Hypoglycemia')
    
    # Mark rapid rises
    for rise in subject_results['rapid_rises']:
        axes[0].plot(
            [rise['start_time'], rise['end_time']], 
            [rise['start_glucose'], rise['end_glucose']], 
            'r-', linewidth=2
        )
        axes[0].scatter(
            rise['end_time'], 
            rise['end_glucose'], 
            color='r', s=50, zorder=5,
            label='_nolegend_'
        )
    
    # Mark hypoglycemic events
    for hypo in subject_results['hypoglycemic_events']:
        axes[0].axvspan(
            hypo['start_time'], 
            hypo['end_time'], 
            alpha=0.2, 
            color='orange',
            label='_nolegend_'
        )
    
    # Create event timeline
    event_types = []
    event_times = []
    event_labels = []
    event_colors = []
    
    # Add rapid rises to timeline
    for rise in subject_results['rapid_rises']:
        event_types.append('Rapid Rise')
        event_times.append(rise['start_time'])
        event_labels.append(f"+{rise['glucose_change']} mg/dL")
        event_colors.append('red')
    
    # Add meal responses to timeline
    for meal in subject_results['meal_responses']:
        event_types.append('Meal Response')
        event_times.append(meal['start_time'])
        event_labels.append(f"+{meal['rise']} mg/dL")
        event_colors.append('green')
    
    # Add hypo events to timeline
    for hypo in subject_results['hypoglycemic_events']:
        event_types.append('Hypoglycemia')
        event_times.append(hypo['start_time'])
        event_labels.append(f"{hypo['min_glucose']} mg/dL")
        event_colors.append('orange')
    
    # Plot event timeline
    for i, (event_type, event_time, label, color) in enumerate(zip(event_types, event_times, event_labels, event_colors)):
        axes[1].scatter(event_time, 0, color=color, s=100, marker='o')
        axes[1].text(event_time, 0.1, label, rotation=45, ha='left')
        axes[1].text(event_time, -0.1, event_type, rotation=45, ha='left')
    
    # Configure timeline axis
    axes[1].set_title('Event Timeline')
    axes[1].set_yticks([])
    axes[1].set_xlabel('Time')
    axes[1].grid(True, axis='x')
    
    # Add legends and adjust layout
    axes[0].legend()
    fig.tight_layout()
    
    return fig


def visualize_glucose_patterns_with_meals(df, results, subject_id=None, meal_times=None):
    """
    Visualize glucose patterns and detected events with meal markers.
    
    Parameters:
    df (pandas.DataFrame): Original glucose data
    results (dict): Results from detect_glucose_patterns
    subject_id (str, optional): Specific subject to visualize
    meal_times (dict, optional): Dictionary with meal times information
                                 e.g., {'breakfast': [datetime objects],
                                        'lunch': [datetime objects],
                                        'dinner': [datetime objects]}
                                 If None, will use typical meal times based on data range
    
    Returns:
    matplotlib.Figure: Figure with visualizations
    """
    if subject_id is None and len(results) > 0:
        # Use the first subject if none specified
        subject_id = list(results.keys())[0]
    
    if subject_id not in results:
        print(f"Subject {subject_id} not found in results")
        return None
    
    # Filter data for this subject
    subject_data = df[df['id'] == subject_id].sort_values(by='time')
    subject_results = results[subject_id]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot glucose profile
    axes[0].plot(subject_data['time'], subject_data['gl'], 'b-', label='Glucose')
    axes[0].set_title(f'Glucose Profile for {subject_id}')
    axes[0].set_ylabel('Glucose (mg/dL)')
    axes[0].grid(True)
    
    # Add horizontal lines for thresholds
    axes[0].axhline(y=180, color='r', linestyle='--', label='Hyperglycemia')
    axes[0].axhline(y=70, color='orange', linestyle='--', label='Hypoglycemia')
    
    # Mark rapid rises
    for rise in subject_results['rapid_rises']:
        axes[0].plot(
            [rise['start_time'], rise['end_time']], 
            [rise['start_glucose'], rise['end_glucose']], 
            'r-', linewidth=2
        )
        axes[0].scatter(
            rise['end_time'], 
            rise['end_glucose'], 
            color='r', s=50, zorder=5,
            label='_nolegend_'
        )
    
    # Mark hypoglycemic events
    for hypo in subject_results['hypoglycemic_events']:
        axes[0].axvspan(
            hypo['start_time'], 
            hypo['end_time'], 
            alpha=0.2, 
            color='orange',
            label='_nolegend_'
        )
    
    # Add meal time markers
    if meal_times is None:
        # If no meal times provided, estimate based on typical meal times
        # Get the date range in the data
        min_date = subject_data['time'].min().date()
        max_date = subject_data['time'].max().date()
        
        breakfast_times = []
        lunch_times = []
        dinner_times = []
        
        # Generate typical meal times for each day in the data
        current_date = min_date
        while current_date <= max_date:
            # Typical meal times - adjust as needed
            breakfast_time = datetime.combine(current_date, time(7, 30))  # 7:30 AM
            lunch_time = datetime.combine(current_date, time(12, 30))     # 12:30 PM
            dinner_time = datetime.combine(current_date, time(18, 30))    # 6:30 PM
            
            breakfast_times.append(breakfast_time)
            lunch_times.append(lunch_time)
            dinner_times.append(dinner_time)
            
            current_date += timedelta(days=1)
        
        meal_times = {
            'breakfast': breakfast_times,
            'lunch': lunch_times,
            'dinner': dinner_times
        }
    
    # Add meal markers to the plot
    meal_colors = {'breakfast': 'green', 'lunch': 'purple', 'dinner': 'brown'}
    meal_markers = []
    
    for meal_name, times in meal_times.items():
        for meal_time in times:
            # Check if this meal time is within our data range
            if (subject_data['time'].min() <= meal_time <= subject_data['time'].max()):
                # Add vertical line for meal time
                line = axes[0].axvline(x=meal_time, color=meal_colors[meal_name], 
                                     linestyle='-', alpha=0.7, linewidth=1.5)
                
                # Only add to legend once
                if meal_name not in meal_markers:
                    line.set_label(meal_name.capitalize())
                    meal_markers.append(meal_name)
    
    # Create event timeline (bottom panel)
    event_types = []
    event_times = []
    event_labels = []
    event_colors = []
    
    # Add rapid rises to timeline
    for rise in subject_results['rapid_rises']:
        event_types.append('Rapid Rise')
        event_times.append(rise['start_time'])
        event_labels.append(f"+{rise['glucose_change']} mg/dL")
        event_colors.append('red')
    
    # Add meal responses to timeline
    for meal in subject_results['meal_responses']:
        event_types.append('Meal Response')
        event_times.append(meal['start_time'])
        event_labels.append(f"+{meal['rise']} mg/dL")
        event_colors.append('green')
    
    # Add hypo events to timeline
    for hypo in subject_results['hypoglycemic_events']:
        event_types.append('Hypoglycemia')
        event_times.append(hypo['start_time'])
        event_labels.append(f"{hypo['min_glucose']} mg/dL")
        event_colors.append('orange')
    
    # Plot event timeline
    for i, (event_type, event_time, label, color) in enumerate(zip(event_types, event_times, event_labels, event_colors)):
        axes[1].scatter(event_time, 0, color=color, s=100, marker='o')
        axes[1].text(event_time, 0.1, label, rotation=45, ha='left')
        axes[1].text(event_time, -0.1, event_type, rotation=45, ha='left')
    
    # Also add meal times to the timeline
    for meal_name, times in meal_times.items():
        for meal_time in times:
            if (subject_data['time'].min() <= meal_time <= subject_data['time'].max()):
                axes[1].scatter(meal_time, 0, color=meal_colors[meal_name], 
                              marker='|', s=100)
                axes[1].text(meal_time, -0.2, meal_name.capitalize(), 
                           rotation=45, ha='left', color=meal_colors[meal_name])
    
    # Configure timeline axis
    axes[1].set_title('Event Timeline')
    axes[1].set_yticks([])
    axes[1].set_xlabel('Time')
    axes[1].grid(True, axis='x')
    
    # Format x-axis to show dates better
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add legends and adjust layout
    axes[0].legend()
    fig.tight_layout()
    
    return fig


def estimate_insulin_needs(results, isf=None, target_glucose=120):
    """
    Estimate insulin needs based on detected patterns.
    
    Parameters:
    results (dict): Results from detect_glucose_patterns
    isf (float, optional): Insulin Sensitivity Factor (mg/dL per unit)
    target_glucose (int): Target glucose level in mg/dL
    
    Returns:
    dict: Insulin estimates by subject
    """
    insulin_estimates = {}
    
    for subject_id, subject_results in results.items():
        # Determine ISF if not provided (using 1800 rule with assumed TDD)
        if isf is None:
            assumed_tdd = 40  # Total Daily Dose assumption
            subject_isf = 1800 / assumed_tdd
        else:
            subject_isf = isf
        
        # Find highest glucose reading from sustained high periods
        highest_glucose = 0
        for period in subject_results['sustained_high']:
            if period['max_glucose'] > highest_glucose:
                highest_glucose = period['max_glucose']
        
        # If no high periods, check rapid rises
        if highest_glucose == 0 and len(subject_results['rapid_rises']) > 0:
            highest_glucose = max(rise['end_glucose'] for rise in subject_results['rapid_rises'])
        
        # Calculate correction dose
        if highest_glucose > target_glucose:
            correction_dose = (highest_glucose - target_glucose) / subject_isf
        else:
            correction_dose = 0
        
        insulin_estimates[subject_id] = {
            'isf': subject_isf,
            'highest_glucose': highest_glucose,
            'correction_dose': correction_dose,
            'target_glucose': target_glucose
        }
    
    return insulin_estimates


def export_to_r(results, file_path='cgm_analysis_results.csv'):
    """
    Export detection results to a CSV format that can be easily imported into R.
    
    Parameters:
    results (dict): Results from detect_glucose_patterns
    file_path (str): Path to save the CSV file
    
    Returns:
    None
    """
    # Prepare data for each event type
    rapid_rises_data = []
    sustained_high_data = []
    meal_responses_data = []
    hypo_events_data = []
    
    for subject_id, subject_results in results.items():
        # Process rapid rises
        for event in subject_results['rapid_rises']:
            event_data = {
                'subject_id': subject_id,
                'event_type': 'rapid_rise',
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'start_glucose': event['start_glucose'],
                'end_glucose': event['end_glucose'],
                'glucose_change': event['glucose_change'],
                'time_change_minutes': event['time_change_minutes'],
                'rate_of_change': event['rate_of_change']
            }
            rapid_rises_data.append(event_data)
        
        # Process sustained high periods
        for event in subject_results['sustained_high']:
            event_data = {
                'subject_id': subject_id,
                'event_type': 'sustained_high',
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'duration_minutes': event['duration_minutes'],
                'avg_glucose': event['avg_glucose'],
                'max_glucose': event['max_glucose']
            }
            sustained_high_data.append(event_data)
        
        # Process meal responses
        for event in subject_results['meal_responses']:
            event_data = {
                'subject_id': subject_id,
                'event_type': 'meal_response',
                'start_time': event['start_time'],
                'peak_time': event['peak_time'],
                'baseline_glucose': event['baseline_glucose'],
                'peak_glucose': event['peak_glucose'],
                'rise': event['rise'],
                'time_to_peak_minutes': event['time_to_peak_minutes']
            }
            meal_responses_data.append(event_data)
        
        # Process hypoglycemic events
        for event in subject_results['hypoglycemic_events']:
            event_data = {
                'subject_id': subject_id,
                'event_type': 'hypoglycemia',
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'duration_minutes': event['duration_minutes'],
                'min_glucose': event['min_glucose'],
                'avg_glucose': event['avg_glucose']
            }
            hypo_events_data.append(event_data)
    
    # Create DataFrames for each event type
    rapid_rises_df = pd.DataFrame(rapid_rises_data) if rapid_rises_data else pd.DataFrame()
    sustained_high_df = pd.DataFrame(sustained_high_data) if sustained_high_data else pd.DataFrame()
    meal_responses_df = pd.DataFrame(meal_responses_data) if meal_responses_data else pd.DataFrame()
    hypo_events_df = pd.DataFrame(hypo_events_data) if hypo_events_data else pd.DataFrame()
    
    # Combine all event types (will have some NA columns)
    all_events = pd.concat([
        rapid_rises_df,
        sustained_high_df,
        meal_responses_df,
        hypo_events_df
    ], ignore_index=True)
    
    # Save to CSV
    all_events.to_csv(file_path, index=False)
    print(f"Results exported to {file_path}")
    
    return None


def main():
    """
    Main function to demonstrate usage of the CGM spike detection.
    """
    # Example usage
    print("CGM Spike Detection Algorithm")
    print("-----------------------------")
    
    # 1. Read data
    file_path = "cgm.csv"  # Update with actual file path
    df = read_cgm_data(file_path)
    print(f"Loaded data for {df['id'].nunique()} subjects with {len(df)} readings")
    
    # 2. Customize parameters (optional)
    params = {
        'rapid_rise_threshold': 30,      # mg/dL
        'rapid_rise_window': 30,         # minutes
        'hyperglycemia_threshold': 180,  # mg/dL
        'hypoglycemia_threshold': 70,    # mg/dL
        'meal_response_window': 120,     # minutes
        'meal_response_threshold': 40,   # mg/dL
        'min_rate_of_change': 1.5        # mg/dL/min
    }
    
    # 3. Detect patterns
    results = detect_glucose_patterns(df, params)
    
    # 4. Print summary statistics
    stats = calculate_statistics(results)
    print("\nSummary Statistics:")
    print(f"Total rapid rises detected: {stats['total_rapid_rises']}")
    print(f"Total sustained hyperglycemia periods: {stats['total_sustained_high']}")
    print(f"Total potential meal responses: {stats['total_meal_responses']}")
    print(f"Total hypoglycemic events: {stats['total_hypo_events']}")
    print(f"Average rise magnitude: {stats['avg_rise_magnitude']:.1f} mg/dL")
    print(f"Average hyperglycemia duration: {stats['avg_hyper_duration']:.1f} minutes")
    print(f"Highest glucose detected: {stats['highest_glucose']} mg/dL")
    print(f"Lowest glucose detected: {stats['lowest_glucose']} mg/dL")
    
    # 5. Estimate insulin needs
    insulin_estimates = estimate_insulin_needs(results)
    print("\nEstimated Insulin Needs:")
    for subject_id, estimate in insulin_estimates.items():
        print(f"{subject_id}: Correction dose of {estimate['correction_dose']:.1f} units " +
              f"for highest reading of {estimate['highest_glucose']} mg/dL")
    
    # 6. Export to R-compatible format
    export_to_r(results, 'cgm_analysis_results.csv')
    
    # 7. Visualize results for the first subject WITH MEAL MARKERS
    if len(results) > 0:
        first_subject = list(results.keys())[0]
        
        # Generate standard visualization
        fig = visualize_glucose_patterns(df, results, first_subject)
        plt.savefig(f"{first_subject}_glucose_analysis.png")
        print(f"Standard visualization saved for {first_subject}")
        
        # Generate visualization with meal markers
        fig_with_meals = visualize_glucose_patterns_with_meals(df, results, first_subject)
        plt.savefig(f"{first_subject}_glucose_analysis_with_meals.png")
        print(f"Visualization with meal markers saved for {first_subject}")
        
        # Show the plots if running in interactive mode
        plt.show()


if __name__ == "__main__":
    main()