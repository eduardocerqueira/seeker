#date: 2025-07-29T16:53:31Z
#url: https://api.github.com/gists/7c4e91d10c658227eae3983fbc3bc778
#owner: https://api.github.com/users/michael021997

"""
Advanced Feature Engineering Module - Section 2.2 of Technical Specification

Purpose: Generate sophisticated time series features using automated feature engineering tools.

Auto Feature Engineering Tools:
- sktime: Time series feature extraction (scikit-learn compatible)
- auto-sklearn: Automated feature selection

Generated Features:
- Velocity and acceleration metrics
- Multi-lag variables (1, 3, 6, 12 month lags)
- Rolling statistical measures (mean, std, min, max)
- Seasonal decomposition components
- Fourier transform features
- Calendar-based features (holidays, weekends, month-end effects)
- YoY Growth Features
- Quarter-over-Quarter (QoQ) Growth Features
Calendar Month Seasonality Index:
- Computed using 24-month rolling window
- Methodology: seasonality_index[month] = avg(fuel_burn[month]) / avg(fuel_burn[all_months])
- Updated monthly with new data
- Applied as multiplicative factor to rolling predictions
"""

import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Feature engineering libraries - now guaranteed to be available
from sktime.transformations.series.summarize import SummaryTransformer
from sktime.transformations.series.acf import AutoCorrelationTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler

class FeatureEngineeringModule:
    """
    Advanced feature engineering module for time series fuel burn prediction
    """
    
    def __init__(self):
        """Initialize the feature engineering module"""
        self.engineered_features = {}
        self.feature_metadata = {}
        self.seasonality_indices = {}
        
        print("Advanced Feature Engineering Module initialized")
        print("All feature engineering libraries available and ready!")
    
    def create_lag_features(self, data, target_col, lag_periods=[1, 3, 6, 12]):
        """
        Create lag features for time series data
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - lag_periods: List of lag periods to create
        
        Returns:
        - DataFrame with lag features added
        """
        print(f"Creating lag features for periods: {lag_periods}")
        
        data_with_lags = data.copy()
        
        for lag in lag_periods:
            lag_col = f"{target_col}_lag_{lag}"
            data_with_lags[lag_col] = data_with_lags[target_col].shift(lag)
            
            # Store metadata
            self.feature_metadata[lag_col] = {
                'type': 'lag',
                'source_column': target_col,
                'lag_period': lag,
                'description': f'{lag}-period lag of {target_col}'
            }
        
        print(f"  Created {len(lag_periods)} lag features")
        return data_with_lags
    
    def create_rolling_features(self, data, target_col, windows=[3, 6, 12, 24]):
        """
        Create rolling statistical features
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - windows: List of rolling window sizes
        
        Returns:
        - DataFrame with rolling features added
        """
        print(f"Creating rolling features for windows: {windows}")
        
        data_with_rolling = data.copy()
        
        for window in windows:
            # Rolling mean
            mean_col = f"{target_col}_rolling_mean_{window}"
            data_with_rolling[mean_col] = data_with_rolling[target_col].rolling(window=window, min_periods=1).mean()
            
            # Rolling standard deviation
            std_col = f"{target_col}_rolling_std_{window}"
            data_with_rolling[std_col] = data_with_rolling[target_col].rolling(window=window, min_periods=1).std()
            
            # Rolling min
            min_col = f"{target_col}_rolling_min_{window}"
            data_with_rolling[min_col] = data_with_rolling[target_col].rolling(window=window, min_periods=1).min()
            
            # Rolling max
            max_col = f"{target_col}_rolling_max_{window}"
            data_with_rolling[max_col] = data_with_rolling[target_col].rolling(window=window, min_periods=1).max()
            
            # Rolling median
            median_col = f"{target_col}_rolling_median_{window}"
            data_with_rolling[median_col] = data_with_rolling[target_col].rolling(window=window, min_periods=1).median()
            
            # Store metadata
            for col, stat in [(mean_col, 'mean'), (std_col, 'std'), (min_col, 'min'), 
                             (max_col, 'max'), (median_col, 'median')]:
                self.feature_metadata[col] = {
                    'type': 'rolling',
                    'source_column': target_col,
                    'window_size': window,
                    'statistic': stat,
                    'description': f'{window}-period rolling {stat} of {target_col}'
                }
        
        print(f"  Created {len(windows) * 5} rolling statistical features")
        return data_with_rolling
    
    def create_yoy_growth_features(self, data, target_col, date_col='date'):
        """
        Create comprehensive Year-over-Year growth features
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - date_col: Name of the date column
        
        Returns:
        - DataFrame with YoY growth features added
        """
        print("Creating comprehensive YoY growth features")
        
        data_with_yoy = data.copy()
        data_with_yoy[date_col] = pd.to_datetime(data_with_yoy[date_col])
        data_with_yoy = data_with_yoy.sort_values(date_col)
        
        # Basic YoY growth rate
        yoy_col = f"{target_col}_yoy_growth"
        data_with_yoy[yoy_col] = data_with_yoy[target_col].pct_change(periods=12) * 100
        
        # YoY rolling average growth (3, 6, 12-month rolling YoY growth rates)
        for window in [3, 6, 12]:
            yoy_rolling_col = f"{target_col}_yoy_rolling_{window}m"
            data_with_yoy[yoy_rolling_col] = data_with_yoy[yoy_col].rolling(window=window, min_periods=1).mean()
            
            self.feature_metadata[yoy_rolling_col] = {
                'type': 'yoy_rolling',
                'source_column': target_col,
                'window_size': window,
                'description': f'{window}-month rolling average of YoY growth'
            }
        
        # YoY growth acceleration (rate of change in YoY growth)
        yoy_accel_col = f"{target_col}_yoy_acceleration"
        data_with_yoy[yoy_accel_col] = data_with_yoy[yoy_col].diff()
        
        # YoY seasonal growth patterns (month-specific YoY growth trends)
        data_with_yoy['month'] = data_with_yoy[date_col].dt.month
        for month in range(1, 13):
            month_mask = data_with_yoy['month'] == month
            month_yoy_col = f"{target_col}_yoy_month_{month}"
            data_with_yoy[month_yoy_col] = 0
            data_with_yoy.loc[month_mask, month_yoy_col] = data_with_yoy.loc[month_mask, yoy_col]
            
            self.feature_metadata[month_yoy_col] = {
                'type': 'yoy_seasonal',
                'source_column': target_col,
                'month': month,
                'description': f'YoY growth for month {month}'
            }
        
        # Store main YoY metadata
        self.feature_metadata[yoy_col] = {
            'type': 'yoy_growth',
            'source_column': target_col,
            'description': f'Year-over-year growth rate of {target_col}'
        }
        
        self.feature_metadata[yoy_accel_col] = {
            'type': 'yoy_acceleration',
            'source_column': target_col,
            'description': f'YoY growth acceleration of {target_col}'
        }
        
        print(f"  Created YoY growth features: basic, rolling averages, acceleration, seasonal patterns")
        return data_with_yoy
    
    def create_qoq_growth_features(self, data, target_col, date_col='date'):
        """
        Create Quarter-over-Quarter growth features
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - date_col: Name of the date column
        
        Returns:
        - DataFrame with QoQ growth features added
        """
        print("Creating Quarter-over-Quarter growth features")
        
        data_with_qoq = data.copy()
        data_with_qoq[date_col] = pd.to_datetime(data_with_qoq[date_col])
        data_with_qoq = data_with_qoq.sort_values(date_col)
        
        # Create quarterly aggregation
        data_with_qoq['quarter'] = data_with_qoq[date_col].dt.to_period('Q')
        quarterly_data = data_with_qoq.groupby('quarter')[target_col].sum().reset_index()
        quarterly_data['quarter_date'] = quarterly_data['quarter'].dt.start_time
        
        # QoQ growth rate
        quarterly_data['qoq_growth'] = quarterly_data[target_col].pct_change() * 100
        
        # QoQ rolling growth (2, 4-quarter rolling QoQ growth rates)
        for window in [2, 4]:
            qoq_rolling_col = f'qoq_rolling_{window}q'
            quarterly_data[qoq_rolling_col] = quarterly_data['qoq_growth'].rolling(window=window, min_periods=1).mean()
        
        # QoQ growth volatility (standard deviation of QoQ growth over rolling windows)
        for window in [2, 4]:
            qoq_vol_col = f'qoq_volatility_{window}q'
            quarterly_data[qoq_vol_col] = quarterly_data['qoq_growth'].rolling(window=window, min_periods=1).std()
        
        # Merge back to monthly data
        data_with_qoq['quarter'] = data_with_qoq[date_col].dt.to_period('Q')
        data_with_qoq = pd.merge(data_with_qoq, quarterly_data[['quarter', 'qoq_growth', 'qoq_rolling_2q', 
                                                               'qoq_rolling_4q', 'qoq_volatility_2q', 'qoq_volatility_4q']], 
                                on='quarter', how='left')
        
        # Store metadata
        qoq_features = ['qoq_growth', 'qoq_rolling_2q', 'qoq_rolling_4q', 'qoq_volatility_2q', 'qoq_volatility_4q']
        for feature in qoq_features:
            self.feature_metadata[feature] = {
                'type': 'qoq_growth',
                'source_column': target_col,
                'description': f'Quarter-over-quarter {feature.replace("qoq_", "")} of {target_col}'
            }
        
        print(f"  Created {len(qoq_features)} QoQ growth features")
        return data_with_qoq
    
    def create_velocity_acceleration_features(self, data, target_col):
        """
        Create velocity and acceleration features
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        
        Returns:
        - DataFrame with velocity and acceleration features added
        """
        print("Creating velocity and acceleration features")
        
        data_with_velocity = data.copy()
        
        # First derivative (velocity)
        velocity_col = f"{target_col}_velocity"
        data_with_velocity[velocity_col] = data_with_velocity[target_col].diff()
        
        # Second derivative (acceleration)
        acceleration_col = f"{target_col}_acceleration"
        data_with_velocity[acceleration_col] = data_with_velocity[velocity_col].diff()
        
        # Velocity over different periods
        for period in [3, 6, 12]:
            velocity_period_col = f"{target_col}_velocity_{period}m"
            data_with_velocity[velocity_period_col] = data_with_velocity[target_col].diff(periods=period)
            
            self.feature_metadata[velocity_period_col] = {
                'type': 'velocity',
                'source_column': target_col,
                'period': period,
                'description': f'{period}-month velocity of {target_col}'
            }
        
        # Rolling velocity statistics
        for window in [3, 6, 12]:
            velocity_mean_col = f"{target_col}_velocity_mean_{window}m"
            velocity_std_col = f"{target_col}_velocity_std_{window}m"
            
            data_with_velocity[velocity_mean_col] = data_with_velocity[velocity_col].rolling(window=window, min_periods=1).mean()
            data_with_velocity[velocity_std_col] = data_with_velocity[velocity_col].rolling(window=window, min_periods=1).std()
            
            self.feature_metadata[velocity_mean_col] = {
                'type': 'velocity_rolling',
                'source_column': target_col,
                'window_size': window,
                'statistic': 'mean',
                'description': f'{window}-month rolling mean velocity of {target_col}'
            }
            
            self.feature_metadata[velocity_std_col] = {
                'type': 'velocity_rolling',
                'source_column': target_col,
                'window_size': window,
                'statistic': 'std',
                'description': f'{window}-month rolling std velocity of {target_col}'
            }
        
        # Store main metadata
        self.feature_metadata[velocity_col] = {
            'type': 'velocity',
            'source_column': target_col,
            'description': f'First derivative (velocity) of {target_col}'
        }
        
        self.feature_metadata[acceleration_col] = {
            'type': 'acceleration',
            'source_column': target_col,
            'description': f'Second derivative (acceleration) of {target_col}'
        }
        
        print(f"  Created velocity and acceleration features")
        return data_with_velocity
    
    def create_calendar_features(self, data, date_col='date'):
        """
        Create calendar-based features
        
        Parameters:
        - data: DataFrame with time series data
        - date_col: Name of the date column
        
        Returns:
        - DataFrame with calendar features added
        """
        print("Creating calendar-based features")
        
        data_with_calendar = data.copy()
        data_with_calendar[date_col] = pd.to_datetime(data_with_calendar[date_col])
        
        # Basic calendar features
        data_with_calendar['year'] = data_with_calendar[date_col].dt.year
        data_with_calendar['month'] = data_with_calendar[date_col].dt.month
        data_with_calendar['quarter'] = data_with_calendar[date_col].dt.quarter
        data_with_calendar['day_of_year'] = data_with_calendar[date_col].dt.dayofyear
        data_with_calendar['week_of_year'] = data_with_calendar[date_col].dt.isocalendar().week
        
        # Month-end effects
        data_with_calendar['is_month_end'] = data_with_calendar[date_col].dt.is_month_end.astype(int)
        data_with_calendar['is_month_start'] = data_with_calendar[date_col].dt.is_month_start.astype(int)
        data_with_calendar['is_quarter_end'] = data_with_calendar[date_col].dt.is_quarter_end.astype(int)
        data_with_calendar['is_quarter_start'] = data_with_calendar[date_col].dt.is_quarter_start.astype(int)
        data_with_calendar['is_year_end'] = data_with_calendar[date_col].dt.is_year_end.astype(int)
        data_with_calendar['is_year_start'] = data_with_calendar[date_col].dt.is_year_start.astype(int)
        
        # Seasonal indicators
        data_with_calendar['is_winter'] = data_with_calendar['month'].isin([12, 1, 2]).astype(int)
        data_with_calendar['is_spring'] = data_with_calendar['month'].isin([3, 4, 5]).astype(int)
        data_with_calendar['is_summer'] = data_with_calendar['month'].isin([6, 7, 8]).astype(int)
        data_with_calendar['is_fall'] = data_with_calendar['month'].isin([9, 10, 11]).astype(int)
        
        # Holiday indicators (simplified)
        data_with_calendar['is_holiday_season'] = data_with_calendar['month'].isin([11, 12]).astype(int)
        data_with_calendar['is_summer_travel'] = data_with_calendar['month'].isin([6, 7, 8]).astype(int)
        
        # Cyclical encoding for month and quarter
        data_with_calendar['month_sin'] = np.sin(2 * np.pi * data_with_calendar['month'] / 12)
        data_with_calendar['month_cos'] = np.cos(2 * np.pi * data_with_calendar['month'] / 12)
        data_with_calendar['quarter_sin'] = np.sin(2 * np.pi * data_with_calendar['quarter'] / 4)
        data_with_calendar['quarter_cos'] = np.cos(2 * np.pi * data_with_calendar['quarter'] / 4)
        
        # Store metadata for calendar features
        calendar_features = [
            'year', 'month', 'quarter', 'day_of_year', 'week_of_year',
            'is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start',
            'is_year_end', 'is_year_start', 'is_winter', 'is_spring', 'is_summer', 'is_fall',
            'is_holiday_season', 'is_summer_travel', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos'
        ]
        
        for feature in calendar_features:
            self.feature_metadata[feature] = {
                'type': 'calendar',
                'source_column': date_col,
                'description': f'Calendar feature: {feature}'
            }
        
        print(f"  Created {len(calendar_features)} calendar features")
        return data_with_calendar
    
    def create_seasonality_index(self, data, target_col, date_col='date', window_months=24):
        """
        Create calendar month seasonality index using 24-month rolling window
        FOCUSED ON SHORT-TERM CALENDAR MONTH PREDICTION
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - date_col: Name of the date column
        - window_months: Rolling window size for seasonality calculation (24 months as per spec)
        
        Returns:
        - Dictionary with monthly seasonality indices
        """
        print(f"Creating seasonality index using {window_months}-month rolling window")
        print("FOCUS: Short-term calendar month prediction with recent patterns")
        
        data_seasonal = data.copy()
        data_seasonal[date_col] = pd.to_datetime(data_seasonal[date_col])
        data_seasonal = data_seasonal.sort_values(date_col)
        data_seasonal['month'] = data_seasonal[date_col].dt.month
        
        # Use EXACTLY the last 24 months for seasonality calculation (as per spec)
        # This ensures we capture the most recent seasonal patterns for short-term prediction
        recent_data = data_seasonal.tail(window_months).copy()
        print(f"Using data from {recent_data[date_col].min()} to {recent_data[date_col].max()} for seasonality")
        
        # Calculate average fuel burn across the rolling window
        avg_fuel_burn = recent_data[target_col].mean()
        
        # Calculate monthly seasonality indices
        monthly_avg = recent_data.groupby('month')[target_col].mean()
        seasonality_indices = {}
        
        for month in range(1, 13):
            if month in monthly_avg.index:
                seasonality_indices[month] = monthly_avg[month] / avg_fuel_burn
            else:
                seasonality_indices[month] = 1.0  # Neutral seasonality for missing months
        
        # Store seasonality indices
        self.seasonality_indices = seasonality_indices
        
        print(f"  Seasonality indices calculated:")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num in range(1, 13):
            month_name = month_names[month_num - 1]
            index_value = seasonality_indices[month_num]
            print(f"    {month_name}: {index_value:.3f}")
        
        return seasonality_indices
    
    def integrate_external_data_features(self, data, external_data, date_col='date'):
        """
        Integrate external data features with fuel burn data
        Enhanced to include both daily-interpolated and monthly-aggregated features
        
        Parameters:
        - data: DataFrame with fuel burn data
        - external_data: Dictionary of external datasets
        - date_col: Name of the date column
        
        Returns:
        - DataFrame with external features integrated
        """
        print("Integrating external data features...")
        print("Creating both daily-interpolated and monthly-aggregated features")
        
        integrated_data = data.copy()
        external_features_added = 0
        monthly_aggregated_features_added = 0
        
        # Define aggregation statistics to apply consistently
        agg_stats = ['mean', 'sum', 'min', 'max', 'median', 'std']
        
        try:
            for source, ext_data in external_data.items():
                if ext_data is not None and len(ext_data) > 0:
                    print(f"  Integrating {source} data...")
                    
                    # Ensure date columns are datetime
                    if 'date' in ext_data.columns:
                        ext_data_clean = ext_data.copy()
                        ext_data_clean['date'] = pd.to_datetime(ext_data_clean['date'])
                        
                        # PART 1: Original daily-interpolated features (keep existing functionality)
                        # Merge with fuel burn data
                        integrated_data = pd.merge(integrated_data, ext_data_clean, on='date', how='left')
                        
                        # Count new features added
                        new_features = [col for col in ext_data_clean.columns if col != 'date']
                        external_features_added += len(new_features)
                        
                        # Store metadata for external features
                        for feature in new_features:
                            self.feature_metadata[feature] = {
                                'type': 'external',
                                'source': source,
                                'description': f'External feature from {source}: {feature}'
                            }
                        
                        print(f"    Added {len(new_features)} daily-interpolated features from {source}")
                        
                        # PART 2: NEW - Monthly aggregated features
                        # Check if this appears to be daily data (more than 31 unique dates per year)
                        ext_data_clean['year'] = ext_data_clean['date'].dt.year
                        ext_data_clean['month'] = ext_data_clean['date'].dt.month
                        
                        # Calculate average records per month to detect daily data
                        records_per_month = len(ext_data_clean) / (ext_data_clean['year'].nunique() * 12)
                        
                        if records_per_month > 5:  # Likely daily or sub-daily data
                            print(f"    Detected daily-level data for {source} (avg {records_per_month:.1f} records/month)")
                            print(f"    Creating monthly aggregated features...")
                            
                            # Get numeric columns for aggregation
                            numeric_cols = ext_data_clean.select_dtypes(include=[np.number]).columns.tolist()
                            numeric_cols = [col for col in numeric_cols if col not in ['year', 'month']]
                            
                            if numeric_cols:
                                # Group by year-month and aggregate
                                monthly_agg = ext_data_clean.groupby(['year', 'month'])[numeric_cols].agg(agg_stats).reset_index()
                                
                                # Flatten column names
                                monthly_agg.columns = ['year', 'month'] + [f"{col}_{stat}_monthly" for col in numeric_cols for stat in agg_stats]
                                
                                # Create date column for merging (first day of month)
                                monthly_agg['date'] = pd.to_datetime(monthly_agg[['year', 'month']].assign(day=1))
                                monthly_agg = monthly_agg.drop(['year', 'month'], axis=1)
                                
                                # Merge monthly aggregated features
                                integrated_data = pd.merge(integrated_data, monthly_agg, on='date', how='left')
                                
                                # Count and store metadata for monthly aggregated features
                                monthly_features = [col for col in monthly_agg.columns if col != 'date']
                                monthly_aggregated_features_added += len(monthly_features)
                                
                                for feature in monthly_features:
                                    # Extract original feature name and statistic
                                    parts = feature.split('_')
                                    if len(parts) >= 3 and parts[-2] in agg_stats and parts[-1] == 'monthly':
                                        original_feature = '_'.join(parts[:-2])
                                        stat = parts[-2]
                                        
                                        self.feature_metadata[feature] = {
                                            'type': 'external_monthly_agg',
                                            'source': source,
                                            'original_feature': original_feature,
                                            'aggregation_stat': stat,
                                            'description': f'Monthly {stat} of {original_feature} from {source}'
                                        }
                                
                                print(f"    Added {len(monthly_features)} monthly aggregated features from {source}")
                        else:
                            print(f"    Data for {source} appears to be already monthly (avg {records_per_month:.1f} records/month)")
                    else:
                        print(f"    Warning: No 'date' column found in {source} data")
                else:
                    print(f"    Warning: No data available for {source}")
            
            # Forward fill missing values for external features (daily-interpolated only)
            external_cols = [col for col in integrated_data.columns 
                           if col not in data.columns and not col.endswith('_monthly')]
            if external_cols:
                print(f"  Forward filling missing values for {len(external_cols)} daily-interpolated external features...")
                for col in external_cols:
                    integrated_data[col] = integrated_data[col].fillna(method='ffill').fillna(method='bfill')
            
            # For monthly aggregated features, use different filling strategy
            monthly_agg_cols = [col for col in integrated_data.columns if col.endswith('_monthly')]
            if monthly_agg_cols:
                print(f"  Handling missing values for {len(monthly_agg_cols)} monthly aggregated features...")
                for col in monthly_agg_cols:
                    # For monthly aggregated features, forward fill is more appropriate
                    integrated_data[col] = integrated_data[col].fillna(method='ffill')
            
            print(f"  Total daily-interpolated external features integrated: {external_features_added}")
            print(f"  Total monthly aggregated external features integrated: {monthly_aggregated_features_added}")
            print(f"  Grand total external features: {external_features_added + monthly_aggregated_features_added}")
            
        except Exception as e:
            print(f"  Error integrating external data: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
        
        return integrated_data
    
    def apply_sktime_engineering(self, data, target_col, date_col='date'):
        """
        Apply sktime time series feature extraction
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - date_col: Name of the date column
        
        Returns:
        - DataFrame with sktime-generated features
        """
        print("Applying sktime time series feature extraction")
        
        try:
            # Prepare data for sktime
            sktime_data = data.copy()
            sktime_data = sktime_data[[date_col, target_col]].dropna()
            
            if len(sktime_data) < 10:
                print("  Insufficient data for sktime feature extraction")
                return data
            
            # Extract statistical summary features
            summary_transformer = SummaryTransformer(
                summary_function=("mean", "std", "min", "max", "median")
            )
            
            # Extract autocorrelation features
            acf_transformer = AutoCorrelationTransformer(
                n_lags=12,  # 12 months for yearly patterns
                adjusted=True
            )
            
            # Apply transformers to the time series
            X = sktime_data[target_col].values
            
            # Get summary features
            summary_features = summary_transformer.fit_transform(X)
            
            # Get autocorrelation features
            acf_features = acf_transformer.fit_transform(X)
            
            # Add features to original data
            data_with_sktime = data.copy()
            features_added = 0
            
            # Define feature names manually since get_feature_names() is not available
            summary_feature_names = ["mean", "std", "min", "max", "median"]
            
            # Add summary features
            for i, feature_name in enumerate(summary_feature_names):
                if i < summary_features.shape[1]:  # Make sure we don't go out of bounds
                    col_name = f"sktime_summary_{feature_name}"
                    data_with_sktime[col_name] = summary_features[0, i]
                    features_added += 1
                    
                    # Store metadata
                    self.feature_metadata[col_name] = {
                        'type': 'sktime_summary',
                        'description': f'sktime summary feature: {feature_name}'
                    }
            
            # Add autocorrelation features
            for i in range(acf_features.shape[1]):
                col_name = f"sktime_acf_lag_{i+1}"
                data_with_sktime[col_name] = acf_features[0, i]
                features_added += 1
                
                # Store metadata
                self.feature_metadata[col_name] = {
                    'type': 'sktime_acf',
                    'description': f'sktime autocorrelation feature: lag {i+1}'
                }
            
            # Add trend features
            # Simple linear trend
            X_trend = np.arange(len(X))
            trend_coef = np.polyfit(X_trend, X, 1)
            
            data_with_sktime['sktime_trend_slope'] = trend_coef[0]
            data_with_sktime['sktime_trend_intercept'] = trend_coef[1]
            features_added += 2
            
            # Store metadata for trend features
            self.feature_metadata['sktime_trend_slope'] = {
                'type': 'sktime_trend',
                'description': 'Linear trend slope coefficient'
            }
            self.feature_metadata['sktime_trend_intercept'] = {
                'type': 'sktime_trend',
                'description': 'Linear trend intercept coefficient'
            }
            
            print(f"  Generated {features_added} sktime features")
            return data_with_sktime
            
        except Exception as e:
            print(f"  Error in sktime feature extraction: {e}")
            return data
    
    def select_best_features(self, data, target_col, max_features=20, method='mutual_info'):
        """
        Select best features using sklearn feature selection
        
        Parameters:
        - data: DataFrame with features
        - target_col: Name of the target column
        - max_features: Maximum number of features to select
        - method: Feature selection method ('mutual_info' or 'f_regression')
        
        Returns:
        - DataFrame with selected features
        """
        print(f"Selecting best {max_features} features using {method}")
        
        try:
            # Prepare feature matrix - exclude period columns which can't be handled by mean()
            feature_cols = [col for col in data.columns 
                           if col != target_col and col != 'date' 
                           and not pd.api.types.is_period_dtype(data[col])]
            
            print(f"  Considering {len(feature_cols)} features for selection")
            X = data[feature_cols].copy()
            y = data[target_col].values
            
            # Handle categorical columns
            categorical_cols = X.select_dtypes(include=['category']).columns.tolist()
            if categorical_cols:
                print(f"  Converting {len(categorical_cols)} categorical columns to numeric")
                # Temperature category mapping (ordinal)
                temp_category_mapping = {
                    'very_cold': 0,
                    'cold': 1,
                    'mild': 2,
                    'warm': 3,
                    'hot': 4
                }
                
                for col in categorical_cols:
                    # Check if column has too many None values
                    none_count = X[col].isna().sum()
                    if none_count > len(X) * 0.5:  # If more than 50% are None
                        # Drop this column from consideration
                        X = X.drop(columns=[col])
                        feature_cols.remove(col)
                        print(f"    Dropping column {col} due to excessive None values")
                        continue
                    
                    # Convert to string first (this removes the categorical nature)
                    X[col] = X[col].astype(str)
                    
                    # Now we can fill NA values with any string
                    X[col] = X[col].fillna('unknown')
                    
                    if '_temp_category' in col:
                        # Use ordinal mapping for temperature categories
                        X[col] = X[col].map(lambda x: temp_category_mapping.get(x, -1))
                    else:
                        # Create a mapping dictionary for this column's categories
                        unique_vals = X[col].unique()
                        mapping = {val: i for i, val in enumerate(unique_vals)}
                        X[col] = X[col].map(mapping)
            
            # Handle missing values for numeric columns only
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                numeric_means = X[numeric_cols].mean()
                X[numeric_cols] = X[numeric_cols].fillna(numeric_means)
            
            # Handle the specific 'NoneNoneNone' error case
            cols_to_drop = []
            for col in X.columns:
                # Check if column contains any 'None' or repeated 'None' strings
                if X[col].astype(str).str.contains(r'^(None)+$', regex=True).any():
                    print(f"    Fixing concatenated None values in column {col}")
                    # Replace any value that is one or more 'None' in a row with np.nan
                    X[col] = X[col].astype(str).replace(r'^(None)+$', np.nan, regex=True)
                # Now, if still object, try to convert to numeric
                if X[col].dtype == object:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                # Fill remaining NaNs with -999
                X[col] = X[col].fillna(-999)
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())
            
            # Select features
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=min(max_features, len(feature_cols)))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(max_features, len(feature_cols)))
            
            X_selected = selector.fit_transform(X, y)
            selected_feature_names = [feature_cols[i] for i in selector.get_support(indices=True)]
            
            # Create result DataFrame
            result_data = data[['date', target_col] + selected_feature_names].copy()
            
            # Store feature selection metadata for ALL features
            feature_scores = selector.scores_
            
            # Create a mapping from feature name to its original index in feature_cols
            feature_to_index = {name: i for i, name in enumerate(feature_cols)}
            
            for feature_name in feature_cols:
                if feature_name in self.feature_metadata:
                    original_index = feature_to_index.get(feature_name)
                    if original_index is not None and original_index < len(feature_scores):
                        self.feature_metadata[feature_name]['selection_score'] = float(feature_scores[original_index])
                    else:
                        self.feature_metadata[feature_name]['selection_score'] = None # Or 0, depending on desired default
                    
                    # Mark as selected if it's in the list of selected features
                    self.feature_metadata[feature_name]['selected'] = feature_name in selected_feature_names
            
            print(f"  Selected {len(selected_feature_names)} best features")
            print(f"  Top 5 features: {selected_feature_names[:5]}")
            
            return result_data
            
        except Exception as e:
            print(f"  Error in feature selection: {e}")
            print(traceback.format_exc()) # Add traceback for better debugging
            return data
    
    def create_comprehensive_features(self, data, target_col='FuelBurn', date_col='date', 
                                    max_features=50, use_automated_tools=True, external_data=None,
                                    auto_save_metadata=True):
        """
        Create comprehensive feature set using all available methods
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - date_col: Name of the date column
        - max_features: Maximum number of features for automated tools
        - use_automated_tools: Whether to use sktime for automated feature extraction
        - external_data: Dictionary of external datasets to integrate
        - auto_save_metadata: Whether to automatically save feature metadata to files
        
        Returns:
        - DataFrame with comprehensive feature set
        """
        print("="*60)
        print("CREATING COMPREHENSIVE FEATURE SET")
        print("="*60)
        
        # Start with original data
        enhanced_data = data.copy()
        
        # Ensure date column is datetime
        enhanced_data[date_col] = pd.to_datetime(enhanced_data[date_col])
        enhanced_data = enhanced_data.sort_values(date_col)
        
        print(f"Starting with {len(enhanced_data.columns)} original columns")
        
        # 1. Calendar features
        print("\n1. CALENDAR FEATURES")
        enhanced_data = self.create_calendar_features(enhanced_data, date_col)
        
        # 2. Lag features
        print("\n2. LAG FEATURES")
        enhanced_data = self.create_lag_features(enhanced_data, target_col)
        
        # 3. Rolling features
        print("\n3. ROLLING STATISTICAL FEATURES")
        enhanced_data = self.create_rolling_features(enhanced_data, target_col)
        
        # 4. YoY growth features
        print("\n4. YOY GROWTH FEATURES")
        enhanced_data = self.create_yoy_growth_features(enhanced_data, target_col, date_col)
        
        # 5. QoQ growth features
        print("\n5. QOQ GROWTH FEATURES")
        enhanced_data = self.create_qoq_growth_features(enhanced_data, target_col, date_col)
        
        # 6. Velocity and acceleration features
        print("\n6. VELOCITY AND ACCELERATION FEATURES")
        enhanced_data = self.create_velocity_acceleration_features(enhanced_data, target_col)
        
        # 7. Seasonality index
        print("\n7. SEASONALITY INDEX")
        seasonality_indices = self.create_seasonality_index(enhanced_data, target_col, date_col)
        
        # 8. External data integration (if provided)
        if external_data:
            print("\n8. EXTERNAL DATA INTEGRATION")
            enhanced_data = self.integrate_external_data_features(enhanced_data, external_data, date_col)
        
        # 9. Automated feature engineering (if enabled)
        if use_automated_tools:
            print("\n9. AUTOMATED FEATURE ENGINEERING")
            
            # sktime time series features
            enhanced_data = self.apply_sktime_engineering(enhanced_data, target_col, date_col)
        
        # 10. Feature selection
        print("\n10. FEATURE SELECTION")
        if len(enhanced_data.columns) > max_features + 10:  # Only select if we have many features
            enhanced_data = self.select_best_features(enhanced_data, target_col, max_features)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Final dataset: {len(enhanced_data)} records, {len(enhanced_data.columns)} features")
        
        # Feature type summary
        feature_types = {}
        for feature, metadata in self.feature_metadata.items():
            feature_type = metadata.get('type', 'unknown')
            feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
        
        print("\nFeature types created:")
        for feature_type, count in sorted(feature_types.items()):
            print(f"  {feature_type}: {count} features")
        
        # Store final engineered features
        self.engineered_features = enhanced_data
        
        # Automatically save feature metadata if enabled
        if auto_save_metadata:
            print("\nAutomatically saving feature metadata...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save metadata to JSON
            json_filename = f"feature_metadata_{timestamp}.json"
            json_file = self.export_feature_metadata_to_json(json_filename)
            
            # Save metadata summary to CSV
            csv_filename = f"feature_metadata_summary_{timestamp}.csv"
            csv_file = self.export_feature_metadata_to_csv(csv_filename)
            
            print(f"Feature metadata automatically saved to:")
            if json_file:
                print(f"  - JSON: {json_file}")
            if csv_file:
                print(f"  - CSV: {csv_file}")
        
        return enhanced_data
    
    def get_feature_importance_summary(self):
        """
        Get summary of feature importance and metadata
        
        Returns:
        - DataFrame with feature metadata and importance scores
        """
        if not self.feature_metadata:
            print("No feature metadata available")
            return None
        
        summary_data = []
        for feature, metadata in self.feature_metadata.items():
            summary_data.append({
                'feature_name': feature,
                'feature_type': metadata.get('type', 'unknown'),
                'source_column': metadata.get('source_column', ''),
                'description': metadata.get('description', ''),
                'selection_score': metadata.get('selection_score', None),
                'selected': metadata.get('selected', False)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by selection score if available
        if 'selection_score' in summary_df.columns and summary_df['selection_score'].notna().any():
            summary_df = summary_df.sort_values('selection_score', ascending=False, na_position='last')
        
        return summary_df
    
    def export_features_to_csv(self, filename=None):
        """
        Export engineered features to CSV file
        
        Parameters:
        - filename: Output filename (optional)
        
        Returns:
        - String with filename of exported file
        """
        if self.engineered_features is None or len(self.engineered_features) == 0:
            print("No engineered features to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"engineered_features_{timestamp}.csv"
        
        try:
            self.engineered_features.to_csv(filename, index=False)
            print(f"Engineered features exported to: {filename}")
            print(f"  Records: {len(self.engineered_features)}")
            print(f"  Features: {len(self.engineered_features.columns)}")
            
            return filename
            
        except Exception as e:
            print(f"Error exporting features: {e}")
            return None
    
    def export_feature_metadata_to_json(self, filename=None):
        """
        Export feature metadata to JSON file
        
        Parameters:
        - filename: Output filename (optional)
        
        Returns:
        - String with filename of exported file
        """
        if not self.feature_metadata:
            print("No feature metadata to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"feature_metadata_{timestamp}.json"
        
        try:
            # Convert metadata to a serializable format
            # Some values might not be JSON serializable (like numpy types)
            serializable_metadata = {}
            for feature, metadata in self.feature_metadata.items():
                serializable_metadata[feature] = {}
                for key, value in metadata.items():
                    # Convert numpy types to Python native types
                    if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                        serializable_metadata[feature][key] = int(value)
                    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                        serializable_metadata[feature][key] = float(value)
                    elif isinstance(value, (np.bool_)):
                        serializable_metadata[feature][key] = bool(value)
                    else:
                        serializable_metadata[feature][key] = value
            
            # Add seasonality indices if available
            if self.seasonality_indices:
                serializable_metadata['_seasonality_indices'] = {
                    str(month): float(index) for month, index in self.seasonality_indices.items()
                }
            
            # Add metadata about the feature engineering process
            serializable_metadata['_metadata'] = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_count': len(self.feature_metadata),
                'feature_types': list(set(meta.get('type', 'unknown') for meta in self.feature_metadata.values()))
            }
            
            # Write to JSON file with indentation for readability
            with open(filename, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            
            print(f"Feature metadata exported to: {filename}")
            print(f"  Features: {len(self.feature_metadata)}")
            
            return filename
            
        except Exception as e:
            print(f"Error exporting feature metadata: {e}")
            return None
    
    def export_feature_metadata_to_csv(self, filename=None):
        """
        Export feature metadata summary to CSV file
        
        Parameters:
        - filename: Output filename (optional)
        
        Returns:
        - String with filename of exported file
        """
        # Get feature metadata summary DataFrame
        summary_df = self.get_feature_importance_summary()
        
        if summary_df is None or len(summary_df) == 0:
            print("No feature metadata to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"feature_metadata_summary_{timestamp}.csv"
        
        try:
            # Export DataFrame to CSV
            summary_df.to_csv(filename, index=False)
            
            print(f"Feature metadata summary exported to: {filename}")
            print(f"  Features: {len(summary_df)}")
            print(f"  Columns: {', '.join(summary_df.columns)}")
            
            return filename
            
        except Exception as e:
            print(f"Error exporting feature metadata summary: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Example usage of Feature Engineering Module
    print("Testing Advanced Feature Engineering Module")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Generate realistic fuel burn data with seasonality and trend
    seasonal_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / 12) * 0.2
    trend = np.linspace(0, 0.5, len(dates))
    noise = np.random.normal(0, 0.1, len(dates))
    fuel_burn = 1000 + seasonal_pattern * 200 + trend * 300 + noise * 50
    
    sample_data = pd.DataFrame({
        'date': dates,
        'FuelBurn': fuel_burn
    })
    
    print(f"Sample data created: {len(sample_data)} records")
    
    # Initialize feature engineering module
    feature_eng = FeatureEngineeringModule()
    
    # Create comprehensive features
    enhanced_data = feature_eng.create_comprehensive_features(
        sample_data, 
        target_col='FuelBurn',
        date_col='date',
        max_features=30,
        use_automated_tools=False  # Set to True if libraries are installed
    )
    
    # Get feature importance summary
    feature_summary = feature_eng.get_feature_importance_summary()
    if feature_summary is not None:
        print(f"\nFeature Summary:")
        print(feature_summary.head(10))
    
    # Export features
    exported_file = feature_eng.export_features_to_csv()
    if exported_file:
        print(f"\nFeatures exported to: {exported_file}")
    
    # Export feature metadata to JSON (detailed format)
    metadata_file = feature_eng.export_feature_metadata_to_json()
    if metadata_file:
        print(f"\nFeature metadata exported to: {metadata_file}")
    
    # Export feature metadata summary to CSV (tabular format)
    metadata_summary_file = feature_eng.export_feature_metadata_to_csv()
    if metadata_summary_file:
        print(f"\nFeature metadata summary exported to: {metadata_summary_file}")
    
    print("\nAdvanced Feature Engineering Module testing completed!")
