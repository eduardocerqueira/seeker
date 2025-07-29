#date: 2025-07-29T16:53:31Z
#url: https://api.github.com/gists/7c4e91d10c658227eae3983fbc3bc778
#owner: https://api.github.com/users/michael021997

"""
EDA & Deep Dive Analysis Module - Section 2.3 of Technical Specification

Purpose: Perform comprehensive exploratory data analysis to extract insights from 
more granular daily and/or weekly data that inform subsequent modules with the goal 
of calendar monthly predictions.

Analysis Components:
- Trend Analysis and Decomposition
- Seasonality Pattern Identification
- Correlation Analysis with External Factors
- Anomaly Detection and Outlier Analysis
- Distribution Analysis and Stationarity Tests
- Deep Dive Sectoral Analysis

Insight Integration:
- Feature Engineering Priorities
- Model Architecture Guidance
- External Data Source Selection
- Data Preprocessing Strategies
- Model Validation Approaches

Key Findings and Conclusions Framework:
- Trend Analysis Findings
- Seasonality Conclusions
- External Factor Correlation Results
- Anomaly Analysis Results
- Sectoral Analysis Insights
- Distribution and Stationarity Conclusions
- Actionable Recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import json
import traceback
warnings.filterwarnings('ignore')

# Statistical analysis libraries
from scipy import stats
from scipy.stats import jarque_bera, shapiro, anderson
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

class EDAModule:
    """
    Comprehensive EDA and Deep Dive Analysis module for fuel burn prediction
    """
    
    def __init__(self, feature_engineering_module=None):
        """Initialize the EDA module"""
        self.analysis_results = {}
        self.insights = {}
        self.recommendations = {}
        self.figures = {}
        self.feature_engineering_module = feature_engineering_module
        self.engineered_features = None
        
        print("EDA & Deep Dive Analysis Module initialized")
        print("All statistical libraries available and ready")
        print("Ready to analyze both original data and engineered features")
        
        # Set default visualization parameters
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['axes.grid'] = True
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def perform_trend_analysis(self, data, target_col='FuelBurn', date_col='date'):
        """
        Perform comprehensive trend analysis and decomposition
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - date_col: Name of the date column
        
        Returns:
        - Dictionary with trend analysis results
        """
        print("="*50)
        print("TREND ANALYSIS AND DECOMPOSITION")
        print("="*50)
        
        trend_results = {}
        
        # Prepare data
        analysis_data = data.copy()
        analysis_data[date_col] = pd.to_datetime(analysis_data[date_col])
        analysis_data = analysis_data.sort_values(date_col).reset_index(drop=True)
        
        # Basic trend statistics
        print("1. BASIC TREND STATISTICS")
        trend_results['basic_stats'] = {
            'total_records': len(analysis_data),
            'date_range': {
                'start': analysis_data[date_col].min(),
                'end': analysis_data[date_col].max(),
                'duration_years': (analysis_data[date_col].max() - analysis_data[date_col].min()).days / 365.25
            },
            'fuel_burn_stats': {
                'mean': analysis_data[target_col].mean(),
                'median': analysis_data[target_col].median(),
                'std': analysis_data[target_col].std(),
                'min': analysis_data[target_col].min(),
                'max': analysis_data[target_col].max(),
                'cv': analysis_data[target_col].std() / analysis_data[target_col].mean()
            }
        }
        
        print(f"  Data span: {trend_results['basic_stats']['date_range']['duration_years']:.1f} years")
        print(f"  Mean fuel burn: {trend_results['basic_stats']['fuel_burn_stats']['mean']:.2f}")
        print(f"  Coefficient of variation: {trend_results['basic_stats']['fuel_burn_stats']['cv']:.3f}")
        
        # Long-term trend analysis
        print("\n2. LONG-TERM TREND ANALYSIS")
        analysis_data['time_index'] = range(len(analysis_data))
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            analysis_data['time_index'], analysis_data[target_col]
        )
        
        trend_results['linear_trend'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'annual_growth_rate': (slope * 12) / analysis_data[target_col].mean() * 100,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat'
        }
        
        print(f"  Annual growth rate: {trend_results['linear_trend']['annual_growth_rate']:.2f}%")
        print(f"  Trend direction: {trend_results['linear_trend']['trend_direction']}")
        print(f"  R-squared: {trend_results['linear_trend']['r_squared']:.3f}")
        
        # Growth rate analysis
        print("\n3. GROWTH RATE ANALYSIS")
        analysis_data['growth_rate'] = analysis_data[target_col].pct_change() * 100
        analysis_data['yoy_growth'] = analysis_data[target_col].pct_change(periods=12) * 100
        
        trend_results['growth_analysis'] = {
            'monthly_growth': {
                'mean': analysis_data['growth_rate'].mean(),
                'std': analysis_data['growth_rate'].std(),
                'median': analysis_data['growth_rate'].median()
            },
            'yoy_growth': {
                'mean': analysis_data['yoy_growth'].mean(),
                'std': analysis_data['yoy_growth'].std(),
                'median': analysis_data['yoy_growth'].median()
            }
        }
        
        print(f"  Average monthly growth: {trend_results['growth_analysis']['monthly_growth']['mean']:.2f}%")
        print(f"  Average YoY growth: {trend_results['growth_analysis']['yoy_growth']['mean']:.2f}%")
        
        # Structural break detection
        print("\n4. STRUCTURAL BREAK DETECTION")
        trend_results['structural_breaks'] = self._detect_structural_breaks(analysis_data, target_col)
        
        # Time series decomposition
        print("\n5. TIME SERIES DECOMPOSITION")
        if len(analysis_data) >= 24:
            try:
                # Set date as index for decomposition
                decomp_data = analysis_data.set_index(date_col)[target_col]
                decomposition = seasonal_decompose(decomp_data, model='additive', period=12)
                
                trend_results['decomposition'] = {
                    'trend_component': decomposition.trend.dropna().values,
                    'seasonal_component': decomposition.seasonal.dropna().values,
                    'residual_component': decomposition.resid.dropna().values,
                    'seasonal_strength': np.var(decomposition.seasonal.dropna()) / np.var(decomp_data.dropna()),
                    'trend_strength': np.var(decomposition.trend.dropna()) / np.var(decomp_data.dropna())
                }
                
                print(f"  Seasonal strength: {trend_results['decomposition']['seasonal_strength']:.3f}")
                print(f"  Trend strength: {trend_results['decomposition']['trend_strength']:.3f}")
                
            except Exception as e:
                print(f"  Error in decomposition: {e}")
                trend_results['decomposition'] = None
        else:
            print("  Insufficient data for decomposition")
            trend_results['decomposition'] = None
        
        self.analysis_results['trend_analysis'] = trend_results
        return trend_results
    
    def _detect_structural_breaks(self, data, target_col):
        """Detect structural breaks in the time series"""
        breaks = []
        
        try:
            # Simple approach: detect significant changes in rolling mean
            window = min(12, len(data) // 4)
            rolling_mean = data[target_col].rolling(window=window, center=True).mean()
            rolling_std = data[target_col].rolling(window=window, center=True).std()
            
            # Detect points where rolling mean changes significantly
            mean_changes = rolling_mean.diff().abs()
            threshold = rolling_std.mean() * 2
            
            break_points = data[mean_changes > threshold].copy()
            
            for idx, row in break_points.iterrows():
                breaks.append({
                    'date': row['date'],
                    'magnitude': mean_changes.iloc[idx],
                    'before_mean': rolling_mean.iloc[max(0, idx-6):idx].mean(),
                    'after_mean': rolling_mean.iloc[idx:min(len(data), idx+6)].mean()
                })
            
            print(f"  Detected {len(breaks)} potential structural breaks")
            
        except Exception as e:
            print(f"  Error in structural break detection: {e}")
        
        return breaks
    
    def analyze_seasonality_patterns(self, data, target_col='FuelBurn', date_col='date'):
        """
        Identify and analyze seasonality patterns
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - date_col: Name of the date column
        
        Returns:
        - Dictionary with seasonality analysis results
        """
        print("="*50)
        print("SEASONALITY PATTERN IDENTIFICATION")
        print("="*50)
        
        seasonality_results = {}
        
        # Prepare data
        analysis_data = data.copy()
        analysis_data[date_col] = pd.to_datetime(analysis_data[date_col])
        analysis_data = analysis_data.sort_values(date_col).reset_index(drop=True)
        
        # Add time components
        analysis_data['year'] = analysis_data[date_col].dt.year
        analysis_data['month'] = analysis_data[date_col].dt.month
        analysis_data['quarter'] = analysis_data[date_col].dt.quarter
        analysis_data['day_of_week'] = analysis_data[date_col].dt.dayofweek
        
        # Monthly seasonal patterns
        print("1. MONTHLY SEASONAL PATTERNS")
        monthly_stats = analysis_data.groupby('month')[target_col].agg(['mean', 'std', 'count']).reset_index()
        overall_mean = analysis_data[target_col].mean()
        monthly_stats['seasonal_index'] = monthly_stats['mean'] / overall_mean
        monthly_stats['seasonal_strength'] = (monthly_stats['mean'] - overall_mean) / overall_mean * 100
        
        seasonality_results['monthly_patterns'] = {
            'monthly_stats': monthly_stats.to_dict('records'),
            'peak_month': monthly_stats.loc[monthly_stats['seasonal_index'].idxmax(), 'month'],
            'trough_month': monthly_stats.loc[monthly_stats['seasonal_index'].idxmin(), 'month'],
            'seasonal_range': monthly_stats['seasonal_index'].max() - monthly_stats['seasonal_index'].min(),
            'coefficient_of_variation': monthly_stats['seasonal_index'].std() / monthly_stats['seasonal_index'].mean()
        }
        
        print(f"  Peak month: {seasonality_results['monthly_patterns']['peak_month']}")
        print(f"  Trough month: {seasonality_results['monthly_patterns']['trough_month']}")
        print(f"  Seasonal range: {seasonality_results['monthly_patterns']['seasonal_range']:.3f}")
        
        # Quarterly patterns
        print("\n2. QUARTERLY PATTERNS")
        quarterly_stats = analysis_data.groupby('quarter')[target_col].agg(['mean', 'std', 'count']).reset_index()
        quarterly_stats['seasonal_index'] = quarterly_stats['mean'] / overall_mean
        
        seasonality_results['quarterly_patterns'] = {
            'quarterly_stats': quarterly_stats.to_dict('records'),
            'peak_quarter': quarterly_stats.loc[quarterly_stats['seasonal_index'].idxmax(), 'quarter'],
            'trough_quarter': quarterly_stats.loc[quarterly_stats['seasonal_index'].idxmin(), 'quarter']
        }
        
        print(f"  Peak quarter: Q{seasonality_results['quarterly_patterns']['peak_quarter']}")
        print(f"  Trough quarter: Q{seasonality_results['quarterly_patterns']['trough_quarter']}")
        
        # Holiday and special event analysis
        print("\n3. HOLIDAY AND SPECIAL EVENT ANALYSIS")
        seasonality_results['holiday_analysis'] = self._analyze_holiday_effects(analysis_data, target_col)
        
        # Annual cycle evolution
        print("\n4. ANNUAL CYCLE EVOLUTION")
        seasonality_results['cycle_evolution'] = self._analyze_cycle_evolution(analysis_data, target_col)
        
        # Weekly patterns (if daily data available)
        if 'day_of_week' in analysis_data.columns:
            print("\n5. WEEKLY PATTERNS")
            weekly_stats = analysis_data.groupby('day_of_week')[target_col].agg(['mean', 'std', 'count']).reset_index()
            weekly_stats['day_name'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_stats['seasonal_index'] = weekly_stats['mean'] / overall_mean
            
            seasonality_results['weekly_patterns'] = {
                'weekly_stats': weekly_stats.to_dict('records'),
                'weekday_avg': weekly_stats[weekly_stats['day_of_week'] < 5]['mean'].mean(),
                'weekend_avg': weekly_stats[weekly_stats['day_of_week'] >= 5]['mean'].mean()
            }
            
            if seasonality_results['weekly_patterns']['weekday_avg'] > 0:
                weekend_ratio = seasonality_results['weekly_patterns']['weekend_avg'] / seasonality_results['weekly_patterns']['weekday_avg']
                seasonality_results['weekly_patterns']['weekend_ratio'] = weekend_ratio
                print(f"  Weekend vs Weekday ratio: {weekend_ratio:.3f}")
        
        self.analysis_results['seasonality_analysis'] = seasonality_results
        return seasonality_results
    
    def _analyze_holiday_effects(self, data, target_col):
        """Analyze holiday and special event effects"""
        holiday_analysis = {}
        
        try:
            # Define holiday months (simplified)
            holiday_months = {
                11: 'Thanksgiving',
                12: 'Christmas/New Year',
                7: 'Summer Holiday',
                3: 'Spring Break'
            }
            
            overall_mean = data[target_col].mean()
            
            for month, holiday_name in holiday_months.items():
                holiday_data = data[data['month'] == month]
                if len(holiday_data) > 0:
                    holiday_mean = holiday_data[target_col].mean()
                    effect = (holiday_mean - overall_mean) / overall_mean * 100
                    
                    holiday_analysis[holiday_name] = {
                        'month': month,
                        'average_fuel_burn': holiday_mean,
                        'effect_percentage': effect,
                        'sample_size': len(holiday_data)
                    }
            
            print(f"  Analyzed {len(holiday_analysis)} holiday periods")
            
        except Exception as e:
            print(f"  Error in holiday analysis: {e}")
        
        return holiday_analysis
    
    def _analyze_cycle_evolution(self, data, target_col):
        """Analyze how annual cycles evolve over time"""
        cycle_evolution = {}
        
        try:
            # Calculate seasonal indices by year
            yearly_seasonal = []
            
            for year in data['year'].unique():
                year_data = data[data['year'] == year]
                if len(year_data) >= 6:  # Need at least 6 months of data
                    monthly_means = year_data.groupby('month')[target_col].mean()
                    year_mean = year_data[target_col].mean()
                    seasonal_indices = monthly_means / year_mean
                    
                    yearly_seasonal.append({
                        'year': year,
                        'seasonal_indices': seasonal_indices.to_dict(),
                        'seasonal_strength': seasonal_indices.std()
                    })
            
            if len(yearly_seasonal) > 1:
                # Analyze evolution of seasonal strength
                seasonal_strengths = [item['seasonal_strength'] for item in yearly_seasonal]
                years = [item['year'] for item in yearly_seasonal]
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, seasonal_strengths)
                
                cycle_evolution = {
                    'yearly_patterns': yearly_seasonal,
                    'seasonal_strength_trend': {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'interpretation': 'strengthening' if slope > 0 else 'weakening' if slope < 0 else 'stable'
                    }
                }
                
                print(f"  Seasonal pattern trend: {cycle_evolution['seasonal_strength_trend']['interpretation']}")
            
        except Exception as e:
            print(f"  Error in cycle evolution analysis: {e}")
        
        return cycle_evolution
    
    def perform_correlation_analysis(self, data, external_data=None, target_col='FuelBurn'):
        """
        Perform correlation analysis with external factors
        
        Parameters:
        - data: DataFrame with fuel burn data
        - external_data: Dictionary of external datasets
        - target_col: Name of the target column
        
        Returns:
        - Dictionary with correlation analysis results
        """
        print("="*50)
        print("CORRELATION ANALYSIS WITH EXTERNAL FACTORS")
        print("="*50)
        
        correlation_results = {}
        
        # Internal correlations (within fuel burn data)
        print("1. INTERNAL CORRELATIONS")
        internal_corr = self._analyze_internal_correlations(data, target_col)
        correlation_results['internal'] = internal_corr
        
        # External correlations (if external data provided)
        if external_data:
            print("\n2. EXTERNAL FACTOR CORRELATIONS")
            external_corr = self._analyze_external_correlations(data, external_data, target_col)
            correlation_results['external'] = external_corr
        else:
            print("\n2. EXTERNAL FACTOR CORRELATIONS")
            print("  No external data provided - skipping external correlation analysis")
            correlation_results['external'] = {}
        
        # Cross-correlation analysis with time lags
        print("\n3. CROSS-CORRELATION WITH TIME LAGS")
        lag_corr = self._analyze_lag_correlations(data, target_col)
        correlation_results['lag_correlations'] = lag_corr
        
        self.analysis_results['correlation_analysis'] = correlation_results
        return correlation_results
    
    def _analyze_internal_correlations(self, data, target_col):
        """Analyze correlations within the fuel burn dataset"""
        internal_corr = {}
        
        try:
            # Create lagged versions for internal correlation
            analysis_data = data.copy()
            analysis_data[f'{target_col}_lag1'] = analysis_data[target_col].shift(1)
            analysis_data[f'{target_col}_lag3'] = analysis_data[target_col].shift(3)
            analysis_data[f'{target_col}_lag6'] = analysis_data[target_col].shift(6)
            analysis_data[f'{target_col}_lag12'] = analysis_data[target_col].shift(12)
            
            # Calculate correlations
            corr_cols = [f'{target_col}_lag1', f'{target_col}_lag3', f'{target_col}_lag6', f'{target_col}_lag12']
            
            for col in corr_cols:
                if col in analysis_data.columns:
                    correlation = analysis_data[target_col].corr(analysis_data[col])
                    internal_corr[col] = correlation
            
            print(f"  Lag correlations calculated for {len(internal_corr)} periods")
            
            # Autocorrelation analysis
            try:
                # Ljung-Box test for autocorrelation
                lb_stat, lb_pvalue = acorr_ljungbox(analysis_data[target_col].dropna(), lags=12, return_df=False)
                internal_corr['ljung_box_test'] = {
                    'statistic': lb_stat,
                    'p_value': lb_pvalue,
                    'has_autocorrelation': lb_pvalue < 0.05
                }
                
                print(f"  Ljung-Box test p-value: {lb_pvalue:.4f}")
                
            except Exception as e:
                print(f"  Error in autocorrelation test: {e}")
            
        except Exception as e:
            print(f"  Error in internal correlation analysis: {e}")
        
        return internal_corr
    
    def _analyze_external_correlations(self, data, external_data, target_col):
        """Analyze correlations with external factors"""
        external_corr = {}
        
        try:
            # Merge fuel burn data with external data
            merged_data = data.copy()
            
            for source, ext_data in external_data.items():
                if ext_data is not None and len(ext_data) > 0:
                    print(f"  Analyzing {source} correlations...")
                    
                    # Merge on date
                    if 'date' in ext_data.columns:
                        merged_temp = pd.merge(merged_data, ext_data, on='date', how='inner')
                        
                        # Calculate correlations for numeric columns
                        numeric_cols = merged_temp.select_dtypes(include=[np.number]).columns
                        source_correlations = {}
                        
                        for col in numeric_cols:
                            if col != target_col and col != 'date':
                                try:
                                    correlation = merged_temp[target_col].corr(merged_temp[col])
                                    if not np.isnan(correlation):
                                        source_correlations[col] = correlation
                                except:
                                    continue
                        
                        # Sort by absolute correlation
                        sorted_corr = dict(sorted(source_correlations.items(), 
                                                key=lambda x: abs(x[1]), reverse=True))
                        
                        external_corr[source] = {
                            'correlations': sorted_corr,
                            'top_positive': max(sorted_corr.items(), key=lambda x: x[1]) if sorted_corr else None,
                            'top_negative': min(sorted_corr.items(), key=lambda x: x[1]) if sorted_corr else None,
                            'strongest_absolute': max(sorted_corr.items(), key=lambda x: abs(x[1])) if sorted_corr else None
                        }
                        
                        if external_corr[source]['strongest_absolute']:
                            strongest = external_corr[source]['strongest_absolute']
                            print(f"    Strongest correlation: {strongest[0]} ({strongest[1]:.3f})")
            
        except Exception as e:
            print(f"  Error in external correlation analysis: {e}")
        
        return external_corr
    
    def _analyze_lag_correlations(self, data, target_col):
        """Analyze cross-correlations with time lags"""
        lag_corr = {}
        
        try:
            # Calculate correlations at different lags
            max_lag = min(12, len(data) // 4)
            
            for lag in range(1, max_lag + 1):
                lagged_series = data[target_col].shift(lag)
                correlation = data[target_col].corr(lagged_series)
                lag_corr[f'lag_{lag}'] = correlation
            
            # Find optimal lag
            if lag_corr:
                optimal_lag = max(lag_corr.items(), key=lambda x: abs(x[1]))
                lag_corr['optimal_lag'] = {
                    'lag_period': optimal_lag[0],
                    'correlation': optimal_lag[1]
                }
                
                print(f"  Optimal lag: {optimal_lag[0]} (correlation: {optimal_lag[1]:.3f})")
            
        except Exception as e:
            print(f"  Error in lag correlation analysis: {e}")
        
        return lag_corr
    
    def detect_anomalies_and_outliers(self, data, target_col='FuelBurn', date_col='date'):
        """
        Detect anomalies and outliers in the data
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        - date_col: Name of the date column
        
        Returns:
        - Dictionary with anomaly detection results
        """
        print("="*50)
        print("ANOMALY DETECTION AND OUTLIER ANALYSIS")
        print("="*50)
        
        anomaly_results = {}
        
        # Prepare data
        analysis_data = data.copy()
        analysis_data[date_col] = pd.to_datetime(analysis_data[date_col])
        analysis_data = analysis_data.sort_values(date_col).reset_index(drop=True)
        
        # Statistical outlier detection
        print("1. STATISTICAL OUTLIER DETECTION")
        statistical_outliers = self._detect_statistical_outliers(analysis_data, target_col)
        anomaly_results['statistical_outliers'] = statistical_outliers
        
        # Contextual anomaly detection
        print("\n2. CONTEXTUAL ANOMALY DETECTION")
        contextual_anomalies = self._detect_contextual_anomalies(analysis_data, target_col, date_col)
        anomaly_results['contextual_anomalies'] = contextual_anomalies
        
        # Machine learning-based anomaly detection
        print("\n3. MACHINE LEARNING ANOMALY DETECTION")
        ml_anomalies = self._detect_ml_anomalies(analysis_data, target_col)
        anomaly_results['ml_anomalies'] = ml_anomalies
        
        # Event-driven anomaly analysis
        print("\n4. EVENT-DRIVEN ANOMALY ANALYSIS")
        event_anomalies = self._analyze_event_driven_anomalies(analysis_data, target_col, date_col)
        anomaly_results['event_anomalies'] = event_anomalies
        
        self.analysis_results['anomaly_analysis'] = anomaly_results
        return anomaly_results
    
    def _detect_statistical_outliers(self, data, target_col):
        """Detect statistical outliers using various methods"""
        outliers = {}
        
        try:
            values = data[target_col].values
            
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            z_outliers = data[z_scores > 3].copy()
            outliers['z_score'] = {
                'count': len(z_outliers),
                'percentage': len(z_outliers) / len(data) * 100,
                'outlier_indices': z_outliers.index.tolist(),
                'threshold': 3
            }
            
            # IQR method
            Q1 = data[target_col].quantile(0.25)
            Q3 = data[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data[target_col] < lower_bound) | (data[target_col] > upper_bound)]
            outliers['iqr'] = {
                'count': len(iqr_outliers),
                'percentage': len(iqr_outliers) / len(data) * 100,
                'outlier_indices': iqr_outliers.index.tolist(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # Modified Z-score (using median)
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
            outliers['modified_z_score'] = {
                'count': len(modified_z_outliers),
                'percentage': len(modified_z_outliers) / len(data) * 100,
                'outlier_indices': modified_z_outliers.index.tolist(),
                'threshold': 3.5
            }
            
            print(f"  Z-score outliers: {outliers['z_score']['count']} ({outliers['z_score']['percentage']:.1f}%)")
            print(f"  IQR outliers: {outliers['iqr']['count']} ({outliers['iqr']['percentage']:.1f}%)")
            print(f"  Modified Z-score outliers: {outliers['modified_z_score']['count']} ({outliers['modified_z_score']['percentage']:.1f}%)")
            
        except Exception as e:
            print(f"  Error in statistical outlier detection: {e}")
        
        return outliers
    
    def _detect_contextual_anomalies(self, data, target_col, date_col):
        """Detect contextual anomalies based on seasonal patterns"""
        contextual_anomalies = {}
        
        try:
            # Add time components
            data_with_time = data.copy()
            data_with_time['month'] = data_with_time[date_col].dt.month
            data_with_time['year'] = data_with_time[date_col].dt.year
            
            # Calculate seasonal expectations
            monthly_stats = data_with_time.groupby('month')[target_col].agg(['mean', 'std']).reset_index()
            monthly_stats.columns = ['month', 'monthly_mean', 'monthly_std']
            
            # Merge seasonal expectations with data
            data_with_expectations = pd.merge(data_with_time, monthly_stats, on='month', how='left')
            
            # Calculate seasonal anomalies
            data_with_expectations['seasonal_residual'] = (
                data_with_expectations[target_col] - data_with_expectations['monthly_mean']
            ) / data_with_expectations['monthly_std']
            
            # Identify contextual anomalies (residuals > 2 standard deviations)
            contextual_outliers = data_with_expectations[
                np.abs(data_with_expectations['seasonal_residual']) > 2
            ].copy()
            
            contextual_anomalies = {
                'count': len(contextual_outliers),
                'percentage': len(contextual_outliers) / len(data) * 100,
                'outlier_indices': contextual_outliers.index.tolist(),
                'threshold': 2,
                'seasonal_residuals': contextual_outliers['seasonal_residual'].tolist()
            }
            
            print(f"  Contextual anomalies: {contextual_anomalies['count']} ({contextual_anomalies['percentage']:.1f}%)")
            
        except Exception as e:
            print(f"  Error in contextual anomaly detection: {e}")
        
        return contextual_anomalies
    
    def _detect_ml_anomalies(self, data, target_col):
        """Detect anomalies using machine learning methods"""
        ml_anomalies = {}
        
        try:
            # Prepare features for ML anomaly detection
            features = data[[target_col]].copy()
            
            # Add rolling statistics as features
            for window in [3, 6, 12]:
                features[f'rolling_mean_{window}'] = features[target_col].rolling(window=window, min_periods=1).mean()
                features[f'rolling_std_{window}'] = features[target_col].rolling(window=window, min_periods=1).std()
            
            # Add lag features
            for lag in [1, 3, 6]:
                features[f'lag_{lag}'] = features[target_col].shift(lag)
            
            # Fill missing values
            features = features.fillna(features.mean())
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_outliers = iso_forest.fit_predict(features)
            iso_anomalies = data[iso_outliers == -1].copy()
            
            ml_anomalies['isolation_forest'] = {
                'count': len(iso_anomalies),
                'percentage': len(iso_anomalies) / len(data) * 100,
                'outlier_indices': iso_anomalies.index.tolist(),
                'contamination': 0.1
            }
            
            print(f"  Isolation Forest anomalies: {ml_anomalies['isolation_forest']['count']} ({ml_anomalies['isolation_forest']['percentage']:.1f}%)")
            
        except Exception as e:
            print(f"  Error in ML anomaly detection: {e}")
        
        return ml_anomalies
    
    def _analyze_event_driven_anomalies(self, data, target_col, date_col):
        """Analyze event-driven anomalies"""
        event_anomalies = {}
        
        try:
            # Define potential event periods (simplified)
            event_periods = [
                {'name': '2008 Financial Crisis', 'start': '2008-01-01', 'end': '2009-12-31'},
                {'name': '2020 COVID-19 Pandemic', 'start': '2020-03-01', 'end': '2021-06-30'},
                {'name': '2001 9/11 Impact', 'start': '2001-09-01', 'end': '2002-03-31'}
            ]
            
            for event in event_periods:
                event_start = pd.to_datetime(event['start'])
                event_end = pd.to_datetime(event['end'])
                
                # Filter data for event period
                event_data = data[
                    (data[date_col] >= event_start) & (data[date_col] <= event_end)
                ].copy()
                
                if len(event_data) > 0:
                    # Calculate pre-event baseline
                    pre_event_data = data[data[date_col] < event_start].copy()
                    if len(pre_event_data) > 0:
                        baseline_mean = pre_event_data[target_col].mean()
                        baseline_std = pre_event_data[target_col].std()
                        
                        # Calculate event impact
                        event_mean = event_data[target_col].mean()
                        impact_magnitude = (event_mean - baseline_mean) / baseline_mean * 100
                        
                        event_anomalies[event['name']] = {
                            'period': f"{event['start']} to {event['end']}",
                            'baseline_mean': baseline_mean,
                            'event_mean': event_mean,
                            'impact_magnitude': impact_magnitude,
                            'sample_size': len(event_data),
                            'z_score': (event_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
                        }
            
            print(f"  Analyzed {len(event_anomalies)} potential event periods")
            
        except Exception as e:
            print(f"  Error in event-driven anomaly analysis: {e}")
        
        return event_anomalies
    
    def perform_distribution_analysis(self, data, target_col='FuelBurn'):
        """
        Perform distribution analysis and stationarity tests
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column
        
        Returns:
        - Dictionary with distribution analysis results
        """
        print("="*50)
        print("DISTRIBUTION ANALYSIS AND STATIONARITY TESTS")
        print("="*50)
        
        distribution_results = {}
        
        # Prepare data
        analysis_data = data.copy()
        values = analysis_data[target_col].dropna().values
        
        # Basic distribution characteristics
        print("1. BASIC DISTRIBUTION CHARACTERISTICS")
        distribution_results['basic_stats'] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'variance': np.var(values),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25)
        }
        
        if distribution_results['basic_stats']['skewness'] is not None:
            print(f"  Skewness: {distribution_results['basic_stats']['skewness']:.3f}")
            print(f"  Kurtosis: {distribution_results['basic_stats']['kurtosis']:.3f}")
        
        # Normality tests
        print("\n2. NORMALITY TESTS")
        distribution_results['normality_tests'] = {}
        
        # Shapiro-Wilk test (for smaller samples)
        if len(values) <= 5000:
            shapiro_stat, shapiro_p = shapiro(values)
            distribution_results['normality_tests']['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
            print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")
        
        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(values)
        distribution_results['normality_tests']['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'is_normal': jb_p > 0.05
        }
        print(f"  Jarque-Bera p-value: {jb_p:.4f}")
        
        # Anderson-Darling test
        ad_stat, ad_critical, ad_significance = anderson(values, dist='norm')
        distribution_results['normality_tests']['anderson_darling'] = {
            'statistic': ad_stat,
            'critical_values': ad_critical.tolist(),
            'significance_levels': ad_significance.tolist(),
            'is_normal': ad_stat < ad_critical[2]  # 5% significance level
        }
        print(f"  Anderson-Darling statistic: {ad_stat:.4f}")
        
        # Stationarity tests
        print("\n3. STATIONARITY TESTS")
        distribution_results['stationarity_tests'] = {}
        
        # Augmented Dickey-Fuller test
        try:
            adf_stat, adf_p, adf_lags, adf_nobs, adf_critical, adf_icbest = adfuller(values, autolag='AIC')
            distribution_results['stationarity_tests']['adf'] = {
                'statistic': adf_stat,
                'p_value': adf_p,
                'lags_used': adf_lags,
                'critical_values': adf_critical,
                'is_stationary': adf_p < 0.05
            }
            print(f"  ADF test p-value: {adf_p:.4f}")
            
        except Exception as e:
            print(f"  Error in ADF test: {e}")
        
        # KPSS test
        try:
            kpss_stat, kpss_p, kpss_lags, kpss_critical = kpss(values, regression='c')
            distribution_results['stationarity_tests']['kpss'] = {
                'statistic': kpss_stat,
                'p_value': kpss_p,
                'lags_used': kpss_lags,
                'critical_values': kpss_critical,
                'is_stationary': kpss_p > 0.05
            }
            print(f"  KPSS test p-value: {kpss_p:.4f}")
            
        except Exception as e:
            print(f"  Error in KPSS test: {e}")
        
        # Variance stability analysis
        print("\n4. VARIANCE STABILITY ANALYSIS")
        distribution_results['variance_analysis'] = self._analyze_variance_stability(analysis_data, target_col)
        
        self.analysis_results['distribution_analysis'] = distribution_results
        return distribution_results
    
    def _analyze_variance_stability(self, data, target_col):
        """Analyze variance stability over time"""
        variance_analysis = {}
        
        try:
            # Calculate rolling variance
            rolling_windows = [6, 12, 24]
            
            for window in rolling_windows:
                if len(data) >= window:
                    rolling_var = data[target_col].rolling(window=window, min_periods=1).var()
                    
                    variance_analysis[f'rolling_var_{window}'] = {
                        'mean_variance': rolling_var.mean(),
                        'std_variance': rolling_var.std(),
                        'cv_variance': rolling_var.std() / rolling_var.mean() if rolling_var.mean() > 0 else 0,
                        'min_variance': rolling_var.min(),
                        'max_variance': rolling_var.max()
                    }
            
            # Test for heteroscedasticity (simplified)
            if len(data) > 24:
                first_half = data[target_col].iloc[:len(data)//2]
                second_half = data[target_col].iloc[len(data)//2:]
                
                # F-test for equal variances
                f_stat = np.var(second_half, ddof=1) / np.var(first_half, ddof=1)
                df1 = len(second_half) - 1
                df2 = len(first_half) - 1
                p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
                
                variance_analysis['heteroscedasticity_test'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'has_heteroscedasticity': p_value < 0.05,
                    'first_half_var': np.var(first_half, ddof=1),
                    'second_half_var': np.var(second_half, ddof=1)
                }
                
                print(f"  Heteroscedasticity test p-value: {p_value:.4f}")
            
        except Exception as e:
            print(f"  Error in variance stability analysis: {e}")
        
        return variance_analysis
    
    def generate_comprehensive_insights(self, auto_export=False):
        """
        Generate comprehensive insights and recommendations based on all analyses
        
        Parameters:
        - auto_export: Whether to automatically export results after generation
        
        Returns:
        - Dictionary with insights and recommendations
        """
        print("="*60)
        print("GENERATING COMPREHENSIVE INSIGHTS AND RECOMMENDATIONS")
        print("="*60)
        
        insights = {}
        recommendations = {}
        
        # Trend insights
        if 'trend_analysis' in self.analysis_results:
            insights['trend_insights'] = self._generate_trend_insights()
        
        # Seasonality insights
        if 'seasonality_analysis' in self.analysis_results:
            insights['seasonality_insights'] = self._generate_seasonality_insights()
        
        # Correlation insights
        if 'correlation_analysis' in self.analysis_results:
            insights['correlation_insights'] = self._generate_correlation_insights()
        
        # Anomaly insights
        if 'anomaly_analysis' in self.analysis_results:
            insights['anomaly_insights'] = self._generate_anomaly_insights()
        
        # Distribution insights
        if 'distribution_analysis' in self.analysis_results:
            insights['distribution_insights'] = self._generate_distribution_insights()
        
        # Generate actionable recommendations
        recommendations = self._generate_actionable_recommendations(insights)
        
        # Store results
        self.insights = insights
        self.recommendations = recommendations
        
        print("\nInsights and recommendations generated successfully!")
        
        # Auto-export if enabled
        if auto_export:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = f"eda_results_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            # Export results with visualizations
            self.export_analysis_results(
                filename=f"eda_analysis_results_{timestamp}.json",
                export_dir=export_dir,
                include_visualizations=True
            )
            
            print(f"\nResults automatically exported to directory: {export_dir}")
        
        return {'insights': insights, 'recommendations': recommendations}
    
    def _generate_trend_insights(self):
        """Generate insights from trend analysis"""
        trend_data = self.analysis_results['trend_analysis']
        insights = []
        
        try:
            # Growth rate insights
            if 'linear_trend' in trend_data:
                annual_growth = trend_data['linear_trend']['annual_growth_rate']
                trend_direction = trend_data['linear_trend']['trend_direction']
                r_squared = trend_data['linear_trend']['r_squared']
                
                insights.append(f"Fuel burn shows {annual_growth:.2f}% annual growth with {trend_direction} trend (R² = {r_squared:.3f})")
            
            # Structural breaks
            if 'structural_breaks' in trend_data and len(trend_data['structural_breaks']) > 0:
                break_count = len(trend_data['structural_breaks'])
                insights.append(f"Detected {break_count} structural breaks indicating significant regime changes")
            
            # Decomposition insights
            if trend_data.get('decomposition'):
                seasonal_strength = trend_data['decomposition']['seasonal_strength']
                trend_strength = trend_data['decomposition']['trend_strength']
                insights.append(f"Seasonal component explains {seasonal_strength:.1%} of variance, trend explains {trend_strength:.1%}")
        
        except Exception as e:
            insights.append(f"Error generating trend insights: {e}")
        
        return insights
    
    def _generate_seasonality_insights(self):
        """Generate insights from seasonality analysis"""
        seasonality_data = self.analysis_results['seasonality_analysis']
        insights = []
        
        try:
            # Monthly patterns
            if 'monthly_patterns' in seasonality_data:
                peak_month = seasonality_data['monthly_patterns']['peak_month']
                trough_month = seasonality_data['monthly_patterns']['trough_month']
                seasonal_range = seasonality_data['monthly_patterns']['seasonal_range']
                
                month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                
                insights.append(f"Peak fuel consumption in {month_names[peak_month]}, lowest in {month_names[trough_month]}")
                insights.append(f"Seasonal variation range: {seasonal_range:.3f} (seasonal index)")
            
            # Holiday effects
            if 'holiday_analysis' in seasonality_data:
                for holiday, data in seasonality_data['holiday_analysis'].items():
                    effect = data['effect_percentage']
                    insights.append(f"{holiday} period shows {effect:+.1f}% deviation from average")
        
        except Exception as e:
            insights.append(f"Error generating seasonality insights: {e}")
        
        return insights
    
    def _generate_correlation_insights(self):
        """Generate insights from correlation analysis"""
        correlation_data = self.analysis_results['correlation_analysis']
        insights = []
        
        try:
            # Internal correlations
            if 'internal' in correlation_data:
                if 'ljung_box_test' in correlation_data['internal']:
                    has_autocorr = correlation_data['internal']['ljung_box_test']['has_autocorrelation']
                    insights.append(f"Time series {'shows' if has_autocorr else 'does not show'} significant autocorrelation")
            
            # External correlations
            if 'external' in correlation_data:
                for source, data in correlation_data['external'].items():
                    if data.get('strongest_absolute'):
                        feature, correlation = data['strongest_absolute']
                        insights.append(f"Strongest {source} correlation: {feature} ({correlation:.3f})")
        
        except Exception as e:
            insights.append(f"Error generating correlation insights: {e}")
        
        return insights
    
    def _generate_anomaly_insights(self):
        """Generate insights from anomaly analysis"""
        anomaly_data = self.analysis_results['anomaly_analysis']
        insights = []
        
        try:
            # Statistical outliers
            if 'statistical_outliers' in anomaly_data:
                for method, data in anomaly_data['statistical_outliers'].items():
                    if isinstance(data, dict) and 'percentage' in data:
                        insights.append(f"{method.replace('_', ' ').title()} method identifies {data['percentage']:.1f}% outliers")
            
            # Event-driven anomalies
            if 'event_anomalies' in anomaly_data:
                for event, data in anomaly_data['event_anomalies'].items():
                    impact = data['impact_magnitude']
                    insights.append(f"{event}: {impact:+.1f}% impact on fuel consumption")
        
        except Exception as e:
            insights.append(f"Error generating anomaly insights: {e}")
        
        return insights
    
    def _generate_distribution_insights(self):
        """Generate insights from distribution analysis"""
        distribution_data = self.analysis_results['distribution_analysis']
        insights = []
        
        try:
            # Basic distribution characteristics
            if 'basic_stats' in distribution_data:
                skewness = distribution_data['basic_stats'].get('skewness')
                kurtosis = distribution_data['basic_stats'].get('kurtosis')
                
                if skewness is not None:
                    skew_desc = 'right-skewed' if skewness > 0.5 else 'left-skewed' if skewness < -0.5 else 'approximately symmetric'
                    insights.append(f"Distribution is {skew_desc} (skewness: {skewness:.3f})")
                
                if kurtosis is not None:
                    kurt_desc = 'heavy-tailed' if kurtosis > 1 else 'light-tailed' if kurtosis < -1 else 'normal-tailed'
                    insights.append(f"Distribution has {kurt_desc} characteristics (kurtosis: {kurtosis:.3f})")
            
            # Stationarity
            if 'stationarity_tests' in distribution_data:
                adf_result = distribution_data['stationarity_tests'].get('adf', {}).get('is_stationary')
                kpss_result = distribution_data['stationarity_tests'].get('kpss', {}).get('is_stationary')
                
                if adf_result is not None and kpss_result is not None:
                    if adf_result and kpss_result:
                        insights.append("Time series is stationary (confirmed by both ADF and KPSS tests)")
                    elif not adf_result and not kpss_result:
                        insights.append("Time series is non-stationary (confirmed by both ADF and KPSS tests)")
                    else:
                        insights.append("Mixed stationarity test results - requires further investigation")
        
        except Exception as e:
            insights.append(f"Error generating distribution insights: {e}")
        
        return insights
    
    def _generate_actionable_recommendations(self, insights):
        """Generate actionable recommendations based on insights"""
        recommendations = {
            'feature_engineering': [],
            'model_architecture': [],
            'data_preprocessing': [],
            'external_data_sources': [],
            'validation_strategies': []
        }
        
        try:
            # Feature engineering recommendations
            if any('seasonal' in str(insight).lower() for insight in insights.get('seasonality_insights', [])):
                recommendations['feature_engineering'].append("Include seasonal features and calendar effects")
                recommendations['feature_engineering'].append("Create month-specific dummy variables")
            
            if any('autocorrelation' in str(insight).lower() for insight in insights.get('correlation_insights', [])):
                recommendations['feature_engineering'].append("Include lag features (1, 3, 6, 12 months)")
                recommendations['feature_engineering'].append("Consider autoregressive components")
            
            # Model architecture recommendations
            if any('trend' in str(insight).lower() for insight in insights.get('trend_insights', [])):
                recommendations['model_architecture'].append("Use models that can capture trend components")
                recommendations['model_architecture'].append("Consider Prophet or ARIMA models for trend modeling")
            
            if any('non-stationary' in str(insight).lower() for insight in insights.get('distribution_insights', [])):
                recommendations['model_architecture'].append("Apply differencing or use integrated models")
                recommendations['model_architecture'].append("Consider ARIMA or Vector Error Correction models")
            
            # Data preprocessing recommendations
            if any('outlier' in str(insight).lower() for insight in insights.get('anomaly_insights', [])):
                recommendations['data_preprocessing'].append("Implement robust outlier detection and handling")
                recommendations['data_preprocessing'].append("Consider outlier-resistant transformations")
            
            if any('skewed' in str(insight).lower() for insight in insights.get('distribution_insights', [])):
                recommendations['data_preprocessing'].append("Apply log or Box-Cox transformation")
                recommendations['data_preprocessing'].append("Use robust scaling methods")
            
            # External data source recommendations
            if any('correlation' in str(insight).lower() for insight in insights.get('correlation_insights', [])):
                recommendations['external_data_sources'].append("Prioritize external factors with high correlations")
                recommendations['external_data_sources'].append("Include economic indicators and weather data")
            
            # Validation strategy recommendations
            if any('structural break' in str(insight).lower() for insight in insights.get('trend_insights', [])):
                recommendations['validation_strategies'].append("Use time-aware cross-validation")
                recommendations['validation_strategies'].append("Test model stability across different periods")
            
            if any('seasonal' in str(insight).lower() for insight in insights.get('seasonality_insights', [])):
                recommendations['validation_strategies'].append("Ensure validation sets cover full seasonal cycles")
                recommendations['validation_strategies'].append("Test seasonal pattern consistency")
        
        except Exception as e:
            recommendations['error'] = f"Error generating recommendations: {e}"
        
        return recommendations
    
    def aggregate_to_monthly(self, data, date_col='date', target_col='FuelBurn'):
        """
        Aggregate daily data to monthly totals
        
        Parameters:
        - data: DataFrame with daily data
        - date_col: Name of the date column
        - target_col: Name of the target column
        
        Returns:
        - DataFrame with monthly aggregated data
        """
        print("Aggregating daily data to monthly totals...")
        
        try:
            # Prepare data
            monthly_data = data.copy()
            monthly_data[date_col] = pd.to_datetime(monthly_data[date_col])
            
            # Create year-month period for grouping
            monthly_data['year_month'] = monthly_data[date_col].dt.to_period('M')
            
            # Group by month and sum the target column
            aggregated = monthly_data.groupby('year_month')[target_col].sum().reset_index()
            
            # Convert period back to datetime (first day of month)
            aggregated[date_col] = aggregated['year_month'].dt.start_time
            
            # Keep only essential columns and sort by date
            result = aggregated[[date_col, target_col]].sort_values(date_col).reset_index(drop=True)
            
            print(f"  Aggregated {len(data)} daily records to {len(result)} monthly records")
            print(f"  Date range: {result[date_col].min()} to {result[date_col].max()}")
            
            return result
            
        except Exception as e:
            print(f"Error aggregating to monthly: {e}")
            return None
    
    def perform_daily_weekly_analysis(self, data, target_col='FuelBurn', date_col='date'):
        """
        Perform comprehensive daily and weekly pattern analysis
        
        Parameters:
        - data: DataFrame with daily data
        - target_col: Name of the target column
        - date_col: Name of the date column
        
        Returns:
        - Dictionary with daily/weekly analysis results
        """
        print("="*50)
        print("DAILY AND WEEKLY PATTERN ANALYSIS")
        print("="*50)
        
        daily_weekly_results = {}
        
        # Prepare data
        analysis_data = data.copy()
        analysis_data[date_col] = pd.to_datetime(analysis_data[date_col])
        analysis_data = analysis_data.sort_values(date_col).reset_index(drop=True)
        
        # Add time components
        analysis_data['year'] = analysis_data[date_col].dt.year
        analysis_data['month'] = analysis_data[date_col].dt.month
        analysis_data['day_of_week'] = analysis_data[date_col].dt.dayofweek
        analysis_data['day_name'] = analysis_data[date_col].dt.day_name()
        analysis_data['week_of_year'] = analysis_data[date_col].dt.isocalendar().week
        analysis_data['is_weekend'] = (analysis_data['day_of_week'] >= 5).astype(int)
        
        # Daily pattern analysis
        print("1. DAILY PATTERN ANALYSIS")
        daily_stats = {
            'total_days': len(analysis_data),
            'date_range': {
                'start': analysis_data[date_col].min(),
                'end': analysis_data[date_col].max(),
                'duration_days': (analysis_data[date_col].max() - analysis_data[date_col].min()).days
            },
            'daily_fuel_burn': {
                'mean': analysis_data[target_col].mean(),
                'median': analysis_data[target_col].median(),
                'std': analysis_data[target_col].std(),
                'min': analysis_data[target_col].min(),
                'max': analysis_data[target_col].max(),
                'cv': analysis_data[target_col].std() / analysis_data[target_col].mean()
            }
        }
        
        daily_weekly_results['daily_stats'] = daily_stats
        print(f"  Analyzed {daily_stats['total_days']} days of data")
        print(f"  Average daily fuel burn: {daily_stats['daily_fuel_burn']['mean']:.2f}")
        print(f"  Daily coefficient of variation: {daily_stats['daily_fuel_burn']['cv']:.3f}")
        
        # Weekly pattern analysis
        print("\n2. WEEKLY PATTERN ANALYSIS")
        weekly_stats = analysis_data.groupby('day_of_week')[target_col].agg(['mean', 'std', 'count']).reset_index()
        weekly_stats['day_name'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_stats['day_name_short'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Calculate weekly pattern metrics
        overall_mean = analysis_data[target_col].mean()
        weekly_stats['relative_to_average'] = weekly_stats['mean'] / overall_mean
        weekly_stats['percentage_difference'] = (weekly_stats['mean'] - overall_mean) / overall_mean * 100
        
        daily_weekly_results['weekly_patterns'] = {
            'weekly_stats': weekly_stats.to_dict('records'),
            'highest_day': weekly_stats.loc[weekly_stats['mean'].idxmax(), 'day_name'],
            'lowest_day': weekly_stats.loc[weekly_stats['mean'].idxmin(), 'day_name'],
            'weekly_range': weekly_stats['mean'].max() - weekly_stats['mean'].min(),
            'weekly_cv': weekly_stats['mean'].std() / weekly_stats['mean'].mean()
        }
        
        print(f"  Highest consumption day: {daily_weekly_results['weekly_patterns']['highest_day']}")
        print(f"  Lowest consumption day: {daily_weekly_results['weekly_patterns']['lowest_day']}")
        print(f"  Weekly range: {daily_weekly_results['weekly_patterns']['weekly_range']:.2f}")
        
        # Weekday vs Weekend analysis
        print("\n3. WEEKDAY VS WEEKEND ANALYSIS")
        weekday_data = analysis_data[analysis_data['is_weekend'] == 0]
        weekend_data = analysis_data[analysis_data['is_weekend'] == 1]
        
        weekday_weekend_analysis = {
            'weekday_stats': {
                'count': len(weekday_data),
                'mean': weekday_data[target_col].mean(),
                'std': weekday_data[target_col].std(),
                'median': weekday_data[target_col].median()
            },
            'weekend_stats': {
                'count': len(weekend_data),
                'mean': weekend_data[target_col].mean(),
                'std': weekend_data[target_col].std(),
                'median': weekend_data[target_col].median()
            }
        }
        
        if weekday_weekend_analysis['weekday_stats']['mean'] > 0:
            weekend_ratio = weekday_weekend_analysis['weekend_stats']['mean'] / weekday_weekend_analysis['weekday_stats']['mean']
            weekend_difference = weekday_weekend_analysis['weekend_stats']['mean'] - weekday_weekend_analysis['weekday_stats']['mean']
            weekend_percentage = (weekend_difference / weekday_weekend_analysis['weekday_stats']['mean']) * 100
            
            weekday_weekend_analysis['comparison'] = {
                'weekend_to_weekday_ratio': weekend_ratio,
                'absolute_difference': weekend_difference,
                'percentage_difference': weekend_percentage,
                'interpretation': 'higher' if weekend_ratio > 1 else 'lower' if weekend_ratio < 1 else 'similar'
            }
            
            print(f"  Weekend vs Weekday ratio: {weekend_ratio:.3f}")
            print(f"  Weekend consumption is {weekend_percentage:+.1f}% vs weekdays")
        
        daily_weekly_results['weekday_weekend'] = weekday_weekend_analysis
        
        # Monthly patterns from daily data
        print("\n4. MONTHLY PATTERNS FROM DAILY DATA")
        monthly_from_daily = analysis_data.groupby('month')[target_col].agg(['mean', 'std', 'count']).reset_index()
        monthly_from_daily['month_name'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_from_daily['seasonal_index'] = monthly_from_daily['mean'] / overall_mean
        monthly_from_daily['percentage_difference'] = (monthly_from_daily['mean'] - overall_mean) / overall_mean * 100
        
        daily_weekly_results['monthly_from_daily'] = {
            'monthly_stats': monthly_from_daily.to_dict('records'),
            'peak_month': monthly_from_daily.loc[monthly_from_daily['seasonal_index'].idxmax(), 'month_name'],
            'trough_month': monthly_from_daily.loc[monthly_from_daily['seasonal_index'].idxmin(), 'month_name'],
            'seasonal_range': monthly_from_daily['seasonal_index'].max() - monthly_from_daily['seasonal_index'].min()
        }
        
        print(f"  Peak month (from daily data): {daily_weekly_results['monthly_from_daily']['peak_month']}")
        print(f"  Trough month (from daily data): {daily_weekly_results['monthly_from_daily']['trough_month']}")
        
        # Multi-frequency insights
        print("\n5. MULTI-FREQUENCY INSIGHTS")
        multi_freq_insights = self._generate_multi_frequency_insights(analysis_data, target_col)
        daily_weekly_results['multi_frequency_insights'] = multi_freq_insights
        
        self.analysis_results['daily_weekly_analysis'] = daily_weekly_results
        return daily_weekly_results
    
    def perform_deep_dive_sectoral_analysis(self, data, external_data=None, target_col='FuelBurn', date_col='date'):
        """
        Perform deep dive sectoral analysis as required by technical specification
        
        Parameters:
        - data: DataFrame with fuel burn data
        - external_data: Dictionary of external datasets
        - target_col: Name of the target column
        - date_col: Name of the date column
        
        Returns:
        - Dictionary with sectoral analysis results
        """
        print("="*50)
        print("DEEP DIVE SECTORAL ANALYSIS")
        print("="*50)
        
        sectoral_results = {}
        
        # Prepare data
        analysis_data = data.copy()
        analysis_data[date_col] = pd.to_datetime(analysis_data[date_col])
        analysis_data = analysis_data.sort_values(date_col).reset_index(drop=True)
        
        # 1. Luxury consumption patterns and fuel burn correlation
        print("1. LUXURY CONSUMPTION PATTERNS ANALYSIS")
        sectoral_results['luxury_analysis'] = self._analyze_luxury_consumption_patterns(
            analysis_data, external_data, target_col, date_col
        )
        
        # 2. Travel sector demand patterns (business vs. leisure travel)
        print("\n2. TRAVEL SECTOR DEMAND PATTERNS")
        sectoral_results['travel_analysis'] = self._analyze_travel_sector_patterns(
            analysis_data, external_data, target_col, date_col
        )
        
        # 3. Regional fuel consumption variations
        print("\n3. REGIONAL FUEL CONSUMPTION VARIATIONS")
        sectoral_results['regional_analysis'] = self._analyze_regional_variations(
            analysis_data, external_data, target_col, date_col
        )
        
        # 4. Demographic and socioeconomic factor analysis
        print("\n4. DEMOGRAPHIC AND SOCIOECONOMIC ANALYSIS")
        sectoral_results['demographic_analysis'] = self._analyze_demographic_factors(
            analysis_data, external_data, target_col, date_col
        )
        
        # 5. Policy impact analysis (fuel taxes, environmental regulations)
        print("\n5. POLICY IMPACT ANALYSIS")
        sectoral_results['policy_analysis'] = self._analyze_policy_impacts(
            analysis_data, external_data, target_col, date_col
        )
        
        self.analysis_results['sectoral_analysis'] = sectoral_results
        return sectoral_results
    
    def _analyze_luxury_consumption_patterns(self, data, external_data, target_col, date_col):
        """Analyze luxury consumption patterns and fuel burn correlation"""
        luxury_analysis = {}
        
        try:
            print("  Analyzing luxury sector correlations with fuel consumption...")
            
            # If external data contains luxury sector information
            if external_data and 'stocks' in external_data:
                stocks_data = external_data['stocks']
                
                # Look for luxury-related columns
                luxury_indicators = []
                for col in stocks_data.columns:
                    if any(keyword in col.lower() for keyword in ['luxury', 'lvmh', 'tesla', 'nike']):
                        luxury_indicators.append(col)
                
                if luxury_indicators:
                    # Merge with fuel data
                    merged_data = pd.merge(data, stocks_data, on='date', how='inner')
                    
                    luxury_correlations = {}
                    for indicator in luxury_indicators:
                        if indicator in merged_data.columns:
                            correlation = merged_data[target_col].corr(merged_data[indicator])
                            if not np.isnan(correlation):
                                luxury_correlations[indicator] = correlation
                    
                    # Lead/lag analysis for luxury indicators
                    lead_lag_analysis = {}
                    for indicator in luxury_indicators[:3]:  # Analyze top 3
                        if indicator in merged_data.columns:
                            correlations_by_lag = {}
                            for lag in range(-6, 7):  # -6 to +6 months
                                if lag == 0:
                                    corr = merged_data[target_col].corr(merged_data[indicator])
                                elif lag > 0:
                                    corr = merged_data[target_col].corr(merged_data[indicator].shift(lag))
                                else:
                                    corr = merged_data[target_col].shift(-lag).corr(merged_data[indicator])
                                
                                if not np.isnan(corr):
                                    correlations_by_lag[lag] = corr
                            
                            if correlations_by_lag:
                                optimal_lag = max(correlations_by_lag.items(), key=lambda x: abs(x[1]))
                                lead_lag_analysis[indicator] = {
                                    'optimal_lag': optimal_lag[0],
                                    'correlation': optimal_lag[1],
                                    'interpretation': 'leads' if optimal_lag[0] < 0 else 'lags' if optimal_lag[0] > 0 else 'concurrent'
                                }
                    
                    luxury_analysis = {
                        'correlations': luxury_correlations,
                        'lead_lag_analysis': lead_lag_analysis,
                        'strongest_luxury_indicator': max(luxury_correlations.items(), key=lambda x: abs(x[1])) if luxury_correlations else None,
                        'luxury_indicators_found': len(luxury_indicators)
                    }
                    
                    print(f"    Found {len(luxury_indicators)} luxury indicators")
                    if luxury_analysis['strongest_luxury_indicator']:
                        strongest = luxury_analysis['strongest_luxury_indicator']
                        print(f"    Strongest luxury correlation: {strongest[0]} ({strongest[1]:.3f})")
                else:
                    print("    No luxury indicators found in external data")
                    luxury_analysis['message'] = "No luxury indicators available in external data"
            else:
                print("    No stock market data available for luxury analysis")
                luxury_analysis['message'] = "No external stock data available"
            
            # Seasonal luxury consumption patterns
            data['month'] = data[date_col].dt.month
            luxury_seasonal_patterns = {}
            
            # Holiday seasons typically associated with luxury consumption
            luxury_months = [11, 12, 6, 7]  # November, December (holidays), June, July (summer)
            for month in luxury_months:
                month_data = data[data['month'] == month]
                if len(month_data) > 0:
                    month_avg = month_data[target_col].mean()
                    overall_avg = data[target_col].mean()
                    luxury_seasonal_patterns[month] = {
                        'average_consumption': month_avg,
                        'relative_to_annual': month_avg / overall_avg,
                        'percentage_difference': (month_avg - overall_avg) / overall_avg * 100
                    }
            
            luxury_analysis['seasonal_patterns'] = luxury_seasonal_patterns
            
        except Exception as e:
            print(f"    Error in luxury consumption analysis: {e}")
            luxury_analysis['error'] = str(e)
        
        return luxury_analysis
    
    def _analyze_travel_sector_patterns(self, data, external_data, target_col, date_col):
        """Analyze travel sector demand patterns"""
        travel_analysis = {}
        
        try:
            print("  Analyzing travel sector patterns...")
            
            # Business vs leisure travel patterns (inferred from weekly patterns)
            data['day_of_week'] = data[date_col].dt.dayofweek
            data['is_business_day'] = (data['day_of_week'] < 5).astype(int)
            
            business_days = data[data['is_business_day'] == 1]
            leisure_days = data[data['is_business_day'] == 0]
            
            if len(business_days) > 0 and len(leisure_days) > 0:
                business_avg = business_days[target_col].mean()
                leisure_avg = leisure_days[target_col].mean()
                
                travel_patterns = {
                    'business_travel_proxy': {
                        'average_consumption': business_avg,
                        'sample_size': len(business_days)
                    },
                    'leisure_travel_proxy': {
                        'average_consumption': leisure_avg,
                        'sample_size': len(leisure_days)
                    },
                    'business_to_leisure_ratio': business_avg / leisure_avg if leisure_avg > 0 else 0,
                    'interpretation': 'business_higher' if business_avg > leisure_avg else 'leisure_higher'
                }
                
                travel_analysis['business_vs_leisure'] = travel_patterns
                print(f"    Business vs Leisure ratio: {travel_patterns['business_to_leisure_ratio']:.3f}")
            
            # Travel sector stock correlations
            if external_data and 'stocks' in external_data:
                stocks_data = external_data['stocks']
                travel_indicators = []
                
                for col in stocks_data.columns:
                    if any(keyword in col.lower() for keyword in ['travel', 'airline', 'aal', 'ual', 'dal', 'ccl']):
                        travel_indicators.append(col)
                
                if travel_indicators:
                    merged_data = pd.merge(data, stocks_data, on='date', how='inner')
                    
                    travel_correlations = {}
                    for indicator in travel_indicators:
                        if indicator in merged_data.columns:
                            correlation = merged_data[target_col].corr(merged_data[indicator])
                            if not np.isnan(correlation):
                                travel_correlations[indicator] = correlation
                    
                    travel_analysis['travel_stock_correlations'] = travel_correlations
                    
                    if travel_correlations:
                        strongest_travel = max(travel_correlations.items(), key=lambda x: abs(x[1]))
                        travel_analysis['strongest_travel_indicator'] = strongest_travel
                        print(f"    Strongest travel correlation: {strongest_travel[0]} ({strongest_travel[1]:.3f})")
            
            # Seasonal travel patterns
            data['month'] = data[date_col].dt.month
            seasonal_travel = {}
            
            # Peak travel months
            peak_travel_months = [6, 7, 8, 11, 12]  # Summer and holiday seasons
            for month in peak_travel_months:
                month_data = data[data['month'] == month]
                if len(month_data) > 0:
                    month_avg = month_data[target_col].mean()
                    overall_avg = data[target_col].mean()
                    seasonal_travel[month] = {
                        'average_consumption': month_avg,
                        'relative_to_annual': month_avg / overall_avg,
                        'percentage_difference': (month_avg - overall_avg) / overall_avg * 100
                    }
            
            travel_analysis['seasonal_travel_patterns'] = seasonal_travel
            
        except Exception as e:
            print(f"    Error in travel sector analysis: {e}")
            travel_analysis['error'] = str(e)
        
        return travel_analysis
    
    def _analyze_regional_variations(self, data, external_data, target_col, date_col):
        """Analyze regional fuel consumption variations"""
        regional_analysis = {}
        
        try:
            print("  Analyzing regional variations...")
            
            # NOAA regional weather-based analysis
            if external_data and 'weather' in external_data:
                weather_data = external_data['weather']
                
                # Merge with fuel data
                merged_data = pd.merge(data, weather_data, on='date', how='inner')
                
                # Analyze regional temperature patterns from NOAA data
                regional_temp_cols = [col for col in merged_data.columns if '_TAVG' in col]
                
                if regional_temp_cols:
                    print(f"    Found {len(regional_temp_cols)} regional temperature indicators")
                    
                    regional_patterns = {}
                    for temp_col in regional_temp_cols:
                        region_name = temp_col.replace('_TAVG', '')
                        
                        # Calculate correlation between regional temperature and fuel consumption
                        if temp_col in merged_data.columns and merged_data[temp_col].notna().sum() > 0:
                            correlation = merged_data[target_col].corr(merged_data[temp_col])
                            
                            # Create temperature bands for this region
                            temp_data = merged_data[merged_data[temp_col].notna()]
                            if len(temp_data) > 0:
                                temp_bands = pd.cut(temp_data[temp_col], 
                                                  bins=3, labels=['Cold', 'Moderate', 'Warm'])
                                
                                band_analysis = {}
                                for band in temp_bands.unique():
                                    if pd.notna(band):
                                        band_data = temp_data[temp_bands == band]
                                        if len(band_data) > 0:
                                            band_analysis[str(band)] = {
                                                'average_consumption': band_data[target_col].mean(),
                                                'sample_size': len(band_data),
                                                'temperature_range': [band_data[temp_col].min(), band_data[temp_col].max()]
                                            }
                                
                                regional_patterns[region_name] = {
                                    'temperature_correlation': correlation,
                                    'temperature_bands': band_analysis,
                                    'sample_size': len(temp_data)
                                }
                    
                    regional_analysis['noaa_regional_patterns'] = regional_patterns
                    
                    # Calculate overall regional variation
                    if len(regional_patterns) > 1:
                        correlations = [v['temperature_correlation'] for v in regional_patterns.values() if not np.isnan(v['temperature_correlation'])]
                        if correlations:
                            regional_analysis['regional_correlation_stats'] = {
                                'mean_correlation': np.mean(correlations),
                                'std_correlation': np.std(correlations),
                                'max_correlation': np.max(correlations),
                                'min_correlation': np.min(correlations)
                            }
                            print(f"    Regional temperature correlation range: {np.min(correlations):.3f} to {np.max(correlations):.3f}")
                
                # Analyze regional precipitation patterns
                regional_precip_cols = [col for col in merged_data.columns if '_PRCP' in col]
                
                if regional_precip_cols:
                    print(f"    Found {len(regional_precip_cols)} regional precipitation indicators")
                    
                    precip_patterns = {}
                    for precip_col in regional_precip_cols:
                        region_name = precip_col.replace('_PRCP', '')
                        
                        if precip_col in merged_data.columns and merged_data[precip_col].notna().sum() > 0:
                            correlation = merged_data[target_col].corr(merged_data[precip_col])
                            
                            precip_patterns[region_name] = {
                                'precipitation_correlation': correlation,
                                'sample_size': merged_data[precip_col].notna().sum()
                            }
                    
                    regional_analysis['regional_precipitation_patterns'] = precip_patterns
            
            # Disaster-based regional impact analysis
            if external_data and 'disasters' in external_data:
                disaster_data = external_data['disasters']
                
                # Merge with fuel data
                merged_data = pd.merge(data, disaster_data, on='date', how='inner')
                
                # Analyze impact of disasters on fuel consumption
                if 'total_disaster_impact' in merged_data.columns:
                    # High disaster impact periods
                    high_impact_threshold = merged_data['total_disaster_impact'].quantile(0.9)
                    high_impact_periods = merged_data[merged_data['total_disaster_impact'] > high_impact_threshold]
                    normal_periods = merged_data[merged_data['total_disaster_impact'] <= high_impact_threshold]
                    
                    if len(high_impact_periods) > 0 and len(normal_periods) > 0:
                        disaster_impact = {
                            'high_disaster_consumption': high_impact_periods[target_col].mean(),
                            'normal_consumption': normal_periods[target_col].mean(),
                            'disaster_impact_ratio': high_impact_periods[target_col].mean() / normal_periods[target_col].mean(),
                            'high_impact_periods': len(high_impact_periods)
                        }
                        
                        regional_analysis['disaster_impact'] = disaster_impact
                        print(f"    Disaster impact ratio: {disaster_impact['disaster_impact_ratio']:.3f}")
            
            # Economic-based regional proxy
            if external_data and 'economics' in external_data:
                econ_data = external_data['economics']
                
                # Look for regional economic indicators
                regional_econ_indicators = []
                for col in econ_data.columns:
                    if any(keyword in col.lower() for keyword in ['regional', 'state', 'gdp', 'employment']):
                        regional_econ_indicators.append(col)
                
                if regional_econ_indicators:
                    merged_data = pd.merge(data, econ_data, on='date', how='inner')
                    
                    econ_regional_correlations = {}
                    for indicator in regional_econ_indicators:
                        if indicator in merged_data.columns:
                            correlation = merged_data[target_col].corr(merged_data[indicator])
                            if not np.isnan(correlation):
                                econ_regional_correlations[indicator] = correlation
                    
                    regional_analysis['economic_regional_correlations'] = econ_regional_correlations
            
        except Exception as e:
            print(f"    Error in regional analysis: {e}")
            regional_analysis['error'] = str(e)
        
        return regional_analysis
    
    def _analyze_demographic_factors(self, data, external_data, target_col, date_col):
        """Analyze demographic and socioeconomic factors"""
        demographic_analysis = {}
        
        try:
            print("  Analyzing demographic and socioeconomic factors...")
            
            # Economic indicators as demographic proxies
            if external_data and 'economics' in external_data:
                econ_data = external_data['economics']
                
                # Merge with fuel data
                merged_data = pd.merge(data, econ_data, on='date', how='inner')
                
                # Key demographic/socioeconomic indicators
                demographic_indicators = []
                for col in econ_data.columns:
                    if any(keyword in col.lower() for keyword in ['unemployment', 'employment', 'gdp', 'income', 'consumer', 'confidence']):
                        demographic_indicators.append(col)
                
                if demographic_indicators:
                    demographic_correlations = {}
                    for indicator in demographic_indicators:
                        if indicator in merged_data.columns:
                            correlation = merged_data[target_col].corr(merged_data[indicator])
                            if not np.isnan(correlation):
                                demographic_correlations[indicator] = correlation
                    
                    # Rank by correlation strength
                    sorted_correlations = dict(sorted(demographic_correlations.items(), 
                                                    key=lambda x: abs(x[1]), reverse=True))
                    
                    demographic_analysis['economic_correlations'] = sorted_correlations
                    
                    if sorted_correlations:
                        strongest_demo = max(sorted_correlations.items(), key=lambda x: abs(x[1]))
                        demographic_analysis['strongest_demographic_factor'] = strongest_demo
                        print(f"    Strongest demographic correlation: {strongest_demo[0]} ({strongest_demo[1]:.3f})")
                    
                    # Economic cycle analysis
                    if len(demographic_indicators) > 0:
                        # Use first available economic indicator as proxy for economic cycles
                        econ_indicator = demographic_indicators[0]
                        if econ_indicator in merged_data.columns:
                            # Identify economic expansion vs contraction periods
                            econ_growth = merged_data[econ_indicator].pct_change()
                            expansion_periods = merged_data[econ_growth > 0]
                            contraction_periods = merged_data[econ_growth <= 0]
                            
                            if len(expansion_periods) > 0 and len(contraction_periods) > 0:
                                economic_cycle_analysis = {
                                    'expansion_consumption': expansion_periods[target_col].mean(),
                                    'contraction_consumption': contraction_periods[target_col].mean(),
                                    'expansion_periods': len(expansion_periods),
                                    'contraction_periods': len(contraction_periods),
                                    'cycle_impact_ratio': expansion_periods[target_col].mean() / contraction_periods[target_col].mean() if contraction_periods[target_col].mean() > 0 else 0
                                }
                                
                                demographic_analysis['economic_cycle_impact'] = economic_cycle_analysis
                                print(f"    Economic cycle impact ratio: {economic_cycle_analysis['cycle_impact_ratio']:.3f}")
            
            # Wealth indicators from stock market data
            if external_data and 'stocks' in external_data:
                stocks_data = external_data['stocks']
                
                # Look for wealth/consumer discretionary indicators
                wealth_indicators = []
                for col in stocks_data.columns:
                    if any(keyword in col.lower() for keyword in ['consumer', 'discretionary', 'luxury', 'market_avg']):
                        wealth_indicators.append(col)
                
                if wealth_indicators:
                    merged_data = pd.merge(data, stocks_data, on='date', how='inner')
                    
                    wealth_correlations = {}
                    for indicator in wealth_indicators:
                        if indicator in merged_data.columns:
                            correlation = merged_data[target_col].corr(merged_data[indicator])
                            if not np.isnan(correlation):
                                wealth_correlations[indicator] = correlation
                    
                    demographic_analysis['wealth_indicators'] = wealth_correlations
                    
                    if wealth_correlations:
                        strongest_wealth = max(wealth_correlations.items(), key=lambda x: abs(x[1]))
                        demographic_analysis['strongest_wealth_indicator'] = strongest_wealth
                        print(f"    Strongest wealth correlation: {strongest_wealth[0]} ({strongest_wealth[1]:.3f})")
            
            # Age/generation proxy analysis using technology adoption patterns
            # (Inferred from data patterns - younger demographics may have different consumption patterns)
            data['year'] = data[date_col].dt.year
            yearly_trends = data.groupby('year')[target_col].mean().reset_index()
            
            if len(yearly_trends) > 2:
                # Calculate year-over-year changes
                yearly_trends['yoy_change'] = yearly_trends[target_col].pct_change() * 100
                
                # Analyze trend acceleration/deceleration (proxy for demographic shifts)
                trend_acceleration = yearly_trends['yoy_change'].diff()
                
                demographic_analysis['generational_proxy'] = {
                    'average_annual_growth': yearly_trends['yoy_change'].mean(),
                    'growth_volatility': yearly_trends['yoy_change'].std(),
                    'trend_acceleration': trend_acceleration.mean(),
                    'interpretation': 'accelerating' if trend_acceleration.mean() > 0 else 'decelerating'
                }
                
                print(f"    Generational trend: {demographic_analysis['generational_proxy']['interpretation']}")
            
        except Exception as e:
            print(f"    Error in demographic analysis: {e}")
            demographic_analysis['error'] = str(e)
        
        return demographic_analysis
    
    def _analyze_policy_impacts(self, data, external_data, target_col, date_col):
        """Analyze policy impacts (fuel taxes, environmental regulations)"""
        policy_analysis = {}
        
        try:
            print("  Analyzing policy impacts...")
            
            # Define major policy periods (simplified)
            policy_periods = [
                {'name': 'Paris Agreement Implementation', 'start': '2016-01-01', 'end': '2020-12-31'},
                {'name': 'COVID-19 Policy Response', 'start': '2020-03-01', 'end': '2021-12-31'},
                {'name': 'Post-Financial Crisis Regulations', 'start': '2010-01-01', 'end': '2015-12-31'}
            ]
            
            policy_impacts = {}
            
            for policy in policy_periods:
                policy_start = pd.to_datetime(policy['start'])
                policy_end = pd.to_datetime(policy['end'])
                
                # Filter data for policy period
                policy_data = data[
                    (data[date_col] >= policy_start) & (data[date_col] <= policy_end)
                ].copy()
                
                if len(policy_data) > 0:
                    # Calculate pre-policy baseline
                    pre_policy_data = data[data[date_col] < policy_start].copy()
                    post_policy_data = data[data[date_col] > policy_end].copy()
                    
                    if len(pre_policy_data) > 0:
                        pre_policy_mean = pre_policy_data[target_col].mean()
                        policy_mean = policy_data[target_col].mean()
                        
                        policy_impact = {
                            'pre_policy_consumption': pre_policy_mean,
                            'policy_period_consumption': policy_mean,
                            'impact_magnitude': (policy_mean - pre_policy_mean) / pre_policy_mean * 100,
                            'policy_period_length': len(policy_data),
                            'statistical_significance': 'significant' if abs((policy_mean - pre_policy_mean) / pre_policy_mean) > 0.1 else 'not_significant'
                        }
                        
                        # Post-policy analysis if data available
                        if len(post_policy_data) > 0:
                            post_policy_mean = post_policy_data[target_col].mean()
                            policy_impact['post_policy_consumption'] = post_policy_mean
                            policy_impact['recovery_ratio'] = post_policy_mean / pre_policy_mean
                        
                        policy_impacts[policy['name']] = policy_impact
                        print(f"    {policy['name']}: {policy_impact['impact_magnitude']:+.1f}% impact")
            
            policy_analysis['policy_period_impacts'] = policy_impacts
            
            # Energy price correlation (proxy for fuel tax/energy policy impacts)
            if external_data and 'economics' in external_data:
                econ_data = external_data['economics']
                
                # Look for energy price indicators
                energy_indicators = []
                for col in econ_data.columns:
                    if any(keyword in col.lower() for keyword in ['oil', 'energy', 'gas', 'fuel']):
                        energy_indicators.append(col)
                
                if energy_indicators:
                    merged_data = pd.merge(data, econ_data, on='date', how='inner')
                    
                    energy_correlations = {}
                    for indicator in energy_indicators:
                        if indicator in merged_data.columns:
                            correlation = merged_data[target_col].corr(merged_data[indicator])
                            if not np.isnan(correlation):
                                energy_correlations[indicator] = correlation
                    
                    policy_analysis['energy_price_correlations'] = energy_correlations
                    
                    if energy_correlations:
                        strongest_energy = max(energy_correlations.items(), key=lambda x: abs(x[1]))
                        policy_analysis['strongest_energy_correlation'] = strongest_energy
                        print(f"    Strongest energy price correlation: {strongest_energy[0]} ({strongest_energy[1]:.3f})")
            
            # Environmental regulation proxy analysis
            # Look for patterns around major environmental policy dates
            environmental_events = [
                {'date': '2015-12-12', 'event': 'Paris Agreement Signed'},
                {'date': '2017-06-01', 'event': 'US Paris Agreement Withdrawal Announced'},
                {'date': '2021-01-20', 'event': 'US Rejoined Paris Agreement'}
            ]
            
            environmental_impacts = {}
            for event in environmental_events:
                event_date = pd.to_datetime(event['date'])
                
                # Analyze 6 months before and after
                before_period = data[
                    (data[date_col] >= event_date - timedelta(days=180)) & 
                    (data[date_col] < event_date)
                ]
                after_period = data[
                    (data[date_col] > event_date) & 
                    (data[date_col] <= event_date + timedelta(days=180))
                ]
                
                if len(before_period) > 0 and len(after_period) > 0:
                    before_mean = before_period[target_col].mean()
                    after_mean = after_period[target_col].mean()
                    
                    environmental_impacts[event['event']] = {
                        'before_consumption': before_mean,
                        'after_consumption': after_mean,
                        'impact_percentage': (after_mean - before_mean) / before_mean * 100,
                        'sample_size_before': len(before_period),
                        'sample_size_after': len(after_period)
                    }
            
            policy_analysis['environmental_policy_impacts'] = environmental_impacts
            
        except Exception as e:
            print(f"    Error in policy analysis: {e}")
            policy_analysis['error'] = str(e)
        
        return policy_analysis
    
    def _generate_multi_frequency_insights(self, data, target_col):
        """Generate insights across different time frequencies"""
        insights = {}
        
        try:
            # Daily volatility
            daily_returns = data[target_col].pct_change().dropna()
            insights['daily_volatility'] = {
                'daily_return_std': daily_returns.std(),
                'daily_return_mean': daily_returns.mean(),
                'volatility_annualized': daily_returns.std() * np.sqrt(252)  # Assuming 252 trading days
            }
            
            # Weekly aggregation insights
            data_weekly = data.copy()
            data_weekly['week_year'] = data_weekly['date'].dt.strftime('%Y-%U')
            weekly_agg = data_weekly.groupby('week_year')[target_col].sum().reset_index()
            
            if len(weekly_agg) > 1:
                weekly_returns = weekly_agg[target_col].pct_change().dropna()
                insights['weekly_patterns'] = {
                    'weekly_return_std': weekly_returns.std(),
                    'weekly_return_mean': weekly_returns.mean(),
                    'weekly_vs_daily_volatility_ratio': weekly_returns.std() / daily_returns.std() if daily_returns.std() > 0 else 0
                }
            
            # Intra-month patterns
            data['day_of_month'] = data['date'].dt.day
            day_of_month_stats = data.groupby('day_of_month')[target_col].mean()
            
            insights['intra_month_patterns'] = {
                'beginning_of_month_avg': day_of_month_stats.iloc[:10].mean(),  # First 10 days
                'middle_of_month_avg': day_of_month_stats.iloc[10:20].mean(),   # Middle 10 days
                'end_of_month_avg': day_of_month_stats.iloc[20:].mean(),        # Last days
                'month_end_effect': day_of_month_stats.iloc[20:].mean() / day_of_month_stats.iloc[:10].mean() if day_of_month_stats.iloc[:10].mean() > 0 else 1
            }
            
            print(f"  Daily volatility (annualized): {insights['daily_volatility']['volatility_annualized']:.3f}")
            if 'weekly_patterns' in insights:
                print(f"  Weekly vs daily volatility ratio: {insights['weekly_patterns']['weekly_vs_daily_volatility_ratio']:.3f}")
            print(f"  Month-end effect ratio: {insights['intra_month_patterns']['month_end_effect']:.3f}")
            
        except Exception as e:
            print(f"  Error in multi-frequency analysis: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def export_analysis_results(self, filename=None, export_dir=None, include_visualizations=False):
        """
        Export analysis results to file with enhanced error handling
        
        Parameters:
        - filename: Output filename (optional)
        - export_dir: Directory to save the results (optional)
        - include_visualizations: Whether to include visualizations (optional)
        
        Returns:
        - String with filename of exported file
        """
        # Create timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set default filename if not provided
        if filename is None:
            filename = f"eda_analysis_results_{timestamp}.json"
        
        # Create export directory if provided
        full_path = filename
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            full_path = os.path.join(export_dir, filename)
        
        try:
            # Prepare results for JSON serialization
            export_data = {
                'analysis_results': self._serialize_results(self.analysis_results),
                'insights': self.insights,
                'recommendations': self.recommendations,
                'export_timestamp': datetime.now().isoformat(),
                'metadata': {
                    'version': '1.0',
                    'generated_by': 'EDAModule',
                    'analysis_components': list(self.analysis_results.keys())
                }
            }
            
            # Create the file
            with open(full_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Analysis results exported to: {full_path}")
            
            # Generate visualizations if requested
            if include_visualizations and self.analysis_results:
                viz_dir = export_dir if export_dir else '.'
                viz_path = os.path.join(viz_dir, f"eda_visualizations_{timestamp}")
                os.makedirs(viz_path, exist_ok=True)
                
                print(f"Generating visualizations in: {viz_path}")
                self._generate_visualizations(viz_path)
            
            return full_path
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error exporting analysis results: {e}")
            print(f"Error details: {error_details}")
            
            # Try to save error log
            try:
                error_log = os.path.join(os.path.dirname(full_path), f"error_log_{timestamp}.txt")
                with open(error_log, 'w') as f:
                    f.write(f"Error exporting analysis results: {e}\n\n")
                    f.write(error_details)
                print(f"Error details saved to: {error_log}")
            except:
                pass
                
            return None
    
    def _serialize_results(self, obj):
        """Convert numpy arrays and other non-serializable objects to lists"""
        if isinstance(obj, dict):
            return {key: self._serialize_results(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_results(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _generate_visualizations(self, output_dir):
        """Generate visualizations based on analysis results"""
        try:
            # Create visualization directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Set plot style
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # Track generated visualizations
            generated_plots = []
            
            # 1. Trend visualization if available
            if 'trend_analysis' in self.analysis_results:
                trend_data = self.analysis_results['trend_analysis']
                
                if 'linear_trend' in trend_data:
                    # Create trend plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot original data if available
                    if 'basic_stats' in trend_data and hasattr(self, 'engineered_features'):
                        data = self.engineered_features
                        if 'date' in data.columns and 'FuelBurn' in data.columns:
                            ax.plot(data['date'], data['FuelBurn'], 'o-', label='Actual Fuel Burn')
                            
                            # Add trend line
                            x = np.arange(len(data))
                            slope = trend_data['linear_trend']['slope']
                            intercept = trend_data['linear_trend']['intercept']
                            trend_line = slope * x + intercept
                            ax.plot(data['date'], trend_line, 'r--', label='Linear Trend')
                            
                            ax.set_title('Fuel Burn Trend Analysis')
                            ax.set_xlabel('Date')
                            ax.set_ylabel('Fuel Burn')
                            ax.legend()
                            
                            # Save plot
                            trend_plot_path = os.path.join(output_dir, 'trend_analysis.png')
                            plt.tight_layout()
                            plt.savefig(trend_plot_path)
                            plt.close()
                            
                            generated_plots.append(trend_plot_path)
            
            # 2. Seasonality visualization if available
            if 'seasonality_analysis' in self.analysis_results:
                seasonality_data = self.analysis_results['seasonality_analysis']
                
                if 'monthly_patterns' in seasonality_data:
                    monthly_stats = pd.DataFrame(seasonality_data['monthly_patterns']['monthly_stats'])
                    
                    if not monthly_stats.empty:
                        # Create seasonality plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot monthly seasonal indices
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[int(x)-1])
                        
                        sns.barplot(x='month_name', y='seasonal_index', data=monthly_stats, ax=ax)
                        ax.axhline(y=1.0, color='r', linestyle='--', label='Average')
                        
                        ax.set_title('Monthly Seasonality Pattern')
                        ax.set_xlabel('Month')
                        ax.set_ylabel('Seasonal Index')
                        ax.legend()
                        
                        # Save plot
                        seasonality_plot_path = os.path.join(output_dir, 'monthly_seasonality.png')
                        plt.tight_layout()
                        plt.savefig(seasonality_plot_path)
                        plt.close()
                        
                        generated_plots.append(seasonality_plot_path)
            
            # 3. Anomaly visualization if available
            if 'anomaly_analysis' in self.analysis_results and hasattr(self, 'engineered_features'):
                anomaly_data = self.analysis_results['anomaly_analysis']
                data = self.engineered_features
                
                if 'statistical_outliers' in anomaly_data and 'date' in data.columns and 'FuelBurn' in data.columns:
                    # Create anomaly plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot original data
                    ax.plot(data['date'], data['FuelBurn'], 'o-', label='Fuel Burn', alpha=0.7)
                    
                    # Plot outliers if any
                    if 'z_score' in anomaly_data['statistical_outliers']:
                        outlier_indices = anomaly_data['statistical_outliers']['z_score'].get('outlier_indices', [])
                        if outlier_indices:
                            outliers = data.iloc[outlier_indices]
                            ax.scatter(outliers['date'], outliers['FuelBurn'], color='red', s=100, 
                                      label='Z-score Outliers', zorder=5)
                    
                    ax.set_title('Fuel Burn with Detected Anomalies')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Fuel Burn')
                    ax.legend()
                    
                    # Save plot
                    anomaly_plot_path = os.path.join(output_dir, 'anomaly_detection.png')
                    plt.tight_layout()
                    plt.savefig(anomaly_plot_path)
                    plt.close()
                    
                    generated_plots.append(anomaly_plot_path)
            
            # 4. Distribution visualization if available
            if 'distribution_analysis' in self.analysis_results and hasattr(self, 'engineered_features'):
                data = self.engineered_features
                
                if 'FuelBurn' in data.columns:
                    # Create distribution plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Histogram
                    sns.histplot(data['FuelBurn'].dropna(), kde=True, ax=ax1)
                    ax1.set_title('Fuel Burn Distribution')
                    ax1.set_xlabel('Fuel Burn')
                    ax1.set_ylabel('Frequency')
                    
                    # QQ Plot
                    from scipy import stats
                    stats.probplot(data['FuelBurn'].dropna(), dist="norm", plot=ax2)
                    ax2.set_title('Q-Q Plot (Normal Distribution)')
                    
                    # Save plot
                    dist_plot_path = os.path.join(output_dir, 'distribution_analysis.png')
                    plt.tight_layout()
                    plt.savefig(dist_plot_path)
                    plt.close()
                    
                    generated_plots.append(dist_plot_path)
            
            # Create visualization index
            if generated_plots:
                index_path = os.path.join(output_dir, 'visualization_index.html')
                with open(index_path, 'w') as f:
                    f.write('<html><head><title>EDA Visualization Results</title>')
                    f.write('<style>body{font-family:Arial;margin:20px;} img{max-width:100%;}</style>')
                    f.write('</head><body>')
                    f.write('<h1>EDA Visualization Results</h1>')
                    f.write(f'<p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
                    
                    for plot_path in generated_plots:
                        plot_name = os.path.basename(plot_path)
                        f.write(f'<h2>{plot_name.replace(".png", "").replace("_", " ").title()}</h2>')
                        f.write(f'<img src="{plot_name}" />')
                        f.write('<hr>')
                    
                    f.write('</body></html>')
                
                print(f"Generated {len(generated_plots)} visualizations")
                print(f"Visualization index: {index_path}")
                
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            print(traceback.format_exc())


# Example usage and testing
if __name__ == "__main__":
    # Example usage of EDA Module
    print("Testing EDA & Deep Dive Analysis Module")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Generate realistic fuel burn data with seasonality, trend, and anomalies
    seasonal_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / 12) * 0.2
    trend = np.linspace(0, 0.5, len(dates))
    noise = np.random.normal(0, 0.1, len(dates))
    
    # Add some anomalies
    anomaly_indices = [10, 25, 35]
    anomaly_values = [0.8, -0.6, 1.2]
    
    fuel_burn = 1000 + seasonal_pattern * 200 + trend * 300 + noise * 50
    for idx, anomaly in zip(anomaly_indices, anomaly_values):
        if idx < len(fuel_burn):
            fuel_burn[idx] += anomaly * 200
    
    sample_data = pd.DataFrame({
        'date': dates,
        'FuelBurn': fuel_burn
    })
    
    print(f"Sample data created: {len(sample_data)} records")
    
    # Initialize EDA module
    eda = EDAModule()
    
    # Perform comprehensive analysis
    print("\n" + "="*60)
    print("PERFORMING COMPREHENSIVE EDA ANALYSIS")
    print("="*60)
    
    # Trend analysis
    trend_results = eda.perform_trend_analysis(sample_data)
    
    # Seasonality analysis
    seasonality_results = eda.analyze_seasonality_patterns(sample_data)
    
    # Correlation analysis (without external data for this example)
    correlation_results = eda.perform_correlation_analysis(sample_data)
    
    # Anomaly detection
    anomaly_results = eda.detect_anomalies_and_outliers(sample_data)
    
    # Distribution analysis
    distribution_results = eda.perform_distribution_analysis(sample_data)
    
    # Generate insights and recommendations
    insights_and_recommendations = eda.generate_comprehensive_insights()
    
    # Export results
    exported_file = eda.export_analysis_results()
    if exported_file:
        print(f"\nAnalysis results exported to: {exported_file}")
    
    print("\nEDA & Deep Dive Analysis Module testing completed!")
