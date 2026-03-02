#date: 2026-03-02T17:37:55Z
#url: https://api.github.com/gists/3c963e0a183861e15f3b6233bff7df83
#owner: https://api.github.com/users/Quinut-McGee

"""
Pandas accessor methods for NEAR blockchain analytics.

This module provides custom pandas DataFrame accessor methods for common
NEAR blockchain analytics tasks like daily transaction counts, volume analysis,
and account ranking.
"""

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
from typing import Optional


@register_dataframe_accessor("near")
class NearAccessor:
    """
    Pandas DataFrame accessor for NEAR blockchain analytics.
    
    Provides methods for common analytics tasks on NEAR blockchain data.
    """
    
    def __init__(self, pandas_obj):
        """Initialize the accessor with a pandas DataFrame."""
        self._validate(pandas_obj)
        self._df = pandas_obj
    
    def _validate(self, obj):
        """Validate that the DataFrame has required columns for NEAR analytics."""
        # Basic validation - at minimum should have some NEAR-related data
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("NearAccessor only works with pandas DataFrames")
    
    def daily_tx_count(self) -> pd.Series:
        """
        Calculate daily transaction counts.
        
        Returns:
            pandas Series with daily transaction counts, indexed by date
            
        Raises:
            ValueError: If required timestamp column is missing
        """
        if 'block_timestamp' not in self._df.columns:
            raise ValueError("DataFrame must contain 'block_timestamp' column for daily analysis")
        
        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(self._df['block_timestamp']):
            self._df['block_timestamp'] = pd.to_datetime(self._df['block_timestamp'])
        
        # Group by date and count transactions
        daily_counts = self._df.groupby(self._df['block_timestamp'].dt.date).size()
        daily_counts.name = 'transaction_count'
        
        return daily_counts
    
    def daily_volume(self) -> pd.Series:
        """
        Calculate daily transaction volume in NEAR.
        
        Returns:
            pandas Series with daily transaction volume in NEAR, indexed by date
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {'block_timestamp', 'amount'}
        missing_cols = required_cols - set(self._df.columns)
        
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(self._df['block_timestamp']):
            self._df['block_timestamp'] = pd.to_datetime(self._df['block_timestamp'])
        
        # Convert amount from yoctoNEAR to NEAR and group by date
        df_copy = self._df.copy()
        df_copy['amount_near'] = df_copy['amount'] / 1e24
        
        daily_volume = df_copy.groupby(df_copy['block_timestamp'].dt.date)['amount_near'].sum()
        daily_volume.name = 'daily_volume_near'
        
        return daily_volume
    
    def top_accounts(self, n: int = 10, by: str = 'transaction_count') -> pd.DataFrame:
        """
        Identify top accounts by transaction count or volume.
        
        Args:
            n: Number of top accounts to return (default: 10)
            by: Metric to rank by - 'transaction_count' or 'volume' (default: 'transaction_count')
            
        Returns:
            pandas DataFrame with top accounts and their metrics
            
        Raises:
            ValueError: If invalid ranking metric or missing required columns
        """
        if by not in {'transaction_count', 'volume'}:
            raise ValueError("by parameter must be 'transaction_count' or 'volume'")
        
        if by == 'transaction_count':
            if 'signer_id' not in self._df.columns:
                raise ValueError("DataFrame must contain 'signer_id' column for transaction count ranking")
            
            # Count transactions per signer
            top_accounts = (
                self._df['signer_id']
                .value_counts()
                .head(n)
                .reset_index()
            )
            top_accounts.columns = ['account_id', 'transaction_count']
            
        else:  # by == 'volume'
            required_cols = {'signer_id', 'amount'}
            missing_cols = required_cols - set(self._df.columns)
            
            if missing_cols:
                raise ValueError(f"DataFrame missing required columns for volume ranking: {missing_cols}")
            
            # Sum transaction volume per signer (in NEAR)
            df_copy = self._df.copy()
            df_copy['amount_near'] = df_copy['amount'] / 1e24
            
            top_accounts = (
                df_copy.groupby('signer_id')['amount_near']
                .sum()
                .sort_values(ascending=False)
                .head(n)
                .reset_index()
            )
            top_accounts.columns = ['account_id', 'total_volume_near']
        
        return top_accounts
    
    def transaction_success_rate(self) -> float:
        """
        Calculate the overall transaction success rate.
        
        Returns:
            Success rate as a float between 0 and 1
            
        Raises:
            ValueError: If success column is missing
        """
        if 'success' not in self._df.columns:
            raise ValueError("DataFrame must contain 'success' column for success rate calculation")
        
        if self._df.empty:
            return 0.0
        
        success_count = self._df['success'].sum()
        total_count = len(self._df)
        
        return success_count / total_count if total_count > 0 else 0.0
    
    def average_transaction_amount(self) -> float:
        """
        Calculate the average transaction amount in NEAR.
        
        Returns:
            Average transaction amount in NEAR
            
        Raises:
            ValueError: If amount column is missing
        """
        if 'amount' not in self._df.columns:
            raise ValueError("DataFrame must contain 'amount' column for average calculation")
        
        if self._df.empty:
            return 0.0
        
        # Convert to NEAR and calculate average
        amounts_near = self._df['amount'] / 1e24
        return amounts_near.mean()
    
    def time_series_analysis(self, freq: str = 'D') -> pd.DataFrame:
        """
        Perform comprehensive time series analysis of transaction data.
        
        Args:
            freq: Frequency for resampling ('D' for daily, 'H' for hourly, etc.)
            
        Returns:
            DataFrame with time series metrics
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {'block_timestamp', 'amount', 'success'}
        missing_cols = required_cols - set(self._df.columns)
        
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns for time series analysis: {missing_cols}")
        
        # Ensure timestamp is datetime and set as index
        df_copy = self._df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['block_timestamp']):
            df_copy['block_timestamp'] = pd.to_datetime(df_copy['block_timestamp'])
        
        df_copy.set_index('block_timestamp', inplace=True)
        df_copy['amount_near'] = df_copy['amount'] / 1e24
        
        # Resample and calculate metrics
        time_series = pd.DataFrame()
        time_series['transaction_count'] = df_copy.resample(freq).size()
        time_series['total_volume'] = df_copy['amount_near'].resample(freq).sum()
        time_series['avg_transaction_size'] = df_copy['amount_near'].resample(freq).mean()
        time_series['success_rate'] = df_copy['success'].resample(freq).mean()
        
        return time_series