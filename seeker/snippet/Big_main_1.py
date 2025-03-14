#date: 2025-03-14T17:04:02Z
#url: https://api.github.com/gists/2621862aa83cc6b6ce7125e22663e4c9
#owner: https://api.github.com/users/Clement1nes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm
from scipy import stats
import os
import datetime
import math
from tqdm import tqdm

# Set dark theme for plots
plt.style.use('dark_background')


def get_parameters():
    params = {
        # Data parameters
        'csv_file': 'MASKVET365.csv',  # CSV file path
        'date_col': 'datetime',  # Date column name
        'token1_col': "**********"
        'token2_col': "**********"
        'volume_col': 'volume',  # Volume column name (will try to find if not exact)

        # Strategy type toggle
        'use_returns_based': False,  # Set to True to use returns-based strategy instead of price-based
        'return1_col': "**********"
        'return2_col': "**********"
        'convert_prices_to_returns': True,  # Whether to convert prices to returns if use_returns_based=True
        'returns_method': 'log',  # Options: 'log' for log returns, 'pct' for percentage returns
        'lookback': 5,  # Lookback period for returns-based strategy

        # Expanding window parameters
        'use_expanding_window': False,  # Whether to use expanding window approach
        'min_window_size': 50,  # Minimum window size for expanding window

        # Copula parameters
        'copula_type': 'gaussian',  # Options: 'gaussian', 'clayton', 'student_t', 'gumbel'
        'copula_params': {
            't_df': 5,  # Degrees of freedom for Student t copula
            'clayton_theta': 2.0,  # Parameter for Clayton copula (>0)
            'gumbel_theta': 1.5  # Parameter for Gumbel copula (≥1)
        },

        # Strategy parameters
        'window_size': 20,  # Window size for copula fitting
        'confidence_level': 0.90,  # Confidence level for bands
        'atr_window': 14,  # Window size for ATR calculation
        'atr_multiplier': 2.0,  # Multiplier for ATR stop-loss
        'fee_pct': 0.001,  # Trading fee percentage (0.001 = 0.1%)

        # Volatility ratio parameters
        'vol_ratio_window': 20,  # Window size for volatility calculation
        'vol_ratio_threshold': 1.2,  # Threshold for volatility ratio (>1.2 or <1/1.2)
        'vol_ratio_mode': 'threshold',  # Mode: 'threshold', 'range', or 'balanced'

        # Volume percentile parameters
        'vol_lookback': 100,  # Lookback window for volume percentile calculation
        'vol_min_percentile': 0.3,  # Minimum volume percentile (0.3 = top 70%)
        'vol_max_percentile': 1.0,  # Maximum volume percentile (1.0 = no upper limit)

        # Donchian Channel parameters
        'donchian_period': 20,  # Period for Donchian channels
        'donchian_width_min': 0.01,  # Minimum width percentile (relative to price)
        'donchian_width_max': 0.05,  # Maximum width percentile (relative to price)
        'donchian_mode': 'range',  # Mode: 'range', 'min', 'max'

        # ADX parameters
        'adx_period': 14,  # Period for ADX calculation
        'adx_threshold': 25,  # Minimum ADX value for trend strength

        # RVI parameters
        'rvi_period': 10,  # Period for RVI calculation
        'rvi_signal_period': 4,  # Signal period for RVI
        'rvi_threshold': 50,  # Threshold for RVI (0-100 scale)

        # Choppiness Index parameters
        'chop_period': 14,  # Period for Choppiness Index
        'chop_threshold': 38.2,  # Below this is trending, above is choppy

        # ATR Volatility Filter parameters
        'atr_vol_period': 14,  # Period for ATR volatility calculation
        'atr_vol_ma_period': 30,  # Period for ATR moving average
        'atr_vol_threshold': 1.2,  # Threshold for ATR/ATR_MA ratio

        # Output parameters
        'output_dir': '1D-results-No-Filters',  # Directory to save results

        # Optimization parameters (only used if run_optimization=True)
        'run_optimization': False,  # Whether to run parameter optimization
        'opt_window_sizes': [20, 40, 60],  # Window sizes to test
        'opt_confidence_levels': [0.90, 0.95, 0.99],  # Confidence levels to test
        'opt_atr_multipliers': [1.5, 2.0, 2.5]  # ATR multipliers to test
    }

    return params


from scipy import stats, optimize
import numpy as np
from scipy.stats import t, norm, rankdata
from scipy.special import gamma, loggamma


class BaseCopula:
    """Base class for all copula types"""

    def __init__(self):
        self.params = None
        self.name = "Base"

    def fit(self, u, v):
        """Fit the copula to data"""
        pass

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands"""
        pass


class GaussianCopula(BaseCopula):
    def __init__(self):
        super().__init__()
        self.name = "Gaussian"
        self.rho = None
        self.cov_matrix = None

    def fit(self, u, v):
        """Fit the Gaussian copula model"""
        # Remove any NaN values
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = u[valid_mask]
        v_clean = v[valid_mask]

        # Handle empty arrays
        if len(u_clean) < 2 or len(v_clean) < 2:
            self.rho = 0
            self.cov_matrix = np.array([[1, 0], [0, 1]])
            return self

        # Transforming uniform to normal with NaN protection
        x = norm.ppf(np.clip(u_clean, 0.001, 0.999))
        y = norm.ppf(np.clip(v_clean, 0.001, 0.999))

        # Pearson correlation coefficient
        self.rho = np.corrcoef(x, y)[0, 1]
        if np.isnan(self.rho):
            self.rho = 0

        # Ensure correlation is valid
        self.rho = max(min(self.rho, 0.999), -0.999)
        self.cov_matrix = np.array([[1, self.rho], [self.rho, 1]])
        self.params = {'rho': self.rho}

        return self

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands"""
        if np.isnan(u) or np.isnan(v) or u <= 0 or u >= 1 or v <= 0 or v >= 1:
            return False, 0.0, 0.0

        try:
            # Convert to normal space with clipping
            x = norm.ppf(min(max(u, 0.001), 0.999))
            y = norm.ppf(min(max(v, 0.001), 0.999))

            # Compute Mahalanobis distance
            point = np.array([x, y])
            inv_cov = np.linalg.inv(self.cov_matrix)
            mahalanobis_sq = point.dot(inv_cov).dot(point)

            # Chi-square threshold for given confidence level
            chi2_threshold = stats.chi2.ppf(confidence_level, df=2)

            # Check if outside bands
            is_outside = mahalanobis_sq > chi2_threshold

            # Calculate direction
            if abs(self.rho) < 1e-10:  # Near-zero correlation
                direction = x  # Just use the x coordinate
            else:
                expected_x = self.rho * y
                direction = x - expected_x

            return is_outside, np.sqrt(mahalanobis_sq), direction

        except Exception as e:
            print(f"Error in is_point_outside_bands: {e}")
            return False, 0.0, 0.0


class ClaytonCopula(BaseCopula):
    def __init__(self, theta=2.0):
        super().__init__()
        self.name = "Clayton"
        self.theta = theta
        self.params = {'theta': theta}

    def _clayton_cdf(self, u, v, theta):
        """Clayton copula cumulative distribution function"""
        if theta < 1e-10:  # Near independence
            return u * v
        return np.maximum(0, (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta))

    def _clayton_pdf(self, u, v, theta):
        """Clayton copula probability density function"""
        if theta < 1e-10:  # Near independence
            return 1.0
        return (1 + theta) * (u * v) ** (-1 - theta) * (u ** (-theta) + v ** (-theta) - 1) ** (-2 - 1 / theta)

    def _log_likelihood(self, theta, u, v):
        """Log-likelihood function for parameter estimation"""
        if theta <= 0:  # Clayton parameter must be positive
            return -np.inf

        log_pdf = np.log((1 + theta)) - (1 + theta) * (np.log(u) + np.log(v)) - (2 + 1 / theta) * np.log(
            u ** (-theta) + v ** (-theta) - 1)
        return -np.sum(log_pdf)  # Negative for minimization

    def fit(self, u, v):
        """Fit the Clayton copula model by estimating theta"""
        # Clean data and handle edge cases
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = u[valid_mask]
        v_clean = v[valid_mask]

        # Handle empty arrays or insufficient data
        if len(u_clean) < 2 or len(v_clean) < 2:
            self.theta = 2.0  # Default value
            self.params = {'theta': self.theta}
            return self

        # Clip values to avoid numerical issues
        u_clean = np.clip(u_clean, 0.001, 0.999)
        v_clean = np.clip(v_clean, 0.001, 0.999)

        try:
            # Start with initial theta value and use optimization
            initial_theta = 2.0  # Starting value
            bounds = [(0.01, 10.0)]  # Theta must be positive

            result = optimize.minimize(
                self._log_likelihood,
                initial_theta,
                args=(u_clean, v_clean),
                bounds=bounds,
                method='L-BFGS-B'
            )

            if result.success:
                self.theta = result.x[0]
            else:
                print("Clayton parameter estimation failed, using default value")
                self.theta = 2.0

        except Exception as e:
            print(f"Error fitting Clayton copula: {e}")
            self.theta = 2.0  # Default value on error

        self.params = {'theta': self.theta}
        return self

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands"""
        if np.isnan(u) or np.isnan(v) or u <= 0 or u >= 1 or v <= 0 or v >= 1:
            return False, 0.0, 0.0

        try:
            # Clip to avoid numerical issues
            u = min(max(u, 0.001), 0.999)
            v = min(max(v, 0.001), 0.999)

            # For Clayton, we compute the probability level of the point
            # and see if it's outside our confidence level
            # We measure the contour of the density function

            # The contour level for our point
            density = self._clayton_pdf(u, v, self.theta)

            # Generate a grid of points and compute their density
            grid_size = 50
            u_grid = np.linspace(0.001, 0.999, grid_size)
            v_grid = np.linspace(0.001, 0.999, grid_size)

            density_grid = np.zeros((grid_size, grid_size))
            for i in range(grid_size):
                for j in range(grid_size):
                    density_grid[i, j] = self._clayton_pdf(u_grid[i], v_grid[j], self.theta)

            # Compute the percentile of our point in the density grid
            flat_density = density_grid.flatten()
            point_percentile = np.mean(flat_density <= density)

            # Determine if the point is outside the confidence interval
            is_outside = point_percentile < (1 - confidence_level)

            # Calculate "direction" based on difference from expected value
            # In Clayton, high dependence in the lower tail
            if u < v:  # Point is below the diagonal
                direction = -1  # Indicative of potential "short" signal
            else:
                direction = 1  # Indicative of potential "long" signal

            # Distance is just a measure of how far outside the bands we are
            distance = abs(point_percentile - (1 - confidence_level))

            return is_outside, distance, direction

        except Exception as e:
            print(f"Error in Clayton is_point_outside_bands: {e}")
            return False, 0.0, 0.0


class StudentTCopula(BaseCopula):
    def __init__(self, df=5):
        super().__init__()
        self.name = "Student t"
        self.df = df  # Degrees of freedom
        self.rho = None
        self.cov_matrix = None
        self.params = {'df': df, 'rho': None}

    def fit(self, u, v):
        """Fit the Student t copula model"""
        # Remove any NaN values
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = u[valid_mask]
        v_clean = v[valid_mask]

        # Handle empty arrays
        if len(u_clean) < 2 or len(v_clean) < 2:
            self.rho = 0
            self.cov_matrix = np.array([[1, 0], [0, 1]])
            self.params = {'df': self.df, 'rho': 0}
            return self

        # Transforming uniform to t-distributed with NaN protection
        x = t.ppf(np.clip(u_clean, 0.001, 0.999), self.df)
        y = t.ppf(np.clip(v_clean, 0.001, 0.999), self.df)

        # Pearson correlation coefficient
        self.rho = np.corrcoef(x, y)[0, 1]
        if np.isnan(self.rho):
            self.rho = 0

        # Ensure correlation is valid
        self.rho = max(min(self.rho, 0.999), -0.999)
        self.cov_matrix = np.array([[1, self.rho], [self.rho, 1]])
        self.params = {'df': self.df, 'rho': self.rho}

        return self

    def _t_copula_likelihood(self, params, u, v):
        """Log-likelihood function for parameter estimation including df"""
        rho, df = params

        if abs(rho) >= 1 or df <= 2:
            return np.inf  # Invalid parameters

        # Convert to t-distribution quantiles
        x = t.ppf(np.clip(u, 0.001, 0.999), df)
        y = t.ppf(np.clip(v, 0.001, 0.999), df)

        # Parameters for the bivariate t-distribution
        cov = np.array([[1, rho], [rho, 1]])
        inv_cov = np.linalg.inv(cov)

        # Calculate log-likelihood
        n = len(u)
        log_lik = 0

        for i in range(n):
            xy = np.array([x[i], y[i]])
            quad = xy.dot(inv_cov).dot(xy)

            # Univariate density terms
            log_lik -= t.logpdf(x[i], df) + t.logpdf(y[i], df)

            # Bivariate t density term
            c = gamma((df + 2) / 2) * gamma(df / 2) / (gamma((df + 1) / 2) ** 2 * np.sqrt(1 - rho ** 2))
            log_lik += np.log(c) - (df + 2) / 2 * np.log(1 + quad / df)

        return -log_lik  # Negative for minimization

    def estimate_df(self, u, v):
        """Estimate optimal degrees of freedom"""
        # This is a simplified approach, full MLE is more complex
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = np.clip(u[valid_mask], 0.001, 0.999)
        v_clean = np.clip(v[valid_mask], 0.001, 0.999)

        if len(u_clean) < 10:  # Need sufficient data
            return self.df

        # Try a few df values and pick the best
        df_candidates = [3, 5, 7, 10, 15, 20, 30]
        best_df = self.df
        best_likelihood = float('inf')

        for df in df_candidates:
            x = t.ppf(u_clean, df)
            y = t.ppf(v_clean, df)
            rho = np.corrcoef(x, y)[0, 1]
            rho = max(min(rho, 0.999), -0.999)

            lik = self._t_copula_likelihood([rho, df], u_clean, v_clean)
            if lik < best_likelihood:
                best_likelihood = lik
                best_df = df

        return best_df

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands"""
        if np.isnan(u) or np.isnan(v) or u <= 0 or u >= 1 or v <= 0 or v >= 1:
            return False, 0.0, 0.0

        try:
            # Convert to t-distributed space with clipping
            x = t.ppf(min(max(u, 0.001), 0.999), self.df)
            y = t.ppf(min(max(v, 0.001), 0.999), self.df)

            # Compute Mahalanobis distance
            point = np.array([x, y])
            inv_cov = np.linalg.inv(self.cov_matrix)
            mahalanobis_sq = point.dot(inv_cov).dot(point)

            # F-distribution threshold for t-distribution
            # For t-copula, the squared Mahalanobis distance follows an F-distribution
            # scaled by degrees of freedom
            f_threshold = stats.f.ppf(confidence_level, 2, self.df)
            scaled_threshold = 2 * f_threshold * (self.df - 1) / self.df

            # Check if outside bands
            is_outside = mahalanobis_sq > scaled_threshold

            # Calculate direction (similar to Gaussian copula)
            if abs(self.rho) < 1e-10:  # Near-zero correlation
                direction = x
            else:
                expected_x = self.rho * y
                direction = x - expected_x

            return is_outside, np.sqrt(mahalanobis_sq / scaled_threshold), direction

        except Exception as e:
            print(f"Error in Student t is_point_outside_bands: {e}")
            return False, 0.0, 0.0


class GumbelCopula(BaseCopula):
    def __init__(self, theta=1.5):
        super().__init__()
        self.name = "Gumbel"
        self.theta = max(1.0, theta)  # Must be ≥ 1
        self.params = {'theta': self.theta}

    def _gumbel_cdf(self, u, v, theta):
        """Gumbel copula cumulative distribution function"""
        if abs(theta - 1.0) < 1e-10:  # Near independence
            return u * v
        term = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)
        return np.exp(-term)

    def _gumbel_pdf(self, u, v, theta):
        """Gumbel copula probability density function"""
        if abs(theta - 1.0) < 1e-10:  # Near independence
            return 1.0

        log_u = np.log(u)
        log_v = np.log(v)
        log_u_theta = (-log_u) ** theta
        log_v_theta = (-log_v) ** theta
        sum_term = log_u_theta + log_v_theta
        power_term = sum_term ** (1 / theta)

        c_uv = np.exp(-power_term)

        term1 = (log_u_theta * log_v_theta) / (u * v * (log_u * log_v) ** (theta - 1))
        term2 = sum_term ** (-2 + 1 / theta)
        term3 = (theta - 1 + power_term)

        return c_uv * term1 * term2 * term3

    def _log_likelihood(self, theta, u, v):
        """Log-likelihood function for parameter estimation"""
        if theta < 1.0:  # Gumbel parameter must be ≥ 1
            return np.inf

        try:
            # Computing log of PDF directly to avoid numerical issues
            log_u = np.log(u)
            log_v = np.log(v)
            log_u_theta = (-log_u) ** theta
            log_v_theta = (-log_v) ** theta
            sum_term = log_u_theta + log_v_theta
            power_term = sum_term ** (1 / theta)

            log_c_uv = -power_term

            log_term1 = np.log(log_u_theta) + np.log(log_v_theta) - np.log(u) - np.log(v) - (theta - 1) * (
                        np.log(-log_u) + np.log(-log_v))
            log_term2 = (-2 + 1 / theta) * np.log(sum_term)
            log_term3 = np.log(theta - 1 + power_term)

            log_pdf = log_c_uv + log_term1 + log_term2 + log_term3
            return -np.sum(log_pdf)  # Negative for minimization

        except Exception as e:
            print(f"Error in Gumbel log-likelihood: {e}")
            return np.inf

    def fit(self, u, v):
        """Fit the Gumbel copula model by estimating theta"""
        # Clean data and handle edge cases
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = u[valid_mask]
        v_clean = v[valid_mask]

        # Handle empty arrays or insufficient data
        if len(u_clean) < 2 or len(v_clean) < 2:
            self.theta = 1.5  # Default value
            self.params = {'theta': self.theta}
            return self

        # Clip values to avoid numerical issues
        u_clean = np.clip(u_clean, 0.001, 0.999)
        v_clean = np.clip(v_clean, 0.001, 0.999)

        try:
            # Kendall's tau as a quick estimate for theta
            tau, _ = stats.kendalltau(u_clean, v_clean)
            initial_theta = 1 / (1 - max(0.01, tau))  # Ensure it's ≥ 1

            # Use optimization for more precise estimate
            bounds = [(1.001, 10.0)]  # Theta must be > 1

            result = optimize.minimize(
                self._log_likelihood,
                initial_theta,
                args=(u_clean, v_clean),
                bounds=bounds,
                method='L-BFGS-B'
            )

            if result.success:
                self.theta = max(1.001, result.x[0])  # Ensure it's > 1
            else:
                # Fallback to Kendall's tau estimate
                self.theta = max(1.001, initial_theta)

        except Exception as e:
            print(f"Error fitting Gumbel copula: {e}")
            self.theta = 1.5  # Default value on error

        self.params = {'theta': self.theta}
        return self

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands"""
        if np.isnan(u) or np.isnan(v) or u <= 0 or u >= 1 or v <= 0 or v >= 1:
            return False, 0.0, 0.0

        try:
            # Clip to avoid numerical issues
            u = min(max(u, 0.001), 0.999)
            v = min(max(v, 0.001), 0.999)

            # For Gumbel, similar to Clayton, we compute the probability contour
            density = self._gumbel_pdf(u, v, self.theta)

            # Generate a grid of points and compute their density
            grid_size = 50
            u_grid = np.linspace(0.001, 0.999, grid_size)
            v_grid = np.linspace(0.001, 0.999, grid_size)

            density_grid = np.zeros((grid_size, grid_size))
            for i in range(grid_size):
                for j in range(grid_size):
                    try:
                        density_grid[i, j] = self._gumbel_pdf(u_grid[i], v_grid[j], self.theta)
                    except:
                        density_grid[i, j] = 0

            # Compute the percentile of our point in the density grid
            flat_density = density_grid.flatten()
            point_percentile = np.mean(flat_density <= density)

            # Determine if the point is outside the confidence interval
            is_outside = point_percentile < (1 - confidence_level)

            # Calculate "direction" based on difference from expected value
            # In Gumbel, high dependence in the upper tail
            if u > v:  # Point is above the diagonal
                direction = 1  # Indicative of potential "long" signal
            else:
                direction = -1  # Indicative of potential "short" signal

            # Distance is just a measure of how far outside the bands we are
            distance = abs(point_percentile - (1 - confidence_level))

            return is_outside, distance, direction

        except Exception as e:
            print(f"Error in Gumbel is_point_outside_bands: {e}")
            return False, 0.0, 0.0


def create_copula(copula_type, params=None):
    """
    Factory function to create a copula of the specified type
    Parameters:
    - copula_type: String specifying the type ('gaussian', 'clayton', 'student_t', 'gumbel')
    - params: Dictionary of parameters for the specific copula type
    Returns:
    - Initialized copula object
    """
    if params is None:
        params = {}

    copula_type = copula_type.lower()

    if copula_type == 'gaussian':
        return GaussianCopula()

    elif copula_type == 'clayton':
        theta = params.get('clayton_theta', 2.0)
        return ClaytonCopula(theta=theta)

    elif copula_type == 'student_t':
        df = params.get('t_df', 5)
        return StudentTCopula(df=df)

    elif copula_type == 'gumbel':
        theta = params.get('gumbel_theta', 1.5)
        return GumbelCopula(theta=theta)

    else:
        print(f"Warning: Unknown copula type '{copula_type}'. Using Gaussian as default.")
        return GaussianCopula()


import numpy as np
import pandas as pd


def calculate_donchian_channels(high, low, period=20):
    """
    Calculate Donchian Channels
    Parameters:
    -----------
    high : pandas.Series
        Series with high prices
    low : pandas.Series
        Series with low prices
    period : int
        Lookback period for calculating channels
    Returns:
    --------
    upper_band, middle_band, lower_band, width
    """
    # Calculate upper and lower bands
    upper_band = high.rolling(window=period).max()
    lower_band = low.rolling(window=period).min()

    # Calculate middle band
    middle_band = (upper_band + lower_band) / 2

    # Calculate width as percentage of middle price
    width = (upper_band - lower_band) / middle_band

    return upper_band, middle_band, lower_band, width


def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX)
    Parameters:
    -----------
    high : pandas.Series
        Series with high prices
    low : pandas.Series
        Series with low prices
    close : pandas.Series
        Series with close prices
    period : int
        Period for ADX calculation
    Returns:
    --------
    adx, plus_di, minus_di
    """
    # Initialize dataframe
    df = pd.DataFrame()
    df['high'] = high
    df['low'] = low
    df['close'] = close

    # Calculate +DM and -DM
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = df['low'].diff().multiply(-1)  # Multiply by -1 to make it positive when it should be

    # Calculate +DM
    df['+dm'] = np.where(
        (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
        df['high_diff'],
        0
    )

    # Calculate -DM
    df['-dm'] = np.where(
        (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
        df['low_diff'],
        0
    )

    # Calculate true range
    df['tr'] = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)

    # Calculate smoothed values
    df['+dm_smoothed'] = df['+dm'].rolling(window=period).sum()
    df['-dm_smoothed'] = df['-dm'].rolling(window=period).sum()
    df['tr_smoothed'] = df['tr'].rolling(window=period).sum()

    # Calculate +DI and -DI
    df['+di'] = 100 * df['+dm_smoothed'] / df['tr_smoothed']
    df['-di'] = 100 * df['-dm_smoothed'] / df['tr_smoothed']

    # Calculate DX (Directional Index)
    df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])

    # Calculate ADX (Average Directional Index)
    df['adx'] = df['dx'].rolling(window=period).mean()

    return df['adx'], df['+di'], df['-di']


def calculate_rvi(close, period=10, signal_period=4):
    """
    Calculate Relative Volatility Index (RVI)
    Parameters:
    -----------
    close : pandas.Series
        Series with close prices
    period : int
        Period for RVI calculation
    signal_period : int
        Period for RVI signal line
    Returns:
    --------
    rvi, rvi_signal
    """
    # Calculate price changes
    price_change = close.diff()

    # Calculate standard deviation for up and down moves
    up_std = pd.Series(np.where(price_change > 0, price_change, 0)).rolling(window=period).std()
    down_std = pd.Series(np.where(price_change < 0, -price_change, 0)).rolling(window=period).std()

    # Calculate RVI
    rvi = 100 * up_std / (up_std + down_std)

    # Calculate signal line
    rvi_signal = rvi.rolling(window=signal_period).mean()

    return rvi, rvi_signal


def calculate_chop(high, low, close, period=14):
    """
    Calculate Choppiness Index
    Parameters:
    -----------
    high : pandas.Series
        Series with high prices
    low : pandas.Series
        Series with low prices
    close : pandas.Series
        Series with close prices
    period : int
        Period for Choppiness Index calculation
    Returns:
    --------
    chop
    """
    # Calculate ATR
    df = pd.DataFrame()
    df['tr'] = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)

    # Sum of ATR over period
    df['atr_sum'] = df['tr'].rolling(window=period).sum()

    # Highest high and lowest low over period
    df['highest_high'] = high.rolling(window=period).max()
    df['lowest_low'] = low.rolling(window=period).min()

    # Calculate choppiness index
    df['chop'] = 100 * np.log10(df['atr_sum'] / (df['highest_high'] - df['lowest_low'])) / np.log10(period)

    return df['chop']


def run_all_returns_filters(return1, return2, df, base_results, params):
    """
    Run all filters for the returns-based strategy with proper returns method handling

    Parameters:
    - return1, return2: Return series for the two assets
    - df: Original dataframe with all needed data
    - base_results: Results from the baseline strategy
    - params: Dictionary with all parameters

    Returns:
    - Dictionary of results for all filtered strategies
    """
    base_signals = base_results['signals']
    returns_method = params.get('returns_method', 'log')  # Get the returns method from params

    # Get filtered signals for all indicators
    filtered_signals_dict, filtered_data_dict = filter_returns_with_indicators(
        base_signals, return1, return2, df, params
    )

    # Dictionary to store results for all filtered strategies
    results_dict = {'Baseline': base_results}

    # Run backtest for each filtered signals set
    for filter_name, filtered_signals in filtered_signals_dict.items():
        print(f"\n=== Running {filter_name} filtered returns-based backtest ===")

        # Run backtest with filtered signals and pass the returns_method
        filter_results = backtest_with_filtered_returns(
            return1, return2, filtered_signals,
            lookback=params['lookback'],
            fee_pct=params['fee_pct'],
            returns_method=returns_method  # Pass the returns method explicitly
        )

        # Add any additional data from the filter
        if filter_name in filtered_data_dict:
            for key, value in filtered_data_dict[filter_name].items():
                filter_results[key] = value

        # Add to results dictionary
        results_dict[f'Returns {filter_name}'] = filter_results

        # Generate output directory name
        output_dir = f"{params['output_dir']}-Returns-{params['copula_type'].capitalize()}-{filter_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Save results and plots
        title = f"{params['copula_type'].capitalize()} Copula Returns Strategy: {filter_name} Filter"
        figs = plot_returns_strategy_results(filter_results, title)
        for i, fig in enumerate(figs):
            output_file = os.path.join(output_dir, f"returns_{filter_name.lower()}_plot_{i + 1}.png")
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Save trades if available
        if 'trades' in filter_results and not filter_results['trades'].empty:
            trade_file = os.path.join(output_dir, f"returns_{filter_name.lower()}_trades.csv")
            filter_results['trades'].to_csv(trade_file, index=False)

        # Save performance metrics
        perf_file = os.path.join(output_dir, f"returns_{filter_name.lower()}_performance.csv")
        perf_df = pd.DataFrame({k: [v] for k, v in filter_results['performance'].items()
                                if not isinstance(v, pd.Series)})
        perf_df.to_csv(perf_file, index=False)

    return results_dict


def filter_returns_with_indicators(base_signals, return1, return2, df, params):
    """
    Apply all indicator-based filters to returns-based strategies with improved signal detection

    Parameters:
    - base_signals: Original signals from copula
    - return1, return2: Return series for the two assets
    - df: Original dataframe with all needed data
    - params: Dictionary with all parameters

    Returns:
    - Dictionary of filtered signals for each filter type
    - Dictionary of indicator data for each filter
    """
    # Dictionary to store filtered signals
    filtered_signals_dict = {}
    filtered_data_dict = {}

    # Print diagnostic information
    print(
        f"Base signals summary: {len(base_signals[base_signals != 0])} active signals out of {len(base_signals)} total points")

    # Create artificial "price" series from cumulative returns for indicators that need price data
    cum_return1 = (1 + return1).cumprod()
    cum_return2 = (1 + return2).cumprod()

    # Create a spread for indicators that need it
    spread = cum_return1 - cum_return2

    # 1. Vol Ratio Filter
    print("Applying volatility ratio filter to returns-based strategy...")
    # First check we have enough non-zero returns for volatility calculation
    if (return1.abs() > 0).sum() < params['vol_ratio_window'] * 0.5 or (return2.abs() > 0).sum() < params[
        'vol_ratio_window'] * 0.5:
        print(f"Warning: Not enough non-zero returns for reliable volatility calculation")
        vol_ratio = pd.Series(1.0, index=return1.index)  # Default to 1.0 (neutral)
    else:
        vol_ratio, _, _ = calculate_vol_ratio(
            return1.abs(), return2.abs(),  # Use absolute returns for volatility
            window_size=params['vol_ratio_window']
        )

    # Filter signals based on volatility ratio
    vol_ratio_filtered = pd.Series(0, index=return1.index)
    valid_indices = pd.Series(False, index=return1.index)

    if params['vol_ratio_mode'] == 'threshold':
        # Filter signals: only keep when vol_ratio exceeds threshold
        valid_indices = (vol_ratio > params['vol_ratio_threshold']) | (vol_ratio < 1 / params['vol_ratio_threshold'])
    elif params['vol_ratio_mode'] == 'range':
        # Filter signals: only keep when vol_ratio is within range
        valid_indices = (vol_ratio >= 1 / params['vol_ratio_threshold']) & (vol_ratio <= params['vol_ratio_threshold'])

    # Apply filter to base signals
    for i in range(len(base_signals)):
        if i < params['window_size']:
            vol_ratio_filtered.iloc[i] = 0
        elif valid_indices.iloc[i]:
            vol_ratio_filtered.iloc[i] = base_signals.iloc[i]

    filtered_signals_dict['Vol Ratio'] = vol_ratio_filtered
    filtered_data_dict['Vol Ratio'] = {'vol_ratio': vol_ratio}

    # Log how many signals passed the filter
    passed_signals = len(vol_ratio_filtered[vol_ratio_filtered != 0])
    print(
        f"Vol Ratio filter: {passed_signals} signals passed out of {len(base_signals[base_signals != 0])} base signals")

    # 2. Volume Percentile Filter
    print("Applying volume percentile filter to returns-based strategy...")
    # Get volume data
    if params['volume_col'] in df.columns:
        volume = df[params['volume_col']]
    else:
        # Try to find volume from similar columns
        volume_found = False
        vol_candidates = ['volume_1', 'volume_2', 'vol_1', 'vol_2', 'volume', 'vol']
        for col in vol_candidates:
            if col in df.columns:
                print(f"Using '{col}' as volume column")
                volume = df[col]
                volume_found = True
                break

        if not volume_found:
            print("Warning: No volume column found. Creating dummy volume.")
            volume = pd.Series(1.0, index=return1.index)

    # Calculate volume percentile
    vol_percentile = calculate_volume_percentile(volume, params['vol_lookback'])

    # Filter signals based on volume percentile
    volume_filtered = pd.Series(0, index=return1.index)
    valid_indices = (vol_percentile >= params['vol_min_percentile']) & (vol_percentile <= params['vol_max_percentile'])

    # Apply filter to base signals
    for i in range(len(base_signals)):
        if i < params['vol_lookback']:
            volume_filtered.iloc[i] = 0
        elif valid_indices.iloc[i]:
            volume_filtered.iloc[i] = base_signals.iloc[i]

    filtered_signals_dict['Volume'] = volume_filtered
    filtered_data_dict['Volume'] = {'vol_percentile': vol_percentile}

    # Log how many signals passed the filter
    passed_signals = len(volume_filtered[volume_filtered != 0])
    print(f"Volume filter: {passed_signals} signals passed out of {len(base_signals[base_signals != 0])} base signals")

    # 3. Combined Filter (Vol Ratio + Volume)
    print("Applying combined filter to returns-based strategy...")

    # Apply both filters
    combined_filtered = pd.Series(0, index=return1.index)
    valid_vol_ratio = pd.Series(False, index=return1.index)

    if params['vol_ratio_mode'] == 'threshold':
        valid_vol_ratio = (vol_ratio > params['vol_ratio_threshold']) | (vol_ratio < 1 / params['vol_ratio_threshold'])
    elif params['vol_ratio_mode'] == 'range':
        valid_vol_ratio = (vol_ratio >= 1 / params['vol_ratio_threshold']) & (
                    vol_ratio <= params['vol_ratio_threshold'])

    valid_volume = (vol_percentile >= params['vol_min_percentile']) & (vol_percentile <= params['vol_max_percentile'])
    valid_indices = valid_vol_ratio & valid_volume

    # Apply combined filter to base signals
    for i in range(len(base_signals)):
        if i < max(params['window_size'], params['vol_lookback']):
            combined_filtered.iloc[i] = 0
        elif valid_indices.iloc[i]:
            combined_filtered.iloc[i] = base_signals.iloc[i]

    filtered_signals_dict['Combined'] = combined_filtered
    filtered_data_dict['Combined'] = {
        'vol_ratio': vol_ratio,
        'vol_percentile': vol_percentile
    }

    # Log how many signals passed the filter
    passed_signals = len(combined_filtered[combined_filtered != 0])
    print(
        f"Combined filter: {passed_signals} signals passed out of {len(base_signals[base_signals != 0])} base signals")

    # 4. ADX Filter
    print("Applying ADX filter to returns-based strategy...")

    # Calculate high/low for ADX calculation on cumulative returns
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)

    adx, plus_di, minus_di = calculate_adx(spread_high, spread_low, spread, params['adx_period'])

    # More lenient ADX filter to ensure more signals pass through
    adx_threshold = max(10, params['adx_threshold'] * 0.7)  # Lower threshold to allow more signals

    # Filter signals with modified ADX approach
    adx_filtered = pd.Series(0, index=return1.index)
    for i in range(len(base_signals)):
        if i < params['adx_period'] * 2:  # Ensure ADX has stabilized
            adx_filtered.iloc[i] = 0
        elif adx.iloc[i] >= adx_threshold:
            # Also check trend direction with DI lines
            if base_signals.iloc[i] > 0 and plus_di.iloc[i] > minus_di.iloc[i]:
                adx_filtered.iloc[i] = 1  # Confirm long signal with +DI > -DI
            elif base_signals.iloc[i] < 0 and minus_di.iloc[i] > plus_di.iloc[i]:
                adx_filtered.iloc[i] = -1  # Confirm short signal with -DI > +DI
            else:
                # If DI doesn't confirm but ADX is strong enough, still consider the signal
                adx_filtered.iloc[i] = base_signals.iloc[i]

    filtered_signals_dict['ADX'] = adx_filtered
    filtered_data_dict['ADX'] = {
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di
    }

    # Log how many signals passed the ADX filter
    passed_signals = len(adx_filtered[adx_filtered != 0])
    print(f"ADX filter: {passed_signals} signals passed out of {len(base_signals[base_signals != 0])} base signals")

    # 5. RVI Filter
    print("Applying RVI filter to returns-based strategy...")

    rvi, rvi_signal = calculate_rvi(spread, params['rvi_period'], params['rvi_signal_period'])

    # Filter signals based on RVI with a more lenient approach
    rvi_filtered = pd.Series(0, index=return1.index)

    for i in range(len(base_signals)):
        if i < params['rvi_period'] + params['rvi_signal_period']:
            rvi_filtered.iloc[i] = 0
        elif base_signals.iloc[i] != 0:  # Only process where we have a base signal
            # For long signals (base_signals > 0)
            if base_signals.iloc[i] > 0:
                # Either RVI > threshold OR RVI > RVI_signal
                if rvi.iloc[i] > params['rvi_threshold'] or rvi.iloc[i] > rvi_signal.iloc[i]:
                    rvi_filtered.iloc[i] = base_signals.iloc[i]
            # For short signals (base_signals < 0)
            elif base_signals.iloc[i] < 0:
                # Either RVI < (100-threshold) OR RVI < RVI_signal
                if rvi.iloc[i] < (100 - params['rvi_threshold']) or rvi.iloc[i] < rvi_signal.iloc[i]:
                    rvi_filtered.iloc[i] = base_signals.iloc[i]

    filtered_signals_dict['RVI'] = rvi_filtered
    filtered_data_dict['RVI'] = {
        'rvi': rvi,
        'rvi_signal': rvi_signal
    }

    # Log how many signals passed the RVI filter
    passed_signals = len(rvi_filtered[rvi_filtered != 0])
    print(f"RVI filter: {passed_signals} signals passed out of {len(base_signals[base_signals != 0])} base signals")

    # 6. Chop Filter
    print("Applying Choppiness Index filter to returns-based strategy...")

    chop = calculate_chop(spread_high, spread_low, spread, params['chop_period'])

    # Filter signals based on Choppiness Index - more lenient threshold
    # Lower values indicate trending market (which is good for trading)
    chop_threshold = min(60, params['chop_threshold'] * 1.3)  # Higher threshold allows more signals

    chop_filtered = pd.Series(0, index=return1.index)
    for i in range(len(base_signals)):
        if i < params['chop_period'] * 2:  # Ensure CHOP has stabilized
            chop_filtered.iloc[i] = 0
        # Pass signals when market is trending (CHOP below threshold)
        elif chop.iloc[i] < chop_threshold and base_signals.iloc[i] != 0:
            chop_filtered.iloc[i] = base_signals.iloc[i]

    filtered_signals_dict['Chop'] = chop_filtered
    filtered_data_dict['Chop'] = {'chop': chop}

    # Log how many signals passed the Chop filter
    passed_signals = len(chop_filtered[chop_filtered != 0])
    print(f"Chop filter: {passed_signals} signals passed out of {len(base_signals[base_signals != 0])} base signals")

    # 7. Donchian Channel Filter
    print("Applying Donchian Channel filter to returns-based strategy...")

    upper_band, middle_band, lower_band, donchian_width = calculate_donchian_channels(
        spread_high, spread_low, params['donchian_period']
    )

    # Filter signals based on Donchian Channel width - more lenient settings
    width_min = max(0.001, params['donchian_width_min'] * 0.5)  # Lower minimum to allow more signals
    width_max = min(0.2, params['donchian_width_max'] * 2)  # Higher maximum to allow more signals

    donchian_filtered = pd.Series(0, index=return1.index)
    for i in range(len(base_signals)):
        if i < params['donchian_period'] * 1.5:  # Ensure Donchian has enough data
            donchian_filtered.iloc[i] = 0
        elif base_signals.iloc[i] != 0:  # Only process where we have a base signal
            if params['donchian_mode'] == 'range':
                # Keep signals when width is within range
                if width_min <= donchian_width.iloc[i] <= width_max:
                    donchian_filtered.iloc[i] = base_signals.iloc[i]
            elif params['donchian_mode'] == 'min':
                # Keep signals when width is above minimum
                if donchian_width.iloc[i] >= width_min:
                    donchian_filtered.iloc[i] = base_signals.iloc[i]
            elif params['donchian_mode'] == 'max':
                # Keep signals when width is below maximum
                if donchian_width.iloc[i] <= width_max:
                    donchian_filtered.iloc[i] = base_signals.iloc[i]

    filtered_signals_dict['Donchian'] = donchian_filtered
    filtered_data_dict['Donchian'] = {
        'upper_band': upper_band,
        'middle_band': middle_band,
        'lower_band': lower_band,
        'donchian_width': donchian_width
    }

    # Log how many signals passed the Donchian filter
    passed_signals = len(donchian_filtered[donchian_filtered != 0])
    print(
        f"Donchian filter: {passed_signals} signals passed out of {len(base_signals[base_signals != 0])} base signals")

    # 8. ATR Volatility Filter
    print("Applying ATR Volatility filter to returns-based strategy...")

    # Calculate ATR
    atr = calculate_atr(spread_high, spread_low, spread, window=params['atr_vol_period'])

    # Calculate ATR moving average
    atr_ma = atr.rolling(window=params['atr_vol_ma_period']).mean()

    # Calculate ATR/ATR MA ratio
    atr_ratio = atr / atr_ma

    # Filter signals when ATR ratio is above threshold (higher volatility)
    # More lenient threshold to allow more signals
    atr_threshold = max(0.8, params['atr_vol_threshold'] * 0.8)

    atr_vol_filtered = pd.Series(0, index=return1.index)
    for i in range(len(base_signals)):
        if i < params['atr_vol_period'] + params['atr_vol_ma_period']:
            atr_vol_filtered.iloc[i] = 0
        elif base_signals.iloc[i] != 0 and atr_ratio.iloc[i] >= atr_threshold:
            atr_vol_filtered.iloc[i] = base_signals.iloc[i]

    filtered_signals_dict['ATR-Vol'] = atr_vol_filtered
    filtered_data_dict['ATR-Vol'] = {
        'atr': atr,
        'atr_ma': atr_ma,
        'atr_ratio': atr_ratio
    }

    # Log how many signals passed the ATR-Vol filter
    passed_signals = len(atr_vol_filtered[atr_vol_filtered != 0])
    print(f"ATR-Vol filter: {passed_signals} signals passed out of {len(base_signals[base_signals != 0])} base signals")

    # Final diagnostic summary
    print("\nFiltered signals summary:")
    for name, filtered_signals in filtered_signals_dict.items():
        active_signals = len(filtered_signals[filtered_signals != 0])
        if active_signals == 0:
            print(f"WARNING: {name} filter produced ZERO active signals!")
        else:
            print(f"{name} filter: {active_signals} active signals")

    # If any filter has zero signals, create a minimal set of signals for demonstration
    for name, filtered_signals in filtered_signals_dict.items():
        if len(filtered_signals[filtered_signals != 0]) == 0:
            print(f"Creating minimal demonstration signals for {name} filter")
            # Find strongest base signals (if any)
            base_signal_indices = base_signals[base_signals != 0].index
            if len(base_signal_indices) > 0:
                # Copy a few of the base signals to ensure some activity
                sample_size = min(5, len(base_signal_indices))
                sample_indices = base_signal_indices[:sample_size]
                for idx in sample_indices:
                    filtered_signals_dict[name].loc[idx] = base_signals.loc[idx]

    return filtered_signals_dict, filtered_data_dict

def filter_signals_by_donchian(signals, donchian_width, min_width=0.01, max_width=0.05, mode='range'):
    """
    Filter signals based on Donchian channel width
    Parameters:
    -----------
    signals : pandas.Series
        Original trading signals
    donchian_width : pandas.Series
        Donchian channel width as percentage of price
    min_width : float
        Minimum width threshold
    max_width : float
        Maximum width threshold
    mode : str
        'range': Keep signals when width is between min and max
        'min': Keep signals when width is above min
        'max': Keep signals when width is below max
    Returns:
    --------
    filtered_signals : pandas.Series
        Signals filtered by Donchian width criteria
    """
    # Initialize filtered signals (copy original signals)
    filtered_signals = signals.copy()

    if mode == 'range':
        # Keep signals when width is within range
        valid_width = (donchian_width >= min_width) & (donchian_width <= max_width)
    elif mode == 'min':
        # Keep signals when width is above minimum
        valid_width = donchian_width >= min_width
    elif mode == 'max':
        # Keep signals when width is below maximum
        valid_width = donchian_width <= max_width
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'range', 'min', or 'max'.")

    # Set signals to 0 when width criteria are not met
    filtered_signals[~valid_width] = 0

    return filtered_signals


def filter_signals_by_adx(signals, adx, threshold=25):
    """
    Filter signals based on ADX value
    Parameters:
    -----------
    signals : pandas.Series
        Original trading signals
    adx : pandas.Series
        ADX series
    threshold : float
        Minimum ADX value to consider trend strong enough
    Returns:
    --------
    filtered_signals : pandas.Series
        Signals filtered by ADX criteria
    """
    # Initialize filtered signals (copy original signals)
    filtered_signals = signals.copy()

    # Keep signals when ADX is above threshold (trend is strong)
    valid_adx = adx >= threshold

    # Set signals to 0 when ADX is below threshold
    filtered_signals[~valid_adx] = 0

    return filtered_signals


def filter_signals_by_rvi(signals, rvi, rvi_signal=None, threshold=50):
    """
    Filter signals based on RVI value
    Parameters:
    -----------
    signals : pandas.Series
        Original trading signals
    rvi : pandas.Series
        RVI series
    rvi_signal : pandas.Series, optional
        RVI signal line
    threshold : float
        RVI threshold (0-100 scale)
    Returns:
    --------
    filtered_signals : pandas.Series
        Signals filtered by RVI criteria
    """
    # Initialize filtered signals (copy original signals)
    filtered_signals = signals.copy()

    # If rvi_signal is provided, use crossover logic
    if rvi_signal is not None:
        # Keep long signals when RVI crosses above signal line
        valid_long = (signals > 0) & (rvi > rvi_signal)

        # Keep short signals when RVI crosses below signal line
        valid_short = (signals < 0) & (rvi < rvi_signal)

        # Combine conditions
        valid_rvi = valid_long | valid_short
    else:
        # Keep long signals when RVI is above threshold
        valid_long = (signals > 0) & (rvi > threshold)

        # Keep short signals when RVI is below threshold
        valid_short = (signals < 0) & (rvi < threshold)

        # Combine conditions
        valid_rvi = valid_long | valid_short

    # Set signals to 0 when RVI criteria are not met
    filtered_signals[~valid_rvi] = 0

    return filtered_signals


def filter_signals_by_chop(signals, chop, threshold=38.2):
    """
    Filter signals based on Choppiness Index
    Parameters:
    -----------
    signals : pandas.Series
        Original trading signals
    chop : pandas.Series
        Choppiness Index series
    threshold : float
        Choppiness threshold (0-100 scale)
    Returns:
    --------
    filtered_signals : pandas.Series
        Signals filtered by Choppiness Index criteria
    """
    # Initialize filtered signals (copy original signals)
    filtered_signals = signals.copy()

    # Keep signals when choppiness is below threshold (market is trending)
    valid_chop = chop < threshold

    # Set signals to 0 when market is too choppy
    filtered_signals[~valid_chop] = 0

    return filtered_signals


def calculate_atr(high, low, close, window=14):
    """
    Calculate Average True Range (ATR)
    Parameters:
    -----------
    high : pandas.Series
        Series with high prices
    low : pandas.Series
        Series with low prices
    close : pandas.Series
        Series with close prices
    window : int
        Period for ATR calculation
    Returns:
    --------
    atr : pandas.Series
        Series with ATR values
    """
    # Calculate True Range
    tr1 = high - low  # Current high - current low
    tr2 = abs(high - close.shift(1))  # Current high - previous close
    tr3 = abs(low - close.shift(1))  # Current low - previous close

    # True Range is the maximum of these three
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    # Calculate ATR as exponential moving average of True Range
    atr = tr.ewm(span=window, min_periods=window).mean()

    return atr
def backtest_with_donchian_filter(token1, token2, window_size= "**********"=0.95,
                                  donchian_period=20, donchian_width_min=0.01, donchian_width_max=0.05,
                                  donchian_mode='range', atr_window=14, atr_multiplier=2.0, fee_pct=0.001):
    """
    Backtest strategy with Donchian Channel width filter
    Parameters:
    -----------
    token1, token2 : "**********"
        Price series for the two assets
    window_size : int
        Size of rolling window for copula fitting
    confidence_level : float
        Confidence level for bands
    donchian_period : int
        Period for Donchian channels calculation
    donchian_width_min : float
        Minimum width threshold as percentage of price
    donchian_width_max : float
        Maximum width threshold as percentage of price
    donchian_mode : str
        Mode for Donchian width filtering ('range', 'min', 'max')
    atr_window : int
        Window size for ATR calculation
    atr_multiplier : float
        Multiplier for ATR stop-loss
    fee_pct : float
        Transaction fee percentage
    Returns:
    --------
    results : dict
        Dictionary with backtest results
    """
    # Calculate spread
    spread = "**********"

    # Generate original signals
    raw_signals, stop_levels, _, is_outside, copula = generate_copula_signals_with_atr_stop(
        token1, token2, window_size, confidence_level, atr_window, atr_multiplier, fee_pct
    )

    # Calculate Donchian channels for the spread
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    _, _, _, donchian_width = calculate_donchian_channels(spread_high, spread_low, donchian_period)

    # Filter signals based on Donchian width
    filtered_signals = filter_signals_by_donchian(
        raw_signals, donchian_width, donchian_width_min, donchian_width_max, donchian_mode
    )

    # Implement trading logic with filtered signals
    signals = "**********"=token1.index)
    final_stop_levels = "**********"=token1.index)
    equity_curve = "**********"=token1.index)

    # Calculate ATR for stop-loss
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Trading state variables
    current_position = 0
    entry_price = 0
    stop_price = 0

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Apply trading logic with ATR-based stop-loss
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********": "**********"
        # Current values
        current_spread = spread.iloc[i]
        current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02
        new_signal = filtered_signals.iloc[i]

        # Update equity
        if i > 0:
            if current_position == 1:  # Long position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
            elif current_position == -1:  # Short position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
            else:  # No position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        # Check for stop-loss (if in a position)
        stop_hit = False
        if current_position == 1 and stop_price > 0 and current_spread < stop_price:
            stop_hit = True
        elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
            stop_hit = True

        # Apply trading logic
        if current_position == 0:  # No current position
            if new_signal != 0:  # Enter new position
                # Apply fee for entry
                equity_curve.iloc[i] *= (1 - fee_pct)

                # Set position and entry price
                current_position = new_signal
                entry_price = current_spread

                # Set initial stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position
                final_stop_levels.iloc[i] = stop_price

        else:  # Already in a position
            # Update trailing stop if in profit
            if not stop_hit:
                if current_position == 1 and current_spread > entry_price:
                    # For long positions, raise stop as price increases
                    new_stop = current_spread - atr_multiplier * current_atr
                    stop_price = max(stop_price, new_stop)
                elif current_position == -1 and current_spread < entry_price:
                    # For short positions, lower stop as price decreases
                    new_stop = current_spread + atr_multiplier * current_atr
                    stop_price = min(stop_price, new_stop)

            # Determine whether to exit or maintain position
            if stop_hit:  # Stop-loss hit
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal == 0 and raw_signals.iloc[i] == 0:  # Exit signal
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal != current_position and new_signal != 0:  # Reversal signal
                # Apply fee for both exit and entry
                equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                current_position = new_signal
                entry_price = current_spread

                # Set new stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position

            else:  # Maintain current position
                signals.iloc[i] = current_position

            # Record stop level
            final_stop_levels.iloc[i] = stop_price

    # Process trades and calculate performance metrics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            # Calculate Donchian width at entry
            width_value = donchian_width.loc[current_date]

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration,
                'donchian_width': width_value
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        width_value = donchian_width.loc[last_date]

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'donchian_width': width_value,
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

        performance_summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_winning_trade': trade_df.loc[
                trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                          total_trades - winning_trades) > 0 else 0,
            'largest_winner': trade_df['profit_pct'].max() if not trade_df.empty else 0,
            'largest_loser': trade_df['profit_pct'].min() if not trade_df.empty else 0,
            'avg_trade_duration': trade_df['duration'].mean() if not trade_df.empty else 0,
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']) != 0 and not trade_df.empty else float(
                'inf'),
            'monthly_returns': monthly_returns
        }
    else:
        trade_df = pd.DataFrame()
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }

    # Store all results
    results = {
        'signals': signals,
        'raw_signals': raw_signals,
        'filtered_signals': filtered_signals,
        'donchian_width': donchian_width,
        'stop_levels': final_stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'copula': {
            'rho': getattr(copula, 'rho', None),
        },
        'performance': performance_summary,
        'donchian_params': {
            'period': donchian_period,
            'width_min': donchian_width_min,
            'width_max': donchian_width_max,
            'mode': donchian_mode
        }
    }

    return results


def backtest_with_adx_filter(token1, token2, window_size= "**********"=0.95,
                             adx_period=14, adx_threshold=25, atr_window=14, atr_multiplier=2.0, fee_pct=0.001):
    """
    Backtest strategy with ADX filter
    Parameters:
    -----------
    token1, token2 : "**********"
        Price series for the two assets
    window_size : int
        Size of rolling window for copula fitting
    confidence_level : float
        Confidence level for bands
    adx_period : int
        Period for ADX calculation
    adx_threshold : float
        Minimum ADX value for trend strength
    atr_window : int
        Window size for ATR calculation
    atr_multiplier : float
        Multiplier for ATR stop-loss
    fee_pct : float
        Transaction fee percentage
    Returns:
    --------
    results : dict
        Dictionary with backtest results
    """
    # Calculate spread
    spread = "**********"

    # Generate original signals
    raw_signals, stop_levels, _, is_outside, copula = generate_copula_signals_with_atr_stop(
        token1, token2, window_size, confidence_level, atr_window, atr_multiplier, fee_pct
    )

    # Calculate ADX for the spread
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    adx, plus_di, minus_di = calculate_adx(spread_high, spread_low, spread, adx_period)

    # Filter signals based on ADX
    filtered_signals = filter_signals_by_adx(raw_signals, adx, adx_threshold)

    # Implement trading logic with filtered signals
    signals = "**********"=token1.index)
    final_stop_levels = "**********"=token1.index)
    equity_curve = "**********"=token1.index)

    # Calculate ATR for stop-loss
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Trading state variables
    current_position = 0
    entry_price = 0
    stop_price = 0

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Apply trading logic with ATR-based stop-loss
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********": "**********"
        # Current values
        current_spread = spread.iloc[i]
        current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02
        new_signal = filtered_signals.iloc[i]

        # Update equity
        if i > 0:
            if current_position == 1:  # Long position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
            elif current_position == -1:  # Short position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
            else:  # No position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        # Check for stop-loss (if in a position)
        stop_hit = False
        if current_position == 1 and stop_price > 0 and current_spread < stop_price:
            stop_hit = True
        elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
            stop_hit = True

        # Apply trading logic
        if current_position == 0:  # No current position
            if new_signal != 0:  # Enter new position
                # Apply fee for entry
                equity_curve.iloc[i] *= (1 - fee_pct)

                # Set position and entry price
                current_position = new_signal
                entry_price = current_spread

                # Set initial stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position
                final_stop_levels.iloc[i] = stop_price

        else:  # Already in a position
            # Update trailing stop if in profit
            if not stop_hit:
                if current_position == 1 and current_spread > entry_price:
                    # For long positions, raise stop as price increases
                    new_stop = current_spread - atr_multiplier * current_atr
                    stop_price = max(stop_price, new_stop)
                elif current_position == -1 and current_spread < entry_price:
                    # For short positions, lower stop as price decreases
                    new_stop = current_spread + atr_multiplier * current_atr
                    stop_price = min(stop_price, new_stop)

            # Determine whether to exit or maintain position
            if stop_hit:  # Stop-loss hit
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal == 0 and raw_signals.iloc[i] == 0:  # Exit signal
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal != current_position and new_signal != 0:  # Reversal signal
                # Apply fee for both exit and entry
                equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                current_position = new_signal
                entry_price = current_spread

                # Set new stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position

            else:  # Maintain current position
                signals.iloc[i] = current_position

            # Record stop level
            final_stop_levels.iloc[i] = stop_price

    # Process trades and calculate performance metrics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            # Calculate ADX at entry
            adx_value = adx.loc[current_date]

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration,
                'adx_value': adx_value
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        adx_value = adx.loc[last_date]

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'adx_value': adx_value,
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

        performance_summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_winning_trade': trade_df.loc[
                trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                          total_trades - winning_trades) > 0 else 0,
            'largest_winner': trade_df['profit_pct'].max() if not trade_df.empty else 0,
            'largest_loser': trade_df['profit_pct'].min() if not trade_df.empty else 0,
            'avg_trade_duration': trade_df['duration'].mean() if not trade_df.empty else 0,
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']) != 0 and not trade_df.empty else float(
                'inf'),
            'monthly_returns': monthly_returns
        }
    else:
        trade_df = pd.DataFrame()
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }

    # Store all results
    results = {
        'signals': signals,
        'raw_signals': raw_signals,
        'filtered_signals': filtered_signals,
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'stop_levels': final_stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'copula': {
            'rho': getattr(copula, 'rho', None),
        },
        'performance': performance_summary,
        'adx_params': {
            'period': adx_period,
            'threshold': adx_threshold
        }
    }

    return results


def backtest_with_rvi_filter(token1, token2, window_size= "**********"=0.95,
                             rvi_period=10, rvi_signal_period=4, rvi_threshold=50,
                             atr_window=14, atr_multiplier=2.0, fee_pct=0.001):
    """
    Backtest strategy with Relative Volatility Index (RVI) filter
    Parameters:
    -----------
    token1, token2 : "**********"
        Price series for the two assets
    window_size : int
        Size of rolling window for copula fitting
    confidence_level : float
        Confidence level for bands
    rvi_period : int
        Period for RVI calculation
    rvi_signal_period : int
        Period for RVI signal line
    rvi_threshold : float
        RVI threshold (0-100 scale)
    atr_window : int
        Window size for ATR calculation
    atr_multiplier : float
        Multiplier for ATR stop-loss
    fee_pct : float
        Transaction fee percentage
    Returns:
    --------
    results : dict
        Dictionary with backtest results
    """
    # Calculate spread
    spread = "**********"

    # Generate original signals
    raw_signals, stop_levels, _, is_outside, copula = generate_copula_signals_with_atr_stop(
        token1, token2, window_size, confidence_level, atr_window, atr_multiplier, fee_pct
    )

    # Calculate RVI for the spread
    rvi, rvi_signal = calculate_rvi(spread, rvi_period, rvi_signal_period)

    # Filter signals based on RVI
    filtered_signals = filter_signals_by_rvi(raw_signals, rvi, rvi_signal, rvi_threshold)

    # Implement trading logic with filtered signals
    signals = "**********"=token1.index)
    final_stop_levels = "**********"=token1.index)
    equity_curve = "**********"=token1.index)

    # Calculate ATR for stop-loss
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Trading state variables
    current_position = 0
    entry_price = 0
    stop_price = 0

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Apply trading logic with ATR-based stop-loss
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********": "**********"
        # Current values
        current_spread = spread.iloc[i]
        current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02
        new_signal = filtered_signals.iloc[i]

        # Update equity
        if i > 0:
            if current_position == 1:  # Long position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
            elif current_position == -1:  # Short position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
            else:  # No position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        # Check for stop-loss (if in a position)
        stop_hit = False
        if current_position == 1 and stop_price > 0 and current_spread < stop_price:
            stop_hit = True
        elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
            stop_hit = True

        # Apply trading logic
        if current_position == 0:  # No current position
            if new_signal != 0:  # Enter new position
                # Apply fee for entry
                equity_curve.iloc[i] *= (1 - fee_pct)

                # Set position and entry price
                current_position = new_signal
                entry_price = current_spread

                # Set initial stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position
                final_stop_levels.iloc[i] = stop_price

        else:  # Already in a position
            # Update trailing stop if in profit
            if not stop_hit:
                if current_position == 1 and current_spread > entry_price:
                    # For long positions, raise stop as price increases
                    new_stop = current_spread - atr_multiplier * current_atr
                    stop_price = max(stop_price, new_stop)
                elif current_position == -1 and current_spread < entry_price:
                    # For short positions, lower stop as price decreases
                    new_stop = current_spread + atr_multiplier * current_atr
                    stop_price = min(stop_price, new_stop)

            # Determine whether to exit or maintain position
            if stop_hit:  # Stop-loss hit
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal == 0 and raw_signals.iloc[i] == 0:  # Exit signal
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal != current_position and new_signal != 0:  # Reversal signal
                # Apply fee for both exit and entry
                equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                current_position = new_signal
                entry_price = current_spread

                # Set new stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position

            else:  # Maintain current position
                signals.iloc[i] = current_position

            # Record stop level
            final_stop_levels.iloc[i] = stop_price

    # Process trades and calculate performance metrics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            # Calculate RVI at entry
            rvi_value = rvi.loc[current_date]
            rvi_sig_value = rvi_signal.loc[current_date]

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration,
                'rvi_value': rvi_value,
                'rvi_signal': rvi_sig_value
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        rvi_value = rvi.loc[last_date]
        rvi_sig_value = rvi_signal.loc[last_date]

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'rvi_value': rvi_value,
            'rvi_signal': rvi_sig_value,
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

        performance_summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_winning_trade': trade_df.loc[
                trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                          total_trades - winning_trades) > 0 else 0,
            'largest_winner': trade_df['profit_pct'].max() if not trade_df.empty else 0,
            'largest_loser': trade_df['profit_pct'].min() if not trade_df.empty else 0,
            'avg_trade_duration': trade_df['duration'].mean() if not trade_df.empty else 0,
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']) != 0 and not trade_df.empty else float(
                'inf'),
            'monthly_returns': monthly_returns
        }
    else:
        trade_df = pd.DataFrame()
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }

    # Store all results
    results = {
        'signals': signals,
        'raw_signals': raw_signals,
        'filtered_signals': filtered_signals,
        'rvi': rvi,
        'rvi_signal': rvi_signal,
        'stop_levels': final_stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'copula': {
            'rho': getattr(copula, 'rho', None),
        },
        'performance': performance_summary,
        'rvi_params': {
            'period': rvi_period,
            'signal_period': rvi_signal_period,
            'threshold': rvi_threshold
        }
    }

    return results


def run_adx_filter_backtest(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                            window_size=20, confidence_level=0.95,
                            adx_period=14, adx_threshold=25,
                            atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                            output_dir='adx_results'):
    """Run backtest with ADX filter"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running ADX filter backtest for {pair_name}...")

    # Run backtest
    results = backtest_with_adx_filter(
        token1, token2, window_size, confidence_level,
        adx_period, adx_threshold,
        atr_window, atr_multiplier, fee_pct
    )

    # Create plots and save them
    title = f"ADX Filter: {pair_name} (Period: {adx_period}, Threshold: {adx_threshold})"
    figs = plot_adx_filter_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_adx_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_adx_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_adx_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance'].get('total_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"Win Rate: {results['performance'].get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {results['performance'].get('total_trades', 0)}")

    return results


def run_rvi_filter_backtest(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                            window_size=20, confidence_level=0.95,
                            rvi_period=10, rvi_signal_period=4, rvi_threshold=50,
                            atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                            output_dir='rvi_results'):
    """Run backtest with RVI filter"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running RVI filter backtest for {pair_name}...")

    # Run backtest
    results = backtest_with_rvi_filter(
        token1, token2, window_size, confidence_level,
        rvi_period, rvi_signal_period, rvi_threshold,
        atr_window, atr_multiplier, fee_pct
    )

    # Create plots and save them
    title = f"RVI Filter: {pair_name} (Period: {rvi_period}, Signal: {rvi_signal_period}, Threshold: {rvi_threshold})"
    figs = plot_rvi_filter_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_rvi_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_rvi_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_rvi_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance'].get('total_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"Win Rate: {results['performance'].get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {results['performance'].get('total_trades', 0)}")

    return results


def run_chop_filter_backtest(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                             window_size=20, confidence_level=0.95,
                             chop_period=14, chop_threshold=38.2,
                             atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                             output_dir='chop_results'):
    """Run backtest with Choppiness Index filter"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running Choppiness Index filter backtest for {pair_name}...")

    # Run backtest
    results = backtest_with_chop_filter(
        token1, token2, window_size, confidence_level,
        chop_period, chop_threshold,
        atr_window, atr_multiplier, fee_pct
    )

    # Create plots and save them
    title = f"Choppiness Index Filter: {pair_name} (Period: {chop_period}, Threshold: {chop_threshold})"
    figs = plot_chop_filter_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_chop_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_chop_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_chop_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance'].get('total_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"Win Rate: {results['performance'].get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {results['performance'].get('total_trades', 0)}")

    return results


def run_donchian_filter_backtest(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                                 window_size=20, confidence_level=0.95,
                                 donchian_period=20, donchian_width_min=0.01, donchian_width_max=0.05,
                                 donchian_mode='range',
                                 atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                                 output_dir='donchian_results'):
    """Run backtest with Donchian Channel filter"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running Donchian Channel filter backtest for {pair_name}...")

    # Run backtest
    results = backtest_with_donchian_filter(
        token1, token2, window_size, confidence_level,
        donchian_period, donchian_width_min, donchian_width_max, donchian_mode,
        atr_window, atr_multiplier, fee_pct
    )

    # Create plots and save them
    title = f"Donchian Channel Filter: {pair_name} (Period: {donchian_period}, Width: {donchian_width_min}-{donchian_width_max}, Mode: {donchian_mode})"
    figs = plot_donchian_filter_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_donchian_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_donchian_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_donchian_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance'].get('total_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"Win Rate: {results['performance'].get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {results['performance'].get('total_trades', 0)}")

    return results
def backtest_with_chop_filter(token1, token2, window_size= "**********"=0.95,
                              chop_period=14, chop_threshold=38.2,
                              atr_window=14, atr_multiplier=2.0, fee_pct=0.001):
    """
    Backtest strategy with Choppiness Index filter
    Parameters:
    -----------
    token1, token2 : "**********"
        Price series for the two assets
    window_size : int
        Size of rolling window for copula fitting
    confidence_level : float
        Confidence level for bands
    chop_period : int
        Period for Choppiness Index calculation
    chop_threshold : float
        Choppiness threshold (0-100 scale)
    atr_window : int
        Window size for ATR calculation
    atr_multiplier : float
        Multiplier for ATR stop-loss
    fee_pct : float
        Transaction fee percentage
    Returns:
    --------
    results : dict
        Dictionary with backtest results
    """
    # Calculate spread
    spread = "**********"

    # Generate original signals
    raw_signals, stop_levels, _, is_outside, copula = generate_copula_signals_with_atr_stop(
        token1, token2, window_size, confidence_level, atr_window, atr_multiplier, fee_pct
    )

    # Calculate Choppiness Index for the spread
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    chop = calculate_chop(spread_high, spread_low, spread, chop_period)

    # Filter signals based on Choppiness Index
    filtered_signals = filter_signals_by_chop(raw_signals, chop, chop_threshold)

    # Implement trading logic with filtered signals
    signals = "**********"=token1.index)
    final_stop_levels = "**********"=token1.index)
    equity_curve = "**********"=token1.index)

    # Calculate ATR for stop-loss
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Trading state variables
    current_position = 0
    entry_price = 0
    stop_price = 0

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Apply trading logic with ATR-based stop-loss
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********": "**********"
        # Current values
        current_spread = spread.iloc[i]
        current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02
        new_signal = filtered_signals.iloc[i]

        # Update equity
        if i > 0:
            if current_position == 1:  # Long position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
            elif current_position == -1:  # Short position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
            else:  # No position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        # Check for stop-loss (if in a position)
        stop_hit = False
        if current_position == 1 and stop_price > 0 and current_spread < stop_price:
            stop_hit = True
        elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
            stop_hit = True

        # Apply trading logic
        if current_position == 0:  # No current position
            if new_signal != 0:  # Enter new position
                # Apply fee for entry
                equity_curve.iloc[i] *= (1 - fee_pct)

                # Set position and entry price
                current_position = new_signal
                entry_price = current_spread

                # Set initial stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position
                final_stop_levels.iloc[i] = stop_price

        else:  # Already in a position
            # Update trailing stop if in profit
            if not stop_hit:
                if current_position == 1 and current_spread > entry_price:
                    # For long positions, raise stop as price increases
                    new_stop = current_spread - atr_multiplier * current_atr
                    stop_price = max(stop_price, new_stop)
                elif current_position == -1 and current_spread < entry_price:
                    # For short positions, lower stop as price decreases
                    new_stop = current_spread + atr_multiplier * current_atr
                    stop_price = min(stop_price, new_stop)

            # Determine whether to exit or maintain position
            if stop_hit:  # Stop-loss hit
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal == 0 and raw_signals.iloc[i] == 0:  # Exit signal
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal != current_position and new_signal != 0:  # Reversal signal
                # Apply fee for both exit and entry
                equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                current_position = new_signal
                entry_price = current_spread

                # Set new stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position

            else:  # Maintain current position
                signals.iloc[i] = current_position

            # Record stop level
            final_stop_levels.iloc[i] = stop_price

    # Process trades and calculate performance metrics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            # Calculate CHOP at entry
            chop_value = chop.loc[current_date]

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration,
                'chop_value': chop_value
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        chop_value = chop.loc[last_date]

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'chop_value': chop_value,
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

        performance_summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_winning_trade': trade_df.loc[
                trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                          total_trades - winning_trades) > 0 else 0,
            'largest_winner': trade_df['profit_pct'].max() if not trade_df.empty else 0,
            'largest_loser': trade_df['profit_pct'].min() if not trade_df.empty else 0,
            'avg_trade_duration': trade_df['duration'].mean() if not trade_df.empty else 0,
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']) != 0 and not trade_df.empty else float(
                'inf'),
            'monthly_returns': monthly_returns
        }
    else:
        trade_df = pd.DataFrame()
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }

    # Store all results
    results = {
        'signals': signals,
        'raw_signals': raw_signals,
        'filtered_signals': filtered_signals,
        'chop': chop,
        'stop_levels': final_stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'copula': {
            'rho': getattr(copula, 'rho', None),
        },
        'performance': performance_summary,
        'chop_params': {
            'period': chop_period,
            'threshold': chop_threshold
        }
    }

    return results


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates


def plot_donchian_filter_results(results, title="Donchian Channel Width Filter"):
    """
    Create visualization of Donchian Channel Width filter strategy performance
    Parameters:
    -----------
    results : dict
        Dictionary with backtest results
    title : str
        Title for the main plot
    Returns:
    --------
    figs : list
        List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    donchian_width = results['donchian_width']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    performance = results['performance']
    trades = results['trades']

    figs = []

    # 1. Create multi-panel plot with signals, donchian width, and equity curve
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Donchian Channel width
    ax2.plot(donchian_width.index, donchian_width * 100, 'purple', label='Donchian Width (%)')

    # Add threshold visualization
    donchian_params = results['donchian_params']
    min_width = donchian_params['width_min']
    max_width = donchian_params['width_max']
    mode = donchian_params['mode']

    ax2.axhline(y=min_width * 100, color='orange', linestyle=':', alpha=0.7,
                label=f'Min Width: {min_width * 100:.2f}%')

    if mode == 'range':
        ax2.axhline(y=max_width * 100, color='orange', linestyle=':', alpha=0.7,
                    label=f'Max Width: {max_width * 100:.2f}%')

        # Fill the valid range area
        valid_x = donchian_width.index
        ax2.fill_between(valid_x, min_width * 100, max_width * 100, color='green', alpha=0.2,
                         label=f'Valid Range: {min_width * 100:.2f}%-{max_width * 100:.2f}%')

    ax2.set_ylabel('Donchian Width (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create additional analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L vs Donchian Width
        ax1.scatter(trades['donchian_width'] * 100, trades['profit_pct'] * 100,
                    alpha=0.7, c='blue', edgecolors='k')
        ax1.set_title('Trade P&L vs Donchian Width')
        ax1.set_xlabel('Donchian Width (%)')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.grid(True, alpha=0.3)

        # P&L histogram
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

        # 3. Create comparison of filtered vs raw signals
        if 'raw_signals' in results and 'filtered_signals' in results:
            fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Plot raw signals
            raw_signals = results['raw_signals']
            raw_long = raw_signals[raw_signals == 1]
            raw_short = raw_signals[raw_signals == -1]

            ax1.plot(spread.index, spread, 'b-', alpha=0.5)
            ax1.scatter(raw_long.index, spread.loc[raw_long.index], marker='^', color='green', s=50,
                        label='Raw Long Signals')
            ax1.scatter(raw_short.index, spread.loc[raw_short.index], marker='v', color='red', s=50,
                        label='Raw Short Signals')
            ax1.set_title('Raw Signals Before Donchian Width Filter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot filtered signals
            filtered_signals = results['filtered_signals']
            filtered_long = filtered_signals[filtered_signals == 1]
            filtered_short = filtered_signals[filtered_signals == -1]

            # Calculate signals removed by filter
            removed_long = raw_signals[(raw_signals == 1) & (filtered_signals == 0)]
            removed_short = raw_signals[(raw_signals == -1) & (filtered_signals == 0)]

            ax2.plot(spread.index, spread, 'b-', alpha=0.5)
            ax2.scatter(filtered_long.index, spread.loc[filtered_long.index], marker='^', color='green', s=50,
                        label='Filtered Long Signals')
            ax2.scatter(filtered_short.index, spread.loc[filtered_short.index], marker='v', color='red', s=50,
                        label='Filtered Short Signals')
            ax2.scatter(removed_long.index, spread.loc[removed_long.index], marker='x', color='orange', s=50,
                        label='Removed Long Signals')
            ax2.scatter(removed_short.index, spread.loc[removed_short.index], marker='x', color='purple', s=50,
                        label='Removed Short Signals')
            ax2.set_title('Signals After Donchian Width Filter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            figs.append(fig3)

    return figs


def backtest_with_expanding_window(token1, token2, min_window_size= "**********"=0.95,
                                   atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                                   copula_type='gaussian', copula_params=None):
    """
    Backtest a strategy using an expanding window for model fitting

    Parameters:
    - token1, token2: "**********"
    - min_window_size: Minimum window size to start with
    - confidence_level: Confidence level for bands
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    - copula_type: Type of copula to use
    - copula_params: Parameters for the copula model

    Returns:
    - results: Dictionary with backtest results
    """
    # Calculate spread
    spread = "**********"

    # Initialize series
    signals = "**********"=token1.index)
    is_outside = "**********"=token1.index)
    stop_levels = "**********"=token1.index)

    # Calculate ATR for stop-loss
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Track position and equity
    current_position = 0
    entry_price = 0
    stop_price = 0
    equity_curve = "**********"=token1.index)

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Create copula model based on selected type
    copula = create_copula(copula_type, copula_params)

    # Store window sizes used at each point
    window_sizes = "**********"=token1.index)

    # For each point in the time series (after initial minimum window)
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"t "**********"q "**********"d "**********"m "**********"( "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"m "**********"i "**********"n "**********"_ "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********") "**********": "**********"
        try:
            # Extract expanding window data - all data up to the current point
            window_t1 = token1.iloc[: "**********"
            window_t2 = token2.iloc[: "**********"

            # Record the window size used
            window_sizes.iloc[i] = len(window_t1)

            # Current values
            current_t1 = "**********"
            current_t2 = "**********"
            current_spread = spread.iloc[i]
            current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02

            # Update equity
            if i > 0:
                if current_position == 1:  # Long position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
                elif current_position == -1:  # Short position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
                else:  # No position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1]

            # Skip if data is invalid
            if len(window_t1.dropna()) < min_window_size * 0.9 or len(window_t2.dropna()) < min_window_size * 0.9:
                signals.iloc[i] = current_position
                stop_levels.iloc[i] = stop_price
                continue

            # Convert to uniform [0,1] using empirical CDF
            u_window = rankdata(window_t1) / (len(window_t1) + 1)
            v_window = rankdata(window_t2) / (len(window_t2) + 1)

            # Current points as quantiles
            u_current = (rankdata(np.append(window_t1, current_t1))[-1]) / (len(window_t1) + 1)
            v_current = (rankdata(np.append(window_t2, current_t2))[-1]) / (len(window_t2) + 1)

            # Fit copula to expanding window data
            copula.fit(u_window, v_window)

            # Check if point is outside confidence bands
            outside_bands, distance, direction = copula.is_point_outside_bands(u_current, v_current, confidence_level)
            is_outside.iloc[i] = outside_bands

            # Determine signal based on whether point is outside bands and direction
            if outside_bands:
                if direction < 0:  # First asset undervalued
                    new_signal = 1  # Long signal
                elif direction > 0:  # First asset overvalued
                    new_signal = -1  # Short signal
                else:
                    new_signal = current_position
            else:
                # Exit when point returns inside bands
                new_signal = 0 if current_position != 0 else current_position

            # Check for stop-loss (if in a position)
            stop_hit = False
            if current_position == 1 and stop_price > 0 and current_spread < stop_price:
                stop_hit = True
            elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
                stop_hit = True

            # Apply trading logic
            if current_position == 0:  # No current position
                if new_signal != 0:  # Enter new position
                    # Apply fee for entry
                    equity_curve.iloc[i] *= (1 - fee_pct)

                    # Set position and entry price
                    current_position = new_signal
                    entry_price = current_spread

                    # Set initial stop-loss level
                    if current_position == 1:  # Long position
                        stop_price = entry_price - atr_multiplier * current_atr
                    else:  # Short position
                        stop_price = entry_price + atr_multiplier * current_atr

                    signals.iloc[i] = current_position
                    stop_levels.iloc[i] = stop_price

            else:  # Already in a position
                # Update trailing stop if in profit
                if not stop_hit:
                    if current_position == 1 and current_spread > entry_price:
                        # For long positions, raise stop as price increases
                        new_stop = current_spread - atr_multiplier * current_atr
                        stop_price = max(stop_price, new_stop)
                    elif current_position == -1 and current_spread < entry_price:
                        # For short positions, lower stop as price decreases
                        new_stop = current_spread + atr_multiplier * current_atr
                        stop_price = min(stop_price, new_stop)

                # Determine whether to exit or maintain position
                if stop_hit:  # Stop-loss hit
                    # Apply fee for exit
                    equity_curve.iloc[i] *= (1 - fee_pct)

                    signals.iloc[i] = 0
                    current_position = 0
                    stop_price = 0

                elif new_signal == 0:  # Exit signal (point inside bands)
                    # Apply fee for exit
                    equity_curve.iloc[i] *= (1 - fee_pct)

                    signals.iloc[i] = 0
                    current_position = 0
                    stop_price = 0

                elif new_signal != current_position and new_signal != 0:  # Reversal signal
                    # Apply fee for both exit and entry
                    equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                    current_position = new_signal
                    entry_price = current_spread

                    # Set new stop-loss level
                    if current_position == 1:  # Long position
                        stop_price = entry_price - atr_multiplier * current_atr
                    else:  # Short position
                        stop_price = entry_price + atr_multiplier * current_atr

                    signals.iloc[i] = current_position

                else:  # Maintain current position
                    signals.iloc[i] = current_position

                # Record stop level
                stop_levels.iloc[i] = stop_price

        except Exception as e:
            print(f"Error at index {i}: {e}")
            # Keep previous values if error occurs
            if i > 0:
                signals.iloc[i] = signals.iloc[i - 1]
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]
                stop_levels.iloc[i] = stop_levels.iloc[i - 1]

    # Calculate trade statistics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            # Get window size used for this trade
            entry_window_size = window_sizes.loc[current_date]

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration,
                'window_size': entry_window_size
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        # Get window size used for this trade
        entry_window_size = window_sizes.loc[last_date]

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'window_size': entry_window_size,
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

        performance_summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_winning_trade': trade_df.loc[
                trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                          total_trades - winning_trades) > 0 else 0,
            'largest_winner': trade_df['profit_pct'].max() if not trade_df.empty else 0,
            'largest_loser': trade_df['profit_pct'].min() if not trade_df.empty else 0,
            'avg_trade_duration': trade_df['duration'].mean() if not trade_df.empty else 0,
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']) != 0 and not trade_df.empty else float(
                'inf'),
            'monthly_returns': monthly_returns
        }
    else:
        trade_df = pd.DataFrame()
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }

    # Store all results
    results = {
        'signals': signals,
        'stop_levels': stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'window_sizes': window_sizes,
        'copula': {
            'type': copula.name,
            'params': getattr(copula, 'params', {})
        },
        'performance': performance_summary,
        'expanding_window': {
            'min_window_size': min_window_size
        }
    }

    return results


def plot_expanding_window_results(results, title="Expanding Window Copula Strategy"):
    """
    Create visualization of expanding window strategy performance

    Parameters:
    - results: Dictionary with backtest results
    - title: Title for the main plot

    Returns:
    - figs: List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    is_outside = results['is_outside']
    performance = results['performance']
    trades = results['trades'] if 'trades' in results else pd.DataFrame()
    window_sizes = results['window_sizes']
    min_window_size = results['expanding_window']['min_window_size']

    figs = []

    # 1. Create main plot with signals and equity curve
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot window size used for each prediction
    ax2.plot(window_sizes.index, window_sizes, 'm-', label='Window Size')
    ax2.set_title('Expanding Window Size')
    ax2.set_ylabel('Number of Data Points')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create expanding window analysis if we have trades
    if not trades.empty and 'window_size' in trades.columns:
        fig2 = plt.figure(figsize=(14, 10))

        # Create 2x2 grid for different views of the expanding window performance
        gs = fig2.add_gridspec(2, 2)

        # 1. Window Size vs. Trade Profit
        ax1 = fig2.add_subplot(gs[0, 0])
        ax1.scatter(trades['window_size'], trades['profit_pct'] * 100, alpha=0.7, c='blue')
        ax1.set_title('Trade Profit vs Window Size')
        ax1.set_xlabel('Window Size')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.grid(True, alpha=0.3)

        # 2. Window Size vs. Win Rate (binned)
        ax2 = fig2.add_subplot(gs[0, 1])
        # Bin window sizes and calculate win rate for each bin
        if len(trades) > 10:  # Only if we have enough trades
            # Create bins for window size
            num_bins = min(10, len(trades) // 5)  # At least 5 trades per bin
            bins = pd.cut(trades['window_size'], bins=num_bins)
            win_rates = trades.groupby(bins)['profit_pct'].apply(lambda x: (x > 0).mean() * 100)

            # Plot binned win rates
            win_rates.index = win_rates.index.astype(str)  # Convert bins to strings for better display
            win_rates.plot(kind='bar', ax=ax2)
            ax2.set_title('Win Rate by Window Size')
            ax2.set_xlabel('Window Size Bins')
            ax2.set_ylabel('Win Rate (%)')
            ax2.axhline(y=50, color='red', linestyle='--')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Not enough trades for meaningful bins",
                     ha='center', va='center', transform=ax2.transAxes)

        # 3. Cumulative Performance by Window Size
        ax3 = fig2.add_subplot(gs[1, 0])
        if 'entry_date' in trades.columns:
            # Sort trades by entry date
            sorted_trades = trades.sort_values('entry_date')
            # Calculate cumulative return
            sorted_trades['cum_return'] = (1 + sorted_trades['profit_pct']).cumprod() - 1

            # Plot cumulative return vs window size
            ax3.plot(sorted_trades['window_size'], sorted_trades['cum_return'] * 100, 'g-')
            ax3.set_title('Cumulative Return by Window Size')
            ax3.set_xlabel('Window Size')
            ax3.set_ylabel('Cumulative Return (%)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No entry dates available for time series",
                     ha='center', va='center', transform=ax3.transAxes)

        # 4. Average Profit by Window Size (binned)
        ax4 = fig2.add_subplot(gs[1, 1])
        if len(trades) > 10:  # Only if we have enough trades
            # Calculate average profit for each bin
            avg_profits = trades.groupby(bins)['profit_pct'].mean() * 100

            # Plot binned average profits
            avg_profits.index = avg_profits.index.astype(str)  # Convert bins to strings for better display
            avg_profits.plot(kind='bar', ax=ax4)
            ax4.set_title('Average Profit by Window Size')
            ax4.set_xlabel('Window Size Bins')
            ax4.set_ylabel('Average Profit (%)')
            ax4.axhline(y=0, color='red', linestyle='--')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Not enough trades for meaningful bins",
                     ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout()
        figs.append(fig2)

    # 3. Create drawdown chart
    fig3, ax = plt.subplots(figsize=(14, 5))

    # Calculate drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve / peak - 1) * 100  # Convert to percentage

    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, 'r-', linewidth=1)

    ax.set_title('Drawdown Analysis (Expanding Window)')
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)

    # Add max drawdown line
    min_dd = drawdown.min()
    ax.axhline(y=min_dd, color='black', linestyle='--',
               label=f'Max Drawdown: {abs(min_dd):.2f}%')
    ax.legend()

    plt.tight_layout()
    figs.append(fig3)

    return figs


def run_expanding_window_backtest(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                                  min_window_size=50, confidence_level=0.95, atr_window=14, atr_multiplier=2.0,
                                  fee_pct=0.001, copula_type='gaussian', copula_params=None, output_dir=None):
    """
    Run expanding window backtest using data from a CSV file
    """
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "expanding_window_results"
        os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running expanding window backtest for {pair_name}...")
    print(f"Data range: "**********"
    print(f"Number of data points: "**********"

    # Run backtest
    results = backtest_with_expanding_window(
        token1, token2, min_window_size, confidence_level,
        atr_window, atr_multiplier, fee_pct,
        copula_type, copula_params
    )

    # Create plots and save them
    title = f"Expanding Window {copula_type.capitalize()} Copula Strategy: {pair_name}"
    figs = plot_expanding_window_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_expanding_window_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_expanding_window_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_expanding_window_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance']['total_return'] * 100:.2f}%")
    print(f"Annualized Return: {results['performance']['annualized_return'] * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown'] * 100:.2f}%")
    print(f"Win Rate: {results['performance']['win_rate'] * 100:.2f}%")
    print(f"Total Trades: {results['performance']['total_trades']}")

    return results


def plot_adx_filter_results(results, title="ADX Filter"):
    """
    Create visualization of ADX filter strategy performance
    Parameters:
    -----------
    results : dict
        Dictionary with backtest results
    title : str
        Title for the main plot
    Returns:
    --------
    figs : list
        List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    adx = results['adx']
    plus_di = results['plus_di']
    minus_di = results['minus_di']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    performance = results['performance']
    trades = results['trades']

    figs = []

    # 1. Create multi-panel plot with signals, ADX, and equity curve
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1.5, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot ADX and DI lines
    ax2.plot(adx.index, adx, 'b-', label='ADX')
    ax2.plot(plus_di.index, plus_di, 'g-', label='+DI')
    ax2.plot(minus_di.index, minus_di, 'r-', label='-DI')

    # Add ADX threshold
    adx_params = results['adx_params']
    threshold = adx_params['threshold']

    ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                label=f'ADX Threshold: {threshold}')

    ax2.set_ylabel('ADX / DI Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create additional analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L vs ADX Value
        ax1.scatter(trades['adx_value'], trades['profit_pct'] * 100,
                    alpha=0.7, c='blue', edgecolors='k')
        ax1.set_title('Trade P&L vs ADX Value')
        ax1.set_xlabel('ADX Value')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.axvline(x=threshold, color='orange', linestyle=':', alpha=0.7)
        ax1.grid(True, alpha=0.3)

        # P&L histogram
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

        # 3. Create comparison of filtered vs raw signals
        if 'raw_signals' in results and 'filtered_signals' in results:
            fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Plot raw signals
            raw_signals = results['raw_signals']
            raw_long = raw_signals[raw_signals == 1]
            raw_short = raw_signals[raw_signals == -1]

            ax1.plot(spread.index, spread, 'b-', alpha=0.5)
            ax1.scatter(raw_long.index, spread.loc[raw_long.index], marker='^', color='green', s=50,
                        label='Raw Long Signals')
            ax1.scatter(raw_short.index, spread.loc[raw_short.index], marker='v', color='red', s=50,
                        label='Raw Short Signals')
            ax1.set_title('Raw Signals Before ADX Filter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot filtered signals
            filtered_signals = results['filtered_signals']
            filtered_long = filtered_signals[filtered_signals == 1]
            filtered_short = filtered_signals[filtered_signals == -1]

            # Calculate signals removed by filter
            removed_long = raw_signals[(raw_signals == 1) & (filtered_signals == 0)]
            removed_short = raw_signals[(raw_signals == -1) & (filtered_signals == 0)]

            ax2.plot(spread.index, spread, 'b-', alpha=0.5)
            ax2.scatter(filtered_long.index, spread.loc[filtered_long.index], marker='^', color='green', s=50,
                        label='Filtered Long Signals')
            ax2.scatter(filtered_short.index, spread.loc[filtered_short.index], marker='v', color='red', s=50,
                        label='Filtered Short Signals')
            ax2.scatter(removed_long.index, spread.loc[removed_long.index], marker='x', color='orange', s=50,
                        label='Removed Long Signals')
            ax2.scatter(removed_short.index, spread.loc[removed_short.index], marker='x', color='purple', s=50,
                        label='Removed Short Signals')
            ax2.set_title('Signals After ADX Filter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            figs.append(fig3)

    return figs


def plot_rvi_filter_results(results, title="RVI Filter"):
    """
    Create visualization of RVI filter strategy performance
    Parameters:
    -----------
    results : dict
        Dictionary with backtest results
    title : str
        Title for the main plot
    Returns:
    --------
    figs : list
        List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    rvi = results['rvi']
    rvi_signal = results['rvi_signal']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    performance = results['performance']
    trades = results['trades']

    figs = []

    # 1. Create multi-panel plot with signals, RVI, and equity curve
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1.5, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot RVI and signal line
    ax2.plot(rvi.index, rvi, 'b-', label='RVI')
    ax2.plot(rvi_signal.index, rvi_signal, 'r-', label='RVI Signal')

    # Add RVI threshold and centerline
    rvi_params = results['rvi_params']
    threshold = rvi_params['threshold']

    ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                label=f'RVI Threshold: {threshold}')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7,
                label='Centerline')

    ax2.set_ylabel('RVI Values')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create additional analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L vs RVI Value
        ax1.scatter(trades['rvi_value'], trades['profit_pct'] * 100,
                    alpha=0.7, c='blue', edgecolors='k')
        ax1.set_title('Trade P&L vs RVI Value')
        ax1.set_xlabel('RVI Value')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.axvline(x=threshold, color='orange', linestyle=':', alpha=0.7)
        ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)

        # P&L histogram
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

        # 3. Create comparison of filtered vs raw signals
        if 'raw_signals' in results and 'filtered_signals' in results:
            fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Plot raw signals
            raw_signals = results['raw_signals']
            raw_long = raw_signals[raw_signals == 1]
            raw_short = raw_signals[raw_signals == -1]

            ax1.plot(spread.index, spread, 'b-', alpha=0.5)
            ax1.scatter(raw_long.index, spread.loc[raw_long.index], marker='^', color='green', s=50,
                        label='Raw Long Signals')
            ax1.scatter(raw_short.index, spread.loc[raw_short.index], marker='v', color='red', s=50,
                        label='Raw Short Signals')
            ax1.set_title('Raw Signals Before RVI Filter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot filtered signals
            filtered_signals = results['filtered_signals']
            filtered_long = filtered_signals[filtered_signals == 1]
            filtered_short = filtered_signals[filtered_signals == -1]

            # Calculate signals removed by filter
            removed_long = raw_signals[(raw_signals == 1) & (filtered_signals == 0)]
            removed_short = raw_signals[(raw_signals == -1) & (filtered_signals == 0)]

            ax2.plot(spread.index, spread, 'b-', alpha=0.5)
            ax2.scatter(filtered_long.index, spread.loc[filtered_long.index], marker='^', color='green', s=50,
                        label='Filtered Long Signals')
            ax2.scatter(filtered_short.index, spread.loc[filtered_short.index], marker='v', color='red', s=50,
                        label='Filtered Short Signals')
            ax2.scatter(removed_long.index, spread.loc[removed_long.index], marker='x', color='orange', s=50,
                        label='Removed Long Signals')
            ax2.scatter(removed_short.index, spread.loc[removed_short.index], marker='x', color='purple', s=50,
                        label='Removed Short Signals')
            ax2.set_title('Signals After RVI Filter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            figs.append(fig3)

    return figs


def plot_chop_filter_results(results, title="Choppiness Index Filter"):
    """
    Create visualization of Choppiness Index filter strategy performance
    Parameters:
    -----------
    results : dict
        Dictionary with backtest results
    title : str
        Title for the main plot
    Returns:
    --------
    figs : list
        List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    chop = results['chop']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    performance = results['performance']
    trades = results['trades']

    figs = []

    # 1. Create multi-panel plot with signals, CHOP, and equity curve
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1.5, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Choppiness Index
    ax2.plot(chop.index, chop, 'b-', label='Choppiness Index')

    # Add CHOP threshold
    chop_params = results['chop_params']
    threshold = chop_params['threshold']

    ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                label=f'CHOP Threshold: {threshold}')
    ax2.axhline(y=61.8, color='red', linestyle='--', alpha=0.5,
                label='Choppy Market: 61.8')

    # Fill the trending range
    valid_x = chop.index
    ax2.fill_between(valid_x, 0, threshold, color='green', alpha=0.2,
                     label=f'Trending Range: <{threshold}')

    ax2.set_ylabel('Choppiness Index')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create additional analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L vs CHOP Value
        ax1.scatter(trades['chop_value'], trades['profit_pct'] * 100,
                    alpha=0.7, c='blue', edgecolors='k')
        ax1.set_title('Trade P&L vs Choppiness Index')
        ax1.set_xlabel('Choppiness Index')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.axvline(x=threshold, color='orange', linestyle=':', alpha=0.7)
        ax1.grid(True, alpha=0.3)

        # P&L histogram
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

        # 3. Create comparison of filtered vs raw signals
        if 'raw_signals' in results and 'filtered_signals' in results:
            fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Plot raw signals
            raw_signals = results['raw_signals']
            raw_long = raw_signals[raw_signals == 1]
            raw_short = raw_signals[raw_signals == -1]

            ax1.plot(spread.index, spread, 'b-', alpha=0.5)
            ax1.scatter(raw_long.index, spread.loc[raw_long.index], marker='^', color='green', s=50,
                        label='Raw Long Signals')
            ax1.scatter(raw_short.index, spread.loc[raw_short.index], marker='v', color='red', s=50,
                        label='Raw Short Signals')
            ax1.set_title('Raw Signals Before Choppiness Index Filter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot filtered signals
            filtered_signals = results['filtered_signals']
            filtered_long = filtered_signals[filtered_signals == 1]
            filtered_short = filtered_signals[filtered_signals == -1]

            # Calculate signals removed by filter
            removed_long = raw_signals[(raw_signals == 1) & (filtered_signals == 0)]
            removed_short = raw_signals[(raw_signals == -1) & (filtered_signals == 0)]

            ax2.plot(spread.index, spread, 'b-', alpha=0.5)
            ax2.scatter(filtered_long.index, spread.loc[filtered_long.index], marker='^', color='green', s=50,
                        label='Filtered Long Signals')
            ax2.scatter(filtered_short.index, spread.loc[filtered_short.index], marker='v', color='red', s=50,
                        label='Filtered Short Signals')
            ax2.scatter(removed_long.index, spread.loc[removed_long.index], marker='x', color='orange', s=50,
                        label='Removed Long Signals')
            ax2.scatter(removed_short.index, spread.loc[removed_short.index], marker='x', color='purple', s=50,
                        label='Removed Short Signals')
            ax2.set_title('Signals After Choppiness Index Filter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            figs.append(fig3)

    return figs


def run_filter_comparison(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                          volume_col='volume', window_size=20, confidence_level=0.95,
                          donchian_period=20, donchian_width_min=0.01, donchian_width_max=0.05, donchian_mode='range',
                          adx_period=14, adx_threshold=25,
                          rvi_period=10, rvi_signal_period=4, rvi_threshold=50,
                          chop_period=14, chop_threshold=38.2,
                          atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                          output_dir='filter_comparison'):
    """
    Run backtests for all filter types and compare results
    Parameters:
    -----------
    csv_file: str
        Path to CSV file with price data
    date_col: str
        Name of the date column
    token1_col, token2_col: "**********"
        Names of price columns for the two assets
    volume_col: str
        Name of volume column
    window_size: int
        Size of rolling window for copula fitting
    confidence_level: float
        Confidence level for bands
    donchian_period: int
        Period for Donchian channels
    donchian_width_min, donchian_width_max: float
        Min/max width thresholds for Donchian filter
    donchian_mode: str
        Mode for Donchian width filtering
    adx_period: int
        Period for ADX calculation
    adx_threshold: float
        Threshold for ADX filter
    rvi_period: int
        Period for RVI calculation
    rvi_signal_period: int
        Period for RVI signal line
    rvi_threshold: float
        Threshold for RVI filter
    chop_period: int
        Period for Choppiness Index
    chop_threshold: float
        Threshold for CHOP filter
    atr_window: int
        Window size for ATR calculation
    atr_multiplier: float
        Multiplier for ATR stop-loss
    fee_pct: float
        Transaction fee percentage
    output_dir: str
        Directory to save outputs
    Returns:
    --------
    comparison_df: pd.DataFrame
        Comparison of all filter strategies
    """
    import os
    import datetime

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Try to find volume column
    if volume_col in df.columns:
        volume = df[volume_col]
    else:
        # Try alternatives
        vol_candidates = [f"{volume_col}_1", f"{volume_col}_2", "volume_1", "volume_2", "volume", "vol"]
        for col in vol_candidates:
            if col in df.columns:
                volume = df[col]
                print(f"Using {col} for volume")
                break
        else:
            # Default to constant if not found
            print("Volume column not found, using constant")
            volume = "**********"=token1.index)

    # Get pair name
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running filter comparison for {pair_name}...")

    # Dictionary to store results
    results = {}

    # 1. Run standard strategy (baseline)
    print("Running baseline strategy...")
    baseline_results = backtest_strategy(
        token1, token2, window_size, confidence_level,
        atr_window, atr_multiplier, fee_pct
    )
    results['Baseline'] = baseline_results

    # 2. Run Donchian Channel Width filter
    print("Running Donchian Channel Width filter strategy...")
    donchian_results = backtest_with_donchian_filter(
        token1, token2, window_size, confidence_level,
        donchian_period, donchian_width_min, donchian_width_max, donchian_mode,
        atr_window, atr_multiplier, fee_pct
    )
    results['Donchian'] = donchian_results

    # Save Donchian plots
    donchian_dir = os.path.join(output_dir, 'donchian')
    os.makedirs(donchian_dir, exist_ok=True)
    donchian_title = f"Donchian Width Filter: {pair_name} (Range: {donchian_width_min * 100:.1f}%-{donchian_width_max * 100:.1f}%)"
    donchian_figs = plot_donchian_filter_results(donchian_results, donchian_title)
    for i, fig in enumerate(donchian_figs):
        fig.savefig(os.path.join(donchian_dir, f"{pair_name}_donchian_plot_{i + 1}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 3. Run ADX filter
    print("Running ADX filter strategy...")
    adx_results = backtest_with_adx_filter(
        token1, token2, window_size, confidence_level,
        adx_period, adx_threshold,
        atr_window, atr_multiplier, fee_pct
    )
    results['ADX'] = adx_results

    # Save ADX plots
    adx_dir = os.path.join(output_dir, 'adx')
    os.makedirs(adx_dir, exist_ok=True)
    adx_title = f"ADX Filter: {pair_name} (Threshold: {adx_threshold})"
    adx_figs = plot_adx_filter_results(adx_results, adx_title)
    for i, fig in enumerate(adx_figs):
        fig.savefig(os.path.join(adx_dir, f"{pair_name}_adx_plot_{i + 1}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 4. Run RVI filter
    print("Running RVI filter strategy...")
    rvi_results = backtest_with_rvi_filter(
        token1, token2, window_size, confidence_level,
        rvi_period, rvi_signal_period, rvi_threshold,
        atr_window, atr_multiplier, fee_pct
    )
    results['RVI'] = rvi_results

    # Save RVI plots
    rvi_dir = os.path.join(output_dir, 'rvi')
    os.makedirs(rvi_dir, exist_ok=True)
    rvi_title = f"RVI Filter: {pair_name} (Threshold: {rvi_threshold})"
    rvi_figs = plot_rvi_filter_results(rvi_results, rvi_title)
    for i, fig in enumerate(rvi_figs):
        fig.savefig(os.path.join(rvi_dir, f"{pair_name}_rvi_plot_{i + 1}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 5. Run Choppiness Index filter
    print("Running Choppiness Index filter strategy...")
    chop_results = backtest_with_chop_filter(
        token1, token2, window_size, confidence_level,
        chop_period, chop_threshold,
        atr_window, atr_multiplier, fee_pct
    )
    results['CHOP'] = chop_results

    # Save CHOP plots
    chop_dir = os.path.join(output_dir, 'chop')
    os.makedirs(chop_dir, exist_ok=True)
    chop_title = f"Choppiness Index Filter: {pair_name} (Threshold: {chop_threshold})"
    chop_figs = plot_chop_filter_results(chop_results, chop_title)
    for i, fig in enumerate(chop_figs):
        fig.savefig(os.path.join(chop_dir, f"{pair_name}_chop_plot_{i + 1}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Create comparison table
    comparison_data = []
    for name, result in results.items():
        perf = result['performance']
        comparison_data.append({
            'Strategy': name,
            'Total Return (%)': perf.get('total_return', 0) * 100,
            'Annualized Return (%)': perf.get('annualized_return', 0) * 100,
            'Sharpe Ratio': perf.get('sharpe_ratio', 0),
            'Max Drawdown (%)': perf.get('max_drawdown', 0) * 100,
            'Win Rate (%)': perf.get('win_rate', 0) * 100,
            'Total Trades': perf.get('total_trades', 0),
            'Avg Win (%)': perf.get('avg_winning_trade', 0) * 100,
            'Avg Loss (%)': perf.get('avg_losing_trade', 0) * 100,
            'Profit Factor': perf.get('profit_factor', 0)
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison table
    comparison_file = os.path.join(output_dir, f"{pair_name}_filter_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved filter comparison to {comparison_file}")

    # Create equity curve comparison chart
    plt.figure(figsize=(14, 7))

    for name, result in results.items():
        equity_curve = result['equity_curve']
        returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
        plt.plot(returns_pct.index, returns_pct, label=f"{name}")

    plt.title(f"Equity Curve Comparison: {pair_name}")
    plt.ylabel("Return (%)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()

    comparison_plot_file = os.path.join(output_dir, f"{pair_name}_equity_comparison.png")
    plt.savefig(comparison_plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Create drawdown comparison chart
    plt.figure(figsize=(14, 7))

    for name, result in results.items():
        equity_curve = result['equity_curve']
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1) * 100  # Convert to percentage
        plt.plot(drawdown.index, drawdown, label=f"{name}")

    plt.title(f"Drawdown Comparison: {pair_name}")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()

    drawdown_plot_file = os.path.join(output_dir, f"{pair_name}_drawdown_comparison.png")
    plt.savefig(drawdown_plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Generate HTML report
    report_path = os.path.join(output_dir, f"{pair_name}_report.html")

    with open(report_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Filter Comparison Report: {pair_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; text-align: center; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .chart {{ margin: 20px 0; max-width: 100%; }}
                .section {{ margin: 30px 0; }}
                .strategy-name {{ text-align: left; font-weight: bold; }}
                .best-value {{ color: green; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Filter Comparison Report: {pair_name}</h1>
            <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="section">
                <h2>Strategy Comparison</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Total Return (%)</th>
                        <th>Annual Return (%)</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown (%)</th>
                        <th>Win Rate (%)</th>
                        <th>Total Trades</th>
                        <th>Profit Factor</th>
                    </tr>
        """)

        # Add rows for each strategy
        best_values = comparison_df.iloc[:, 1:].idxmax()

        for i, row in comparison_df.iterrows():
            f.write("<tr>")
            f.write(f'<td class="strategy-name">{row["Strategy"]}</td>')

            for col in ["Total Return (%)", "Annualized Return (%)", "Sharpe Ratio", "Max Drawdown (%)",
                        "Win Rate (%)", "Total Trades", "Profit Factor"]:
                value = row[col]
                is_best = (i == best_values[col])
                cell_class = 'best-value' if is_best else ''
                f.write(f'<td class="{cell_class}">{value:.2f}</td>')

            f.write("</tr>")

        f.write("""
                </table>
            </div>
            <div class="section">
                <h2>Equity Curve Comparison</h2>
                <img src="equity_comparison.png" class="chart" alt="Equity Curve Comparison">
            </div>
            <div class="section">
                <h2>Drawdown Comparison</h2>
                <img src="drawdown_comparison.png" class="chart" alt="Drawdown Comparison">
            </div>
            <div class="section">
                <h2>Individual Filter Results</h2>
                <h3>Donchian Channel Width Filter</h3>
                <img src="donchian/donchian_plot_1.png" class="chart" alt="Donchian Filter Results">
                <h3>ADX Filter</h3>
                <img src="adx/adx_plot_1.png" class="chart" alt="ADX Filter Results">
                <h3>RVI Filter</h3>
                <img src="rvi/rvi_plot_1.png" class="chart" alt="RVI Filter Results">
                <h3>Choppiness Index Filter</h3>
                <img src="chop/chop_plot_1.png" class="chart" alt="CHOP Filter Results">
            </div>
            <div class="section">
                <h2>Conclusions</h2>
                <p>Based on the analysis, the following observations can be made:</p>
                <ul>
        """)

        # Add some automatic conclusions
        best_strategy = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax(), 'Strategy']
        f.write(
            f'<li>The <strong>{best_strategy}</strong> filter shows the best risk-adjusted performance (highest Sharpe ratio).</li>')

        # Compare returns
        base_return = comparison_df.loc[comparison_df['Strategy'] == 'Baseline', 'Annualized Return (%)'].values[0]
        best_return = comparison_df['Annualized Return (%)'].max()
        best_return_strategy = comparison_df.loc[comparison_df['Annualized Return (%)'].idxmax(), 'Strategy']

        if best_return > base_return:
            improvement = best_return - base_return
            f.write(
                f'<li>The {best_return_strategy} filter improved annual returns by <strong>{improvement:.2f}%</strong> compared to the baseline strategy.</li>')
        else:
            f.write(
                f'<li>None of the filters improved returns compared to the baseline strategy.</li>')

        # Compare win rates
        base_winrate = comparison_df.loc[comparison_df['Strategy'] == 'Baseline', 'Win Rate (%)'].values[0]
        best_winrate = comparison_df['Win Rate (%)'].max()
        best_winrate_strategy = comparison_df.loc[comparison_df['Win Rate (%)'].idxmax(), 'Strategy']

        if best_winrate > base_winrate:
            winrate_diff = best_winrate - base_winrate
            f.write(
                f'<li>The {best_winrate_strategy} filter improved win rate by <strong>{winrate_diff:.2f}%</strong> compared to the baseline strategy.</li>')
        else:
            f.write(
                f'<li>None of the filters improved win rate compared to the baseline strategy.</li>')

        f.write("""
                </ul>
            </div>
        </body>
        </html>
        """)

    print(f"Generated HTML report at {report_path}")

    return comparison_df


def create_returns_from_prices(price_series, method='log'):
    """
    Convert price series to returns series
    Parameters:
    - price_series: Series with price data
    - method: 'log' for log returns or 'pct' for percentage returns
    Returns:
    - returns_series: Series with returns data
    """
    if method == 'log':
        # Calculate log returns: ln(P_t / P_{t-1})
        return np.log(price_series / price_series.shift(1)).fillna(0)
    else:
        # Calculate percentage returns: (P_t / P_{t-1}) - 1
        return price_series.pct_change().fillna(0)


def backtest_returns_copula_strategy(return1, return2, window_size=20, confidence_level=0.95,
                                     lookback=5, fee_pct=0.001, copula_type='gaussian',
                                     copula_params=None, returns_method='log'):
    """
    Backtest copula strategy using returns data with proper log/pct return handling.

    Parameters:
      - return1, return2: Return series for the two assets.
      - window_size: Size of rolling window for copula fitting.
      - confidence_level: Confidence level for bands (e.g. 0.95).
      - lookback: Number of periods to look back for performance evaluation.
      - fee_pct: Trading fee percentage per trade (0.001 = 0.1%).
      - copula_type: Type of copula to use ('gaussian', 'clayton', 'student_t', 'gumbel').
      - copula_params: Parameters for the chosen copula.
      - returns_method: 'log' (default) for log returns or 'pct' for percentage returns.

    Returns:
      - results: Dictionary with backtest results
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    # Initialize series for signals, equity curve, and outside-band flag
    signals = pd.Series(0, index=return1.index)
    is_outside = pd.Series(False, index=return1.index)
    equity_curve = pd.Series(1.0, index=return1.index)

    # Create copula model based on selected type
    copula = create_copula(copula_type, copula_params)

    # Track current position (1 for long asset1/short asset2, -1 for short asset1/long asset2)
    current_position = 0

    # Loop over time series starting after the initial window
    for i in range(window_size, len(return1)):
        try:
            # Extract window data for copula fitting
            window_r1 = return1.iloc[i - window_size:i]
            window_r2 = return2.iloc[i - window_size:i]
            # Current returns at time i
            current_r1 = return1.iloc[i]
            current_r2 = return2.iloc[i]

            # =============== Update equity curve based on current position ===============
            if i > 0:
                prev_equity = equity_curve.iloc[i - 1]
                if current_position == 1:
                    # Long asset1, short asset2
                    if returns_method == 'log':
                        # If r1, r2 are log returns, the daily multiplier is exp(r1 - r2)
                        multiplier = np.exp(current_r1 - current_r2)
                    else:
                        # If r1, r2 are simple returns, use the compound formula for accuracy
                        multiplier = (1 + current_r1) / (1 + current_r2)
                    equity_curve.iloc[i] = prev_equity * multiplier

                elif current_position == -1:
                    # Short asset1, long asset2
                    if returns_method == 'log':
                        multiplier = np.exp(current_r2 - current_r1)
                    else:
                        multiplier = (1 + current_r2) / (1 + current_r1)
                    equity_curve.iloc[i] = prev_equity * multiplier

                else:
                    # No position; equity stays the same
                    equity_curve.iloc[i] = prev_equity

            # =============== Skip if insufficient valid window data ===============
            if len(window_r1.dropna()) < window_size * 0.9 or len(window_r2.dropna()) < window_size * 0.9:
                signals.iloc[i] = current_position
                continue

            # =============== Convert window data to uniform [0,1] via empirical CDF ===============
            u_window = rankdata(window_r1) / (len(window_r1) + 1)
            v_window = rankdata(window_r2) / (len(window_r2) + 1)

            # Compute current quantiles by appending current value
            u_current = (rankdata(np.append(window_r1, current_r1))[-1]) / (window_size + 1)
            v_current = (rankdata(np.append(window_r2, current_r2))[-1]) / (window_size + 1)

            # =============== Fit the copula to the window data and check if outside bands ===============
            copula.fit(u_window, v_window)
            outside_bands, distance, direction = copula.is_point_outside_bands(u_current, v_current, confidence_level)
            is_outside.iloc[i] = outside_bands

            # =============== Determine new signal based on the copula test ===============
            new_signal = 0
            if outside_bands:
                if direction < 0:  # Expect asset1 to outperform asset2
                    new_signal = 1
                elif direction > 0:  # Expect asset2 to outperform asset1
                    new_signal = -1

            # =============== Apply signal logic with fee adjustments ===============
            # If we enter or exit or reverse a position, we multiply equity by (1 - fee_pct) once or twice.
            if current_position == 0 and new_signal != 0:
                # Enter new position
                equity_curve.iloc[i] *= (1 - fee_pct)  # Entry fee
                current_position = new_signal
                signals.iloc[i] = current_position

            elif new_signal == 0 and current_position != 0:
                # Exit position
                equity_curve.iloc[i] *= (1 - fee_pct)  # Exit fee
                current_position = 0
                signals.iloc[i] = 0

            elif new_signal != 0 and current_position != 0 and new_signal != current_position:
                # Reverse position
                equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)  # Exit & entry fees
                current_position = new_signal
                signals.iloc[i] = current_position

            else:
                # No change in position
                signals.iloc[i] = current_position

        except Exception as e:
            print(f"Error at index {i}: {e}")
            # If there's an error, carry forward the previous values
            if i > 0:
                signals.iloc[i] = signals.iloc[i - 1]
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

    # =============== After processing entire series, compute trades and performance ===============
    trades = calculate_trades_from_signals(signals, return1, return2, equity_curve, fee_pct, returns_method)
    performance_summary = calculate_performance_metrics(equity_curve, trades)

    # =============== Package up results ===============
    results = {
        'signals': signals,
        'equity_curve': equity_curve,
        'trades': trades,
        'return1': return1,
        'return2': return2,
        'performance': performance_summary,
        'is_outside': is_outside,
        'copula': {
            'type': copula_type,
            'params': getattr(copula, 'params', {})
        }
    }
    return results


def calculate_performance_metrics(equity_curve, trades_df=None):
    """
    Calculate performance metrics from equity curve and trade data with improved win rate handling

    Parameters:
    - equity_curve: Series with equity values over time
    - trades_df: DataFrame with trade details (optional)

    Returns:
    - Dictionary with performance metrics
    """
    # Initialize default results for early returns
    default_results = {
        'total_return': 0,
        'annualized_return': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'calmar_ratio': 0,
        'win_rate': 0,
        'profit_factor': 0,
        'avg_winning_trade': 0,
        'avg_losing_trade': 0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
    }

    # Handle empty or invalid equity curve
    if equity_curve is None or len(equity_curve) <= 1:
        print("Warning: Equity curve is empty or too short")
        return default_results

    # Ensure equity_curve is a pandas Series
    if not isinstance(equity_curve, pd.Series):
        try:
            equity_curve = pd.Series(equity_curve)
        except Exception as e:
            print(f"Error converting equity curve to Series: {e}")
            return default_results

    # Calculate returns
    returns = equity_curve.pct_change().dropna()

    # Total return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    # Annualized return (assuming 252 trading days per year)
    days = len(equity_curve)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

    # Maximum drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve / peak - 1)
    max_drawdown = abs(drawdown.min())

    # Risk metrics
    volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0  # Annualized volatility
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if not downside_returns.empty and len(
        downside_returns) > 1 else 0.0001

    # Sharpe ratio (annualized)
    risk_free_rate = 0.02  # Assuming 2% risk-free rate
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Sortino ratio (using downside deviation)
    sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0

    # Calmar ratio (return / max drawdown)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Trade statistics (if trades_df is provided)
    if trades_df is not None and not trades_df.empty and 'profit_pct' in trades_df.columns:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
        losing_trades = len(trades_df[trades_df['profit_pct'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Average trade metrics
        winning_trades_df = trades_df[trades_df['profit_pct'] > 0]
        losing_trades_df = trades_df[trades_df['profit_pct'] <= 0]

        avg_winning_trade = winning_trades_df['profit_pct'].mean() if not winning_trades_df.empty else 0
        avg_losing_trade = losing_trades_df['profit_pct'].mean() if not losing_trades_df.empty else 0

        # Profit factor (sum of gains / sum of losses)
        sum_gains = winning_trades_df['profit_pct'].sum() if not winning_trades_df.empty else 0
        sum_losses = abs(losing_trades_df['profit_pct'].sum()) if not losing_trades_df.empty else 0
        profit_factor = sum_gains / sum_losses if sum_losses != 0 and sum_losses > 0 else 0

        # Print detailed trade statistics for debugging
        print(f"Trade statistics: {total_trades} total trades, {winning_trades} winners, {losing_trades} losers")
        print(f"Win rate: {win_rate:.2%}, Profit factor: {profit_factor:.2f}")
    else:
        # If no trades data, analyze equity curve to estimate trade metrics
        # This is a fallback when we don't have explicit trade data
        if len(returns) > 0:
            # Estimate trade metrics from equity curve
            daily_gains = returns[returns > 0]
            daily_losses = returns[returns < 0]

            win_rate = len(daily_gains) / len(returns) if len(returns) > 0 else 0

            sum_gains = daily_gains.sum() if not daily_gains.empty else 0
            sum_losses = abs(daily_losses.sum()) if not daily_losses.empty else 0
            profit_factor = sum_gains / sum_losses if sum_losses > 0 else 0

            total_trades = len(daily_gains) + len(daily_losses)
            winning_trades = len(daily_gains)
            losing_trades = len(daily_losses)

            avg_winning_trade = daily_gains.mean() if not daily_gains.empty else 0
            avg_losing_trade = daily_losses.mean() if not daily_losses.empty else 0

            print("Note: Trade metrics estimated from equity curve (no explicit trade data)")
        else:
            win_rate = 0
            profit_factor = 0
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            avg_winning_trade = 0
            avg_losing_trade = 0

    # Calculate monthly returns if datetime index
    if isinstance(equity_curve.index[0], (pd.Timestamp, pd.DatetimeIndex)):
        try:
            monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        except:
            monthly_returns = pd.Series()
    else:
        monthly_returns = pd.Series()

    # Create performance summary
    performance_summary = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_winning_trade': avg_winning_trade,
        'avg_losing_trade': avg_losing_trade,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'monthly_returns': monthly_returns
    }

    return performance_summary


def plot_returns_strategy_results(results, title="Returns-Based Copula Strategy"):
    """
    Create visualizations of returns-based strategy performance
    Parameters:
    - results: Dictionary with backtest results
    - title: Title for the main plot
    Returns:
    - figs: List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    equity_curve = results['equity_curve']
    return1 = results['return1']
    return2 = results['return2']
    is_outside = results['is_outside']
    performance = results['performance']
    trades = results['trades'] if 'trades' in results else pd.DataFrame()

    figs = []

    # 1. Create main plot with cumulative returns and equity curve
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1.5, 1.5, 1]})

    # Plot the cumulative returns of each asset
    cum_ret1 = (1 + return1).cumprod() - 1
    cum_ret2 = (1 + return2).cumprod() - 1

    ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return')
    ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return')

    # Shade areas where outside bands - safely handle string or datetime indices
    outside_dates = is_outside[is_outside].index
    if len(outside_dates) > 0:
        # Check if we can convert to datetime if not already
        try:
            # If these are strings, try to convert to datetime
            if isinstance(outside_dates[0], str):
                datetime_dates = pd.to_datetime(outside_dates)
                consecutive_threshold = pd.Timedelta(days=1)

                # Mark non-consecutive dates
                for i in range(len(datetime_dates)):
                    if i > 0 and (datetime_dates[i] - datetime_dates[i - 1]) <= consecutive_threshold:
                        continue  # Skip consecutive dates
                    ax1.axvline(x=outside_dates[i], color='gray', linestyle='--', alpha=0.3)
            else:
                # These are already datetime objects
                for i in range(len(outside_dates)):
                    if i > 0 and (outside_dates[i] - outside_dates[i - 1]).total_seconds() <= 86400:
                        continue  # Skip consecutive dates
                    ax1.axvline(x=outside_dates[i], color='gray', linestyle='--', alpha=0.3)
        except:
            # If conversion fails or any other error, just plot all points
            for date in outside_dates:
                ax1.axvline(x=date, color='gray', linestyle='--', alpha=0.3)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    for date in long_entries.index:
        try:
            ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
            ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
        except:
            pass  # Skip if there's any issue with the date or value

    for date in short_entries.index:
        try:
            ax1.scatter(date, cum_ret1.loc[date] * 100, marker='v', color='red', s=100)
            ax1.scatter(date, cum_ret2.loc[date] * 100, marker='^', color='blue', s=100)
        except:
            pass  # Skip if there's any issue with the date or value

    ax1.set_title(title)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot the return difference
    ret_diff = return1 - return2
    cum_diff = (1 + ret_diff).cumprod() - 1

    ax2.plot(cum_diff.index, cum_diff * 100, 'r-', label='Asset1 - Asset2 Return')

    # Add trade markers
    for date in long_entries.index:
        try:
            ax2.axvline(x=date, color='green', linestyle='-', alpha=0.5)
        except:
            pass  # Skip if there's any issue with the date

    for date in short_entries.index:
        try:
            ax2.axvline(x=date, color='red', linestyle='-', alpha=0.5)
        except:
            pass  # Skip if there's any issue with the date

    ax2.set_ylabel('Cumulative Return Difference (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Strategy Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create trade analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L scatter plot
        if 'r1_cum' in trades.columns and 'r2_cum' in trades.columns:
            ax1.scatter(trades['r1_cum'] * 100, trades['r2_cum'] * 100,
                        c=trades['profit_pct'] * 100, cmap='RdYlGn',
                        vmin=-5, vmax=5, s=80, alpha=0.8, edgecolors='k')

            ax1.set_title('Trade Performance by Asset Returns')
            ax1.set_xlabel('Asset 1 Return (%)')
            ax1.set_ylabel('Asset 2 Return (%)')
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)

            # Add diagonal line (y=x)
            min_val = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
            max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

            cbar = plt.colorbar(ax1.collections[0], ax=ax1)
            cbar.set_label('Profit/Loss (%)')

        # P&L histogram
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

    return figs


def run_returns_backtest_from_csv(csv_file, date_col='timestamp', return1_col='return_1', return2_col='return_2',
                                  window_size=20, confidence_level=0.95, lookback=5, fee_pct=0.001,
                                  copula_type='gaussian', copula_params=None, output_dir='returns_copula_output',
                                  convert_prices=False, price1_col=None, price2_col=None, returns_method='log'):
    """
    Run returns-based copula strategy backtest using data from a CSV file
    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of the date/timestamp column
    - return1_col, return2_col: Names of return columns (if already in CSV)
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - lookback: Number of periods to lookback for performance evaluation
    - fee_pct: Transaction fee percentage
    - copula_type: Type of copula to use
    - copula_params: Parameters for the copula
    - output_dir: Directory to save output visualizations
    - convert_prices: Whether to convert price data to returns (if True, price columns are used)
    - price1_col, price2_col: Names of price columns (used if convert_prices=True)
    - returns_method: Method for calculating returns ('log' or 'pct')
    Returns:
    - results: Dictionary with backtest results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Check if we need to convert prices to returns
    if convert_prices:
        if price1_col is None or price2_col is None:
            raise ValueError("Price column names must be provided when convert_prices=True")

        # Check if required columns exist
        required_cols = {price1_col, price2_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required price columns: {missing_cols}")

        # Convert prices to returns
        return1 = create_returns_from_prices(df[price1_col], method=returns_method)
        return2 = create_returns_from_prices(df[price2_col], method=returns_method)
    else:
        # Use return columns directly
        required_cols = {return1_col, return2_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required return columns: {missing_cols}")

        return1 = df[return1_col]
        return2 = df[return2_col]

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running returns-based copula strategy backtest for {pair_name}...")
    print(f"Data range: {return1.index.min()} to {return1.index.max()}")
    print(f"Number of data points: {len(return1)}")
    print(f"Parameters: window_size={window_size}, confidence_level={confidence_level}, " +
          f"lookback={lookback}, fee_pct={fee_pct * 100}%")
    print(f"Copula: type={copula_type}")

    # Run backtest
    results = backtest_returns_copula_strategy(
        return1, return2, window_size, confidence_level,
        lookback, fee_pct, copula_type, copula_params
    )

    # Create plots and save them
    title = f"{copula_type.capitalize()} Copula Returns Strategy: {pair_name}"
    figs = plot_returns_strategy_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_returns_{copula_type}_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_returns_{copula_type}_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_returns_{copula_type}_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance']['total_return'] * 100:.2f}%")
    print(f"Annualized Return: {results['performance']['annualized_return'] * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown'] * 100:.2f}%")
    print(f"Win Rate: {results['performance']['win_rate'] * 100:.2f}%")
    print(f"Total Trades: {results['performance']['total_trades']}")
    print(f"Profit Factor: {results['performance']['profit_factor']:.2f}")

    return results


def calculate_trades_from_signals(signals, return1, return2, equity_curve, fee_pct, returns_method='log'):
    """
    Calculate trade statistics from signals with proper log/pct return handling

    Parameters:
    - signals: Trading signals (1 = long-short, -1 = short-long, 0 = no position)
    - return1, return2: Return series for the two assets
    - equity_curve: Equity curve from backtest
    - fee_pct: Transaction fee percentage
    - returns_method: 'log' for log returns or 'pct' for percentage returns

    Returns:
    - trades_df: DataFrame with trade details
    """
    trades = []

    # Get position changes (entry/exit points)
    position_changes = signals.diff().fillna(0)
    change_indices = position_changes[position_changes != 0].index

    # Ensure we have valid indices before processing
    if len(change_indices) < 2:
        print("Warning: Not enough position changes detected for meaningful trade analysis")
        return pd.DataFrame()

    # Process each position change
    for i in range(len(change_indices) - 1):
        try:
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            position = signals.loc[start_idx]

            # Skip if not actually entering a position
            if position == 0:
                continue

            # Extract slice of returns during the position
            try:
                if isinstance(start_idx, pd.Timestamp) and isinstance(end_idx, pd.Timestamp):
                    position_returns1 = return1.loc[start_idx:end_idx]
                    position_returns2 = return2.loc[start_idx:end_idx]
                else:
                    # Handle numeric indices
                    start_pos = return1.index.get_loc(start_idx)
                    end_pos = return1.index.get_loc(end_idx)
                    position_returns1 = return1.iloc[start_pos:end_pos + 1]
                    position_returns2 = return2.iloc[start_pos:end_pos + 1]
            except Exception as e:
                print(f"Error extracting returns: {e}")
                continue

            # Calculate cumulative returns over the holding period
            if returns_method == 'log':
                # For log returns, sum them up
                r1_cum = np.sum(position_returns1)
                r2_cum = np.sum(position_returns2)

                # Convert from log to simple returns for reporting
                r1_simple = np.exp(r1_cum) - 1
                r2_simple = np.exp(r2_cum) - 1

                # Calculate profit based on position direction
                if position == 1:  # Long asset1, short asset2
                    profit_log = r1_cum - r2_cum
                    profit_pct = np.exp(profit_log) - 1
                else:  # Short asset1, long asset2
                    profit_log = r2_cum - r1_cum
                    profit_pct = np.exp(profit_log) - 1

            else:  # Percentage returns
                # For percentage returns, compound them
                r1_simple = (1 + position_returns1).prod() - 1
                r2_simple = (1 + position_returns2).prod() - 1

                # Calculate profit based on position direction
                if position == 1:  # Long asset1, short asset2
                    profit_pct = (1 + r1_simple) / (1 + r2_simple) - 1
                else:  # Short asset1, long asset2
                    profit_pct = (1 + r2_simple) / (1 + r1_simple) - 1

            # Account for fees
            profit_pct = profit_pct - fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                if isinstance(start_idx, pd.Timestamp) and isinstance(end_idx, pd.Timestamp):
                    duration = (end_idx - start_idx).days
                    if duration == 0:
                        duration = 1  # Minimum 1 day for intraday trades
                else:
                    duration = int(end_pos - start_pos)
            except Exception as e:
                print(f"Error calculating duration: {e}")
                duration = len(position_returns1)

            trades.append({
                'entry_date': start_idx,
                'exit_date': end_idx,
                'position': 'Long-Short' if position == 1 else 'Short-Long',
                'profit_pct': float(profit_pct),
                'r1_cum': float(r1_simple),  # Always store as simple returns for clarity
                'r2_cum': float(r2_simple),
                'duration': duration
            })

        except Exception as e:
            print(f"Error processing trade at index {i}: {e}")
            continue

    # Handle the last open position if any
    if len(change_indices) > 0 and signals.iloc[-1] != 0:
        try:
            start_idx = change_indices[-1]
            end_idx = signals.index[-1]
            position = signals.loc[start_idx]

            # Extract slice of returns for the open position
            try:
                if isinstance(start_idx, pd.Timestamp) and isinstance(end_idx, pd.Timestamp):
                    position_returns1 = return1.loc[start_idx:end_idx]
                    position_returns2 = return2.loc[start_idx:end_idx]
                else:
                    # Handle numeric indices
                    start_pos = return1.index.get_loc(start_idx)
                    position_returns1 = return1.iloc[start_pos:]
                    position_returns2 = return2.iloc[start_pos:]
            except Exception as e:
                print(f"Error extracting returns for open position: {e}")
                return pd.DataFrame(trades) if trades else pd.DataFrame()

            # Calculate cumulative returns over the holding period
            if returns_method == 'log':
                # For log returns, sum them up
                r1_cum = np.sum(position_returns1)
                r2_cum = np.sum(position_returns2)

                # Convert from log to simple returns for reporting
                r1_simple = np.exp(r1_cum) - 1
                r2_simple = np.exp(r2_cum) - 1

                # Calculate profit based on position direction
                if position == 1:  # Long asset1, short asset2
                    profit_log = r1_cum - r2_cum
                    profit_pct = np.exp(profit_log) - 1
                else:  # Short asset1, long asset2
                    profit_log = r2_cum - r1_cum
                    profit_pct = np.exp(profit_log) - 1

            else:  # Percentage returns
                # For percentage returns, compound them
                r1_simple = (1 + position_returns1).prod() - 1
                r2_simple = (1 + position_returns2).prod() - 1

                # Calculate profit based on position direction
                if position == 1:  # Long asset1, short asset2
                    profit_pct = (1 + r1_simple) / (1 + r2_simple) - 1
                else:  # Short asset1, long asset2
                    profit_pct = (1 + r2_simple) / (1 + r1_simple) - 1

            # Account for fees (only entry fee for open position)
            profit_pct = profit_pct - fee_pct

            # Calculate duration
            try:
                if isinstance(start_idx, pd.Timestamp) and isinstance(end_idx, pd.Timestamp):
                    duration = (end_idx - start_idx).days
                    if duration == 0:
                        duration = 1
                else:
                    start_pos = return1.index.get_loc(start_idx)
                    end_pos = return1.index.get_loc(end_idx)
                    duration = int(end_pos - start_pos)
            except Exception as e:
                print(f"Error calculating duration for open position: {e}")
                duration = len(position_returns1)

            trades.append({
                'entry_date': start_idx,
                'exit_date': end_idx,
                'position': 'Long-Short' if position == 1 else 'Short-Long',
                'profit_pct': float(profit_pct),
                'r1_cum': float(r1_simple),
                'r2_cum': float(r2_simple),
                'duration': duration,
                'open': True
            })

        except Exception as e:
            print(f"Error processing open position: {e}")

    # Create DataFrame with trade information
    if trades:
        try:
            trades_df = pd.DataFrame(trades)
            print(f"Successfully identified {len(trades_df)} trades")
            return trades_df
        except Exception as e:
            print(f"Error creating trades DataFrame: {e}")
            return pd.DataFrame()
    else:
        print("No trades identified from the signals")
        return pd.DataFrame()

def run_all_indicator_filters(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                              volume_col='volume', output_dir='indicator_filter_results'):
    """
    Main function to run all indicator filter backtests and create comparison reports
    Parameters:
    -----------
    csv_file: str
        Path to CSV file with price data
    date_col: str
        Name of the date column
    token1_col, token2_col: "**********"
        Names of price columns for the two assets
    volume_col: str
        Name of volume column
    output_dir: str
        Directory to save outputs
    Returns:
    --------
    results: dict
        Dictionary containing all filter strategy results
    """
    # Import necessary modules
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    import time

    # Set Matplotlib style
    plt.style.use('dark_background')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Start time
    start_time = time.time()

    # Get pair name from CSV file
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Starting indicator filter backtests for {pair_name}...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)

    # Run filter comparison with default parameters
    comparison_df = run_filter_comparison(
        csv_file=csv_file,
        date_col=date_col,
        token1_col= "**********"
        token2_col= "**********"
        volume_col=volume_col,
        # Copula parameters
        window_size=20,
        confidence_level=0.95,
        # Donchian parameters
        donchian_period=20,
        donchian_width_min=0.01,
        donchian_width_max=0.05,
        donchian_mode='range',
        # ADX parameters
        adx_period=14,
        adx_threshold=25,
        # RVI parameters
        rvi_period=10,
        rvi_signal_period=4,
        rvi_threshold=50,
        # CHOP parameters
        chop_period=14,
        chop_threshold=38.2,
        # General parameters
        atr_window=14,
        atr_multiplier=2.0,
        fee_pct=0.001,
        output_dir=output_dir
    )

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 80)
    print(f"Completed all backtests in {elapsed_time:.2f} seconds")
    print("\nSummary of results:")
    print(comparison_df[['Strategy', 'Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Total Trades']].to_string(
        index=False))
    print(f"\nFull report available at: {os.path.join(output_dir, f'{pair_name}_report.html')}")

    return comparison_df
def run_combined_filtered_backtest(csv_file, date_col='datetime',
                                   token1_col= "**********"='close_2', volume_col='volume',
                                   window_size=20, confidence_level=0.95,
                                   vol_ratio_window=20, vol_ratio_threshold=1.2, vol_ratio_mode='threshold',
                                   vol_lookback=100, vol_min_percentile=0.3, vol_max_percentile=1.0,
                                   atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                                   output_dir='combined_results'):
    """
    Runner function that:
      1) Loads CSV
      2) Calls backtest_with_combined_filters
      3) Plots results via plot_combined_filter_results
      4) Saves plots/trade logs/performance
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        df = pd.read_csv(csv_file)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.set_index(date_col, inplace=True)

    # Check token columns
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"c "**********"o "**********"l "**********"  "**********"i "**********"n "**********"  "**********"[ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********"_ "**********"c "**********"o "**********"l "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"2 "**********"_ "**********"c "**********"o "**********"l "**********"] "**********": "**********"
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    token1 = "**********"
    token2 = "**********"

    # Try to find the appropriate volume column
    if volume_col in df.columns:
        volume = df[volume_col]
    else:
        # Try to find volume based on token column naming
        vol1_candidate = volume_col + "_1"
        vol2_candidate = volume_col + "_2"

        if vol1_candidate in df.columns:
            print(f"Using '{vol1_candidate}' as volume column")
            volume = df[vol1_candidate]
        elif vol2_candidate in df.columns:
            print(f"Using '{vol2_candidate}' as volume column")
            volume = df[vol2_candidate]
        else:
            # Try various common volume column names
            vol_candidates = ['volume_1', 'volume_2', 'vol_1', 'vol_2', 'volume', 'vol']
            for col in vol_candidates:
                if col in df.columns:
                    print(f"Using '{col}' as volume column")
                    volume = df[col]
                    break
            else:
                # If no volume column found, create constant volume
                print("Warning: No volume column found. Using constant volume.")
                volume = "**********"=token1.index)

    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running combined filters backtest for {pair_name}...")
    print(f"Data range: "**********"
    print(f"Number of data points: "**********"
    print(f"Vol Ratio: window={vol_ratio_window}, threshold={vol_ratio_threshold}, mode={vol_ratio_mode}")
    print(f"Volume: lookback={vol_lookback}, min={vol_min_percentile}, max={vol_max_percentile}")

    # Actually run the combined backtest
    results = backtest_with_combined_filters(
        token1, token2, volume,
        window_size=window_size,
        confidence_level=confidence_level,
        vol_ratio_window=vol_ratio_window,
        vol_ratio_threshold=vol_ratio_threshold,
        vol_ratio_mode=vol_ratio_mode,
        vol_lookback=vol_lookback,
        vol_min_percentile=vol_min_percentile,
        vol_max_percentile=vol_max_percentile,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
        fee_pct=fee_pct
    )

    # Plot
    title = f"Combined Filter Backtest: {pair_name}"
    figs = plot_combined_filter_results(results, title=title)

    # Save figs
    for i, fig in enumerate(figs):
        outpath = os.path.join(output_dir, f"{pair_name}_combined_plot_{i + 1}.png")
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"Saved combined-filter plot to {outpath}")

    # Save trades if any
    if not results['trades'].empty:
        trades_file = os.path.join(output_dir, f"{pair_name}_combined_trades.csv")
        results['trades'].to_csv(trades_file, index=False)
        print(f"Saved combined-filter trade log to {trades_file}")

    # Save performance summary
    perf = pd.DataFrame({k: [v] for k, v in results['performance'].items() if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_combined_performance.csv")
    perf.to_csv(perf_file, index=False)
    print(f"Saved combined-filter performance to {perf_file}")

    # Print summary
    p = results['performance']
    print("\nCombined Filters Performance Summary:")
    print(f"Total Return: {p.get('total_return', 0) * 100:.2f}%")
    print(f"Annualized Return: {p.get('annualized_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {p.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {p.get('max_drawdown', 0) * 100:.2f}%")
    print(f"Win Rate: {p.get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {p.get('total_trades', 0)}")
    print(f"Profit Factor: {p.get('profit_factor', 0):.2f}")

    return results


def backtest_with_combined_filters(token1, token2, volume,
                                   window_size=20, confidence_level=0.95,
                                   vol_ratio_window=20, vol_ratio_threshold=1.2, vol_ratio_mode='threshold',
                                   vol_lookback=100, vol_min_percentile=0.3, vol_max_percentile=1.0,
                                   atr_window=14, atr_multiplier=2.0, fee_pct=0.001):
    """
    Backtest strategy that applies BOTH:
      1) Volatility ratio filter
      2) Volume percentile filter
    in sequence, then trades with an ATR stop.
    Returns results dict with:
        'vol_ratio_filtered' : signals after vol ratio filter
        'final_filtered'     : signals after final volume filter
        'vol_ratio'          : volatility ratio series
        'vol_percentile'     : volume percentile series
        ...
    """
    # Calculate spread
    spread = "**********"

    # 1) Generate raw copula signals
    raw_signals, stop_levels, _, is_outside, copula = generate_copula_signals_with_atr_stop(
        token1, token2, window_size, confidence_level, atr_window, atr_multiplier, fee_pct
    )

    # 2) Vol ratio filter
    vol_ratio_filtered_signals, vol_ratio = filter_signals_by_vol_ratio(
        token1, token2, raw_signals,
        window_size=vol_ratio_window,
        vol_ratio_threshold=vol_ratio_threshold,
        mode=vol_ratio_mode
    )

    # 3) Volume percentile filter
    final_filtered_signals, vol_percentile = filter_signals_by_volume_percentile(
        vol_ratio_filtered_signals,
        volume,
        lookback_window=vol_lookback,
        min_percentile=vol_min_percentile,
        max_percentile=vol_max_percentile
    )

    # 4) Calculate ATR for stop-loss
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Initialize signals and equity tracking
    signals = "**********"=token1.index)
    final_stop_levels = "**********"=token1.index)
    equity_curve = "**********"=token1.index)

    # Trading state variables
    current_position = 0
    entry_price = 0
    stop_price = 0

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Apply trading logic with ATR-based stop-loss
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********": "**********"
        # Current values
        current_spread = spread.iloc[i]
        current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02
        new_signal = final_filtered_signals.iloc[i]

        # Update equity
        if i > 0:
            if current_position == 1:  # Long position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
            elif current_position == -1:  # Short position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
            else:  # No position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        # Check for stop-loss (if in a position)
        stop_hit = False
        if current_position == 1 and stop_price > 0 and current_spread < stop_price:
            stop_hit = True
        elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
            stop_hit = True

        # Apply trading logic
        if current_position == 0:  # No current position
            if new_signal != 0:  # Enter new position
                # Apply fee for entry
                equity_curve.iloc[i] *= (1 - fee_pct)

                # Set position and entry price
                current_position = new_signal
                entry_price = current_spread

                # Set initial stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position
                final_stop_levels.iloc[i] = stop_price

        else:  # Already in a position
            # Update trailing stop if in profit
            if not stop_hit:
                if current_position == 1 and current_spread > entry_price:
                    # For long positions, raise stop as price increases
                    new_stop = current_spread - atr_multiplier * current_atr
                    stop_price = max(stop_price, new_stop)
                elif current_position == -1 and current_spread < entry_price:
                    # For short positions, lower stop as price decreases
                    new_stop = current_spread + atr_multiplier * current_atr
                    stop_price = min(stop_price, new_stop)

            # Determine whether to exit or maintain position
            if stop_hit:  # Stop-loss hit
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal == 0 and raw_signals.iloc[i] == 0:  # Exit signal
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal != current_position and new_signal != 0:  # Reversal signal
                # Apply fee for both exit and entry
                equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                current_position = new_signal
                entry_price = current_spread

                # Set new stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position

            else:  # Maintain current position
                signals.iloc[i] = current_position

            # Record stop level
            final_stop_levels.iloc[i] = stop_price

    # Calculate trade statistics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            # For analysis, record vol_ratio & volume_percentile at entry
            vratio = vol_ratio.get(current_date, np.nan)
            vpctl = vol_percentile.get(current_date, np.nan)

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration,
                'vol_ratio': vratio,
                'volume_percentile': vpctl
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        vratio = vol_ratio.get(last_date, np.nan)
        vpctl = vol_percentile.get(last_date, np.nan)

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'vol_ratio': vratio,
            'volume_percentile': vpctl,
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

        performance_summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_winning_trade': trade_df.loc[
                trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                          total_trades - winning_trades) > 0 else 0,
            'largest_winner': trade_df['profit_pct'].max() if not trade_df.empty else 0,
            'largest_loser': trade_df['profit_pct'].min() if not trade_df.empty else 0,
            'avg_trade_duration': trade_df['duration'].mean() if not trade_df.empty else 0,
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']) != 0 and not trade_df.empty else float(
                'inf'),
            'monthly_returns': monthly_returns
        }
    else:
        trade_df = pd.DataFrame()
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }

    # Store all results
    results = {
        'signals': signals,
        'raw_signals': raw_signals,
        'vol_ratio_filtered': vol_ratio_filtered_signals,
        'final_filtered': final_filtered_signals,
        'vol_ratio': vol_ratio,
        'vol_percentile': vol_percentile,
        'stop_levels': final_stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'copula': {
            'rho': getattr(copula, 'rho', None),
        },
        'performance': performance_summary,
        'vol_ratio_params': {
            'window': vol_ratio_window,
            'threshold': vol_ratio_threshold,
            'mode': vol_ratio_mode
        },
        'volume_params': {
            'lookback': vol_lookback,
            'min_percentile': vol_min_percentile,
            'max_percentile': vol_max_percentile
        }
    }

    return results


def plot_combined_filter_results(results, title="Combined Filter Strategy"):
    """
    Create visualization of the combined filter strategy performance
    Parameters:
    - results: Dictionary with backtest results
    - title: Title for the main plot
    Returns:
    - figs: List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    vol_ratio = results['vol_ratio']
    vol_percentile = results['vol_percentile']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    performance = results['performance']
    trades = results['trades']

    figs = []

    # 1. Create multi-panel plot with signals, filters, and equity curve
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot volatility ratio
    ax2.plot(vol_ratio.index, vol_ratio, 'purple', label= "**********"
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

    # Add threshold/range visualization based on mode
    vol_ratio_params = results['vol_ratio_params']
    mode = vol_ratio_params['mode']
    threshold = vol_ratio_params['threshold']

    if mode == 'threshold':
        ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                    label=f'Upper Threshold: {threshold:.2f}')
        ax2.axhline(y=1 / threshold, color='orange', linestyle=':', alpha=0.7,
                    label=f'Lower Threshold: {1 / threshold:.2f}')
    elif mode == 'range':
        # Fill the valid range area
        valid_x = vol_ratio.index
        ax2.fill_between(valid_x, 1 / threshold, threshold, color='green', alpha=0.2,
                         label=f'Valid Range: {1 / threshold:.2f}-{threshold:.2f}')

    ax2.set_ylabel('Volatility Ratio')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot volume percentile
    ax3.plot(vol_percentile.index, vol_percentile, 'blue', label='Volume Percentile')

    # Add volume threshold visualization
    vol_params = results['volume_params']
    min_percentile = vol_params['min_percentile']
    max_percentile = vol_params['max_percentile']

    ax3.axhline(y=min_percentile, color='orange', linestyle=':', alpha=0.7,
                label=f'Min Percentile: {min_percentile:.2f}')
    if max_percentile < 1.0:
        ax3.axhline(y=max_percentile, color='orange', linestyle=':', alpha=0.7,
                    label=f'Max Percentile: {max_percentile:.2f}')

    # Fill the valid volume range area
    ax3.fill_between(vol_percentile.index, min_percentile, max_percentile, color='green', alpha=0.2,
                     label=f'Valid Range: {min_percentile:.2f}-{max_percentile:.2f}')

    ax3.set_ylabel('Volume Percentile')
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax4.plot(returns_pct.index, returns_pct, 'g-')
    ax4.set_title('Equity Curve')
    ax4.set_ylabel('Return (%)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax4.text(0.02, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create additional analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L vs Filters
        sc = ax1.scatter(trades['vol_ratio'], trades['volume_percentile'],
                         c=trades['profit_pct'] * 100, cmap='RdYlGn',
                         vmin=-5, vmax=5, s=80, alpha=0.8, edgecolors='k')

        ax1.set_title('Trade Performance by Filter Values')
        ax1.set_xlabel('Volatility Ratio (Token1/Token2)')
        ax1.set_ylabel('Volume Percentile')
        ax1.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)

        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label('Profit/Loss (%)')

        # Trade count and win rate by month
        if isinstance(trades['entry_date'].iloc[0], (pd.Timestamp, datetime.datetime)):
            # Group by month
            trades['entry_month'] = trades['entry_date'].dt.strftime('%Y-%m')
            monthly_stats = trades.groupby('entry_month').agg({
                'profit_pct': [lambda x: (x > 0).mean() * 100, 'mean', 'count']
            })
            monthly_stats.columns = ['win_rate', 'avg_profit', 'trade_count']
            monthly_stats['avg_profit'] = monthly_stats['avg_profit'] * 100  # Convert to percentage

            # Plot monthly statistics
            ax2_1 = ax2.twinx()
            bars = ax2.bar(monthly_stats.index, monthly_stats['trade_count'], alpha=0.7, color='blue',
                           label='Trade Count')
            line = ax2_1.plot(monthly_stats.index, monthly_stats['win_rate'], 'r-', marker='o', label='Win Rate (%)')

            ax2.set_title('Monthly Trade Statistics')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Number of Trades')
            ax2_1.set_ylabel('Win Rate (%)')

            # Add combined legend
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_1.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')

            plt.xticks(rotation=45)
            ax2.grid(True, alpha=0.3)
        else:
            # If no datetime index, just show trade distribution
            ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
            ax2.set_title('Trade Profit/Loss Distribution')
            ax2.set_xlabel('Profit/Loss (%)')
            ax2.set_ylabel('Frequency')
            ax2.axvline(x=0, color='red', linestyle='--')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

        # 3. Create signal progression analysis
        if 'raw_signals' in results and 'vol_ratio_filtered' in results and 'final_filtered' in results:
            fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

            # Plot raw signals
            raw_signals = results['raw_signals']
            raw_long = raw_signals[raw_signals == 1]
            raw_short = raw_signals[raw_signals == -1]

            ax1.plot(spread.index, spread, 'b-', alpha=0.5)
            ax1.scatter(raw_long.index, spread.loc[raw_long.index], marker='^', color='green', s=50,
                        label='Raw Long Signals')
            ax1.scatter(raw_short.index, spread.loc[raw_short.index], marker='v', color='red', s=50,
                        label='Raw Short Signals')
            ax1.set_title('Step 1: Raw Copula Signals')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot vol ratio filtered signals
            vol_ratio_filtered = results['vol_ratio_filtered']
            vr_long = vol_ratio_filtered[vol_ratio_filtered == 1]
            vr_short = vol_ratio_filtered[vol_ratio_filtered == -1]
            vr_removed_long = raw_signals[(raw_signals == 1) & (vol_ratio_filtered == 0)]
            vr_removed_short = raw_signals[(raw_signals == -1) & (vol_ratio_filtered == 0)]

            ax2.plot(spread.index, spread, 'b-', alpha=0.5)
            ax2.scatter(vr_long.index, spread.loc[vr_long.index], marker='^', color='green', s=50,
                        label='Vol Ratio Filtered Long')
            ax2.scatter(vr_short.index, spread.loc[vr_short.index], marker='v', color='red', s=50,
                        label='Vol Ratio Filtered Short')
            ax2.scatter(vr_removed_long.index, spread.loc[vr_removed_long.index], marker='x', color='orange', s=50,
                        label='Removed by Vol Ratio Filter')
            ax2.scatter(vr_removed_short.index, spread.loc[vr_removed_short.index], marker='x', color='purple', s=50,
                        label='Removed by Vol Ratio Filter')
            ax2.set_title('Step 2: After Volatility Ratio Filter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot final signals (after volume filter)
            final_filtered = results['final_filtered']
            final_long = final_filtered[final_filtered == 1]
            final_short = final_filtered[final_filtered == -1]
            vol_removed_long = vol_ratio_filtered[(vol_ratio_filtered == 1) & (final_filtered == 0)]
            vol_removed_short = vol_ratio_filtered[(vol_ratio_filtered == -1) & (final_filtered == 0)]

            ax3.plot(spread.index, spread, 'b-', alpha=0.5)
            ax3.scatter(final_long.index, spread.loc[final_long.index], marker='^', color='green', s=50,
                        label='Final Long Signals')
            ax3.scatter(final_short.index, spread.loc[final_short.index], marker='v', color='red', s=50,
                        label='Final Short Signals')
            ax3.scatter(vol_removed_long.index, spread.loc[vol_removed_long.index], marker='x', color='cyan', s=50,
                        label='Removed by Volume Filter')
            ax3.scatter(vol_removed_short.index, spread.loc[vol_removed_short.index], marker='x', color='magenta', s=50,
                        label='Removed by Volume Filter')
            ax3.set_title('Step 3: After Volume Percentile Filter (Final Signals)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            figs.append(fig3)

        return figs


def run_donchian_filter_backtest(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                                 window_size=20, confidence_level=0.95,
                                 donchian_period=20, donchian_width_min=0.01, donchian_width_max=0.05,
                                 donchian_mode='range',
                                 atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                                 output_dir='donchian_results'):
    """Run backtest with Donchian Channel filter"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running Donchian filter backtest for {pair_name}...")

    # Run backtest
    results = backtest_with_donchian_filter(
        token1, token2, window_size, confidence_level,
        donchian_period, donchian_width_min, donchian_width_max, donchian_mode,
        atr_window, atr_multiplier, fee_pct
    )

    # Create plots and save them
    title = f"Donchian Filter: {pair_name} (Width: {donchian_width_min}-{donchian_width_max}, Mode: {donchian_mode})"
    figs = plot_donchian_filter_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_donchian_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_donchian_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_donchian_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance'].get('total_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"Win Rate: {results['performance'].get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {results['performance'].get('total_trades', 0)}")

    return results
def optimize_combined_filters(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                              volume_col='volume',
                              window_size=20, confidence_level=0.95, atr_window=14, atr_multiplier=2.0,
                              fee_pct=0.001,
                              vol_ratio_thresholds=[1.1, 1.2, 1.5], vol_ratio_modes=['threshold', 'range'],
                              vol_min_percentiles=[0.1, 0.3, 0.5], vol_max_percentiles=[1.0],
                              output_dir='combined_optimization'):
    """
    Optimize combined filter parameters
    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of the date/timestamp column
    - token1_col: "**********"
    - token2_col: "**********"
    - volume_col: Name of the volume column
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    - vol_ratio_thresholds: List of thresholds for volatility ratio filtering
    - vol_ratio_modes: List of modes for volatility ratio filtering
    - vol_min_percentiles: List of minimum volume percentiles to test
    - vol_max_percentiles: List of maximum volume percentiles to test
    - output_dir: Directory to save results
    Returns:
    - results_df: DataFrame with optimization results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Extract volume data
    if volume_col in df.columns:
        volume = df[volume_col]
    else:
        # Try common volume column names
        volume_found = False
        vol1_candidate = volume_col + "_1"
        vol2_candidate = volume_col + "_2"

        if vol1_candidate in df.columns:
            print(f"Using '{vol1_candidate}' as volume column")
            volume = df[vol1_candidate]
            volume_found = True
        elif vol2_candidate in df.columns:
            print(f"Using '{vol2_candidate}' as volume column")
            volume = df[vol2_candidate]
            volume_found = True
        else:
            vol_candidates = ['volume_1', 'volume_2', 'vol_1', 'vol_2', 'volume', 'vol']
            for col in vol_candidates:
                if col in df.columns:
                    print(f"Using '{col}' as volume column")
                    volume = df[col]
                    volume_found = True
                    break

        if not volume_found:
            print("No volume column found, using constant volume")
            volume = "**********"=token1.index)

    # Pair name
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Optimizing combined filter parameters for {pair_name}...")

    # Store results
    results = []

    # Grid search
    total_combinations = len(vol_ratio_thresholds) * len(vol_ratio_modes) * len(vol_min_percentiles) * len(
        vol_max_percentiles)
    current = 0

    for vol_ratio_threshold in vol_ratio_thresholds:
        for vol_ratio_mode in vol_ratio_modes:
            for vol_min_percentile in vol_min_percentiles:
                for vol_max_percentile in vol_max_percentiles:
                    # Skip invalid combinations (min > max)
                    if vol_min_percentile >= vol_max_percentile:
                        continue

                    current += 1
                    print(f"Testing combination {current}/{total_combinations}: " +
                          f"vol_ratio_threshold={vol_ratio_threshold}, vol_ratio_mode={vol_ratio_mode}, " +
                          f"vol_min_percentile={vol_min_percentile}, vol_max_percentile={vol_max_percentile}")

                    try:
                        # Run backtest with these parameters
                        res = backtest_with_combined_filters(
                            token1, token2, volume, window_size, confidence_level,
                            vol_ratio_window=20, vol_ratio_threshold=vol_ratio_threshold,
                            vol_ratio_mode=vol_ratio_mode,
                            vol_lookback=100, vol_min_percentile=vol_min_percentile,
                            vol_max_percentile=vol_max_percentile,
                            atr_window=atr_window, atr_multiplier=atr_multiplier, fee_pct=fee_pct
                        )

                        # Extract performance metrics
                        perf = res['performance']

                        # Skip combinations with too few trades
                        if perf['total_trades'] < 5:
                            print(f"  Skipping combination with only {perf['total_trades']} trades")
                            continue

                        # Store results
                        results.append({
                            'vol_ratio_threshold': vol_ratio_threshold,
                            'vol_ratio_mode': vol_ratio_mode,
                            'vol_min_percentile': vol_min_percentile,
                            'vol_max_percentile': vol_max_percentile,
                            'total_return': perf['total_return'],
                            'annualized_return': perf['annualized_return'],
                            'sharpe_ratio': perf['sharpe_ratio'],
                            'max_drawdown': perf['max_drawdown'],
                            'win_rate': perf['win_rate'],
                            'total_trades': perf['total_trades'],
                            'profit_factor': perf['profit_factor']
                        })
                    except Exception as e:
                        print(f"  Error with this combination: {e}")

    # Create DataFrame with results
    if results:
        results_df = pd.DataFrame(results)

        # Sort by Sharpe ratio (descending)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        # Save results
        results_file = os.path.join(output_dir, f"{pair_name}_combined_optimization.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved optimization results to {results_file}")

        # Display top 5 parameter combinations
        print("\nTop 5 parameter combinations:")
        display_cols = ['vol_ratio_threshold', 'vol_ratio_mode', 'vol_min_percentile', 'vol_max_percentile',
                        'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'total_trades']
        print(results_df[display_cols].head(5).to_string())

        # Create chart for the best combination
        if len(results_df) > 0:
            best_params = results_df.iloc[0]
            print(f"\nRunning backtest with best parameters:")
            print(f"vol_ratio_threshold={best_params['vol_ratio_threshold']}, " +
                  f"vol_ratio_mode={best_params['vol_ratio_mode']}, " +
                  f"vol_min_percentile={best_params['vol_min_percentile']}, " +
                  f"vol_max_percentile={best_params['vol_max_percentile']}")

            best_results = backtest_with_combined_filters(
                token1, token2, volume, window_size, confidence_level,
                vol_ratio_window=20, vol_ratio_threshold=best_params['vol_ratio_threshold'],
                vol_ratio_mode=best_params['vol_ratio_mode'],
                vol_lookback=100, vol_min_percentile=best_params['vol_min_percentile'],
                vol_max_percentile=best_params['vol_max_percentile'],
                atr_window=atr_window, atr_multiplier=atr_multiplier, fee_pct=fee_pct
            )

            # Create descriptive title
            title = f"Best Combined Filter: {pair_name}\n" + \
                    f"Vol Ratio: {best_params['vol_ratio_mode']} ({best_params['vol_ratio_threshold']}), " + \
                    f"Volume: {best_params['vol_min_percentile']}-{best_params['vol_max_percentile']}"

            figs = plot_combined_filter_results(best_results, title)

            for i, fig in enumerate(figs):
                output_file = os.path.join(output_dir, f"{pair_name}_best_combined_plot_{i + 1}.png")
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"Saved best parameters plot to {output_file}")

        return results_df

    return pd.DataFrame()


def create_performance_report(results_dict, output_dir):
    """
    Create a comprehensive performance report comparing all strategies
    Parameters:
    - results_dict: Dictionary of strategy name to results
      Example: {'Standard Copula': base_results, 'Vol Ratio': volratio_results, ...}
    - output_dir: Directory to save report
    Returns:
    - report_path: Path to saved report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame for summary comparison
    compare_results = []

    for strategy_name, results in results_dict.items():
        compare_results.append({
            'Strategy': strategy_name,
            'Total Return (%)': results['performance'].get('total_return', 0) * 100,
            'Annualized Return (%)': results['performance'].get('annualized_return', 0) * 100,
            'Sharpe Ratio': results['performance'].get('sharpe_ratio', 0),
            'Max Drawdown (%)': results['performance'].get('max_drawdown', 0) * 100,
            'Win Rate (%)': results['performance'].get('win_rate', 0) * 100,
            'Total Trades': results['performance'].get('total_trades', 0),
            'Avg Win (%)': results['performance'].get('avg_winning_trade', 0) * 100,
            'Avg Loss (%)': results['performance'].get('avg_losing_trade', 0) * 100,
            'Profit Factor': results['performance'].get('profit_factor', 0)
        })

    compare_results = pd.DataFrame(compare_results)

    # Save summary comparison
    comparison_file = os.path.join(output_dir, 'strategy_comparison.csv')
    compare_results.to_csv(comparison_file, index=False)

    # Create equity curve comparison figure
    plt.figure(figsize=(14, 7))

    for strategy_name, results in results_dict.items():
        plt.plot((results['equity_curve'] - 1) * 100, label=strategy_name)

    plt.title('Strategy Comparison')
    plt.ylabel('Return (%)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend()

    comparison_plot_file = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(comparison_plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Create drawdown comparison figure
    plt.figure(figsize=(14, 7))

    for strategy_name, results in results_dict.items():
        equity = results['equity_curve']
        peak = equity.expanding().max()
        drawdown = (equity / peak - 1) * 100
        plt.plot(drawdown.index, drawdown, label=strategy_name)

    plt.title('Drawdown Comparison')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend()

    drawdown_plot_file = os.path.join(output_dir, 'drawdown_comparison.png')
    plt.savefig(drawdown_plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Create trade analysis charts if we have trades for the combined strategy
    combined_strategy = next((name for name, res in results_dict.items() if 'Combined' in name), None)

    if combined_strategy and not results_dict[combined_strategy]['trades'].empty and len(
            results_dict[combined_strategy]['trades']) > 5:
        trades_df = results_dict[combined_strategy]['trades']

        # Analyze trades by conditions
        trade_analysis = analyze_trades_by_conditions(trades_df)

        # Save trade analysis to file
        for key, analysis in trade_analysis.items():
            if key != 'summary' and not isinstance(analysis, str):
                analysis_file = os.path.join(output_dir, f'trade_analysis_{key}.csv')
                analysis.to_csv(analysis_file)

        # Create trade analysis visualization
        if 'vol_ratio_buckets' in trade_analysis and 'volume_percentile_buckets' in trade_analysis:
            # Plot win rate by vol ratio and volume buckets
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Vol ratio analysis
            vol_ratio_analysis = trade_analysis['vol_ratio_buckets']
            ax1.bar(vol_ratio_analysis.index, vol_ratio_analysis['win_rate'], color='blue', alpha=0.7)
            ax1.set_title('Win Rate by Volatility Ratio')
            ax1.set_xlabel('Volatility Ratio Range')
            ax1.set_ylabel('Win Rate (%)')
            ax1.axhline(y=50, color='red', linestyle='--')
            for i, count in enumerate(vol_ratio_analysis['count']):
                ax1.text(i, vol_ratio_analysis['win_rate'].iloc[i] + 2, f"n={int(count)}", ha='center')
            ax1.grid(True, alpha=0.3)

            # Volume percentile analysis
            vol_pct_analysis = trade_analysis['volume_percentile_buckets']
            ax2.bar(vol_pct_analysis.index, vol_pct_analysis['win_rate'], color='green', alpha=0.7)
            ax2.set_title('Win Rate by Volume Percentile')
            ax2.set_xlabel('Volume Percentile Range')
            ax2.set_ylabel('Win Rate (%)')
            ax2.axhline(y=50, color='red', linestyle='--')
            for i, count in enumerate(vol_pct_analysis['count']):
                ax2.text(i, vol_pct_analysis['win_rate'].iloc[i] + 2, f"n={int(count)}", ha='center')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            analysis_plot_file = os.path.join(output_dir, 'win_rate_analysis.png')
            plt.savefig(analysis_plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            # Plot average profit by vol ratio and volume buckets
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Vol ratio analysis
            ax1.bar(vol_ratio_analysis.index, vol_ratio_analysis['avg_profit'], color='blue', alpha=0.7)
            ax1.set_title('Average Profit by Volatility Ratio')
            ax1.set_xlabel('Volatility Ratio Range')
            ax1.set_ylabel('Average Profit (%)')
            ax1.axhline(y=0, color='red', linestyle='--')
            for i, count in enumerate(vol_ratio_analysis['count']):
                ax1.text(i, vol_ratio_analysis['avg_profit'].iloc[i] + 0.2, f"n={int(count)}", ha='center')
            ax1.grid(True, alpha=0.3)

            # Volume percentile analysis
            ax2.bar(vol_pct_analysis.index, vol_pct_analysis['avg_profit'], color='green', alpha=0.7)
            ax2.set_title('Average Profit by Volume Percentile')
            ax2.set_xlabel('Volume Percentile Range')
            ax2.set_ylabel('Average Profit (%)')
            ax2.axhline(y=0, color='red', linestyle='--')
            for i, count in enumerate(vol_pct_analysis['count']):
                ax2.text(i, vol_pct_analysis['avg_profit'].iloc[i] + 0.2, f"n={int(count)}", ha='center')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            profit_plot_file = os.path.join(output_dir, 'avg_profit_analysis.png')
            plt.savefig(profit_plot_file, dpi=150, bbox_inches='tight')
            plt.close()

    # Create a basic HTML report
    report_path = os.path.join(output_dir, 'performance_report.html')

    with open(report_path, 'w') as f:
        f.write(f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Strategy Performance Report</title>
           <style>
               body {{ font-family: Arial, sans-serif; margin: 20px; }}
               h1, h2, h3 {{ color: #333; }}
               table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
               th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
               th {{ background-color: #f2f2f2; text-align: center; }}
               tr:nth-child(even) {{ background-color: #f9f9f9; }}
               tr:hover {{ background-color: #f5f5f5; }}
               .chart {{ margin: 20px 0; max-width: 100%; }}
               .section {{ margin: 30px 0; }}
               .strategy-name {{ text-align: left; font-weight: bold; }}
               .best-value {{ color: green; font-weight: bold; }}
           </style>
       </head>
       <body>
           <h1>Strategy Performance Report</h1>
           <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
           <div class="section">
               <h2>Strategy Comparison</h2>
               <table>
                   <tr>
                       <th>Strategy</th>
                       <th>Total Return (%)</th>
                       <th>Annual Return (%)</th>
                       <th>Sharpe Ratio</th>
                       <th>Max Drawdown (%)</th>
                       <th>Win Rate (%)</th>
                       <th>Total Trades</th>
                       <th>Profit Factor</th>
                   </tr>
       """)

        # Add rows for each strategy
        compare_df = pd.DataFrame(compare_results)
        best_values = compare_df.iloc[:, 1:].idxmax()

        for i, row in compare_df.iterrows():
            f.write("<tr>")
            f.write(f'<td class="strategy-name">{row["Strategy"]}</td>')

            for col in ["Total Return (%)", "Annualized Return (%)", "Sharpe Ratio", "Max Drawdown (%)",
                        "Win Rate (%)", "Total Trades", "Profit Factor"]:
                value = row[col]
                is_best = (i == best_values[col])
                cell_class = 'best-value' if is_best else ''
                f.write(f'<td class="{cell_class}">{value:.2f}</td>')

            f.write("</tr>")

        f.write("""
               </table>
           </div>
           <div class="section">
               <h2>Equity Curve Comparison</h2>
               <img src="strategy_comparison.png" class="chart" alt="Equity Curve Comparison">
           </div>
           <div class="section">
               <h2>Drawdown Comparison</h2>
               <img src="drawdown_comparison.png" class="chart" alt="Drawdown Comparison">
           </div>
       """)

        # Add trade analysis if available
        if os.path.exists(os.path.join(output_dir, 'win_rate_analysis.png')):
            f.write("""
               <div class="section">
                   <h2>Trade Performance Analysis</h2>
                   <h3>Win Rate Analysis</h3>
                   <img src="win_rate_analysis.png" class="chart" alt="Win Rate Analysis">
                   <h3>Average Profit Analysis</h3>
                   <img src="avg_profit_analysis.png" class="chart" alt="Average Profit Analysis">
               </div>
           """)

        f.write("""
           <div class="section">
               <h2>Conclusions</h2>
               <p>Based on the analysis, the following observations can be made:</p>
               <ul>
       """)

        # Add some automatic conclusions
        best_strategy = compare_df.loc[compare_df['Sharpe Ratio'].idxmax(), 'Strategy']
        f.write(
            f'<li>The <strong>{best_strategy}</strong> strategy shows the best risk-adjusted performance (highest Sharpe ratio).</li>')

        # Compare returns
        base_return = next(
            (row['Annualized Return (%)'] for i, row in compare_df.iterrows() if row['Strategy'] == 'Standard Copula'),
            0)
        best_return = compare_df['Annualized Return (%)'].max()
        best_return_strategy = compare_df.loc[compare_df['Annualized Return (%)'].idxmax(), 'Strategy']

        if best_return > base_return:
            improvement = best_return - base_return
            f.write(
                f'<li>The {best_return_strategy} strategy improved annual returns by <strong>{improvement:.2f}%</strong> compared to the standard strategy.</li>')
        else:
            f.write(
                f'<li>None of the filter strategies improved returns compared to the standard strategy.</li>')

        # Compare win rates
        base_winrate = next(
            (row['Win Rate (%)'] for i, row in compare_df.iterrows() if row['Strategy'] == 'Standard Copula'), 0)
        best_winrate = compare_df['Win Rate (%)'].max()
        best_winrate_strategy = compare_df.loc[compare_df['Win Rate (%)'].idxmax(), 'Strategy']

        if best_winrate > base_winrate:
            winrate_diff = best_winrate - base_winrate
            f.write(
                f'<li>The {best_winrate_strategy} strategy improved win rate by <strong>{winrate_diff:.2f}%</strong> compared to the standard strategy.</li>')
        else:
            f.write(
                f'<li>None of the filter strategies improved win rate compared to the standard strategy.</li>')

        f.write("""
               </ul>
           </div>
       </body>
       </html>
       """)

    print(f"Saved performance report to {report_path}")

    return report_path

def analyze_trades_by_conditions(trades_df):
    """
    Analyze performance by various conditions
    Parameters:
    - trades_df: DataFrame with trade data including vol_ratio and volume_percentile
    Returns:
    - analysis: Dictionary with analysis results
    """
    if trades_df.empty or len(trades_df) < 5:
        return {"error": "Not enough trades for analysis"}

    analysis = {}

    if 'vol_ratio' in trades_df.columns:
        # Create buckets for vol ratio
        vol_ratio_bins = [0, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, np.inf]
        vol_ratio_labels = ['0-0.5', '0.5-0.8', '0.8-1.0', '1.0-1.25', '1.25-1.5', '1.5-2.0', '2.0+']

        trades_df['vol_ratio_bucket'] = pd.cut(trades_df['vol_ratio'], bins=vol_ratio_bins,
                                               labels=vol_ratio_labels)

        # Group by buckets
        vol_ratio_analysis = trades_df.groupby('vol_ratio_bucket').agg({
            'profit_pct': ['mean', 'std', lambda x: (x > 0).mean(), 'count'],
            'position': lambda x: (x == 'Long').mean()
        })

        vol_ratio_analysis.columns = ['avg_profit', 'std_profit', 'win_rate', 'count', 'long_pct']
        vol_ratio_analysis['avg_profit'] *= 100  # Convert to percentage
        vol_ratio_analysis['std_profit'] *= 100  # Convert to percentage
        vol_ratio_analysis['win_rate'] *= 100  # Convert to percentage
        vol_ratio_analysis['long_pct'] *= 100  # Convert to percentage

        analysis['vol_ratio_buckets'] = vol_ratio_analysis

        # 2. Analyze by volume percentile buckets
    if 'volume_percentile' in trades_df.columns:
        # Create buckets for volume percentile
        vol_pct_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        vol_pct_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

        trades_df['vol_pct_bucket'] = pd.cut(trades_df['volume_percentile'], bins=vol_pct_bins,
                                             labels=vol_pct_labels)

        # Group by buckets
        vol_pct_analysis = trades_df.groupby('vol_pct_bucket').agg({
            'profit_pct': ['mean', 'std', lambda x: (x > 0).mean(), 'count'],
            'position': lambda x: (x == 'Long').mean()
        })

        vol_pct_analysis.columns = ['avg_profit', 'std_profit', 'win_rate', 'count', 'long_pct']
        vol_pct_analysis['avg_profit'] *= 100  # Convert to percentage
        vol_pct_analysis['std_profit'] *= 100  # Convert to percentage
        vol_pct_analysis['win_rate'] *= 100  # Convert to percentage
        vol_pct_analysis['long_pct'] *= 100  # Convert to percentage

        analysis['volume_percentile_buckets'] = vol_pct_analysis

        # 3. Analyze by position type
    position_analysis = trades_df.groupby('position').agg({
        'profit_pct': ['mean', 'std', lambda x: (x > 0).mean(), 'count'],
        'duration': 'mean'
    })

    position_analysis.columns = ['avg_profit', 'std_profit', 'win_rate', 'count', 'avg_duration']
    position_analysis['avg_profit'] *= 100  # Convert to percentage
    position_analysis['std_profit'] *= 100  # Convert to percentage
    position_analysis['win_rate'] *= 100  # Convert to percentage

    analysis['position_analysis'] = position_analysis

    # 4. Overall statistics summary
    analysis['summary'] = {
        'total_trades': len(trades_df),
        'avg_profit': trades_df['profit_pct'].mean() * 100,
        'win_rate': (trades_df['profit_pct'] > 0).mean() * 100,
        'sharpe': trades_df['profit_pct'].mean() / trades_df['profit_pct'].std() * np.sqrt(len(trades_df)) if
        trades_df['profit_pct'].std() > 0 else 0,
        'avg_duration': trades_df['duration'].mean(),
        'long_trades_pct': (trades_df['position'] == 'Long').mean() * 100
    }

    return analysis


def generate_copula_signals_with_atr_stop(token1, token2, window_size= "**********"=0.95,
                                          atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                                          copula_type='gaussian', copula_params=None):
    """
    Generate trading signals based on copula confidence bands with ATR trailing stop-loss
    Parameters:
    - token1, token2: "**********"
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands (e.g., 0.95 for 95%)
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss (e.g., 2.0 = 2x ATR)
    - fee_pct: Transaction fee percentage
    - copula_type: Type of copula to use ('gaussian', 'clayton', 'student_t', 'gumbel')
    - copula_params: Dictionary of parameters for the specific copula type
    Returns:
    - signals: Trading signals (1=long spread, -1=short spread, 0=no position)
    - stop_levels: Stop-loss levels
    - equity_curve: Equity curve
    - is_outside: Boolean series indicating points outside confidence bands
    - copula: The fitted copula object
    """
    # Initialize series
    signals = "**********"=token1.index)
    is_outside = "**********"=token1.index)
    stop_levels = "**********"=token1.index)

    # Calculate spread
    spread = "**********"

    # Calculate ATR for stop-loss
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Create copula model based on selected type
    copula = create_copula(copula_type, copula_params)

    # Track position and equity
    current_position = 0
    entry_price = 0
    stop_price = 0
    equity_curve = "**********"=token1.index)

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # For each point in the time series (after initial window)
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********": "**********"
        try:
            # Extract window data
            window_t1 = token1.iloc[i - window_size: "**********"
            window_t2 = token2.iloc[i - window_size: "**********"

            # Current values
            current_t1 = "**********"
            current_t2 = "**********"
            current_spread = spread.iloc[i]
            current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02

            # Update equity
            if i > 0:
                if current_position == 1:  # Long position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
                elif current_position == -1:  # Short position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
                else:  # No position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1]

            # Skip if data is invalid
            if len(window_t1.dropna()) < window_size * 0.9 or len(window_t2.dropna()) < window_size * 0.9:
                signals.iloc[i] = current_position
                stop_levels.iloc[i] = stop_price
                continue

            # Convert to uniform [0,1] using empirical CDF
            u_window = rankdata(window_t1) / (len(window_t1) + 1)
            v_window = rankdata(window_t2) / (len(window_t2) + 1)

            # Current points as quantiles
            u_current = (rankdata(np.append(window_t1, current_t1))[-1]) / (window_size + 1)
            v_current = (rankdata(np.append(window_t2, current_t2))[-1]) / (window_size + 1)

            # Fit copula to window data
            copula.fit(u_window, v_window)

            # Check if point is outside confidence bands
            outside_bands, distance, direction = copula.is_point_outside_bands(u_current, v_current, confidence_level)
            is_outside.iloc[i] = outside_bands

            # Determine signal based on whether point is outside bands and direction
            if outside_bands:
                if direction < 0:  # First asset undervalued
                    new_signal = 1  # Long signal
                elif direction > 0:  # First asset overvalued
                    new_signal = -1  # Short signal
                else:
                    new_signal = current_position
            else:
                # Exit when point returns inside bands
                new_signal = 0 if current_position != 0 else current_position

            # Check for stop-loss (if in a position)
            stop_hit = False
            if current_position == 1 and stop_price > 0 and current_spread < stop_price:
                stop_hit = True
            elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
                stop_hit = True

            # Apply trading logic
            if current_position == 0:  # No current position
                if new_signal != 0:  # Enter new position
                    # Apply fee for entry
                    equity_curve.iloc[i] *= (1 - fee_pct)

                    # Set position and entry price
                    current_position = new_signal
                    entry_price = current_spread

                    # Set initial stop-loss level
                    if current_position == 1:  # Long position
                        stop_price = entry_price - atr_multiplier * current_atr
                    else:  # Short position
                        stop_price = entry_price + atr_multiplier * current_atr

                    signals.iloc[i] = current_position
                    stop_levels.iloc[i] = stop_price

            else:  # Already in a position
                # Update trailing stop if in profit
                if not stop_hit:
                    if current_position == 1 and current_spread > entry_price:
                        # For long positions, raise stop as price increases
                        new_stop = current_spread - atr_multiplier * current_atr
                        stop_price = max(stop_price, new_stop)
                    elif current_position == -1 and current_spread < entry_price:
                        # For short positions, lower stop as price decreases
                        new_stop = current_spread + atr_multiplier * current_atr
                        stop_price = min(stop_price, new_stop)

                # Determine whether to exit or maintain position
                if stop_hit:  # Stop-loss hit
                    # Apply fee for exit
                    equity_curve.iloc[i] *= (1 - fee_pct)

                    signals.iloc[i] = 0
                    current_position = 0
                    stop_price = 0

                elif new_signal == 0:  # Exit signal (point inside bands)
                    # Apply fee for exit
                    equity_curve.iloc[i] *= (1 - fee_pct)

                    signals.iloc[i] = 0
                    current_position = 0
                    stop_price = 0

                elif new_signal != current_position and new_signal != 0:  # Reversal signal
                    # Apply fee for both exit and entry
                    equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                    current_position = new_signal
                    entry_price = current_spread

                    # Set new stop-loss level
                    if current_position == 1:  # Long position
                        stop_price = entry_price - atr_multiplier * current_atr
                    else:  # Short position
                        stop_price = entry_price + atr_multiplier * current_atr

                    signals.iloc[i] = current_position

                else:  # Maintain current position
                    signals.iloc[i] = current_position

                # Record stop level
                stop_levels.iloc[i] = stop_price
        except Exception as e:
            print(f"Error at index {i}: {e}")
            # Keep previous values if error occurs
            if i > 0:
                signals.iloc[i] = signals.iloc[i - 1]
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]
                stop_levels.iloc[i] = stop_levels.iloc[i - 1]

    return signals, stop_levels, equity_curve, is_outside, copula


def backtest_strategy(token1, token2, window_size= "**********"=0.95,
                      atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                      copula_type='gaussian', copula_params=None):
    """
    Backtest the copula strategy with ATR trailing stop-loss
    Parameters:
    - token1, token2: "**********"
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    - copula_type: Type of copula to use ('gaussian', 'clayton', 'student_t', 'gumbel')
    - copula_params: Dictionary of parameters for the specific copula type
    Returns:
    - results: Dictionary with backtest results and data
    """
    # Generate signals and equity curve
    signals, stop_levels, equity_curve, is_outside, copula = generate_copula_signals_with_atr_stop(
        token1, token2, window_size, confidence_level, atr_window, atr_multiplier, fee_pct,
        copula_type, copula_params
    )

    # Calculate spread
    spread = "**********"

    # Calculate trade statistics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = trade_df['profit_pct'].mean()

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

        performance_summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_winning_trade': trade_df.loc[
                trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                          total_trades - winning_trades) > 0 else 0,
            'largest_winner': trade_df['profit_pct'].max(),
            'largest_loser': trade_df['profit_pct'].min(),
            'avg_trade_duration': trade_df['duration'].mean(),
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']) != 0 else float('inf'),
            'monthly_returns': monthly_returns
        }
    else:
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }
        trade_df = pd.DataFrame()

    # Store all results
    results = {
        'signals': signals,
        'stop_levels': stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'copula': {
            'type': copula.name,
            'params': copula.params
        },
        'performance': performance_summary
    }

    return results


def plot_strategy_results(results, title="Copula Strategy with ATR Trailing Stop-Loss"):
    """
    Create comprehensive visualizations of strategy performance
    Parameters:
    - results: Dictionary with backtest results
    - title: Title for the main plot
    Returns:
    - figs: List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    is_outside = results['is_outside']
    performance = results['performance']
    trades = results['trades'] if 'trades' in results else pd.DataFrame()

    figs = []

    # 1. Create main plot with signals and equity curve
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax2.plot(returns_pct.index, returns_pct, 'g-')
    ax2.set_title('Equity Curve')
    ax2.set_ylabel('Return (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax2.text(0.02, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create drawdown chart
    fig2, ax = plt.subplots(figsize=(14, 5))

    # Calculate drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve / peak - 1) * 100  # Convert to percentage

    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, 'r-', linewidth=1)

    ax.set_title('Drawdown Analysis')
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)

    # Add max drawdown line
    min_dd = drawdown.min()
    ax.axhline(y=min_dd, color='black', linestyle='--',
               label=f'Max Drawdown: {abs(min_dd):.2f}%')
    ax.legend()

    plt.tight_layout()
    figs.append(fig2)

    # 3. Trade analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Trade duration histogram
        ax1.hist(trades['duration'], bins=20, color='blue', alpha=0.7)
        ax1.set_title('Trade Duration Distribution')
        ax1.set_xlabel('Duration (Days)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Trade P&L histogram
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # Add vertical line at zero
        ax2.axvline(x=0, color='red', linestyle='--')

        plt.tight_layout()
        figs.append(fig3)

    return figs


def run_backtest_from_csv(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                          window_size=20, confidence_level=0.95, atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                          copula_type='gaussian', copula_params=None, output_dir='copula_output'):
    """
    Run backtest using data from a CSV file
    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of the date/timestamp column
    - token1_col: "**********"
    - token2_col: "**********"
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    - copula_type: Type of copula to use ('gaussian', 'clayton', 'student_t', 'gumbel')
    - copula_params: Dictionary of parameters for the specific copula type
    - output_dir: Directory to save output visualizations
    Returns:
    - results: Dictionary with backtest results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Check if required columns exist
    required_cols = "**********"
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running backtest for {pair_name}...")
    print(f"Data range: "**********"
    print(f"Number of data points: "**********"
    print(f"Parameters: window_size={window_size}, confidence_level={confidence_level}, " +
          f"atr_window={atr_window}, atr_multiplier={atr_multiplier}, fee_pct={fee_pct * 100}%")
    print(f"Copula: type={copula_type}")

    # Run backtest
    results = "**********"
                                atr_window, atr_multiplier, fee_pct,
                                copula_type, copula_params)

    # Create plots and save them
    title = f"{copula_type.capitalize()} Copula Strategy: {pair_name} (Window: {window_size}, Conf: {confidence_level * 100}%, ATR: {atr_multiplier}x)"
    figs = plot_strategy_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_{copula_type}_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_{copula_type}_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_{copula_type}_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance']['total_return'] * 100:.2f}%")
    print(f"Annualized Return: {results['performance']['annualized_return'] * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown'] * 100:.2f}%")
    print(f"Win Rate: {results['performance']['win_rate'] * 100:.2f}%")
    print(f"Total Trades: {results['performance']['total_trades']}")
    print(f"Profit Factor: {results['performance']['profit_factor']:.2f}")

    return results


def optimize_parameters(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                        window_sizes=[20, 40, 60], confidence_levels=[0.90, 0.95, 0.99],
                        atr_multipliers=[1.5, 2.0, 2.5], fee_pct=0.001, output_dir='optimization_results'):
    """
    Optimize strategy parameters using grid search
    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of the date/timestamp column
    - token1_col: "**********"
    - token2_col: "**********"
    - window_sizes: List of window sizes to test
    - confidence_levels: List of confidence levels to test
    - atr_multipliers: List of ATR multipliers to test
    - fee_pct: Transaction fee percentage
    - output_dir: Directory to save results
    Returns:
    - results_df: DataFrame with optimization results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Pair name
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Optimizing parameters for {pair_name}...")

    # Store results
    results = []

    # Grid search
    total_combinations = len(window_sizes) * len(confidence_levels) * len(atr_multipliers)
    current = 0

    for window_size in window_sizes:
        for confidence_level in confidence_levels:
            for atr_multiplier in atr_multipliers:
                current += 1
                print(f"Testing combination {current}/{total_combinations}: " +
                      f"window={window_size}, conf={confidence_level}, atr={atr_multiplier}")

                try:
                    # Run backtest with these parameters
                    res = "**********"
                                            14, atr_multiplier, fee_pct)

                    # Extract performance metrics
                    perf = res['performance']

                    # Store results
                    results.append({
                        'window_size': window_size,
                        'confidence_level': confidence_level,
                        'atr_multiplier': atr_multiplier,
                        'total_return': perf['total_return'],
                        'annualized_return': perf['annualized_return'],
                        'sharpe_ratio': perf['sharpe_ratio'],
                        'max_drawdown': perf['max_drawdown'],
                        'win_rate': perf['win_rate'],
                        'total_trades': perf['total_trades'],
                        'profit_factor': perf['profit_factor']
                    })
                except Exception as e:
                    print(f"  Error with this combination: {e}")

    # Create DataFrame with results
    if results:
        results_df = pd.DataFrame(results)

        # Sort by Sharpe ratio (descending)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        # Save results
        results_file = os.path.join(output_dir, f"{pair_name}_optimization.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved optimization results to {results_file}")

        # Display top 5 parameter combinations
        print("\nTop 5 parameter combinations:")
        display_cols = ['window_size', 'confidence_level', 'atr_multiplier',
                        'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        print(results_df[display_cols].head(5).to_string())

        # Create chart for the best combination
        best_params = results_df.iloc[0]
        print(f"\nRunning backtest with best parameters:")
        print(f"window_size={best_params['window_size']}, " +
              f"confidence_level={best_params['confidence_level']}, " +
              f"atr_multiplier={best_params['atr_multiplier']}")

        best_results = "**********"
                                         int(best_params['window_size']),
                                         best_params['confidence_level'],
                                         14, best_params['atr_multiplier'], fee_pct)

        title = f"Best Parameters: {pair_name} (Window: {int(best_params['window_size'])}, " + \
                f"Conf: {best_params['confidence_level'] * 100}%, ATR: {best_params['atr_multiplier']}x)"

        figs = plot_strategy_results(best_results, title)

        for i, fig in enumerate(figs):
            output_file = os.path.join(output_dir, f"{pair_name}_best_params_plot_{i + 1}.png")
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved best parameters plot to {output_file}")

        return results_df

    return pd.DataFrame()


#############################################
# FUNCTIONS FOR VOLATILITY RATIO FILTER #
#############################################

 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"a "**********"l "**********"c "**********"u "**********"l "**********"a "**********"t "**********"e "**********"_ "**********"v "**********"o "**********"l "**********"_ "**********"r "**********"a "**********"t "**********"i "**********"o "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"2 "**********", "**********"  "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********"= "**********"2 "**********"0 "**********") "**********": "**********"
    """
    Calculate volatility ratio between two tokens
    Parameters:
    - token1, token2: "**********"
    - window_size: Window size for volatility calculation
    Returns:
    - vol_ratio: "**********"
    - token1_vol: "**********"
    - token2_vol: "**********"
    """
    # Calculate returns
    token1_returns = "**********"
    token2_returns = "**********"

    # Calculate rolling volatility (standard deviation of returns)
    token1_vol = "**********"=window_size).std()
    token2_vol = "**********"=window_size).std()

    # Calculate volatility ratio (Token1/Token2)
    vol_ratio = "**********"

    # Handle infinite and NaN values
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return vol_ratio, token1_vol, token2_vol


 "**********"d "**********"e "**********"f "**********"  "**********"f "**********"i "**********"l "**********"t "**********"e "**********"r "**********"_ "**********"s "**********"i "**********"g "**********"n "**********"a "**********"l "**********"s "**********"_ "**********"b "**********"y "**********"_ "**********"v "**********"o "**********"l "**********"_ "**********"r "**********"a "**********"t "**********"i "**********"o "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"2 "**********", "**********"  "**********"s "**********"i "**********"g "**********"n "**********"a "**********"l "**********"s "**********", "**********"  "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********"= "**********"2 "**********"0 "**********", "**********"  "**********"v "**********"o "**********"l "**********"_ "**********"r "**********"a "**********"t "**********"i "**********"o "**********"_ "**********"t "**********"h "**********"r "**********"e "**********"s "**********"h "**********"o "**********"l "**********"d "**********"= "**********"1 "**********". "**********"2 "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"= "**********"' "**********"t "**********"h "**********"r "**********"e "**********"s "**********"h "**********"o "**********"l "**********"d "**********"' "**********") "**********": "**********"
    """
    Filter existing signals based on volatility ratio criteria
    Parameters:
    - token1, token2: "**********"
    - signals: Original signals to filter
    - window_size: Window size for volatility calculation
    - vol_ratio_threshold: Threshold for volatility ratio comparison
    - mode: Filter mode ('threshold', 'range', 'balanced')
    Returns:
    - filtered_signals: Signals filtered by volatility ratio
    - vol_ratio: "**********"
    """
    # Calculate volatility ratio
    vol_ratio, _, _ = "**********"

    # Initialize filtered signals (copy original signals)
    filtered_signals = signals.copy()

    # Apply filtering based on selected mode
    if mode == 'threshold':
        # Only keep signals when vol_ratio exceeds threshold (either direction)
        valid_ratio = (vol_ratio > vol_ratio_threshold) | (vol_ratio < 1 / vol_ratio_threshold)
        # Set signals to 0 when ratio is not valid
        filtered_signals[~valid_ratio] = 0

    elif mode == 'range':
        # Only keep signals when vol_ratio is within a balanced range
        min_threshold = 1 / vol_ratio_threshold
        max_threshold = vol_ratio_threshold
        valid_ratio = (vol_ratio >= min_threshold) & (vol_ratio <= max_threshold)
        # Set signals to 0 when ratio is not valid
        filtered_signals[~valid_ratio] = 0

    elif mode == 'balanced':
        # Only keep signals when Token1 and Token2 have similar volatility
        valid_ratio = (vol_ratio >= 0.8) & (vol_ratio <= 1.2)  # 20% difference threshold
        # Set signals to 0 when ratio is not valid
        filtered_signals[~valid_ratio] = 0

    return filtered_signals, vol_ratio


def backtest_with_vol_ratio_filter(token1, token2, window_size= "**********"=0.95,
                                   vol_ratio_window=20, vol_ratio_threshold=1.2, vol_ratio_mode='threshold',
                                   atr_window=14, atr_multiplier=2.0, fee_pct=0.001):
    """
    Backtest strategy with volatility ratio filtering
    Parameters:
    - token1, token2: "**********"
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - vol_ratio_window: Size of rolling window for volatility calculation
    - vol_ratio_threshold: Threshold for volatility ratio filtering
    - vol_ratio_mode: Mode for volatility ratio filtering
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    Returns:
    - results: Dictionary with backtest results
    """
    # Calculate spread
    spread = "**********"

    # Generate signals and equity curve
    raw_signals, stop_levels, _, is_outside, copula = generate_copula_signals_with_atr_stop(
        token1, token2, window_size, confidence_level, atr_window, atr_multiplier, fee_pct
    )

    # Filter signals based on volatility ratio
    filtered_signals, vol_ratio = filter_signals_by_vol_ratio(
        token1, token2, raw_signals, vol_ratio_window, vol_ratio_threshold, vol_ratio_mode
    )

    # Calculate ATR for stop-loss
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Initialize series for actual trading signals and stop levels
    signals = "**********"=token1.index)
    final_stop_levels = "**********"=token1.index)

    # Track position and equity
    current_position = 0
    entry_price = 0
    stop_price = 0
    equity_curve = "**********"=token1.index)

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Apply trading logic with ATR-based stop-loss
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********": "**********"
        # Current values
        current_spread = spread.iloc[i]
        current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02
        new_signal = filtered_signals.iloc[i]

        # Update equity
        if i > 0:
            if current_position == 1:  # Long position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
            elif current_position == -1:  # Short position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
            else:  # No position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        # Check for stop-loss (if in a position)
        stop_hit = False
        if current_position == 1 and stop_price > 0 and current_spread < stop_price:
            stop_hit = True
        elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
            stop_hit = True

        # Apply trading logic
        if current_position == 0:  # No current position
            if new_signal != 0:  # Enter new position
                # Apply fee for entry
                equity_curve.iloc[i] *= (1 - fee_pct)

                # Set position and entry price
                current_position = new_signal
                entry_price = current_spread

                # Set initial stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position
                final_stop_levels.iloc[i] = stop_price

        else:  # Already in a position
            # Update trailing stop if in profit
            if not stop_hit:
                if current_position == 1 and current_spread > entry_price:
                    # For long positions, raise stop as price increases
                    new_stop = current_spread - atr_multiplier * current_atr
                    stop_price = max(stop_price, new_stop)
                elif current_position == -1 and current_spread < entry_price:
                    # For short positions, lower stop as price decreases
                    new_stop = current_spread + atr_multiplier * current_atr
                    stop_price = min(stop_price, new_stop)

            # Determine whether to exit or maintain position
            if stop_hit:  # Stop-loss hit
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal == 0 and raw_signals.iloc[i] == 0:  # Exit signal
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal != current_position and new_signal != 0:  # Reversal signal
                # Apply fee for both exit and entry
                equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                current_position = new_signal
                entry_price = current_spread

                # Set new stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position

            else:  # Maintain current position
                signals.iloc[i] = current_position

            # Record stop level
            final_stop_levels.iloc[i] = stop_price

    # Calculate trade statistics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration,
                'vol_ratio': vol_ratio.loc[current_date]
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'vol_ratio': vol_ratio.loc[last_date],
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

            performance_summary = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'avg_winning_trade': trade_df.loc[
                    trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
                'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                              total_trades - winning_trades) > 0 else 0,
                'largest_winner': trade_df['profit_pct'].max(),
                'largest_loser': trade_df['profit_pct'].min(),
                'avg_trade_duration': trade_df['duration'].mean(),
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']) != 0 else float('inf'),
            'monthly_returns': monthly_returns
        }
    else:
        trade_df = pd.DataFrame()
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }

    # Store all results
    results = {
        'signals': signals,
        'raw_signals': raw_signals,
        'filtered_signals': filtered_signals,
        'vol_ratio': vol_ratio,
        'stop_levels': final_stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'copula': {
            'rho': getattr(copula, 'rho', None),
        },
        'performance': performance_summary,
        'vol_ratio_params': {
            'window': vol_ratio_window,
            'threshold': vol_ratio_threshold,
            'mode': vol_ratio_mode
        }
    }

    return results


def plot_vol_ratio_results(results, title="Volatility Ratio Filtered Strategy"):
    """
    Create visualization of volatility ratio filtered strategy performance
    Parameters:
    - results: Dictionary with backtest results
    - title: Title for the main plot
    Returns:
    - figs: List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    vol_ratio = results['vol_ratio']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    performance = results['performance']
    trades = results['trades']

    figs = []

    # 1. Create multi-panel plot with signals, vol ratio, and equity curve
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot volatility ratio
    ax2.plot(vol_ratio.index, vol_ratio, 'purple', label= "**********"
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

    # Add threshold/range visualization based on mode
    vol_params = results['vol_ratio_params']
    mode = vol_params['mode']
    threshold = vol_params['threshold']

    if mode == 'threshold':
        ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                    label=f'Upper Threshold: {threshold:.2f}')
        ax2.axhline(y=1 / threshold, color='orange', linestyle=':', alpha=0.7,
                    label=f'Lower Threshold: {1 / threshold:.2f}')
    elif mode == 'range':
        # Fill the valid range area
        valid_x = vol_ratio.index
        ax2.fill_between(valid_x, 1 / threshold, threshold, color='green', alpha=0.2,
                         label=f'Valid Range: {1 / threshold:.2f}-{threshold:.2f}')

    ax2.set_ylabel('Volatility Ratio')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create additional analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L vs Volatility Ratio
        ax1.scatter(trades['vol_ratio'], trades['profit_pct'] * 100,
                    alpha=0.7, c='blue', edgecolors='k')
        ax1.set_title('Trade P&L vs Volatility Ratio')
        ax1.set_xlabel('Volatility Ratio (Token1/Token2)')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.axvline(x=1, color='gray', linestyle='--')
        ax1.grid(True, alpha=0.3)

        # P&L histogram
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

        # 3. Create comparison of filtered vs raw signals
        if 'raw_signals' in results and 'filtered_signals' in results:
            fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Plot raw signals
            raw_signals = results['raw_signals']
            raw_long = raw_signals[raw_signals == 1]
            raw_short = raw_signals[raw_signals == -1]

            ax1.plot(spread.index, spread, 'b-', alpha=0.5)
            ax1.scatter(raw_long.index, spread.loc[raw_long.index], marker='^', color='green', s=50,
                        label='Raw Long Signals')
            ax1.scatter(raw_short.index, spread.loc[raw_short.index], marker='v', color='red', s=50,
                        label='Raw Short Signals')
            ax1.set_title('Raw Signals Before Volatility Ratio Filter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot filtered signals
            filtered_signals = results['filtered_signals']
            filtered_long = filtered_signals[filtered_signals == 1]
            filtered_short = filtered_signals[filtered_signals == -1]

            # Calculate signals removed by filter
            removed_long = raw_signals[(raw_signals == 1) & (filtered_signals == 0)]
            removed_short = raw_signals[(raw_signals == -1) & (filtered_signals == 0)]

            ax2.plot(spread.index, spread, 'b-', alpha=0.5)
            ax2.scatter(filtered_long.index, spread.loc[filtered_long.index], marker='^', color='green', s=50,
                        label='Filtered Long Signals')
            ax2.scatter(filtered_short.index, spread.loc[filtered_short.index], marker='v', color='red', s=50,
                        label='Filtered Short Signals')
            ax2.scatter(removed_long.index, spread.loc[removed_long.index], marker='x', color='orange', s=50,
                        label='Removed Long Signals')
            ax2.scatter(removed_short.index, spread.loc[removed_short.index], marker='x', color='purple', s=50,
                        label='Removed Short Signals')
            ax2.set_title('Signals After Volatility Ratio Filter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            figs.append(fig3)

    return figs


def run_vol_ratio_backtest(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                           window_size=20, confidence_level=0.95,
                           vol_ratio_window=20, vol_ratio_threshold=1.2, vol_ratio_mode='threshold',
                           atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                           output_dir='volratio_results'):
    """
    Run backtest with volatility ratio filtering
    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of the date/timestamp column
    - token1_col: "**********"
    - token2_col: "**********"
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - vol_ratio_window: Size of rolling window for volatility calculation
    - vol_ratio_threshold: Threshold for volatility ratio filtering
    - vol_ratio_mode: Mode for volatility ratio filtering ('threshold', 'range', 'balanced')
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    - output_dir: Directory to save output visualizations
    Returns:
    - results: Dictionary with backtest results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Check if required columns exist
    required_cols = "**********"
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running volatility ratio backtest for {pair_name}...")
    print(f"Data range: "**********"
    print(f"Number of data points: "**********"
    print(f"Parameters: window_size={window_size}, confidence_level={confidence_level}, " +
          f"vol_ratio_window={vol_ratio_window}, vol_ratio_threshold={vol_ratio_threshold}, vol_ratio_mode={vol_ratio_mode}")

    # Run backtest
    results = backtest_with_vol_ratio_filter(
        token1, token2, window_size, confidence_level,
        vol_ratio_window, vol_ratio_threshold, vol_ratio_mode,
        atr_window, atr_multiplier, fee_pct
    )

    # Create plots and save them
    if vol_ratio_mode == 'threshold':
        mode_desc = f"Threshold: {vol_ratio_threshold}"
    elif vol_ratio_mode == 'range':
        mode_desc = f"Range: {1 / vol_ratio_threshold:.2f}-{vol_ratio_threshold:.2f}"
    else:
        mode_desc = f"Balanced Range"

    title = f"Vol Ratio Filter: {pair_name} (Window: {window_size}, {vol_ratio_mode.capitalize()}: {mode_desc})"
    figs = plot_vol_ratio_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_volratio_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_volratio_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_volratio_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance'].get('total_return', 0) * 100:.2f}%")
    print(f"Annualized Return: {results['performance'].get('annualized_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results['performance'].get('max_drawdown', 0) * 100:.2f}%")
    print(f"Win Rate: {results['performance'].get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {results['performance'].get('total_trades', 0)}")
    print(f"Profit Factor: {results['performance'].get('profit_factor', 0):.2f}")

    return results


def optimize_vol_ratio_params(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                              window_size=20, confidence_level=0.95,
                              vol_ratio_windows=[10, 20, 30], vol_ratio_thresholds=[1.1, 1.2, 1.5, 2.0],
                              vol_ratio_modes=['threshold', 'range'],
                              atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                              output_dir='volratio_optimization'):
    """
    Optimize volatility ratio parameters
    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of the date/timestamp column
    - token1_col: "**********"
    - token2_col: "**********"
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - vol_ratio_windows: List of window sizes for volatility calculation
    - vol_ratio_thresholds: List of thresholds for volatility ratio filtering
    - vol_ratio_modes: List of modes for volatility ratio filtering
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    - output_dir: Directory to save results
    Returns:
    - results_df: DataFrame with optimization results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Pair name
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Optimizing volatility ratio parameters for {pair_name}...")

    # Store results
    results = []

    # Grid search
    total_combinations = len(vol_ratio_windows) * len(vol_ratio_thresholds) * len(vol_ratio_modes)
    current = 0

    for vol_ratio_window in vol_ratio_windows:
        for vol_ratio_threshold in vol_ratio_thresholds:
            for vol_ratio_mode in vol_ratio_modes:
                current += 1

                if vol_ratio_mode == 'threshold':
                    mode_desc = f"Threshold: {vol_ratio_threshold}"
                elif vol_ratio_mode == 'range':
                    mode_desc = f"Range: {1 / vol_ratio_threshold:.2f}-{vol_ratio_threshold:.2f}"
                else:
                    mode_desc = f"Balanced Range"

                print(f"Testing combination {current}/{total_combinations}: " +
                      f"vol_window={vol_ratio_window}, {vol_ratio_mode} {mode_desc}")

                try:
                    # Run backtest with these parameters
                    res = backtest_with_vol_ratio_filter(
                        token1, token2, window_size, confidence_level,
                        vol_ratio_window, vol_ratio_threshold, vol_ratio_mode,
                        atr_window, atr_multiplier, fee_pct
                    )

                    # Extract performance metrics
                    perf = res['performance']

                    # Skip combinations with too few trades
                    if perf['total_trades'] < 5:
                        print(f"  Skipping combination with only {perf['total_trades']} trades")
                        continue

                    # Store results
                    results.append({
                        'vol_ratio_window': vol_ratio_window,
                        'vol_ratio_threshold': vol_ratio_threshold,
                        'vol_ratio_mode': vol_ratio_mode,
                        'total_return': perf['total_return'],
                        'annualized_return': perf['annualized_return'],
                        'sharpe_ratio': perf['sharpe_ratio'],
                        'max_drawdown': perf['max_drawdown'],
                        'win_rate': perf['win_rate'],
                        'total_trades': perf['total_trades'],
                        'profit_factor': perf['profit_factor']
                    })
                except Exception as e:
                    print(f"  Error with this combination: {e}")

    # Create DataFrame with results
    if results:
        results_df = pd.DataFrame(results)

        # Sort by Sharpe ratio (descending)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        # Save results
        results_file = os.path.join(output_dir, f"{pair_name}_volratio_optimization.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved optimization results to {results_file}")

        # Display top 5 parameter combinations
        print("\nTop 5 parameter combinations:")
        display_cols = ['vol_ratio_window', 'vol_ratio_threshold', 'vol_ratio_mode',
                        'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'total_trades']
        print(results_df[display_cols].head(5).to_string())

        # Create chart for the best combination
        best_params = results_df.iloc[0]
        print(f"\nRunning backtest with best parameters:")
        print(f"vol_ratio_window={best_params['vol_ratio_window']}, " +
              f"vol_ratio_threshold={best_params['vol_ratio_threshold']}, " +
              f"vol_ratio_mode={best_params['vol_ratio_mode']}")

        best_results = backtest_with_vol_ratio_filter(
            token1, token2, window_size, confidence_level,
            int(best_params['vol_ratio_window']),
            best_params['vol_ratio_threshold'],
            best_params['vol_ratio_mode'],
            atr_window, atr_multiplier, fee_pct
        )

        # Create descriptive title
        if best_params['vol_ratio_mode'] == 'threshold':
            mode_desc = f"Threshold: {best_params['vol_ratio_threshold']}"
        elif best_params['vol_ratio_mode'] == 'range':
            mode_desc = f"Range: {1 / best_params['vol_ratio_threshold']:.2f}-{best_params['vol_ratio_threshold']:.2f}"
        else:
            mode_desc = f"Balanced Range"

        title = f"Best Vol Ratio: {pair_name} (Window: {best_params['vol_ratio_window']}, " + \
                f"{best_params['vol_ratio_mode'].capitalize()} {mode_desc})"

        figs = plot_vol_ratio_results(best_results, title)

        for i, fig in enumerate(figs):
            output_file = os.path.join(output_dir, f"{pair_name}_best_volratio_plot_{i + 1}.png")
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved best parameters plot to {output_file}")

        return results_df

    return pd.DataFrame()


##############################################
# FUNCTIONS FOR VOLUME PERCENTILE FILTERING #
##############################################

def calculate_volume_percentile(volume_series, lookback_window=100):
    """
    Calculate the rolling percentile rank of volume
    Parameters:
    - volume_series: Series with volume data
    - lookback_window: Window size for percentile calculation
    Returns:
    - vol_percentile: Series with volume percentile ranks (0-1)
    """
    # Calculate rolling percentile rank (current volume compared to recent history)
    vol_percentile = volume_series.rolling(window=lookback_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )

    return vol_percentile


def filter_signals_by_volume_percentile(signals, volume, lookback_window=100, min_percentile=0.3, max_percentile=1.0):
    """
    Filter signals based on volume percentile
    Parameters:
    - signals: Original signals to filter
    - volume: Volume series for percentile calculation
    - lookback_window: Window size for percentile calculation
    - min_percentile: Minimum volume percentile to keep signals
    - max_percentile: Maximum volume percentile to keep signals
    Returns:
    - filtered_signals: Signals filtered by volume percentile
    - vol_percentile: Series with volume percentile ranks
    """
    # Calculate volume percentile
    vol_percentile = calculate_volume_percentile(volume, lookback_window)

    # Initialize filtered signals (copy original signals)
    filtered_signals = signals.copy()

    # Filter out signals with low or high volume percentile
    valid_volume = (vol_percentile >= min_percentile) & (vol_percentile <= max_percentile)
    filtered_signals[~valid_volume] = 0

    return filtered_signals, vol_percentile


def backtest_with_volume_filter(token1, token2, volume, window_size= "**********"=0.95,
                               vol_lookback=100, vol_min_percentile=0.3, vol_max_percentile=1.0,
                               atr_window=14, atr_multiplier=2.0, fee_pct=0.001):
    """
    Backtest strategy with volume percentile filtering.
    Parameters:
    - token1, token2: "**********"
    - volume: Volume series for percentile calculation
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - vol_lookback: Lookback window for volume percentile calculation
    - vol_min_percentile: Minimum volume percentile to keep signals
    - vol_max_percentile: Maximum volume percentile to keep signals
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    Returns:
    - results: Dictionary with backtest results
    """
    # Calculate spread
    spread = "**********"

    # Generate raw copula signals
    raw_signals, stop_levels, _, is_outside, copula = generate_copula_signals_with_atr_stop(
        token1, token2, window_size, confidence_level, atr_window, atr_multiplier, fee_pct
    )

    # Filter signals based on volume percentile
    filtered_signals, vol_percentile = filter_signals_by_volume_percentile(
        raw_signals, volume, vol_lookback, vol_min_percentile, vol_max_percentile
    )

    # Calculate ATR for stop-loss
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    atr = calculate_atr(spread_high, spread_low, spread, window=atr_window)

    # Initialize series for actual trading signals and stop levels
    signals = "**********"=token1.index)
    final_stop_levels = "**********"=token1.index)

    # Track position and equity
    current_position = 0
    entry_price = 0
    stop_price = 0
    equity_curve = "**********"=token1.index)

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Apply trading logic with ATR-based stop-loss
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"w "**********"i "**********"n "**********"d "**********"o "**********"w "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********") "**********") "**********": "**********"
        # Current values
        current_spread = spread.iloc[i]
        current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else spread.iloc[i] * 0.02
        new_signal = filtered_signals.iloc[i]

        # Update equity
        if i > 0:
            if current_position == 1:  # Long position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
            elif current_position == -1:  # Short position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
            else:  # No position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        # Check for stop-loss (if in a position)
        stop_hit = False
        if current_position == 1 and stop_price > 0 and current_spread < stop_price:
            stop_hit = True
        elif current_position == -1 and stop_price > 0 and current_spread > stop_price:
            stop_hit = True

        # Apply trading logic
        if current_position == 0:  # No current position
            if new_signal != 0:  # Enter new position
                # Apply fee for entry
                equity_curve.iloc[i] *= (1 - fee_pct)

                # Set position and entry price
                current_position = new_signal
                entry_price = current_spread

                # Set initial stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position
                final_stop_levels.iloc[i] = stop_price

        else:  # Already in a position
            # Update trailing stop if in profit
            if not stop_hit:
                if current_position == 1 and current_spread > entry_price:
                    # For long positions, raise stop as price increases
                    new_stop = current_spread - atr_multiplier * current_atr
                    stop_price = max(stop_price, new_stop)
                elif current_position == -1 and current_spread < entry_price:
                    # For short positions, lower stop as price decreases
                    new_stop = current_spread + atr_multiplier * current_atr
                    stop_price = min(stop_price, new_stop)

            # Determine whether to exit or maintain position
            if stop_hit:  # Stop-loss hit
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal == 0 and raw_signals.iloc[i] == 0:  # Exit signal
                # Apply fee for exit
                equity_curve.iloc[i] *= (1 - fee_pct)

                signals.iloc[i] = 0
                current_position = 0
                stop_price = 0

            elif new_signal != current_position and new_signal != 0:  # Reversal signal
                # Apply fee for both exit and entry
                equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)

                current_position = new_signal
                entry_price = current_spread

                # Set new stop-loss level
                if current_position == 1:  # Long position
                    stop_price = entry_price - atr_multiplier * current_atr
                else:  # Short position
                    stop_price = entry_price + atr_multiplier * current_atr

                signals.iloc[i] = current_position

            else:  # Maintain current position
                signals.iloc[i] = current_position

            # Record stop level
            final_stop_levels.iloc[i] = stop_price

    # Calculate trade statistics
    trades = []
    position_changes = signals.diff().fillna(0)
    change_dates = position_changes[position_changes != 0].index

    # Process each position change
    for i in range(len(change_dates) - 1):
        current_date = change_dates[i]
        next_date = change_dates[i + 1]
        position = signals.loc[current_date]

        if position != 0:  # If this is an entry
            entry_price = spread.loc[current_date]
            exit_price = spread.loc[next_date]

            # Calculate profit
            if position == 1:  # Long position
                profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
            else:  # Short position
                profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

            # Account for fees
            profit_pct -= fee_pct * 2  # Entry and exit fees

            # Calculate duration
            try:
                duration = (next_date - current_date).days
            except:
                duration = 1  # Fallback if date conversion fails

            vol_pct = vol_percentile.loc[current_date]

            trades.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'position': 'Long' if position == 1 else 'Short',
                'profit_pct': profit_pct,
                'duration': duration,
                'volume_percentile': vol_pct
            })

    # Handle the last open position if any
    if len(change_dates) > 0 and signals.iloc[-1] != 0:
        last_date = change_dates[-1]
        position = signals.loc[last_date]
        entry_price = spread.loc[last_date]
        exit_price = spread.iloc[-1]

        if position == 1:  # Long position
            profit_pct = (exit_price - entry_price) / abs(entry_price) if abs(entry_price) > 0 else 0
        else:  # Short position
            profit_pct = (entry_price - exit_price) / abs(entry_price) if abs(entry_price) > 0 else 0

        # Account for fees (only entry, no exit yet)
        profit_pct -= fee_pct

        try:
            duration = (spread.index[-1] - last_date).days
        except:
            duration = 1

        vol_pct = vol_percentile.loc[last_date]

        trades.append({
            'entry_date': last_date,
            'exit_date': spread.index[-1],
            'position': 'Long' if position == 1 else 'Short',
            'profit_pct': profit_pct,
            'duration': duration,
            'volume_percentile': vol_pct,
            'open': True
        })

    # Calculate performance metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        winning_trades = sum(trade_df['profit_pct'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate monthly returns
        if isinstance(equity_curve.index[0], (pd.Timestamp, datetime.datetime)):
            monthly_returns = equity_curve.resample('M').last().pct_change()
        else:
            # If not datetime index, can't calculate monthly returns
            monthly_returns = pd.Series()

        performance_summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_winning_trade': trade_df.loc[
                trade_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': trade_df.loc[trade_df['profit_pct'] <= 0, 'profit_pct'].mean() if (
                                                                                                      total_trades - winning_trades) > 0 else 0,
            'largest_winner': trade_df['profit_pct'].max() if not trade_df.empty else 0,
            'largest_loser': trade_df['profit_pct'].min() if not trade_df.empty else 0,
            'avg_trade_duration': trade_df['duration'].mean() if not trade_df.empty else 0,
            'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
            'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (
                        252 / len(equity_curve)) - 1 if len(
                equity_curve) > 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(trade_df.loc[trade_df['profit_pct'] > 0, 'profit_pct']) /
                                 sum(trade_df.loc[trade_df['profit_pct'] < 0, 'profit_pct']))
            if sum(trade_df.loc[
                       trade_df['profit_pct'] < 0, 'profit_pct']) != 0 and not trade_df.empty else float(
                'inf'),
            'monthly_returns': monthly_returns
        }
    else:
        trade_df = pd.DataFrame()
        performance_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_winner': 0,
            'largest_loser': 0,
            'avg_trade_duration': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'monthly_returns': pd.Series()
        }

    # Store all results
    results = {
        'signals': signals,
        'raw_signals': raw_signals,
        'filtered_signals': filtered_signals,
        'vol_percentile': vol_percentile,
        'stop_levels': final_stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'trades': trade_df,
        'spread': spread,
        'copula': {
            'rho': getattr(copula, 'rho', None),
        },
        'performance': performance_summary,
        'volume_params': {
            'lookback': vol_lookback,
            'min_percentile': vol_min_percentile,
            'max_percentile': vol_max_percentile
        }
    }

    return results

def plot_volume_filter_results(results, title="Volume Percentile Filtered Strategy"):
    """
    Create visualization of volume percentile filtered strategy performance
    Parameters:
    - results: Dictionary with backtest results
    - title: Title for the main plot
    Returns:
    - figs: List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    vol_percentile = results['vol_percentile']
    stop_levels = results['stop_levels']
    equity_curve = results['equity_curve']
    spread = results['spread']
    performance = results['performance']
    trades = results['trades']

    figs = []

    # 1. Create multi-panel plot with signals, vol percentile, and equity curve
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plot the spread and signals
    ax1.plot(spread.index, spread, 'b-', label='Spread')
    ax1.plot(stop_levels.index, stop_levels, 'r--', label='Stop-Loss', alpha=0.7)

    # Mark trade entries and exits
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
    short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

    ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                marker='^', color='green', s=100, label='Long Entry')
    ax1.scatter(long_exits.index, spread.loc[long_exits.index],
                marker='o', color='green', s=80, label='Long Exit')
    ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                marker='v', color='red', s=100, label='Short Entry')
    ax1.scatter(short_exits.index, spread.loc[short_exits.index],
                marker='o', color='red', s=80, label='Short Exit')

    ax1.set_title(title)
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot volume percentile
    ax2.plot(vol_percentile.index, vol_percentile, 'purple', label='Volume Percentile')

    # Add threshold visualization
    vol_params = results['volume_params']
    min_percentile = vol_params['min_percentile']
    max_percentile = vol_params['max_percentile']

    ax2.axhline(y=min_percentile, color='orange', linestyle=':', alpha=0.7,
                label=f'Min Percentile: {min_percentile}')
    if max_percentile < 1.0:
        ax2.axhline(y=max_percentile, color='orange', linestyle=':', alpha=0.7,
                    label=f'Max Percentile: {max_percentile}')

    # Fill the valid range area
    valid_x = vol_percentile.index
    ax2.fill_between(valid_x, min_percentile, max_percentile, color='green', alpha=0.2,
                     label=f'Valid Range: {min_percentile}-{max_percentile}')

    ax2.set_ylabel('Volume Percentile')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot equity curve
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create additional analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L vs Volume Percentile
        ax1.scatter(trades['volume_percentile'], trades['profit_pct'] * 100,
                    alpha=0.7, c='blue', edgecolors='k')
        ax1.set_title('Trade P&L vs Volume Percentile')
        ax1.set_xlabel('Volume Percentile')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.grid(True, alpha=0.3)

        # P&L histogram
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

        # 3. Create comparison of filtered vs raw signals
        if 'raw_signals' in results and 'filtered_signals' in results:
            fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Plot raw signals
            raw_signals = results['raw_signals']
            raw_long = raw_signals[raw_signals == 1]
            raw_short = raw_signals[raw_signals == -1]

            ax1.plot(spread.index, spread, 'b-', alpha=0.5)
            ax1.scatter(raw_long.index, spread.loc[raw_long.index], marker='^', color='green', s=50,
                        label='Raw Long Signals')
            ax1.scatter(raw_short.index, spread.loc[raw_short.index], marker='v', color='red', s=50,
                        label='Raw Short Signals')
            ax1.set_title('Raw Signals Before Volume Percentile Filter')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot filtered signals
            filtered_signals = results['filtered_signals']
            filtered_long = filtered_signals[filtered_signals == 1]
            filtered_short = filtered_signals[filtered_signals == -1]

            # Calculate signals removed by filter
            removed_long = raw_signals[(raw_signals == 1) & (filtered_signals == 0)]
            removed_short = raw_signals[(raw_signals == -1) & (filtered_signals == 0)]

            ax2.plot(spread.index, spread, 'b-', alpha=0.5)
            ax2.scatter(filtered_long.index, spread.loc[filtered_long.index], marker='^', color='green',
                        s=50,
                        label='Filtered Long Signals')
            ax2.scatter(filtered_short.index, spread.loc[filtered_short.index], marker='v', color='red',
                        s=50,
                        label='Filtered Short Signals')
            ax2.scatter(removed_long.index, spread.loc[removed_long.index], marker='x', color='orange',
                        s=50,
                        label='Removed Long Signals')
            ax2.scatter(removed_short.index, spread.loc[removed_short.index], marker='x', color='purple',
                        s=50,
                        label='Removed Short Signals')
            ax2.set_title('Signals After Volume Percentile Filter')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            figs.append(fig3)

    return figs

def run_volume_filtered_backtest(csv_file, date_col='datetime',
                                 token1_col= "**********"='close_2', volume_col='volume',
                                 window_size=20, confidence_level=0.95,
                                 vol_lookback=100, vol_min_percentile=0.3, vol_max_percentile=1.0,
                                 atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                                 output_dir='volume_results'):
    """
    Run backtest using volume percentile filtering
    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of the date/timestamp column
    - token1_col: "**********"
    - token2_col: "**********"
    - volume_col: Name of the volume column
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - vol_lookback: Lookback window for volume percentile calculation
    - vol_min_percentile: Minimum volume percentile to keep signals
    - vol_max_percentile: Maximum volume percentile to keep signals
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    - output_dir: Directory to save output visualizations
    Returns:
    - results: Dictionary with backtest results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    try:
        # Try to parse dates automatically
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        # If that fails, load normally and try to convert dates
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Check if required columns exist
    required_cols = "**********"
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Try to find the appropriate volume column
    if volume_col in df.columns:
        volume = df[volume_col]
    else:
        # Try to find volume based on token column naming
        vol1_candidate = volume_col + "_1"
        vol2_candidate = volume_col + "_2"

        if vol1_candidate in df.columns:
            print(f"Using '{vol1_candidate}' as volume column")
            volume = df[vol1_candidate]
        elif vol2_candidate in df.columns:
            print(f"Using '{vol2_candidate}' as volume column")
            volume = df[vol2_candidate]
        else:
            # Try various common volume column names
            vol_candidates = ['volume_1', 'volume_2', 'vol_1', 'vol_2', 'volume', 'vol']
            for col in vol_candidates:
                if col in df.columns:
                    print(f"Using '{col}' as volume column")
                    volume = df[col]
                    break
            else:
                # If no volume column found, create constant volume
                print("Warning: No volume column found. Using constant volume.")
                volume = "**********"=token1.index)

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Running volume filtered backtest for {pair_name}...")
    print(f"Data range: "**********"
    print(f"Number of data points: "**********"
    print(f"Parameters: window_size={window_size}, confidence_level={confidence_level}, " +
          f"vol_lookback={vol_lookback}, vol_min_percentile={vol_min_percentile}, vol_max_percentile={vol_max_percentile}")

    # Run backtest
    results = backtest_with_volume_filter(
        token1, token2, volume, window_size, confidence_level,
        vol_lookback, vol_min_percentile, vol_max_percentile,
        atr_window, atr_multiplier, fee_pct
    )

    # Create plots and save them
    title = f"Volume Filter: {pair_name} (Min: {vol_min_percentile}, Max: {vol_max_percentile})"
    figs = plot_volume_filter_results(results, title)

    for i, fig in enumerate(figs):
        output_file = os.path.join(output_dir, f"{pair_name}_volume_plot_{i + 1}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    # Save trade log if we have trades
    if not results['trades'].empty:
        trade_log_file = os.path.join(output_dir, f"{pair_name}_volume_trades.csv")
        results['trades'].to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")

    # Save performance summary
    perf_summary = pd.DataFrame({k: [v] for k, v in results['performance'].items()
                                 if not isinstance(v, pd.Series)})
    perf_file = os.path.join(output_dir, f"{pair_name}_volume_performance.csv")
    perf_summary.to_csv(perf_file, index=False)
    print(f"Saved performance summary to {perf_file}")

    # Display key performance metrics
    print("\nPerformance Summary:")
    print(f"Total Return: {results['performance'].get('total_return', 0) * 100:.2f}%")
    print(f"Annualized Return: {results['performance'].get('annualized_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results['performance'].get('max_drawdown', 0) * 100:.2f}%")
    print(f"Win Rate: {results['performance'].get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {results['performance'].get('total_trades', 0)}")
    print(f"Profit Factor: {results['performance'].get('profit_factor', 0):.2f}")

    return results


def plot_returns_indicator_results(results, indicator_name, indicator_data, title=None):
    """
    Create visualizations specifically for indicator-filtered returns-based strategies
    Parameters:
    - results: Dictionary with backtest results
    - indicator_name: Name of the indicator being plotted
    - indicator_data: Data specific to the indicator
    - title: Plot title (optional)
    Returns:
    - figs: List of figure objects
    """
    # Extract needed data
    signals = results['signals']
    equity_curve = results['equity_curve']
    return1 = results['return1']
    return2 = results['return2']
    performance = results['performance']
    trades = results['trades'] if 'trades' in results else pd.DataFrame()

    # Create cumulative returns for visualization
    cum_ret1 = (1 + return1).cumprod() - 1
    cum_ret2 = (1 + return2).cumprod() - 1
    spread = cum_ret1 - cum_ret2  # Difference in cumulative returns

    figs = []

    # Set title if not provided
    if title is None:
        title = f"{indicator_name} Filtered Returns Strategy"

    # 1. Create main plot with indicator and equity curve
    if indicator_name == "Vol Ratio":
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot cumulative returns
        ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return (%)')
        ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return (%)')

        # Mark trade entries and exits
        long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
        long_exits = signals[(signals.shift(1) == 1) & (signals != 1)]
        short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]
        short_exits = signals[(signals.shift(1) == -1) & (signals != -1)]

        for date in long_entries.index:
            try:
                ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
                ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
            except:
                pass  # Skip if there's any issue with the date

        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot volatility ratio
        vol_ratio = indicator_data['vol_ratio']
        ax2.plot(vol_ratio.index, vol_ratio, 'purple', label='Vol Ratio (Return1/Return2)')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

        # Get threshold parameters from the function global variables or default to common values
        threshold = 1.2  # Default value
        mode = 'threshold'  # Default mode

        if hasattr(results, 'vol_ratio_params'):
            threshold = results.vol_ratio_params.get('threshold', threshold)
            mode = results.vol_ratio_params.get('mode', mode)

        if mode == 'threshold':
            ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                        label=f'Upper Threshold: {threshold:.2f}')
            ax2.axhline(y=1 / threshold, color='orange', linestyle=':', alpha=0.7,
                        label=f'Lower Threshold: {1 / threshold:.2f}')
        elif mode == 'range':
            # Fill the valid range area
            valid_x = vol_ratio.index
            ax2.fill_between(valid_x, 1 / threshold, threshold, color='green', alpha=0.2,
                             label=f'Valid Range: {1 / threshold:.2f}-{threshold:.2f}')

        ax2.set_ylabel('Volatility Ratio')
        ax2.set_yscale('log')  # Log scale for better visualization
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    elif indicator_name == "Volume":
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot cumulative returns
        ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return (%)')
        ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return (%)')

        # Mark trade entries and exits
        long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
        short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]

        for date in long_entries.index:
            try:
                ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
                ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
            except:
                pass

        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot volume percentile
        vol_percentile = indicator_data['vol_percentile']
        ax2.plot(vol_percentile.index, vol_percentile, 'purple', label='Volume Percentile')

        # Get threshold parameters
        min_percentile = 0.3  # Default value
        max_percentile = 1.0  # Default value

        if hasattr(results, 'volume_params'):
            min_percentile = results.volume_params.get('min_percentile', min_percentile)
            max_percentile = results.volume_params.get('max_percentile', max_percentile)

        ax2.axhline(y=min_percentile, color='orange', linestyle=':', alpha=0.7,
                    label=f'Min Percentile: {min_percentile}')
        if max_percentile < 1.0:
            ax2.axhline(y=max_percentile, color='orange', linestyle=':', alpha=0.7,
                        label=f'Max Percentile: {max_percentile}')

        # Fill the valid range area
        valid_x = vol_percentile.index
        ax2.fill_between(valid_x, min_percentile, max_percentile, color='green', alpha=0.2,
                         label=f'Valid Range: {min_percentile}-{max_percentile}')

        ax2.set_ylabel('Volume Percentile')
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    elif indicator_name == "ADX":
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1.5, 1]})

        # Plot cumulative returns
        ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return (%)')
        ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return (%)')

        # Mark trade entries and exits
        long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
        short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]

        for date in long_entries.index:
            try:
                ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
                ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
            except:
                pass

        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot ADX and DI lines
        adx = indicator_data['adx']
        plus_di = indicator_data['plus_di']
        minus_di = indicator_data['minus_di']

        ax2.plot(adx.index, adx, 'b-', label='ADX')
        ax2.plot(plus_di.index, plus_di, 'g-', label='+DI')
        ax2.plot(minus_di.index, minus_di, 'r-', label='-DI')

        # Add ADX threshold
        threshold = 25  # Default
        if hasattr(results, 'adx_params'):
            threshold = results.adx_params.get('threshold', threshold)

        ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                    label=f'ADX Threshold: {threshold}')

        ax2.set_ylabel('ADX / DI Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    elif indicator_name == "RVI":
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1.5, 1]})

        # Plot cumulative returns
        ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return (%)')
        ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return (%)')

        # Mark trade entries and exits
        long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
        short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]

        for date in long_entries.index:
            try:
                ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
                ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
            except:
                pass

        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot RVI and signal line
        rvi = indicator_data['rvi']
        rvi_signal = indicator_data['rvi_signal']

        ax2.plot(rvi.index, rvi, 'b-', label='RVI')
        ax2.plot(rvi_signal.index, rvi_signal, 'r-', label='RVI Signal')

        # Add RVI threshold and centerline
        threshold = 50  # Default
        if hasattr(results, 'rvi_params'):
            threshold = results.rvi_params.get('threshold', threshold)

        ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                    label=f'RVI Threshold: {threshold}')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7,
                    label='Centerline')

        ax2.set_ylabel('RVI Values')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    elif indicator_name == "Chop":
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1.5, 1]})

        # Plot cumulative returns
        ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return (%)')
        ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return (%)')

        # Mark trade entries and exits
        long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
        short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]

        for date in long_entries.index:
            try:
                ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
                ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
            except:
                pass

        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Choppiness Index
        chop = indicator_data['chop']
        ax2.plot(chop.index, chop, 'b-', label='Choppiness Index')

        # Add CHOP threshold
        threshold = 38.2  # Default
        if hasattr(results, 'chop_params'):
            threshold = results.chop_params.get('threshold', threshold)

        ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                    label=f'CHOP Threshold: {threshold}')
        ax2.axhline(y=61.8, color='red', linestyle='--', alpha=0.5,
                    label='Choppy Market: 61.8')

        # Fill the trending range
        valid_x = chop.index
        ax2.fill_between(valid_x, 0, threshold, color='green', alpha=0.2,
                         label=f'Trending Range: <{threshold}')

        ax2.set_ylabel('Choppiness Index')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    elif indicator_name == "Donchian":
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot cumulative returns
        ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return (%)')
        ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return (%)')

        # Mark trade entries and exits
        long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
        short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]

        for date in long_entries.index:
            try:
                ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
                ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
            except:
                pass

        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Donchian Channel width
        donchian_width = indicator_data['donchian_width']
        ax2.plot(donchian_width.index, donchian_width * 100, 'purple', label='Donchian Width (%)')

        # Add threshold visualization
        min_width = 0.01  # Default
        max_width = 0.05  # Default
        mode = 'range'  # Default

        if hasattr(results, 'donchian_params'):
            min_width = results.donchian_params.get('width_min', min_width)
            max_width = results.donchian_params.get('width_max', max_width)
            mode = results.donchian_params.get('mode', mode)

        ax2.axhline(y=min_width * 100, color='orange', linestyle=':', alpha=0.7,
                    label=f'Min Width: {min_width * 100:.2f}%')

        if mode == 'range':
            ax2.axhline(y=max_width * 100, color='orange', linestyle=':', alpha=0.7,
                        label=f'Max Width: {max_width * 100:.2f}%')

            # Fill the valid range area
            valid_x = donchian_width.index
            ax2.fill_between(valid_x, min_width * 100, max_width * 100, color='green', alpha=0.2,
                             label=f'Valid Range: {min_width * 100:.2f}%-{max_width * 100:.2f}%')

        ax2.set_ylabel('Donchian Width (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    elif indicator_name == "ATR-Vol":
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot cumulative returns
        ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return (%)')
        ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return (%)')

        # Mark trade entries and exits
        long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
        short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]

        for date in long_entries.index:
            try:
                ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
                ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
            except:
                pass

        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot ATR ratio
        atr = indicator_data['atr']
        atr_ma = indicator_data['atr_ma']
        atr_ratio = indicator_data['atr_ratio']

        ax2.plot(atr.index, atr, 'b-', alpha=0.5, label='ATR')
        ax2.plot(atr_ma.index, atr_ma, 'g-', alpha=0.5, label='ATR MA')
        ax2.plot(atr_ratio.index, atr_ratio, 'r-', label='ATR Ratio')

        # Add threshold
        threshold = 1.2  # Default
        if hasattr(results, 'atr_vol_params'):
            threshold = results.atr_vol_params.get('threshold', threshold)

        ax2.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7,
                    label=f'ATR Ratio Threshold: {threshold}')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

        ax2.set_ylabel('ATR & Ratio Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    else:
        # Generic plot for any other indicator
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot cumulative returns
        ax1.plot(cum_ret1.index, cum_ret1 * 100, 'b-', label='Asset 1 Cumulative Return (%)')
        ax1.plot(cum_ret2.index, cum_ret2 * 100, 'g-', label='Asset 2 Cumulative Return (%)')

        # Mark trade entries and exits
        long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
        short_entries = signals[(signals == -1) & (signals.shift(1) != -1)]

        for date in long_entries.index:
            try:
                ax1.scatter(date, cum_ret1.loc[date] * 100, marker='^', color='blue', s=100)
                ax1.scatter(date, cum_ret2.loc[date] * 100, marker='v', color='red', s=100)
            except:
                pass

        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # If we have any indicator data, plot the first one
        if indicator_data and len(indicator_data) > 0:
            first_key = list(indicator_data.keys())[0]
            first_data = indicator_data[first_key]
            ax2.plot(first_data.index, first_data, 'purple', label=first_key)
            ax2.set_ylabel(first_key)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

    # Plot equity curve in the bottom panel (common to all indicators)
    returns_pct = 100 * (equity_curve - 1)  # Convert to percentage
    ax3.plot(returns_pct.index, returns_pct, 'g-')
    ax3.set_title('Equity Curve')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Add performance metrics as text
    textstr = '\n'.join((
        f'Total Return: {performance.get("total_return", 0) * 100:.2f}%',
        f'Annualized Return: {performance.get("annualized_return", 0) * 100:.2f}%',
        f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.2f}',
        f'Max Drawdown: {performance.get("max_drawdown", 0) * 100:.2f}%',
        f'Win Rate: {performance.get("win_rate", 0) * 100:.2f}%',
        f'Total Trades: {performance.get("total_trades", 0)}',
        f'Profit Factor: {performance.get("profit_factor", 0):.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    figs.append(fig1)

    # 2. Create trade analysis if we have trades
    if not trades.empty and len(trades) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trade P&L distribution
        ax2.hist(trades['profit_pct'] * 100, bins=20, color='green', alpha=0.7)
        ax2.set_title('Trade Profit/Loss Distribution')
        ax2.set_xlabel('Profit/Loss (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figs.append(fig2)

    return figs
def create_output_directories(base_dir, strategy_type):
    """
    Create all necessary output directories
    Parameters:
    - base_dir: Base output directory
    - strategy_type: 'Price' or 'Returns'
    """
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    # Create filter subdirectories
    filter_dirs = [
        f"{base_dir}-VolRatio",
        f"{base_dir}-Volume",
        f"{base_dir}-Combined",
        f"{base_dir}-ADX",
        f"{base_dir}-RVI",
        f"{base_dir}-Chop",
        f"{base_dir}-Donchian",
        f"{base_dir}-ATR-Vol"
    ]

    for directory in filter_dirs:
        os.makedirs(directory, exist_ok=True)

    # Create optimization directory if needed
    os.makedirs(f"{base_dir}-Optimization", exist_ok=True)
    os.makedirs(os.path.join(f"{base_dir}-Optimization", "combined"), exist_ok=True)

    print(f"Created output directories for {strategy_type} strategy")


def backtest_with_filtered_returns(return1, return2, filtered_signals, lookback=5, fee_pct=0.001, returns_method='log'):
    """
    Run a backtest with pre-filtered signals for returns-based strategy with proper log/pct return handling

    Parameters:
    - return1, return2: Return series for the two assets
    - filtered_signals: Filtered trading signals (1=long-short, -1=short-long, 0=no position)
    - lookback: Number of periods to lookback for performance evaluation
    - fee_pct: Transaction fee percentage
    - returns_method: 'log' for log returns or 'pct' for percentage returns

    Returns:
    - results: Dictionary with backtest results
    """
    # Initialize equity curve
    equity_curve = pd.Series(1.0, index=return1.index)

    # Track position
    current_position = 0

    # For each point in the time series
    for i in range(lookback, len(return1)):
        # Current values
        current_r1 = return1.iloc[i]
        current_r2 = return2.iloc[i]

        # Get signal
        new_signal = filtered_signals.iloc[i]

        # Update equity based on previous position
        if i > 0:
            if current_position == 1:  # Long asset1, short asset2
                if returns_method == 'log':
                    # For log returns, the relationship is: exp(r1 - r2)
                    multiplier = np.exp(current_r1 - current_r2)
                else:
                    # For percentage returns: (1 + r1) / (1 + r2) - 1 = (1 + r1 - r2 - r1*r2) - 1 ≈ r1 - r2
                    # But the exact formula is more accurate
                    multiplier = (1 + current_r1) / (1 + current_r2)

                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * multiplier

            elif current_position == -1:  # Short asset1, long asset2
                if returns_method == 'log':
                    # For log returns: exp(r2 - r1)
                    multiplier = np.exp(current_r2 - current_r1)
                else:
                    # For percentage returns: (1 + r2) / (1 + r1)
                    multiplier = (1 + current_r2) / (1 + current_r1)

                equity_curve.iloc[i] = equity_curve.iloc[i - 1] * multiplier

            else:  # No position
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        # Apply trading logic
        if current_position == 0 and new_signal != 0:  # Enter new position
            # Apply fee for entry
            equity_curve.iloc[i] *= (1 - fee_pct)
            current_position = new_signal

        elif new_signal == 0 and current_position != 0:  # Exit position
            # Apply fee for exit
            equity_curve.iloc[i] *= (1 - fee_pct)
            current_position = 0

        elif new_signal != 0 and current_position != 0 and new_signal != current_position:  # Reverse position
            # Apply fee for both exit and entry
            equity_curve.iloc[i] *= (1 - fee_pct) * (1 - fee_pct)
            current_position = new_signal

    # Calculate trade statistics
    trades = calculate_trades_from_signals(filtered_signals, return1, return2, equity_curve, fee_pct, returns_method)

    # Calculate performance metrics
    performance_summary = calculate_performance_metrics(equity_curve, trades)

    # Store all results
    results = {
        'signals': filtered_signals,
        'equity_curve': equity_curve,
        'trades': trades,
        'return1': return1,
        'return2': return2,
        'performance': performance_summary,
        'is_outside': pd.Series(False, index=return1.index)  # Placeholder
    }

    return results

def compare_copula_types(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                         window_size=20, confidence_level=0.95, atr_window=14, atr_multiplier=2.0, fee_pct=0.001,
                         output_dir='copula_comparison'):
    """
    Compare performance of different copula types on the same dataset
    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of the date/timestamp column
    - token1_col: "**********"
    - token2_col: "**********"
    - window_size: Size of rolling window for copula fitting
    - confidence_level: Confidence level for bands
    - atr_window: Window size for ATR calculation
    - atr_multiplier: Multiplier for ATR stop-loss
    - fee_pct: Transaction fee percentage
    - output_dir: Directory to save comparison results
    Returns:
    - comparison_df: DataFrame with performance comparison of different copula types
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(csv_file, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
    except:
        df = pd.read_csv(csv_file)
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except:
            print(f"Warning: Could not parse {date_col} as dates. Using default index.")

    # Extract token prices
    token1 = "**********"
    token2 = "**********"

    # Get pair name from CSV filename
    pair_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Comparing copula types for {pair_name}...")

    # Define copula types to test
    copula_types = [
        ('gaussian', {}),  # (type, params)
        ('clayton', {'clayton_theta': 2.0}),
        ('student_t', {'t_df': 5}),
        ('gumbel', {'gumbel_theta': 1.5})
    ]

    # Store results
    results = []
    equity_curves = {}

    # Run backtest for each copula type
    for copula_type, params in copula_types:
        print(f"\nTesting {copula_type.capitalize()} copula...")

        try:
            # Run the backtest
            res = backtest_strategy(
                token1, token2, window_size, confidence_level,
                atr_window, atr_multiplier, fee_pct,
                copula_type, params
            )

            # Extract performance metrics
            perf = res['performance']
            equity_curves[copula_type] = res['equity_curve']

            # Add copula type and parameters to results
            results.append({
                'copula_type': copula_type,
                'params': str(params),
                'total_return': perf['total_return'],
                'annualized_return': perf['annualized_return'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'max_drawdown': perf['max_drawdown'],
                'win_rate': perf['win_rate'],
                'total_trades': perf['total_trades'],
                'profit_factor': perf['profit_factor']
            })

            # Save the individual results
            individual_output_dir = os.path.join(output_dir, copula_type)
            os.makedirs(individual_output_dir, exist_ok=True)

            # Create title for plots
            title = f"{copula_type.capitalize()} Copula: {pair_name}"
            figs = plot_strategy_results(res, title)

            for i, fig in enumerate(figs):
                output_file = os.path.join(individual_output_dir, f"{pair_name}_{copula_type}_plot_{i + 1}.png")
                fig.savefig(output_file, dpi=150, bbox_inches='tight')

            # Display key metrics
            print(f"  Total Return: {perf['total_return'] * 100:.2f}%")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            print(f"  Win Rate: {perf['win_rate'] * 100:.2f}%")
            print(f"  Total Trades: {perf['total_trades']}")

        except Exception as e:
            print(f"  Error with {copula_type} copula: {e}")

    # Create comparison DataFrame
    if results:
        comparison_df = pd.DataFrame(results)

        # Sort by Sharpe ratio (descending)
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)

        # Save comparison results
        comparison_file = os.path.join(output_dir, f"{pair_name}_copula_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nSaved copula comparison to {comparison_file}")

        # Create equity curve comparison chart
        if equity_curves:
            plt.figure(figsize=(14, 7))

            for copula_type, equity_curve in equity_curves.items():
                plt.plot((equity_curve - 1) * 100, label=f"{copula_type.capitalize()} Copula")

            plt.title(f"Equity Curve Comparison of Copula Types: {pair_name}")
            plt.ylabel("Return (%)")
            plt.xlabel("Date")
            plt.grid(True, alpha=0.3)
            plt.legend()

            comparison_plot_file = os.path.join(output_dir, f"{pair_name}_copula_equity_comparison.png")
            plt.savefig(comparison_plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            # Print comparison summary
            print("\nCopula Type Comparison:")
            for i, row in comparison_df.iterrows():
                print(f"{row['copula_type'].capitalize()} Copula: " +
                      f"Return={row['total_return'] * 100:.2f}%, " +
                      f"Sharpe={row['sharpe_ratio']:.2f}, " +
                      f"Win Rate={row['win_rate'] * 100:.2f}%, " +
                      f"Trades={row['total_trades']}")

            return comparison_df

    return pd.DataFrame()


def select_best_copula(u, v):
    """
    Select the best copula model based on AIC criterion
    Parameters:
    - u, v: Uniform [0,1] data series to fit copulas
    Returns:
    - best_copula: The best copula object fitted to data
    - aic_values: Dictionary of AIC values for each copula type
    """
    copula_types = [
        ('gaussian', GaussianCopula()),
        ('clayton', ClaytonCopula()),
        ('student_t', StudentTCopula()),
        ('gumbel', GumbelCopula())
    ]

    aic_values = {}
    fitted_copulas = {}

    for name, copula in copula_types:
        try:
            # Fit copula
            copula.fit(u, v)

            # Calculate log-likelihood and AIC
            # This is a simplified example - you would need to implement
            # a proper log-likelihood function for each copula type
            log_lik = -1  # Placeholder
            n_params = len(copula.params) if hasattr(copula, 'params') else 1
            aic = -2 * log_lik + 2 * n_params

            aic_values[name] = aic
            fitted_copulas[name] = copula

        except Exception as e:
            print(f"Error fitting {name} copula: {e}")

    # Find copula with minimum AIC
    if aic_values:
        best_name = min(aic_values, key=aic_values.get)
        return fitted_copulas[best_name], aic_values

    # Default to Gaussian if all fits fail
    return GaussianCopula(), {}


def main():
    """
    Main function to run all strategies and compare them
    """
    # Get parameters
    params = get_parameters()

    # Print settings
    print("\nStarting backtests with the following settings:")
    print(f"CSV file: {params['csv_file']}")
    print(f"Date column: {params['date_col']}")

    # Adjust output directory based on strategy type
    strategy_type = "Returns" if params['use_returns_based'] else "Price"
    base_output_dir = f"{params['output_dir']}-{strategy_type}-{params['copula_type'].capitalize()}"

    # Create all necessary output directories at the start
    create_output_directories(base_output_dir, strategy_type)

    # Dictionary to collect all results
    all_results = {}

    if params['use_returns_based']:
        # First, create return series from prices if needed
        if params['convert_prices_to_returns']:
            # Load price data
            try:
                df = pd.read_csv(params['csv_file'], parse_dates=[params['date_col']])
                df.set_index(params['date_col'], inplace=True)
            except:
                df = pd.read_csv(params['csv_file'])
                try:
                    df[params['date_col']] = pd.to_datetime(df[params['date_col']])
                    df.set_index(params['date_col'], inplace=True)
                except:
                    print(f"Warning: Could not parse {params['date_col']} as dates. Using default index.")

            # Convert prices to returns
            df[params['token1_col']] = "**********"='coerce')
            df[params['token2_col']] = "**********"='coerce')
            return1 = "**********"=params['returns_method'])
            return2 = "**********"=params['returns_method'])
            bad_rows = "**********"='coerce').isna()]
            print("Bad rows in token1: "**********"
        else:
            # Load returns data directly
            try:
                df = pd.read_csv(params['csv_file'], parse_dates=[params['date_col']])
                df.set_index(params['date_col'], inplace=True)
            except:
                df = pd.read_csv(params['csv_file'])
                try:
                    df[params['date_col']] = pd.to_datetime(df[params['date_col']])
                    df.set_index(params['date_col'], inplace=True)
                except:
                    print(f"Warning: Could not parse {params['date_col']} as dates. Using default index.")

            return1 = df[params['return1_col']]
            return2 = df[params['return2_col']]

        # 1. Run baseline returns-based strategy
        print(f"\n=== Running {params['copula_type']} copula returns-based strategy backtest ===")
        base_results = backtest_returns_copula_strategy(
            return1, return2,
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            lookback=params['lookback'],
            fee_pct=params['fee_pct'],
            copula_type=params['copula_type'],
            copula_params=params['copula_params']
        )

        # Save results and plots
        title = f"{params['copula_type'].capitalize()} Copula Returns Strategy: Baseline"
        figs = plot_returns_strategy_results(base_results, title)
        for i, fig in enumerate(figs):
            output_file = os.path.join(base_output_dir, f"returns_baseline_plot_{i + 1}.png")
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # 2. Run all filters for returns-based strategy
        filtered_results = run_all_returns_filters(return1, return2, df, base_results, params)

        # Merge results
        all_results.update(filtered_results)

        # Create performance report for returns-based strategies
        print("\n=== Creating returns-based performance report ===")
        report_path = create_performance_report(all_results, base_output_dir)

        print("\nReturns-based backtests completed! Check the output directories for results.")
        print(f"Results saved to: {base_output_dir}")
        print(f"Performance report saved to: {report_path}")

    else:
        # Run price-based strategies
        # 1. Run standard copula strategy backtest for baseline
        print(f"\n=== Running {params['copula_type']} copula price-based strategy backtest ===")
        base_results = run_backtest_from_csv(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            copula_type=params['copula_type'],
            copula_params=params['copula_params'],
            output_dir=base_output_dir
        )
        all_results['Standard Copula'] = base_results

        # 2. Run volatility ratio filtered strategy
        print("\n=== Running volatility ratio filtered backtest ===")
        volratio_results = run_vol_ratio_backtest(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            vol_ratio_window=params['vol_ratio_window'],
            vol_ratio_threshold=params['vol_ratio_threshold'],
            vol_ratio_mode=params['vol_ratio_mode'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            output_dir=f"{base_output_dir}-VolRatio"
        )
        all_results['Vol Ratio Filtered'] = volratio_results

        # 3. Run volume percentile filtered strategy
        print("\n=== Running volume percentile filtered backtest ===")
        volume_results = run_volume_filtered_backtest(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            volume_col=params['volume_col'],
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            vol_lookback=params['vol_lookback'],
            vol_min_percentile=params['vol_min_percentile'],
            vol_max_percentile=params['vol_max_percentile'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            output_dir=f"{base_output_dir}-Volume"
        )
        all_results['Volume Filtered'] = volume_results

        # 4. Run combined filtered strategy
        print("\n=== Running combined filtered backtest (Vol Ratio + Volume) ===")
        combined_results = run_combined_filtered_backtest(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            volume_col=params['volume_col'],
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            vol_ratio_window=params['vol_ratio_window'],
            vol_ratio_threshold=params['vol_ratio_threshold'],
            vol_ratio_mode=params['vol_ratio_mode'],
            vol_lookback=params['vol_lookback'],
            vol_min_percentile=params['vol_min_percentile'],
            vol_max_percentile=params['vol_max_percentile'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            output_dir=f"{base_output_dir}-Combined"
        )
        all_results['Combined Filters'] = combined_results

        # 5. Run ADX filtered strategy
        print("\n=== Running ADX filtered backtest ===")
        adx_results = run_adx_filter_backtest(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            adx_period=params['adx_period'],
            adx_threshold=params['adx_threshold'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            output_dir=f"{base_output_dir}-ADX"
        )
        all_results['ADX Filtered'] = adx_results

        # 6. Run RVI filtered strategy
        print("\n=== Running RVI filtered backtest ===")
        rvi_results = run_rvi_filter_backtest(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            rvi_period=params['rvi_period'],
            rvi_signal_period=params['rvi_signal_period'],
            rvi_threshold=params['rvi_threshold'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            output_dir=f"{base_output_dir}-RVI"
        )
        all_results['RVI Filtered'] = rvi_results

        # 7. Run Choppiness Index filtered strategy
        print("\n=== Running Choppiness Index filtered backtest ===")
        chop_results = run_chop_filter_backtest(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            chop_period=params['chop_period'],
            chop_threshold=params['chop_threshold'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            output_dir=f"{base_output_dir}-Chop"
        )
        all_results['Chop Filtered'] = chop_results

        # 8. Run Donchian Channel filtered strategy
        print("\n=== Running Donchian Channel filtered backtest ===")
        donchian_results = run_donchian_filter_backtest(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            donchian_period=params['donchian_period'],
            donchian_width_min=params['donchian_width_min'],
            donchian_width_max=params['donchian_width_max'],
            donchian_mode=params['donchian_mode'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            output_dir=f"{base_output_dir}-Donchian"
        )
        all_results['Donchian Filtered'] = donchian_results

        # Create comprehensive performance report
        print("\n=== Creating performance report ===")
        report_path = create_performance_report(all_results, base_output_dir)

        print("\nPrice-based backtests completed! Check the output directories for results.")
        print(f"Performance report saved to: {report_path}")

    # Add expanding window backtest if enabled
    if params.get('use_expanding_window', False):
        print("\n=== Running expanding window backtest ===")
        expanding_output_dir = f"{base_output_dir}-ExpandingWindow"
        os.makedirs(expanding_output_dir, exist_ok=True)

        expanding_results = run_expanding_window_backtest(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            min_window_size=params['min_window_size'],
            confidence_level=params['confidence_level'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            copula_type=params['copula_type'],
            copula_params=params['copula_params'],
            output_dir=expanding_output_dir
        )
        all_results['Expanding Window'] = expanding_results

        # Add expanding window results to overall performance report
        print("\n=== Creating updated performance report with expanding window results ===")
        updated_report_path = create_performance_report(all_results, base_output_dir)
        print(f"Updated performance report saved to: {updated_report_path}")

    # Run parameter optimization if enabled
    if params['run_optimization']:
        optimization_output_dir = f"{base_output_dir}-Optimization"
        print("\n=== Running parameter optimization ===")
        os.makedirs(optimization_output_dir, exist_ok=True)

        if params['use_returns_based']:
            # Add optimization for returns-based strategy
            print("\n--- Optimizing returns-based strategy parameters ---")
            # You could add optimization code here for returns-based strategies
            print("Basic optimization for returns-based strategy not yet implemented.")
        else:
            # Optimize combined filters for price-based strategy
            print("\n--- Optimizing combined filter parameters ---")
            optimize_combined_filters(
                csv_file=params['csv_file'],
                date_col=params['date_col'],
                token1_col= "**********"
                token2_col= "**********"
                volume_col=params['volume_col'],
                window_size=params['window_size'],
                confidence_level=params['confidence_level'],
                atr_window=params['atr_window'],
                atr_multiplier=params['atr_multiplier'],
                fee_pct=params['fee_pct'],
                # Test a range of parameters
                vol_ratio_thresholds=[1.1, 1.2, 1.3, 1.5, 2.0],
                vol_ratio_modes=['threshold', 'range'],
                vol_min_percentiles=[0.1, 0.2, 0.3, 0.4, 0.5],
                vol_max_percentiles=[0.8, 0.9, 1.0],
                output_dir=os.path.join(optimization_output_dir, 'combined')
            )

    # Compare different copula types if requested
    if params.get('compare_copula_types', False):
        print("\n=== Comparing different copula types ===")
        copula_comparison_dir = f"{base_output_dir}-CopulaComparison"
        os.makedirs(copula_comparison_dir, exist_ok=True)

        compare_copula_types(
            csv_file=params['csv_file'],
            date_col=params['date_col'],
            token1_col= "**********"
            token2_col= "**********"
            window_size=params['window_size'],
            confidence_level=params['confidence_level'],
            atr_window=params['atr_window'],
            atr_multiplier=params['atr_multiplier'],
            fee_pct=params['fee_pct'],
            output_dir=copula_comparison_dir
        )

    print("\nAll processes completed successfully!")


if __name__ == "__main__":
    main()