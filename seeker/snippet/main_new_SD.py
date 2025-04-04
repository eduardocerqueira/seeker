#date: 2025-04-04T16:55:17Z
#url: https://api.github.com/gists/dd7b20e4d2d306cb7bcf77fa93ac0a73
#owner: https://api.github.com/users/Clement1nes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import rankdata, norm, t, multivariate_t
from scipy import stats
from scipy.optimize import minimize
import os
import datetime
from tqdm import tqdm


def get_parameters():
    """
    Define default strategy parameters with option for user input
    (Now ensures the SD filter is always enabled)
    """
    default_params = {
        # Data parameters
        'csv_file': 'ALGOUSDT_XTZUSDT_1d_price_2020-03-04_to_2025-04-03.csv',  # CSV file path
        'date_col': 'timestamp',  # Date column name
        'token1_col': "**********"
        'token2_col': "**********"

        # Strategy parameters
        'window_size': 20,  # Window size for copula fitting
        'confidence_level': 0.90,  # Confidence level for bands
        'atr_window': 14,  # Window size for ATR calculation
        'atr_multiplier': 2.0,  # Multiplier for ATR stop-loss
        'fee_pct': 0.001,  # Trading fee percentage (0.001 = 0.1%)
        'copula_type': 'student_t',  # Type of copula to use: 'gaussian', 'student_t', 'clayton', 'gumbel', 'frank'

        # Copula-specific parameters
        'student_t_df': 5,  # Degrees of freedom for Student-t copula

        # ATR calculation basis
        'atr_on_equity': True,  # Whether to calculate ATR on equity curve (True) or spread (False)

        # Volatility filter parameters
        'vol_filter_enabled': False,  # Whether to use volatility filter
        'vol_window': 20,  # Window size for volatility calculation
        'vol_lookback': 100,  # Lookback window for percentile calculation
        'vol_min_percentile': 30,  # Minimum volatility percentile to take trades
        'vol_max_percentile': 75,  # Maximum volatility percentile to take trades

        # SD Volatility filter parameters - Always enabled now
        'sd_filter_enabled': True,  # Whether to use SD volatility filter (always True now)
        'sd_window': 20,  # Window size for SD calculation
        'sd_lookback_windows': 5,  # Number of previous windows to average for comparison
        'sd_threshold_ratio': 1.0,  # Ratio threshold for current SD vs average SD (1.0 = equal)

        # Output parameters
        'output_dir': 'dynamic_wfo_results',  # Directory to save results
    }

    # Ask if user wants to use default parameters or customize
    print("Default parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")

    choice = input("\nDo you want to use default parameters? (y/n): ").lower()

    if choice == 'y':
        return default_params

    # Allow user to customize parameters
    params = default_params.copy()

    print("\nEnter new values (press Enter to keep default):")

    # Data parameters
    csv_file = input(f"CSV file path [{params['csv_file']}]: ")
    if csv_file:
        params['csv_file'] = csv_file

    date_col = input(f"Date column name [{params['date_col']}]: ")
    if date_col:
        params['date_col'] = date_col

    token1_col = input(f"First token price column [{params['token1_col']}]: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********"_ "**********"c "**********"o "**********"l "**********": "**********"
        params['token1_col'] = "**********"

    token2_col = input(f"Second token price column [{params['token2_col']}]: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"2 "**********"_ "**********"c "**********"o "**********"l "**********": "**********"
        params['token2_col'] = "**********"

    # SD filter parameters can be customized, but enabled status is fixed
    print("\nSD filter is always enabled. You can customize its parameters:")

    sd_window = input(f"SD window size [{params['sd_window']}]: ")
    if sd_window:
        params['sd_window'] = int(sd_window)

    sd_lookback_windows = input(f"SD lookback windows [{params['sd_lookback_windows']}]: ")
    if sd_lookback_windows:
        params['sd_lookback_windows'] = int(sd_lookback_windows)

    sd_threshold_ratio = input(f"SD threshold ratio [{params['sd_threshold_ratio']}]: ")
    if sd_threshold_ratio:
        params['sd_threshold_ratio'] = float(sd_threshold_ratio)

    # Force SD filter to be enabled
    params['sd_filter_enabled'] = True

    # Other parameters can be handled by the dynamic WFO function

    print("\nParameters updated successfully.")
    return params


class BaseCopula:
    """Base class for all copula implementations"""

    def __init__(self):
        pass

    def fit(self, u, v):
        """Fit the copula model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")


class GaussianCopula(BaseCopula):
    def __init__(self):
        super().__init__()
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


class StudentTCopula(BaseCopula):
    def __init__(self, df=5):
        super().__init__()
        self.rho = None
        self.df = df  # Degrees of freedom
        self.cov_matrix = None

    def fit(self, u, v):
        """Fit the Student-t copula model"""
        # Remove any NaN values
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = u[valid_mask]
        v_clean = v[valid_mask]

        # Handle empty arrays
        if len(u_clean) < 2 or len(v_clean) < 2:
            self.rho = 0
            self.cov_matrix = np.array([[1, 0], [0, 1]])
            return self

        # Transform to t-distributed variables
        x = t.ppf(np.clip(u_clean, 0.001, 0.999), self.df)
        y = t.ppf(np.clip(v_clean, 0.001, 0.999), self.df)

        # Calculate correlation
        self.rho = np.corrcoef(x, y)[0, 1]
        if np.isnan(self.rho):
            self.rho = 0

        # Ensure correlation is valid
        self.rho = max(min(self.rho, 0.999), -0.999)
        self.cov_matrix = np.array([[1, self.rho], [self.rho, 1]])
        return self

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands for Student-t copula"""
        if np.isnan(u) or np.isnan(v) or u <= 0 or u >= 1 or v <= 0 or v >= 1:
            return False, 0.0, 0.0

        try:
            # Convert to t-space with clipping
            x = t.ppf(min(max(u, 0.001), 0.999), self.df)
            y = t.ppf(min(max(v, 0.001), 0.999), self.df)

            # Compute Mahalanobis distance
            point = np.array([x, y])
            inv_cov = np.linalg.inv(self.cov_matrix)
            mahalanobis_sq = point.dot(inv_cov).dot(point)

            # F-distribution threshold for given confidence level (for Student t)
            # The test statistic follows F(2, df) distribution
            f_threshold = stats.f.ppf(confidence_level, 2, self.df)
            threshold = 2 * (self.df - 1) * f_threshold / self.df

            # Check if outside bands
            is_outside = mahalanobis_sq > threshold

            # Calculate direction
            if abs(self.rho) < 1e-10:  # Near-zero correlation
                direction = x  # Just use the x coordinate
            else:
                expected_x = self.rho * y
                direction = x - expected_x

            return is_outside, np.sqrt(mahalanobis_sq), direction

        except Exception as e:
            print(f"Error in Student-t is_point_outside_bands: {e}")
            return False, 0.0, 0.0


class ClaytonCopula(BaseCopula):
    def __init__(self):
        super().__init__()
        self.theta = None  # Clayton copula parameter

    def clayton_cdf(self, u, v, theta):
        """Clayton copula CDF"""
        if theta < 1e-10:  # Near independence
            return u * v
        return max(0, (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta))

    def fit(self, u, v):
        """Fit the Clayton copula model"""
        # Remove any NaN values
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = u[valid_mask]
        v_clean = v[valid_mask]

        # Handle empty arrays
        if len(u_clean) < 2 or len(v_clean) < 2:
            self.theta = 0
            return self

        # Clip values to avoid numerical issues
        u_clean = np.clip(u_clean, 0.001, 0.999)
        v_clean = np.clip(v_clean, 0.001, 0.999)

        # Calculate Kendall's tau
        tau, _ = stats.kendalltau(u_clean, v_clean)
        if np.isnan(tau):
            self.theta = 0
        else:
            # Convert Kendall's tau to Clayton theta
            # For Clayton: tau = theta / (theta + 2)
            if tau <= 0:
                self.theta = 0
            else:
                self.theta = 2 * tau / (1 - tau)

        return self

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands for Clayton copula"""
        if np.isnan(u) or np.isnan(v) or u <= 0 or u >= 1 or v <= 0 or v >= 1:
            return False, 0.0, 0.0

        try:
            u = min(max(u, 0.001), 0.999)
            v = min(max(v, 0.001), 0.999)

            # For Clayton, use the deviation from the expected conditional distribution
            if self.theta < 1e-10:  # Near independence
                expected_v = v
                direction = u - 0.5  # Use u's deviation from center
            else:
                # Calculate conditional distribution: F(v|u)
                # For Clayton: F(v|u) = u^(-theta-1) * (u^(-theta) + v^(-theta) - 1)^(-1/theta-1) * v^(-theta-1)
                # Simplified measure: Calculate difference from median of conditional distribution

                # Convert to normal for measuring distance and direction
                x = norm.ppf(u)
                y = norm.ppf(v)

                # Use correlation-based approach for direction
                kendall_corr = self.theta / (self.theta + 2)  # Convert theta to correlation
                spearman_corr = 3 * kendall_corr / 2  # Approximate conversion

                # Use a similar approach to Gaussian for direction
                expected_x = spearman_corr * y
                direction = x - expected_x

            # Calculate a pseudo-distance based on probability
            joint_prob = self.clayton_cdf(u, v, self.theta)
            expected_prob = u * v  # Independence case

            # Calculate normalized deviation
            deviation = abs(joint_prob - expected_prob) / max(expected_prob, 1e-10)
            threshold = 1 - confidence_level

            # Determine if outside bands
            is_outside = deviation > threshold

            return is_outside, deviation, direction

        except Exception as e:
            print(f"Error in Clayton is_point_outside_bands: {e}")
            return False, 0.0, 0.0


class GumbelCopula(BaseCopula):
    def __init__(self):
        super().__init__()
        self.theta = None  # Gumbel copula parameter

    def gumbel_cdf(self, u, v, theta):
        """Gumbel copula CDF"""
        if theta <= 1 + 1e-10:  # Near independence
            return u * v
        return np.exp(-(((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)))

    def fit(self, u, v):
        """Fit the Gumbel copula model"""
        # Remove any NaN values
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = u[valid_mask]
        v_clean = v[valid_mask]

        # Handle empty arrays
        if len(u_clean) < 2 or len(v_clean) < 2:
            self.theta = 1  # Independence
            return self

        # Clip values to avoid numerical issues
        u_clean = np.clip(u_clean, 0.001, 0.999)
        v_clean = np.clip(v_clean, 0.001, 0.999)

        # Calculate Kendall's tau
        tau, _ = stats.kendalltau(u_clean, v_clean)
        if np.isnan(tau):
            self.theta = 1
        else:
            # Convert Kendall's tau to Gumbel theta
            # For Gumbel: tau = 1 - 1/theta
            if tau <= 0:
                self.theta = 1  # Independence
            else:
                self.theta = 1 / (1 - tau)

        return self

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands for Gumbel copula"""
        if np.isnan(u) or np.isnan(v) or u <= 0 or u >= 1 or v <= 0 or v >= 1:
            return False, 0.0, 0.0

        try:
            u = min(max(u, 0.001), 0.999)
            v = min(max(v, 0.001), 0.999)

            # Convert to normal for measuring distance and direction
            x = norm.ppf(u)
            y = norm.ppf(v)

            # For Gumbel, similar approach to Clayton
            if self.theta <= 1 + 1e-10:  # Near independence
                direction = x  # Just use the x coordinate
            else:
                # Use correlation-based approach for direction
                kendall_corr = 1 - 1 / self.theta  # Convert theta to correlation
                spearman_corr = 3 * kendall_corr / 2  # Approximate conversion

                # Use a similar approach to Gaussian for direction
                expected_x = spearman_corr * y
                direction = x - expected_x

            # Calculate a pseudo-distance based on probability
            joint_prob = self.gumbel_cdf(u, v, self.theta)
            expected_prob = u * v  # Independence case

            # Calculate normalized deviation
            deviation = abs(joint_prob - expected_prob) / max(expected_prob, 1e-10)
            threshold = 1 - confidence_level

            # Determine if outside bands
            is_outside = deviation > threshold

            return is_outside, deviation, direction

        except Exception as e:
            print(f"Error in Gumbel is_point_outside_bands: {e}")
            return False, 0.0, 0.0


class FrankCopula(BaseCopula):
    def __init__(self):
        super().__init__()
        self.theta = None  # Frank copula parameter

    def frank_cdf(self, u, v, theta):
        """Frank copula CDF"""
        if abs(theta) < 1e-10:  # Near independence
            return u * v
        return -1 / theta * np.log(1 + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1))

    def fit(self, u, v):
        """Fit the Frank copula model"""
        # Remove any NaN values
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        u_clean = u[valid_mask]
        v_clean = v[valid_mask]

        # Handle empty arrays
        if len(u_clean) < 2 or len(v_clean) < 2:
            self.theta = 0  # Independence
            return self

        # Clip values to avoid numerical issues
        u_clean = np.clip(u_clean, 0.001, 0.999)
        v_clean = np.clip(v_clean, 0.001, 0.999)

        # Calculate Kendall's tau
        tau, _ = stats.kendalltau(u_clean, v_clean)
        if np.isnan(tau):
            self.theta = 0
        else:
            # Frank copula has a more complex relationship with Kendall's tau
            # For small tau values, use approximation; otherwise, use numerical approach
            if abs(tau) < 0.1:
                self.theta = tau * 12  # Approximation for small tau
            else:
                # Use empirical relationship (approximation)
                if tau > 0:
                    self.theta = 15 * tau  # Positive association
                else:
                    self.theta = 15 * tau  # Negative association

        return self

    def is_point_outside_bands(self, u, v, confidence_level=0.95):
        """Check if a point (u,v) is outside the confidence bands for Frank copula"""
        if np.isnan(u) or np.isnan(v) or u <= 0 or u >= 1 or v <= 0 or v >= 1:
            return False, 0.0, 0.0

        try:
            u = min(max(u, 0.001), 0.999)
            v = min(max(v, 0.001), 0.999)

            # Convert to normal for measuring distance and direction
            x = norm.ppf(u)
            y = norm.ppf(v)

            # For Frank copula
            if abs(self.theta) < 1e-10:  # Near independence
                direction = x  # Just use the x coordinate
            else:
                # Use correlation-based approach for direction
                # For Frank, approximate Kendall's tau using debye function
                # Since we don't have Debye function easily accessible, use simplified relationship
                kendall_corr = self.theta / 10 if abs(self.theta) < 5 else np.sign(self.theta) * 0.5
                spearman_corr = 3 * kendall_corr / 2  # Approximate conversion

                # Use a similar approach to Gaussian for direction
                expected_x = spearman_corr * y
                direction = x - expected_x

            # Calculate a pseudo-distance based on probability
            joint_prob = self.frank_cdf(u, v, self.theta)
            expected_prob = u * v  # Independence case

            # Calculate normalized deviation
            deviation = abs(joint_prob - expected_prob) / max(expected_prob, 1e-10)
            threshold = 1 - confidence_level

            # Determine if outside bands
            is_outside = deviation > threshold

            return is_outside, deviation, direction

        except Exception as e:
            print(f"Error in Frank is_point_outside_bands: {e}")
            return False, 0.0, 0.0


def create_copula(copula_type, **kwargs):
    """Factory function to create the requested copula type"""
    if copula_type.lower() == 'gaussian':
        return GaussianCopula()
    elif copula_type.lower() == 'student_t':
        df = kwargs.get('student_t_df', 5)
        return StudentTCopula(df=df)
    elif copula_type.lower() == 'clayton':
        return ClaytonCopula()
    elif copula_type.lower() == 'gumbel':
        return GumbelCopula()
    elif copula_type.lower() == 'frank':
        return FrankCopula()
    else:
        print(f"Warning: Unknown copula type '{copula_type}'. Using Gaussian copula instead.")
        return GaussianCopula()


def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range (ATR)"""
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']  # Current high - current low
    df['tr2'] = abs(df['high'] - df['prev_close'])  # Current high - previous close
    df['tr3'] = abs(df['low'] - df['prev_close'])  # Current low - previous close
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=window).mean()
    return df['atr']


 "**********"d "**********"e "**********"f "**********"  "**********"f "**********"i "**********"n "**********"d "**********"_ "**********"b "**********"e "**********"s "**********"t "**********"_ "**********"f "**********"i "**********"t "**********"_ "**********"c "**********"o "**********"p "**********"u "**********"l "**********"a "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"2 "**********") "**********": "**********"
    """
    Find the copula type that best fits the joint distribution of two token price series.
    Parameters:
    - token1, token2: "**********"
    Returns:
    - best_copula_type: String indicating the best-fitting copula
    """
    # Convert to uniform [0,1] using empirical CDF
    u_train = "**********"
    v_train = "**********"

    # Test each copula type
    copula_types = ['gaussian', 'student_t', 'clayton', 'gumbel', 'frank']
    copula_fits = {}
    fitted_copulas = {}

    for copula_type in copula_types:
        # Set copula specific parameters
        copula_params = {}
        if copula_type == 'student_t':
            copula_params['student_t_df'] = 5

        # Create and fit copula
        copula = create_copula(copula_type, **copula_params)

        # Fit copula to data
        copula.fit(u_train, v_train)

        # Store fitted copula
        fitted_copulas[copula_type] = copula

        # Calculate log-likelihood or AIC for this copula fit
        log_likelihood = 0

        # For Gaussian copula
        if copula_type == 'gaussian':
            # Transform to standard normal
            x = norm.ppf(np.clip(u_train, 0.001, 0.999))
            y = norm.ppf(np.clip(v_train, 0.001, 0.999))

            # Calculate log-likelihood
            rho = copula.rho
            rho_sq = rho ** 2
            for idx in range(len(x)):
                # Bivariate normal density (logarithm)
                z = (x[idx] ** 2 - 2 * rho * x[idx] * y[idx] + y[idx] ** 2) / (1 - rho_sq)
                log_likelihood += -0.5 * (np.log(1 - rho_sq) + z)

        # For Student-t copula
        elif copula_type == 'student_t':
            # Transform to t-distributed variables
            df_val = copula.df
            x = t.ppf(np.clip(u_train, 0.001, 0.999), df_val)
            y = t.ppf(np.clip(v_train, 0.001, 0.999), df_val)

            # Calculate log-likelihood - approximation based on correlation
            rho = copula.rho
            log_likelihood = -len(x) * np.log(1 - rho ** 2)  # Simplified measure

        # For Clayton, Gumbel, and Frank copulas - use pseudo-likelihood approach
        elif copula_type in ['clayton', 'gumbel', 'frank']:
            # Use the empirical observation that a better fit copula results in
            # data points being more evenly distributed around the copula contours

            # For each (u,v) point, check how well it fits the copula model
            outside_count = 0
            total_distance = 0

            for idx in range(len(u_train)):
                if idx % 10 == 0:  # Skip some points for efficiency
                    is_outside, distance, _ = copula.is_point_outside_bands(u_train[idx], v_train[idx], 0.5)
                    total_distance += distance
                    if is_outside:
                        outside_count += 1

            # A better fit should have a more balanced distribution of points inside/outside
            # and lower total distance from expected values
            balance_score = abs(outside_count / (len(u_train) / 10) - 0.5)  # 0.5 is ideal for 50% confidence band
            log_likelihood = -balance_score - total_distance / len(u_train)  # Higher is better

        # Store fit metrics
        aic = -2 * log_likelihood + 2  # Simple AIC calculation (1 parameter for all except t which has 2)
        if copula_type == 'student_t':
            aic += 2  # Student-t has extra parameter (df)

        bic = -2 * log_likelihood + np.log(len(u_train))  # BIC calculation
        if copula_type == 'student_t':
            bic += np.log(len(u_train))  # Student-t has extra parameter

        copula_fits[copula_type] = {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic
        }

    # Determine best copula type based on AIC (lower is better)
    best_copula_type = min(copula_fits.items(), key=lambda x: x[1]['aic'])[0]

    return best_copula_type


def dynamic_walk_forward_optimization(csv_file, date_col= "**********"='close_1', token2_col='close_2',
                                      param_ranges=None, output_dir='dynamic_wfo_results',
                                      optimization_goal='sharpe_ratio', min_trades=10,
                                      copula_window_days=365, update_frequency_days=30,
                                      num_cycles=4):
    """
    Dynamic walk-forward optimization that uses a rolling window for copula determination
    and separates parameter optimization from out-of-sample testing:

    - Uses a rolling window (default 365 days) to determine the best copula type
    - Updates the copula model regularly (default every 30 days) during testing
    - Optimizes strategy parameters in-sample
    - Tests the optimized parameters out-of-sample while dynamically updating the copula type

    Parameters:
    - csv_file: Path to CSV file
    - date_col: Name of date column
    - token1_col, token2_col: "**********"
    - param_ranges: Dictionary with parameter ranges to test
    - output_dir: Directory to save results
    - optimization_goal: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
    - min_trades: Minimum number of trades required for a valid strategy
    - copula_window_days: Number of days to use for determining copula type
    - update_frequency_days: How often to update the copula model (in days)
    - num_cycles: Number of cycles for the walk-forward analysis

    Returns:
    - Dictionary with walk-forward results
    """
    # Load data with verbose debugging
    print(f"Loading data from: {csv_file}")
    try:
        # Make sure we're loading as a DataFrame
        df = pd.read_csv(csv_file.strip())  # Strip to remove any trailing spaces

        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Set date as index
        df.set_index(date_col, inplace=True)

        # Debug info
        print(f"Data loaded successfully with {len(df)} rows")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        # Create year column for reference
        df['year'] = df.index.year
        print(f"Years in data: {sorted(df.index.year.unique())}")

        # Print data counts per year
        yearly_counts = df.groupby('year').size()
        print(f"Number of data points per year:")
        for year, count in yearly_counts.items():
            print(f"  {year}: {count} data points")

    except Exception as e:
        print(f"Error during data loading: {e}")
        raise

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set default parameter ranges if not provided
    if param_ranges is None:
        param_ranges = {
            'window_size': [20, 40, 60],
            'confidence_level': [0.90, 0.95, 0.99],
            'atr_multiplier': [1.5, 2.0, 2.5],
            'atr_on_equity': [False, True]
        }

    # Calculate segment sizes based on date range
    total_date_range = df.index.max() - df.index.min()
    days_per_segment = total_date_range.days // (num_cycles * 2)  # Each cycle has 2 segments (IS and OOS)

    print(f"Total date range: {total_date_range.days} days")
    print(f"Using {num_cycles} cycles with approximately {days_per_segment} days per segment")

    # Ensure we have at least copula_window_days of data before the first segment
    min_start_date = df.index.min() + pd.Timedelta(days=copula_window_days)

    # Create segment boundaries
    segment_boundaries = []
    current_date = min_start_date

    for i in range(num_cycles * 2 + 1):  # +1 to include the end boundary
        if current_date > df.index.max():
            current_date = df.index.max()

        segment_boundaries.append(current_date)

        if current_date == df.index.max():
            break

        current_date = current_date + pd.Timedelta(days=days_per_segment)

    # Preview all segment boundaries
    print("\nSegment boundaries:")
    for i, boundary in enumerate(segment_boundaries):
        print(f"  Boundary {i + 1}: {boundary.strftime('%Y-%m-%d')}")

    # Initialize results dictionary
    results = {}

    # For each cycle
    for cycle in range(num_cycles):
        print(f"\n{'-' * 80}")
        print(f"PROCESSING CYCLE {cycle + 1}/{num_cycles}")
        print(f"{'-' * 80}")

        # Get segment boundaries for this cycle
        is_start = segment_boundaries[cycle * 2]
        is_end = segment_boundaries[cycle * 2 + 1]
        oos_start = segment_boundaries[cycle * 2 + 1]
        oos_end = segment_boundaries[cycle * 2 + 2] if cycle * 2 + 2 < len(segment_boundaries) else df.index.max()

        # Make sure we don't go beyond available data
        if is_start >= df.index.max():
            print(f"Warning: Cycle {cycle + 1} would start after available data ends. Stopping cycles.")
            break

        # Extract data for each segment using date ranges
        is_data = df[(df.index >= is_start) & (df.index < is_end)]
        oos_data = df[(df.index >= oos_start) & (df.index <= oos_end)]

        # Skip if any segment has insufficient data
        if len(is_data) < 30 or len(oos_data) < 30:
            print(f"  Skipping cycle {cycle + 1}: Insufficient data in one or more segments")
            continue

        # Get date ranges for easier reference
        is_range = f"{is_data.index.min().strftime('%Y-%m-%d')} to {is_data.index.max().strftime('%Y-%m-%d')}"
        oos_range = f"{oos_data.index.min().strftime('%Y-%m-%d')} to {oos_data.index.max().strftime('%Y-%m-%d')}"

        print(f"  In-Sample: {is_range} ({len(is_data)} data points)")
        print(f"  Out-of-Sample: {oos_range} ({len(oos_data)} data points)")

        # Extract token prices for IS data
        is_token1 = "**********"
        is_token2 = "**********"

        # Determine the optimal parameters using the in-sample data
        # For each date in IS, we'll use a rolling window of copula_window_days prior to that date
        print(f"  Optimizing parameters on in-sample data with dynamic copula selection...")
        best_params, is_results = optimize_parameters_dynamic_copula(
            df, is_start, is_end,
            token1_col, token2_col,
            copula_window_days, update_frequency_days,
            param_ranges, optimization_goal, min_trades
        )

        # If optimization fails, skip this cycle
        if best_params is None:
            print(f"  Skipping cycle {cycle + 1}: Parameter optimization failed")
            continue

        print(f"  Best parameters found:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")

        # Print In-Sample performance
        is_perf = is_results['performance']
        print(f"\n  In-Sample Performance:")
        print(f"    Total Return: {is_perf['total_return'] * 100:.2f}%")
        print(f"    Sharpe Ratio: {is_perf['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: {is_perf['max_drawdown'] * 100:.2f}%")
        print(f"    Win Rate: {is_perf['win_rate'] * 100:.2f}%")
        print(f"    Total Trades: {is_perf['total_trades']}")

        # Test the optimized parameters on the out-of-sample data
        # While continuing to update the copula model
        print(f"\n  Testing out-of-sample data with optimized parameters and dynamic copula updates...")
        oos_results = backtest_with_dynamic_copula(
            df, oos_start, oos_end,
            token1_col, token2_col,
            copula_window_days, update_frequency_days,
            best_params
        )

        # Print Out-of-Sample performance
        oos_perf = oos_results['performance']
        print(f"\n  Out-of-Sample Performance:")
        print(f"    Total Return: {oos_perf['total_return'] * 100:.2f}%")
        print(f"    Sharpe Ratio: {oos_perf['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: {oos_perf['max_drawdown'] * 100:.2f}%")
        print(f"    Win Rate: {oos_perf['win_rate'] * 100:.2f}%")
        print(f"    Total Trades: {oos_perf['total_trades']}")

        # Store results for this cycle
        cycle_id = f"Cycle_{cycle + 1}"
        results[cycle_id] = {
            'is_range': is_range,
            'oos_range': oos_range,
            'best_params': best_params,
            'is_results': is_results,
            'oos_results': oos_results,
            'copula_updates': is_results.get('copula_updates', []) + oos_results.get('copula_updates', [])
        }

        # Generate and save performance plots
        create_dynamic_wfo_plots(
            is_range, oos_range,
            best_params,
            is_results, oos_results,
            os.path.join(output_dir, cycle_id)
        )

    # Calculate consolidated metrics
    if results:
        consolidated_metrics = calculate_dynamic_wfo_consolidated_results(results)

        # Generate consolidated visualizations
        create_dynamic_wfo_consolidated_visualizations(results, consolidated_metrics, output_dir)

        # Print overall performance comparison
        print("\n" + "=" * 80)
        print("CONSOLIDATED PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Overall In-Sample Return: {consolidated_metrics['overall_is_return']:.2f}%")
        print(f"Overall Out-of-Sample Return: {consolidated_metrics['overall_oos_return']:.2f}%")
        print(f"Average In-Sample Sharpe: {consolidated_metrics['avg_is_sharpe']:.2f}")
        print(f"Average Out-of-Sample Sharpe: {consolidated_metrics['avg_oos_sharpe']:.2f}")
        print(f"Robustness Ratio (OOS/IS Return): {consolidated_metrics['robustness_ratio']:.2f}")
        print(
            f"Positive Out-of-Sample Segments: {consolidated_metrics.get('positive_oos_segments', 0)} out of {consolidated_metrics.get('total_cycles', 0)}")

    else:
        print("\nNo valid cycles found for analysis.")

    return results


def optimize_parameters_dynamic_copula(df, start_date, end_date, token1_col, token2_col,
                                       copula_window_days, update_frequency_days,
                                       param_ranges, optimization_goal='sharpe_ratio', min_trades=10):
    """
    Optimize parameters using dynamic copula updating during the in-sample period.
    Now ensures SD filter is always used regardless of parameter settings.

    Parameters:
    - df: Full dataframe with all data
    - start_date, end_date: Date boundaries for the in-sample period
    - token1_col, token2_col: "**********"
    - copula_window_days: Number of days to use for copula determination
    - update_frequency_days: How often to update the copula model
    - param_ranges: Dictionary with parameter ranges to test
    - optimization_goal: Metric to optimize
    - min_trades: Minimum trades required for a valid strategy

    Returns:
    - best_params: Dictionary with best parameters
    - best_results: Results dictionary for the best parameters
    """
    # Set constant parameters
    fixed_params = {
        'fee_pct': 0.001,
        'atr_window': 14,
        'vol_filter_enabled': False,
        'sd_filter_enabled': True  # Always enable SD filter
    }

    # Calculate total combinations
    total_combinations = (
            len(param_ranges.get('window_size', [20])) *
            len(param_ranges.get('confidence_level', [0.95])) *
            len(param_ranges.get('atr_multiplier', [2.0])) *
            len(param_ranges.get('atr_on_equity', [False]))
    )

    print(f"    Optimizing with {total_combinations} parameter combinations...")

    # Track best results
    best_metric_value = float('-inf') if optimization_goal != 'max_drawdown' else float('inf')
    best_params = None
    best_results = None
    valid_strategies = 0

    # Grid search through all parameter combinations
    for window_size in param_ranges.get('window_size', [20]):
        for confidence_level in param_ranges.get('confidence_level', [0.95]):
            for atr_multiplier in param_ranges.get('atr_multiplier', [2.0]):
                for atr_on_equity in param_ranges.get('atr_on_equity', [False]):
                    # Create parameter set for this combination
                    params = {
                        'window_size': window_size,
                        'confidence_level': confidence_level,
                        'atr_multiplier': atr_multiplier,
                        'atr_on_equity': atr_on_equity,
                        # SD filter parameters - use fixed values or from param_ranges if available
                        'sd_window': param_ranges.get('sd_window', [20])[0] if isinstance(param_ranges.get('sd_window', [20]), list) else param_ranges.get('sd_window', 20),
                        'sd_lookback_windows': param_ranges.get('sd_lookback_windows', [5])[0] if isinstance(param_ranges.get('sd_lookback_windows', [5]), list) else param_ranges.get('sd_lookback_windows', 5),
                        'sd_threshold_ratio': param_ranges.get('sd_threshold_ratio', [1.0])[0] if isinstance(param_ranges.get('sd_threshold_ratio', [1.0]), list) else param_ranges.get('sd_threshold_ratio', 1.0),
                        **fixed_params
                    }

                    try:
                        # Run backtest with these parameters and dynamic copula updating
                        results = backtest_with_dynamic_copula(
                            df, start_date, end_date,
                            token1_col, token2_col,
                            copula_window_days, update_frequency_days,
                            params
                        )

                        # Check if strategy is valid based on trade count
                        if results['performance']['total_trades'] < min_trades:
                            continue

                        valid_strategies += 1

                        # Get the optimization metric
                        metric_value = results['performance'].get(optimization_goal, 0)

                        # Check if this is the best so far
                        better_metric = False
                        if optimization_goal in ['max_drawdown', 'volatility']:
                            better_metric = metric_value < best_metric_value
                        else:
                            better_metric = metric_value > best_metric_value

                        if better_metric:
                            best_metric_value = metric_value
                            best_params = params
                            best_results = results

                            # Print progress update for best result so far
                            print(f"      New best {optimization_goal}: {metric_value:.4f} with params:")
                            for param_name, param_value in params.items():
                                if param_name in ['window_size', 'confidence_level', 'atr_multiplier', 'atr_on_equity', 'sd_window', 'sd_lookback_windows', 'sd_threshold_ratio']:
                                    print(f"        {param_name}: {param_value}")
                    except Exception as e:
                        print(f"      Error testing parameters {params}: {e}")

    print(f"    Evaluated {valid_strategies} valid strategies (with minimum {min_trades} trades)")

    if best_params is None:
        print(f"    Warning: No valid parameter combinations found with at least {min_trades} trades")
        return None, None

    return best_params, best_results


def backtest_with_dynamic_copula(df, start_date, end_date, token1_col, token2_col,
                                 copula_window_days, update_frequency_days,
                                 strategy_params):
    """
    Backtest the strategy with dynamic copula updates.

    The copula model is determined using a rolling window of historical data
    and updated at regular intervals during the backtest.
    Now includes SD filter implementation regardless of parameter setting.

    Parameters:
    - df: Full dataframe with all data
    - start_date, end_date: Date boundaries for the test period
    - token1_col, token2_col: "**********"
    - copula_window_days: Number of days to use for copula determination
    - update_frequency_days: How often to update the copula model
    - strategy_params: Dictionary with strategy parameters

    Returns:
    - results: Dictionary with backtest results
    """
    # Extract data for the test period
    test_data = df[(df.index >= start_date) & (df.index <= end_date)].copy()

    # Extract token prices for the test period
    token1 = "**********"
    token2 = "**********"

    # Initialize signals, equity curve, and other result series
    signals = pd.Series(0, index=test_data.index)
    stop_levels = pd.Series(np.nan, index=test_data.index)
    equity_curve = pd.Series(1.0, index=test_data.index)
    is_outside = pd.Series(False, index=test_data.index)
    sd_filter_status = pd.Series(False, index=test_data.index)  # Track SD filter status

    # Calculate spread
    spread = "**********"

    # Calculate ATR for spread
    spread_high = pd.Series(np.maximum(spread, spread.shift(1)), index=spread.index)
    spread_low = pd.Series(np.minimum(spread, spread.shift(1)), index=spread.index)
    spread_atr = calculate_atr(spread_high, spread_low, spread, window=strategy_params.get('atr_window', 14))

    # Initialize SD filter calculations
    sd_window = strategy_params.get('sd_window', 20)
    sd_lookback_windows = strategy_params.get('sd_lookback_windows', 5)
    sd_threshold_ratio = strategy_params.get('sd_threshold_ratio', 1.0)

    # Calculate rolling standard deviation for the spread
    spread_sd = spread.rolling(window=sd_window).std()

    # Initialize position tracking variables
    current_position = 0
    entry_price = 0
    stop_price = 0
    entry_date = None

    # Calculate returns for equity tracking
    pct_change = spread.pct_change().fillna(0).clip(-0.20, 0.20)  # Limit extreme moves

    # Initialize variables for equity ATR if needed
    if strategy_params.get('atr_on_equity', False):
        equity_high = pd.Series(equity_curve.copy(), index=equity_curve.index)
        equity_low = pd.Series(equity_curve.copy(), index=equity_curve.index)
        equity_atr = pd.Series(np.nan, index=equity_curve.index)

    # Initialize copula updates tracking
    copula_updates = []
    current_copula_type = None
    current_copula = None
    last_update_date = None

    # Create a list of update dates
    update_dates = [start_date]
    current_update = start_date
    while current_update < end_date:
        current_update = current_update + pd.Timedelta(days=update_frequency_days)
        if current_update <= end_date:
            update_dates.append(current_update)

    # Record all trades for analysis
    trades = []

    # Process each date in the test period
    for i, current_date in enumerate(test_data.index):
        try:
            # Check if we need to update the copula model
            update_copula = False

            # Update on the first date or on scheduled update dates
            if last_update_date is None or current_date in update_dates:
                update_copula = True
                last_update_date = current_date

            if update_copula:
                # Calculate the start of the copula window
                copula_window_start = current_date - pd.Timedelta(days=copula_window_days)

                # Get data for the copula window
                copula_data = df[(df.index >= copula_window_start) & (df.index < current_date)]

                if len(copula_data) >= 30:  # Ensure we have enough data
                    # Find the best-fit copula for this window
                    copula_token1 = "**********"
                    copula_token2 = "**********"

                    new_copula_type = "**********"

                    # Create a new copula of the determined type
                    copula_params = {}
                    if new_copula_type == 'student_t':
                        copula_params['student_t_df'] = 5

                    current_copula = create_copula(new_copula_type, **copula_params)

                    # Fit the copula to the window data
                    u_window = "**********"
                    v_window = "**********"
                    current_copula.fit(u_window, v_window)

                    # Update the current copula type if changed
                    copula_change = (current_copula_type is not None and current_copula_type != new_copula_type)
                    current_copula_type = new_copula_type

                    # Record the update
                    copula_updates.append({
                        'date': current_date,
                        'copula_type': current_copula_type,
                        'window_start': copula_window_start,
                        'window_end': current_date,
                        'window_size': len(copula_data),
                        'copula_changed': copula_change
                    })

                    # Only print if this is the first copula determination or if the type changed
                    if current_copula is None or copula_change:
                        print(f"      Updated copula to {current_copula_type} on {current_date.strftime('%Y-%m-%d')}")

            # Skip if we don't have a copula model yet
            if current_copula is None:
                continue

            # Current values
            current_t1 = "**********"
            current_t2 = "**********"
            current_spread = spread.iloc[i]

            # Current spread ATR (used if not using equity ATR)
            current_spread_atr = spread_atr.iloc[i] if not np.isnan(spread_atr.iloc[i]) else current_spread * 0.02

            # Update equity based on previous position
            if i > 0:
                if current_position == 1:  # Long position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + pct_change.iloc[i])
                elif current_position == -1:  # Short position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 - pct_change.iloc[i])
                else:  # No position
                    equity_curve.iloc[i] = equity_curve.iloc[i - 1]

                # Update equity high/low values for ATR calculation if using equity ATR
                if strategy_params.get('atr_on_equity', False) and i >= 5:
                    equity_high.iloc[i] = max(equity_curve.iloc[i], equity_curve.iloc[i - 1])
                    equity_low.iloc[i] = min(equity_curve.iloc[i], equity_curve.iloc[i - 1])

                    # Calculate rolling ATR on equity if we have enough data
                    if i >= strategy_params.get('atr_window', 14) + 5:
                        lookback_start = max(0, i - strategy_params.get('atr_window', 14) - 10)
                        equity_atr.iloc[i] = calculate_atr(
                            equity_high.iloc[lookback_start:i + 1],
                            equity_low.iloc[lookback_start:i + 1],
                            equity_curve.iloc[lookback_start:i + 1],
                            window=strategy_params.get('atr_window', 14)
                        ).iloc[-1]

            # Get current ATR (either spread or equity based)
            if strategy_params.get('atr_on_equity', False) and i >= strategy_params.get('atr_window',
                                                                                        14) + 5 and not np.isnan(
                equity_atr.iloc[i]):
                current_atr = equity_atr.iloc[i]
                # Scale ATR to the spread's magnitude
                current_atr = current_atr / equity_curve.iloc[i] * current_spread
            else:
                current_atr = current_spread_atr

            # SD Filter logic - Ensure we have enough data for SD calculations
            sd_filter_passes = False
            if i >= sd_window + sd_lookback_windows:
                current_sd = spread_sd.iloc[i]
                # Calculate average SD over the previous lookback windows
                lookback_start = max(0, i - sd_window * sd_lookback_windows)
                lookback_end = i - sd_window + 1  # +1 because we want to exclude current window

                # Ensure we have valid lookback period
                if lookback_end > lookback_start:
                    # Calculate average SD over previous windows (excluding current window)
                    avg_sd = spread_sd.iloc[lookback_start:lookback_end].mean()

                    # Check if current SD is above threshold relative to average
                    if not np.isnan(avg_sd) and avg_sd > 0:
                        sd_ratio = current_sd / avg_sd
                        sd_filter_passes = sd_ratio >= sd_threshold_ratio
                else:
                    # Not enough data for lookback, default to pass
                    sd_filter_passes = True
            else:
                # Not enough data yet, default to pass
                sd_filter_passes = True

            # Record SD filter status
            sd_filter_status.iloc[i] = sd_filter_passes

            # Convert current points to quantiles
            recent_window = 30  # Use a small window for ranking
            recent_start = max(0, i - recent_window)
            recent_t1 = token1.iloc[recent_start: "**********"
            recent_t2 = token2.iloc[recent_start: "**********"
            u_current = rankdata(recent_t1)[-1] / (len(recent_t1) + 1)
            v_current = rankdata(recent_t2)[-1] / (len(recent_t2) + 1)

            # Check if point is outside confidence bands
            outside_bands, distance, direction = current_copula.is_point_outside_bands(
                u_current, v_current, strategy_params.get('confidence_level', 0.95))
            is_outside.iloc[i] = outside_bands

            # Determine signal based on whether point is outside bands, direction, and SD filter
            if outside_bands and sd_filter_passes:  # Apply SD filter regardless of parameter setting
                if direction < 0:  # First asset undervalued
                    new_signal = 1  # Long signal
                elif direction > 0:  # First asset overvalued
                    new_signal = -1  # Short signal
                else:
                    new_signal = current_position
            else:
                # Exit when point returns inside bands or SD filter fails
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
                    equity_curve.iloc[i] *= (1 - strategy_params.get('fee_pct', 0.001))

                    # Set position and entry price
                    current_position = new_signal
                    entry_price = current_spread
                    entry_date = current_date

                    # Set initial stop-loss level
                    if current_position == 1:  # Long position
                        stop_price = entry_price - strategy_params.get('atr_multiplier', 2.0) * current_atr
                    else:  # Short position
                        stop_price = entry_price + strategy_params.get('atr_multiplier', 2.0) * current_atr

                    signals.iloc[i] = current_position
                    stop_levels.iloc[i] = stop_price

            else:  # Already in a position
                # Update trailing stop if in profit
                if not stop_hit:
                    if current_position == 1 and current_spread > entry_price:
                        # For long positions, raise stop as price increases
                        new_stop = current_spread - strategy_params.get('atr_multiplier', 2.0) * current_atr
                        stop_price = max(stop_price, new_stop)
                    elif current_position == -1 and current_spread < entry_price:
                        # For short positions, lower stop as price decreases
                        new_stop = current_spread + strategy_params.get('atr_multiplier', 2.0) * current_atr
                        stop_price = min(stop_price, new_stop)

                # Determine whether to exit or maintain position
                if stop_hit:  # Stop-loss hit
                    # Record trade
                    if current_position == 1:
                        profit_pct = (current_spread - entry_price) / abs(entry_price) - strategy_params.get('fee_pct',
                                                                                                             0.001) * 2
                    else:
                        profit_pct = (entry_price - current_spread) / abs(entry_price) - strategy_params.get('fee_pct',
                                                                                                             0.001) * 2

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'position': 'Long' if current_position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': current_spread,
                        'profit_pct': profit_pct,
                        'duration': (current_date - entry_date).days,
                        'exit_reason': 'Stop Loss',
                        'copula_type': current_copula_type
                    })

                    # Apply fee for exit
                    equity_curve.iloc[i] *= (1 - strategy_params.get('fee_pct', 0.001))

                    signals.iloc[i] = 0
                    current_position = 0
                    stop_price = 0

                elif new_signal == 0:  # Exit signal (point inside bands)
                    # Record trade
                    if current_position == 1:
                        profit_pct = (current_spread - entry_price) / abs(entry_price) - strategy_params.get('fee_pct',
                                                                                                             0.001) * 2
                    else:
                        profit_pct = (entry_price - current_spread) / abs(entry_price) - strategy_params.get('fee_pct',
                                                                                                             0.001) * 2

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'position': 'Long' if current_position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': current_spread,
                        'profit_pct': profit_pct,
                        'duration': (current_date - entry_date).days,
                        'exit_reason': 'Signal',
                        'copula_type': current_copula_type
                    })

                    # Apply fee for exit
                    equity_curve.iloc[i] *= (1 - strategy_params.get('fee_pct', 0.001))

                    signals.iloc[i] = 0
                    current_position = 0
                    stop_price = 0

                elif new_signal != current_position and new_signal != 0:  # Reversal signal
                    # Record existing trade
                    if current_position == 1:
                        profit_pct = (current_spread - entry_price) / abs(entry_price) - strategy_params.get('fee_pct',
                                                                                                             0.001) * 2
                    else:
                        profit_pct = (entry_price - current_spread) / abs(entry_price) - strategy_params.get('fee_pct',
                                                                                                             0.001) * 2

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'position': 'Long' if current_position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': current_spread,
                        'profit_pct': profit_pct,
                        'duration': (current_date - entry_date).days,
                        'exit_reason': 'Reversal',
                        'copula_type': current_copula_type
                    })

                    # Apply fee for both exit and entry
                    equity_curve.iloc[i] *= (1 - strategy_params.get('fee_pct', 0.001)) * (
                            1 - strategy_params.get('fee_pct', 0.001))

                    # Update position
                    current_position = new_signal
                    entry_price = current_spread
                    entry_date = current_date

                    # Set new stop-loss level
                    if current_position == 1:  # Long position
                        stop_price = entry_price - strategy_params.get('atr_multiplier', 2.0) * current_atr
                    else:  # Short position
                        stop_price = entry_price + strategy_params.get('atr_multiplier', 2.0) * current_atr

                    signals.iloc[i] = current_position

                else:  # Maintain current position
                    signals.iloc[i] = current_position

                # Record stop level
                stop_levels.iloc[i] = stop_price

        except Exception as e:
            print(f"Error at date {current_date}: {e}")
            # Keep previous values if error occurs
            if i > 0:
                signals.iloc[i] = signals.iloc[i - 1]
                equity_curve.iloc[i] = equity_curve.iloc[i - 1]
                stop_levels.iloc[i] = stop_levels.iloc[i - 1]
                sd_filter_status.iloc[i] = sd_filter_status.iloc[i - 1]

    # Handle the last open position if any
    if current_position != 0:
        # Calculate profit for the last position
        last_date = test_data.index[-1]
        last_spread = spread.iloc[-1]

        if current_position == 1:
            profit_pct = (last_spread - entry_price) / abs(entry_price) - strategy_params.get('fee_pct', 0.001)
        else:
            profit_pct = (entry_price - last_spread) / abs(entry_price) - strategy_params.get('fee_pct', 0.001)

        trades.append({
            'entry_date': entry_date,
            'exit_date': last_date,
            'position': 'Long' if current_position == 1 else 'Short',
            'entry_price': entry_price,
            'exit_price': last_spread,
            'profit_pct': profit_pct,
            'duration': (last_date - entry_date).days,
            'exit_reason': 'End of Test',
            'copula_type': current_copula_type,
            'open': True
        })

    # Convert trades list to DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Calculate performance metrics
    performance = calculate_detailed_performance(equity_curve, trades_df)

    # Prepare results dictionary
    results = {
        'signals': signals,
        'stop_levels': stop_levels,
        'equity_curve': equity_curve,
        'is_outside': is_outside,
        'sd_filter_status': sd_filter_status,  # Add SD filter status to results
        'spread': spread,
        'spread_sd': spread_sd,  # Add spread SD to results
        'trades': trades_df,
        'copula_updates': copula_updates,
        'performance': performance
    }

    return results


def calculate_detailed_performance(equity_curve, trades_df):
    """
    Calculate detailed performance metrics based on equity curve and trades.

    Parameters:
    - equity_curve: Series of equity values
    - trades_df: DataFrame containing trade details

    Returns:
    - Dictionary with performance metrics
    """
    if len(equity_curve) < 2:
        return {
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
            'volatility': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0
        }

    # Calculate basic metrics
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    # Calculate returns and volatility for Sharpe ratio
    returns = equity_curve.pct_change().dropna()

    # Calculate volatility
    volatility = returns.std() * np.sqrt(252)  # Annualized

    # Calculate max drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve / peak - 1)
    max_drawdown = abs(drawdown.min())

    # Define risk-free rate (5% annual)
    risk_free_rate_annual = 0.05
    risk_free_rate_daily = risk_free_rate_annual / 365  # Daily risk-free rate

    # Calculate annualized metrics
    days_in_test = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days_in_test > 0:
        annualized_return = (1 + total_return) ** (365 / days_in_test) - 1
    else:
        annualized_return = 0

    # Calculate excess return over risk-free rate
    excess_return = annualized_return - risk_free_rate_annual

    # Calculate Sharpe ratio
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Trade statistics
    if not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = sum(trades_df['profit_pct'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_winning_trade = trades_df.loc[trades_df['profit_pct'] > 0, 'profit_pct'].mean() if winning_trades > 0 else 0
        avg_losing_trade = trades_df.loc[trades_df['profit_pct'] <= 0, 'profit_pct'].mean() if losing_trades > 0 else 0
        largest_winner = trades_df['profit_pct'].max() if not trades_df.empty else 0
        largest_loser = trades_df['profit_pct'].min() if not trades_df.empty else 0
        avg_trade_duration = trades_df['duration'].mean() if not trades_df.empty else 0

        # Profit factor
        gross_profit = sum(trades_df.loc[trades_df['profit_pct'] > 0, 'profit_pct'])
        gross_loss = abs(sum(trades_df.loc[trades_df['profit_pct'] < 0, 'profit_pct']))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0
        avg_winning_trade = 0
        avg_losing_trade = 0
        largest_winner = 0
        largest_loser = 0
        avg_trade_duration = 0
        profit_factor = 0

    # Return metrics dictionary
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_winning_trade': avg_winning_trade,
        'avg_losing_trade': avg_losing_trade,
        'largest_winner': largest_winner,
        'largest_loser': largest_loser,
        'avg_trade_duration': avg_trade_duration,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor
    }


def calculate_dynamic_wfo_consolidated_results(results):
    """
    Calculate consolidated metrics across all walk-forward cycles.

    Parameters:
    - results: Dictionary with cycle results

    Returns:
    - Dictionary with consolidated metrics
    """
    # Initialize metrics
    total_cycles = 0
    is_return_total = 1.0
    oos_return_total = 1.0

    is_returns = []
    oos_returns = []
    is_sharpes = []
    oos_sharpes = []
    is_drawdowns = []
    oos_drawdowns = []
    is_win_rates = []
    oos_win_rates = []
    is_trade_counts = []
    oos_trade_counts = []

    return_ratios = []
    sharpe_ratios = []
    positive_oos_segments = 0

    copula_frequencies = {}

    # Calculate metrics for each cycle
    for cycle_id, result in results.items():
        is_perf = result['is_results']['performance']
        oos_perf = result['oos_results']['performance']

        total_cycles += 1

        # Track copula frequencies
        for update in result.get('copula_updates', []):
            copula_type = update['copula_type']
            if copula_type in copula_frequencies:
                copula_frequencies[copula_type] += 1
            else:
                copula_frequencies[copula_type] = 1

        # Compound returns
        is_return_total *= (1 + is_perf['total_return'])
        oos_return_total *= (1 + oos_perf['total_return'])

        # Store metrics
        is_returns.append(is_perf['total_return'] * 100)  # Convert to percentage
        oos_returns.append(oos_perf['total_return'] * 100)  # Convert to percentage
        is_sharpes.append(is_perf['sharpe_ratio'])
        oos_sharpes.append(oos_perf['sharpe_ratio'])
        is_drawdowns.append(is_perf['max_drawdown'] * 100)  # Convert to percentage
        oos_drawdowns.append(oos_perf['max_drawdown'] * 100)  # Convert to percentage
        is_win_rates.append(is_perf['win_rate'] * 100)  # Convert to percentage
        oos_win_rates.append(oos_perf['win_rate'] * 100)  # Convert to percentage
        is_trade_counts.append(is_perf['total_trades'])
        oos_trade_counts.append(oos_perf['total_trades'])

        # Calculate ratios
        if is_perf['total_return'] != 0:
            return_ratios.append(oos_perf['total_return'] / is_perf['total_return'])

        if is_perf['sharpe_ratio'] != 0:
            sharpe_ratios.append(oos_perf['sharpe_ratio'] / is_perf['sharpe_ratio'])

        # Count positive out-of-sample segments
        if oos_perf['total_return'] > 0:
            positive_oos_segments += 1

    # Calculate averages and overall metrics
    consolidated_metrics = {
        'total_cycles': total_cycles,
        'overall_is_return': (is_return_total - 1) * 100,  # Convert to percentage
        'overall_oos_return': (oos_return_total - 1) * 100,  # Convert to percentage
        'avg_is_return': np.mean(is_returns),
        'avg_oos_return': np.mean(oos_returns),
        'avg_is_sharpe': np.mean(is_sharpes),
        'avg_oos_sharpe': np.mean(oos_sharpes),
        'avg_is_drawdown': np.mean(is_drawdowns),
        'avg_oos_drawdown': np.mean(oos_drawdowns),
        'avg_is_win_rate': np.mean(is_win_rates),
        'avg_oos_win_rate': np.mean(oos_win_rates),
        'avg_is_trades': np.mean(is_trade_counts),
        'avg_oos_trades': np.mean(oos_trade_counts),
        'avg_return_ratio': np.mean(return_ratios) if return_ratios else 0,
        'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
        'positive_oos_segments': positive_oos_segments,
        'is_returns': is_returns,
        'oos_returns': oos_returns,
        'is_sharpes': is_sharpes,
        'oos_sharpes': oos_sharpes,
        'copula_frequencies': copula_frequencies,
    }

    # Calculate robustness ratio (out-of-sample vs in-sample performance)
    if is_return_total > 1:
        robustness_ratio = (oos_return_total - 1) / (is_return_total - 1)
    else:
        robustness_ratio = 0

    consolidated_metrics['robustness_ratio'] = robustness_ratio

    return consolidated_metrics


def create_dynamic_wfo_plots(is_range, oos_range, best_params, is_results, oos_results, output_dir):
    """
    Create plots comparing performance across in-sample and out-of-sample periods.
    Now includes additional SD filter visualization.

    Parameters:
    - is_range, oos_range: Date ranges for segments
    - best_params: Optimized parameters
    - is_results, oos_results: Results dictionaries
    - output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Equity Curves Comparison
    fig, ax = plt.subplots(figsize=(14, 7))

    # Normalize equity curves to start at 100
    is_equity = is_results['equity_curve'] * 100 / is_results['equity_curve'].iloc[0]
    oos_equity = oos_results['equity_curve'] * 100 / oos_results['equity_curve'].iloc[0]

    # Plot equity curves
    is_equity.plot(ax=ax, label=f'In-Sample ({is_range})', color='blue')
    oos_equity.plot(ax=ax, label=f'Out-of-Sample ({oos_range})', color='green')

    # Add labels and title
    param_str = f"Window: {best_params['window_size']}, Conf: {best_params['confidence_level']}, " \
                f"ATR: {best_params['atr_multiplier']}, ATR on Equity: {best_params['atr_on_equity']}"
    sd_str = f"SD Window: {best_params['sd_window']}, Lookback: {best_params['sd_lookback_windows']}, " \
             f"Threshold: {best_params['sd_threshold_ratio']}"
    ax.set_title(f"Equity Curves Comparison - {param_str}\n{sd_str}")
    ax.set_ylabel("Equity Value (Starting = 100)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add performance metrics
    is_perf = is_results['performance']
    oos_perf = oos_results['performance']

    metrics_text = (
        f"In-Sample ({is_range}):\n"
        f"Return: {is_perf['total_return'] * 100:.2f}%\n"
        f"Sharpe: {is_perf['sharpe_ratio']:.2f}\n"
        f"Drawdown: {is_perf['max_drawdown'] * 100:.2f}%\n"
        f"Win Rate: {is_perf['win_rate'] * 100:.2f}%\n"
        f"Trades: {is_perf['total_trades']}\n\n"
        f"Out-of-Sample ({oos_range}):\n"
        f"Return: {oos_perf['total_return'] * 100:.2f}%\n"
        f"Sharpe: {oos_perf['sharpe_ratio']:.2f}\n"
        f"Drawdown: {oos_perf['max_drawdown'] * 100:.2f}%\n"
        f"Win Rate: {oos_perf['win_rate'] * 100:.2f}%\n"
        f"Trades: {oos_perf['total_trades']}"
    )

    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equity_comparison.png"), dpi=150)
    plt.close()

    # 2. Copula Updates Visualization
    is_updates = is_results.get('copula_updates', [])
    oos_updates = oos_results.get('copula_updates', [])

    if is_updates or oos_updates:
        fig, ax = plt.subplots(figsize=(14, 5))

        # Create a map of copula types to y-axis positions
        copula_types = set([update['copula_type'] for update in is_updates + oos_updates])
        copula_positions = {copula: i for i, copula in enumerate(sorted(copula_types))}

        # Create a color map for copula types
        color_map = {
            'gaussian': 'blue',
            'student_t': 'green',
            'clayton': 'red',
            'gumbel': 'orange',
            'frank': 'purple'
        }

        # Plot In-Sample copula updates
        for update in is_updates:
            date = update['date']
            copula = update['copula_type']
            ax.scatter(date, copula_positions[copula],
                       marker='o', s=100,
                       color=color_map.get(copula, 'gray'),
                       edgecolor='black', linewidth=1.5,
                       label=f'{copula} (IS)')

        # Plot Out-of-Sample copula updates
        for update in oos_updates:
            date = update['date']
            copula = update['copula_type']
            ax.scatter(date, copula_positions[copula],
                       marker='s', s=100,
                       color=color_map.get(copula, 'gray'),
                       edgecolor='black', linewidth=1.5,
                       label=f'{copula} (OOS)')

        # Add vertical line to separate IS and OOS periods
        is_end = pd.to_datetime(is_range.split(' to ')[1])
        ax.axvline(x=is_end, color='black', linestyle='--',
                   label='IS/OOS Boundary')

        # Set y-axis ticks to copula names
        ax.set_yticks(list(copula_positions.values()))
        ax.set_yticklabels(list(copula_positions.keys()))

        # Add title and labels
        ax.set_title("Dynamic Copula Updates During Backtest")
        ax.set_xlabel("Date")
        ax.set_ylabel("Copula Type")

        # Handle duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "copula_updates.png"), dpi=150)
        plt.close()

    # 3. Trade Distribution Comparison
    if 'trades' in is_results and 'trades' in oos_results:
        is_trades = is_results.get('trades', pd.DataFrame())
        oos_trades = oos_results.get('trades', pd.DataFrame())

        if not is_trades.empty and not oos_trades.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Plot in-sample trade distribution
            is_returns = is_trades['profit_pct'] * 100  # Convert to percentage
            ax1.hist(is_returns, bins=20, alpha=0.7, color='blue', label=f'In-Sample ({is_range})')
            ax1.axvline(x=0, color='black', linestyle='--')
            ax1.set_title(f"In-Sample Trade P&L Distribution")
            ax1.set_xlabel("Trade P&L (%)")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot out-of-sample trade distribution
            oos_returns = oos_trades['profit_pct'] * 100  # Convert to percentage
            ax2.hist(oos_returns, bins=20, alpha=0.7, color='green', label=f'Out-of-Sample ({oos_range})')
            ax2.axvline(x=0, color='black', linestyle='--')
            ax2.set_title(f"Out-of-Sample Trade P&L Distribution")
            ax2.set_xlabel("Trade P&L (%)")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "trade_distribution.png"), dpi=150)
            plt.close()

    # 4. SD Filter Visualization (New)
    if 'spread_sd' in is_results and 'spread_sd' in oos_results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # IS Period: Plot spread with SD
        is_spread = is_results['spread']
        is_sd = is_results['spread_sd']
        is_filter_status = is_results.get('sd_filter_status', pd.Series(False, index=is_spread.index))

        # Plot spread
        ax1.plot(is_spread.index, is_spread, label='Spread', color='blue', alpha=0.7)

        # Plot SD
        ax1_twin = ax1.twinx()
        ax1_twin.plot(is_sd.index, is_sd, label='Standard Deviation', color='red', linestyle='--')

        # Mark periods where SD filter passes
        if not is_filter_status.empty:
            sd_passes = is_filter_status[is_filter_status]
            if not sd_passes.empty:
                ax1.scatter(sd_passes.index, is_spread.loc[sd_passes.index],
                            color='green', alpha=0.4, s=20, marker='^',
                            label='SD Filter Pass')

        # Setup legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.set_title(f"In-Sample: Spread & SD Filter ({is_range})")
        ax1.set_ylabel("Spread Value")
        ax1_twin.set_ylabel("Standard Deviation")
        ax1.grid(True, alpha=0.3)

        # OOS Period: Plot spread with SD
        oos_spread = oos_results['spread']
        oos_sd = oos_results['spread_sd']
        oos_filter_status = oos_results.get('sd_filter_status', pd.Series(False, index=oos_spread.index))

        # Plot spread
        ax2.plot(oos_spread.index, oos_spread, label='Spread', color='blue', alpha=0.7)

        # Plot SD
        ax2_twin = ax2.twinx()
        ax2_twin.plot(oos_sd.index, oos_sd, label='Standard Deviation', color='red', linestyle='--')

        # Mark periods where SD filter passes
        if not oos_filter_status.empty:
            sd_passes = oos_filter_status[oos_filter_status]
            if not sd_passes.empty:
                ax2.scatter(sd_passes.index, oos_spread.loc[sd_passes.index],
                            color='green', alpha=0.4, s=20, marker='^',
                            label='SD Filter Pass')

        # Setup legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax2.set_title(f"Out-of-Sample: Spread & SD Filter ({oos_range})")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Spread Value")
        ax2_twin.set_ylabel("Standard Deviation")
        ax2.grid(True, alpha=0.3)

        # Format date axis
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sd_filter_visualization.png"), dpi=150)
        plt.close()

    # 5. Trade Analysis by Copula Type
    if 'trades' in is_results and 'trades' in oos_results:
        all_trades = pd.concat([
            is_results['trades'].assign(period='In-Sample') if not is_results['trades'].empty else pd.DataFrame(),
            oos_results['trades'].assign(period='Out-of-Sample') if not oos_results['trades'].empty else pd.DataFrame()
        ])

        if not all_trades.empty and 'copula_type' in all_trades.columns:
            # Create a pivot table of trade performance by copula type
            try:
                copula_analysis = all_trades.groupby(['period', 'copula_type']).agg({
                    'profit_pct': ['count', 'mean', 'std', 'min', 'max'],
                    'duration': 'mean'
                })

                # Convert profit_pct to percentage
                for stat in ['mean', 'std', 'min', 'max']:
                    copula_analysis[('profit_pct', stat)] *= 100

                # Calculate win rate by copula type using agg instead of apply
                win_rates = all_trades.groupby(['period', 'copula_type']).agg(
                    win_rate=(('profit_pct'), lambda x: (x > 0).mean() * 100)
                )

                # Use concat instead of merge to avoid index level issues
                copula_analysis = pd.concat([copula_analysis, win_rates], axis=1)

                # Save analysis to CSV
                copula_analysis.to_csv(os.path.join(output_dir, "trade_by_copula_type.csv"))

                # Create a bar chart of performance by copula type
                fig, ax = plt.subplots(figsize=(12, 6))

                # Prepare data for plotting - handle multi-index correctly
                performance_by_copula = all_trades.groupby(['period', 'copula_type'])['profit_pct'].mean() * 100

                # Try to reshape for plotting or fallback to a simpler plot if it fails
                try:
                    performance_plot_data = performance_by_copula.unstack(level=0)
                    performance_plot_data.plot(kind='bar', ax=ax)
                except:
                    # Fallback to a simpler plot
                    performance_by_copula.plot(kind='bar', ax=ax)

                ax.set_title("Average Trade P&L by Copula Type")
                ax.set_ylabel("Average P&L (%)")
                ax.set_xlabel("Copula Type")
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "performance_by_copula.png"), dpi=150)
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create copula type analysis plots. Error: {e}")

    # 6. Save detailed results summary as text
    with open(os.path.join(output_dir, "results_summary.txt"), 'w') as f:
        f.write(f"DYNAMIC COPULA WALK-FORWARD OPTIMIZATION\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"DATE RANGES:\n")
        f.write(f"In-Sample: {is_range}\n")
        f.write(f"Out-of-Sample: {oos_range}\n\n")

        f.write("OPTIMIZED PARAMETERS:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")

        f.write("\nSD FILTER PARAMETERS:\n")
        f.write(f"SD Window: {best_params['sd_window']}\n")
        f.write(f"SD Lookback Windows: {best_params['sd_lookback_windows']}\n")
        f.write(f"SD Threshold Ratio: {best_params['sd_threshold_ratio']}\n")

        f.write("\nCOPULA UPDATES:\n")
        is_copula_counts = {}
        oos_copula_counts = {}

        for update in is_results.get('copula_updates', []):
            copula = update['copula_type']
            is_copula_counts[copula] = is_copula_counts.get(copula, 0) + 1

        for update in oos_results.get('copula_updates', []):
            copula = update['copula_type']
            oos_copula_counts[copula] = oos_copula_counts.get(copula, 0) + 1

        f.write("In-Sample Copula Distribution:\n")
        for copula, count in is_copula_counts.items():
            f.write(f"  {copula}: {count} updates\n")

        f.write("\nOut-of-Sample Copula Distribution:\n")
        for copula, count in oos_copula_counts.items():
            f.write(f"  {copula}: {count} updates\n")

        f.write("\nPERFORMANCE METRICS:\n")
        f.write("In-Sample Performance:\n")
        for metric, value in is_perf.items():
            if metric in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate', 'volatility']:
                f.write(f"  {metric}: {value * 100:.2f}%\n")
            else:
                f.write(f"  {metric}: {value:.4f}\n")

        f.write("\nOut-of-Sample Performance:\n")
        for metric, value in oos_perf.items():
            if metric in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate', 'volatility']:
                f.write(f"  {metric}: {value * 100:.2f}%\n")
            else:
                f.write(f"  {metric}: {value:.4f}\n")

        # Calculate performance comparison
        if is_perf['total_return'] != 0:
            return_ratio = oos_perf['total_return'] / is_perf['total_return']
            f.write(f"\nOut-of-Sample/In-Sample Return Ratio: {return_ratio:.2f}\n")

        if is_perf['sharpe_ratio'] != 0:
            sharpe_ratio = oos_perf['sharpe_ratio'] / is_perf['sharpe_ratio']
            f.write(f"Out-of-Sample/In-Sample Sharpe Ratio: {sharpe_ratio:.2f}\n")

        # SD filter analysis
        if 'sd_filter_status' in is_results and 'sd_filter_status' in oos_results:
            is_sd_status = is_results['sd_filter_status']
            oos_sd_status = oos_results['sd_filter_status']

            f.write("\nSD FILTER ANALYSIS:\n")
            is_pass_rate = 100 * is_sd_status.mean() if not is_sd_status.empty else 0
            oos_pass_rate = 100 * oos_sd_status.mean() if not oos_sd_status.empty else 0

            f.write(f"In-Sample SD Filter Pass Rate: {is_pass_rate:.2f}%\n")
            f.write(f"Out-of-Sample SD Filter Pass Rate: {oos_pass_rate:.2f}%\n")

        # Trade analysis summary
        if 'trades' in is_results and 'trades' in oos_results:
            is_trades = is_results['trades']
            oos_trades = oos_results['trades']

            f.write("\nTRADE ANALYSIS:\n")
            f.write(f"In-Sample Trades: {len(is_trades)}\n")
            f.write(f"Out-of-Sample Trades: {len(oos_trades)}\n")

            if not is_trades.empty and 'exit_reason' in is_trades.columns:
                is_exit_counts = is_trades['exit_reason'].value_counts()
                f.write("\nIn-Sample Exit Reasons:\n")
                for reason, count in is_exit_counts.items():
                    f.write(f"  {reason}: {count} trades\n")

            if not oos_trades.empty and 'exit_reason' in oos_trades.columns:
                oos_exit_counts = oos_trades['exit_reason'].value_counts()
                f.write("\nOut-of-Sample Exit Reasons:\n")
                for reason, count in oos_exit_counts.items():
                    f.write(f"  {reason}: {count} trades\n")


def create_dynamic_wfo_consolidated_visualizations(results, consolidated_metrics, output_dir):
    """
    Create consolidated visualizations for all walk-forward cycles.
    Now includes SD filter visualization.

    Parameters:
    - results: Dictionary with cycle results
    - consolidated_metrics: Dictionary with consolidated metrics
    - output_dir: Directory to save visualizations
    """
    # Extract cycle IDs
    cycle_ids = list(results.keys())

    # 1. In-Sample vs Out-of-Sample Returns Comparison
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(cycle_ids))
    width = 0.35

    # Plot in-sample and out-of-sample returns
    ax.bar(x - width / 2, consolidated_metrics['is_returns'], width,
           label='In-Sample', color='blue', alpha=0.7)
    ax.bar(x + width / 2, consolidated_metrics['oos_returns'], width,
           label='Out-of-Sample', color='green', alpha=0.7)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_title('In-Sample vs Out-of-Sample Returns by Cycle')
    ax.set_ylabel('Return (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(cycle_ids, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text with overall metrics
    metrics_text = (
        f"Overall In-Sample Return: {consolidated_metrics['overall_is_return']:.2f}%\n"
        f"Overall Out-of-Sample Return: {consolidated_metrics['overall_oos_return']:.2f}%\n"
        f"Average In-Sample Return: {consolidated_metrics['avg_is_return']:.2f}%\n"
        f"Average Out-of-Sample Return: {consolidated_metrics['avg_oos_return']:.2f}%\n"
        f"Robustness Ratio: {consolidated_metrics['robustness_ratio']:.2f}"
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "consolidated_returns_comparison.png"), dpi=150)
    plt.close()

    # 2. SD Filter Pass Rate Comparison (New)
    if 'is_sd_pass_rates' in consolidated_metrics and 'oos_sd_pass_rates' in consolidated_metrics:
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot in-sample and out-of-sample SD pass rates
        ax.bar(x - width / 2, consolidated_metrics['is_sd_pass_rates'], width,
               label='In-Sample', color='blue', alpha=0.7)
        ax.bar(x + width / 2, consolidated_metrics['oos_sd_pass_rates'], width,
               label='Out-of-Sample', color='green', alpha=0.7)

        # Add labels and title
        ax.set_title('SD Filter Pass Rates by Cycle')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(cycle_ids, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add text with overall metrics
        sd_metrics_text = (
            f"Average In-Sample SD Pass Rate: {consolidated_metrics['avg_is_sd_pass_rate']:.2f}%\n"
            f"Average Out-of-Sample SD Pass Rate: {consolidated_metrics['avg_oos_sd_pass_rate']:.2f}%\n"
            f"SD Filter: Window={results[cycle_ids[0]]['best_params']['sd_window']}, "
            f"Lookback={results[cycle_ids[0]]['best_params']['sd_lookback_windows']}, "
            f"Threshold={results[cycle_ids[0]]['best_params']['sd_threshold_ratio']}"
        )

        ax.text(0.02, 0.98, sd_metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sd_filter_pass_rates.png"), dpi=150)
        plt.close()

        # 3. Copula Type Distribution
    if 'copula_frequencies' in consolidated_metrics:
        copula_counts = consolidated_metrics['copula_frequencies']

        if copula_counts:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create color map for consistency
            color_map = {
                'gaussian': 'blue',
                'student_t': 'green',
                'clayton': 'red',
                'gumbel': 'orange',
                'frank': 'purple'
            }

            # Create pie chart
            labels = list(copula_counts.keys())
            sizes = list(copula_counts.values())
            colors = [color_map.get(label, 'gray') for label in labels]

            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

            plt.title('Copula Type Distribution Across All Cycles')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "copula_type_distribution.png"), dpi=150)
            plt.close()

        # 4. Sharpe Ratio Comparison
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot in-sample and out-of-sample Sharpe ratios
    ax.bar(x - width / 2, consolidated_metrics['is_sharpes'], width,
           label='In-Sample', color='blue', alpha=0.7)
    ax.bar(x + width / 2, consolidated_metrics['oos_sharpes'], width,
           label='Out-of-Sample', color='green', alpha=0.7)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_title('In-Sample vs Out-of-Sample Sharpe Ratios by Cycle')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(cycle_ids, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text with overall metrics
    metrics_text = (
        f"Average In-Sample Sharpe: {consolidated_metrics['avg_is_sharpe']:.2f}\n"
        f"Average Out-of-Sample Sharpe: {consolidated_metrics['avg_oos_sharpe']:.2f}\n"
        f"Sharpe Ratio OOS/IS: {consolidated_metrics['avg_sharpe_ratio']:.2f}"
    )

    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "consolidated_sharpe_comparison.png"), dpi=150)
    plt.close()

    # 5. Create consolidated performance summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Define summary metrics
    summary_metrics = {
        'Overall In-Sample Return (%)': consolidated_metrics['overall_is_return'],
        'Overall Out-of-Sample Return (%)': consolidated_metrics['overall_oos_return'],
        'Average In-Sample Return (%)': consolidated_metrics['avg_is_return'],
        'Average Out-of-Sample Return (%)': consolidated_metrics['avg_oos_return'],
        'Average In-Sample Sharpe': consolidated_metrics['avg_is_sharpe'],
        'Average Out-of-Sample Sharpe': consolidated_metrics['avg_oos_sharpe'],
        'Average In-Sample Drawdown (%)': consolidated_metrics['avg_is_drawdown'],
        'Average Out-of-Sample Drawdown (%)': consolidated_metrics['avg_oos_drawdown'],
        'Robustness Ratio (OOS/IS Return)': consolidated_metrics['robustness_ratio'],
        'Average Sharpe Ratio (OOS/IS)': consolidated_metrics['avg_sharpe_ratio'],
        'Positive Out-of-Sample Segments': f"{consolidated_metrics.get('positive_oos_segments', 0)} of {consolidated_metrics.get('total_cycles', 0)}"
    }

    # Add SD filter metrics if available
    if 'avg_is_sd_pass_rate' in consolidated_metrics:
        summary_metrics['Average In-Sample SD Pass Rate (%)'] = consolidated_metrics['avg_is_sd_pass_rate']

    if 'avg_oos_sd_pass_rate' in consolidated_metrics:
        summary_metrics['Average Out-of-Sample SD Pass Rate (%)'] = consolidated_metrics['avg_oos_sd_pass_rate']

    # Create the table
    table_data = [[metric, f"{value:.2f}" if isinstance(value, (float, int)) else value]
                  for metric, value in summary_metrics.items()]

    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                     loc='center', cellLoc='center')

    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.title(f"Dynamic Copula Walk-Forward Optimization: Consolidated Performance Summary", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "consolidated_summary_table.png"), dpi=150)
    plt.close()

    # 6. Scatter plot of In-Sample vs Out-of-Sample returns
    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot with points labeled by cycle ID
    scatter = ax.scatter(consolidated_metrics['is_returns'],
                         consolidated_metrics['oos_returns'],
                         c=range(len(cycle_ids)), cmap='viridis', s=100, alpha=0.7)

    # Add 45-degree line (perfect correlation)
    min_val = min(min(consolidated_metrics['is_returns']),
                  min(consolidated_metrics['oos_returns']))
    max_val = max(max(consolidated_metrics['is_returns']),
                  max(consolidated_metrics['oos_returns']))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # Add zero lines
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='-', alpha=0.3)

    # Add cycle labels
    for i, cycle_id in enumerate(cycle_ids):
        ax.annotate(cycle_id,
                    (consolidated_metrics['is_returns'][i],
                     consolidated_metrics['oos_returns'][i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')

    # Add labels and title
    ax.set_title('In-Sample vs Out-of-Sample Returns')
    ax.set_xlabel('In-Sample Return (%)')
    ax.set_ylabel('Out-of-Sample Return (%)')
    ax.grid(True, alpha=0.3)

    # Add quadrant labels
    ax.text(0.25 * max_val, 0.75 * max_val, "IS+, OOS+\nIdeal",
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.25 * max_val, 0.75 * min_val, "IS+, OOS-\nOverfit?",
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.25 * min_val, 0.75 * max_val, "IS-, OOS+\nUnderfit?",
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.25 * min_val, 0.75 * min_val, "IS-, OOS-\nInvalid",
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "in_sample_vs_out_sample_scatter.png"), dpi=150)
    plt.close()

    # 7. Save consolidated metrics as a text file
    with open(os.path.join(output_dir, "consolidated_performance.txt"), 'w') as f:
        f.write("DYNAMIC COPULA WALK-FORWARD OPTIMIZATION: CONSOLIDATED PERFORMANCE\n")
        f.write("=" * 60 + "\n\n")

        f.write("OVERALL PERFORMANCE METRICS:\n")
        for metric, value in summary_metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.2f}\n")
            else:
                f.write(f"{metric}: {value}\n")

        # Add SD filter info
        f.write("\nSD FILTER CONFIGURATION:\n")
        sample_params = results[cycle_ids[0]]['best_params']
        f.write(f"Window Size: {sample_params['sd_window']}\n")
        f.write(f"Lookback Windows: {sample_params['sd_lookback_windows']}\n")
        f.write(f"Threshold Ratio: {sample_params['sd_threshold_ratio']}\n")

        f.write("\n" + "=" * 60 + "\n\n")
        f.write("CYCLE-BY-CYCLE BREAKDOWN:\n\n")

        for i, cycle_id in enumerate(cycle_ids):
            result = results[cycle_id]
            f.write(f"{cycle_id}:\n")

            # Segment ranges
            f.write(f"  In-Sample: {result['is_range']}\n")
            f.write(f"  Out-of-Sample: {result['oos_range']}\n")

            # Parameters
            f.write("  Optimized Parameters:\n")
            for param, value in result['best_params'].items():
                f.write(f"    {param}: {value}\n")

            # Performance
            f.write(f"  In-Sample Performance:\n")
            f.write(f"    Return: {consolidated_metrics['is_returns'][i]:.2f}%\n")
            f.write(f"    Sharpe: {consolidated_metrics['is_sharpes'][i]:.2f}\n")

            # SD Filter Pass Rate
            if 'is_sd_pass_rates' in consolidated_metrics:
                f.write(f"    SD Pass Rate: {consolidated_metrics['is_sd_pass_rates'][i]:.2f}%\n")

            f.write(f"  Out-of-Sample Performance:\n")
            f.write(f"    Return: {consolidated_metrics['oos_returns'][i]:.2f}%\n")
            f.write(f"    Sharpe: {consolidated_metrics['oos_sharpes'][i]:.2f}\n")

            # SD Filter Pass Rate
            if 'oos_sd_pass_rates' in consolidated_metrics:
                f.write(f"    SD Pass Rate: {consolidated_metrics['oos_sd_pass_rates'][i]:.2f}%\n")

            f.write("\n")

        # Copula frequency summary
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("COPULA TYPE DISTRIBUTION:\n\n")

        for copula_type, count in consolidated_metrics.get('copula_frequencies', {}).items():
            percentage = count / sum(consolidated_metrics['copula_frequencies'].values()) * 100
            f.write(f"{copula_type}: {count} instances ({percentage:.1f}%)\n")


def main():
    """
    Main function to run the dynamic copula walk-forward optimization.
    Now ensures the SD filter is always applied.
    """
    # Get parameters
    params = get_parameters()

    # Set parameter ranges for optimization
    param_ranges = {
        'window_size': [20, 30, 40, 50, 60],
        'confidence_level': [0.80, 0.85, 0.90, 0.95, 0.99],
        'atr_multiplier': [1.5, 2.0, 2.5],
        'atr_on_equity': [False, True],
        'sd_window': [20],  # SD filter parameters
        'sd_lookback_windows': [5],
        'sd_threshold_ratio': [1.0]
    }

    # Ask if user wants to optimize SD filter parameters
    optimize_sd = input("\nDo you want to optimize SD filter parameters? (y/n): ").lower() == 'y'

    if optimize_sd:
        # Allow user to set SD parameter ranges
        print("\nEnter SD filter parameter ranges (comma-separated values):")

        sd_window_input = input("SD window sizes [20]: ") or "20"
        sd_window_values = [int(x.strip()) for x in sd_window_input.split(',')]
        param_ranges['sd_window'] = sd_window_values

        sd_lookback_input = input("SD lookback windows [5]: ") or "5"
        sd_lookback_values = [int(x.strip()) for x in sd_lookback_input.split(',')]
        param_ranges['sd_lookback_windows'] = sd_lookback_values

        sd_threshold_input = input("SD threshold ratios [1.0]: ") or "1.0"
        sd_threshold_values = [float(x.strip()) for x in sd_threshold_input.split(',')]
        param_ranges['sd_threshold_ratio'] = sd_threshold_values

        print(f"\nOptimizing with SD parameters:")
        print(f"  Window sizes: {param_ranges['sd_window']}")
        print(f"  Lookback windows: {param_ranges['sd_lookback_windows']}")
        print(f"  Threshold ratios: {param_ranges['sd_threshold_ratio']}")
    else:
        # Use fixed SD values from params
        param_ranges['sd_window'] = [params['sd_window']]
        param_ranges['sd_lookback_windows'] = [params['sd_lookback_windows']]
        param_ranges['sd_threshold_ratio'] = [params['sd_threshold_ratio']]

        print(f"\nUsing fixed SD parameters:")
        print(f"  Window size: {params['sd_window']}")
        print(f"  Lookback windows: {params['sd_lookback_windows']}")
        print(f"  Threshold ratio: {params['sd_threshold_ratio']}")

    # Ask user for dynamic copula parameters
    while True:
        try:
            copula_window_days = int(
                input("Enter the number of days for copula determination window (default 365): ") or "365")
            if copula_window_days < 30:
                print("Window should be at least 30 days. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    while True:
        try:
            update_frequency_days = int(
                input("Enter the frequency in days to update the copula model (default 30): ") or "30")
            if update_frequency_days < 1:
                print("Update frequency should be at least 1 day. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    while True:
        try:
            num_cycles = int(
                input("Enter the number of cycles for walk-forward optimization (4-10 recommended): ") or "4")
            if num_cycles < 1:
                print("Number of cycles must be at least 1. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Run dynamic walk-forward optimization
    results = dynamic_walk_forward_optimization(
        csv_file=params['csv_file'],
        date_col=params['date_col'],
        token1_col= "**********"
        token2_col= "**********"
        param_ranges=param_ranges,
        output_dir=params['output_dir'],
        optimization_goal='sharpe_ratio',
        min_trades=10,
        copula_window_days=copula_window_days,
        update_frequency_days=update_frequency_days,
        num_cycles=num_cycles
    )

    print(f"\nDynamic copula walk-forward optimization completed!")
    print(f"Check the output directory '{params['output_dir']}' for detailed results.")


if __name__ == "__main__":
    main()