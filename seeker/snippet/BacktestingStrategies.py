#date: 2025-03-05T17:03:45Z
#url: https://api.github.com/gists/0983c5f4a0af82dc3f19de9161af65f0
#owner: https://api.github.com/users/Clement1nes

import os
import pandas as pd
import numpy as np
import panel as pn
import datetime
import ta
import ccxt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from binance.client import Client
from binance.enums import *
from datetime import date, timedelta
import itertools
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Configure Panel theme and extensions
pn.extension('tabulator', sizing_mode='stretch_width')


class BaseStrategy(Strategy):
    """Base strategy with separate trailing stops for long and short positions"""

    sl_pct = 0.02  # Stop-loss percentage
    ts_long_pct = 0.015  # Trailing stop for long positions
    ts_short_pct = 0.02  # Trailing stop for short positions
    tp_pct = 0.03  # Take-profit percentage
    use_sl = True
    use_ts = True
    use_tp = True
    exit_on_reversal = True

    def init(self):
        self.sl_price = None
        self.ts_price = None
        self.tp_price = None
        self.peak_price = None

    def set_exit_prices(self, entry_price, position_type='long'):
        """Set stop loss, take profit, and trailing stop based on position type."""
        if position_type == 'long':
            self.sl_price = entry_price * (1 - self.sl_pct) if self.use_sl else None
            self.tp_price = entry_price * (1 + self.tp_pct) if self.use_tp else None
            self.peak_price = entry_price if self.use_ts else None
            self.ts_price = self.peak_price * (1 - self.ts_long_pct) if self.use_ts else None
        else:
            self.sl_price = entry_price * (1 + self.sl_pct) if self.use_sl else None
            self.tp_price = entry_price * (1 - self.tp_pct) if self.use_tp else None
            self.peak_price = entry_price if self.use_ts else None
            self.ts_price = self.peak_price * (1 + self.ts_short_pct) if self.use_ts else None

    def check_exits(self):
        """Checks and handles exit conditions for both long and short positions."""
        if not self.position:
            return
        current_price = self.data.Close[-1]
        position_closed = False  # Flag to check if position was closed in this method

        if self.position.is_long:
            # Update trailing stop for longs
            if self.use_ts and self.peak_price is not None and current_price > self.peak_price:
                self.peak_price = current_price
                self.ts_price = self.peak_price * (1 - self.ts_long_pct)

            # Check exit conditions for longs
            if (self.use_sl and self.sl_price is not None and current_price <= self.sl_price) or \
                    (self.use_ts and self.ts_price is not None and current_price <= self.ts_price) or \
                    (self.use_tp and self.tp_price is not None and current_price >= self.tp_price):
                self.position.close()
                position_closed = True

        else:  # Short position
            # Update trailing stop for shorts
            if self.use_ts and self.peak_price is not None and current_price < self.peak_price:
                self.peak_price = current_price
                self.ts_price = self.peak_price * (1 + self.ts_short_pct)

            # Check exit conditions for shorts
            if (self.use_sl and self.sl_price is not None and current_price >= self.sl_price) or \
                    (self.use_ts and self.ts_price is not None and current_price >= self.ts_price) or \
                    (self.use_tp and self.tp_price is not None and current_price <= self.tp_price):
                self.position.close()
                position_closed = True

        if position_closed:
            # Reset exit prices
            self.sl_price = None
            self.tp_price = None
            self.ts_price = None
            self.peak_price = None


class AdaptiveSupertrendStrategy(BaseStrategy):
    """
    Adaptive Strategy that combines Supertrend and RSI/Bollinger strategies with multi-timeframe support:
    - Uses Choppiness Index to determine market regime
    - Supertrend for trending markets (low choppiness)
    - RSI/Bollinger for choppy markets (high choppiness) with custom timeframe
    """
    # Choppiness Index parameters
    choppiness_period = 14
    choppiness_threshold = 61.8

    # Supertrend parameters
    atr_period = 10
    atr_multiplier = 3

    # RSI/Bollinger parameters
    rsi_period = 14
    bb_period = 20
    bb_std_dev = 2.0
    lookback_range = 3
    rsi_bb_timeframe = '1H'  # Added timeframe parameter

    def init(self):
        super().init()
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)

        # Calculate Choppiness Index for regime detection
        self.choppiness = self.I(self.calculate_choppiness, high, low, close, self.choppiness_period)

        # Initialize Supertrend components (always on original timeframe)
        self.supertrend = self.I(self.calculate_supertrend, high, low, close, self.atr_period, self.atr_multiplier)

        # Initialize RSI/Bollinger components with resampling
        if self.rsi_bb_timeframe != '0':  # '0' means use original timeframe
            # Convert timeframe string to pandas offset
            timeframe_map = {
                '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T',
                '1H': '1H', '2H': '2H', '4H': '4H', '6H': '6H', '8H': '8H', '12H': '12H',
                '1D': '1D'
            }
            offset = timeframe_map.get(self.rsi_bb_timeframe, '1H')

            # Resample close prices for RSI/Bollinger calculations
            resampled_close = close.resample(offset).last().ffill()
            resampled_high = high.resample(offset).max().ffill()
            resampled_low = low.resample(offset).min().ffill()

            # Calculate RSI on resampled timeframe
            self.rsi = self.I(ta.momentum.rsi, resampled_close, self.rsi_period)

            # Calculate Bollinger Bands on resampled timeframe
            bb_indicator = ta.volatility.BollingerBands(
                close=resampled_close,
                window=self.bb_period,
                window_dev=self.bb_std_dev
            )
            self.bb_upper = self.I(lambda: bb_indicator.bollinger_hband())
            self.bb_lower = self.I(lambda: bb_indicator.bollinger_lband())

        else:
            # Calculate on original timeframe
            self.rsi = self.I(ta.momentum.rsi, close, self.rsi_period)
            bb_indicator = ta.volatility.BollingerBands(
                close=close,
                window=self.bb_period,
                window_dev=self.bb_std_dev
            )
            self.bb_upper = self.I(lambda: bb_indicator.bollinger_hband())
            self.bb_lower = self.I(lambda: bb_indicator.bollinger_lband())

        self.current_regime = None
        self.regime_change = False

    def calculate_choppiness(self, high, low, close, period):
        """Calculate Choppiness Index"""
        # Convert inputs to pandas Series
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

        # Calculate ATR
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).sum()

        # Calculate highest high and lowest low
        high_low = high.rolling(period).max() - low.rolling(period).min()

        # Calculate Choppiness Index
        chop = 100 * np.log10(atr / high_low) / np.log10(period)

        # Replace infinite values with 0 or 100
        chop = np.where(np.isinf(chop), 100, chop)
        chop = np.where(np.isnan(chop), 50, chop)  # Use 50 as neutral value for NaN

        return chop

    def calculate_supertrend(self, high, low, close, atr_period, atr_multiplier):
        """Calculate Supertrend"""
        # Calculate ATR
        atr = ta.volatility.average_true_range(high, low, close, window=atr_period)

        # Calculate basic upper and lower bands
        upperband = ((high + low) / 2) + (atr_multiplier * atr)
        lowerband = ((high + low) / 2) - (atr_multiplier * atr)

        # Initialize Supertrend
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        # First value
        supertrend.iloc[0] = upperband.iloc[0]
        direction.iloc[0] = 1

        # Calculate Supertrend
        for i in range(1, len(close)):
            if upperband.iloc[i] < supertrend.iloc[i - 1] or close.iloc[i - 1] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = min(upperband.iloc[i], supertrend.iloc[i - 1])
            else:
                supertrend.iloc[i] = max(lowerband.iloc[i], supertrend.iloc[i - 1])

            if direction.iloc[i - 1] == 1:  # Previous trend was up
                if close.iloc[i] <= supertrend.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upperband.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lowerband.iloc[i]
            else:  # Previous trend was down
                if close.iloc[i] >= supertrend.iloc[i]:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lowerband.iloc[i]
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upperband.iloc[i]

        return supertrend

    def check_regime(self):
        """Determine market regime based on Choppiness Index"""
        current_chop = self.choppiness[-1]
        previous_regime = self.current_regime

        # Update regime (high choppiness = choppy market)
        self.current_regime = 'choppy' if current_chop > self.choppiness_threshold else 'trend'

        # Check for regime change
        self.regime_change = previous_regime != self.current_regime

        return self.current_regime

    def supertrend_signals(self):
        """Generate Supertrend signals for trending market"""
        if len(self.data) < 2:
            return None

        close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        supertrend = self.supertrend[-1]
        prev_supertrend = self.supertrend[-2]

        # Buy when price crosses above Supertrend
        if prev_close <= prev_supertrend and close > supertrend:
            return 'buy'
        # Sell when price crosses below Supertrend
        elif prev_close >= prev_supertrend and close < supertrend:
            return 'sell'
        return None

    def rsi_bollinger_signals(self):
        """Generate RSI/Bollinger signals for choppy market"""
        recent_highs = self.data.High[-self.lookback_range:]
        recent_lows = self.data.Low[-self.lookback_range:]

        price_touched_upper = any(recent_highs >= self.bb_upper[-1])
        price_touched_lower = any(recent_lows <= self.bb_lower[-1])

        if price_touched_lower and self.rsi[-1] < 30:
            return 'buy'
        elif price_touched_upper and self.rsi[-1] > 70:
            return 'sell'
        return None

    def next(self):
        if len(self.data) < max(self.choppiness_period, self.atr_period, self.bb_period) + 2:
            return

        # Check exits first
        self.check_exits()

        # Determine market regime
        regime = self.check_regime()

        # If regime changed, close existing positions
        if self.regime_change and self.position:
            self.position.close()
            return

        price = self.data.Close[-1]

        # Generate signals based on current regime
        if regime == 'trend':
            signal = self.supertrend_signals()
        else:  # choppy
            signal = self.rsi_bollinger_signals()

        # Execute trades
        if signal == 'buy':
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy()
                self.set_exit_prices(price, 'long')
        elif signal == 'sell':
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell()
                self.set_exit_prices(price, 'short')

class BBATRIntradayStrategy(BaseStrategy):
    """ATR Bollinger Bands Strategy"""

    # Strategy parameters
    bblength = 55
    bbstdev = 0.3
    lena = 20
    mult = 0.6
    use_filter = True
    use_atr_bb = True

    def init(self):
        super().init()

        price = pd.Series(self.data.Close)
        hl2 = (pd.Series(self.data.High) + pd.Series(self.data.Low)) / 2

        # Calculate basis - either filtered or SMA
        if self.use_filter:
            self.basis = self.I(self.calculate_filter, hl2)
        else:
            self.basis = self.I(ta.trend.sma_indicator, price, self.bblength)

        # Calculate ATR (custom implementation matching TradingView)
        self.atr = self.I(self.calculate_custom_atr, self.data.High, self.data.Low, self.data.Close, self.lena)

        # Calculate standard deviation and bands
        self.std = self.I(lambda: price.rolling(window=self.bblength).std())

        # Calculate final bands based on settings
        self.upper_band = self.I(lambda: self.basis + (self.bbstdev * self.std if not self.use_atr_bb
                                                       else self.mult * self.atr * self.basis))
        self.lower_band = self.I(lambda: self.basis - (self.bbstdev * self.std if not self.use_atr_bb
                                                       else self.mult * self.atr * self.basis))

    def calculate_filter(self, price):
        """Implements the custom filter from TradingView code"""
        gamma = 0.87
        length = len(price)
        l0 = np.zeros(length)
        l1 = np.zeros(length)
        l2 = np.zeros(length)
        l3 = np.zeros(length)

        # Initial values
        l0[0] = price.iloc[0]
        l1[0] = l0[0]
        l2[0] = l1[0]
        l3[0] = l2[0]

        # Calculate filtered values
        for i in range(1, length):
            l0[i] = (1 - gamma) * price.iloc[i] + gamma * l0[i - 1]
            l1[i] = -gamma * l0[i] + l0[i - 1] + gamma * l1[i - 1]
            l2[i] = -gamma * l1[i] + l1[i - 1] + gamma * l2[i - 1]
            l3[i] = -gamma * l2[i] + l2[i - 1] + gamma * l3[i - 1]

        return pd.Series((l0 + 2 * l1 + 2 * l2 + l3) / 6, index=price.index)

    def calculate_custom_atr(self, high, low, close, period):
        """Implements the custom ATR calculation from TradingView code"""
        lh = high - low
        pc = close.shift(1)
        hc = abs(high - pc)
        lc = abs(low - pc)

        # Find maximum of the three measures
        tr = pd.concat([lh, hc, lc], axis=1).max(axis=1)

        # Calculate ATR using exponential smoothing
        alpha = 2 / (period + 1)
        atr = tr.ewm(alpha=alpha, adjust=False).mean()

        return atr

    def next(self):
        if len(self.data) <= self.bblength:
            return

        price = self.data.Close[-1]

        if pd.isna(self.upper_band[-1]) or pd.isna(self.lower_band[-1]):
            return

        # Entry logic
        if not self.position:
            if price > self.upper_band[-1]:
                self.buy()
                self.set_exit_prices(price, 'long')
            elif price < self.lower_band[-1]:
                self.sell()
                self.set_exit_prices(price, 'short')

        # Exit logic handled by BaseStrategy's check_exits()
        self.check_exits()


class RSITrendReversalStrategy(BaseStrategy):
    """RSI Trend Reversal Strategy with Heikin-Ashi and custom RSI level calculations"""

    # Strategy parameters
    rsi_length = 8  # RSI length
    rsi_mult = 1.5  # RSI multiplier
    lookback = 1  # Delay to prevent idealization
    sltp = 10  # Minimum difference
    use_fixed_sltp = True  # New parameter to toggle fixed SLTP

    def init(self):
        super().init()
        # Calculate Heikin-Ashi prices
        self.ha_close = self.I(self.calculate_heikin_ashi_close)

        # Initialize state variables first
        self.prev_direction = 1  # Start with upward direction
        self.bars_since_signal = 0

        # Calculate RSI level after initializing state variables
        self.rsi_level = self.I(self.calculate_rsi_level)

    def calculate_heikin_ashi_close(self):
        """Calculate Heikin-Ashi close prices"""
        df = pd.Series(self.data.Close)
        open_series = pd.Series(self.data.Open)
        high_series = pd.Series(self.data.High)
        low_series = pd.Series(self.data.Low)

        ha_close = (open_series + high_series + low_series + df) / 4
        return ha_close

    def calculate_rsi_level(self):
        """Calculate RSI-based threshold levels"""
        # Get price data
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Calculate RSI
        rsi = ta.momentum.rsi(close, self.rsi_length)

        # Calculate ATR
        atr = ta.volatility.average_true_range(high, low, close, self.rsi_length)

        # Initialize arrays
        thresholds = np.zeros(len(close))

        # Calculate thresholds
        for i in range(self.rsi_length, len(close)):
            # Get RSI and ATR values
            curr_rsi = max(0.1, min(99.9, rsi[i]))  # Bound RSI to prevent extreme values
            curr_atr = atr[i]
            curr_close = close[i]

            # Calculate RSI-based multiplier
            rsi_mult = 1 / (curr_rsi / 100) if curr_close > thresholds[i - 1] else 1 / ((100 - curr_rsi) / 100)

            # Calculate threshold
            atr_component = curr_atr * self.rsi_mult * rsi_mult
            threshold = curr_close

            if curr_close > thresholds[i - 1]:
                if self.use_fixed_sltp:
                    threshold = max(curr_close - atr_component, curr_close * (1 - self.sltp / 100))
                else:
                    threshold = curr_close - atr_component
            else:
                if self.use_fixed_sltp:
                    threshold = min(curr_close + atr_component, curr_close * (1 + self.sltp / 100))
                else:
                    threshold = curr_close + atr_component

            thresholds[i] = threshold

        return pd.Series(thresholds)

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.rsi_length + 2:
            return

        # Get current values
        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        threshold = self.rsi_level[-1]
        prev_threshold = self.rsi_level[-2]

        # Generate signals
        buy_signal = prev_price <= prev_threshold and price > threshold
        sell_signal = prev_price >= prev_threshold and price < threshold

        # Apply signals with lookback delay
        if len(self.data) >= self.lookback:
            # Check exits first
            self.check_exits()

            # Entry logic
            if buy_signal:
                if self.position.is_short:
                    self.position.close()
                if not self.position.is_long:
                    self.buy()
                    self.set_exit_prices(price, 'long')

            elif sell_signal:
                if self.position.is_long:
                    self.position.close()
                if not self.position.is_short:
                    self.sell()
                    self.set_exit_prices(price, 'short')


class LiquidationStrategy(BaseStrategy):
    """
    Strategy that trades based on liquidation data:
    - Enters longs when there are large long liquidations (reversal expectation)
    - Uses volume and price action confirmation
    - Includes dynamic position sizing based on liquidation volume
    """
    # Liquidation parameters
    liq_threshold = 1000000  # Minimum USD value of liquidations to trigger signal
    liq_window = 12  # Rolling window to sum liquidations (periods)
    liq_spike_mult = 2.0  # How many times above average to consider a spike

    # Confirmation parameters
    volume_mult = 1.5  # Required volume spike multiplier
    price_rebound = 0.01  # Required price rebound percentage
    rsi_period = 14  # RSI period for oversold confirmation
    rsi_oversold = 30  # RSI oversold threshold

    def init(self):
        super().init()
        # Price/volume data
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)

        # Calculate indicators
        self.volume_ma = self.I(ta.trend.sma_indicator, volume, self.liq_window)
        self.rsi = self.I(ta.momentum.rsi, close, self.rsi_period)

        # Store liquidation data as indicator
        self.long_liqs = self.I(self.calculate_liquidations, self.data.Long_Liquidations)
        self.liq_ma = self.I(ta.trend.sma_indicator, self.long_liqs, self.liq_window)

    def calculate_liquidations(self, liquidation_series):
        """Calculate rolling sum of liquidations"""
        return pd.Series(liquidation_series).rolling(self.liq_window).sum()

    def next(self):
        if len(self.data) < self.liq_window + 2:
            return

        # Check exits first
        self.check_exits()

        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        volume = self.data.Volume[-1]

        # Get current liquidation metrics
        current_liqs = self.long_liqs[-1]
        avg_liqs = self.liq_ma[-1]

        # Check for liquidation spike
        liq_spike = (current_liqs > self.liq_threshold and
                     current_liqs > avg_liqs * self.liq_spike_mult)

        # Confirmation conditions
        volume_spike = volume > self.volume_ma[-1] * self.volume_mult
        price_bounce = price > prev_price * (1 + self.price_rebound)
        oversold = self.rsi[-1] < self.rsi_oversold

        # Long entry logic
        if not self.position:
            if (liq_spike and
                    volume_spike and
                    price_bounce and
                    oversold):
                # Size position based on liquidation volume
                size = self.calculate_position_size(current_liqs)
                self.buy(size=size)
                self.set_exit_prices(price, 'long')

    def calculate_position_size(self, liquidation_amount):
        """Calculate position size based on liquidation volume"""
        # Base position size as percentage of equity
        base_size = self.equity * 0.02

        # Scale up size with larger liquidations, cap at 5x
        liq_scale = min(5.0, liquidation_amount / self.liq_threshold)
        position_size = base_size * liq_scale

        # Ensure minimum position size
        return max(1.0, position_size)

    def validate_data(self, data):
        """Ensure required columns exist"""
        required = ['Long_Liquidations']
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")




class AdaptiveRSIVolumeStrategy(BaseStrategy):
    """
    Adaptive RSI Volume Strategy with smart filters and dynamic risk management.
    Combines RSI, volume analysis, and market volatility filters for improved trade selection.
    """

    # Core parameters
    rsi_length = 8  # RSI length
    rsi_mult = 1.5  # RSI multiplier
    lookback = 1  # Delay to prevent idealization
    sltp = 10  # Minimum difference

    # Filter parameters
    atr_period = 14  # ATR period for volatility filter
    volatility_mult = 1.5  # Multiplier for volatility threshold
    trend_ema = 200  # EMA period for trend filter
    min_volume_mult = 1.2  # Minimum volume multiplier
    volume_ma_period = 20  # Volume MA period
    rsi_overbought = 70  # RSI overbought level
    rsi_oversold = 30  # RSI oversold level
    min_swing_pct = 0.5  # Minimum swing percentage
    max_risk_per_trade = 0.02  # Maximum risk per trade (2%)

    def init(self):
        super().init()
        # Calculate Heikin-Ashi prices
        self.ha_close = self.I(self.calculate_heikin_ashi_close)

        # Initialize state variables
        self.prev_direction = 1
        self.bars_since_signal = 0

        # Price series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Core indicators
        self.rsi_level = self.I(self.calculate_rsi_level)
        self.rsi = self.I(ta.momentum.rsi, close, self.rsi_length)

        # Volatility and trend indicators
        self.atr = self.I(ta.volatility.average_true_range, high, low, close, self.atr_period)
        self.ema = self.I(ta.trend.ema_indicator, close, self.trend_ema)
        self.volume_ma = self.I(ta.trend.sma_indicator, volume, self.volume_ma_period)

        # Trading state tracking
        self.consecutive_losses = 0
        self.last_trade_profit = None
        self.current_trade_entry = None
        self.peak_price = None
        self.valley_price = None

    def calculate_heikin_ashi_close(self):
        """Calculate Heikin-Ashi close prices for smoother trend identification"""
        df = pd.Series(self.data.Close)
        open_series = pd.Series(self.data.Open)
        high_series = pd.Series(self.data.High)
        low_series = pd.Series(self.data.Low)

        ha_close = (open_series + high_series + low_series + df) / 4
        return ha_close

    def calculate_rsi_level(self):
        """Calculate adaptive RSI thresholds based on market conditions"""
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        rsi = ta.momentum.rsi(close, self.rsi_length)
        atr = ta.volatility.average_true_range(high, low, close, self.rsi_length)
        thresholds = np.zeros(len(close))

        for i in range(self.rsi_length, len(close)):
            curr_rsi = max(0.1, min(99.9, rsi[i]))
            curr_atr = atr[i]
            curr_close = close[i]

            # Adaptive RSI multiplier
            rsi_mult = 1 / (curr_rsi / 100) if curr_close > thresholds[i - 1] else 1 / ((100 - curr_rsi) / 100)

            # Threshold calculation with ATR component
            atr_component = curr_atr * self.rsi_mult * rsi_mult
            threshold = curr_close

            if curr_close > thresholds[i - 1]:
                threshold = max(curr_close - atr_component, curr_close * (1 - self.sltp / 100))
            else:
                threshold = min(curr_close + atr_component, curr_close * (1 + self.sltp / 100))

            thresholds[i] = threshold

        return pd.Series(thresholds)

    def calculate_position_size(self, price):
        """Dynamic position sizing based on market conditions and account risk"""
        if not self.atr[-1]:
            return 1

        # ATR-based stop loss distance
        stop_distance = self.atr[-1] * self.volatility_mult

        # Risk-based position sizing
        account_value = self.equity
        risk_amount = account_value * self.max_risk_per_trade

        # Adjust risk based on market conditions and past performance
        if self.consecutive_losses >= 2:
            risk_amount *= 0.5  # Reduce risk after losses

        # Calculate final position size
        position_size = risk_amount / stop_distance
        return max(1, int(position_size))

    def check_market_conditions(self, price, signal_type='buy'):
        """Comprehensive market condition analysis"""

        # 1. Volatility Check
        current_volatility = self.atr[-1]
        avg_volatility = sum(self.atr[-20:]) / 20
        if current_volatility > avg_volatility * self.volatility_mult:
            return False

        # 2. Trend Analysis
        trend_aligned = (
                (signal_type == 'buy' and price > self.ema[-1]) or
                (signal_type == 'sell' and price < self.ema[-1])
        )
        if not trend_aligned:
            return False

        # 3. Volume Analysis
        if self.data.Volume[-1] < self.volume_ma[-1] * self.min_volume_mult:
            return False

        # 4. RSI Check
        current_rsi = self.rsi[-1]
        if signal_type == 'buy' and current_rsi > self.rsi_overbought:
            return False
        if signal_type == 'sell' and current_rsi < self.rsi_oversold:
            return False

        # 5. Price Swing Analysis
        recent_high = max(self.data.High[-5:])
        recent_low = min(self.data.Low[-5:])
        swing_size = (recent_high - recent_low) / recent_low * 100
        if swing_size < self.min_swing_pct:
            return False

        # 6. Risk Management Check
        if self.consecutive_losses >= 3:
            return False

        return True

    def next(self):
        if len(self.data) < max(self.rsi_length + 2, self.trend_ema):
            return

        # Get current market data
        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        threshold = self.rsi_level[-1]
        prev_threshold = self.rsi_level[-2]

        # Update trade tracking
        if not self.position and self.current_trade_entry is not None:
            if self.last_trade_profit and self.last_trade_profit < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            self.current_trade_entry = None
            self.last_trade_profit = None

        # Generate trade signals
        buy_signal = prev_price <= prev_threshold and price > threshold
        sell_signal = prev_price >= prev_threshold and price < threshold

        # Apply trading logic with filters
        if len(self.data) >= self.lookback:
            self.check_exits()

            if buy_signal and self.check_market_conditions(price, 'buy'):
                if self.position.is_short:
                    self.position.close()
                if not self.position.is_long:
                    size = self.calculate_position_size(price)
                    self.buy(size=size)
                    self.current_trade_entry = price
                    self.set_exit_prices(price, 'long')
                    self.peak_price = price

            elif sell_signal and self.check_market_conditions(price, 'sell'):
                if self.position.is_long:
                    self.position.close()
                if not self.position.is_short:
                    size = self.calculate_position_size(price)
                    self.sell(size=size)
                    self.current_trade_entry = price
                    self.set_exit_prices(price, 'short')
                    self.valley_price = price

        # Update trade performance tracking
        if self.position and self.current_trade_entry is not None:
            if self.position.is_long:
                self.last_trade_profit = (price - self.current_trade_entry) / self.current_trade_entry
                self.peak_price = max(self.peak_price, price) if self.peak_price else price
            else:
                self.last_trade_profit = (self.current_trade_entry - price) / self.current_trade_entry
                self.valley_price = min(self.valley_price, price) if self.valley_price else price

class IchimokuTKCrossStrategy(BaseStrategy):
    """
    Ichimoku TK Cross Strategy with filters
    - Uses Tenkan-sen (Conversion Line) and Kijun-sen (Base Line) crossovers
    - Additional filters: RSI, ADX, and Cloud position
    - Trades both long and short directions
    """

    # Ichimoku parameters
    conversion_periods = 20  # Tenkan-sen period
    base_periods = 60  # Kijun-sen period
    lagging_span2_periods = 120  # Senkou Span B period
    displacement = 30  # Chikou Span displacement

    # Filter parameters
    adx_period = 14  # ADX period
    adx_threshold = 25  # Minimum ADX value for trend strength
    rsi_period = 14  # RSI period
    rsi_overbought = 70  # RSI overbought threshold
    rsi_oversold = 30  # RSI oversold threshold
    use_cloud_filter = True  # Whether to use cloud position as filter

    def init(self):
        super().init()

        # Price series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Calculate Ichimoku components
        self.conversion_line = self.I(self.donchian_avg, high, low, self.conversion_periods)
        self.base_line = self.I(self.donchian_avg, high, low, self.base_periods)

        # Calculate Lead lines (Cloud)
        self.lead_line1 = self.I(lambda: (self.conversion_line + self.base_line) / 2)
        self.lead_line2 = self.I(self.donchian_avg, high, low, self.lagging_span2_periods)

        # Calculate ADX
        self.adx = self.I(ta.trend.adx, high, low, close, self.adx_period)

        # Calculate RSI
        self.rsi = self.I(ta.momentum.rsi, close, self.rsi_period)

    def donchian_avg(self, high, low, periods):
        """Calculate Donchian Channel Average"""
        upper = high.rolling(periods).max()
        lower = low.rolling(periods).min()
        return (upper + lower) / 2

    def check_filters(self, direction='long'):
        """Check if all filters allow trading"""
        # Get current values
        price = self.data.Close[-1]
        adx_value = self.adx[-1]
        rsi_value = self.rsi[-1]

        # ADX filter - Check for strong trend
        trend_strength = adx_value > self.adx_threshold

        # RSI filter - Different conditions for long/short
        if direction == 'long':
            rsi_filter = rsi_value < self.rsi_overbought
        else:  # short
            rsi_filter = rsi_value > self.rsi_oversold

        # Cloud filter - Price position relative to cloud
        if self.use_cloud_filter:
            current_lead1 = self.lead_line1[-self.displacement] if len(self.lead_line1) > self.displacement else None
            current_lead2 = self.lead_line2[-self.displacement] if len(self.lead_line2) > self.displacement else None

            if current_lead1 is not None and current_lead2 is not None:
                cloud_top = max(current_lead1, current_lead2)
                cloud_bottom = min(current_lead1, current_lead2)

                if direction == 'long':
                    cloud_filter = price > cloud_top
                else:  # short
                    cloud_filter = price < cloud_bottom
            else:
                cloud_filter = True  # If not enough data, ignore cloud filter
        else:
            cloud_filter = True

        return trend_strength and rsi_filter and cloud_filter

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.conversion_periods, self.base_periods,
                                self.lagging_span2_periods, self.adx_period,
                                self.rsi_period):
            return

        # Check exits first
        self.check_exits()

        # Get prices and compute crosses
        price = self.data.Close[-1]

        tk_cross_up = (self.conversion_line[-1] > self.base_line[-1] and
                       self.conversion_line[-2] <= self.base_line[-2])
        tk_cross_down = (self.conversion_line[-1] < self.base_line[-1] and
                         self.conversion_line[-2] >= self.base_line[-2])

        # Position management
        if not self.position:
            # Check for long entry
            if tk_cross_up and self.check_filters('long'):
                self.buy()
                self.set_exit_prices(price, 'long')

            # Check for short entry
            elif tk_cross_down and self.check_filters('short'):
                self.sell()
                self.set_exit_prices(price, 'short')

        # Exit conditions
        elif self.position.is_long:
            if tk_cross_down:
                self.position.close()
                if self.check_filters('short'):  # Reverse position if filters allow
                    self.sell()
                    self.set_exit_prices(price, 'short')

        elif self.position.is_short:
            if tk_cross_up:
                self.position.close()
                if self.check_filters('long'):  # Reverse position if filters allow
                    self.buy()
                    self.set_exit_prices(price, 'long')
class IchimokuEMAStrategy(BaseStrategy):
    """
    Ichimoku EMA Cross Strategy
    - Uses Ichimoku Cloud components and EMA 200
    - Entry on TK Cross above EMA200
    - Exit on TK Cross below
    """

    # Default parameters
    conversion_periods = 20  # Tenkan-sen (Conversion Line) period
    base_periods = 60  # Kijun-sen (Base Line) period
    lagging_span2_periods = 120  # Senkou Span B period
    displacement = 30  # Displacement (Chikou Span)
    ema_length = 200  # EMA period

    def init(self):
        super().init()

        # Price series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Calculate EMA 200
        self.ema200 = self.I(ta.trend.ema_indicator, close, self.ema_length)

        # Calculate Ichimoku components
        self.conversion_line = self.I(self.donchian_avg, high, low, self.conversion_periods)
        self.base_line = self.I(self.donchian_avg, high, low, self.base_periods)

        # Calculate Lead lines
        self.lead_line1 = self.I(lambda: (self.conversion_line + self.base_line) / 2)
        self.lead_line2 = self.I(self.donchian_avg, high, low, self.lagging_span2_periods)

        # Lagging Span (shifted close price)
        self.lagging_span = close.shift(-self.displacement)

    def donchian_avg(self, high, low, periods):
        """Calculate Donchian Channel Average"""
        upper = high.rolling(periods).max()
        lower = low.rolling(periods).min()
        return (upper + lower) / 2

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.conversion_periods, self.base_periods,
                                self.lagging_span2_periods, self.ema_length):
            return

        # Check exits first
        self.check_exits()

        # Get current values
        price = self.data.Close[-1]

        # Entry/exit logic
        tk_cross_up = (self.conversion_line[-1] > self.base_line[-1] and
                       self.conversion_line[-2] <= self.base_line[-2])
        tk_cross_down = (self.conversion_line[-1] < self.base_line[-1] and
                         self.conversion_line[-2] >= self.base_line[-2])

        above_ema = price > self.ema200[-1]

        # Long entry conditions
        if not self.position:
            if tk_cross_up and above_ema:
                self.buy()
                self.set_exit_prices(price, 'long')

        # Exit conditions
        elif self.position.is_long:
            if tk_cross_down:
                self.position.close()


class SRVWAPStrategy(BaseStrategy):
    """
    Support/Resistance VWAP Strategy
    - Uses SR levels for entries and stops
    - VWAP color and position for confirmation
    - Combines SR retest signals with VWAP trend
    """

    # SR Parameters
    pivot_length = 15  # Length for pivot points
    sr_strength = 2  # Minimum strength required for SR level
    atr_period = 20  # Period for ATR calculation
    zone_size_atr = 0.075  # Size of SR zones in ATR multiplier
    min_bars_between_signals = 15  # Minimum bars between signals

    # VWAP Parameters
    vwap_ma_length = 20  # Length for VWAP smoothing
    min_vwap_trend_strength = 0.001  # Minimum VWAP slope for trend confirmation

    def init(self):
        super().init()

        # Price series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Calculate ATR and VWAP
        self.atr = self.I(ta.volatility.average_true_range, high, low, close, self.atr_period)
        self.vwap = self.I(self.calculate_vwap, close, volume)

        # Track SR levels
        self.sr_levels = []  # List to store SR levels
        self.last_signal_bar = 0

    def calculate_vwap(self, close, volume):
        """Calculate VWAP"""
        typical_price = close
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    def detect_sr_level(self):
        """Detect new support/resistance levels"""
        high = self.data.High[-self.pivot_length:]
        low = self.data.Low[-self.pivot_length:]

        # Detect pivot highs
        if high[-self.pivot_length // 2] == max(high):
            return {'price': high[-self.pivot_length // 2], 'type': 'resistance'}

        # Detect pivot lows
        if low[-self.pivot_length // 2] == min(low):
            return {'price': low[-self.pivot_length // 2], 'type': 'support'}

        return None

    def is_retesting_level(self, price, level, level_type):
        """Check if price is retesting an SR level"""
        zone_size = self.atr[-1] * self.zone_size_atr

        if level_type == 'resistance':
            return abs(price - level) <= zone_size and price < level
        else:
            return abs(price - level) <= zone_size and price > level

    def check_vwap_trend(self):
        """Check VWAP trend direction and strength"""
        vwap_slope = self.vwap[-1] - self.vwap[-2]

        if abs(vwap_slope) < self.min_vwap_trend_strength:
            return None

        return 'up' if vwap_slope > 0 else 'down'

    def price_vs_vwap(self):
        """Check price position relative to VWAP"""
        price = self.data.Close[-1]
        return 'above' if price > self.vwap[-1] else 'below'

    def next(self):
        if len(self.data) < self.pivot_length:
            return

        self.check_exits()

        current_bar = len(self.data)
        price = self.data.Close[-1]

        # Update SR levels
        sr_level = self.detect_sr_level()
        if sr_level:
            self.sr_levels.append(sr_level)

        # Clean old levels
        self.sr_levels = [level for level in self.sr_levels
                          if abs(price - level['price']) <= self.atr[-1] * 5]

        # Skip if too soon after last signal
        if current_bar - self.last_signal_bar < self.min_bars_between_signals:
            return

        vwap_trend = self.check_vwap_trend()
        price_position = self.price_vs_vwap()

        # Trading logic
        if not self.position:
            # Look for entries
            for level in self.sr_levels:
                # Long setup: Retesting support, price above VWAP, VWAP trending up
                if (level['type'] == 'support' and
                        self.is_retesting_level(price, level['price'], level['type']) and
                        price_position == 'above' and
                        vwap_trend == 'up'):

                    self.buy()
                    # Set stop below the support level
                    stop_price = level['price'] - self.atr[-1] * 0.5
                    self.sl_price = stop_price
                    self.last_signal_bar = current_bar
                    break

                # Short setup: Retesting resistance, price below VWAP, VWAP trending down
                elif (level['type'] == 'resistance' and
                      self.is_retesting_level(price, level['price'], level['type']) and
                      price_position == 'below' and
                      vwap_trend == 'down'):

                    self.sell()
                    # Set stop above the resistance level
                    stop_price = level['price'] + self.atr[-1] * 0.5
                    self.sl_price = stop_price
                    self.last_signal_bar = current_bar
                    break

        # Exit conditions
        elif self.position.is_long:
            # Exit if price crosses below VWAP while VWAP is trending down
            if price_position == 'below' and vwap_trend == 'down':
                self.position.close()

        elif self.position.is_short:
            # Exit if price crosses above VWAP while VWAP is trending up
            if price_position == 'above' and vwap_trend == 'up':
                self.position.close()

class MultiIndicatorMomentumStrategy(BaseStrategy):
    """
    Multi-Indicator Momentum Strategy combining SMA, MACD, RSI, and Bollinger Bands
    with ATR-based risk management for both long and short positions
    """

    # SMA Parameters
    sma_period = 20

    # MACD Parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # RSI Parameters
    rsi_period = 14
    rsi_lower = 40  # Long entry lower bound
    rsi_upper = 70  # Long entry upper bound
    rsi_short_lower = 30  # Short entry lower bound
    rsi_short_upper = 60  # Short entry upper bound
    rsi_exit_lower = 30  # Exit long position below this
    rsi_exit_upper = 70  # Exit short position above this

    # Bollinger Bands Parameters
    bb_period = 20
    bb_std = 2.0

    # ATR Risk Management
    atr_period = 14
    atr_stop_multiplier = 2.0
    max_drawdown_pct = 15.0  # Maximum allowed drawdown percentage

    def init(self):
        super().init()

        # Price series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Calculate SMA
        self.sma = self.I(ta.trend.sma_indicator, close, self.sma_period)

        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            close=close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        self.macd_line = self.I(lambda: macd_indicator.macd())
        self.macd_signal_line = self.I(lambda: macd_indicator.macd_signal())

        # Calculate RSI
        self.rsi = self.I(ta.momentum.rsi, close, self.rsi_period)

        # Calculate Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            close=close,
            window=self.bb_period,
            window_dev=self.bb_std
        )
        self.bb_upper = self.I(lambda: bb_indicator.bollinger_hband())
        self.bb_lower = self.I(lambda: bb_indicator.bollinger_lband())

        # Calculate ATR for position sizing and stops
        self.atr = self.I(ta.volatility.average_true_range, high, low, close, self.atr_period)

        # Track equity curve for drawdown monitoring
        self.peak_equity = self.equity

        # Track last signal for stop loss
        self.last_signal = None
        self.signal_price = None

    def check_macd_crossover(self):
        """Check for MACD line crossing above/below signal line"""
        if self.macd_line[-2] <= self.macd_signal_line[-2] and self.macd_line[-1] > self.macd_signal_line[-1]:
            return 'buy'
        elif self.macd_line[-2] >= self.macd_signal_line[-2] and self.macd_line[-1] < self.macd_signal_line[-1]:
            return 'sell'
        return None

    def check_sma_cross(self):
        """Check for price crossing above/below SMA"""
        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]

        if prev_price <= self.sma[-2] and price > self.sma[-1]:
            return 'buy'
        elif prev_price >= self.sma[-2] and price < self.sma[-1]:
            return 'sell'
        return None

    def calculate_position_size(self):
        """Calculate position size based on ATR"""
        if not self.atr[-1] or self.atr[-1] == 0:
            return 0

        risk_per_trade = self.equity * 0.02  # Risk 2% per trade
        stop_distance = self.atr[-1] * self.atr_stop_multiplier

        if stop_distance == 0:
            return 0

        position_size = risk_per_trade / stop_distance
        return max(1, int(position_size))  # Ensure at least 1 unit

    def check_drawdown(self):
        """Monitor drawdown and return True if exceeded"""
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        if current_drawdown > self.max_drawdown_pct:
            return True
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        return False

    def next(self):
        # Check exits first
        self.check_exits()

        # Skip if not enough data
        if len(self.data) < max(self.sma_period, self.macd_slow, self.rsi_period, self.bb_period):
            return

        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Get current values
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        current_atr = self.atr[-1]

        # Check for exit conditions first
        if self.position:
            # Check drawdown
            if self.check_drawdown():
                self.position.close()
                return

            # Exit long position
            if self.position.is_long:
                if (price < self.sma[-1] or  # Price below SMA
                        price <= self.bb_lower[-1] or  # Price breaches lower BB
                        self.check_macd_crossover() == 'sell' or  # MACD cross down
                        rsi <= self.rsi_exit_lower or  # RSI oversold
                        rsi >= self.rsi_exit_upper):  # RSI overbought
                    self.position.close()
                    return

            # Exit short position
            elif self.position.is_short:
                if (price > self.sma[-1] or  # Price above SMA
                        price >= self.bb_upper[-1] or  # Price breaches upper BB
                        self.check_macd_crossover() == 'buy' or  # MACD cross up
                        rsi >= self.rsi_exit_upper or  # RSI overbought
                        rsi <= self.rsi_exit_lower):  # RSI oversold
                    self.position.close()
                    return

        # Entry signals
        if not self.position:
            macd_signal = self.check_macd_crossover()
            sma_signal = self.check_sma_cross()

            # Long entry conditions
            if (macd_signal == 'buy' and
                    sma_signal == 'buy' and
                    self.rsi_lower <= rsi <= self.rsi_upper):

                size = self.calculate_position_size()
                if size > 0:
                    self.buy(size=size)
                    stop_price = price - current_atr * self.atr_stop_multiplier
                    self.set_exit_prices(price, 'long')

            # Short entry conditions
            elif (macd_signal == 'sell' and
                  sma_signal == 'sell' and
                  self.rsi_short_lower <= rsi <= self.rsi_short_upper):

                size = self.calculate_position_size()
                if size > 0:
                    self.sell(size=size)
                    stop_price = price + current_atr * self.atr_stop_multiplier
                    self.set_exit_prices(price, 'short')

class AlligatorStrategy(BaseStrategy):
    """Alligator Strategy with ADX filter"""

    # Define parameters
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_offset = 8
    teeth_offset = 5
    lips_offset = 3
    adx_period = 14
    adx_threshold = 25
    atr_period = 14
    atr_multiplier = 2.0

    # Risk parameters
    sl_pct = 0.02  # Stop-loss percentage
    ts_pct = 0.015  # Trailing stop percentage
    tp_pct = 0.03  # Take-profit percentage
    use_sl = True
    use_ts = True
    use_tp = True
    exit_on_reversal = True

    def init(self):
        super().init()
        # Calculate Alligator components
        close = pd.Series(self.data.Close)
        self.jaw = self.I(self.calculate_sma_offset, self.data, self.jaw_period, self.jaw_offset)
        self.teeth = self.I(self.calculate_sma_offset, self.data, self.teeth_period, self.teeth_offset)
        self.lips = self.I(self.calculate_sma_offset, self.data, self.lips_period, self.lips_offset)

        # Calculate ADX
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        self.adx = self.I(ta.trend.adx, high, low, close, window=self.adx_period)

        # Calculate ATR for position sizing
        self.atr = self.I(self.calculate_atr)

    def calculate_sma_offset(self, data, period, offset):
        """Calculate offset SMA"""
        close = pd.Series(data.Close)
        sma = ta.trend.sma_indicator(close, period)
        # Implement offset by shifting the SMA
        return sma.shift(offset)

    def calculate_atr(self):
        """Calculate ATR"""
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        return ta.volatility.AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=self.atr_period
        ).average_true_range()

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.jaw_period + self.jaw_offset,
                                self.teeth_period + self.teeth_offset,
                                self.lips_period + self.lips_offset,
                                self.adx_period):
            return

        # Check exits first
        self.check_exits()

        # Get current values
        price = self.data.Close[-1]
        current_atr = self.atr[-1]
        adx_value = self.adx[-1]

        # Skip if ATR is not valid
        if not current_atr or current_atr == 0:
            return

        # Strong trend condition
        strong_trend = adx_value > self.adx_threshold

        # Check for reversal exits
        if self.exit_on_reversal:
            if self.position.is_long and price < self.lips[-1]:
                self.position.close()
            elif self.position.is_short and price > self.lips[-1]:
                self.position.close()

        # Entry signals
        if not self.position:
            # Long entry
            if price > self.lips[-1] and strong_trend:
                stop_price = price - current_atr * self.atr_multiplier
                size = self.calculate_position_size(current_atr)
                if size > 0:
                    self.buy(size=size)
                    self.set_exit_prices(price, 'long')

            # Short entry
            elif price < self.lips[-1] and strong_trend:
                stop_price = price + current_atr * self.atr_multiplier
                size = self.calculate_position_size(current_atr)
                if size > 0:
                    self.sell(size=size)
                    self.set_exit_prices(price, 'short')

    def calculate_position_size(self, current_atr):
        """Calculate position size based on ATR"""
        if not current_atr or current_atr == 0:
            return 0

        # Use ATR for position sizing
        risk_per_trade = self.equity * 0.02  # Risk 2% per trade
        position_size = risk_per_trade / (current_atr * self.atr_multiplier)

        # Round down to nearest whole number and ensure it's positive
        position_size = max(1, int(position_size))

        return position_size

class TurtleStrategy(BaseStrategy):
    """Turtle Trading Strategy adapted for the backtesting platform"""

    # [Previous parameters remain the same]
    period = 20
    atr_multiplier = 2.0
    profit_target = 0.2
    tenkan_period = 9
    kijun_period = 26
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    stochastic_k_period = 14
    stochastic_d_period = 3
    overbought_threshold = 70
    oversold_threshold = 30
    divergence_lookback = 5
    divergence_threshold = 1.0
    volume_ma_period = 20
    volume_std_period = 20
    min_volume_mult = 1.2
    bb_period = 20
    bb_std = 2.0
    ema_short = 10
    ema_medium = 21
    ema_long = 50
    min_adx_strength = 20
    rsi_overbought = 60
    rsi_oversold = 40
    volume_mult = 1.5

    # Turtle system parameters
    sys1_entry = 20
    sys1_exit = 10
    sys2_entry = 55
    sys2_exit = 20
    atr_period = 20

    # Risk parameters
    risk_level = 2
    r_max = 0.02
    unit_limit = 4
    sl_pct = 0.02
    tp_pct = 0.06
    ts_pct = 0.02
    use_sl = True
    use_tp = True
    use_ts = True
    exit_on_reversal = True

    def init(self):
        super().init()

        # Calculate ATR for position sizing
        self.atr = self.I(self._calculate_atr)

        # System 1 (Fast) Breakout Levels
        self.sys1_high = self.I(self._calculate_donchian_high, self.sys1_entry)
        self.sys1_low = self.I(self._calculate_donchian_low, self.sys1_entry)
        self.sys1_exit_high = self.I(self._calculate_donchian_high, self.sys1_exit)
        self.sys1_exit_low = self.I(self._calculate_donchian_low, self.sys1_exit)

        # System 2 (Slow) Breakout Levels
        self.sys2_high = self.I(self._calculate_donchian_high, self.sys2_entry)
        self.sys2_low = self.I(self._calculate_donchian_low, self.sys2_entry)
        self.sys2_exit_high = self.I(self._calculate_donchian_high, self.sys2_exit)
        self.sys2_exit_low = self.I(self._calculate_donchian_low, self.sys2_exit)

        # State tracking
        self.last_breakout_win = False
        self.current_units = 0
        self.entry_price = None

    def _calculate_atr(self):
        high_series = pd.Series(self.data.High)
        low_series = pd.Series(self.data.Low)
        close_series = pd.Series(self.data.Close)
        return ta.volatility.AverageTrueRange(
            high=high_series,
            low=low_series,
            close=close_series,
            window=self.atr_period
        ).average_true_range()

    def _calculate_donchian_high(self, period):
        return pd.Series(self.data.High).rolling(window=period).max()

    def _calculate_donchian_low(self, period):
        return pd.Series(self.data.Low).rolling(window=period).min()

    def calculate_position_size(self, current_atr):
        """Calculate position size based on ATR"""
        if not current_atr or current_atr == 0:
            return 0

        # Calculate dollar volatility
        dollar_volatility = current_atr * self.risk_level

        # Calculate position size
        position_size = (self.equity * self.r_max) / dollar_volatility

        # Round down to nearest whole number and ensure it's positive
        position_size = max(1, int(position_size))

        return position_size

    def next(self):
        # Check exits first
        self.check_exits()

        # Skip if not enough data
        if len(self.data) < max(self.sys1_entry, self.sys2_entry):
            return

        price = self.data.Close[-1]
        current_atr = self.atr[-1]

        # Skip if ATR is not valid
        if not current_atr or current_atr == 0:
            return

        # System 1 (Fast) Entry Signals
        sys1_long = price > self.sys1_high[-2]
        sys1_short = price < self.sys1_low[-2]

        # System 2 (Slow) Entry Signals
        sys2_long = price > self.sys2_high[-2]
        sys2_short = price < self.sys2_low[-2]

        # Exit Signals
        sys1_exit_long = price < self.sys1_exit_low[-1]
        sys1_exit_short = price > self.sys1_exit_high[-1]
        sys2_exit_long = price < self.sys2_exit_low[-1]
        sys2_exit_short = price > self.sys2_exit_high[-1]

        # Position Management
        if self.position.is_long:
            # Check for profit target
            if self.entry_price and price >= self.entry_price * (1 + self.profit_target):
                self.position.close()
                self.last_breakout_win = True
                self.current_units = 0
                self.entry_price = None
                return

            # Check for exits
            if ((sys1_exit_long and self.current_units <= 2) or
                    (sys2_exit_long and self.current_units > 2)):
                self.position.close()
                self.last_breakout_win = self.entry_price and price > self.entry_price
                self.current_units = 0
                self.entry_price = None
                return

            # Check for pyramiding
            elif (self.current_units < self.unit_limit and self.entry_price and
                  price > self.entry_price + current_atr):
                size = self.calculate_position_size(current_atr)
                if size > 0:
                    self.buy(size=size)
                    self.set_exit_prices(price, 'long')
                    self.current_units += 1

        elif self.position.is_short:
            # Check for profit target
            if self.entry_price and price <= self.entry_price * (1 - self.profit_target):
                self.position.close()
                self.last_breakout_win = True
                self.current_units = 0
                self.entry_price = None
                return

            # Check for exits
            if ((sys1_exit_short and self.current_units <= 2) or
                    (sys2_exit_short and self.current_units > 2)):
                self.position.close()
                self.last_breakout_win = self.entry_price and price < self.entry_price
                self.current_units = 0
                self.entry_price = None
                return

            # Check for pyramiding
            elif (self.current_units < self.unit_limit and self.entry_price and
                  price < self.entry_price - current_atr):
                size = self.calculate_position_size(current_atr)
                if size > 0:
                    self.sell(size=size)
                    self.set_exit_prices(price, 'short')
                    self.current_units += 1

        # New Position Entry
        else:
            # Calculate position size
            size = self.calculate_position_size(current_atr)

            if size > 0:
                # System 1 or 2 Entry
                if (sys1_long or sys2_long) and not self.last_breakout_win:
                    self.buy(size=size)
                    self.set_exit_prices(price, 'long')
                    self.current_units = 1
                    self.entry_price = price

                elif (sys1_short or sys2_short) and not self.last_breakout_win:
                    self.sell(size=size)
                    self.set_exit_prices(price, 'short')
                    self.current_units = 1
                    self.entry_price = price

class MACrossStrategy(BaseStrategy):
    """Enhanced Moving Average Crossover Strategy"""
    n1 = 10
    n2 = 20

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        self.ma1 = self.I(ta.trend.sma_indicator, close, self.n1)
        self.ma2 = self.I(ta.trend.sma_indicator, close, self.n2)

    def next(self):
        # Check exits first
        self.check_exits()
        # Check for reversal signal exit
        if self.exit_on_reversal:
            if self.position.is_long and crossover(self.ma2, self.ma1):
                self.position.close()
            elif self.position.is_short and crossover(self.ma1, self.ma2):
                self.position.close()
        # Entry signals
        if crossover(self.ma1, self.ma2):
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        elif crossover(self.ma2, self.ma1):
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')


class RSIStrategy(BaseStrategy):
    """Enhanced RSI Strategy"""
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        self.rsi = self.I(ta.momentum.rsi, close, self.rsi_period)

    def next(self):
        self.check_exits()

        # Check for reversal signal exit
        if self.exit_on_reversal:
            if self.position.is_long and self.rsi[-1] > self.rsi_overbought:
                self.position.close()
            elif self.position.is_short and self.rsi[-1] < self.rsi_oversold:
                self.position.close()

        # Entry signals
        if self.rsi[-1] < self.rsi_oversold:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        elif self.rsi[-1] > self.rsi_overbought:
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')


class EMACrossWithVolumeStrategy(BaseStrategy):
    """EMA Cross strategy with volume confirmation"""
    n1 = 10  # Fast EMA
    n2 = 20  # Slow EMA
    volume_ma = 20  # Volume MA period
    volume_mult = 1.5  # Volume spike multiplier

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)
        self.ema1 = self.I(ta.trend.ema_indicator, close, self.n1)
        self.ema2 = self.I(ta.trend.ema_indicator, close, self.n2)
        self.volume_sma = self.I(ta.trend.sma_indicator, volume, self.volume_ma)

    def next(self):
        self.check_exits()

        volume_spike = self.data.Volume[-1] > self.volume_sma[-1] * self.volume_mult

        if self.exit_on_reversal:
            if self.position.is_long and crossover(self.ema2, self.ema1):
                self.position.close()
            elif self.position.is_short and crossover(self.ema1, self.ema2):
                self.position.close()

        if crossover(self.ema1, self.ema2) and volume_spike:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        elif crossover(self.ema2, self.ema1) and volume_spike:
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy"""
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_threshold = 50

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        self.bb_indicator = ta.volatility.BollingerBands(
            close, window=self.bb_period, window_dev=self.bb_std)
        self.bb_upper = self.I(lambda: self.bb_indicator.bollinger_hband())
        self.bb_lower = self.I(lambda: self.bb_indicator.bollinger_lband())
        self.rsi = self.I(ta.momentum.rsi, close, self.rsi_period)

    def next(self):
        self.check_exits()

        price = self.data.Close[-1]

        if self.position:
            if self.position.is_long and price >= self.bb_upper[-1]:
                self.position.close()
            elif self.position.is_short and price <= self.bb_lower[-1]:
                self.position.close()

        else:
            if price <= self.bb_lower[-1] and self.rsi[-1] < self.rsi_threshold:
                self.buy()
                self.set_exit_prices(price, 'long')
            elif price >= self.bb_upper[-1] and self.rsi[-1] > self.rsi_threshold:
                self.sell()
                self.set_exit_prices(price, 'short')


class MACDWithTrendStrategy(BaseStrategy):
    """MACD strategy with trend filter"""
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    trend_ma = 200
    min_trend_strength = 0.02  # 2% minimum trend strength

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        macd_indicator = ta.trend.MACD(close,
                                       window_fast=self.macd_fast,
                                       window_slow=self.macd_slow,
                                       window_sign=self.macd_signal)
        self.macd_line = self.I(lambda: macd_indicator.macd())
        self.macd_signal_line = self.I(lambda: macd_indicator.macd_signal())
        self.trend_ema = self.I(ta.trend.ema_indicator, close, self.trend_ma)

    def next(self):
        self.check_exits()

        trend_strength = (self.data.Close[-1] - self.trend_ema[-1]) / self.trend_ema[-1]
        strong_uptrend = trend_strength > self.min_trend_strength
        strong_downtrend = trend_strength < -self.min_trend_strength

        if self.exit_on_reversal:
            if self.position.is_long and crossover(self.macd_signal_line, self.macd_line):
                self.position.close()
            elif self.position.is_short and crossover(self.macd_line, self.macd_signal_line):
                self.position.close()

        if crossover(self.macd_line, self.macd_signal_line) and strong_uptrend:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        elif crossover(self.macd_signal_line, self.macd_line) and strong_downtrend:
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')


class RSIDivergenceStrategy(BaseStrategy):
    """RSI Divergence Strategy with dynamic lookback range."""
    rsi_period = 14
    lookback_range = 5  # Default lookback range
    min_lookback_range = 3
    max_lookback_range = 10

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        self.rsi = self.I(ta.momentum.rsi, close, self.rsi_period)

    def detect_divergence(self, lookback_range):
        """Detect bullish or bearish RSI divergence."""
        # Recent data for price and RSI
        recent_prices = pd.Series(self.data.Close[-lookback_range:])
        recent_rsi = pd.Series(self.rsi[-lookback_range:])

        # Find local min/max in price and RSI
        price_low = recent_prices.idxmin()
        price_high = recent_prices.idxmax()
        rsi_low = recent_rsi.idxmin()
        rsi_high = recent_rsi.idxmax()

        # Bullish divergence: Price lower low, RSI higher low
        bullish_divergence = (
                price_low < price_high and rsi_low > rsi_high
        )

        # Bearish divergence: Price higher high, RSI lower high
        bearish_divergence = (
                price_high > price_low and rsi_high < rsi_low
        )

        return bullish_divergence, bearish_divergence

    def next(self):
        self.check_exits()

        # Test for divergence across the range of lookback values
        for lookback_range in range(self.min_lookback_range, self.max_lookback_range + 1):
            bullish_divergence, bearish_divergence = self.detect_divergence(lookback_range)

            # Trading logic
            if bullish_divergence:
                self.buy()
                self.set_exit_prices(self.data.Close[-1], "long")
                return  # Exit after first valid signal
            elif bearish_divergence:
                self.sell()
                self.set_exit_prices(self.data.Close[-1], "short")
                return  # Exit after first valid signal


class EnhancedRSIBollingerStrategy(BaseStrategy):
    """
    Enhanced RSI and Bollinger Bands Strategy with MA trend and RVOL filters
    - Uses MA for trend direction bias
    - RVOL for volume confirmation
    - RSI and Bollinger Bands for entry signals
    """
    # Default parameters
    rsi_period = 14
    bb_period = 20
    bb_std_dev = 2.0
    recent_candles = 5
    lookback_range = 3

    # New Parameters
    ma_period = 200  # Moving average period for trend filter
    rvol_period = 20  # Period for relative volume calculation
    rvol_threshold = 1.5  # Minimum RVOL for trade confirmation
    rsi_oversold = 30  # RSI oversold threshold
    rsi_overbought = 70  # RSI overbought threshold
    counter_trend_rsi = 80  # RSI threshold for counter-trend shorts

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)

        # Initialize RSI
        self.rsi = self.I(ta.momentum.rsi, close, self.rsi_period)

        # Initialize Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            close=close,
            window=self.bb_period,
            window_dev=self.bb_std_dev
        )
        self.bb_upper = self.I(lambda: bb_indicator.bollinger_hband())
        self.bb_lower = self.I(lambda: bb_indicator.bollinger_lband())
        self.bb_middle = self.I(lambda: bb_indicator.bollinger_mavg())

        # Initialize Moving Average
        self.ma = self.I(ta.trend.sma_indicator, close, self.ma_period)

        # Initialize RVOL (Relative Volume)
        self.avg_volume = self.I(ta.trend.sma_indicator, volume, self.rvol_period)
        self.rvol = self.I(lambda: volume / self.avg_volume)

    def check_rvol(self):
        """Check if relative volume is above threshold"""
        return self.rvol[-1] > self.rvol_threshold

    def next(self):
        self.check_exits()

        price = self.data.Close[-1]
        above_ma = price > self.ma[-1]

        # Check if we're near BB bands
        near_upper_band = abs(price - self.bb_upper[-1]) / self.bb_upper[-1] < 0.002  # Within 0.2%
        near_lower_band = abs(price - self.bb_lower[-1]) / self.bb_lower[-1] < 0.002  # Within 0.2%

        # Volume confirmation
        volume_confirmed = self.check_rvol()

        if not self.position:
            # Long Setup (Price near lower band, oversold RSI, above MA)
            if (price <= self.bb_lower[-1] and  # Price at or below lower band
                    self.rsi[-1] < self.rsi_oversold and  # RSI oversold
                    above_ma and  # Above MA trend
                    volume_confirmed):  # Volume confirmation

                self.buy()
                self.set_exit_prices(price, "long")

            # Short Setup (Price near upper band, overbought RSI, either below MA or extreme RSI)
            elif (price >= self.bb_upper[-1] and  # Price at or above upper band
                  self.rsi[-1] > self.rsi_overbought and  # RSI overbought
                  (not above_ma or self.rsi[-1] > self.counter_trend_rsi) and  # Below MA or extremely overbought
                  volume_confirmed):  # Volume confirmation

                self.sell()
                self.set_exit_prices(price, "short")

        # Exit conditions
        elif self.position.is_long:
            # Exit long if:
            # 1. Price crosses above upper band
            # 2. RSI becomes overbought
            # 3. Price crosses below MA with high volume
            if (price >= self.bb_upper[-1] or
                    self.rsi[-1] > self.rsi_overbought or
                    (not above_ma and volume_confirmed)):
                self.position.close()

        elif self.position.is_short:
            # Exit short if:
            # 1. Price crosses below lower band
            # 2. RSI becomes oversold
            # 3. Price crosses above MA with high volume
            if (price <= self.bb_lower[-1] or
                    self.rsi[-1] < self.rsi_oversold or
                    (above_ma and volume_confirmed)):
                self.position.close()




class RSIBollingerStrategy(BaseStrategy):
    """Stochastic RSI and Bollinger Bands Strategy with SMA Filter and Risk Management"""

    # Default parameters
    stoch_rsi_period = 14
    stoch_k_period = 3
    stoch_d_period = 3
    bb_period = 20
    bb_std_dev = 2.0
    recent_candles = 5
    lookback_range = 3
    ma_period = 200  # Moving average for trend filtering
    sma_filter_period = 50  # NEW: SMA filter for trend validation
    atr_period = 14  # ATR filter to avoid high volatility trades
    atr_multiplier = 1.5  # ATR threshold for entry filtering

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)

        # Compute Stochastic RSI
        self.stochrsi_k = self.I(lambda: ta.momentum.stochrsi_k(
            close, self.stoch_rsi_period, self.stoch_k_period))
        self.stochrsi_d = self.I(lambda: ta.momentum.stochrsi_d(
            close, self.stoch_rsi_period, self.stoch_k_period, self.stoch_d_period))

        # Initialize Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            close=close,
            window=self.bb_period,
            window_dev=self.bb_std_dev
        )
        self.bb_upper = self.I(lambda: bb_indicator.bollinger_hband())
        self.bb_lower = self.I(lambda: bb_indicator.bollinger_lband())

        # Initialize Moving Averages
        self.ma = self.I(ta.trend.sma_indicator, close, self.ma_period)
        self.sma_filter = self.I(ta.trend.sma_indicator, close, self.sma_filter_period)  # NEW SMA FILTER

        # Compute ATR (Average True Range) for volatility filter
        self.atr = self.I(ta.volatility.average_true_range, pd.Series(self.data.High),
                          pd.Series(self.data.Low), close, self.atr_period)

    def next(self):
        self.check_exits()

        # Use lookback_range for recent price checks
        recent_highs = self.data.High[-self.lookback_range:]
        recent_lows = self.data.Low[-self.lookback_range:]
        price_touched_upper = all(recent_highs >= self.bb_upper[-1])  # Stronger condition (multiple touches)
        price_touched_lower = all(recent_lows <= self.bb_lower[-1])  # Stronger condition (multiple touches)

        # Check trend position using SMA filter
        price_above_sma = self.data.Close[-1] > self.sma_filter[-1]
        price_below_sma = self.data.Close[-1] < self.sma_filter[-1]

        # Check MA slope to confirm trend
        ma_slope_up = self.ma[-1] > self.ma[-2]  # Ensures trend direction
        ma_slope_down = self.ma[-1] < self.ma[-2]

        # ATR-based filter to avoid high volatility trades
        recent_range = abs(self.data.Close[-1] - self.data.Open[-1])
        low_volatility = recent_range < (self.atr_multiplier * self.atr[-1])

        # Stochastic RSI confirmation (K crossing D)
        stoch_confirm_long = self.stochrsi_k[-2] < self.stochrsi_d[-2] and self.stochrsi_k[-1] > self.stochrsi_d[-1]
        stoch_confirm_short = self.stochrsi_k[-2] > self.stochrsi_d[-2] and self.stochrsi_k[-1] < self.stochrsi_d[-1]

        # **Updated Entry Signals with SMA Filter**
        if (price_touched_lower and self.stochrsi_k[-1] < 20 and price_above_sma and
                ma_slope_up and stoch_confirm_long and low_volatility):  # Long condition
            self.buy()
            self.set_exit_prices(self.data.Close[-1], "long")
        elif (price_touched_upper and self.stochrsi_k[-1] > 80 and price_below_sma and
              ma_slope_down and stoch_confirm_short and low_volatility):  # Short condition
            self.sell()
            self.set_exit_prices(self.data.Close[-1], "short")

class DonchianChannelStrategy(BaseStrategy):
    """Donchian Channel Breakout Strategy"""
    dc_period = 20

    def init(self):
        super().init()
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        self.dc_indicator = ta.volatility.DonchianChannel(
            high=high, low=low, close=close, window=self.dc_period)
        self.dc_high = self.I(lambda: self.dc_indicator.donchian_channel_hband())
        self.dc_low = self.I(lambda: self.dc_indicator.donchian_channel_lband())

    def next(self):
        self.check_exits()

        if self.data.Close[-1] > self.dc_high[-1]:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        elif self.data.Close[-1] < self.dc_low[-1]:
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')


class ATRChandelierExitStrategy(BaseStrategy):
    """ATR Strategy with Chandelier Exit"""
    atr_period = 14
    atr_multiplier = 3.0

    def init(self):
        super().init()
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        self.atr = self.I(ta.volatility.average_true_range, high, low, close, self.atr_period)
        self.chandelier_exit_long = self.I(lambda: high.rolling(self.atr_period).max() - self.atr_multiplier * self.atr)
        self.chandelier_exit_short = self.I(lambda: low.rolling(self.atr_period).min() + self.atr_multiplier * self.atr)

    def next(self):
        self.check_exits()

        if not self.position:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        else:
            if self.position.is_long and self.data.Close[-1] < self.chandelier_exit_long[-1]:
                self.position.close()
            elif self.position.is_short and self.data.Close[-1] > self.chandelier_exit_short[-1]:
                self.position.close()


class KeltnerChannelStrategy(BaseStrategy):
    """Keltner Channel Strategy"""
    kc_period = 20
    kc_multiplier = 2.0

    def init(self):
        super().init()
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        self.kc_indicator = ta.volatility.KeltnerChannel(high, low, close,
                                                         window=self.kc_period,
                                                         window_atr=self.kc_period,
                                                         original_version=False)
        self.kc_middle = self.I(lambda: self.kc_indicator.keltner_channel_mband())
        self.kc_upper = self.I(lambda: self.kc_indicator.keltner_channel_hband())
        self.kc_lower = self.I(lambda: self.kc_indicator.keltner_channel_lband())

    def next(self):
        self.check_exits()

        if self.data.Close[-1] > self.kc_upper[-1]:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        elif self.data.Close[-1] < self.kc_lower[-1]:
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')


class ChannelBreakoutStrategy(BaseStrategy):
    """Channel Breakout Strategy"""
    channel_period = 20

    def init(self):
        super().init()
        self.high_channel = self.I(lambda: pd.Series(self.data.High).rolling(self.channel_period).max())
        self.low_channel = self.I(lambda: pd.Series(self.data.Low).rolling(self.channel_period).min())

    def next(self):
        self.check_exits()

        if self.data.Close[-1] > self.high_channel[-1]:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        elif self.data.Close[-1] < self.low_channel[-1]:
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')


class MACDDivergenceStrategy(BaseStrategy):
    """MACD Divergence Strategy"""
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        macd_indicator = ta.trend.MACD(close,
                                       window_fast=self.macd_fast,
                                       window_slow=self.macd_slow,
                                       window_sign=self.macd_signal)
        self.macd_line = self.I(lambda: macd_indicator.macd())
        self.macd_signal_line = self.I(lambda: macd_indicator.macd_signal())

    def next(self):
        self.check_exits()

        # Implement divergence detection logic (simplified for example)
        if self.data.Close[-1] > self.data.Close[-2] and self.macd_line[-1] < self.macd_line[-2]:
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')
        elif self.data.Close[-1] < self.data.Close[-2] and self.macd_line[-1] > self.macd_line[-2]:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')


class LinearRegressionStrategy(BaseStrategy):
    """Linear Regression Strategy"""
    lr_window = 14  # Window size for linear regression

    def init(self):
        super().init()
        self.close = pd.Series(self.data.Close)
        self.lr_slope = self.I(self.compute_lr_slope)

    def compute_lr_slope(self):
        slopes = [0] * (self.lr_window - 1)  # Initial zeros
        for i in range(self.lr_window - 1, len(self.close)):
            y = self.close[i - self.lr_window + 1:i + 1].values
            x = np.arange(self.lr_window)
            slope, intercept = np.polyfit(x, y, 1)
            slopes.append(slope)
        return np.array(slopes)

    def next(self):
        self.check_exits()

        current_slope = self.lr_slope[-1]
        previous_slope = self.lr_slope[-2] if len(self.lr_slope) > 1 else 0

        # Entry signals
        if current_slope > 0 and previous_slope <= 0:
            self.buy()
            self.set_exit_prices(self.data.Close[-1], 'long')
        elif current_slope < 0 and previous_slope >= 0:
            self.sell()
            self.set_exit_prices(self.data.Close[-1], 'short')


class ATRVortexStrategy(BaseStrategy):
    """ATR Bands with Vortex Indicator Strategy"""

    # ATR Band Parameters
    first_atr_length = 14
    first_atr_mult = 1.5
    second_atr_length = 14
    second_atr_mult = 3.0

    # Vortex Indicator Parameters
    vi_period = 14
    apply_tema = False

    # Additional Parameters
    fixed_targets = True
    tracking_sl_length = 2
    use_strategy_specific_exits = True
    repaints = True  # If True, uses current bar, if False uses previous bar signals

    def init(self):
        super().init()

        # Price series
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        # Calculate ATR Bands
        self.first_atr = self.I(ta.volatility.average_true_range, high, low, close, self.first_atr_length)
        self.second_atr = self.I(ta.volatility.average_true_range, high, low, close, self.second_atr_length)

        # Calculate ATR Bands
        self.first_upper = self.I(lambda: close + self.first_atr * self.first_atr_mult)
        self.first_lower = self.I(lambda: close - self.first_atr * self.first_atr_mult)
        self.second_upper = self.I(lambda: close + self.second_atr * self.second_atr_mult)
        self.second_lower = self.I(lambda: close - self.second_atr * self.second_atr_mult)

        # Calculate Vortex Indicator
        self.vip = self.I(self._calculate_vortex_plus, high, low)
        self.vim = self.I(self._calculate_vortex_minus, high, low)

        # Track last signal for stop loss
        self.last_signal = None
        self.signal_price = None

    def _calculate_tema(self, source, length):
        """Calculate Triple EMA"""
        ema1 = pd.Series(source).ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        tema = 3 * (ema1 - ema2) + ema3
        return tema

    def _calculate_vortex_plus(self, high, low):
        """Calculate Vortex Indicator Plus"""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        vm_plus = np.abs(high_series - low_series.shift(1)).rolling(window=self.vi_period).sum()
        tr = ta.volatility.average_true_range(high_series, low_series, pd.Series(self.data.Close), self.vi_period)
        vi_plus = vm_plus / tr.rolling(window=self.vi_period).sum()

        if self.apply_tema:
            return self._calculate_tema(vi_plus, self.vi_period)
        return vi_plus

    def _calculate_vortex_minus(self, high, low):
        """Calculate Vortex Indicator Minus"""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        vm_minus = np.abs(low_series - high_series.shift(1)).rolling(window=self.vi_period).sum()
        tr = ta.volatility.average_true_range(high_series, low_series, pd.Series(self.data.Close), self.vi_period)
        vi_minus = vm_minus / tr.rolling(window=self.vi_period).sum()

        if self.apply_tema:
            return self._calculate_tema(vi_minus, self.vi_period)
        return vi_minus

    def _check_vortex_signals(self):
        """Check for Vortex crossover signals"""
        index = 0 if self.repaints else 1

        # Crossover signals
        if self.vip[-1 - index] <= self.vim[-1 - index] and self.vip[-2 - index] > self.vim[-2 - index]:
            return 'short'
        elif self.vip[-1 - index] >= self.vim[-1 - index] and self.vip[-2 - index] < self.vim[-2 - index]:
            return 'long'
        return None

    def next(self):
        # Check base strategy exits first
        self.check_exits()

        # Skip if not enough data
        if len(self.data) < max(self.first_atr_length, self.second_atr_length, self.vi_period):
            return

        # Get current values
        close = self.data.Close[-1]
        first_upper = self.first_upper[-1]
        first_lower = self.first_lower[-1]
        second_upper = self.second_upper[-1]
        second_lower = self.second_lower[-1]

        # Check for Vortex signals
        signal = self._check_vortex_signals()

        # Entry logic
        if not self.position:
            if signal == 'long':
                self.buy()
                self.last_signal = 'long'
                self.signal_price = close
                self.set_exit_prices(close, 'long')

            elif signal == 'short':
                self.sell()
                self.last_signal = 'short'
                self.signal_price = close
                self.set_exit_prices(close, 'short')

        # Exit logic
        if self.use_strategy_specific_exits and self.position:
            if self.position.is_long:
                # Fixed target exits
                if self.fixed_targets:
                    if close >= second_upper:
                        self.position.close()
                    elif close <= first_lower:  # Stop loss
                        self.position.close()
                # Trailing stop logic
                else:
                    trailing_stop = first_upper
                    if len(self.data) >= self.tracking_sl_length:
                        for i in range(self.tracking_sl_length):
                            trailing_stop = min(trailing_stop, self.first_upper[-i - 1])
                    if close < trailing_stop:
                        self.position.close()

            elif self.position.is_short:
                # Fixed target exits
                if self.fixed_targets:
                    if close <= second_lower:
                        self.position.close()
                    elif close >= first_upper:  # Stop loss
                        self.position.close()
                # Trailing stop logic
                else:
                    trailing_stop = first_lower
                    if len(self.data) >= self.tracking_sl_length:
                        for i in range(self.tracking_sl_length):
                            trailing_stop = max(trailing_stop, self.first_lower[-i - 1])
                    if close > trailing_stop:
                        self.position.close()
class ConorSwitchStrategy(BaseStrategy):
    """Enhanced Conor Switch Strategy using Tenkan-sen for longs and MACD divergence for shorts"""

    # Define ALL parameters as class variables
    tenkan_period = 9  # Default period for Tenkan-sen
    stochastic_k_period = 14  # Required for compatibility
    stochastic_d_period = 3  # Required for compatibility

    # MACD parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # RSI parameters (kept for entry/exit conditions)
    rsi_period = 14
    overbought_threshold = 70
    oversold_threshold = 30

    # Divergence parameters
    divergence_lookback = 5
    divergence_threshold = 1.0

    def init(self):
        super().init()
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        # Calculate Tenkan-sen
        self.tenkan_sen = self.I(lambda: (high.rolling(window=self.tenkan_period).max() +
                                          low.rolling(window=self.tenkan_period).min()) / 2)

        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            close=close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        self.macd_line = self.I(lambda: macd_indicator.macd())
        self.macd_signal_line = self.I(lambda: macd_indicator.macd_signal())
        self.macd_histogram = self.I(lambda: macd_indicator.macd_diff())

        # Calculate RSI (kept for entry/exit conditions)
        self.rsi = self.I(lambda: ta.momentum.RSIIndicator(close=close,
                                                           window=self.rsi_period).rsi())

        # Calculate Stochastic (for plotting compatibility)
        stoch = ta.momentum.StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=self.stochastic_k_period,
            smooth_window=self.stochastic_d_period
        )
        self.stoch_k = self.I(lambda: stoch.stoch())
        self.stoch_d = self.I(lambda: stoch.stoch_signal())

        # Memory arrays for divergence
        self.price_memory = []  # Store recent highs
        self.macd_memory = []  # Store recent MACD values

    def check_bearish_macd_divergence(self):
        """Check for bearish MACD divergence"""
        # Get current values
        current_high = self.data.High[-1]
        current_macd = self.macd_histogram[-1]

        # Update memories
        self.price_memory.append(current_high)
        self.macd_memory.append(current_macd)

        # Keep only lookback period
        if len(self.price_memory) > self.divergence_lookback:
            self.price_memory.pop(0)
            self.macd_memory.pop(0)

        # Need enough data points
        if len(self.price_memory) < self.divergence_lookback:
            return False

        # Check for higher highs in price
        price_change = ((current_high - min(self.price_memory)) / min(self.price_memory)) * 100
        if price_change < self.divergence_threshold:
            return False

        # Check for MACD divergence
        price_making_higher_high = current_high > max(self.price_memory[:-1])
        macd_making_lower_high = current_macd < max(self.macd_memory[:-1])

        # Additional MACD conditions
        macd_below_signal = self.macd_line[-1] < self.macd_signal_line[-1]
        histogram_decreasing = (self.macd_histogram[-1] < self.macd_histogram[-2] < self.macd_histogram[-3])

        return (price_making_higher_high and
                macd_making_lower_high and
                macd_below_signal and
                histogram_decreasing)

    def check_trend(self):
        """Check the current trend"""
        # MACD trend
        macd_trend = self.macd_line[-1] > self.macd_signal_line[-1]
        # Price trend
        price_trend = self.data.Close[-1] > self.tenkan_sen[-1]
        return macd_trend and price_trend
    def next(self):
        self.check_exits()

        high = self.data.High[-1]
        close = self.data.Close[-1]
        tenkan_sen = self.tenkan_sen[-1]
        rsi = self.rsi[-1]

        # Long entry conditions
        if not self.position.is_long:
            if (high > tenkan_sen and
                    rsi < self.overbought_threshold and
                    self.check_trend()):
                self.buy()
                self.set_exit_prices(close, 'long')

        # Short entry based on MACD divergence
        if self.check_bearish_macd_divergence():
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell()
                self.set_exit_prices(close, 'short')

        # Exit conditions
        # Exit long if MACD crosses below signal
        if self.position.is_long and self.macd_line[-1] < self.macd_signal_line[-1]:
            self.position.close()

        # Exit short if MACD crosses above signal or RSI oversold
        if self.position.is_short:
            if (self.macd_line[-1] > self.macd_signal_line[-1] or
                    rsi < self.oversold_threshold):
                self.position.close()


class TripleEMAStrategy(Strategy):
    """
    Triple EMA Strategy with ADX Filter, ATR-based stops, and Volume Filter
    """

    # EMA Parameters
    ema_fast = 8
    ema_medium = 21
    ema_slow = 55

    # ADX Parameters
    adx_period = 14
    adx_threshold = 25  # Minimum ADX value for trend confirmation

    # ATR Parameters for Dynamic Stops
    atr_period = 14
    atr_multiplier = 2.0

    # Trend Confirmation Lookback
    trend_lookback = 3

    # Volume Filter
    min_volume_threshold = 1000

    # Position Management
    use_strategy_specific_exits = True
    split_position = True

    def init(self):
        super().init()

        # Price series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Calculate EMAs
        self.ema_f = self.I(ta.trend.ema_indicator, close, self.ema_fast)
        self.ema_m = self.I(ta.trend.ema_indicator, close, self.ema_medium)
        self.ema_s = self.I(ta.trend.ema_indicator, close, self.ema_slow)

        # Calculate ADX
        self.adx = self.I(ta.trend.adx, high, low, close, self.adx_period)

        # ATR for dynamic stops
        self.atr = self.I(ta.volatility.average_true_range, high, low, close, self.atr_period)

        # Store volume for filtering
        self.volume = volume

        # Variables to store stop loss and take profit
        self.sl = None
        self.tp = None

    def check_trend(self):
        """Determine trend direction based on multiple-bar EMA alignment."""
        ema_f_list = self.ema_f[-self.trend_lookback:]
        ema_m_list = self.ema_m[-self.trend_lookback:]
        ema_s_list = self.ema_s[-self.trend_lookback:]

        uptrend = all(ema_f_list[i] > ema_m_list[i] > ema_s_list[i] for i in range(self.trend_lookback))
        downtrend = all(ema_f_list[i] < ema_m_list[i] < ema_s_list[i] for i in range(self.trend_lookback))

        if uptrend:
            return "uptrend"
        elif downtrend:
            return "downtrend"
        return "ranging"

    def set_exit_prices(self, current_close, position_type):
        """Set ATR-based stop loss and take profit levels."""
        current_atr = self.atr[-1]
        if position_type == 'long':
            self.sl = current_close - self.atr_multiplier * current_atr
            self.tp = current_close + self.atr_multiplier * current_atr
        else:  # short position
            self.sl = current_close + self.atr_multiplier * current_atr
            self.tp = current_close - self.atr_multiplier * current_atr

    def check_exits(self):
        """Check if current price hits stop loss or take profit."""
        if self.position:
            current_close = self.data.Close[-1]

            if self.position.is_long:
                # Move stop to break-even if desired, etc. For now, just basic SL/TP.
                if current_close <= self.sl or current_close >= self.tp:
                    self.position.close()
            elif self.position.is_short:
                if current_close >= self.sl or current_close <= self.tp:
                    self.position.close()

    def next(self):
        if self.use_strategy_specific_exits:
            self.check_exits()
        else:
            self.check_exits()  # Use base strategy exits

        # Get current values
        close = self.data.Close[-1]
        adx_value = self.adx[-1]
        trend = self.check_trend()
        curr_volume = self.volume.iloc[-1]

        # Skip if ADX is too low (ranging market)
        if adx_value < self.adx_threshold:
            return

        # Only trade if volume is above threshold
        if curr_volume < self.min_volume_threshold:
            return

        # Entry logic
        if not self.position:
            if trend == "uptrend":
                self.buy()
                self.set_exit_prices(close, 'long')
            elif trend == "downtrend":
                self.sell()
                self.set_exit_prices(close, 'short')

class OptimizationResults:
    def __init__(self, params, metrics, equity_curve, trades):
        self.params = params
        self.metrics = metrics
        self.equity_curve = equity_curve
        self.trades = trades


class StrategyOptimizer:
    def __init__(self, strategy_class, data):
        self.strategy_class = strategy_class
        self.data = data

    def optimize_grid(self, param_ranges, base_params, cash, metric='Net Profit'):
        """
        Perform grid optimization over parameter ranges

        Parameters:
        param_ranges: Dict of parameters to optimize and their ranges
        base_params: Dict of fixed parameters
        cash: Initial cash amount
        metric: Metric to optimize for (default: 'Net Profit')
        """
        print("\nStarting optimization with:")
        print("Parameter ranges:", param_ranges)
        print("Base parameters:", base_params)
        print("Initial cash:", cash)

        all_results = []

        try:
            # Verify data is valid
            if self.data.empty:
                raise ValueError("Empty data provided")

            if not all(col in self.data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                raise ValueError("Data missing required OHLCV columns")

            # Generate all parameter combinations
            combinations = list(self._generate_combinations(param_ranges))
            total_combinations = len(combinations)
            print(f"\nTesting {total_combinations} parameter combinations")

            # Test each parameter combination
            for i, params in enumerate(combinations, 1):
                current_params = base_params.copy()
                current_params.update(params)

                try:
                    # Run backtest with current parameters
                    bt = Backtest(self.data, self.strategy_class, cash=cash, commission=.002)
                    stats = bt.run(**current_params)

                    # Extract metrics
                    net_profit = stats['Equity Final [$]'] - cash
                    return_pct = stats.get('Return [%]', 0)
                    sharpe = stats.get('Sharpe Ratio', 0)
                    max_dd = stats.get('Max. Drawdown [%]', 0)
                    trades = stats.get('# Trades', 0)

                    print(f"\nCombination {i}/{total_combinations}:")
                    print(f"Net Profit: ${net_profit:.2f}")
                    print(f"Return: {return_pct:.2f}%")
                    print(f"Sharpe: {sharpe:.2f}")
                    print(f"Max DD: {max_dd:.2f}%")
                    print(f"Trades: {trades}")

                    # Store all results
                    all_results.append({
                        'Params': current_params,
                        'Net Profit': net_profit,
                        'Return [%]': return_pct,
                        'Sharpe Ratio': sharpe,
                        'Max Drawdown [%]': max_dd,
                        'Trades': trades,
                        'Metrics': stats,
                        'Equity Curve': stats._equity_curve,
                        'Trades Data': stats._trades
                    })

                except Exception as e:
                    print(f"Error testing combination {i}: {str(e)}")
                    continue

            if not all_results:
                print("No valid results found")
                return None, []

            # Sort results based on specified metric
            metric_mapping = {
                'Net Profit': 'Net Profit',
                'Sharpe Ratio': 'Sharpe Ratio',
                'Return [%]': 'Return [%]'
            }

            sort_key = metric_mapping.get(metric, 'Net Profit')
            all_results.sort(key=lambda x: x[sort_key], reverse=True)

            best_result = all_results[0]

            print("\nOptimization completed")
            print("\nBest parameters found:")
            print(f"Parameters: {best_result['Params']}")
            print(f"Net Profit: ${best_result['Net Profit']:.2f}")
            print(f"Return: {best_result['Return [%]']:.2f}%")
            print(f"Sharpe: {best_result['Sharpe Ratio']:.2f}")
            print(f"Max DD: {best_result['Max Drawdown [%]']:.2f}%")
            print(f"Trades: {best_result['Trades']}")

            return best_result, all_results

        except Exception as e:
            print(f"Optimization error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, []

    def _generate_combinations(self, param_ranges):
        """Generate all possible combinations of parameters"""
        if not param_ranges:
            yield {}
            return

        # Get all parameter names and their possible values
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        # Generate all combinations
        for values in itertools.product(*param_values):
            yield dict(zip(param_names, values))


class CryptoBacktester:
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"k "**********"e "**********"y "**********"= "**********"N "**********"o "**********"n "**********"e "**********", "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"= "**********"N "**********"o "**********"n "**********"e "**********") "**********": "**********"
        self.strategies = {
            'Moving Average Crossover': MACrossStrategy,
            'Liquidation Strategy': LiquidationStrategy,
            'ATR Vortex': ATRVortexStrategy,  # Make sure this is added
            'RSI': RSIStrategy,
            'Enhanced RSI Bollinger': EnhancedRSIBollingerStrategy,
            'Adaptive RSI Volume': AdaptiveRSIVolumeStrategy,
            'Adaptive Supertrend': AdaptiveSupertrendStrategy,
            'RSI Trend Reversal': RSITrendReversalStrategy,
            'ATR BB': BBATRIntradayStrategy,  # Changed from 'ATR Bollinger Bands'
            'RSI DIVER': RSIDivergenceStrategy,
            'SR VWAP Strategy': SRVWAPStrategy,
            'EMA Cross with Volume': EMACrossWithVolumeStrategy,
            'Ichimoku TK Cross': IchimokuTKCrossStrategy,
            'Ichimoku EMA Cross': IchimokuEMAStrategy,
            'Bollinger Bands': BollingerBandsStrategy,
            'MACD with Trend': MACDWithTrendStrategy,
            'RSI Divergence': RSIDivergenceStrategy,
            'RSI and Bollinger': RSIBollingerStrategy,
            'Donchian Channel Breakout': DonchianChannelStrategy,
            'ATR Chandelier Exit': ATRChandelierExitStrategy,
            'Keltner Channel': KeltnerChannelStrategy,
            'Channel Breakout': ChannelBreakoutStrategy,
            'MACD Divergence': MACDDivergenceStrategy,
            'Linear Regression': LinearRegressionStrategy,
            'Conor Switch': ConorSwitchStrategy,
            'Alligator': AlligatorStrategy,
            'Turtle Trading': TurtleStrategy,
            'Triple EMA': TripleEMAStrategy
        }
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********"  "**********"o "**********"r "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
            api_key = os.getenv('Change me')
            api_secret = "**********"
        self.client = "**********"

    def optimize_strategy(self, strategy_name, data, param_ranges, base_params, cash):
        strategy_class = self.strategies[strategy_name]
        optimizer = StrategyOptimizer(strategy_class, data)
        return optimizer.optimize_grid(param_ranges, base_params, cash)

    def create_plot(self, data, bt, stats):
        print("Creating plot with data:")
        print("Data shape:", data.shape)
        print("Data head:", data.head())

        strategy = stats._strategy

        # Calculate buy & hold returns
        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        initial_cash = stats['_equity_curve'].iloc[0]['Equity']
        buy_hold_shares = initial_cash / initial_price
        buy_hold_equity = buy_hold_shares * data['Close']

        # Determine the number of rows based on the strategy
        if isinstance(strategy, ConorSwitchStrategy):
            rows = 3
            row_heights = [0.6, 0.2, 0.2]
            subplot_titles = ('Price and Signals', 'Strategy vs Buy & Hold', 'Stochastic Oscillator')
        else:
            rows = 2
            row_heights = [0.7, 0.3]
            subplot_titles = ('Price and Signals', 'Strategy vs Buy & Hold')

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add strategy-specific indicators
        if isinstance(strategy, MACrossStrategy):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=strategy.ma1,
                    name='Fast MA',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=strategy.ma2,
                    name='Slow MA',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        elif isinstance(strategy, LinearRegressionStrategy):
            # Plot Linear Regression Line
            lr_line = strategy.I(lambda: strategy.lr_slope * np.arange(len(data)) + strategy.close[0])
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=lr_line,
                    name='Linear Regression Line',
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
        elif isinstance(strategy, ConorSwitchStrategy):
            # Plot Tenkan-sen line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=strategy.tenkan_sen,
                    name='Tenkan-sen',
                    line=dict(color='orange')
                ),
                row=1, col=1
            )

            # Plot RSI instead of Stochastic
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=strategy.rsi,
                    name='RSI',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )

            # Add RSI levels
            fig.add_hline(y=strategy.overbought_threshold, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=strategy.oversold_threshold, line_dash="dash", line_color="green", row=3, col=1)

        # Add strategy equity curve and buy & hold comparison
        equity_curve = stats['_equity_curve']
        equity_curve.index = pd.to_datetime(equity_curve.index)

        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['Equity'],
                name='Strategy',
                line=dict(color='blue')
            ),
            row=rows - 1, col=1
        )

        # Add buy & hold equity curve
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=buy_hold_equity,
                name='Buy & Hold',
                line=dict(color='gray', dash='dash')
            ),
            row=rows - 1, col=1
        )

        # Calculate and add strategy vs buy & hold metrics to the title
        strategy_return = ((equity_curve['Equity'].iloc[-1] / equity_curve['Equity'].iloc[0]) - 1) * 100
        buy_hold_return = ((buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0]) - 1) * 100

        fig.update_layout(
            height=800,
            title_text=f"Backtest Results (Strategy: {strategy_return:.1f}% vs Buy & Hold: {buy_hold_return:.1f}%)",
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        # Print debug info about the plot
        print("\nPlot information:")
        print("Number of traces:", len(fig.data))
        print("First trace type:", fig.data[0].type if fig.data else "No traces")

        return fig

    def fetch_data(self, symbol, interval, start_date, end_date):
        try:
            print(f"\nFetching data for {symbol} from {start_date} to {end_date}")

            # Ensure the symbol is correctly formatted
            symbol_input = symbol.upper()
            if symbol_input.endswith('USDT'):
                symbol_pair = symbol_input
            else:
                symbol_pair = symbol_input + 'USDT'

            # Convert dates to strings in the required format
            start_str = start_date.strftime('%d %b %Y')
            end_str = end_date.strftime('%d %b %Y')

            print(f"Requesting klines with parameters:")
            print(f"Symbol: {symbol_pair}")
            print(f"Interval: {interval}")
            print(f"Start Time: {start_str}")
            print(f"End Time: {end_str}")

            # Handle custom intervals (8h and 12h)
            if interval in ['8h', '12h']:
                # First fetch 1-hour data
                base_klines = self.client.get_historical_klines(
                    symbol=symbol_pair,
                    interval=KLINE_INTERVAL_1HOUR,
                    start_str=start_str,
                    end_str=end_str
                )

                # Convert to DataFrame
                df = pd.DataFrame(base_klines, columns=[
                    'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Convert string values to float
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Handle missing values
                df = df.dropna()  # Remove rows with NaN values

                # Resample to desired timeframe
                hours = 8 if interval == '8h' else 12
                df_resampled = df.resample(f'{hours}H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })

                # Handle any missing values after resampling
                df_resampled = df_resampled.dropna()

                return df_resampled

            else:
                # Handle regular intervals
                klines = self.client.get_historical_klines(
                    symbol=symbol_pair,
                    interval=interval,
                    start_str=start_str,
                    end_str=end_str
                )

                if not klines:
                    print(f"No data received for {symbol}")
                    return None

                print(f"Received {len(klines)} candlesticks")

                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

                # Convert timestamp to datetime index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Convert string values to float and handle missing values
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Clean data
                df = df.dropna()  # Remove rows with NaN values

                # Additional data validation
                if df.empty:
                    print(f"No valid data after cleaning for {symbol}")
                    return None

                # Verify OHLC relationships
                df = df[
                    (df['High'] >= df['Low']) &
                    (df['High'] >= df['Open']) &
                    (df['High'] >= df['Close']) &
                    (df['Low'] <= df['Open']) &
                    (df['Low'] <= df['Close'])
                    ]

                # Final check for sufficient data
                if len(df) < 50:  # Minimum required for most indicators
                    print(f"Insufficient data points ({len(df)}) after cleaning")
                    return None

                print(f"Final cleaned dataset contains {len(df)} rows")
                return df

        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # Functions to compute monthly and yearly analysis
    def compute_monthly_analysis(self, trades, equity_curve):
        # Convert index to datetime if not already
        equity_curve.index = pd.to_datetime(equity_curve.index)
        equity_curve['Month'] = equity_curve.index.to_period('M')
        monthly_returns = equity_curve.groupby('Month')['Equity'].last().pct_change().fillna(0) * 100
        df = monthly_returns.reset_index().rename(columns={'Equity': 'Return [%]'})
        df['Month'] = df['Month'].dt.strftime('%Y-%m')  # Format Month as 'YYYY-MM'
        return df

    def compute_yearly_analysis(self, trades, equity_curve):
        equity_curve.index = pd.to_datetime(equity_curve.index)
        equity_curve['Year'] = equity_curve.index.year
        yearly_returns = equity_curve.groupby('Year')['Equity'].last().pct_change().fillna(0) * 100
        df = yearly_returns.reset_index().rename(columns={'Equity': 'Return [%]'})
        df['Year'] = df['Year'].astype(str)  # Ensure Year is of string type
        return df

    def create_dashboard(self):
        default_end = date.today()
        default_start = default_end - timedelta(days=365)
        optimization_progress = pn.widgets.Progress(name='Optimization Progress', value=0, width=300, visible=False)
        # Input widgets
        symbol = pn.widgets.TextInput(name='Symbol', value='BTC', width=200)
        interval_select = pn.widgets.Select(name='Interval', options={
            '5m': KLINE_INTERVAL_5MINUTE,
            '15m': KLINE_INTERVAL_15MINUTE,
            '30m': KLINE_INTERVAL_30MINUTE,
            '1h': KLINE_INTERVAL_1HOUR,
            '2h': KLINE_INTERVAL_2HOUR,
            '4h': KLINE_INTERVAL_4HOUR,
            '8h': '8h',  # Custom interval
            '12h': '12h',  # Custom interval
            '1d': KLINE_INTERVAL_1DAY,
        }, value=KLINE_INTERVAL_1DAY)

        start_date = pn.widgets.DatePicker(name='Start Date', value=default_start, start=date(2010, 1, 1),
                                           end=default_end)
        end_date = pn.widgets.DatePicker(name='End Date', value=default_end, start=date(2010, 1, 1),
                                         end=default_end)

        strategy_select = pn.widgets.Select(name='Strategy', options=list(self.strategies.keys()),
                                            value='Moving Average Crossover')

        # Add initial cash input widget
        initial_cash = pn.widgets.IntInput(name='Initial Equity [$]', value=100000, start=1000, step=1000)

        commission_bps = pn.widgets.FloatSlider(
            name='Commission (basis points)',
            value=20.0,
            start=0.0,
            end=100.0,
            step=0.5,
            format='0.0',
        )
        # Strategy parameter widgets
        ma_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize Fast MA', value=False),
            pn.widgets.IntSlider(name='Fast MA Period', value=10, start=5, end=50),
            pn.widgets.Checkbox(name='Optimize Slow MA', value=False),
            pn.widgets.IntSlider(name='Slow MA Period', value=20, start=10, end=100),
            visible=True,
            name='MA Parameters'
        )
        ichimoku_tk_params = pn.Column(
            # Ichimoku Parameters
            pn.widgets.Checkbox(name='Optimize Conversion Period', value=False),
            pn.widgets.IntSlider(name='Conversion Period', value=20, start=10, end=30),
            pn.widgets.Checkbox(name='Optimize Base Period', value=False),
            pn.widgets.IntSlider(name='Base Period', value=60, start=40, end=80),
            pn.widgets.Checkbox(name='Optimize Lagging Span 2 Period', value=False),
            pn.widgets.IntSlider(name='Lagging Span 2 Period', value=120, start=80, end=160),
            pn.widgets.Checkbox(name='Optimize Displacement', value=False),
            pn.widgets.IntSlider(name='Displacement', value=30, start=20, end=40),

            # Filter Parameters
            pn.widgets.Checkbox(name='Optimize ADX Period', value=False),
            pn.widgets.IntSlider(name='ADX Period', value=14, start=10, end=30),
            pn.widgets.Checkbox(name='Optimize ADX Threshold', value=False),
            pn.widgets.IntSlider(name='ADX Threshold', value=25, start=15, end=40),

            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize RSI Bounds', value=False),
            pn.widgets.IntRangeSlider(name='RSI Bounds', value=(30, 70), start=20, end=80),

            pn.widgets.Checkbox(name='Use Cloud Filter', value=True),

            visible=False,
            name='Ichimoku TK Parameters'
        )
        liquidation_params = pn.Column(
            # Liquidation parameters
            pn.widgets.Checkbox(name='Optimize Liquidation Threshold', value=False),
            pn.widgets.IntSlider(name='Liquidation Threshold', value=1000000, start=100000, end=5000000, step=100000),

            pn.widgets.Checkbox(name='Optimize Liquidation Window', value=False),
            pn.widgets.IntSlider(name='Liquidation Window', value=12, start=6, end=24),

            pn.widgets.Checkbox(name='Optimize Liquidation Spike Multiplier', value=False),
            pn.widgets.FloatSlider(name='Liquidation Spike Multiplier', value=2.0, start=1.5, end=4.0, step=0.1),

            # Confirmation parameters
            pn.widgets.Checkbox(name='Optimize Volume Multiplier', value=False),
            pn.widgets.FloatSlider(name='Volume Multiplier', value=1.5, start=1.1, end=3.0, step=0.1),

            pn.widgets.Checkbox(name='Optimize Price Rebound', value=False),
            pn.widgets.FloatSlider(name='Price Rebound %', value=1.0, start=0.5, end=3.0, step=0.1),

            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=7, end=28),

            pn.widgets.Checkbox(name='Optimize RSI Oversold', value=False),
            pn.widgets.IntSlider(name='RSI Oversold', value=30, start=20, end=40),

            visible=False,
            name='Liquidation Strategy Parameters'
        )
        bb_atr_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize BB Length', value=False),
            pn.widgets.IntSlider(name='BB Length', value=55, start=20, end=100),
            pn.widgets.Checkbox(name='Optimize BB StdDev', value=False),
            pn.widgets.FloatSlider(name='BB StdDev', value=0.3, start=0.1, end=1.0, step=0.05),
            pn.widgets.Checkbox(name='Optimize ATR Length', value=False),
            pn.widgets.IntSlider(name='ATR Length', value=20, start=10, end=50),
            pn.widgets.Checkbox(name='Optimize ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='ATR Multiplier', value=0.6, start=0.1, end=2.0, step=0.1),
            pn.widgets.Checkbox(name='Use Filter', value=True),
            pn.widgets.Checkbox(name='Use ATR BB', value=True),
            visible=False,
            name='ATR Bollinger Bands Parameters'
        )
        ichimoku_ema_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize Conversion Period', value=False),
            pn.widgets.IntSlider(name='Conversion Period', value=20, start=10, end=30),
            pn.widgets.Checkbox(name='Optimize Base Period', value=False),
            pn.widgets.IntSlider(name='Base Period', value=60, start=40, end=80),
            pn.widgets.Checkbox(name='Optimize Lagging Span 2 Period', value=False),
            pn.widgets.IntSlider(name='Lagging Span 2 Period', value=120, start=80, end=160),
            pn.widgets.Checkbox(name='Optimize Displacement', value=False),
            pn.widgets.IntSlider(name='Displacement', value=30, start=20, end=40),
            pn.widgets.Checkbox(name='Optimize EMA Length', value=False),
            pn.widgets.IntSlider(name='EMA Length', value=200, start=150, end=250),
            visible=False,
            name='Ichimoku EMA Parameters'
        )

        adaptive_supertrend_params = pn.Column(
            # Choppiness Index Parameters
            pn.widgets.Checkbox(name='Optimize Choppiness Period', value=False),
            pn.widgets.IntSlider(name='Choppiness Period', value=14, start=5, end=50, step=1),
            pn.widgets.Checkbox(name='Optimize Choppiness Threshold', value=False),
            pn.widgets.FloatSlider(name='Choppiness Threshold', value=61.8, start=40.0, end=80.0, step=0.1),
            pn.widgets.Select(
                name='RSI/BB Timeframe',
                options={
                    'Same as Chart': '0',
                    '1 minute': '1m',
                    '3 minutes': '3m',
                    '5 minutes': '5m',
                    '15 minutes': '15m',
                    '30 minutes': '30m',
                    '1 hour': '1H',
                    '2 hours': '2H',
                    '4 hours': '4H',
                    '6 hours': '6H',
                    '8 hours': '8H',
                    '12 hours': '12H',
                    '1 day': '1D'
                },
                value='1H'
            ),
            # Supertrend Parameters
            pn.widgets.Checkbox(name='Optimize ATR Period', value=False),
            pn.widgets.IntSlider(name='ATR Period', value=10, start=5, end=30, step=1),
            pn.widgets.Checkbox(name='Optimize ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='ATR Multiplier', value=3.0, start=1.0, end=5.0, step=0.1),

            # RSI/Bollinger Parameters
            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=5, end=30, step=1),
            pn.widgets.Checkbox(name='Optimize BB Period', value=False),
            pn.widgets.IntSlider(name='BB Period', value=20, start=10, end=50, step=1),
            pn.widgets.Checkbox(name='Optimize BB Std Dev', value=False),
            pn.widgets.FloatSlider(name='BB Std Dev', value=2.0, start=1.0, end=3.0, step=0.1),
            pn.widgets.Checkbox(name='Optimize Lookback Range', value=False),
            pn.widgets.IntSlider(name='Lookback Range', value=3, start=1, end=10, step=1),

            visible=False,
            name='Adaptive Supertrend Parameters'
        )

        enhanced_rsibb_params = pn.Column(
            # RSI Parameters
            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=5, end=28),
            pn.widgets.Checkbox(name='Optimize RSI Thresholds', value=False),
            pn.widgets.IntRangeSlider(name='RSI Thresholds', value=(30, 70), start=20, end=80),
            pn.widgets.Checkbox(name='Optimize Counter-Trend RSI', value=False),
            pn.widgets.IntSlider(name='Counter-Trend RSI', value=80, start=70, end=90),

            # Bollinger Bands Parameters
            pn.widgets.Checkbox(name='Optimize BB Period', value=False),
            pn.widgets.IntSlider(name='BB Period', value=20, start=10, end=50),
            pn.widgets.Checkbox(name='Optimize BB Std Dev', value=False),
            pn.widgets.FloatSlider(name='BB Std Dev', value=2.0, start=1.0, end=4.0, step=0.1),

            # MA Parameters
            pn.widgets.Checkbox(name='Optimize MA Period', value=False),
            pn.widgets.IntSlider(name='MA Period', value=200, start=50, end=500),

            # RVOL Parameters
            pn.widgets.Checkbox(name='Optimize RVOL Period', value=False),
            pn.widgets.IntSlider(name='RVOL Period', value=20, start=10, end=50),
            pn.widgets.Checkbox(name='Optimize RVOL Threshold', value=False),
            pn.widgets.FloatSlider(name='RVOL Threshold', value=1.5, start=1.0, end=3.0, step=0.1),

            # Lookback Parameters
            pn.widgets.Checkbox(name='Optimize Recent Candles', value=False),
            pn.widgets.IntSlider(name='Recent Candles', value=5, start=1, end=10),
            pn.widgets.Checkbox(name='Optimize Lookback Range', value=False),
            pn.widgets.IntSlider(name='Lookback Range', value=3, start=1, end=10),

            visible=False,
            name='Enhanced RSI Bollinger Parameters'
        )
        # Create parameter widgets
        sr_vwap_params = pn.Column(
            # SR Parameters
            pn.widgets.Checkbox(name='Optimize Pivot Length', value=False),
            pn.widgets.IntSlider(name='Pivot Length', value=15, start=10, end=30),

            pn.widgets.Checkbox(name='Optimize SR Strength', value=False),
            pn.widgets.IntSlider(name='SR Strength', value=2, start=1, end=5),

            pn.widgets.Checkbox(name='Optimize ATR Period', value=False),
            pn.widgets.IntSlider(name='ATR Period', value=20, start=10, end=40),

            pn.widgets.Checkbox(name='Optimize Zone Size', value=False),
            pn.widgets.FloatSlider(name='Zone Size ATR', value=0.075, start=0.025, end=0.2, step=0.025),

            pn.widgets.Checkbox(name='Optimize Min Bars Between Signals', value=False),
            pn.widgets.IntSlider(name='Min Bars Between Signals', value=15, start=5, end=30),

            # VWAP Parameters
            pn.widgets.Checkbox(name='Optimize VWAP MA Length', value=False),
            pn.widgets.IntSlider(name='VWAP MA Length', value=20, start=10, end=40),

            pn.widgets.Checkbox(name='Optimize VWAP Trend Strength', value=False),
            pn.widgets.FloatSlider(name='Min VWAP Trend Strength', value=0.001, start=0.0005, end=0.005, step=0.0005),

            visible=False,
            name='SR VWAP Parameters'
        )
        # Multi-Indicator Momentum parameter widgets
        multi_momentum_params = pn.Column(
            # SMA Parameters
            pn.widgets.Checkbox(name='Optimize SMA Period', value=False),
            pn.widgets.IntSlider(name='SMA Period', value=20, start=10, end=50),

            # MACD Parameters
            pn.widgets.Checkbox(name='Optimize MACD Fast', value=False),
            pn.widgets.IntSlider(name='MACD Fast Period', value=12, start=8, end=20),
            pn.widgets.Checkbox(name='Optimize MACD Slow', value=False),
            pn.widgets.IntSlider(name='MACD Slow Period', value=26, start=20, end=40),
            pn.widgets.Checkbox(name='Optimize MACD Signal', value=False),
            pn.widgets.IntSlider(name='MACD Signal Period', value=9, start=5, end=15),

            # RSI Parameters
            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize RSI Bounds', value=False),
            pn.widgets.IntRangeSlider(name='RSI Long Entry Bounds', value=(40, 70), start=30, end=80),
            pn.widgets.IntRangeSlider(name='RSI Short Entry Bounds', value=(30, 60), start=20, end=70),
            pn.widgets.IntRangeSlider(name='RSI Exit Bounds', value=(30, 70), start=20, end=80),

            # Bollinger Bands Parameters
            pn.widgets.Checkbox(name='Optimize BB Period', value=False),
            pn.widgets.IntSlider(name='BB Period', value=20, start=10, end=50),
            pn.widgets.Checkbox(name='Optimize BB STD', value=False),
            pn.widgets.FloatSlider(name='BB STD', value=2.0, start=1.0, end=3.0, step=0.1),

            # ATR Parameters
            pn.widgets.Checkbox(name='Optimize ATR Period', value=False),
            pn.widgets.IntSlider(name='ATR Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='ATR Stop Multiplier', value=2.0, start=1.0, end=4.0, step=0.1),
            pn.widgets.FloatSlider(name='Max Drawdown %', value=15.0, start=5.0, end=25.0, step=0.5),

            visible=False,
            name='Multi-Indicator Momentum Parameters'
        )
        rsi_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize Overbought Level', value=False),
            pn.widgets.IntSlider(name='Overbought Level', value=70, start=60, end=90),
            pn.widgets.Checkbox(name='Optimize Oversold Level', value=False),
            pn.widgets.IntSlider(name='Oversold Level', value=30, start=10, end=40),
            visible=False,
            name='RSI Parameters'
        )
        adaptive_rsi_params = pn.Column(
            # Core RSI Parameters
            pn.widgets.Checkbox(name='Optimize RSI Length', value=False),
            pn.widgets.IntSlider(name='RSI Length', value=8, start=5, end=20),
            pn.widgets.Checkbox(name='Optimize RSI Multiplier', value=False),
            pn.widgets.FloatSlider(name='RSI Multiplier', value=1.5, start=1.0, end=3.0, step=0.1),

            # Volatility Parameters
            pn.widgets.Checkbox(name='Optimize ATR Period', value=False),
            pn.widgets.IntSlider(name='ATR Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize Volatility Multiplier', value=False),
            pn.widgets.FloatSlider(name='Volatility Multiplier', value=1.5, start=1.0, end=3.0, step=0.1),

            # Trend Parameters
            pn.widgets.Checkbox(name='Optimize Trend EMA', value=False),
            pn.widgets.IntSlider(name='Trend EMA Period', value=200, start=100, end=400),

            # Volume Parameters
            pn.widgets.Checkbox(name='Optimize Volume MA Period', value=False),
            pn.widgets.IntSlider(name='Volume MA Period', value=20, start=10, end=50),
            pn.widgets.Checkbox(name='Optimize Min Volume Multiplier', value=False),
            pn.widgets.FloatSlider(name='Min Volume Multiplier', value=1.2, start=1.0, end=2.0, step=0.1),

            # RSI Bounds
            pn.widgets.Checkbox(name='Optimize RSI Bounds', value=False),
            pn.widgets.IntRangeSlider(name='RSI Bounds', value=(30, 70), start=20, end=80),

            # Risk Parameters
            pn.widgets.Checkbox(name='Optimize Min Swing Percentage', value=False),
            pn.widgets.FloatSlider(name='Min Swing Percentage', value=0.5, start=0.2, end=2.0, step=0.1),
            pn.widgets.Checkbox(name='Optimize Max Risk Per Trade', value=False),
            pn.widgets.FloatSlider(name='Max Risk Per Trade %', value=2.0, start=0.5, end=5.0, step=0.1),

            visible=False,
            name='Adaptive RSI Parameters'
        )
        # ATR Vortex parameter widgets

        atr_vortex_params = pn.Column(
            # First ATR Band Parameters
            pn.widgets.Checkbox(name='Optimize First ATR Length', value=False),
            pn.widgets.IntSlider(name='First ATR Length', value=14, start=5, end=30),
            pn.widgets.Checkbox(name='Optimize First ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='First ATR Multiplier', value=1.5, start=0.5, end=3.0, step=0.1),

            # Second ATR Band Parameters
            pn.widgets.Checkbox(name='Optimize Second ATR Length', value=False),
            pn.widgets.IntSlider(name='Second ATR Length', value=14, start=5, end=30),
            pn.widgets.Checkbox(name='Optimize Second ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='Second ATR Multiplier', value=3.0, start=1.0, end=5.0, step=0.1),

            # Vortex Parameters
            pn.widgets.Checkbox(name='Optimize VI Period', value=False),
            pn.widgets.IntSlider(name='VI Period', value=14, start=5, end=30),
            pn.widgets.Checkbox(name='Apply TEMA', value=False),

            # Additional Parameters
            pn.widgets.Checkbox(name='Fixed Targets', value=True),
            pn.widgets.Checkbox(name='Repaints', value=True),
            pn.widgets.Checkbox(name='Use Strategy Specific Exits', value=True),
            pn.widgets.Checkbox(name='Optimize Tracking SL Length', value=False),
            pn.widgets.IntSlider(name='Tracking SL Length', value=2, start=1, end=10),

            visible=False,
            name='ATR Vortex Parameters'
        )
        triple_ema_params = pn.Column(
            pn.widgets.Checkbox(name='Use Strategy Specific Exits', value=True),
            pn.widgets.Checkbox(name='Split Position Management', value=True),

            pn.widgets.Checkbox(name='Optimize Fast EMA', value=False),
            pn.widgets.IntSlider(name='Fast EMA Period', value=8, start=5, end=20),

            pn.widgets.Checkbox(name='Optimize Medium EMA', value=False),
            pn.widgets.IntSlider(name='Medium EMA Period', value=21, start=15, end=50),

            pn.widgets.Checkbox(name='Optimize Slow EMA', value=False),
            pn.widgets.IntSlider(name='Slow EMA Period', value=55, start=30, end=100),

            pn.widgets.Checkbox(name='Optimize ADX Period', value=False),
            pn.widgets.IntSlider(name='ADX Period', value=14, start=7, end=28),

            pn.widgets.Checkbox(name='Optimize ADX Threshold', value=False),
            pn.widgets.IntSlider(name='ADX Threshold', value=25, start=15, end=40),

            pn.widgets.Checkbox(name='Optimize ATR Period', value=False),
            pn.widgets.IntSlider(name='ATR Period', value=14, start=7, end=28),

            pn.widgets.Checkbox(name='Optimize ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='ATR Multiplier', value=2.0, start=1.0, end=4.0, step=0.1),

            pn.widgets.Checkbox(name='Optimize Trend Lookback', value=False),
            pn.widgets.IntSlider(name='Trend Lookback', value=3, start=2, end=10),

            pn.widgets.Checkbox(name='Optimize Volume Threshold', value=False),
            pn.widgets.IntSlider(name='Min Volume Threshold', value=1000, start=500, end=5000, step=100),

            visible=False,
            name='Triple EMA Parameters'
        )

        rsi_trend_reversal_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize RSI Length', value=False),
            pn.widgets.IntSlider(name='RSI Length', value=8, start=5, end=20),
            pn.widgets.Checkbox(name='Optimize RSI Multiplier', value=False),
            pn.widgets.FloatSlider(name='RSI Multiplier', value=1.5, start=0.5, end=3.0, step=0.1),
            pn.widgets.Checkbox(name='Optimize Lookback', value=False),
            pn.widgets.IntSlider(name='Lookback', value=1, start=1, end=5),
            pn.widgets.Checkbox(name='Use Fixed SLTP', value=True),  # Add this new checkbox
            pn.widgets.Checkbox(name='Optimize SLTP', value=False),
            pn.widgets.FloatSlider(name='SLTP', value=10, start=5, end=20, step=0.5),
            visible=False,
            name='RSI Trend Reversal Parameters'
        )
        alligator_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize Jaw Period', value=False),
            pn.widgets.IntSlider(name='Jaw Period', value=13, start=8, end=20),
            pn.widgets.Checkbox(name='Optimize Teeth Period', value=False),
            pn.widgets.IntSlider(name='Teeth Period', value=8, start=5, end=15),
            pn.widgets.Checkbox(name='Optimize Lips Period', value=False),
            pn.widgets.IntSlider(name='Lips Period', value=5, start=3, end=10),
            pn.widgets.Checkbox(name='Optimize ADX Period', value=False),
            pn.widgets.IntSlider(name='ADX Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize ADX Threshold', value=False),
            pn.widgets.IntSlider(name='ADX Threshold', value=25, start=15, end=40),
            pn.widgets.Checkbox(name='Optimize ATR Period', value=False),
            pn.widgets.IntSlider(name='ATR Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='ATR Multiplier', value=2.0, start=1.0, end=4.0, step=0.1),
            visible=False,
            name='Alligator Parameters'
        )
        emacross_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize Fast EMA', value=False),
            pn.widgets.IntSlider(name='Fast EMA', value=10, start=5, end=50),
            pn.widgets.Checkbox(name='Optimize Slow EMA', value=False),
            pn.widgets.IntSlider(name='Slow EMA', value=20, start=10, end=100),
            pn.widgets.Checkbox(name='Optimize Volume MA', value=False),
            pn.widgets.IntSlider(name='Volume MA', value=20, start=10, end=50),
            pn.widgets.Checkbox(name='Optimize Volume Multiplier', value=False),
            pn.widgets.FloatSlider(name='Volume Multiplier', value=1.5, start=1.1, end=3.0, step=0.1),
            visible=False,
            name='EMA Parameters'
        )
        turtle_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize Period', value=False),
            pn.widgets.IntSlider(name='Breakout Period', value=55, start=20, end=100),
            pn.widgets.Checkbox(name='Optimize ATR Period', value=False),
            pn.widgets.IntSlider(name='ATR Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='ATR Multiplier', value=2.0, start=1.0, end=4.0, step=0.1),
            pn.widgets.Checkbox(name='Optimize Profit Target', value=False),
            pn.widgets.FloatSlider(name='Profit Target %', value=0.2, start=0.1, end=1.0, step=0.1),
            visible=False,
            name='Turtle Parameters'
        )
        bb_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize BB Period', value=False),
            pn.widgets.IntSlider(name='BB Period', value=20, start=10, end=50),
            pn.widgets.Checkbox(name='Optimize BB Std Dev', value=False),
            pn.widgets.FloatSlider(name='BB Std Dev', value=2.0, start=1.0, end=4.0, step=0.1),
            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize RSI Threshold', value=False),
            pn.widgets.IntSlider(name='RSI Threshold', value=50, start=30, end=70),
            visible=False,
            name='Bollinger Parameters'
        )
        macd_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize MACD Fast', value=False),
            pn.widgets.IntSlider(name='MACD Fast', value=12, start=8, end=20),
            pn.widgets.Checkbox(name='Optimize MACD Slow', value=False),
            pn.widgets.IntSlider(name='MACD Slow', value=26, start=20, end=40),
            pn.widgets.Checkbox(name='Optimize MACD Signal', value=False),
            pn.widgets.IntSlider(name='MACD Signal', value=9, start=5, end=15),
            pn.widgets.Checkbox(name='Optimize Trend MA', value=False),
            pn.widgets.IntSlider(name='Trend MA', value=200, start=100, end=300),
            pn.widgets.Checkbox(name='Optimize Min Trend Strength %', value=False),
            pn.widgets.FloatSlider(name='Min Trend Strength %', value=2.0, start=0.5, end=5.0, step=0.1),
            visible=False,
            name='MACD Parameters'
        )
        # Strategy parameter widgets for new strategies
        rsidiv_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=5, end=28),
            pn.widgets.Checkbox(name='Optimize Min Lookback Range', value=False),
            pn.widgets.IntSlider(name='Min Lookback Range', value=3, start=1, end=10),
            pn.widgets.Checkbox(name='Optimize Max Lookback Range', value=False),
            pn.widgets.IntSlider(name='Max Lookback Range', value=10, start=5, end=60),
            visible=False,
            name='RSI Divergence Parameters'
        )
        rsibb_params = pn.Column(
            # Stochastic RSI Parameters
            pn.widgets.Checkbox(name='Optimize Stoch RSI Period', value=False),  # 0
            pn.widgets.IntSlider(name='Stoch RSI Period', value=14, start=5, end=28),  # 1
            pn.widgets.IntSlider(name='Stoch RSI K Period', value=3, start=1, end=14),  # 2
            pn.widgets.IntSlider(name='Stoch RSI D Period', value=3, start=1, end=14),  # 3

            # Bollinger Bands Parameters
            pn.widgets.Checkbox(name='Optimize BB Period', value=False),  # 4
            pn.widgets.IntSlider(name='BB Period', value=20, start=10, end=50),  # 5
            pn.widgets.Checkbox(name='Optimize BB Std Dev', value=False),  # 6
            pn.widgets.FloatSlider(name='BB Std Dev', value=2.0, start=1.0, end=4.0, step=0.1),  # 7

            # Trading Parameters
            pn.widgets.Checkbox(name='Optimize Recent Candles', value=False),  # 8
            pn.widgets.IntSlider(name='Recent Candles', value=5, start=1, end=10),  # 9

            # Lookback Parameter
            pn.widgets.Checkbox(name='Optimize Lookback Range', value=False),  # 10
            pn.widgets.IntSlider(name='Lookback Range', value=3, start=1, end=10),  # 11

            # NEW: SMA Filter Optimization
            pn.widgets.Checkbox(name='Optimize SMA Filter Period', value=False),  # 12
            pn.widgets.IntSlider(name='SMA Filter Period', value=50, start=20, end=200),  # 13

            visible=False,
            name='RSI and Bollinger Parameters'
        )
        donchian_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize Donchian Period', value=False),
            pn.widgets.IntSlider(name='Donchian Period', value=20, start=10, end=50),
            visible=False,
            name='Donchian Channel Parameters'
        )

        atrchandelier_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize ATR Period', value=False),
            pn.widgets.IntSlider(name='ATR Period', value=14, start=7, end=28),
            pn.widgets.Checkbox(name='Optimize ATR Multiplier', value=False),
            pn.widgets.FloatSlider(name='ATR Multiplier', value=3.0, start=1.0, end=5.0, step=0.1),
            visible=False,
            name='ATR Chandelier Exit Parameters'
        )

        keltner_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize Keltner Period', value=False),
            pn.widgets.IntSlider(name='Keltner Period', value=20, start=10, end=50),
            pn.widgets.Checkbox(name='Optimize Keltner Multiplier', value=False),
            pn.widgets.FloatSlider(name='Keltner Multiplier', value=2.0, start=1.0, end=4.0, step=0.1),
            visible=False,
            name='Keltner Channel Parameters'
        )

        channelbreakout_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize Channel Breakout Period', value=False),
            pn.widgets.IntSlider(name='Channel Breakout Period', value=20, start=10, end=50),
            visible=False,
            name='Channel Breakout Parameters'
        )

        macddiv_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize MACD Fast', value=False),
            pn.widgets.IntSlider(name='MACD Fast', value=12, start=8, end=20),
            pn.widgets.Checkbox(name='Optimize MACD Slow', value=False),
            pn.widgets.IntSlider(name='MACD Slow', value=26, start=20, end=40),
            pn.widgets.Checkbox(name='Optimize MACD Signal', value=False),
            pn.widgets.IntSlider(name='MACD Signal', value=9, start=5, end=15),
            pn.widgets.Checkbox(name='Optimize Divergence Threshold', value=False),
            pn.widgets.FloatSlider(name='Divergence Threshold', value=0.02, start=0.01, end=0.05, step=0.001),
            visible=False,
            name='MACD Divergence Parameters'
        )

        linearreg_params = pn.Column(
            pn.widgets.Checkbox(name='Optimize LR Window', value=False),
            pn.widgets.IntSlider(name='LR Window', value=14, start=5, end=50),
            visible=False,
            name='Linear Regression Parameters'
        )

        # Conor Switch parameter widgets
        conor_switch_params = pn.Column(
            # Tenkan-sen parameters
            pn.widgets.Checkbox(name='Optimize Tenkan Period', value=False),
            pn.widgets.IntSlider(name='Tenkan Period', value=9, start=5, end=20),

            # RSI parameters
            pn.widgets.Checkbox(name='Optimize RSI Period', value=False),
            pn.widgets.IntSlider(name='RSI Period', value=14, start=7, end=28),

            # Threshold parameters
            pn.widgets.Checkbox(name='Optimize Overbought Threshold', value=False),
            pn.widgets.IntSlider(name='Overbought Threshold', value=80, start=70, end=90),
            pn.widgets.Checkbox(name='Optimize Oversold Threshold', value=False),
            pn.widgets.IntSlider(name='Oversold Threshold', value=20, start=10, end=30),

            # Divergence parameters
            pn.widgets.Checkbox(name='Optimize Divergence Lookback', value=False),
            pn.widgets.IntSlider(name='Divergence Lookback', value=5, start=3, end=10),
            pn.widgets.Checkbox(name='Optimize Divergence Threshold', value=False),
            pn.widgets.FloatSlider(name='Divergence Threshold %', value=1.0, start=0.1, end=5.0, step=0.1),

            visible=False,
            name='Conor Switch Parameters'
        )

        # Risk management parameters
        base_params = pn.Column(
            pn.widgets.Checkbox(name='Use Stop Loss', value=True),  # Toggle SL
            pn.widgets.Checkbox(name='Use Trailing Stop', value=True),  # Toggle TS
            pn.widgets.Checkbox(name='Use Take Profit', value=True),  # Toggle TP
            pn.widgets.Checkbox(name='Exit on Reversal', value=True),  # Toggle exit on reversal

            pn.widgets.FloatSlider(name='Stop Loss %', value=2.0, start=0.5, end=10.0, step=0.5),

            # Separate trailing stops for long and short positions
            pn.widgets.FloatSlider(name='Trailing Stop % (Long)', value=1.5, start=0.5, end=10.0, step=0.5),
            pn.widgets.FloatSlider(name='Trailing Stop % (Short)', value=2.0, start=0.5, end=10.0, step=0.5),

            pn.widgets.FloatSlider(name='Take Profit %', value=3.0, start=1.0, end=15.0, step=0.5),

            name='Risk Management'
        )

        # Optimization and backtest controls
        optimize_button = pn.widgets.Button(name='Run Optimization', button_type='primary')
        backtest_button = pn.widgets.Button(name='Run Backtest', button_type='primary')
        walk_forward_button = pn.widgets.Button(name='Run Walk-Forward Analysis', button_type='warning')
        back_button = pn.widgets.Button(name='Back', button_type='primary', width=100, align='center')
        optimize_sl_tp_button = pn.widgets.Button(name='Optimize SL/TP')

        # Results containers
        metrics_table = pn.widgets.Tabulator(sizing_mode='stretch_width', header_align='left', show_index=False)
        trades_table = pn.widgets.Tabulator(sizing_mode='stretch_width', header_align='left', show_index=False)
        optimization_results_table = pn.widgets.Tabulator(sizing_mode='stretch_width', header_align='left',
                                                          show_index=False)
        all_optimization_results_table = pn.widgets.Tabulator(sizing_mode='stretch_width', header_align='left',
                                                              show_index=False)
        monthly_analysis_table = pn.widgets.Tabulator(sizing_mode='stretch_width', header_align='left',
                                                      show_index=False)
        yearly_analysis_table = pn.widgets.Tabulator(sizing_mode='stretch_width', header_align='left',
                                                     show_index=False)
        plot_panel = pn.pane.Plotly(sizing_mode='stretch_width', height=900)
        error_message = pn.pane.Markdown('')

        def update_strategy_params(event):
            """Update visible parameters based on strategy selection"""
            # Hide all parameter groups
            ma_params.visible = False
            adaptive_supertrend_params.visible = False
            adaptive_rsi_params.visible = False
            liquidation_params.visible = False
            multi_momentum_params.visible = False
            bb_atr_params.visible = False
            rsi_params.visible = False
            emacross_params.visible = False
            bb_params.visible = False
            rsi_trend_reversal_params.visible = False
            rsibb_params.visible = False
            atr_vortex_params.visible = False
            triple_ema_params.visible = False
            macd_params.visible = False
            rsidiv_params.visible = False
            donchian_params.visible = False
            atrchandelier_params.visible = False
            keltner_params.visible = False
            channelbreakout_params.visible = False
            macddiv_params.visible = False
            linearreg_params.visible = False
            conor_switch_params.visible = False
            turtle_params.visible = False
            alligator_params.visible = False
            sr_vwap_params.visible = False

            # Show only the relevant parameters
            selected_strategy = event.new
            if selected_strategy == 'Triple EMA':
                triple_ema_params.visible = True
            elif selected_strategy == 'Adaptive RSI Volume':
                adaptive_rsi_params.visible = True
            elif selected_strategy == 'Adaptive Supertrend':
                adaptive_supertrend_params.visible = True
            elif selected_strategy == 'Liquidation':
                liquidation_params.visable = True
            elif selected_strategy == 'ATR Bollinger Bands':
                bb_atr_params.visible = True
            elif selected_strategy == 'RSI Trend Reversal':
                rsi_trend_reversal_params.visible = True
            elif selected_strategy == 'Multi-Indicator Momentum':
                multi_momentum_params.visible = True
            elif selected_strategy == 'Enhanced RSI Bollinger':
                enhanced_rsibb_params.visible = True
            elif selected_strategy == 'SR VWAP Strategy':  # This is correct
                sr_vwap_params.visible = True
            elif selected_strategy == 'Moving Average Crossover':
                ma_params.visible = True
            elif selected_strategy == 'Ichimoku TK Cross':
                ichimoku_tk_params.visible = True
            elif selected_strategy == 'RSI':
                rsi_params.visible = True
            elif selected_strategy == 'Ichimoku EMA Cross':
                ichimoku_ema_params.visible = True
            elif selected_strategy == 'EMA Cross with Volume':
                emacross_params.visible = True
            elif selected_strategy == 'RSI and Bollinger':
                rsibb_params.visible = True
            elif selected_strategy == 'Bollinger Bands':
                bb_params.visible = True
            elif selected_strategy == 'MACD with Trend':
                macd_params.visible = True
            elif selected_strategy == 'RSI Divergence':
                rsidiv_params.visible = True
            elif selected_strategy == 'Donchian Channel Breakout':
                donchian_params.visible = True
            elif selected_strategy == 'ATR Chandelier Exit':
                atrchandelier_params.visible = True
            elif selected_strategy == 'Alligator':
                alligator_params.visible = True
            elif selected_strategy == 'Keltner Channel':
                keltner_params.visible = True
            elif selected_strategy == 'Channel Breakout':
                channelbreakout_params.visible = True
            elif selected_strategy == 'MACD Divergence':
                macddiv_params.visible = True
            elif selected_strategy == 'Linear Regression':
                linearreg_params.visible = True
            elif selected_strategy == 'Conor Switch':
                conor_switch_params.visible = True
            elif selected_strategy == 'Turtle Trading':
                turtle_params.visible = True
            if selected_strategy == 'ATR Vortex':
                atr_vortex_params.visible = True

        strategy_select.param.watch(update_strategy_params, 'value')

        def optimize_strategy(event):
            optimize_button = event.obj
            optimize_button.loading = True
            error_message.object = ''
            try:
                # Fetch data
                print("\nStarting optimization process...")
                data = self.fetch_data(symbol.value, interval_select.value, start_date.value, end_date.value)
                if data is None or data.empty:
                    error_message.object = f"⚠️ No data found for {symbol.value}"
                    logger.warning(f"No data found for {symbol.value}")
                    return

                logger.debug("Data fetched successfully.")
                strategy_class = self.strategies[strategy_select.value]
                cash = initial_cash.value

                # Get parameter ranges and base strategy parameters
                param_ranges = {}
                base_strategy_params = {}

                # Extract strategy-specific parameters and set up optimization ranges
                strategy = strategy_select.value
                if strategy == 'Moving Average Crossover':
                    optimize_fast_ma = ma_params[0].value
                    fast_ma_value = ma_params[1].value
                    optimize_slow_ma = ma_params[2].value
                    slow_ma_value = ma_params[3].value

                    if optimize_fast_ma:
                        param_ranges['n1'] = range(max(1, fast_ma_value - 5), fast_ma_value + 6)
                    else:
                        base_strategy_params['n1'] = fast_ma_value

                    if optimize_slow_ma:
                        param_ranges['n2'] = range(max(1, slow_ma_value - 5), slow_ma_value + 6)
                    else:
                        base_strategy_params['n2'] = slow_ma_value

                if strategy == 'Adaptive RSI Volume':
                    # Get parameter values
                    rsi_length = adaptive_rsi_params[1].value
                    rsi_mult = adaptive_rsi_params[3].value
                    atr_period = adaptive_rsi_params[5].value
                    volatility_mult = adaptive_rsi_params[7].value
                    trend_ema = adaptive_rsi_params[9].value
                    volume_ma_period = adaptive_rsi_params[11].value
                    min_volume_mult = adaptive_rsi_params[13].value
                    rsi_bounds = adaptive_rsi_params[15].value
                    min_swing_pct = adaptive_rsi_params[17].value
                    max_risk_per_trade = adaptive_rsi_params[19].value

                    # Base parameters
                    base_strategy_params = {
                        'rsi_length': rsi_length,
                        'rsi_mult': rsi_mult,
                        'atr_period': atr_period,
                        'volatility_mult': volatility_mult,
                        'trend_ema': trend_ema,
                        'volume_ma_period': volume_ma_period,
                        'min_volume_mult': min_volume_mult,
                        'rsi_oversold': rsi_bounds[0],
                        'rsi_overbought': rsi_bounds[1],
                        'min_swing_pct': min_swing_pct,
                        'max_risk_per_trade': max_risk_per_trade / 100
                    }

                    # Add optimization ranges if selected
                    if adaptive_rsi_params[0].value:  # Optimize RSI Length
                        param_ranges['rsi_length'] = range(max(5, rsi_length - 3), rsi_length + 4)
                        base_strategy_params.pop('rsi_length')

                    if adaptive_rsi_params[2].value:  # Optimize RSI Multiplier
                        param_ranges['rsi_mult'] = [round(rsi_mult + i * 0.1, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('rsi_mult')

                    if adaptive_rsi_params[4].value:  # Optimize ATR Period
                        param_ranges['atr_period'] = range(max(7, atr_period - 3), atr_period + 4)
                        base_strategy_params.pop('atr_period')

                    if adaptive_rsi_params[6].value:  # Optimize Volatility Multiplier
                        param_ranges['volatility_mult'] = [round(volatility_mult + i * 0.1, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('volatility_mult')

                    if adaptive_rsi_params[8].value:  # Optimize Trend EMA
                        param_ranges['trend_ema'] = range(max(100, trend_ema - 50), trend_ema + 51, 25)
                        base_strategy_params.pop('trend_ema')

                    if adaptive_rsi_params[10].value:  # Optimize Volume MA Period
                        param_ranges['volume_ma_period'] = range(max(10, volume_ma_period - 5), volume_ma_period + 6)
                        base_strategy_params.pop('volume_ma_period')

                    if adaptive_rsi_params[12].value:  # Optimize Min Volume Multiplier
                        param_ranges['min_volume_mult'] = [round(min_volume_mult + i * 0.1, 1) for i in range(-2, 3)]
                        base_strategy_params.pop('min_volume_mult')

                    if adaptive_rsi_params[14].value:  # Optimize RSI Bounds
                        param_ranges['rsi_oversold'] = range(max(20, rsi_bounds[0] - 5), rsi_bounds[0] + 6)
                        param_ranges['rsi_overbought'] = range(rsi_bounds[1] - 5, min(80, rsi_bounds[1] + 6))
                        base_strategy_params.pop('rsi_oversold')
                        base_strategy_params.pop('rsi_overbought')

                    if adaptive_rsi_params[16].value:  # Optimize Min Swing Percentage
                        param_ranges['min_swing_pct'] = [round(min_swing_pct + i * 0.1, 1) for i in range(-3, 4)]
                        base_strategy_params.pop('min_swing_pct')

                    if adaptive_rsi_params[18].value:  # Optimize Max Risk Per Trade
                        param_ranges['max_risk_per_trade'] = [round((max_risk_per_trade + i * 0.5) / 100, 3) for i in
                                                              range(-4, 5)]
                        base_strategy_params.pop('max_risk_per_trade')

                elif strategy == 'Adaptive Supertrend':
                    # Get parameter values
                    choppiness_period = adaptive_supertrend_params[1].value
                    choppiness_threshold = adaptive_supertrend_params[3].value
                    atr_period = adaptive_supertrend_params[5].value
                    atr_multiplier = adaptive_supertrend_params[7].value
                    rsi_bb_timeframe = adaptive_supertrend_params[16].value
                    rsi_period = adaptive_supertrend_params[9].value
                    bb_period = adaptive_supertrend_params[11].value
                    bb_std = adaptive_supertrend_params[13].value
                    lookback_range = adaptive_supertrend_params[15].value

                    # Base parameters
                    base_strategy_params = {
                        'choppiness_period': choppiness_period,
                        'choppiness_threshold': choppiness_threshold,
                        'rsi_bb_timeframe': rsi_bb_timeframe,
                        'atr_period': atr_period,
                        'atr_multiplier': atr_multiplier,
                        'rsi_period': rsi_period,
                        'bb_period': bb_period,
                        'bb_std_dev': bb_std,
                        'lookback_range': lookback_range
                    }

                    # Add optimization ranges if selected
                    if adaptive_supertrend_params[0].value:  # Optimize Choppiness Period
                        param_ranges['choppiness_period'] = range(
                            max(5, choppiness_period - 5),
                            choppiness_period + 6
                        )
                        base_strategy_params.pop('choppiness_period')

                    if adaptive_supertrend_params[2].value:  # Optimize Choppiness Threshold
                        param_ranges['choppiness_threshold'] = [
                            round(choppiness_threshold + i * 2.0, 1)
                            for i in range(-5, 6)
                        ]
                        base_strategy_params.pop('choppiness_threshold')

                    if adaptive_supertrend_params[4].value:  # Optimize ATR Period
                        param_ranges['atr_period'] = range(
                            max(5, atr_period - 3),
                            atr_period + 4
                        )
                        base_strategy_params.pop('atr_period')

                    if adaptive_supertrend_params[6].value:  # Optimize ATR Multiplier
                        param_ranges['atr_multiplier'] = [
                            round(atr_multiplier + i * 0.2, 1)
                            for i in range(-5, 6)
                        ]
                        base_strategy_params.pop('atr_multiplier')

                    if adaptive_supertrend_params[8].value:  # Optimize RSI Period
                        param_ranges['rsi_period'] = range(
                            max(5, rsi_period - 3),
                            rsi_period + 4
                        )
                        base_strategy_params.pop('rsi_period')

                    if adaptive_supertrend_params[10].value:  # Optimize BB Period
                        param_ranges['bb_period'] = range(
                            max(10, bb_period - 5),
                            bb_period + 6
                        )
                        base_strategy_params.pop('bb_period')

                    if adaptive_supertrend_params[12].value:  # Optimize BB Std Dev
                        param_ranges['bb_std_dev'] = [
                            round(bb_std + i * 0.1, 1)
                            for i in range(-5, 6)
                        ]
                        base_strategy_params.pop('bb_std_dev')

                    if adaptive_supertrend_params[14].value:  # Optimize Lookback Range
                        param_ranges['lookback_range'] = range(
                            max(1, lookback_range - 2),
                            lookback_range + 3
                        )
                        base_strategy_params.pop('lookback_range')

                    print("\nOptimizing Adaptive Supertrend Strategy")
                    print("Parameter Ranges:")
                    for param, range_vals in param_ranges.items():
                        print(f"{param}: {list(range_vals)}")
                    print("\nBase Parameters:")
                    for param, value in base_strategy_params.items():
                        print(f"{param}: {value}")

                elif strategy == 'Liquidation Strategy':
                    liq_threshold = liquidation_params[1].value
                    liq_window = liquidation_params[3].value
                    liq_spike_mult = liquidation_params[5].value
                    volume_mult = liquidation_params[7].value
                    price_rebound = liquidation_params[9].value
                    rsi_period = liquidation_params[11].value
                    rsi_oversold = liquidation_params[13].value

                    base_strategy_params = {
                        'liq_threshold': liq_threshold,
                        'liq_window': liq_window,
                        'liq_spike_mult': liq_spike_mult,
                        'volume_mult': volume_mult,
                        'price_rebound': price_rebound / 100,  # Convert to decimal
                        'rsi_period': rsi_period,
                        'rsi_oversold': rsi_oversold
                    }

                    if liquidation_params[0].value:
                        param_ranges['liq_threshold'] = range(max(100000, liq_threshold - 500000),
                                                              min(5000000, liq_threshold + 500001),
                                                              100000)
                        base_strategy_params.pop('liq_threshold')

                    if liquidation_params[2].value:
                        param_ranges['liq_window'] = range(max(6, liq_window - 3),
                                                           liq_window + 4)
                        base_strategy_params.pop('liq_window')

                    if liquidation_params[4].value:
                        param_ranges['liq_spike_mult'] = [round(liq_spike_mult + i * 0.1, 1)
                                                          for i in range(-5, 6)]
                        base_strategy_params.pop('liq_spike_mult')

                    if liquidation_params[6].value:
                        param_ranges['volume_mult'] = [round(volume_mult + i * 0.1, 1)
                                                       for i in range(-5, 6)]
                        base_strategy_params.pop('volume_mult')

                    if liquidation_params[8].value:
                        param_ranges['price_rebound'] = [round((price_rebound + i * 0.1) / 100, 3)
                                                         for i in range(-5, 6)]
                        base_strategy_params.pop('price_rebound')

                    if liquidation_params[10].value:
                        param_ranges['rsi_period'] = range(max(7, rsi_period - 3),
                                                           rsi_period + 4)
                        base_strategy_params.pop('rsi_period')

                    if liquidation_params[12].value:
                        param_ranges['rsi_oversold'] = range(max(20, rsi_oversold - 5),
                                                             min(40, rsi_oversold + 6))
                        base_strategy_params.pop('rsi_oversold')
                elif strategy == 'ATR Bollinger Bands':

                    # Get widget values by name

                    bb_widgets = {w.name: w for w in bb_atr_params}

                    base_strategy_params = {

                        'bblength': bb_widgets['BB Length'].value,

                        'bbstdev': bb_widgets['BB StdDev'].value,

                        'lena': bb_widgets['ATR Length'].value,

                        'mult': bb_widgets['ATR Multiplier'].value,

                        'use_filter': bb_widgets['Use Filter'].value,

                        'use_atr_bb': bb_widgets['Use ATR BB'].value

                    }

                    # Add optimization ranges if selected

                    if bb_widgets['Optimize BB Length'].value:
                        param_ranges['bblength'] = range(

                            max(20, base_strategy_params['bblength'] - 10),

                            base_strategy_params['bblength'] + 11

                        )

                        base_strategy_params.pop('bblength')

                    if bb_widgets['Optimize BB StdDev'].value:
                        param_ranges['bbstdev'] = [

                            round(base_strategy_params['bbstdev'] + i * 0.05, 2)

                            for i in range(-4, 5)

                        ]

                        base_strategy_params.pop('bbstdev')

                    if bb_widgets['Optimize ATR Length'].value:
                        param_ranges['lena'] = range(

                            max(10, base_strategy_params['lena'] - 5),

                            base_strategy_params['lena'] + 6

                        )

                        base_strategy_params.pop('lena')

                    if bb_widgets['Optimize ATR Multiplier'].value:
                        param_ranges['mult'] = [

                            round(base_strategy_params['mult'] + i * 0.1, 2)

                            for i in range(-5, 6)

                        ]

                        base_strategy_params.pop('mult')

                elif strategy == 'Enhanced RSI Bollinger':
                    # Get parameter values
                    rsi_period = enhanced_rsibb_params[1].value
                    rsi_thresholds = enhanced_rsibb_params[3].value
                    counter_trend_rsi = enhanced_rsibb_params[5].value
                    bb_period = enhanced_rsibb_params[7].value
                    bb_std = enhanced_rsibb_params[9].value
                    ma_period = enhanced_rsibb_params[11].value
                    rvol_period = enhanced_rsibb_params[13].value
                    rvol_threshold = enhanced_rsibb_params[15].value
                    recent_candles = enhanced_rsibb_params[17].value
                    lookback_range = enhanced_rsibb_params[19].value

                    # Set base parameters
                    base_strategy_params = {
                        'rsi_period': rsi_period,
                        'rsi_oversold': rsi_thresholds[0],
                        'rsi_overbought': rsi_thresholds[1],
                        'counter_trend_rsi': counter_trend_rsi,
                        'bb_period': bb_period,
                        'bb_std_dev': bb_std,
                        'ma_period': ma_period,
                        'rvol_period': rvol_period,
                        'rvol_threshold': rvol_threshold,
                        'recent_candles': recent_candles,
                        'lookback_range': lookback_range
                    }

                    # Add optimization ranges if selected
                    if enhanced_rsibb_params[0].value:  # Optimize RSI Period
                        param_ranges['rsi_period'] = range(max(5, rsi_period - 3), rsi_period + 4)
                        base_strategy_params.pop('rsi_period')

                    if enhanced_rsibb_params[2].value:  # Optimize RSI Thresholds
                        param_ranges['rsi_oversold'] = range(max(20, rsi_thresholds[0] - 5), rsi_thresholds[0] + 6)
                        param_ranges['rsi_overbought'] = range(rsi_thresholds[1] - 5, min(80, rsi_thresholds[1] + 6))
                        base_strategy_params.pop('rsi_oversold')
                        base_strategy_params.pop('rsi_overbought')

                    if enhanced_rsibb_params[4].value:  # Optimize Counter-Trend RSI
                        param_ranges['counter_trend_rsi'] = range(max(70, counter_trend_rsi - 5),
                                                                  min(90, counter_trend_rsi + 6))
                        base_strategy_params.pop('counter_trend_rsi')

                    if enhanced_rsibb_params[6].value:  # Optimize BB Period
                        param_ranges['bb_period'] = range(max(10, bb_period - 5), bb_period + 6)
                        base_strategy_params.pop('bb_period')

                    if enhanced_rsibb_params[8].value:  # Optimize BB Std Dev
                        param_ranges['bb_std_dev'] = [round(bb_std + i * 0.1, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('bb_std_dev')

                    if enhanced_rsibb_params[10].value:  # Optimize MA Period
                        param_ranges['ma_period'] = range(max(50, ma_period - 50), ma_period + 51, 25)
                        base_strategy_params.pop('ma_period')

                    if enhanced_rsibb_params[12].value:  # Optimize RVOL Period
                        param_ranges['rvol_period'] = range(max(10, rvol_period - 5), rvol_period + 6)
                        base_strategy_params.pop('rvol_period')

                    if enhanced_rsibb_params[14].value:  # Optimize RVOL Threshold
                        param_ranges['rvol_threshold'] = [round(rvol_threshold + i * 0.1, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('rvol_threshold')

                    if enhanced_rsibb_params[16].value:  # Optimize Recent Candles
                        param_ranges['recent_candles'] = range(max(1, recent_candles - 2), recent_candles + 3)
                        base_strategy_params.pop('recent_candles')

                    if enhanced_rsibb_params[18].value:  # Optimize Lookback Range
                        param_ranges['lookback_range'] = range(max(1, lookback_range - 2), lookback_range + 3)
                        base_strategy_params.pop('lookback_range')
                elif strategy == 'SR VWAP Strategy':

                    # Get parameter values

                    pivot_length = sr_vwap_params[1].value

                    sr_strength = sr_vwap_params[3].value

                    atr_period = sr_vwap_params[5].value

                    zone_size = sr_vwap_params[7].value

                    min_bars = sr_vwap_params[9].value

                    vwap_length = sr_vwap_params[11].value

                    trend_strength = sr_vwap_params[13].value

                    # Set base parameters

                    base_strategy_params = {

                        'pivot_length': pivot_length,

                        'sr_strength': sr_strength,

                        'atr_period': atr_period,

                        'zone_size_atr': zone_size,

                        'min_bars_between_signals': min_bars,

                        'vwap_ma_length': vwap_length,

                        'min_vwap_trend_strength': trend_strength

                    }

                    # Add optimization ranges if selected

                    if sr_vwap_params[0].value:  # Optimize Pivot Length

                        param_ranges['pivot_length'] = range(

                            max(10, pivot_length - 5),

                            pivot_length + 6

                        )

                        base_strategy_params.pop('pivot_length')

                    if sr_vwap_params[2].value:  # Optimize SR Strength

                        param_ranges['sr_strength'] = range(1, 6)

                        base_strategy_params.pop('sr_strength')

                    if sr_vwap_params[4].value:  # Optimize ATR Period

                        param_ranges['atr_period'] = range(

                            max(10, atr_period - 5),

                            atr_period + 6

                        )

                        base_strategy_params.pop('atr_period')

                    if sr_vwap_params[6].value:  # Optimize Zone Size

                        param_ranges['zone_size_atr'] = [

                            round(zone_size + i * 0.025, 3) for i in range(-2, 3)

                        ]

                        base_strategy_params.pop('zone_size_atr')

                    if sr_vwap_params[8].value:  # Optimize Min Bars

                        param_ranges['min_bars_between_signals'] = range(

                            max(5, min_bars - 5),

                            min_bars + 6

                        )

                        base_strategy_params.pop('min_bars_between_signals')

                    if sr_vwap_params[10].value:  # Optimize VWAP Length

                        param_ranges['vwap_ma_length'] = range(

                            max(10, vwap_length - 5),

                            vwap_length + 6

                        )

                        base_strategy_params.pop('vwap_ma_length')

                    if sr_vwap_params[12].value:  # Optimize Trend Strength

                        param_ranges['min_vwap_trend_strength'] = [

                            round(trend_strength + i * 0.0005, 4) for i in range(-2, 3)

                        ]

                        base_strategy_params.pop('min_vwap_trend_strength')

                elif strategy == 'ATR Vortex':
                    # Get parameters using widget names
                    first_atr_length = next(w for w in atr_vortex_params if w.name == 'First ATR Length').value
                    first_atr_mult = next(w for w in atr_vortex_params if w.name == 'First ATR Multiplier').value
                    second_atr_length = next(w for w in atr_vortex_params if w.name == 'Second ATR Length').value
                    second_atr_mult = next(w for w in atr_vortex_params if w.name == 'Second ATR Multiplier').value
                    vi_period = next(w for w in atr_vortex_params if w.name == 'VI Period').value
                    apply_tema = next(w for w in atr_vortex_params if w.name == 'Apply TEMA').value
                    fixed_targets = next(w for w in atr_vortex_params if w.name == 'Fixed Targets').value
                    repaints = next(w for w in atr_vortex_params if w.name == 'Repaints').value
                    use_strategy_specific_exits = next(
                        w for w in atr_vortex_params if w.name == 'Use Strategy Specific Exits').value
                    tracking_sl_length = next(w for w in atr_vortex_params if w.name == 'Tracking SL Length').value

                    # Set base parameters
                    base_strategy_params = {
                        'first_atr_length': first_atr_length,
                        'first_atr_mult': first_atr_mult,
                        'second_atr_length': second_atr_length,
                        'second_atr_mult': second_atr_mult,
                        'vi_period': vi_period,
                        'apply_tema': apply_tema,
                        'fixed_targets': fixed_targets,
                        'repaints': repaints,
                        'use_strategy_specific_exits': use_strategy_specific_exits,
                        'tracking_sl_length': tracking_sl_length
                    }

                    # Get optimization checkboxes
                    optimize_first_atr_length = next(
                        w for w in atr_vortex_params if w.name == 'Optimize First ATR Length').value
                    optimize_first_atr_mult = next(
                        w for w in atr_vortex_params if w.name == 'Optimize First ATR Multiplier').value
                    optimize_second_atr_length = next(
                        w for w in atr_vortex_params if w.name == 'Optimize Second ATR Length').value
                    optimize_second_atr_mult = next(
                        w for w in atr_vortex_params if w.name == 'Optimize Second ATR Multiplier').value
                    optimize_vi_period = next(w for w in atr_vortex_params if w.name == 'Optimize VI Period').value
                    optimize_tracking_sl = next(
                        w for w in atr_vortex_params if w.name == 'Optimize Tracking SL Length').value

                    # Add optimization ranges if selected
                    if optimize_first_atr_length:
                        param_ranges['first_atr_length'] = range(max(5, first_atr_length - 5), first_atr_length + 6)
                        base_strategy_params.pop('first_atr_length')

                    if optimize_first_atr_mult:
                        param_ranges['first_atr_mult'] = [round(first_atr_mult + i * 0.1, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('first_atr_mult')

                    if optimize_second_atr_length:
                        param_ranges['second_atr_length'] = range(max(5, second_atr_length - 5), second_atr_length + 6)
                        base_strategy_params.pop('second_atr_length')

                    if optimize_second_atr_mult:
                        param_ranges['second_atr_mult'] = [round(second_atr_mult + i * 0.1, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('second_atr_mult')

                    if optimize_vi_period:
                        param_ranges['vi_period'] = range(max(5, vi_period - 5), vi_period + 6)
                        base_strategy_params.pop('vi_period')

                    if optimize_tracking_sl:
                        param_ranges['tracking_sl_length'] = range(max(1, tracking_sl_length - 2),
                                                                   tracking_sl_length + 3)
                        base_strategy_params.pop('tracking_sl_length')

                elif strategy == 'Ichimoku TK Cross':
                    # Get parameter values
                    conversion_period = ichimoku_tk_params[1].value
                    base_period = ichimoku_tk_params[3].value
                    lagging_span2_period = ichimoku_tk_params[5].value
                    displacement = ichimoku_tk_params[7].value
                    adx_period = ichimoku_tk_params[9].value
                    adx_threshold = ichimoku_tk_params[11].value
                    rsi_period = ichimoku_tk_params[13].value
                    rsi_bounds = ichimoku_tk_params[15].value
                    use_cloud = ichimoku_tk_params[16].value

                    # Set base parameters
                    base_strategy_params = {
                        'conversion_periods': conversion_period,
                        'base_periods': base_period,
                        'lagging_span2_periods': lagging_span2_period,
                        'displacement': displacement,
                        'adx_period': adx_period,
                        'adx_threshold': adx_threshold,
                        'rsi_period': rsi_period,
                        'rsi_oversold': rsi_bounds[0],
                        'rsi_overbought': rsi_bounds[1],
                        'use_cloud_filter': use_cloud
                    }

                    # Add optimization ranges if selected
                    if ichimoku_tk_params[0].value:  # Optimize Conversion Period
                        param_ranges['conversion_periods'] = range(
                            max(10, conversion_period - 5),
                            conversion_period + 6
                        )
                        base_strategy_params.pop('conversion_periods')

                    if ichimoku_tk_params[2].value:  # Optimize Base Period
                        param_ranges['base_periods'] = range(
                            max(40, base_period - 10),
                            base_period + 11
                        )
                        base_strategy_params.pop('base_periods')

                    if ichimoku_tk_params[4].value:  # Optimize Lagging Span 2 Period
                        param_ranges['lagging_span2_periods'] = range(
                            max(80, lagging_span2_period - 20),
                            lagging_span2_period + 21
                        )
                        base_strategy_params.pop('lagging_span2_periods')

                    if ichimoku_tk_params[6].value:  # Optimize Displacement
                        param_ranges['displacement'] = range(
                            max(20, displacement - 5),
                            displacement + 6
                        )
                        base_strategy_params.pop('displacement')

                    if ichimoku_tk_params[8].value:  # Optimize ADX Period
                        param_ranges['adx_period'] = range(
                            max(10, adx_period - 5),
                            adx_period + 6
                        )
                        base_strategy_params.pop('adx_period')

                    if ichimoku_tk_params[10].value:  # Optimize ADX Threshold
                        param_ranges['adx_threshold'] = range(
                            max(15, adx_threshold - 5),
                            adx_threshold + 6
                        )
                        base_strategy_params.pop('adx_threshold')

                    if ichimoku_tk_params[12].value:  # Optimize RSI Period
                        param_ranges['rsi_period'] = range(
                            max(7, rsi_period - 3),
                            rsi_period + 4
                        )
                        base_strategy_params.pop('rsi_period')

                    if ichimoku_tk_params[14].value:  # Optimize RSI Bounds
                        param_ranges['rsi_oversold'] = range(
                            max(20, rsi_bounds[0] - 5),
                            rsi_bounds[0] + 6
                        )
                        param_ranges['rsi_overbought'] = range(
                            rsi_bounds[1] - 5,
                            min(80, rsi_bounds[1] + 6)
                        )
                        base_strategy_params.pop('rsi_oversold')
                        base_strategy_params.pop('rsi_overbought')

                elif strategy == 'Ichimoku EMA Cross':
                    # Get parameter values
                    conversion_period = ichimoku_ema_params[1].value
                    base_period = ichimoku_ema_params[3].value
                    lagging_span2_period = ichimoku_ema_params[5].value
                    displacement = ichimoku_ema_params[7].value
                    ema_length = ichimoku_ema_params[9].value

                    # Set base parameters
                    base_strategy_params = {
                        'conversion_periods': conversion_period,
                        'base_periods': base_period,
                        'lagging_span2_periods': lagging_span2_period,
                        'displacement': displacement,
                        'ema_length': ema_length
                    }

                    # Add optimization ranges if selected
                    if ichimoku_ema_params[0].value:  # Optimize Conversion Period
                        param_ranges['conversion_periods'] = range(
                            max(10, conversion_period - 5),
                            conversion_period + 6
                        )
                        base_strategy_params.pop('conversion_periods')

                    if ichimoku_ema_params[2].value:  # Optimize Base Period
                        param_ranges['base_periods'] = range(
                            max(40, base_period - 10),
                            base_period + 11
                        )
                        base_strategy_params.pop('base_periods')

                    if ichimoku_ema_params[4].value:  # Optimize Lagging Span 2 Period
                        param_ranges['lagging_span2_periods'] = range(
                            max(80, lagging_span2_period - 20),
                            lagging_span2_period + 21
                        )
                        base_strategy_params.pop('lagging_span2_periods')

                    if ichimoku_ema_params[6].value:  # Optimize Displacement
                        param_ranges['displacement'] = range(
                            max(20, displacement - 5),
                            displacement + 6
                        )
                        base_strategy_params.pop('displacement')

                    if ichimoku_ema_params[8].value:  # Optimize EMA Length
                        param_ranges['ema_length'] = range(
                            max(150, ema_length - 25),
                            ema_length + 26
                        )
                        base_strategy_params.pop('ema_length')

                elif strategy == 'Multi-Indicator Momentum':
                    # Get parameters
                    sma_period = next(w for w in multi_momentum_params if w.name == 'SMA Period').value
                    macd_fast = next(w for w in multi_momentum_params if w.name == 'MACD Fast Period').value
                    macd_slow = next(w for w in multi_momentum_params if w.name == 'MACD Slow Period').value
                    macd_signal = next(w for w in multi_momentum_params if w.name == 'MACD Signal Period').value
                    rsi_period = next(w for w in multi_momentum_params if w.name == 'RSI Period').value
                    rsi_long_bounds = next(w for w in multi_momentum_params if w.name == 'RSI Long Entry Bounds').value
                    rsi_short_bounds = next(
                        w for w in multi_momentum_params if w.name == 'RSI Short Entry Bounds').value
                    rsi_exit_bounds = next(w for w in multi_momentum_params if w.name == 'RSI Exit Bounds').value
                    bb_period = next(w for w in multi_momentum_params if w.name == 'BB Period').value
                    bb_std = next(w for w in multi_momentum_params if w.name == 'BB STD').value
                    atr_period = next(w for w in multi_momentum_params if w.name == 'ATR Period').value
                    atr_multiplier = next(w for w in multi_momentum_params if w.name == 'ATR Stop Multiplier').value
                    max_drawdown = next(w for w in multi_momentum_params if w.name == 'Max Drawdown %').value

                    # Set base parameters
                    base_strategy_params = {
                        'sma_period': sma_period,
                        'macd_fast': macd_fast,
                        'macd_slow': macd_slow,
                        'macd_signal': macd_signal,
                        'rsi_period': rsi_period,
                        'rsi_lower': rsi_long_bounds[0],
                        'rsi_upper': rsi_long_bounds[1],
                        'rsi_short_lower': rsi_short_bounds[0],
                        'rsi_short_upper': rsi_short_bounds[1],
                        'rsi_exit_lower': rsi_exit_bounds[0],
                        'rsi_exit_upper': rsi_exit_bounds[1],
                        'bb_period': bb_period,
                        'bb_std': bb_std,
                        'atr_period': atr_period,
                        'atr_stop_multiplier': atr_multiplier,
                        'max_drawdown_pct': max_drawdown
                    }

                    # Add optimization ranges if selected
                    if next(w for w in multi_momentum_params if w.name == 'Optimize SMA Period').value:
                        param_ranges['sma_period'] = range(max(10, sma_period - 5), sma_period + 6)
                        base_strategy_params.pop('sma_period')

                    if next(w for w in multi_momentum_params if w.name == 'Optimize MACD Fast').value:
                        param_ranges['macd_fast'] = range(max(8, macd_fast - 2), macd_fast + 3)
                        base_strategy_params.pop('macd_fast')

                    if next(w for w in multi_momentum_params if w.name == 'Optimize MACD Slow').value:
                        param_ranges['macd_slow'] = range(max(20, macd_slow - 3), macd_slow + 4)
                        base_strategy_params.pop('macd_slow')

                    if next(w for w in multi_momentum_params if w.name == 'Optimize MACD Signal').value:
                        param_ranges['macd_signal'] = range(max(5, macd_signal - 2), macd_signal + 3)
                        base_strategy_params.pop


                elif strategy == 'Alligator':
                    # Get parameter values
                    jaw_period = alligator_params[1].value
                    teeth_period = alligator_params[3].value
                    lips_period = alligator_params[5].value
                    adx_period = alligator_params[7].value
                    adx_threshold = alligator_params[9].value
                    atr_period = alligator_params[11].value
                    atr_multiplier = alligator_params[13].value

                    # Base parameters (including fixed offset values)
                    base_strategy_params = {
                        'jaw_period': jaw_period,
                        'teeth_period': teeth_period,
                        'lips_period': lips_period,
                        'adx_period': adx_period,
                        'adx_threshold': adx_threshold,
                        'atr_period': atr_period,
                        'atr_multiplier': atr_multiplier,
                        'jaw_offset': 8,
                        'teeth_offset': 5,
                        'lips_offset': 3
                    }

                    # Add optimization ranges if selected
                    if alligator_params[0].value:  # Optimize Jaw Period
                        param_ranges['jaw_period'] = range(max(8, jaw_period - 3),
                                                           jaw_period + 4)
                        base_strategy_params.pop('jaw_period')

                    if alligator_params[2].value:  # Optimize Teeth Period
                        param_ranges['teeth_period'] = range(max(5, teeth_period - 2),
                                                             teeth_period + 3)
                        base_strategy_params.pop('teeth_period')

                    if alligator_params[4].value:  # Optimize Lips Period
                        param_ranges['lips_period'] = range(max(3, lips_period - 2),
                                                            lips_period + 3)
                        base_strategy_params.pop('lips_period')

                    if alligator_params[6].value:  # Optimize ADX Period
                        param_ranges['adx_period'] = range(max(7, adx_period - 3),
                                                           adx_period + 4)
                        base_strategy_params.pop('adx_period')

                    if alligator_params[8].value:  # Optimize ADX Threshold
                        param_ranges['adx_threshold'] = range(max(15, adx_threshold - 5),
                                                              min(40, adx_threshold + 6))
                        base_strategy_params.pop('adx_threshold')

                    if alligator_params[10].value:  # Optimize ATR Period
                        param_ranges['atr_period'] = range(max(7, atr_period - 3),
                                                           atr_period + 4)
                        base_strategy_params.pop('atr_period')

                    if alligator_params[12].value:  # Optimize ATR Multiplier
                        param_ranges['atr_multiplier'] = [round(atr_multiplier + i * 0.1, 1)
                                                          for i in range(-5, 6)]
                        base_strategy_params.pop('atr_multiplier')

                    print("\nAlligator Strategy Optimization Parameters:")
                    print("Base Parameters:", base_strategy_params)
                    print("Parameter Ranges:", param_ranges)
                elif strategy == 'Triple EMA':
                    # Get parameter values
                    use_strategy_exits = triple_ema_params[0].value
                    split_position = triple_ema_params[1].value
                    ema_fast = triple_ema_params[3].value
                    ema_medium = triple_ema_params[5].value
                    ema_slow = triple_ema_params[7].value
                    adx_period = triple_ema_params[9].value
                    adx_threshold = triple_ema_params[11].value
                    atr_period = triple_ema_params[13].value
                    atr_multiplier = triple_ema_params[15].value
                    trend_lookback = triple_ema_params[17].value
                    volume_threshold = triple_ema_params[19].value

                    # Base parameters
                    base_strategy_params = {
                        'use_strategy_specific_exits': use_strategy_exits,
                        'split_position': split_position,
                        'ema_fast': ema_fast,
                        'ema_medium': ema_medium,
                        'ema_slow': ema_slow,
                        'adx_period': adx_period,
                        'adx_threshold': adx_threshold,
                        'atr_period': atr_period,
                        'atr_multiplier': atr_multiplier,
                        'trend_lookback': trend_lookback,
                        'min_volume_threshold': volume_threshold
                    }

                    # Add optimization ranges if selected
                    if triple_ema_params[2].value:  # Optimize Fast EMA
                        param_ranges['ema_fast'] = range(max(5, ema_fast - 3), ema_fast + 4)
                        base_strategy_params.pop('ema_fast')

                    if triple_ema_params[4].value:  # Optimize Medium EMA
                        param_ranges['ema_medium'] = range(max(15, ema_medium - 5), ema_medium + 6)
                        base_strategy_params.pop('ema_medium')

                    if triple_ema_params[6].value:  # Optimize Slow EMA
                        param_ranges['ema_slow'] = range(max(30, ema_slow - 10), ema_slow + 11)
                        base_strategy_params.pop('ema_slow')

                    if triple_ema_params[8].value:  # Optimize ADX Period
                        param_ranges['adx_period'] = range(max(7, adx_period - 3), adx_period + 4)
                        base_strategy_params.pop('adx_period')

                    if triple_ema_params[10].value:  # Optimize ADX Threshold
                        param_ranges['adx_threshold'] = range(max(15, adx_threshold - 5), adx_threshold + 6)
                        base_strategy_params.pop('adx_threshold')

                    if triple_ema_params[12].value:  # Optimize ATR Period
                        param_ranges['atr_period'] = range(max(7, atr_period - 3), atr_period + 4)
                        base_strategy_params.pop('atr_period')

                    if triple_ema_params[14].value:  # Optimize ATR Multiplier
                        param_ranges['atr_multiplier'] = [round(atr_multiplier + i * 0.1, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('atr_multiplier')

                    if triple_ema_params[16].value:  # Optimize Trend Lookback
                        param_ranges['trend_lookback'] = range(max(2, trend_lookback - 2), trend_lookback + 3)
                        base_strategy_params.pop('trend_lookback')

                    if triple_ema_params[18].value:  # Optimize Volume Threshold
                        param_ranges['min_volume_threshold'] = range(max(500, volume_threshold - 500),
                                                                     volume_threshold + 501, 100)
                        base_strategy_params.pop('min_volume_threshold')


                elif strategy == 'RSI Trend Reversal':
                    # Get current parameter values
                    rsi_length = next(w for w in rsi_trend_reversal_params if w.name == 'RSI Length').value
                    rsi_mult = next(w for w in rsi_trend_reversal_params if w.name == 'RSI Multiplier').value
                    lookback = next(w for w in rsi_trend_reversal_params if w.name == 'Lookback').value
                    use_fixed_sltp = next(w for w in rsi_trend_reversal_params if w.name == 'Use Fixed SLTP').value
                    sltp = next(w for w in rsi_trend_reversal_params if w.name == 'SLTP').value

                    # Set base parameters
                    base_strategy_params = {
                        'rsi_length': rsi_length,
                        'rsi_mult': rsi_mult,
                        'lookback': lookback,
                        'use_fixed_sltp': use_fixed_sltp,
                        'sltp': sltp
                    }

                    # Get optimization checkboxes
                    optimize_rsi_length = next(
                        w for w in rsi_trend_reversal_params if w.name == 'Optimize RSI Length').value
                    optimize_rsi_mult = next(
                        w for w in rsi_trend_reversal_params if w.name == 'Optimize RSI Multiplier').value
                    optimize_lookback = next(
                        w for w in rsi_trend_reversal_params if w.name == 'Optimize Lookback').value
                    optimize_sltp = next(w for w in rsi_trend_reversal_params if w.name == 'Optimize SLTP').value

                    # Add optimization ranges if selected
                    if optimize_rsi_length:
                        param_ranges['rsi_length'] = range(max(2, rsi_length - 3), rsi_length + 4)
                        base_strategy_params.pop('rsi_length')

                    if optimize_rsi_mult:
                        param_ranges['rsi_mult'] = [round(rsi_mult + i * 0.1, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('rsi_mult')

                    if optimize_lookback:
                        param_ranges['lookback'] = range(max(1, lookback - 2), lookback + 3)
                        base_strategy_params.pop('lookback')

                    if optimize_sltp and use_fixed_sltp:  # Only optimize SLTP if fixed SLTP is enabled
                        param_ranges['sltp'] = [round(sltp + i * 0.5, 1) for i in range(-5, 6)]
                        base_strategy_params.pop('sltp')







                elif strategy == 'RSI and Bollinger':

                    # Get parameter values

                    stoch_rsi_period = rsibb_params[1].value

                    stoch_k_period = rsibb_params[2].value

                    stoch_d_period = rsibb_params[3].value

                    bb_period = rsibb_params[5].value

                    bb_std = rsibb_params[7].value

                    recent_candles = rsibb_params[9].value

                    lookback_range = rsibb_params[11].value

                    sma_filter_period = rsibb_params[13].value  # ✅ Now correctly referenced

                    # Base parameters

                    base_strategy_params = {

                        'stoch_rsi_period': stoch_rsi_period,

                        'stoch_k_period': stoch_k_period,

                        'stoch_d_period': stoch_d_period,

                        'bb_period': bb_period,

                        'bb_std_dev': bb_std,

                        'recent_candles': recent_candles,

                        'lookback_range': lookback_range,

                        'sma_filter_period': sma_filter_period  # ✅ SMA filter added

                    }

                    # Optimization ranges if selected

                    if rsibb_params[0].value:  # Optimize Stoch RSI Period

                        param_ranges['stoch_rsi_period'] = range(max(5, stoch_rsi_period - 5), stoch_rsi_period + 6)

                        base_strategy_params.pop('stoch_rsi_period')

                    if rsibb_params[4].value:  # Optimize BB Period

                        param_ranges['bb_period'] = range(max(10, bb_period - 5), bb_period + 6)

                        base_strategy_params.pop('bb_period')

                    if rsibb_params[6].value:  # Optimize BB Std Dev

                        param_ranges['bb_std_dev'] = [round(bb_std + i * 0.1, 1) for i in range(-5, 6)]

                        base_strategy_params.pop('bb_std_dev')

                    if rsibb_params[8].value:  # Optimize Recent Candles

                        param_ranges['recent_candles'] = range(max(1, recent_candles - 2), recent_candles + 3)

                        base_strategy_params.pop('recent_candles')

                    if rsibb_params[10].value:  # Optimize Lookback Range

                        param_ranges['lookback_range'] = range(max(1, lookback_range - 2), lookback_range + 3)

                        base_strategy_params.pop('lookback_range')

                    if rsibb_params[12].value:  # Optimize SMA Filter Period

                        param_ranges['sma_filter_period'] = range(max(20, sma_filter_period - 10),
                                                                  sma_filter_period + 11)

                        base_strategy_params.pop('sma_filter_period')
                elif strategy == 'Turtle Trading':
                    # Get parameter values
                    breakout_period = turtle_params[1].value
                    atr_period = turtle_params[3].value
                    atr_multiplier = turtle_params[5].value
                    profit_target = turtle_params[7].value

                    # Base parameters
                    base_strategy_params = {
                        'period': breakout_period,
                        'atr_period': atr_period,
                        'atr_multiplier': atr_multiplier,
                        'profit_target': profit_target
                    }

                    # Optimization ranges if selected
                    if turtle_params[0].value:  # Optimize Period
                        param_ranges['period'] = range(max(20, breakout_period - 10),
                                                       breakout_period + 11)
                        base_strategy_params.pop('period')

                    if turtle_params[2].value:  # Optimize ATR Period
                        param_ranges['atr_period'] = range(max(7, atr_period - 3),
                                                           atr_period + 4)
                        base_strategy_params.pop('atr_period')

                    if turtle_params[4].value:  # Optimize ATR Multiplier
                        param_ranges['atr_multiplier'] = [round(atr_multiplier + i * 0.1, 1)
                                                          for i in range(-5, 6)]
                        base_strategy_params.pop('atr_multiplier')

                    if turtle_params[6].value:  # Optimize Profit Target
                        param_ranges['profit_target'] = [round(profit_target + i * 0.1, 1)
                                                         for i in range(-5, 6)]
                        base_strategy_params.pop('profit_target')

                elif strategy == 'EMA Cross with Volume':
                    optimize_fast_ema = emacross_params[0].value
                    fast_ema_value = emacross_params[1].value
                    optimize_slow_ema = emacross_params[2].value
                    slow_ema_value = emacross_params[3].value
                    optimize_volume_ma = emacross_params[4].value
                    volume_ma_value = emacross_params[5].value
                    optimize_volume_mult = emacross_params[6].value
                    volume_mult_value = emacross_params[7].value

                    if optimize_fast_ema:
                        param_ranges['n1'] = range(max(1, fast_ema_value - 5), fast_ema_value + 6)
                    else:
                        base_strategy_params['n1'] = fast_ema_value

                    if optimize_slow_ema:
                        param_ranges['n2'] = range(max(1, slow_ema_value - 5), slow_ema_value + 6)
                    else:
                        base_strategy_params['n2'] = slow_ema_value

                    if optimize_volume_ma:
                        param_ranges['volume_ma'] = range(max(1, volume_ma_value - 5), volume_ma_value + 6)
                    else:
                        base_strategy_params['volume_ma'] = volume_ma_value

                    if optimize_volume_mult:
                        param_ranges['volume_mult'] = [round(volume_mult_value + i * 0.1, 2) for i in range(-5, 6)]
                    else:
                        base_strategy_params['volume_mult'] = volume_mult_value

                elif strategy == 'Bollinger Bands':
                    optimize_bb_period = bb_params[0].value
                    bb_period_value = bb_params[1].value
                    optimize_bb_std = bb_params[2].value
                    bb_std_value = bb_params[3].value
                    optimize_rsi_period = bb_params[4].value
                    rsi_period_value = bb_params[5].value
                    optimize_rsi_thresh = bb_params[6].value
                    rsi_thresh_value = bb_params[7].value

                    if optimize_bb_period:
                        param_ranges['bb_period'] = range(max(1, bb_period_value - 5), bb_period_value + 6)
                    else:
                        base_strategy_params['bb_period'] = bb_period_value

                    if optimize_bb_std:
                        param_ranges['bb_std'] = [round(bb_std_value + i * 0.1, 2) for i in range(-5, 6)]
                    else:
                        base_strategy_params['bb_std'] = bb_std_value

                    if optimize_rsi_period:
                        param_ranges['rsi_period'] = range(max(1, rsi_period_value - 3), rsi_period_value + 4)
                    else:
                        base_strategy_params['rsi_period'] = rsi_period_value

                    if optimize_rsi_thresh:
                        param_ranges['rsi_threshold'] = range(max(1, rsi_thresh_value - 5), rsi_thresh_value + 6)
                    else:
                        base_strategy_params['rsi_threshold'] = rsi_thresh_value

                elif strategy == 'MACD with Trend':
                    optimize_macd_fast = macd_params[0].value
                    macd_fast_value = macd_params[1].value
                    optimize_macd_slow = macd_params[2].value
                    macd_slow_value = macd_params[3].value
                    optimize_macd_signal = macd_params[4].value
                    macd_signal_value = macd_params[5].value
                    optimize_trend_ma = macd_params[6].value
                    trend_ma_value = macd_params[7].value
                    optimize_trend_strength = macd_params[8].value
                    trend_strength_value = macd_params[9].value

                    if optimize_macd_fast:
                        param_ranges['macd_fast'] = range(max(1, macd_fast_value - 5), macd_fast_value + 6)
                    else:
                        base_strategy_params['macd_fast'] = macd_fast_value

                    if optimize_macd_slow:
                        param_ranges['macd_slow'] = range(max(1, macd_slow_value - 5), macd_slow_value + 6)
                    else:
                        base_strategy_params['macd_slow'] = macd_slow_value

                    if optimize_macd_signal:
                        param_ranges['macd_signal'] = range(max(1, macd_signal_value - 5), macd_signal_value + 6)
                    else:
                        base_strategy_params['macd_signal'] = macd_signal_value

                    if optimize_trend_ma:
                        param_ranges['trend_ma'] = range(max(1, trend_ma_value - 50), trend_ma_value + 51)
                    else:
                        base_strategy_params['trend_ma'] = trend_ma_value

                    if optimize_trend_strength:
                        param_ranges['min_trend_strength'] = [round(trend_strength_value / 100 + i * 0.001, 3) for i in
                                                              range(-5, 6)]
                    else:
                        base_strategy_params['min_trend_strength'] = round(trend_strength_value / 100, 3)

                # Handle new strategies similarly
                elif strategy == 'RSI Divergence':
                    optimize_rsi_period = rsidiv_params[0].value
                    rsi_period_value = rsidiv_params[1].value
                    optimize_min_lookback = rsidiv_params[2].value
                    min_lookback_value = rsidiv_params[3].value
                    optimize_max_lookback = rsidiv_params[4].value
                    max_lookback_value = rsidiv_params[5].value

                    if optimize_rsi_period:
                        param_ranges['rsi_period'] = range(
                            max(5, rsi_period_value - 5), rsi_period_value + 6
                        )
                    else:
                        base_strategy_params['rsi_period'] = rsi_period_value

                    if optimize_min_lookback:
                        param_ranges['min_lookback_range'] = range(
                            max(1, min_lookback_value - 2), min_lookback_value + 3
                        )
                    else:
                        base_strategy_params['min_lookback_range'] = min_lookback_value

                    if optimize_max_lookback:
                        param_ranges['max_lookback_range'] = range(
                            max(min_lookback_value, max_lookback_value - 2), max_lookback_value + 3
                        )
                    else:
                        base_strategy_params['max_lookback_range'] = max_lookback_value

                elif strategy == 'Donchian Channel Breakout':
                    channel_period_value = donchian_params[1].value

                    base_strategy_params['dc_period'] = channel_period_value

                elif strategy == 'ATR Chandelier Exit':
                    atr_period_value = atrchandelier_params[1].value
                    atr_multiplier_value = atrchandelier_params[3].value

                    base_strategy_params['atr_period'] = atr_period_value
                    base_strategy_params['atr_multiplier'] = atr_multiplier_value

                elif strategy == 'Keltner Channel':
                    kc_period_value = keltner_params[1].value
                    kc_multiplier_value = keltner_params[3].value

                    base_strategy_params['kc_period'] = kc_period_value
                    base_strategy_params['kc_multiplier'] = kc_multiplier_value

                elif strategy == 'Channel Breakout':
                    channel_breakout_period_value = channelbreakout_params[1].value

                    base_strategy_params['channel_period'] = channel_breakout_period_value

                elif strategy == 'MACD Divergence':
                    optimize_macd_fast = macddiv_params[0].value
                    macd_fast_value = macddiv_params[1].value
                    optimize_macd_slow = macddiv_params[2].value
                    macd_slow_value = macddiv_params[3].value
                    optimize_macd_signal = macddiv_params[4].value
                    macd_signal_value = macddiv_params[5].value
                    optimize_divergence_thresh = macddiv_params[6].value
                    divergence_thresh_value = macddiv_params[7].value

                    if optimize_macd_fast:
                        param_ranges['macd_fast'] = range(max(1, macd_fast_value - 5), macd_fast_value + 6)
                    else:
                        base_strategy_params['macd_fast'] = macd_fast_value

                    if optimize_macd_slow:
                        param_ranges['macd_slow'] = range(max(1, macd_slow_value - 5), macd_slow_value + 6)
                    else:
                        base_strategy_params['macd_slow'] = macd_slow_value

                    if optimize_macd_signal:
                        param_ranges['macd_signal'] = range(max(1, macd_signal_value - 5), macd_signal_value + 6)
                    else:
                        base_strategy_params['macd_signal'] = macd_signal_value

                    if optimize_divergence_thresh:
                        param_ranges['divergence_threshold'] = [round(divergence_thresh_value + i * 0.001, 3) for i in
                                                                range(-5, 6)]
                    else:
                        base_strategy_params['divergence_threshold'] = round(divergence_thresh_value, 3)

                elif strategy == 'Linear Regression':
                    optimize_lr_window = linearreg_params[0].value
                    lr_window_value = linearreg_params[1].value

                    if optimize_lr_window:
                        param_ranges['lr_window'] = range(max(5, lr_window_value - 5), lr_window_value + 6)
                    else:
                        base_strategy_params['lr_window'] = lr_window_value





                elif strategy == 'Conor Switch':

                    base_strategy_params = {

                        'tenkan_period': conor_switch_params[1].value,

                        'stochastic_k_period': 14,  # Required parameter

                        'stochastic_d_period': 3,  # Required parameter

                        'rsi_period': conor_switch_params[3].value,

                        'overbought_threshold': conor_switch_params[5].value,

                        'oversold_threshold': conor_switch_params[7].value,

                        'divergence_lookback': conor_switch_params[9].value,

                        'divergence_threshold': conor_switch_params[11].value

                    }

                    if conor_switch_params[0].value:  # Optimize Tenkan Period

                        param_ranges['tenkan_period'] = range(max(5, conor_switch_params[1].value - 5),

                                                              conor_switch_params[1].value + 6)

                        base_strategy_params.pop('tenkan_period')

                    if conor_switch_params[2].value:  # Optimize RSI Period

                        param_ranges['rsi_period'] = range(max(7, conor_switch_params[3].value - 3),

                                                           conor_switch_params[3].value + 4)

                        base_strategy_params.pop('rsi_period')

                    if conor_switch_params[4].value:  # Optimize Overbought Threshold

                        param_ranges['overbought_threshold'] = range(conor_switch_params[5].value - 5,

                                                                     conor_switch_params[5].value + 6)

                        base_strategy_params.pop('overbought_threshold')

                    if conor_switch_params[6].value:  # Optimize Oversold Threshold

                        param_ranges['oversold_threshold'] = range(conor_switch_params[7].value - 5,

                                                                   conor_switch_params[7].value + 6)

                        base_strategy_params.pop('oversold_threshold')

                    if conor_switch_params[8].value:  # Optimize Divergence Lookback

                        param_ranges['divergence_lookback'] = range(max(3, conor_switch_params[9].value - 2),

                                                                    conor_switch_params[9].value + 3)

                        base_strategy_params.pop('divergence_lookback')

                    if conor_switch_params[10].value:  # Optimize Divergence Threshold

                        param_ranges['divergence_threshold'] = [round(conor_switch_params[11].value + i * 0.1, 1)

                                                                for i in range(-5, 6)]

                        base_strategy_params.pop('divergence_threshold')

                # Add base strategy parameters from risk management settings
                use_sl, use_ts, use_tp, exit_reversal, sl_pct, ts_long_pct, ts_short_pct, tp_pct = [param.value for param in base_params]
                base_strategy_params.update({
                    'use_sl': use_sl,
                    'use_ts': use_ts,
                    'use_tp': use_tp,
                    'exit_on_reversal': exit_reversal,
                    'sl_pct': sl_pct / 100,
                    'ts_long_pct': ts_long_pct / 100,
                    'ts_short_pct': ts_short_pct / 100,
                    'tp_pct': tp_pct / 100
                })

                optimizer = StrategyOptimizer(strategy_class, data)

                print("\nRunning grid optimization...")
                # Run grid optimization
                best_result, all_results = optimizer.optimize_grid(param_ranges, base_strategy_params, cash)

                if best_result is None:
                    error_message.object = "⚠️ Optimization failed to find valid results"
                    optimize_button.loading = False
                    return

                print("\nOptimization completed, preparing metrics...")
                # Create detailed metrics dictionary for best result
                metrics = best_result['Metrics']
                final_equity = metrics.get('Equity Final [$]', metrics.get('Equity Final', 0))
                net_profit = final_equity - cash

                detailed_metrics = {
                    'Metric': [
                        'Net Profit',
                        'Return [%]',
                        'Return (Ann.) [%]',
                        'Max Drawdown [%]',
                        'Sharpe Ratio',
                        'Sortino Ratio',
                        'Calmar Ratio',
                        'Max Drawdown Duration',
                        'Total Trades',
                        'Win Rate [%]',
                        'Best Trade [%]',
                        'Worst Trade [%]',
                        'Avg Trade [%]',
                        'Profit Factor',
                        'Expectancy [%]',
                        'SQN',
                        'Avg Trade Duration',
                        'Max Trade Duration',
                        'Equity Final [$]',
                        'Equity Peak [$]'
                    ],
                    'Value': [
                        f"${net_profit:.2f}",
                        f"{metrics.get('Return [%]', 0):.2f}%",
                        f"{metrics.get('Return (Ann.) [%]', 0):.2f}%",
                        f"{metrics.get('Max. Drawdown [%]', 0):.2f}%",
                        f"{metrics.get('Sharpe Ratio', 0):.2f}",
                        f"{metrics.get('Sortino Ratio', 0):.2f}",
                        f"{metrics.get('Calmar Ratio', 0):.2f}",
                        str(metrics.get('Max Drawdown Duration', 'N/A')),
                        metrics.get('# Trades', 0),
                        f"{metrics.get('Win Rate [%]', 0):.2f}%",
                        f"{metrics.get('Best Trade [%]', 0):.2f}%",
                        f"{metrics.get('Worst Trade [%]', 0):.2f}%",
                        f"{metrics.get('Avg. Trade [%]', 0):.2f}%",
                        f"{metrics.get('Profit Factor', 0):.2f}",
                        f"{metrics.get('Expectancy [%]', 0):.2f}%",
                        f"{metrics.get('SQN', 0):.2f}",
                        str(metrics.get('Avg. Trade Duration', 'N/A')),
                        str(metrics.get('Max. Trade Duration', 'N/A')),
                        f"${final_equity:.2f}",
                        f"${metrics.get('Equity Peak [$]', 0):.2f}"
                    ]
                }

                # Update UI components with results
                print("\nUpdating UI components...")
                metrics_table.value = pd.DataFrame(detailed_metrics)
                metrics_table.value = metrics_table.value[['Metric', 'Value']]

                # Display best optimization parameters
                optimized_params_df = pd.DataFrame([best_result['Params']])
                optimization_results_table.value = optimized_params_df.T.reset_index()
                optimization_results_table.value.columns = ['Parameter', 'Value']

                # Display all optimization results if needed
                all_results_df = pd.DataFrame([{
                    **res['Params'],
                    'Net Profit': res['Net Profit'],
                    'Return [%]': res['Return [%]'],
                    'Sharpe Ratio': res['Sharpe Ratio'],
                    'Max Drawdown [%]': res['Max Drawdown [%]'],
                    'Trades': res['Trades']
                } for res in all_results])

                all_optimization_results_table.value = all_results_df

                # Run backtest again with best parameters to get strategy instance and stats for plotting
                print("\nRunning final backtest with best parameters...")
                bt = Backtest(data, strategy_class, cash=cash, commission=commission_bps.value / 10000)
                stats = bt.run(**best_result['Params'])

                # Compute monthly and yearly analysis
                equity_curve = stats['_equity_curve']
                trades_data = stats['_trades']
                monthly_analysis_df = self.compute_monthly_analysis(trades_data, equity_curve)
                yearly_analysis_df = self.compute_yearly_analysis(trades_data, equity_curve)

                # Update the monthly and yearly analysis tables
                monthly_analysis_table.value = monthly_analysis_df
                yearly_analysis_table.value = yearly_analysis_df

                print("\nCreating plot...")
                fig = self.create_plot(data, bt, stats)
                plot_panel.object = fig

                print("\nOptimization process completed successfully.")

            except Exception as e:
                error_msg = f"⚠️ Error during optimization: {str(e)}"
                print(error_msg)
                print("\nFull traceback:")
                import traceback
                traceback.print_exc()
                error_message.object = error_msg
            finally:
                optimize_button.loading = False

        def run_backtest(event):
            backtest_button.loading = True
            error_message.object = ''
            try:
                # Fetch data
                print("\nStarting backtest process...")
                data = self.fetch_data(symbol.value, interval_select.value, start_date.value, end_date.value)
                if data is None or data.empty:
                    error_message.object = f"⚠️ No data found for {symbol.value}"
                    logger.warning(f"No data found for {symbol.value}")
                    return

                logger.debug("Data fetched successfully.")
                strategy_class = self.strategies[strategy_select.value]
                cash = initial_cash.value

                # Get base strategy parameters
                base_strategy_params = {}

                strategy = strategy_select.value
                if strategy == 'Moving Average Crossover':
                    fast_ma_value = ma_params[1].value
                    slow_ma_value = ma_params[3].value

                    base_strategy_params['n1'] = fast_ma_value
                    base_strategy_params['n2'] = slow_ma_value
                elif strategy == 'RSI and Bollinger':
                    # Get parameters from widgets
                    rsi_period = rsibb_params[1].value
                    bb_period = rsibb_params[3].value
                    bb_std = rsibb_params[5].value
                    recent_candles = rsibb_params[7].value
                    lookback_range = rsibb_params[9].value
                    ma_period = rsibb_params[11].value  # Get MA period value

                    # Set base parameters
                    base_strategy_params.update({
                        'rsi_period': rsi_period,
                        'bb_period': bb_period,
                        'bb_std_dev': bb_std,
                        'recent_candles': recent_candles,
                        'lookback_range': lookback_range,
                        'ma_period': ma_period  # Include MA parameter
                    })

                elif strategy == 'Adaptive Supertrend':
                    # Get parameter values
                    choppiness_period = adaptive_supertrend_params[1].value
                    rsi_bb_timeframe = adaptive_supertrend_params[16].value  # Add this line

                    choppiness_threshold = adaptive_supertrend_params[3].value
                    atr_period = adaptive_supertrend_params[5].value
                    atr_multiplier = adaptive_supertrend_params[7].value
                    rsi_period = adaptive_supertrend_params[9].value
                    bb_period = adaptive_supertrend_params[11].value
                    bb_std = adaptive_supertrend_params[13].value
                    lookback_range = adaptive_supertrend_params[15].value

                    # Set base parameters
                    base_strategy_params = {
                        'choppiness_period': choppiness_period,
                        'choppiness_threshold': choppiness_threshold,
                        'rsi_bb_timeframe': rsi_bb_timeframe,  # Add this line
                        'atr_period': atr_period,
                        'atr_multiplier': atr_multiplier,
                        'rsi_period': rsi_period,
                        'bb_period': bb_period,
                        'bb_std_dev': bb_std,
                        'lookback_range': lookback_range
                    }

                    print("\nRunning Backtest for Adaptive Supertrend Strategy")
                    print("Parameters:")
                    print(f"Choppiness Period: {choppiness_period}, Threshold: {choppiness_threshold}")
                    print(f"ATR Period: {atr_period}, Multiplier: {atr_multiplier}")
                    print(f"RSI/BB Timeframe: {rsi_bb_timeframe}")  # Add this line to print

                    print(f"RSI Period: {rsi_period}, BB Period: {bb_period}")
                    print(f"BB Std Dev: {bb_std}, Lookback Range: {lookback_range}")

                elif strategy == 'Triple EMA':
                    # Get parameter values from widgets
                    use_strategy_exits = triple_ema_params[0].value
                    split_position = triple_ema_params[1].value
                    ema_fast = triple_ema_params[3].value
                    ema_medium = triple_ema_params[5].value
                    ema_slow = triple_ema_params[7].value
                    adx_period = triple_ema_params[9].value
                    adx_threshold = triple_ema_params[11].value
                    atr_period = triple_ema_params[13].value
                    atr_multiplier = triple_ema_params[15].value
                    trend_lookback = triple_ema_params[17].value
                    volume_threshold = triple_ema_params[19].value

                    # Set base parameters for backtest
                    base_strategy_params = {
                        'use_strategy_specific_exits': use_strategy_exits,
                        'split_position': split_position,
                        'ema_fast': ema_fast,
                        'ema_medium': ema_medium,
                        'ema_slow': ema_slow,
                        'adx_period': adx_period,
                        'adx_threshold': adx_threshold,
                        'atr_period': atr_period,
                        'atr_multiplier': atr_multiplier,
                        'trend_lookback': trend_lookback,
                        'min_volume_threshold': volume_threshold
                    }
                elif strategy == 'Liquidation Strategy':
                    liq_threshold = liquidation_params[1].value
                    liq_window = liquidation_params[3].value
                    liq_spike_mult = liquidation_params[5].value
                    volume_mult = liquidation_params[7].value
                    price_rebound = liquidation_params[9].value
                    rsi_period = liquidation_params[11].value
                    rsi_oversold = liquidation_params[13].value

                    base_strategy_params = {
                        'liq_threshold': liq_threshold,
                        'liq_window': liq_window,
                        'liq_spike_mult': liq_spike_mult,
                        'volume_mult': volume_mult,
                        'price_rebound': price_rebound / 100,
                        'rsi_period': rsi_period,
                        'rsi_oversold': rsi_oversold
                    }
                elif strategy == 'RSI Trend Reversal':
                    base_strategy_params = {
                        'rsi_length': rsi_trend_reversal_params[1].value,
                        'rsi_mult': rsi_trend_reversal_params[3].value,
                        'lookback': rsi_trend_reversal_params[5].value,
                        'use_fixed_sltp': rsi_trend_reversal_params[6].value,  # Add the new parameter
                        'sltp': rsi_trend_reversal_params[8].value
                    }
                elif strategy == 'ATR Bollinger Bands':
                    # Get widget values by name
                    bb_widgets = {w.name: w for w in bb_atr_params}

                    base_strategy_params = {
                        'bblength': bb_widgets['BB Length'].value,
                        'bbstdev': bb_widgets['BB StdDev'].value,
                        'lena': bb_widgets['ATR Length'].value,
                        'mult': bb_widgets['ATR Multiplier'].value,
                        'use_filter': bb_widgets['Use Filter'].value,
                        'use_atr_bb': bb_widgets['Use ATR BB'].value
                    }
                elif strategy == 'Enhanced RSI Bollinger':
                    # Get parameter values (same as optimization section)
                    rsi_period = enhanced_rsibb_params[1].value
                    rsi_thresholds = enhanced_rsibb_params[3].value
                    counter_trend_rsi = enhanced_rsibb_params[5].value
                    bb_period = enhanced_rsibb_params[7].value
                    bb_std = enhanced_rsibb_params[9].value
                    ma_period = enhanced_rsibb_params[11].value
                    rvol_period = enhanced_rsibb_params[13].value
                    rvol_threshold = enhanced_rsibb_params[15].value
                    recent_candles = enhanced_rsibb_params[17].value
                    lookback_range = enhanced_rsibb_params[19].value

                    base_strategy_params.update({
                        'rsi_period': rsi_period,
                        'rsi_oversold': rsi_thresholds[0],
                        'rsi_overbought': rsi_thresholds[1],
                        'counter_trend_rsi': counter_trend_rsi,
                        'bb_period': bb_period,
                        'bb_std_dev': bb_std,
                        'ma_period': ma_period,
                        'rvol_period': rvol_period,
                        'rvol_threshold': rvol_threshold,
                        'recent_candles': recent_candles,
                        'lookback_range': lookback_range
                    })
                if strategy == 'Adaptive RSI Volume':
                    # Get parameter values
                    base_strategy_params = {
                        'rsi_length': adaptive_rsi_params[1].value,
                        'rsi_mult': adaptive_rsi_params[3].value,
                        'atr_period': adaptive_rsi_params[5].value,
                        'volatility_mult': adaptive_rsi_params[7].value,
                        'trend_ema': adaptive_rsi_params[9].value,
                        'volume_ma_period': adaptive_rsi_params[11].value,
                        'min_volume_mult': adaptive_rsi_params[13].value,
                        'rsi_oversold': adaptive_rsi_params[15].value[0],
                        'rsi_overbought': adaptive_rsi_params[15].value[1],
                        'min_swing_pct': adaptive_rsi_params[17].value,
                        'max_risk_per_trade': adaptive_rsi_params[19].value / 100
                    }
                elif strategy == 'SR VWAP Strategy':

                    # Get parameter values

                    pivot_length = sr_vwap_params[1].value

                    sr_strength = sr_vwap_params[3].value

                    atr_period = sr_vwap_params[5].value

                    zone_size = sr_vwap_params[7].value

                    min_bars = sr_vwap_params[9].value

                    vwap_length = sr_vwap_params[11].value

                    trend_strength = sr_vwap_params[13].value

                    base_strategy_params.update({

                        'pivot_length': pivot_length,

                        'sr_strength': sr_strength,

                        'atr_period': atr_period,

                        'zone_size_atr': zone_size,

                        'min_bars_between_signals': min_bars,

                        'vwap_ma_length': vwap_length,

                        'min_vwap_trend_strength': trend_strength

                    })

                elif strategy == 'Ichimoku TK Cross':
                    # Get parameter values
                    conversion_period = ichimoku_tk_params[1].value
                    base_period = ichimoku_tk_params[3].value
                    lagging_span2_period = ichimoku_tk_params[5].value
                    displacement = ichimoku_tk_params[7].value
                    adx_period = ichimoku_tk_params[9].value
                    adx_threshold = ichimoku_tk_params[11].value
                    rsi_period = ichimoku_tk_params[13].value
                    rsi_bounds = ichimoku_tk_params[15].value
                    use_cloud = ichimoku_tk_params[16].value

                    base_strategy_params.update({
                        'conversion_periods': conversion_period,
                        'base_periods': base_period,
                        'lagging_span2_periods': lagging_span2_period,
                        'displacement': displacement,
                        'adx_period': adx_period,
                        'adx_threshold': adx_threshold,
                        'rsi_period': rsi_period,
                        'rsi_oversold': rsi_bounds[0],
                        'rsi_overbought': rsi_bounds[1],
                        'use_cloud_filter': use_cloud
                    })
                elif strategy == 'ATR Vortex':
                    # Get parameters using widget names
                    first_atr_length = next(w for w in atr_vortex_params if w.name == 'First ATR Length').value
                    first_atr_mult = next(w for w in atr_vortex_params if w.name == 'First ATR Multiplier').value
                    second_atr_length = next(w for w in atr_vortex_params if w.name == 'Second ATR Length').value
                    second_atr_mult = next(w for w in atr_vortex_params if w.name == 'Second ATR Multiplier').value
                    vi_period = next(w for w in atr_vortex_params if w.name == 'VI Period').value
                    apply_tema = next(w for w in atr_vortex_params if w.name == 'Apply TEMA').value
                    fixed_targets = next(w for w in atr_vortex_params if w.name == 'Fixed Targets').value
                    repaints = next(w for w in atr_vortex_params if w.name == 'Repaints').value
                    use_strategy_specific_exits = next(
                        w for w in atr_vortex_params if w.name == 'Use Strategy Specific Exits').value
                    tracking_sl_length = next(w for w in atr_vortex_params if w.name == 'Tracking SL Length').value

                    # Set base parameters
                    base_strategy_params.update({
                        'first_atr_length': first_atr_length,
                        'first_atr_mult': first_atr_mult,
                        'second_atr_length': second_atr_length,
                        'second_atr_mult': second_atr_mult,
                        'vi_period': vi_period,
                        'apply_tema': apply_tema,
                        'fixed_targets': fixed_targets,
                        'repaints': repaints,
                        'use_strategy_specific_exits': use_strategy_specific_exits,
                        'tracking_sl_length': tracking_sl_length
                    })
                elif strategy == 'Ichimoku EMA Cross':
                    # Get parameter values
                    conversion_period = ichimoku_ema_params[1].value
                    base_period = ichimoku_ema_params[3].value
                    lagging_span2_period = ichimoku_ema_params[5].value
                    displacement = ichimoku_ema_params[7].value
                    ema_length = ichimoku_ema_params[9].value

                    base_strategy_params.update({
                        'conversion_periods': conversion_period,
                        'base_periods': base_period,
                        'lagging_span2_periods': lagging_span2_period,
                        'displacement': displacement,
                        'ema_length': ema_length
                    })
                elif strategy == 'Multi-Indicator Momentum':
                    try:
                        # Get all parameter values from widgets
                        sma_period = next(w for w in multi_momentum_params if w.name == 'SMA Period').value
                        macd_fast = next(w for w in multi_momentum_params if w.name == 'MACD Fast Period').value
                        macd_slow = next(w for w in multi_momentum_params if w.name == 'MACD Slow Period').value
                        macd_signal = next(w for w in multi_momentum_params if w.name == 'MACD Signal Period').value
                        rsi_period = next(w for w in multi_momentum_params if w.name == 'RSI Period').value
                        rsi_long_bounds = next(
                            w for w in multi_momentum_params if w.name == 'RSI Long Entry Bounds').value
                        rsi_short_bounds = next(
                            w for w in multi_momentum_params if w.name == 'RSI Short Entry Bounds').value
                        rsi_exit_bounds = next(w for w in multi_momentum_params if w.name == 'RSI Exit Bounds').value
                        bb_period = next(w for w in multi_momentum_params if w.name == 'BB Period').value
                        bb_std = next(w for w in multi_momentum_params if w.name == 'BB STD').value
                        atr_period = next(w for w in multi_momentum_params if w.name == 'ATR Period').value
                        atr_multiplier = next(w for w in multi_momentum_params if w.name == 'ATR Stop Multiplier').value
                        max_drawdown = next(w for w in multi_momentum_params if w.name == 'Max Drawdown %').value

                        # Set strategy parameters
                        base_strategy_params.update({
                            'sma_period': sma_period,
                            'macd_fast': macd_fast,
                            'macd_slow': macd_slow,
                            'macd_signal': macd_signal,
                            'rsi_period': rsi_period,
                            'rsi_lower': rsi_long_bounds[0],
                            'rsi_upper': rsi_long_bounds[1],
                            'rsi_short_lower': rsi_short_bounds[0],
                            'rsi_short_upper': rsi_short_bounds[1],
                            'rsi_exit_lower': rsi_exit_bounds[0],
                            'rsi_exit_upper': rsi_exit_bounds[1],
                            'bb_period': bb_period,
                            'bb_std': bb_std,
                            'atr_period': atr_period,
                            'atr_stop_multiplier': atr_multiplier,
                            'max_drawdown_pct': max_drawdown
                        })

                        print("Multi-Indicator Momentum Strategy Parameters:", base_strategy_params)

                    except Exception as e:
                        error_msg = f"Error setting Multi-Indicator Momentum Strategy parameters: {str(e)}"
                        print(error_msg)
                        print("\nFull traceback:")
                        import traceback
                        traceback.print_exc()
                        error_message.object = error_msg
                elif strategy == 'RSI':
                    rsi_period_value = rsi_params[1].value
                    overbought_value = rsi_params[3].value
                    oversold_value = rsi_params[5].value

                    base_strategy_params['rsi_period'] = rsi_period_value
                    base_strategy_params['rsi_overbought'] = overbought_value
                    base_strategy_params['rsi_oversold'] = oversold_value
                elif strategy == 'RSI and Bollinger':
                    # Get parameters from widgets
                    rsi_period = rsibb_params[1].value
                    bb_period = rsibb_params[3].value
                    bb_std = rsibb_params[5].value
                    recent_candles = rsibb_params[7].value
                    lookback_range = rsibb_params[9].value  # Get the Lookback Range value

                    # Set base parameters
                    base_strategy_params.update({
                        'rsi_period': rsi_period,
                        'bb_period': bb_period,
                        'bb_std_dev': bb_std,
                        'recent_candles': recent_candles,
                        'lookback_range': lookback_range  # Include the new parameter
                    })
                elif strategy == 'EMA Cross with Volume':
                    fast_ema_value = emacross_params[1].value
                    slow_ema_value = emacross_params[3].value
                    volume_ma_value = emacross_params[5].value
                    volume_mult_value = emacross_params[7].value

                    base_strategy_params['n1'] = fast_ema_value
                    base_strategy_params['n2'] = slow_ema_value
                    base_strategy_params['volume_ma'] = volume_ma_value
                    base_strategy_params['volume_mult'] = volume_mult_value

                elif strategy == 'Bollinger Bands':
                    bb_period_value = bb_params[1].value
                    bb_std_value = bb_params[3].value
                    rsi_period_value = bb_params[5].value
                    rsi_thresh_value = bb_params[7].value

                    base_strategy_params['bb_period'] = bb_period_value
                    base_strategy_params['bb_std'] = bb_std_value
                    base_strategy_params['rsi_period'] = rsi_period_value
                    base_strategy_params['rsi_threshold'] = rsi_thresh_value

                elif strategy == 'MACD with Trend':
                    macd_fast_value = macd_params[1].value
                    macd_slow_value = macd_params[3].value
                    macd_signal_value = macd_params[5].value
                    trend_ma_value = macd_params[7].value
                    trend_strength_value = macd_params[9].value

                    base_strategy_params['macd_fast'] = macd_fast_value
                    base_strategy_params['macd_slow'] = macd_slow_value
                    base_strategy_params['macd_signal'] = macd_signal_value
                    base_strategy_params['trend_ma'] = trend_ma_value
                    base_strategy_params['min_trend_strength'] = trend_strength_value / 100

                # Handle new strategies similarly
                elif strategy == 'RSI Divergence':
                    rsi_period_value = rsidiv_params[1].value
                    lookback_range_value = rsidiv_params[3].value

                    base_strategy_params['rsi_period'] = rsi_period_value
                    base_strategy_params['lookback_range'] = lookback_range_value

                    print("\nRunning Backtest for RSI Divergence Strategy")
                    print(f"RSI Period: {rsi_period_value}, Lookback Range: {lookback_range_value}")


                elif strategy == 'Donchian Channel Breakout':
                    channel_period_value = donchian_params[1].value

                    base_strategy_params['dc_period'] = channel_period_value

                elif strategy == 'ATR Chandelier Exit':
                    atr_period_value = atrchandelier_params[1].value
                    atr_multiplier_value = atrchandelier_params[3].value

                    base_strategy_params['atr_period'] = atr_period_value
                    base_strategy_params['atr_multiplier'] = atr_multiplier_value

                elif strategy == 'Keltner Channel':
                    kc_period_value = keltner_params[1].value
                    kc_multiplier_value = keltner_params[3].value

                    base_strategy_params['kc_period'] = kc_period_value
                    base_strategy_params['kc_multiplier'] = kc_multiplier_value
                elif strategy == 'Alligator':
                    # Get Alligator parameter values
                    jaw_period = alligator_params[1].value
                    teeth_period = alligator_params[3].value
                    lips_period = alligator_params[5].value
                    adx_period = alligator_params[7].value
                    adx_threshold = alligator_params[9].value
                    atr_period = alligator_params[11].value
                    atr_multiplier = alligator_params[13].value

                    # Set parameters
                    base_strategy_params = {
                        'jaw_period': jaw_period,
                        'teeth_period': teeth_period,
                        'lips_period': lips_period,
                        'adx_period': adx_period,
                        'adx_threshold': adx_threshold,
                        'atr_period': atr_period,
                        'atr_multiplier': atr_multiplier,
                        'jaw_offset': 8,  # Default offset values
                        'teeth_offset': 5,
                        'lips_offset': 3
                    }
                elif strategy == 'Channel Breakout':
                    channel_breakout_period_value = channelbreakout_params[1].value

                    base_strategy_params['channel_period'] = channel_breakout_period_value

                elif strategy == 'MACD Divergence':
                    macd_fast_value = macddiv_params[1].value
                    macd_slow_value = macddiv_params[3].value
                    macd_signal_value = macddiv_params[5].value
                    divergence_thresh_value = macddiv_params[7].value

                    base_strategy_params['macd_fast'] = macd_fast_value
                    base_strategy_params['macd_slow'] = macd_slow_value
                    base_strategy_params['macd_signal'] = macd_signal_value
                    base_strategy_params['divergence_threshold'] = divergence_thresh_value

                elif strategy == 'Linear Regression':
                    lr_window_value = linearreg_params[1].value

                    base_strategy_params['lr_window'] = lr_window_value


                elif strategy == 'Conor Switch':

                    tenkan_period_value = conor_switch_params[1].value

                    stoch_k_value = conor_switch_params[3].value

                    stoch_d_value = conor_switch_params[5].value

                    overbought_value = conor_switch_params[7].value

                    oversold_value = conor_switch_params[9].value

                    base_strategy_params['tenkan_period'] = tenkan_period_value

                    base_strategy_params['stochastic_k_period'] = stoch_k_value

                    base_strategy_params['stochastic_d_period'] = stoch_d_value

                    base_strategy_params['overbought_threshold'] = overbought_value

                    base_strategy_params['oversold_threshold'] = oversold_value

                elif strategy == 'Turtle Trading':

                    breakout_period = turtle_params[1].value
                    atr_period = turtle_params[3].value
                    atr_multiplier = turtle_params[5].value
                    profit_target = turtle_params[7].value

                    base_strategy_params['period'] = breakout_period
                    base_strategy_params['atr_period'] = atr_period
                    base_strategy_params['atr_multiplier'] = atr_multiplier
                    base_strategy_params['profit_target'] = profit_target
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                # Add base strategy parameters from risk management settings
                use_sl, use_ts, use_tp, exit_reversal, sl_pct, ts_long_pct, ts_short_pct, tp_pct = [param.value for param in base_params]
                base_strategy_params = {
                    'use_sl': use_sl,
                    'use_ts': use_ts,
                    'use_tp': use_tp,
                    'exit_on_reversal': exit_reversal,
                    'sl_pct': sl_pct / 100,
                    'ts_long_pct': ts_long_pct / 100,  # ✅ Corrected: Separate TS for longs
                    'ts_short_pct': ts_short_pct / 100,  # ✅ Corrected: Separate TS for shorts
                    'tp_pct': tp_pct / 100
                }

                # Run backtest
                bt = Backtest(data, strategy_class, cash=cash, commission=commission_bps.value / 10000)
                stats = bt.run(**base_strategy_params)

                # Print available keys in stats for debugging
                print("Available stats keys:", stats.keys())

                # Calculate net profit
                final_equity = stats.get('Equity Final [$]', stats.get('Equity Final', cash))
                net_profit = final_equity - cash

                # Create detailed metrics dictionary
                detailed_metrics = {
                    'Metric': [
                        'Net Profit',
                        'Return [%]',
                        'Return (Ann.) [%]',
                        'Max Drawdown [%]',
                        'Sharpe Ratio',
                        'Sortino Ratio',
                        'Calmar Ratio',
                        'Max Drawdown Duration',
                        'Total Trades',
                        'Win Rate [%]',
                        'Best Trade [%]',
                        'Worst Trade [%]',
                        'Avg Trade [%]',
                        'Profit Factor',
                        'Expectancy [%]',
                        'SQN',
                        'Avg Trade Duration',
                        'Max Trade Duration',
                        'Equity Final [$]',
                        'Equity Peak [$]'
                    ],
                    'Value': [
                        f"${net_profit:.2f}",
                        f"{stats.get('Return [%]', 0):.2f}%",
                        f"{stats.get('Return (Ann.) [%]', 0):.2f}%",
                        f"{stats.get('Max. Drawdown [%]', 0):.2f}%",
                        f"{stats.get('Sharpe Ratio', 0):.2f}",
                        f"{stats.get('Sortino Ratio', 0):.2f}",
                        f"{stats.get('Calmar Ratio', 0):.2f}",
                        str(stats.get('Max Drawdown Duration', 'N/A')),
                        stats.get('# Trades', 0),
                        f"{stats.get('Win Rate [%]', 0):.2f}%",
                        f"{stats.get('Best Trade [%]', 0):.2f}%",
                        f"{stats.get('Worst Trade [%]', 0):.2f}%",
                        f"{stats.get('Avg. Trade [%]', 0):.2f}%",
                        f"{stats.get('Profit Factor', 0):.2f}",
                        f"{stats.get('Expectancy [%]', 0):.2f}%",
                        f"{stats.get('SQN', 0):.2f}",
                        str(stats.get('Avg. Trade Duration', 'N/A')),
                        str(stats.get('Max. Trade Duration', 'N/A')),
                        f"${final_equity:.2f}",
                        f"${stats.get('Equity Peak [$]', stats.get('Equity Peak', 0)):.2f}"
                    ]
                }

                # Update UI components with results
                print("\nUpdating UI components...")
                metrics_table.value = pd.DataFrame(detailed_metrics)
                metrics_table.value = metrics_table.value[['Metric', 'Value']]

                # Update optimized parameters table
                optimized_params_df = pd.DataFrame([base_strategy_params])
                optimization_results_table.value = optimized_params_df.T.reset_index()
                optimization_results_table.value.columns = ['Parameter', 'Value']

                # Update trades table
                trades_table.value = stats['_trades']

                # Compute monthly and yearly analysis
                equity_curve = stats['_equity_curve']
                trades_data = stats['_trades']
                monthly_analysis_df = self.compute_monthly_analysis(trades_data, equity_curve)
                yearly_analysis_df = self.compute_yearly_analysis(trades_data, equity_curve)

                # Update the monthly and yearly analysis tables
                monthly_analysis_table.value = monthly_analysis_df
                yearly_analysis_table.value = yearly_analysis_df

                # Create plot
                fig = self.create_plot(data, bt, stats)
                plot_panel.object = fig

                print("\nBacktest process completed successfully.")

            except Exception as e:
                error_msg = f"⚠️ Error during backtest: {str(e)}"
                print(error_msg)
                print("\nFull traceback:")
                import traceback
                traceback.print_exc()
                error_message.object = error_msg
            finally:
                backtest_button.loading = False

        def run_walk_forward_analysis(event):
            walk_forward_button.loading = True
            error_message.object = ''
            try:
                # Placeholder for walk-forward analysis implementation
                error_message.object = "Walk-forward analysis is not implemented yet."
            except Exception as e:
                error_msg = f"⚠️ Error during walk-forward analysis: {str(e)}"
                print(error_msg)
                print("\nFull traceback:")
                import traceback
                traceback.print_exc()
                error_message.object = error_msg
            finally:
                walk_forward_button.loading = False

        def go_back(event):
            # Show main controls and hide back button
            controls.visible = True
            back_button.visible = False
            # Clear the plot
            plot_panel.object = None

        optimize_button.on_click(optimize_strategy)
        backtest_button.on_click(run_backtest)
        walk_forward_button.on_click(run_walk_forward_analysis)
        back_button.on_click(go_back)

        # Layout components
        # In the create_dashboard function, update the controls definition:
        controls = pn.Column(
            pn.pane.Markdown("<h2 style='font-size: 20pt;'>🚀 Crypto Backtesting Dashboard</h2>"),
            pn.WidgetBox(
                symbol,
                interval_select,
                pn.Row(start_date, end_date),
                initial_cash,
                strategy_select,
                commission_bps,
                # Strategy parameter widgets
                ma_params,
                ichimoku_ema_params,
                rsi_params,
                rsi_trend_reversal_params,
                rsibb_params,
                triple_ema_params,  # Make sure this is here
                emacross_params,
                adaptive_rsi_params,
                bb_params,
                atr_vortex_params,
                multi_momentum_params,
                macd_params,
                liquidation_params,
                sr_vwap_params,
                bb_atr_params,
                rsidiv_params,
                alligator_params,
                enhanced_rsibb_params,
                donchian_params,
                atrchandelier_params,
                adaptive_supertrend_params,
                keltner_params,
                channelbreakout_params,
                ichimoku_tk_params,
                macddiv_params,
                linearreg_params,
                conor_switch_params,
                turtle_params,
                # Risk management parameters
                base_params,
                # Buttons
                pn.Row(optimize_button, optimize_sl_tp_button, backtest_button, walk_forward_button),
                error_message,
                width=300
            )
        )

        controls.visible = True
        back_button.visible = False

        dashboard = pn.Column(
            pn.Row(
                pn.Column(
                    controls,
                    back_button
                ),
                pn.Spacer(width=50),
                pn.Column(
                    "📉 Trading Chart",
                    plot_panel,
                    sizing_mode='stretch_width'
                ),
            ),
            pn.Tabs(
                ('Performance Metrics', metrics_table),
                ('Parameters', optimization_results_table),
                ('Optimization Results', all_optimization_results_table),
                ('Trade Log', trades_table),
                ('Monthly Analysis', monthly_analysis_table),  # New Tab
                ('Yearly Analysis', yearly_analysis_table),    # New Tab
            ),
            sizing_mode='stretch_width'
        )

        return dashboard

    def run(self):
        dashboard = self.create_dashboard()
        dashboard.show()


if __name__ == "__main__":
    # Create the backtester instance
    backtester = CryptoBacktester()

    print("\nTesting Binance API Connection...")
    try:
        # Test basic connection
        result = backtester.client.ping()
        print("API Connection Test:", "Successful" if result == {} else "Failed")

        # Test data fetching
        print("\nFetching sample BTC data...")
        klines = backtester.client.get_historical_klines(
            symbol="BTCUSDT",
            interval=KLINE_INTERVAL_1DAY,
            limit=5  # Just get 5 candles
        )

        if klines:
            print(f"Successfully fetched {len(klines)} candles")
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            print("\nSample data:")
            print(df.head())
        else:
            print("No data received")

    except Exception as e:
        print(f"Error testing API: {str(e)}")

    backtester.run()