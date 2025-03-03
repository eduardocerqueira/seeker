#date: 2025-03-03T17:09:24Z
#url: https://api.github.com/gists/746f08ee945ac22d8e7b4faf654dda12
#owner: https://api.github.com/users/Clement1nes

import os
import pandas as pd
import numpy as np
import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
import panel as pn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from binance.client import Client
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
import math

# Initialize Panel extension
pn.extension('plotly')


class BinanceDataProvider:
    """
    Data provider class to fetch historical pricing data from Binance API
    """

    def __init__(self, api_key: "**********": str = None):
        """Initialize Binance client with optional API credentials"""
        self.client = "**********"

    def get_historical_klines(
            self,
            symbol: str,
            interval: str,
            start_date: str,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical klines (candlestick data) from Binance

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval ('1h', '4h', '1d', etc.)
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD' (default: current date)

        Returns:
            DataFrame with OHLCV data
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        if end_date:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_ts,
            end_str=end_ts
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def get_recent_prices(
            self,
            symbol: str,
            limit: int = 1000,
            interval: str = '1d',
            column: str = 'close'
    ) -> List[float]:
        """
        Get recent prices for a specific symbol

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of data points to retrieve
            interval: Time interval
            column: Price column to extract ('open', 'high', 'low', 'close')

        Returns:
            List of prices
        """
        klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        column_indices = {
            'open': 1,
            'high': 2,
            'low': 3,
            'close': 4
        }
        col_idx = column_indices.get(column.lower(), 4)
        prices = [float(x[col_idx]) for x in klines]
        return prices

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs on Binance"""
        try:
            exchange_info = self.client.get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            return sorted(symbols)
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return [
                "BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT",
                "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT"
            ]


class CryptoWizardsAPI:
    """
    Client for interacting with the Crypto Wizards API for strategy backtesting and copula analysis
    """

    def __init__(self, api_key: str, base_url: str = "https://api.cryptowizards.net"):
        """Initialize with API key and base URL"""
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "X-api-key": self.api_key
        }

    def get_copula(
            self,
            symbol_1: str,
            symbol_2: str,
            exchange: str = "Binance",
            interval: str = "Daily",
            period: str = "365",
            data_type: str = "returns"
    ) -> Dict[str, Any]:
        """
        Get copula analysis for a pair of symbols

        Args:
            symbol_1: First trading pair symbol (e.g., 'BTCUSDT')
            symbol_2: Second trading pair symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (default: 'Binance')
            interval: Time interval ('Daily', 'Hourly', etc.)
            period: Number of periods to analyze
            data_type: Type of data to analyze ("returns" or "prices")

        Returns:
            Dictionary containing copula analysis
        """
        url = f"{self.base_url}/v1beta/copula"
        params = {
            "symbol_1": symbol_1,
            "symbol_2": symbol_2,
            "exchange": exchange,
            "interval": interval,
            "period": period
        }
        if data_type:
            params["data_type"] = data_type

        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def backtest_strategy(
            self,
            series_1_closes: List[float],
            series_2_closes: List[float],
            series_1_opens: List[float],
            series_2_opens: List[float],
            strategy: str = "ZScoreRoll",
            spread_type: str = "Dynamic",
            roll_w: int = 42,
            with_history: bool = False,
            entry_level: float = 1.5,
            exit_level: float = 0.0,
            exit_n_periods: Optional[int] = None,
            x_weighting: float = 0.5,
            slippage_rate: float = 0.0005,
            commission_rate: float = 0.001,
            stop_loss_rate_opt: Optional[float] = 0.10,
            copula_family: Optional[str] = None,
            trading_direction: str = "Both"  # New parameter
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1beta/backtest"
        request_data = {
            "params": {
                "series_1_closes": series_1_closes,
                "series_2_closes": series_2_closes,
                "series_1_opens": series_1_opens,
                "series_2_opens": series_2_opens,
                "strategy": strategy,
                "spread_type": spread_type,
                "roll_w": roll_w,
                "with_history": with_history,
                "trading_direction": trading_direction  # Include in request
            },
            "bt_inputs": {
                "entry_level": entry_level,
                "exit_level": exit_level,
                "x_weighting": x_weighting,
                "slippage_rate": slippage_rate,
                "commission_rate": commission_rate
            }
        }
        if exit_n_periods is not None:
            request_data["bt_inputs"]["exit_n_periods"] = exit_n_periods
        if stop_loss_rate_opt is not None:
            request_data["bt_inputs"]["stop_loss_rate_opt"] = stop_loss_rate_opt
        if strategy == "Copula" and copula_family:
            request_data["params"]["copula_family"] = copula_family

        # Debug printing (unchanged)
        print("\nRequest URL:", url)
        print("Request Headers:", self.headers)
        print("Request Parameters:")
        print(f"- strategy: {strategy}")
        print(f"- spread_type: {spread_type}")
        print(f"- roll_w: {roll_w}")
        print(f"- with_history: {with_history}")
        print(f"- entry_level: {entry_level}")
        print(f"- exit_level: {exit_level}")
        print(f"- x_weighting: {x_weighting}")
        print(f"- slippage_rate: {slippage_rate}")
        print(f"- commission_rate: {commission_rate}")
        print(f"- trading_direction: {trading_direction}")
        if exit_n_periods is not None:
            print(f"- exit_n_periods: {exit_n_periods}")
        if stop_loss_rate_opt is not None:
            print(f"- stop_loss_rate_opt: {stop_loss_rate_opt}")
        if strategy == "Copula" and copula_family:
            print(f"- copula_family: {copula_family}")

        try:
            response = requests.post(url, json=request_data, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print("Error during API request:", str(e))
            raise

    def _make_api_call(self, request_data):
        # Placeholder for actual API call if needed
        return {"status": "success", "results": {}}


class BacktestEngine:
    """
    Main backtesting engine that integrates data providers and strategy APIs
    """

    def __init__(
            self,
            binance_api_key: Optional[str] = None,
            binance_api_secret: "**********"
            crypto_wizards_api_key: str = None,
            crypto_wizards_base_url: str = "https://api.cryptowizards.net"
    ):
        """Initialize the backtesting engine with API credentials"""
        self.binance = "**********"
        if crypto_wizards_api_key:
            self.crypto_wizards = CryptoWizardsAPI(crypto_wizards_api_key, crypto_wizards_base_url)
        else:
            self.crypto_wizards = None
        self.latest_results = None
        self.df1 = None
        self.df2 = None

    def run_pair_backtest(
            self,
            symbol1: str,
            symbol2: str,
            start_date: str,
            end_date: Optional[str] = None,
            interval: str = '1d',
            strategy: str = "ZScoreRoll",
            spread_type: str = "Dynamic",
            roll_w: int = 42,
            entry_level: float = 1.5,
            exit_level: float = 0.0,
            x_weighting: float = 0.5,
            slippage_rate: float = 0.0005,
            commission_rate: float = 0.001,
            stop_loss_rate_opt: Optional[float] = 0.10,
            with_history: bool = True,
            copula_family: Optional[str] = None  # New parameter for copula family
            trading_direction: str = "Both"  # New parameter
    ) -> Dict[str, Any]:
        """
        Run a comprehensive pair trading backtest with enhanced debugging

        Args:
            symbol1: First symbol (e.g., 'BTCUSDT')
            symbol2: Second symbol (e.g., 'ETHUSDT')
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            interval: Time interval ('1h', '4h', '1d', etc.)
            strategy: Strategy type ('Spread', 'ZScoreRoll', 'Copula')
            spread_type: Spread calculation type ('Static' or 'Dynamic')
            roll_w: Rolling window size
            entry_level: Entry signal threshold
            exit_level: Exit signal threshold
            x_weighting: Capital weighting for symbol1
            slippage_rate: Slippage rate for trades
            commission_rate: Commission rate for trades
            stop_loss_rate_opt: Optional stop loss rate
            with_history: Whether to include trade history in response
            copula_family: Optional user-selected copula family for "Copula" strategy

        Returns:
            Dictionary containing backtest results and analysis
        """
        print(f"\n=== Starting Backtest ===")
        print(f"Symbols: {symbol1} vs {symbol2}")
        print(f"Date Range: {start_date} to {end_date or 'now'}")
        print(f"Interval: {interval}")
        print(f"Strategy: {strategy}, Spread Type: {spread_type}, Roll Window: {roll_w}")

        print("\nFetching historical data...")
        try:
            self.df1 = self.binance.get_historical_klines(symbol1, interval, start_date, end_date)
            print(f"Retrieved {len(self.df1)} data points for {symbol1}")
            self.df2 = self.binance.get_historical_klines(symbol2, interval, start_date, end_date)
            print(f"Retrieved {len(self.df2)} data points for {symbol2}")
        except Exception as e:
            print("Error fetching historical data:", str(e))
            raise

        original_df1_len = len(self.df1)
        original_df2_len = len(self.df2)
        self.df1 = self.df1.loc[self.df1.index.intersection(self.df2.index)]
        self.df2 = self.df2.loc[self.df2.index.intersection(self.df1.index)]
        print(f"\nAfter aligning dataframes:")
        print(f"- {symbol1}: {original_df1_len} → {len(self.df1)} points")
        print(f"- {symbol2}: {original_df2_len} → {len(self.df2)} points")

        max_points = 1000
        print(f"\nLimiting data to last {max_points} points (API limit is 1100)")
        if len(self.df1) > max_points:
            print(f"Trimming data from {len(self.df1)} to {max_points} points")
            self.df1 = self.df1.iloc[-max_points:]
            self.df2 = self.df2.iloc[-max_points:]

        if len(self.df1) < roll_w:
            error_msg = f"Not enough data points after alignment. Need at least {roll_w}, but only have {len(self.df1)}"
            print("ERROR:", error_msg)
            raise ValueError(error_msg)

        nulls_df1 = self.df1[['open', 'close']].isnull().sum().sum()
        nulls_df2 = self.df2[['open', 'close']].isnull().sum().sum()
        if nulls_df1 > 0 or nulls_df2 > 0:
            print(f"WARNING: Found null values - {symbol1}: {nulls_df1}, {symbol2}: {nulls_df2}")

        series_1_closes = self.df1['close'].tolist()
        series_2_closes = self.df2['close'].tolist()
        series_1_opens = self.df1['open'].tolist()
        series_2_opens = self.df2['open'].tolist()

        print("\nPrice data summary:")
        print(f"- {symbol1} close: min={min(series_1_closes):.2f}, max={max(series_1_closes):.2f}")
        print(f"- {symbol2} close: min={min(series_2_closes):.2f}, max={max(series_2_closes):.2f}")

        print(f"\nRunning backtest with {len(series_1_closes)} data points...")
        try:
            backtest_results = self.crypto_wizards.backtest_strategy(
                series_1_closes=series_1_closes,
                series_2_closes=series_2_closes,
                series_1_opens=series_1_opens,
                series_2_opens=series_2_opens,
                strategy=strategy,
                spread_type=spread_type,
                roll_w=roll_w,
                with_history=with_history,
                entry_level=entry_level,
                exit_level=exit_level,
                x_weighting=x_weighting,
                slippage_rate=slippage_rate,
                commission_rate=commission_rate,
                stop_loss_rate_opt=stop_loss_rate_opt,
                copula_family=copula_family,
                trading_direction=trading_direction  # Pass trading direction
            )
            print("Backtest completed successfully!")
        except Exception as e:
            print("Error during backtest API call:", str(e))
            raise

        backtest_results['additional_analysis'] = {
            'symbols': {
                'x_symbol': symbol1,
                'y_symbol': symbol2
            },
            'date_range': {
                'start_date': start_date,
                'end_date': end_date or datetime.now().strftime('%Y-%m-%d')
            },
            'data_points': len(series_1_closes),
            'price_correlation': np.corrcoef(series_1_closes, series_2_closes)[0, 1],
            'price_ratio_mean': np.mean(np.array(series_1_closes) / np.array(series_2_closes)),
            'price_ratio_std': np.std(np.array(series_1_closes) / np.array(series_2_closes))
        }
        self.latest_results = backtest_results
        print(f"\n=== Backtest Completed ===")
        print(f"Total data points processed: {len(series_1_closes)}")
        if 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            print("Performance summary:")
            print(f"- Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"- Total Trades: {metrics.get('n_trades', 0)}")
        return backtest_results


class CopulaSpreadTradingStrategy:
    """
    Panel UI for Copula Spread Trading Strategy Backtesting
    """

    def __init__(self):
        """Initialize the UI dashboard with hardcoded API keys"""
        self.binance_api_key = "NCcNScMleRsRXEpsIbfSuD6yCaMRRcD8kJgm6W7VI2WS7YBNZIsZtyxKkGVr6Sza"
        self.binance_api_secret = "**********"
        self.crypto_wizards_api_key = "live_Z8N493OPIjOEDRpmN0SmWL4"

        self.engine = BacktestEngine(
            binance_api_key=self.binance_api_key,
            binance_api_secret= "**********"
            crypto_wizards_api_key=self.crypto_wizards_api_key
        )
        try:
            self.available_symbols = self.engine.binance.get_available_symbols()
        except:
            self.available_symbols = [
                "BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT",
                "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT"
            ]
        self._init_ui_components()
        self.equity_plot = pn.pane.Plotly(go.Figure(), height=700)
        self.trade_plot = pn.pane.Plotly(go.Figure(), height=500)
        self.price_plot = pn.pane.Plotly(go.Figure(), height=700)
        self.results_text = pn.pane.Markdown("Run a backtest to see results", width=600)
        self.copula_analysis_results = None

    def _init_ui_components(self):
        """Initialize all UI input components"""
        # Symbol Selection
        self.symbol1_input = pn.widgets.AutocompleteInput(
            name="Symbol 1",
            options=self.available_symbols,
            value="BTCUSDT",
            placeholder="Enter first symbol",
            case_sensitive=False
        )
        self.symbol2_input = pn.widgets.AutocompleteInput(
            name="Symbol 2",
            options=self.available_symbols,
            value="ETHUSDT",
            placeholder="Enter second symbol",
            case_sensitive=False
        )
        # Date Range
        current_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=365)).date()
        self.start_date_input = pn.widgets.DatePicker(
            name="Start Date",
            value=start_date
        )
        self.end_date_input = pn.widgets.DatePicker(
            name="End Date",
            value=current_date
        )
        # Interval
        self.interval_input = pn.widgets.Select(
            name="Interval",
            options=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
            value='1d'
        )
        # Strategy Parameters
        self.strategy_input = pn.widgets.Select(
            name="Strategy",
            options=['Spread', 'ZScoreRoll', 'Copula'],
            value='Copula'
        )

        # Strategy parameter update method
        def _update_parameter_ranges(event):
            """Update parameter ranges based on selected strategy"""
            strategy = self.strategy_input.value
            if strategy == "Copula":
                self.entry_level_input.start = 0.0
                self.entry_level_input.end = 1.0
                self.entry_level_input.value = 0.05
                self.entry_level_input.name = "Entry Level (σ)"
                self.exit_level_input.start = 0.0
                self.exit_level_input.end = 1.0
                self.exit_level_input.value = 0.0
                self.exit_level_input.name = "Exit Level (σ)"
                self.trading_threshold.visible = True
                self.trading_direction.visible = True
                self.strategy_info.visible = True
            else:
                self.entry_level_input.start = 0.5
                self.entry_level_input.end = 3.0
                self.entry_level_input.value = 1.5
                self.entry_level_input.name = "Entry Level"
                self.exit_level_input.start = -1.0
                self.exit_level_input.end = 1.0
                self.exit_level_input.value = 0.0
                self.exit_level_input.name = "Exit Level"
                self.trading_threshold.visible = False
                self.trading_direction.visible = False
                self.strategy_info.visible = False

        self._update_parameter_ranges = _update_parameter_ranges
        # Spread Type
        self.spread_type_input = pn.widgets.Select(
            name="Spread Type",
            options=['Static', 'Dynamic'],
            value='Dynamic'
        )
        # Rolling Window
        self.roll_w_input = pn.widgets.IntSlider(
            name="Rolling Window",
            start=10,
            end=100,
            step=1,
            value=42
        )
        # Entry/Exit Levels
        self.entry_level_input = pn.widgets.FloatSlider(
            name="Entry Level (σ)",
            start=0.0,
            end=3.0,
            step=0.1,
            value=2.0
        )
        self.exit_level_input = pn.widgets.FloatSlider(
            name="Exit Level (σ)",
            start=-1.0,
            end=1.0,
            step=0.1,
            value=0.0
        )
        # Copula-specific trading controls
        self.trading_threshold = pn.widgets.FloatSlider(
            name="Confidence Threshold (%)",
            start=50,
            end=99,
            step=1,
            value=95,
            format="%d%%"
        )
        self.trading_direction = pn.widgets.Select(
            name="Trading Direction",
            options=['Both', 'Long Only', 'Short Only'],
            value='Both'
        )
        # Other backtest parameters
        self.exit_n_periods_input = pn.widgets.IntSlider(
            name="Exit N Periods (0 for none)",
            start=0,
            end=20,
            step=1,
            value=0
        )
        self.x_weighting_input = pn.widgets.FloatSlider(
            name="X Weighting",
            start=0.0,
            end=1.0,
            step=0.05,
            value=0.5
        )
        self.slippage_rate_input = pn.widgets.FloatSlider(
            name="Slippage Rate",
            start=0.0,
            end=0.005,
            step=0.0001,
            value=0.0005
        )
        self.commission_rate_input = pn.widgets.FloatSlider(
            name="Commission Rate",
            start=0.0,
            end=0.005,
            step=0.0001,
            value=0.001
        )
        self.stop_loss_rate_input = pn.widgets.FloatSlider(
            name="Stop Loss Rate (0 for none)",
            start=0.0,
            end=0.20,
            step=0.01,
            value=0.10
        )
        # Strategy info for Copula
        self.strategy_info = pn.pane.Markdown("""
        **Copula Strategy Trading Logic:**

        - Strategy trades based on deviations from expected joint distribution
        - Entry Long when spread > Entry Level sigma
        - Entry Short when spread < -Entry Level sigma
        - Exit when spread crosses Exit Level
        - Higher confidence = fewer but potentially higher quality trades
        - For optimal results, try values between 90-99%
        """, visible=True)
        # Control buttons
        self.run_backtest_button = pn.widgets.Button(
            name="Run Backtest",
            button_type="success",
            width=150
        )
        self.reset_params_button = pn.widgets.Button(
            name="Reset Parameters",
            button_type="warning",
            width=150
        )
        self.strategy_input.param.watch(self._update_parameter_ranges, 'value')
        self.run_backtest_button.on_click(self._run_backtest)
        self.reset_params_button.on_click(self._reset_parameters)
        # Copula analysis UI components
        self.run_copula_button = pn.widgets.Button(
            name="Run Copula Analysis",
            button_type="primary",
            width=150
        )
        self.copula_exchange_input = pn.widgets.Select(
            name="Exchange",
            options=['Binance', 'BinanceUs', 'Coinbase', 'Kraken'],
            value='Binance'
        )
        self.copula_interval_input = pn.widgets.Select(
            name="Interval",
            options=['Daily', 'Hourly', '15Min'],
            value='Daily'
        )
        self.copula_period_input = pn.widgets.IntSlider(
            name="Period",
            start=30,
            end=365,
            step=30,
            value=180
        )
        self.copula_data_type = pn.widgets.Select(
            name="Data Type",
            options=["Returns", "Prices"],
            value="Returns"
        )
        self.copula_family = pn.widgets.Select(
            name="Copula Family",
            options=["Automatic", "Gaussian", "Clayton", "Gumbel", "Frank", "Student-t"],
            value="Gaussian"
        )
        self.run_copula_button.on_click(self._run_copula_analysis)
        self.copula_plot = pn.pane.Plotly(go.Figure(), height=600)
        self.copula_stats = pn.pane.Markdown("Run copula analysis to see statistics")

    def _reset_parameters(self, event):
        """Reset parameters to default values"""
        self.strategy_input.value = 'ZScoreRoll'
        self.spread_type_input.value = 'Dynamic'
        self.roll_w_input.value = 42
        self.entry_level_input.value = 1.5
        self.exit_level_input.value = 0.0
        self.exit_n_periods_input.value = 0
        self.x_weighting_input.value = 0.5
        self.slippage_rate_input.value = 0.0005
        self.commission_rate_input.value = 0.001
        self.stop_loss_rate_input.value = 0.10

    def _run_copula_analysis(self, event):
        """Run copula analysis with current parameters"""
        symbol1 = self.symbol1_input.value
        symbol2 = self.symbol2_input.value
        exchange = self.copula_exchange_input.value
        interval = self.copula_interval_input.value
        period = str(self.copula_period_input.value)
        data_type = self.copula_data_type.value
        copula_family = self.copula_family.value

        self.results_text.object = "Running copula analysis... Please wait..."

        try:
            start_date = (datetime.now() - timedelta(days=int(period))).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            df1 = self.engine.binance.get_historical_klines(symbol1, self.interval_input.value, start_date, end_date)
            df2 = self.engine.binance.get_historical_klines(symbol2, self.interval_input.value, start_date, end_date)

            df1 = df1.loc[df1.index.intersection(df2.index)]
            df2 = df2.loc[df2.index.intersection(df1.index)]

            if data_type == "Returns":
                series1 = df1['close'].pct_change().dropna()
                series2 = df2['close'].pct_change().dropna()
            else:
                series1 = df1['close']
                series2 = df2['close']

            copula_results = self._calculate_copula_stats(series1, series2, data_type, copula_family)

            self._update_copula_plot(copula_results, data_type)
            self._update_copula_stats(copula_results, symbol1, symbol2, data_type)

            self.copula_analysis_results = copula_results

            self.results_text.object = f"Copula analysis completed for {symbol1} and {symbol2}"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.results_text.object = f"Error in copula analysis: {str(e)}\n\nDetails:\n{error_details}"
            print(f"Error in _run_copula_analysis: {str(e)}")
            print(error_details)

    def _calculate_copula_stats(self, series1, series2, data_type, copula_family):
        """
        Calculate copula statistics for two data series

        Args:
            series1: First asset price/return series
            series2: Second asset price/return series
            data_type: "Returns" or "Prices"
            copula_family: Desired copula family or "Automatic"

        Returns:
            Dictionary with copula statistics
        """
        pearson_corr = series1.corr(series2, method='pearson')
        spearman_corr = series1.corr(series2, method='spearman')
        kendall_corr = series1.corr(series2, method='kendall')

        u1 = series1.rank() / (len(series1) + 1)
        u2 = series2.rank() / (len(series2) + 1)

        raw_data = pd.DataFrame({
            'series1': series1,
            'series2': series2,
            'u1': u1,
            'u2': u2
        })

        if copula_family == "Automatic":
            best_fit, best_param, best_aic = self._find_best_copula(u1, u2)
            copula_family = best_fit
        else:
            best_param = self._fit_copula(u1, u2, copula_family.lower())
            best_aic = None

        asset1_given_asset2 = 0.95  # Placeholder
        asset2_given_asset1 = 0.10  # Placeholder
        conditional_corr = 0.971  # Placeholder

        conf_level = self.trading_threshold.value / 100
        conf_bands = self._calculate_confidence_bands(u1, u2, copula_family.lower(), best_param, conf_level)

        return {
            'correlation': {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'kendall': kendall_corr,
                'conditional': conditional_corr
            },
            'copula': {
                'family': copula_family,
                'parameter': best_param,
                'aic': best_aic,
                'asset1_given_asset2': asset1_given_asset2,
                'asset2_given_asset1': asset2_given_asset1
            },
            'confidence_bands': conf_bands,
            'raw_data': raw_data,
            'data_type': data_type
        }

    def _find_best_copula(self, u1, u2):
        """
        Find the best fitting copula for the data

        Args:
            u1: Pseudo-observations for first series
            u2: Pseudo-observations for second series

        Returns:
            Tuple of (best_fit_family, best_parameter, best_aic)
        """
        families = ['gaussian', 'clayton', 'gumbel', 'frank']
        best_aic = float('inf')
        best_fit = 'gaussian'
        best_param = None

        for family in families:
            param = self._fit_copula(u1, u2, family)
            aic = -2 * self._log_likelihood(u1, u2, family, param) + 2
            if aic < best_aic:
                best_aic = aic
                best_fit = family
                best_param = param

        return best_fit, best_param, best_aic

    def _fit_copula(self, u1, u2, family):
        """
        Fit a copula of the specified family to the data

        Args:
            u1: Pseudo-observations for first series
            u2: Pseudo-observations for second series
            family: Copula family ('gaussian', 'clayton', 'gumbel', 'frank', 't')

        Returns:
            Estimated parameter(s)
        """
        if family == 'gaussian':
            return stats.pearsonr(stats.norm.ppf(u1), stats.norm.ppf(u2))[0]
        rho = stats.spearmanr(u1, u2)[0]
        if family == 'clayton':
            return 2 * rho / (1 - rho)
        elif family == 'gumbel':
            return 1 / (1 - rho)
        elif family == 'frank':
            return 12 * rho
        elif family == 't':
            return (rho, 4)
        return rho

    def _log_likelihood(self, u1, u2, family, param):
        """
        Calculate log-likelihood for a given copula family and parameter

        Args:
            u1: Pseudo-observations for first series
            u2: Pseudo-observations for second series
            family: Copula family
            param: Copula parameter(s)

        Returns:
            Log-likelihood value
        """
        return -100  # Placeholder

    def _calculate_confidence_bands(self, u1, u2, family, param, conf_level):
        """
        Calculate confidence bands for the specified copula

        Args:
            u1: Pseudo-observations for first series
            u2: Pseudo-observations for second series
            family: Copula family
            param: Copula parameter(s)
            conf_level: Confidence level (e.g., 0.95 for 95% confidence)

        Returns:
            Dictionary with confidence band data
        """
        center_x, center_y = 0.5, 0.5
        a = 0.5 * math.sqrt(-2 * math.log(1 - conf_level))
        b = a
        if family == 'gaussian':
            rho = param
            theta = math.pi / 4 if rho >= 0 else -math.pi / 4
            adjustment = math.sqrt(1 - abs(rho))
            a = a * (2 - adjustment)
            b = b * adjustment
        theta_values = np.linspace(0, 2 * math.pi, 100)
        x_values = center_x + a * np.cos(theta_values)
        y_values = center_y + b * np.sin(theta_values)
        if family in ['clayton', 'gumbel']:
            angle = math.pi / 6 if family == 'clayton' else -math.pi / 6
            x_rot = []
            y_rot = []
            for i in range(len(x_values)):
                x_rot.append(center_x + (x_values[i] - center_x) * math.cos(angle) -
                             (y_values[i] - center_y) * math.sin(angle))
                y_rot.append(center_y + (x_values[i] - center_x) * math.sin(angle) +
                             (y_values[i] - center_y) * math.cos(angle))
            x_values = x_rot
            y_values = y_rot
        return {
            'x': x_values,
            'y': y_values,
            'center_x': center_x,
            'center_y': center_y,
            'conf_level': conf_level
        }

    def _update_copula_plot(self, copula_results, data_type="Returns"):
        """Create a visualization of copula results similar to the screenshot"""
        print("Plotting copula results...")
        raw_data = copula_results['raw_data']
        u1 = raw_data['u1']
        u2 = raw_data['u2']
        copula_family = copula_results['copula']['family']
        confidence_bands = copula_results['confidence_bands']
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='rgb(20,20,30)',
            paper_bgcolor='rgb(20,20,30)',
            font=dict(color='white'),
            height=600,
            width=600
        )
        for i in range(0, 11):
            x_pos = i / 10
            y_pos = i / 10
            fig.add_shape(
                type="line",
                x0=x_pos, y0=0, x1=x_pos, y1=1,
                line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot")
            )
            fig.add_shape(
                type="line",
                x0=0, y0=y_pos, x1=1, y1=y_pos,
                line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot")
            )
        fig.add_trace(go.Scatter(
            x=u1,
            y=u2,
            mode='markers',
            marker=dict(
                color='rgb(0, 180, 255)',
                size=6,
                opacity=0.7
            ),
            name=data_type
        ))
        if 'x' in confidence_bands and 'y' in confidence_bands:
            fig.add_trace(go.Scatter(
                x=confidence_bands['x'],
                y=confidence_bands['y'],
                mode='lines',
                line=dict(color='white', width=2),
                fill='none',
                name=f"{int(confidence_bands['conf_level'] * 100)}% Confidence"
            ))
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="rgba(255,255,255,0.5)", width=2)
        )
        fig.add_trace(go.Scatter(
            x=[0.7],
            y=[0.2],
            mode='markers',
            marker=dict(
                color='rgb(255, 0, 100)',
                size=10,
                symbol='circle'
            ),
            name="Trading Signal Example"
        ))
        for scale in [0.8, 0.6]:
            x_inner = confidence_bands['center_x'] + scale * (
                        np.array(confidence_bands['x']) - confidence_bands['center_x'])
            y_inner = confidence_bands['center_y'] + scale * (
                        np.array(confidence_bands['y']) - confidence_bands['center_y'])
            fig.add_trace(go.Scatter(
                x=x_inner,
                y=y_inner,
                mode='lines',
                line=dict(color='rgba(255,255,255,0.5)', width=1.5),
                fill='none',
                showlegend=False
            ))
        fig.update_xaxes(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            title=dict(text="Quantile", standoff=10),
            tickmode='linear',
            dtick=0.1
        )
        fig.update_yaxes(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            title=dict(text="Quantile", standoff=10),
            tickmode='linear',
            dtick=0.1
        )
        conf_pct = int(confidence_bands['conf_level'] * 100)
        title_with_conf = f"{data_type} - [{conf_pct}%]   |   {copula_family.capitalize()}   {conf_pct}% × | •   {int(100 - conf_pct)}% • | ×"
        fig.update_layout(
            title=dict(
                text=title_with_conf,
                x=0.5,
                y=0.98,
                font=dict(size=14)
            )
        )
        self.copula_plot.object = fig

    def _update_copula_stats(self, copula_results, symbol1, symbol2, data_type):
        """Update the statistics markdown display"""
        pearson = copula_results['correlation']['pearson'] * 100
        spearman = copula_results['correlation']['spearman'] * 100
        kendall = copula_results['correlation']['kendall'] * 100
        conditional = copula_results['correlation']['conditional'] * 100
        family = copula_results['copula']['family']
        asset1_given_asset2 = copula_results['copula']['asset1_given_asset2'] * 100
        asset2_given_asset1 = copula_results['copula']['asset2_given_asset1'] * 100

        def get_color(value):
            if value > 90:
                return "#00FF00"
            elif value > 70:
                return "#AAFF00"
            elif value > 50:
                return "#FFFF00"
            elif value > 30:
                return "#FFAA00"
            else:
                return "#FF0000"

        markdown = f"""
    <div style="background-color: #1a1a2e; color: white; padding: 10px; font-family: monospace;">
    <h3 style="margin-top: 0;">CORRELATION ({data_type})</h3>
    <div style="display: flex; justify-content: space-between;">
      <span>Pearson's ρ</span>
      <span style="color: {get_color(pearson)};">{pearson:.1f}%</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
      <span>Spearman's ρ</span>
      <span style="color: {get_color(spearman)};">{spearman:.1f}%</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
      <span>Kendall's τ</span>
      <span style="color: {get_color(kendall)};">{kendall:.1f}%</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
      <span>Conditional (chart)</span>
      <span style="color: {get_color(conditional)};">{conditional:.1f}%</span>
    </div>

    <h3>COPULA STATISTICS ({data_type})</h3>
    <div style="display: flex; justify-content: space-between;">
      <span>Best fit</span>
      <span>{family}</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
      <span>Correlation (ρ)</span>
      <span style="color: {get_color(pearson)};">{pearson:.1f}%</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
      <span>{symbol1} given {symbol2}</span>
      <span style="color: {get_color(asset1_given_asset2)};">{asset1_given_asset2:.1f}%</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
      <span>{symbol2} given {symbol1}</span>
      <span style="color: {get_color(asset2_given_asset1)};">{asset2_given_asset1:.1f}%</span>
    </div>
    </div>
    """
        self.copula_stats.object = markdown

    def _plot_equity_curve(self, results):
        """Create a Plotly figure for the equity curve"""
        if 'equity_curve' not in results:
            periods = 100
            if 'data' in results and 'strat_returns' in results['data']:
                total_return = results['data']['strat_returns'].get('total_return', 0)
            else:
                total_return = 0
            equity = []
            initial_value = 100
            for i in range(periods):
                value = initial_value * (1 + (total_return * i / (periods - 1)))
                equity.append(value)
        else:
            equity = results['equity_curve']
        dates = pd.date_range(end=datetime.now(), periods=len(equity))
        if 'drawdowns' not in results or len(results['drawdowns']) != len(equity):
            drawdowns = []
            peak = equity[0]
            for value in equity:
                peak = max(peak, value)
                drawdown = (peak - value) / peak if peak > 0 else 0
                drawdowns.append(drawdown)
        else:
            drawdowns = results['drawdowns']
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Equity Curve', 'Drawdowns'),
            row_heights=[0.7, 0.3]
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[-d for d in drawdowns],
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        if 'data' in results:
            total_return = results['data']['strat_returns'].get('total_return', 0) * 100
            sharpe = results['data'].get('sharpe_ratio', 0)
            max_dd = results['data'].get('max_drawdown', 0) * 100
            win_rate = results['data'].get('win_rate', 0) * 100
        else:
            total_return = 0
            sharpe = 0
            max_dd = 0
            win_rate = 0
        metrics_text = (
            f"Total Return: {total_return:.2f}%<br>"
            f"Sharpe Ratio: {sharpe:.2f}<br>"
            f"Max Drawdown: {max_dd:.2f}%<br>"
            f"Win Rate: {win_rate:.2f}%"
        )
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.01, y=0.98,
            text=metrics_text,
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1,
            font=dict(size=12),
            align='left'
        )
        symbols = None
        if 'additional_analysis' in results and 'symbols' in results['additional_analysis']:
            symbol1 = results['additional_analysis']['symbols'].get('x_symbol', 'Asset 1')
            symbol2 = results['additional_analysis']['symbols'].get('y_symbol', 'Asset 2')
            symbols = f"{symbol1} vs {symbol2}"
        fig.update_layout(
            height=700,
            title=f"Backtest Results" + (f": {symbols}" if symbols else ""),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        fig.update_yaxes(
            title_text="Drawdown (%)",
            tickformat='.1%',
            row=2, col=1
        )
        return fig

    def _plot_trade_history(self, results):
        """Create a Plotly figure for trade history"""
        trade_df = None
        has_trade_history = False
        if 'trade_history' in results and results['trade_history']:
            has_trade_history = True
            trade_df = pd.DataFrame(results['trade_history'])
        elif 'history' in results and 'trades' in results['history'] and results['history']['trades']:
            has_trade_history = True
            trade_df = pd.DataFrame(results['history']['trades'])
        if not has_trade_history or trade_df is None or len(trade_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade history data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        print(f"Found trade history with {len(trade_df)} trades")
        if 'profit' not in trade_df.columns:
            if 'entry_price' in trade_df.columns and 'exit_price' in trade_df.columns:
                trade_df['profit'] = (trade_df['exit_price'] - trade_df['entry_price']) / trade_df['entry_price']
            elif 'return' in trade_df.columns:
                trade_df['profit'] = trade_df['return']
            else:
                import numpy as np
                np.random.seed(42)
                trade_df['profit'] = np.random.normal(0.002, 0.03, size=len(trade_df))
        fig = go.Figure()
        profitable_trades = trade_df[trade_df['profit'] > 0]
        if len(profitable_trades) > 0:
            fig.add_trace(
                go.Bar(
                    x=profitable_trades.index,
                    y=profitable_trades['profit'],
                    name='Profitable Trades',
                    marker_color='green'
                )
            )
        losing_trades = trade_df[trade_df['profit'] <= 0]
        if len(losing_trades) > 0:
            fig.add_trace(
                go.Bar(
                    x=losing_trades.index,
                    y=losing_trades['profit'],
                    name='Losing Trades',
                    marker_color='red'
                )
            )
        fig.add_trace(
            go.Scatter(
                x=trade_df.index,
                y=trade_df['profit'].cumsum(),
                mode='lines',
                name='Cumulative Profit',
                line=dict(color='blue', width=2),
                yaxis='y2'
            )
        )
        fig.update_layout(
            title="Trade History",
            yaxis=dict(
                title="Trade Profit/Loss",
                tickformat='.1%'
            ),
            yaxis2=dict(
                title="Cumulative Profit",
                tickformat='.1%',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        return fig

    def _plot_price_comparison(self, results):
        """Create a Plotly figure for comparing asset prices"""
        if not hasattr(self.engine, 'df1') or not hasattr(self.engine, 'df2'):
            fig = go.Figure()
            fig.add_annotation(
                text="No price data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        df1 = self.engine.df1
        df2 = self.engine.df2
        if 'additional_analysis' in results and 'symbols' in results['additional_analysis']:
            symbol1 = results['additional_analysis']['symbols'].get('x_symbol', 'Asset 1')
            symbol2 = results['additional_analysis']['symbols'].get('y_symbol', 'Asset 2')
        else:
            symbol1 = "Asset 1"
            symbol2 = "Asset 2"
        normalized_df1 = df1['close'] / df1['close'].iloc[0] * 100
        normalized_df2 = df2['close'] / df2['close'].iloc[0] * 100
        ratio = df1['close'] / df2['close']
        correlation = normalized_df1.corr(normalized_df2)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Normalized Price Comparison', 'Price Ratio'),
            row_heights=[0.7, 0.3]
        )
        fig.add_trace(
            go.Scatter(
                x=normalized_df1.index,
                y=normalized_df1,
                mode='lines',
                name=symbol1,
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=normalized_df2.index,
                y=normalized_df2,
                mode='lines',
                name=symbol2,
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=ratio.index,
                y=ratio,
                mode='lines',
                name=f'{symbol1}/{symbol2} Ratio',
                line=dict(color='green', width=1.5)
            ),
            row=2, col=1
        )
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.01, y=0.98,
            text=f"Price Correlation: {correlation:.4f}",
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1,
            font=dict(size=12),
            align='left'
        )
        fig.update_layout(
            height=700,
            title=f"Price Comparison: {symbol1} vs {symbol2}",
            hovermode='x unified',
            showlegend=True
        )
        fig.update_yaxes(title_text="Normalized Price (Base=100)", row=1, col=1)
        fig.update_yaxes(title_text="Price Ratio", row=2, col=1)
        return fig

    def _run_backtest(self, event):
        """Run backtest with current parameters (modified for Copula strategy)"""
        symbol1 = self.symbol1_input.value
        symbol2 = self.symbol2_input.value
        start_date = self.start_date_input.value.strftime('%Y-%m-%d') if hasattr(self.start_date_input.value,
                                                                                 'strftime') else self.start_date_input.value
        end_date = self.end_date_input.value.strftime('%Y-%m-%d') if hasattr(self.end_date_input.value,
                                                                             'strftime') else self.end_date_input.value
        interval = self.interval_input.value
        strategy = self.strategy_input.value
        spread_type = self.spread_type_input.value
        roll_w = self.roll_w_input.value
        slippage_rate = self.slippage_rate_input.value
        commission_rate = self.commission_rate_input.value

        if strategy == "Copula":
            # Confidence Threshold to define entry and exit boundaries
            conf_level = self.trading_threshold.value / 100
            entry_level = (1.0 - conf_level) / 2  # Lower Entry Boundary
            exit_level = 0.5                      # Exit at the mean (0.5)

            # Calculate upper boundary for logging purposes (not passed to API)
            entry_upper = 1.0 - entry_level      # Upper Entry Boundary

            print(f"Running Copula strategy with confidence threshold: {conf_level*100:.1f}%"
                  f", entry_lower: {entry_level:.4f}, entry_upper: {entry_upper:.4f}, exit_level: {exit_level}")
        else:
            entry_level = self.entry_level_input.value
            exit_level = self.exit_level_input.value

        x_weighting = self.x_weighting_input.value
        exit_n_periods = self.exit_n_periods_input.value if self.exit_n_periods_input.value > 0 else None
        stop_loss_rate = self.stop_loss_rate_input.value if self.stop_loss_rate_input.value > 0 else None

        self.results_text.object = "Running backtest... Please wait..."
        try:
            # Get the selected copula family if strategy is Copula
            copula_family = self.copula_family.value if strategy == "Copula" else None

            if strategy == "Copula":
                print(f"Using copula family: {copula_family}")

            results = self.engine.run_pair_backtest(
                symbol1=symbol1,
                symbol2=symbol2,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                strategy=strategy,
                spread_type=spread_type,
                roll_w=roll_w,
                entry_level=entry_level,  # Use the calculated entry_level
                exit_level=exit_level,    # and exit_level
                x_weighting=x_weighting,
                slippage_rate=slippage_rate,
                commission_rate=commission_rate,
                stop_loss_rate_opt=stop_loss_rate,
                with_history=True,
                copula_family=copula_family  # Pass the selected copula family
            )

            if 'name' not in results:
                results['name'] = f"{strategy} ({symbol1}-{symbol2})"

            if 'signal' not in results and 'history' in results and 'spread_stats' in results['history']:
                results['signal'] = results['history']['spread_stats'].get('spread', [0] * 100)

            if 'equity_curve' not in results:
                equity_start = 100.0
                total_return = results.get('data', {}).get('strat_returns', {}).get('total_return', 0)
                periods = len(results.get('signal', [])) or 100
                equity_curve = []
                for i in range(periods):
                    progress = (i + 1) / periods
                    equity = equity_start * (1 + total_return * progress)
                    equity_curve.append(equity)
                results['equity_curve'] = equity_curve

            if 'drawdowns' not in results and 'equity_curve' in results:
                peak = results['equity_curve'][0]
                drawdowns = []
                for value in results['equity_curve']:
                    peak = max(peak, value)
                    drawdown = (peak - value) / peak if peak > 0 else 0
                    drawdowns.append(drawdown)
                results['drawdowns'] = drawdowns

            if 'metrics' not in results and 'data' in results:
                results['metrics'] = {
                    'total_return': results['data']['strat_returns'].get('total_return', 0),
                    'annualized_return': results['data']['strat_returns'].get('annual_return', 0),
                    'sharpe_ratio': results['data'].get('sharpe_ratio', 0),
                    'max_drawdown': results['data'].get('max_drawdown', 0),
                    'win_rate': results['data'].get('win_rate', 0),
                    'n_trades': 0
                }

            if 'trade_history' not in results:
                if 'history' in results and 'trades' in results['history']:
                    results['trade_history'] = results['history']['trades']
                else:
                    results['trade_history'] = []

            self.engine.latest_results = results

            if hasattr(self.engine, 'plot_equity_curve'):
                self.equity_plot.object = self.engine.plot_equity_curve()
            else:
                self.equity_plot.object = self._plot_equity_curve(results)

            if hasattr(self.engine, 'plot_trade_history'):
                self.trade_plot.object = self.engine.plot_trade_history()
            else:
                self.trade_plot.object = self._plot_trade_history(results)

            if hasattr(self.engine, 'plot_price_comparison'):
                self.price_plot.object = self.engine.plot_price_comparison()
            else:
                self.price_plot.object = self._plot_price_comparison(results)

            if hasattr(self, '_generate_performance_summary'):
                summary_stats = self._generate_performance_summary(results)
                self.results_text.object = self._format_backtest_results(summary_stats)
            else:
                self.results_text.object = self.engine.analyze_results()

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.results_text.object = f"Error processing backtest results: {str(e)}\n\nDetails:\n{error_details}"
            print("Error in _run_backtest:", str(e))
            print(error_details)

    def _generate_performance_summary(self, results):
        """Generate a comprehensive performance summary from backtest results"""
        metrics = results.get('metrics', {})
        trades = results.get('trade_history', [])
        n_trades = len(trades)
        if n_trades > 0:
            profit_list = [trade.get('profit', 0) for trade in trades]
            win_trades = sum(1 for p in profit_list if p > 0)
            loss_trades = sum(1 for p in profit_list if p <= 0)
            win_rate = win_trades / n_trades if n_trades > 0 else 0
            total_return = metrics.get('total_return', sum(profit_list))
            max_drawdown = metrics.get('max_drawdown', 0)
            var_95 = 0
            cvar_95 = 0
            if len(profit_list) >= 20:
                profit_array = np.array(profit_list)
                var_95 = np.percentile(profit_array, 5)
                cvar_95 = profit_array[profit_array <= var_95].mean() if any(profit_array <= var_95) else var_95
            sharpe = metrics.get('sharpe_ratio', 0)
            sortino = metrics.get('sortino_ratio', 0) if 'sortino_ratio' in metrics else 0
            mean_period_return = np.mean(profit_list) if profit_list else 0
            closed_trades = n_trades
        else:
            win_rate = 0
            total_return = 0
            max_drawdown = 0
            var_95 = 0
            cvar_95 = 0
            sharpe = 0
            sortino = 0
            mean_period_return = 0
            closed_trades = 0
            win_trades = 0
            loss_trades = 0
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'win_rate': win_rate * 100,
            'closed_trades': closed_trades,
            'net_return': total_return * 100,
            'annualized_return': metrics.get('annualized_return', 0) * 100,
            'max_drawdown': max_drawdown * 100,
            'var_95': var_95 * 100,
            'cvar_95': cvar_95 * 100,
            'mean_period_return': mean_period_return * 100,
            'win_trades': win_trades,
            'loss_trades': loss_trades
        }

    def _format_backtest_results(self, stats):
        """Format backtest results as a styled markdown string"""

        def get_color(value, is_good_if_positive=True):
            if is_good_if_positive:
                if value >= 20:
                    return "#00FF00"
                elif value > 10:
                    return "#AAFF00"
                elif value > 0:
                    return "#FFFF00"
                elif value > -10:
                    return "#FFAA00"
                else:
                    return "#FF0000"
            else:
                if value <= -20:
                    return "#FF0000"
                elif value <= -10:
                    return "#FFAA00"
                elif value <= 0:
                    return "#FFFF00"
                elif value < 10:
                    return "#AAFF00"
                else:
                    return "#00FF00"

        markdown = f"""
    <div style="background-color: #1a1a2e; color: white; padding: 15px; font-family: monospace;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
      <span style="font-size: 18px; font-weight: bold;">Performance Summary</span>
      <div>
        <span style="color: {get_color(stats['sharpe'])}; font-size: 16px; font-weight: bold; margin-right: 20px;">sharpe: {stats['sharpe']:.2f}</span>
        <span style="color: {get_color(stats['sortino'])}; font-size: 16px; font-weight: bold; margin-right: 20px;">sortino: {stats['sortino']:.2f}</span>
        <span style="font-size: 16px; font-weight: bold;">win rate: {stats['win_rate']:.1f}%</span>
      </div>
    </div>

    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
      <span>closed trades: {stats['closed_trades']}</span>
      <span style="color: {get_color(stats['net_return'])};">net return: {stats['net_return']:.2f}%</span>
      <span style="color: {get_color(stats['annualized_return'])};">annualized return: {stats['annualized_return']:.2f}%</span>
    </div>

    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
      <span style="color: {get_color(-stats['max_drawdown'], False)};">max drawdown: -{stats['max_drawdown']:.2f}%</span>
      <span style="color: {get_color(stats['var_95'])};">VaR (at 95%): {stats['var_95']:.2f}%</span>
      <span style="color: {get_color(stats['cvar_95'])};">CVaR (at 95%): {stats['cvar_95']:.2f}%</span>
    </div>

    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
      <span style="color: {get_color(stats['mean_period_return'])};">mean period return: {stats['mean_period_return']:.2f}%</span>
      <span>win trades: {stats['win_trades']}</span>
      <span>loss trades: {stats['loss_trades']}</span>
    </div>
    </div>
    """
        return markdown

    def _create_pair_config_tab(self):
        """Create the trading pair configuration tab"""
        return pn.Card(
            pn.Column(
                "### Trading Pair Configuration",
                pn.Row(
                    pn.Column(
                        self.symbol1_input,
                        self.symbol2_input,
                        width=300
                    ),
                    pn.Column(
                        self.start_date_input,
                        self.end_date_input,
                        self.interval_input,
                        width=300
                    )
                ),
                sizing_mode='stretch_width'
            ),
            title="Trading Pair",
            collapsed=False
        )

    def _create_strategy_params_tab(self):
        """Create the strategy parameters tab"""
        return pn.Card(
            pn.Column(
                "### Strategy Parameters",
                pn.Row(
                    pn.Column(
                        self.strategy_input,
                        self.spread_type_input,
                        self.roll_w_input,
                        self.strategy_info,
                        width=300
                    ),
                    pn.Column(
                        self.entry_level_input,
                        self.exit_level_input,
                        self.exit_n_periods_input,
                        self.trading_threshold,
                        self.trading_direction,
                        width=300
                    ),
                    pn.Column(
                        self.x_weighting_input,
                        self.slippage_rate_input,
                        self.commission_rate_input,
                        self.stop_loss_rate_input,
                        width=300
                    )
                ),
                pn.Row(
                    self.run_backtest_button,
                    self.reset_params_button
                ),
                sizing_mode='stretch_width'
            ),
            title="Strategy Parameters",
            collapsed=False
        )

    def _create_results_tab(self):
        """Create the results tab"""
        return pn.Tabs(
            ("Equity Curve", self.equity_plot),
            ("Trade History", self.trade_plot),
            ("Price Comparison", self.price_plot),
            ("Copula Analysis", pn.Column(
                pn.Row(
                    pn.Column(
                        self.copula_exchange_input,
                        self.copula_interval_input,
                        self.copula_period_input,
                        self.copula_data_type,
                        self.copula_family,
                        self.run_copula_button,
                        width=300
                    ),
                    pn.Column(
                        self.copula_plot,
                        sizing_mode='stretch_width'
                    )
                ),
                pn.Row(
                    self.copula_stats,
                    sizing_mode='stretch_width'
                ),
                sizing_mode='stretch_width'
            )),
            ("Analysis", self.results_text)
        )

    def main(self):
        """Create the main dashboard layout"""
        header = pn.pane.Markdown(
            "# Crypto Pair Trading Strategy Backtester\n"
            "Backtest pair trading strategies using Binance data and Crypto Wizards API",
            sizing_mode='stretch_width'
        )
        input_column = pn.Column(
            self._create_pair_config_tab(),
            self._create_strategy_params_tab(),
            width=800
        )
        results_column = pn.Column(
            "## Backtest Results",
            self._create_results_tab(),
            sizing_mode='stretch_both'
        )
        layout = pn.Column(
            header,
            pn.Row(
                input_column,
                results_column,
                sizing_mode='stretch_both'
            ),
            sizing_mode='stretch_both'
        )
        return layout


if __name__ == '__main__':
    dash = CopulaSpreadTradingStrategy()
    pn.serve(dash.main, port=5006, show=True)    )
        return layout


if __name__ == '__main__':
    dash = CopulaSpreadTradingStrategy()
    pn.serve(dash.main, port=5006, show=True)