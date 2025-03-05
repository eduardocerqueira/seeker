#date: 2025-03-05T17:02:28Z
#url: https://api.github.com/gists/4f0c45603c39a2af7fa5719630a46ddf
#owner: https://api.github.com/users/Clement1nes

import os
import numpy as np
import threading
import pandas as pd
import panel as pn
import param
from datetime import datetime, timedelta
import holoviews as hv
from bokeh.models import HoverTool
from scipy.stats import pearsonr, norm
from holoviews import opts
from binance.client import Client
from sklearn.covariance import GraphicalLassoCV
from correlation_analysis import CorrelationAnalysis
from beta_analysis import BetaDistributionAnalysis
from statsmodels.tsa.stattools import grangercausalitytests, coint
from dashboard_config import STYLE_CONFIG, CUSTOM_CSS
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from arch import arch_model

pn.extension('tabulator', sizing_mode="stretch_width")
hv.extension('bokeh')


# Binance API keys
api_key = 'CHANGE ME'
api_secret = "**********"
client = "**********"

pd.options.display.float_format = '{:.8f}'.format

# Cache dictionary to store fetched data
data_cache = {}
cache_expiry = timedelta(minutes=5)  # Data expires after 5 minutes


class DashboardBase:
    def create_nav_links(self):
        return pn.Row(
            pn.pane.HTML(f'''
                <div style="display: flex; gap: 10px;">
                    <a href="/" class="nav-link">Main Dashboard</a>
                    <a href="/correlation" class="nav-link">Correlation Analysis</a>
                    <a href="/beta" class="nav-link">Beta Distribution</a>
                </div>
            '''),
            align='end',
            styles={'background': STYLE_CONFIG['background_color']}
        )

    def create_header(self, title):
        return pn.Column(
            pn.pane.HTML(f'''
                <div class="dashboard-header">
                    <h1 style="color: {STYLE_CONFIG['secondary_color']}; 
                             font-family: {STYLE_CONFIG['font_family']}; 
                             margin: 0;">
                        {title}
                    </h1>
                </div>
            '''),
            sizing_mode='stretch_width'
        )



# Function to get historical Spot market data (with caching)
def get_spot_historical_data(symbol, interval='1m', limit=500):
    cache_key = ('spot', symbol, interval)
    current_time = datetime.now()
    # Check if data is in cache and not expired
    if cache_key in data_cache:
        cached_time, data = data_cache[cache_key]
        if current_time - cached_time < cache_expiry:
            return data
    try:
        candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"Error fetching spot data for {symbol} at interval {interval}: {e}")
        return pd.DataFrame()
    if not candles:
        print(f"No spot data returned for {symbol} at interval {interval}")
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    # Store data in cache
    data_cache[cache_key] = (current_time, df)
    return df


def create_market_maker_volume_plot(historical_data):
    """Create a plot showing market maker volume trends"""
    # Calculate market maker metrics
    df = historical_data.copy()
    df['mm_volume'] = df.apply(
        lambda row: row['volume'] if row['close'] > row['open'] else -row['volume'],
        axis=1
    )

    # Calculate rolling metrics
    window = 20  # Adjustable window size
    df['mm_volume_ma'] = df['mm_volume'].abs().rolling(window=window).mean()
    df['mm_volume_trend'] = df['mm_volume'].rolling(window=window).sum()

    # Create volume plot
    volume_plot = hv.Bars(df, 'timestamp', 'mm_volume').opts(
        color=hv.dim('mm_volume').apply(lambda x: 'green' if x > 0 else 'red'),
        alpha=0.6,
        width=800,
        height=200,
        tools=['hover'],
        ylabel='Market Maker Volume',
        title='Market Maker Volume'
    )

    # Create trend line
    trend_plot = hv.Curve(df, 'timestamp', 'mm_volume_trend').opts(
        color='blue',
        line_width=2,
        tools=['hover']
    )

    # Create moving average line
    ma_plot = hv.Curve(df, 'timestamp', 'mm_volume_ma').opts(
        color='yellow',
        line_width=2,
        tools=['hover']
    )

    # Combine plots
    combined_plot = (volume_plot * trend_plot * ma_plot).opts(
        shared_axes=True,
        legend_position='top_right',
        show_grid=True,
        toolbar='above',
        tools=['hover']
    )

    return combined_plot

# Function to get historical Perpetual Futures data (with caching)
def get_perpetual_historical_data(symbol, interval='1m', limit=500):
    cache_key = ('perpetual', symbol, interval)
    current_time = datetime.now()
    # Check if data is in cache and not expired
    if cache_key in data_cache:
        cached_time, data = data_cache[cache_key]
        if current_time - cached_time < cache_expiry:
            return data
    try:
        futures_candles = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"Error fetching perpetual data for {symbol} at interval {interval}: {e}")
        return pd.DataFrame()
    if not futures_candles:
        print(f"No perpetual data returned for {symbol} at interval {interval}")
        return pd.DataFrame()
    df = pd.DataFrame(futures_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    # Store data in cache
    data_cache[cache_key] = (current_time, df)
    return df

# Function to detect volumetric order blocks
def identify_volumetric_order_blocks(data, volume_threshold=1.5):
    if data.empty:
        return pd.DataFrame()
    avg_volume = data['volume'].mean()
    significant_volume = data[data['volume'] >= avg_volume * volume_threshold]
    order_blocks = []
    for index, row in significant_volume.iterrows():
        order_blocks.append({
            'Time': index,
            'Price': row['close'],
            'Volume': row['volume']
        })
    return pd.DataFrame(order_blocks)

# Function to detect and list market maker trades
def identify_market_maker_trades(data, volume_threshold=1.5):
    if data.empty:
        return pd.DataFrame()

    # Create a copy of the data to avoid modifying the original
    df = data.copy()

    # Calculate average volume and identify significant trades
    avg_volume = df['volume'].mean()
    mask = df['volume'] >= avg_volume * volume_threshold

    if not mask.any():
        return pd.DataFrame()

    # Create the significant trades dataframe
    significant_trades = df[mask].copy()

    # Reset the index to turn the timestamp into a column
    significant_trades.reset_index(inplace=True)

    # Calculate trade type
    significant_trades['Trade Type'] = significant_trades.apply(
        lambda row: "Buy" if row['close'] > row['open'] else "Sell",
        axis=1
    )

    # Format price and volumes
    significant_trades['Price'] = significant_trades['close'].astype(float)
    significant_trades['Asset Volume'] = significant_trades['volume'].astype(float)
    significant_trades['USD Volume'] = (significant_trades['Asset Volume'] * significant_trades['Price']).round(2)

    # Select and order the columns we need
    result = significant_trades[['timestamp', 'Price', 'Asset Volume', 'USD Volume', 'Trade Type']].copy()
    result.rename(columns={'timestamp': 'Time'}, inplace=True)

    # Ensure all numeric columns are float
    numeric_columns = ['Price', 'Asset Volume', 'USD Volume']
    result[numeric_columns] = result[numeric_columns].astype(float)

    # Sort by time
    result.sort_values('Time', inplace=True)

    return result

# Function to create liquidity heatmap
def create_liquidity_heatmap(data):
    if data.empty:
        return None

    # Create volume profile
    price_min = data['low'].min()
    price_max = data['high'].max()
    n_bins = 50

    # Create price bins
    bins = np.linspace(price_min, price_max, n_bins)
    volumes = np.zeros(len(bins) - 1)

    # Accumulate volume for each candle across its price range
    for _, row in data.iterrows():
        low_idx = np.digitize(row['low'], bins) - 1
        high_idx = np.digitize(row['high'], bins) - 1
        vol_per_bin = row['volume'] / (high_idx - low_idx + 1) if high_idx >= low_idx else 0
        volumes[low_idx:high_idx + 1] += vol_per_bin

    # Normalize volumes
    max_volume = volumes.max()
    if max_volume > 0:
        volumes = volumes / max_volume

    # Create heatmap data
    heatmap_data = pd.DataFrame({
        'price': bins[:-1],
        'volume': volumes
    })

    return heatmap_data

# Function to create volume profile
# Function to create volume profile
def create_volume_profile(data, bins=50):
    if data.empty:
        return None

    # Define price bins
    price_min = data['low'].min()
    price_max = data['high'].max()
    price_bins = np.linspace(price_min, price_max, bins)

    # Calculate histogram of volumes over price bins
    volume_profile, edges = np.histogram(
        (data['high'] + data['low']) / 2,  # Use average price for each candle
        bins=price_bins,
        weights=data['volume']  # Weight by volume
    )

    # Calculate total volume and percentage of total
    total_volume = np.sum(volume_profile)
    volume_percentages = (volume_profile / total_volume) * 100

    # Create DataFrame for plotting
    volume_profile_df = pd.DataFrame({
        'Price': edges[:-1],
        'Volume': volume_profile,
        'VolumePercent': volume_percentages,  # Percentage of total volume
        'TotalVolume': total_volume  # Store total volume for reference
    })

    return volume_profile_df

# Function to create interactive plot with Volume Profile
def create_interactive_plot(historical_data, title, volume_threshold=1.5,
                            show_buy_trades=True, show_sell_trades=True,
                            sort_by_volume=False, buy_color='green', sell_color='red',
                            order_block_color='orange', show_liquidity_heatmap=False,
                            show_volume_profile=False):
    if historical_data.empty:
        return hv.Curve([]).opts(title=title), pd.DataFrame()

    plot_elements = []

    # Add liquidity heatmap if enabled
    if show_liquidity_heatmap:
        # Create liquidity heatmap
        heatmap_data = create_liquidity_heatmap(historical_data)
        if heatmap_data is not None:
            heatmap = hv.Image(
                (historical_data.index.values, heatmap_data['price'], heatmap_data['volume']),
                ['Time', 'Price'],
                'Volume'
            ).opts(
                cmap='Blues',
                alpha=0.6,
                tools=['hover'],
                colorbar=True,
                width=800,
                height=400
            )
            plot_elements.append(heatmap)

    # Base price line
    price_line = hv.Curve((historical_data.index, historical_data['close']), 'Time', 'Price').opts(
        color='blue', line_width=2, tools=['hover'], height=400, responsive=True)
    plot_elements.append(price_line)

    # Identify volumetric order blocks
    volumetric_blocks = identify_volumetric_order_blocks(historical_data, volume_threshold)
    order_block_lines = []
    for _, row in volumetric_blocks.iterrows():
        hline = hv.HLine(row['Price']).opts(
            color=order_block_color, line_dash='dashed', line_width=2,
            alpha=0.5)
        order_block_lines.append(hline)
    order_blocks = hv.Overlay(order_block_lines)
    plot_elements.append(order_blocks)

    # Identify market maker trades
    filtered_trades = identify_market_maker_trades(historical_data, volume_threshold)

    # Create scatter plot for trades with interactive tooltips
    if not filtered_trades.empty:
        hover_opts = {
            'tools': ['hover'],
            'hover_tooltips': [
                ('Time', '@Time{%F %T}'),
                ('Price', '@Price{0,0.00000000}'),
                ('Asset Volume', '@{Asset Volume}{0,0.00000000}'),
                ('USD Volume', '$@{USD Volume}{0,0.00}'),
                ('Type', '@{Trade Type}')
            ],
            'hover_formatters': {'@Time': 'datetime'}
        }

        if show_buy_trades:
            buy_trades = filtered_trades[filtered_trades['Trade Type'] == 'Buy']
            if not buy_trades.empty:
                buy_scatter = hv.Scatter(
                    data=buy_trades,
                    kdims=['Time'],
                    vdims=['Price', 'Asset Volume', 'USD Volume', 'Trade Type'],
                    label='Buy Trades'
                ).opts(
                    color=buy_color,
                    size=8,
                    marker='triangle',
                    **hover_opts
                )
                plot_elements.append(buy_scatter)

        if show_sell_trades:
            sell_trades = filtered_trades[filtered_trades['Trade Type'] == 'Sell']
            if not sell_trades.empty:
                sell_scatter = hv.Scatter(
                    data=sell_trades,
                    kdims=['Time'],
                    vdims=['Price', 'Asset Volume', 'USD Volume', 'Trade Type'],
                    label='Sell Trades'
                ).opts(
                    color=sell_color,
                    size=8,
                    marker='inverted_triangle',
                    **hover_opts
                )
                plot_elements.append(sell_scatter)

    # Combine all elements into an Overlay
    main_plot = hv.Overlay(plot_elements).opts(
        title=title,
        legend_position='top_left',
        show_legend=True,
        responsive=True
    )

    # Add Volume Profile if enabled
    if show_volume_profile:
        volume_profile_df = create_volume_profile(historical_data)
        if volume_profile_df is not None:
            # Calculate the x-axis range for scaling
            time_range = historical_data.index.max() - historical_data.index.min()
            min_time = historical_data.index.min()

            # Create scaled x-coordinates for the bars
            volume_profile_df['x0'] = min_time
            volume_profile_df['x1'] = min_time + time_range * 0.2 * (volume_profile_df['VolumePercent'] / 100)

            # Create volume profile bars
            volume_bars = hv.Segments(
                data=volume_profile_df,
                kdims=['x0', 'Price', 'x1', 'Price']
            ).opts(
                color='gray',
                line_width=2,
                alpha=0.5,
                tools=['hover'],
                hover_tooltips=[
                    ('Price Level', '@Price{0,0.0000}'),
                    ('Volume at Level', '@Volume{0,0.0}'),
                    ('% of Total Volume', '@VolumePercent{0.00}%'),
                    ('Total Volume', '@TotalVolume{0,0.0}')
                ]
            )

            # Combine main plot with volume profile
            combined_plot = (main_plot * volume_bars).opts(
                show_grid=True
            )
        else:
            combined_plot = main_plot
    else:
        combined_plot = main_plot

    return combined_plot, filtered_trades

# Function to calculate correlations
def calculate_correlations(data1, data2, window_size=30):
    """Calculate static and rolling correlations between two price series"""
    # Align the two series
    df = pd.DataFrame({
        'price1': data1['close'],
        'price2': data2['close']
    }).dropna()

    # Calculate static correlation
    static_corr, p_value = pearsonr(df['price1'], df['price2'])

    # Calculate rolling correlation
    rolling_corr = df['price1'].rolling(window=window_size).corr(df['price2'])

    return static_corr, rolling_corr, df

def calculate_beta_metrics(data1, data2, risk_free_rate=0.02):
    """Calculate beta and related risk metrics"""
    # Calculate returns
    returns1 = data1['close'].pct_change().dropna()
    returns2 = data2['close'].pct_change().dropna()

    # Calculate beta
    covariance = returns1.cov(returns2)
    variance = returns2.var()
    beta = covariance / variance

    # Calculate additional metrics
    volatility1 = returns1.std() * np.sqrt(365)  # Annualized volatility
    volatility2 = returns2.std() * np.sqrt(365)
    sharpe1 = (returns1.mean() * 365 - risk_free_rate) / volatility1
    sharpe2 = (returns2.mean() * 365 - risk_free_rate) / volatility2

    return {
        'beta': beta,
        'volatility_token1': "**********"
        'volatility_token2': "**********"
        'sharpe_token1': "**********"
        'sharpe_token2': "**********"
    }

def analyze_granger_causality(data1, data2, max_lags=5):
    """Test for Granger causality between two price series"""
    # Prepare data
    df = pd.DataFrame({
        'price1': data1['close'],
        'price2': data2['close']
    }).pct_change().dropna()

    # Test both directions
    results_1_2 = grangercausalitytests(df[['price1', 'price2']], maxlag=max_lags, verbose=False)
    results_2_1 = grangercausalitytests(df[['price2', 'price1']], maxlag=max_lags, verbose=False)

    return results_1_2, results_2_1

def analyze_cointegration(data1, data2):
    """Test for cointegration between two price series"""
    _, pvalue, _ = coint(data1['close'], data2['close'])
    return {
        'cointegrated': pvalue < 0.05,
        'p_value': pvalue
    }


def calculate_dtw(data1, data2):
    """Calculate Dynamic Time Warping distance between normalized price series"""
    try:
        # Extract prices and ensure they're 1-D arrays
        prices1 = data1['close'].values.astype(float)  # Convert to float array
        prices2 = data2['close'].values.astype(float)  # Convert to float array

        # Remove any NaN values
        prices1 = prices1[~np.isnan(prices1)]
        prices2 = prices2[~np.isnan(prices2)]

        # Ensure equal lengths
        min_length = min(len(prices1), len(prices2))
        prices1 = prices1[:min_length]
        prices2 = prices2[:min_length]

        # Normalize using sklearn's preprocessing
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        norm1 = scaler.fit_transform(prices1.reshape(-1, 1)).ravel()
        norm2 = scaler.fit_transform(prices2.reshape(-1, 1)).ravel()

        # Convert to list format for fastdtw
        norm1_list = norm1.tolist()
        norm2_list = norm2.tolist()

        # Calculate DTW distance with lists
        distance, path = fastdtw(norm1_list, norm2_list, dist=euclidean)

        # Calculate additional metrics
        mean_dist = distance / len(norm1)
        correlation = np.corrcoef(norm1, norm2)[0, 1]

        return {
            'success': True,
            'distance': distance,
            'path': path,
            'mean_dist': mean_dist,
            'correlation': correlation,
            'length': len(norm1)
        }

    except Exception as e:
        print(f"DTW calculation error: {str(e)}")
        print(f"Data types - prices1: {type(prices1)}, prices2: {type(prices2)}")
        print(f"Data types - norm1: {type(norm1)}, norm2: {type(norm2)}")
        print(f"Shapes - prices1: {prices1.shape if 'prices1' in locals() else 'not created'}")
        print(f"Shapes - prices2: {prices2.shape if 'prices2' in locals() else 'not created'}")
        return {
            'success': False
        }


# Function to perform risk analysis and plot the efficient frontier
def perform_risk_analysis(data1, data2, risk_free_rate=0.02):
    """Perform portfolio risk analysis and plot the efficient frontier"""
    try:
        # Calculate daily returns
        returns1 = data1['close'].pct_change().dropna()
        returns2 = data2['close'].pct_change().dropna()

        # Align the returns
        returns = pd.DataFrame({
            'returns1': returns1,
            'returns2': returns2
        }).dropna()

        mean_returns = returns.mean() * 365  # Annualized returns
        cov_matrix = returns.cov() * 365     # Annualized covariance

        # Generate portfolio weights
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(2)
            weights /= np.sum(weights)
            weights_record.append(weights)

            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            results[0, i] = portfolio_std
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std

        # Convert results to DataFrame
        results_df = pd.DataFrame(results.T, columns=['std_dev', 'return', 'sharpe_ratio'])
        weights_df = pd.DataFrame(weights_record, columns=[data1.name, data2.name])
        portfolio_df = pd.concat([results_df, weights_df], axis=1)

        # Identify optimal portfolios
        min_vol_port = portfolio_df.iloc[portfolio_df['std_dev'].idxmin()]
        max_sharpe_port = portfolio_df.iloc[portfolio_df['sharpe_ratio'].idxmax()]

        # Create the scatter plot for all portfolios using holoviews
        scatter = hv.Points(
            data=portfolio_df,
            kdims=['std_dev', 'return'],
            vdims=['sharpe_ratio']
        ).opts(
            color='sharpe_ratio',
            colorbar=True,
            cmap='Viridis',
            width=1000,
            height=3000,
            tools=['hover'],
            title='Efficient Frontier',
            xlabel='Risk (Standard Deviation)',
            ylabel='Expected Return',
            size=5
        )

        # Add minimum volatility point
        min_vol_point = hv.Points(
            data=pd.DataFrame({
                'std_dev': [min_vol_port['std_dev']],
                'return': [min_vol_port['return']]
            })
        ).opts(
            color='red',
            size=10,
            marker='star'
        )

        # Add maximum Sharpe ratio point
        max_sharpe_point = hv.Points(
            data=pd.DataFrame({
                'std_dev': [max_sharpe_port['std_dev']],
                'return': [max_sharpe_port['return']]
            })
        ).opts(
            color='gold',
            size=10,
            marker='diamond'
        )

        # Combine plots using Overlay
        combined_plot = (scatter * min_vol_point * max_sharpe_point).opts(
            title=f'Efficient Frontier: {data1.name} vs {data2.name}',
            legend_position='top_right'
        )

        return combined_plot, min_vol_port, max_sharpe_port

    except Exception as e:
        print(f"Error in perform_risk_analysis: {str(e)}")
        return None, None, None



class BetaDistributionAnalysis(param.Parameterized):
    tokens = "**********"=['ETHUSDT'], objects=['ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT'])
    custom_token = "**********"='')
    timeframe = param.Selector(default='1h', objects=['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
    window_size = param.Integer(default=30, bounds=(5, 100))
    update_frequency = param.Integer(default=0, bounds=(0, 3600))

    def __init__(self, **params):
        super().__init__(**params)
        self.plot_pane = pn.pane.HoloViews(sizing_mode='stretch_width')
        self.stats_pane = pn.pane.Markdown(sizing_mode='stretch_width')
        self.periodic_callback = None
        self.available_tokens = "**********"
        self.param.tokens.objects = "**********"
        self.custom_tokens = "**********"
        self.update_data()
        self.update_periodic_callback()

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"a "**********"v "**********"a "**********"i "**********"l "**********"a "**********"b "**********"l "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
        return [
            'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT',
            'MATICUSDT', 'SOLUSDT', 'DOTUSDT', 'UNIUSDT', 'LTCUSDT', 'AVAXUSDT',
            'ATOMUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT'
        ]

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"v "**********"a "**********"l "**********"i "**********"d "**********"a "**********"t "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
        """Validate if a token pair exists on Binance"""
        try:
            test_data = "**********"=1)
            return not test_data.empty
        except Exception:
            return False

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"a "**********"d "**********"d "**********"_ "**********"c "**********"u "**********"s "**********"t "**********"o "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
        """Add a custom token if it's valid"""
        token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********". "**********"e "**********"n "**********"d "**********"s "**********"w "**********"i "**********"t "**********"h "**********"( "**********"' "**********"U "**********"S "**********"D "**********"T "**********"' "**********") "**********": "**********"
            token = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"v "**********"a "**********"l "**********"i "**********"d "**********"a "**********"t "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
            self.custom_tokens.add(token)
            current_objects = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"_ "**********"o "**********"b "**********"j "**********"e "**********"c "**********"t "**********"s "**********": "**********"
                current_objects.append(token)
                self.param.tokens.objects = "**********"
                # Automatically select the new token for display in the plot
                self.tokens = "**********"
            return True, f"Successfully added {token}"
        return False, f"Invalid token pair: "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"c "**********"a "**********"l "**********"c "**********"u "**********"l "**********"a "**********"t "**********"e "**********"_ "**********"r "**********"o "**********"l "**********"l "**********"i "**********"n "**********"g "**********"_ "**********"b "**********"e "**********"t "**********"a "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"b "**********"t "**********"c "**********"_ "**********"d "**********"a "**********"t "**********"a "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"d "**********"a "**********"t "**********"a "**********") "**********": "**********"
        """Calculate rolling beta values for a token relative to BTC"""
        btc_returns = btc_data['close'].pct_change().dropna()
        token_returns = "**********"

        # Align the data
        aligned_data = "**********"=1).dropna()
        btc_returns = aligned_data.iloc[:, 0]
        token_returns = aligned_data.iloc[: "**********"

        # Calculate rolling beta
        rolling_cov = "**********"=self.window_size).cov(btc_returns)
        rolling_var = btc_returns.rolling(window=self.window_size).var()
        rolling_beta = rolling_cov / rolling_var

        return rolling_beta.dropna()

    @param.depends('tokens', 'timeframe', 'window_size', 'update_frequency', 'custom_token', watch= "**********"
    def update_data(self):
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"s "**********"t "**********"o "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            success, message = "**********"
            self.custom_token = "**********"
            print(message)  # Optional: Print feedback in console for debugging

        try:
            # Fetch BTC data
            btc_data = get_spot_historical_data('BTCUSDT', self.timeframe)
            if btc_data.empty:
                self.plot_pane.object = None
                self.stats_pane.object = "Error: Unable to fetch BTC data"
                return

            # Initialize storage for beta values and statistics
            token_betas = "**********"
            beta_stats = []

            # Calculate betas for each token
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
                token_data = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"d "**********"a "**********"t "**********"a "**********". "**********"e "**********"m "**********"p "**********"t "**********"y "**********": "**********"
                    beta_values = "**********"
                    token_betas[token] = "**********"

                    # Calculate statistics
                    mean_beta = np.mean(beta_values)
                    std_beta = np.std(beta_values)
                    beta_stats.append({
                        'Token': "**********"
                        'Mean Beta': mean_beta,
                        'Std Dev': std_beta,
                        'Min Beta': np.min(beta_values),
                        'Max Beta': np.max(beta_values)
                    })

            # Create the distribution plot
            dist_plot = hv.Curve([(1, 0)], 'Beta', 'Density', label='BTC (β=1)').opts(
                color='black', line_width=3)

            # Define color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Add distribution curves for each token
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"b "**********"e "**********"t "**********"a "**********"_ "**********"v "**********"a "**********"l "**********"u "**********"e "**********"s "**********") "**********", "**********"  "**********"c "**********"o "**********"l "**********"o "**********"r "**********"  "**********"i "**********"n "**********"  "**********"z "**********"i "**********"p "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"b "**********"e "**********"t "**********"a "**********"s "**********". "**********"i "**********"t "**********"e "**********"m "**********"s "**********"( "**********") "**********", "**********"  "**********"c "**********"o "**********"l "**********"o "**********"r "**********"s "**********") "**********": "**********"
                mean_beta = np.mean(beta_values)
                std_beta = np.std(beta_values)

                # Create x-axis points for the normal distribution
                x = np.linspace(max(0, mean_beta - 3 * std_beta), mean_beta + 3 * std_beta, 200)
                pdf = norm.pdf(x, mean_beta, std_beta)

                # Add the distribution curve
                token_curve = "**********"
                                       label=f'{token} (β={mean_beta: "**********"
                    color=color, line_width=2)
                dist_plot = "**********"

            # Style the plot
            dist_plot = dist_plot.opts(
                title='Beta Distribution Relative to Bitcoin',
                width=800,
                height=400,
                legend_position='right',
                xlabel='Beta Coefficient (β)',
                ylabel='Probability Density',
                show_grid=True,
                fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12},
                tools=['hover']
            )

            # Create statistics summary
            stats_md = """
            ### Beta Analysis Statistics

            | Token | Mean β | Std Dev | Min β | Max β |
            |-------|--------|---------|-------|-------|
            """

            for stat in beta_stats:
                stats_md += f"| {stat['Token']} | {stat['Mean Beta']: "**********":.3f} | {stat['Min Beta']:.3f} | {stat['Max Beta']:.3f} |\n"

            stats_md += """

            **Interpretation:**
            - β = 1: Same volatility as Bitcoin
            - β > 1: More volatile than Bitcoin
            - β < 1: Less volatile than Bitcoin
            """

            # Update display
            self.plot_pane.object = dist_plot
            self.stats_pane.object = stats_md

        except Exception as e:
            print(f"Error in beta distribution analysis: {str(e)}")
            self.plot_pane.object = None
            self.stats_pane.object = f"Error during analysis: {str(e)}"

    @param.depends('update_frequency', watch=True)
    def update_periodic_callback(self):
        if self.periodic_callback:
            self.periodic_callback.stop()
            self.periodic_callback = None
        if self.update_frequency > 0:
            self.periodic_callback = pn.state.add_periodic_callback(
                self.update_data, period=self.update_frequency * 1000)

    def view(self):
        # Create navigation
        nav_links = pn.Row(
            pn.pane.HTML('<a href="/" style="font-size:15px; margin:10px;">Main Dashboard</a>'),
            align='end'
        )

        # Create custom token input
        custom_token_input = "**********"
            name= "**********"
            placeholder= "**********"
            value=''
        )

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"a "**********"d "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"c "**********"a "**********"l "**********"l "**********"b "**********"a "**********"c "**********"k "**********"( "**********"e "**********"v "**********"e "**********"n "**********"t "**********") "**********": "**********"
            if event.new and event.new.strip():
                success, message = "**********"
                if success:
                    custom_token_input.value = "**********"
                custom_token_input.name = "**********"

        custom_token_input.param.watch(add_token_callback, 'value')

        # Create widgets
        tokens_select = "**********"
            name= "**********"
            value= "**********"
            options= "**********"
            sizing_mode='stretch_width'
        )
        tokens_select.link(self, value= "**********"

        timeframe_select = pn.widgets.Select(
            name='Timeframe',
            options=self.param.timeframe.objects,
            value=self.timeframe,
            sizing_mode='stretch_width'
        )
        timeframe_select.link(self, value='timeframe')

        window_slider = pn.widgets.IntSlider(
            name='Rolling Window Size',
            start=5,
            end=100,
            step=1,
            value=self.window_size,
            sizing_mode='stretch_width'
        )
        window_slider.link(self, value='window_size')

        update_freq = pn.widgets.IntInput(
            name='Update Frequency (seconds)',
            value=self.update_frequency,
            step=10,
            sizing_mode='stretch_width'
        )
        update_freq.link(self, value='update_frequency')

        # Create help text
        help_text = pn.pane.Markdown("""
        ### How to Use
        1. Enter any token symbol (e.g., "ETH" or "ETHUSDT") in the custom token input
        2. Select one or more tokens from the dropdown for analysis
        3. Adjust timeframe and window size as needed
        4. The graph will show beta distribution relative to Bitcoin
        """)

        # Create layout
        controls = pn.Column(
            pn.pane.Markdown("### Beta Distribution Controls"),
            help_text,
            custom_token_input,
            tokens_select,
            timeframe_select,
            window_slider,
            update_freq,
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px'}
        )

        # Return template with navigation
        return pn.template.FastListTemplate(
            title='Beta Distribution Analysis',
            header=[nav_links],
            sidebar=[controls],
            main=[self.stats_pane, self.plot_pane],
            accent_base_color='#0A5A9C',
            header_background='#0A5A9C',
            theme_toggle=False,
            main_max_width='80%'
        )

# CryptoDashboard class
class CryptoDashboard(param.Parameterized):
    symbol = param.String(default='BTCUSDT')
    timeframe = param.Selector(default='1m', objects=['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
    volume_threshold = param.Number(default=1.5, bounds=(1, 10))
    show_buy_trades = param.Boolean(default=True)
    show_sell_trades = param.Boolean(default=True)
    sort_by_volume = param.Boolean(default=False)
    show_liquidity_heatmap = param.Boolean(default=False)
    show_volume_profile = param.Boolean(default=False)
    update_frequency = param.Integer(default=0, bounds=(0, 3600))
    buy_color = param.Color(default='#00FF00')
    sell_color = param.Color(default='#FF0000')
    order_block_color = param.Color(default='#FFA500')

    def __init__(self, **params):
        super().__init__(**params)
        # Initialize main components
        self.plot_pane = pn.pane.HoloViews(sizing_mode='stretch_width', min_height=600)
        self.spot_table = pn.widgets.DataFrame(sizing_mode='stretch_width', min_height=200)
        self.perpetual_table = pn.widgets.DataFrame(sizing_mode='stretch_width', min_height=200)
        self.main_content = pn.Column(
            pn.pane.Markdown("""
                    # Cryptocurrency Trading Analysis Dashboard

                    This dashboard provides advanced analysis tools for cryptocurrency trading:

                    ## Features:

                    ### Market Making Analysis
                    - Real-time spot and perpetual futures data
                    - Volume threshold analysis
                    - Liquidity heatmap
                    - Volume profile visualization

                    ### Correlation Analysis
                    - Real-time correlation heatmap
                    - Pairs trading opportunities
                    - Risk metrics and hedging ratios
                    - Funding rate adjusted strategies

                    ### Beta Distribution Analysis
                    - Dynamic beta calculation
                    - Up/Down market beta analysis
                    - Volatility regime detection
                    - Optimal hedge ratio calculation

                    Select an analysis type from the navigation menu above.
                    """),
            sizing_mode='stretch_width'
        )

        self.update_data()
        self.periodic_callback = None
        self.update_periodic_callback()

    @param.depends('symbol', 'timeframe', 'volume_threshold', 'show_buy_trades',
                   'show_sell_trades', 'sort_by_volume', 'buy_color', 'sell_color',
                   'order_block_color', 'show_liquidity_heatmap', 'show_volume_profile', watch=True)
    def update_data(self):
        try:
            # Fetch data
            spot_data = get_spot_historical_data(self.symbol, self.timeframe)
            perpetual_data = get_perpetual_historical_data(self.symbol, self.timeframe)

            # Ensure data is in correct format
            if not isinstance(spot_data, pd.DataFrame):
                spot_data = pd.DataFrame(spot_data)
            if not isinstance(perpetual_data, pd.DataFrame):
                perpetual_data = pd.DataFrame(perpetual_data)

            # Create plots with heatmap and volume profile
            try:
                spot_plot, spot_trades = create_interactive_plot(
                    spot_data,
                    f'Spot Market: {self.symbol} with Order Blocks',
                    self.volume_threshold,
                    self.show_buy_trades,
                    self.show_sell_trades,
                    self.sort_by_volume,
                    self.buy_color,
                    self.sell_color,
                    self.order_block_color,
                    self.show_liquidity_heatmap,
                    self.show_volume_profile
                )

                perpetual_plot, perpetual_trades = create_interactive_plot(
                    perpetual_data,
                    f'Perpetual Futures: {self.symbol} with Order Blocks',
                    self.volume_threshold,
                    self.show_buy_trades,
                    self.show_sell_trades,
                    self.sort_by_volume,
                    self.buy_color,
                    self.sell_color,
                    self.order_block_color,
                    self.show_liquidity_heatmap,
                    self.show_volume_profile
                )

                # Ensure trades data is in correct format
                if not isinstance(spot_trades, pd.DataFrame):
                    spot_trades = pd.DataFrame(spot_trades)
                if not isinstance(perpetual_trades, pd.DataFrame):
                    perpetual_trades = pd.DataFrame(perpetual_trades)

                # Update components with error handling
                if spot_plot is not None:
                    self.plot_pane.object = (spot_plot + perpetual_plot).cols(1)
                else:
                    self.plot_pane.object = None

                # Update tables with proper formatting
                if not spot_trades.empty:
                    self.spot_table.value = spot_trades.copy()
                else:
                    self.spot_table.value = pd.DataFrame()

                if not perpetual_trades.empty:
                    self.perpetual_table.value = perpetual_trades.copy()
                else:
                    self.perpetual_table.value = pd.DataFrame()

            except Exception as plot_error:
                print(f"Error creating plots: {str(plot_error)}")
                self.plot_pane.object = None
                self.spot_table.value = pd.DataFrame()
                self.perpetual_table.value = pd.DataFrame()

        except Exception as e:
            print(f"Error updating dashboard: {str(e)}")
            self.plot_pane.object = None
            self.spot_table.value = pd.DataFrame()
            self.perpetual_table.value = pd.DataFrame()

    def update_periodic_callback(self):
        if self.periodic_callback:
            self.periodic_callback.stop()
            self.periodic_callback = None
        if self.update_frequency > 0:
            self.periodic_callback = pn.state.add_periodic_callback(
                self.update_data, period=self.update_frequency * 1000
            )

    def view(self):
        # Create all widgets
        symbol_input = pn.widgets.TextInput(
            name="Symbol",
            value=self.symbol,
            sizing_mode='stretch_width'
        )
        symbol_input.link(self, value='symbol')

        timeframe_select = pn.widgets.Select(
            name='Timeframe',
            options=self.param.timeframe.objects,
            value=self.timeframe,
            sizing_mode='stretch_width'
        )
        timeframe_select.link(self, value='timeframe')

        volume_slider = pn.widgets.FloatSlider(
            name='Volume Threshold',
            start=1,
            end=10,
            step=0.1,
            value=self.volume_threshold,
            sizing_mode='stretch_width'
        )
        volume_slider.link(self, value='volume_threshold')

        # Call the helper methods to create control groups
        trading_controls = self._create_trading_controls()
        display_controls = self._create_display_controls()
        color_controls = self._create_color_controls()
        update_controls = self._create_update_controls()

        # Combine all sections for sidebar
        sidebar = pn.Column(
            pn.Column(
                pn.pane.Markdown("### Basic Settings"),
                symbol_input,
                timeframe_select,
                volume_slider,
                margin=(0, 0, 20, 0)
            ),
            trading_controls,
            display_controls,
            color_controls,
            update_controls,
            sizing_mode='stretch_width',
            margin=(10, 5)
        )

        # Create navigation
        nav_links = pn.Row(
            pn.pane.HTML("""
                <a href="/" style="
                    font-size: 15px;
                    margin: 10px;
                    padding: 8px 15px;
                    background-color: #0A5A9C;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                ">Dashboard</a>
            """),
            pn.pane.HTML("""
                <a href="/correlation" style="
                    font-size: 15px;
                    margin: 10px;
                    padding: 8px 15px;
                    background-color: #0A5A9C;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                ">Correlation Analysis</a>
            """),
            pn.pane.HTML("""
                <a href="/beta" style="
                    font-size: 15px;
                    margin: 10px;
                    padding: 8px 15px;
                    background-color: #0A5A9C;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                ">Beta Distribution</a>
            """),
            align='end'
        )

        # Return template
        template = pn.template.FastListTemplate(
            title='Crypto Trading Analysis Dashboard',
            header=[nav_links],
            sidebar=[sidebar],
            main=[self.plot_pane, self.spot_table, self.perpetual_table],
            accent_base_color='#0A5A9C',
            header_background='#0A5A9C',
            theme_toggle=False,
            main_max_width='80%'
        )

        return template

    def _create_trading_controls(self):
        show_buy_trades = pn.widgets.Checkbox(
            name='Show Buy Trades', value=self.show_buy_trades
        )
        show_buy_trades.link(self, value='show_buy_trades')

        show_sell_trades = pn.widgets.Checkbox(
            name='Show Sell Trades', value=self.show_sell_trades
        )
        show_sell_trades.link(self, value='show_sell_trades')

        sort_volume = pn.widgets.Checkbox(
            name='Sort by Volume', value=self.sort_by_volume
        )
        sort_volume.link(self, value='sort_by_volume')

        return pn.Column(
            pn.pane.Markdown("### Trading Controls"),
            show_buy_trades,
            show_sell_trades,
            sort_volume,
            sizing_mode='stretch_width',
            margin=(0, 0, 20, 0)
        )

    def _create_display_controls(self):
        show_heatmap = pn.widgets.Checkbox(
            name='Show Liquidity Heatmap',
            value=self.show_liquidity_heatmap
        )
        show_heatmap.link(self, value='show_liquidity_heatmap')

        show_volume_profile = pn.widgets.Checkbox(
            name='Show Volume Profile',
            value=self.show_volume_profile
        )
        show_volume_profile.link(self, value='show_volume_profile')

        return pn.Column(
            pn.pane.Markdown("### Display Options"),
            show_heatmap,
            show_volume_profile,
            sizing_mode='stretch_width',
            margin=(0, 0, 20, 0)
        )

    def _create_color_controls(self):
        buy_color = pn.widgets.ColorPicker(
            name='Buy Trade Color',
            value=self.buy_color,
            width=150
        )
        buy_color.link(self, value='buy_color')

        sell_color = pn.widgets.ColorPicker(
            name='Sell Trade Color',
            value=self.sell_color,
            width=150
        )
        sell_color.link(self, value='sell_color')

        order_block_color = pn.widgets.ColorPicker(
            name='Order Block Color',
            value=self.order_block_color,
            width=150
        )
        order_block_color.link(self, value='order_block_color')

        return pn.Column(
            pn.pane.Markdown("### Color Settings"),
            buy_color,
            sell_color,
            order_block_color,
            sizing_mode='stretch_width',
            margin=(0, 0, 20, 0)
        )

    def _create_update_controls(self):
        update_freq = pn.widgets.IntInput(
            name='Update Frequency (seconds)',
            value=self.update_frequency,
            step=10,
            width=200
        )
        update_freq.link(self, value='update_frequency')

        return pn.Column(
            pn.pane.Markdown("### Update Settings"),
            update_freq,
            sizing_mode='stretch_width',
            margin=(0, 0, 20, 0)
        )


class BetaDistributionAnalysis(param.Parameterized):
    tokens = "**********"=['ETHUSDT'], objects=['ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT'])
    custom_token = "**********"='')
    timeframe = param.Selector(default='1h', objects=['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
    window_size = param.Integer(default=30, bounds=(5, 100))
    update_frequency = param.Integer(default=0, bounds=(0, 3600))

    def __init__(self, **params):
        super().__init__(**params)
        self.plot_pane = pn.pane.HoloViews(sizing_mode='stretch_width')
        self.stats_pane = pn.pane.Markdown(sizing_mode='stretch_width')
        self.periodic_callback = None
        self.available_tokens = "**********"
        self.param.tokens.objects = "**********"
        self.custom_tokens = "**********"
        self.update_data()
        self.update_periodic_callback()

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"a "**********"v "**********"a "**********"i "**********"l "**********"a "**********"b "**********"l "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
        return [
            'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT',
            'MATICUSDT', 'SOLUSDT', 'DOTUSDT', 'UNIUSDT', 'LTCUSDT', 'AVAXUSDT',
            'ATOMUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT'
        ]

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"v "**********"a "**********"l "**********"i "**********"d "**********"a "**********"t "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
        """Validate if a token pair exists on Binance"""
        try:
            test_data = "**********"=1)
            return not test_data.empty
        except Exception:
            return False

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"a "**********"d "**********"d "**********"_ "**********"c "**********"u "**********"s "**********"t "**********"o "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
        """Add a custom token if it's valid and trigger update for graph"""
        token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********". "**********"e "**********"n "**********"d "**********"s "**********"w "**********"i "**********"t "**********"h "**********"( "**********"' "**********"U "**********"S "**********"D "**********"T "**********"' "**********") "**********": "**********"
            token = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"v "**********"a "**********"l "**********"i "**********"d "**********"a "**********"t "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
            self.custom_tokens.add(token)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"p "**********"a "**********"r "**********"a "**********"m "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********". "**********"o "**********"b "**********"j "**********"e "**********"c "**********"t "**********"s "**********": "**********"
                self.param.tokens.objects = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
                self.tokens = "**********"
            return True, f"Successfully added {token}"
        return False, f"Invalid token pair: "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"c "**********"a "**********"l "**********"c "**********"u "**********"l "**********"a "**********"t "**********"e "**********"_ "**********"r "**********"o "**********"l "**********"l "**********"i "**********"n "**********"g "**********"_ "**********"b "**********"e "**********"t "**********"a "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"b "**********"t "**********"c "**********"_ "**********"d "**********"a "**********"t "**********"a "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"d "**********"a "**********"t "**********"a "**********") "**********": "**********"
        """Calculate rolling beta values for a token relative to BTC"""
        btc_returns = btc_data['close'].pct_change().dropna()
        token_returns = "**********"

        # Align the data
        aligned_data = "**********"=1).dropna()
        btc_returns = aligned_data.iloc[:, 0]
        token_returns = aligned_data.iloc[: "**********"

        # Calculate rolling beta
        rolling_cov = "**********"=self.window_size).cov(btc_returns)
        rolling_var = btc_returns.rolling(window=self.window_size).var()
        rolling_beta = rolling_cov / rolling_var

        return rolling_beta.dropna()

    @param.depends('tokens', 'timeframe', 'window_size', 'update_frequency', 'custom_token', watch= "**********"
    def update_data(self):
        # If there's a custom token, attempt to add it and reset the field
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"s "**********"t "**********"o "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            success, message = "**********"
            self.custom_token = "**********"
            print(message)  # Optional: print feedback for debugging

        # Fetch BTC data for baseline
        btc_data = get_spot_historical_data('BTCUSDT', self.timeframe)
        if btc_data.empty:
            self.plot_pane.object = None
            self.stats_pane.object = "Error: Unable to fetch BTC data"
            return

        # Initialize storage for beta values and statistics
        token_betas = "**********"
        beta_stats = []

        # Calculate betas for each selected token
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
            token_data = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"d "**********"a "**********"t "**********"a "**********". "**********"e "**********"m "**********"p "**********"t "**********"y "**********": "**********"
                beta_values = "**********"
                token_betas[token] = "**********"

                # Calculate statistics
                mean_beta = np.mean(beta_values)
                std_beta = np.std(beta_values)
                beta_stats.append({
                    'Token': "**********"
                    'Mean Beta': mean_beta,
                    'Std Dev': std_beta,
                    'Min Beta': np.min(beta_values),
                    'Max Beta': np.max(beta_values)
                })

        # Create the distribution plot
        dist_plot = hv.Curve([(1, 0)], 'Beta', 'Density', label='BTC (β=1)').opts(
            color='black', line_width=3)

        # Define color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Add distribution curves for each token
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"b "**********"e "**********"t "**********"a "**********"_ "**********"v "**********"a "**********"l "**********"u "**********"e "**********"s "**********") "**********", "**********"  "**********"c "**********"o "**********"l "**********"o "**********"r "**********"  "**********"i "**********"n "**********"  "**********"z "**********"i "**********"p "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"b "**********"e "**********"t "**********"a "**********"s "**********". "**********"i "**********"t "**********"e "**********"m "**********"s "**********"( "**********") "**********", "**********"  "**********"c "**********"o "**********"l "**********"o "**********"r "**********"s "**********") "**********": "**********"
            mean_beta = np.mean(beta_values)
            std_beta = np.std(beta_values)

            # Create x-axis points for the normal distribution
            x = np.linspace(max(0, mean_beta - 3 * std_beta), mean_beta + 3 * std_beta, 200)
            pdf = norm.pdf(x, mean_beta, std_beta)

            # Add the distribution curve
            token_curve = "**********"
                                   label=f'{token} (β={mean_beta: "**********"
                color=color, line_width=2)
            dist_plot = "**********"

        # Style the plot
        dist_plot = dist_plot.opts(
            title='Beta Distribution Relative to Bitcoin',
            width=800,
            height=400,
            legend_position='right',
            xlabel='Beta Coefficient (β)',
            ylabel='Probability Density',
            show_grid=True,
            fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12},
            tools=['hover']
        )

        # Create statistics summary
        stats_md = """
        ### Beta Analysis Statistics

        | Token | Mean β | Std Dev | Min β | Max β |
        |-------|--------|---------|-------|-------|
        """

        for stat in beta_stats:
            stats_md += f"| {stat['Token']} | {stat['Mean Beta']: "**********":.3f} | {stat['Min Beta']:.3f} | {stat['Max Beta']:.3f} |\n"

        stats_md += """

        **Interpretation:**
        - β = 1: Same volatility as Bitcoin
        - β > 1: More volatile than Bitcoin
        - β < 1: Less volatile than Bitcoin
        """

        # Update display
        self.plot_pane.object = dist_plot
        self.stats_pane.object = stats_md



    @param.depends('update_frequency', watch=True)
    def update_periodic_callback(self):
        if self.periodic_callback:
            self.periodic_callback.stop()
            self.periodic_callback = None
        if self.update_frequency > 0:
            self.periodic_callback = pn.state.add_periodic_callback(
                self.update_data, period=self.update_frequency * 1000)

    def view(self):
        # Create navigation
        nav_links = pn.Row(
            pn.pane.HTML('<a href="/" style="font-size:15px; margin:10px;">Main Dashboard</a>'),
            align='end'
        )

        # Create custom token input
        custom_token_input = "**********"
            name= "**********"
            placeholder= "**********"
            value=''
        )

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"a "**********"d "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"c "**********"a "**********"l "**********"l "**********"b "**********"a "**********"c "**********"k "**********"( "**********"e "**********"v "**********"e "**********"n "**********"t "**********") "**********": "**********"
            if event.new and event.new.strip():
                success, message = "**********"
                if success:
                    custom_token_input.value = "**********"
                custom_token_input.name = "**********"

        custom_token_input.param.watch(add_token_callback, 'value')

        # Create widgets
        tokens_select = "**********"
            name= "**********"
            value= "**********"
            options= "**********"
            sizing_mode='stretch_width'
        )
        tokens_select.link(self, value= "**********"

        timeframe_select = pn.widgets.Select(
            name='Timeframe',
            options=self.param.timeframe.objects,
            value=self.timeframe,
            sizing_mode='stretch_width'
        )
        timeframe_select.link(self, value='timeframe')

        window_slider = pn.widgets.IntSlider(
            name='Rolling Window Size',
            start=5,
            end=100,
            step=1,
            value=self.window_size,
            sizing_mode='stretch_width'
        )
        window_slider.link(self, value='window_size')

        update_freq = pn.widgets.IntInput(
            name='Update Frequency (seconds)',
            value=self.update_frequency,
            step=10,
            sizing_mode='stretch_width'
        )
        update_freq.link(self, value='update_frequency')

        # Create help text
        help_text = pn.pane.Markdown("""
        ### How to Use
        1. Enter any token symbol (e.g., "ETH" or "ETHUSDT") in the custom token input
        2. Select one or more tokens from the dropdown for analysis
        3. Adjust timeframe and window size as needed
        4. The graph will show beta distribution relative to Bitcoin
        """)

        # Create layout
        controls = pn.Column(
            pn.pane.Markdown("### Beta Distribution Controls"),
            help_text,
            custom_token_input,
            tokens_select,
            timeframe_select,
            window_slider,
            update_freq,
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px'}
        )

        # Return template with navigation
        return pn.template.FastListTemplate(
            title='Beta Distribution Analysis',
            header=[nav_links],
            sidebar=[controls],
            main=[self.stats_pane, self.plot_pane],
            accent_base_color='#0A5A9C',
            header_background='#0A5A9C',
            theme_toggle=False,
            main_max_width='80%'
        )


class EnhancedCorrelationAnalysis(param.Parameterized):
    base_token = "**********"='BTCUSDT')
    comparison_tokens = "**********"=[], objects=['ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT'])
    timeframe = param.Selector(default='1h', objects=['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
    window_size = param.Integer(default=30, bounds=(5, 100))
    update_frequency = param.Integer(default=0, bounds=(0, 3600))
    show_distribution = param.Boolean(default=True)

# CorrelationAnalysis class
class CorrelationAnalysis(param.Parameterized):
    token1 = "**********"='BTCUSDT')
    token2 = "**********"='ETHUSDT')
    timeframe = param.Selector(default='1h', objects=['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
    window_size = param.Integer(default=30, bounds=(5, 100))
    update_frequency = param.Integer(default=0, bounds=(0, 3600))
    show_vol_adjusted = param.Boolean(default=False)
    show_beta = param.Boolean(default=False)
    show_granger = param.Boolean(default=False)
    show_cointegration = param.Boolean(default=False)
    show_dtw = param.Boolean(default=False)
    show_risk_analysis = param.Boolean(default=False)
    risk_free_rate = param.Number(default=0.02, bounds=(0, 0.1))
    show_hedge_analysis = param.Boolean(default=False)
    min_hedge_effectiveness = param.Number(default=0.5, bounds=(0.3, 0.9))

    def __init__(self, **params):
        self.client = "**********"=api_key, api_secret=api_secret)  # Initialize client here
        print("Client initialized:", self.client)  # Debugging line to confirm client initialization
        super().__init__(**params)
        self.plot_pane = pn.pane.HoloViews(sizing_mode='stretch_both')
        self.correlation_info = pn.pane.Markdown(sizing_mode='stretch_width')
        self.periodic_callback = None
        self.update_data()
        self.update_periodic_callback()

    def get_funding_rate(self, symbol):
        """Get current funding rate for a symbol"""
        try:
            print("Accessing client:", self.client)  # Check client access in the method
            funding_rate = self.client.futures_funding_rate(symbol=symbol)
            if funding_rate:
                return float(funding_rate[0]['fundingRate'])
            return 0.0
        except Exception as e:
            print(f"Error fetching funding rate for {symbol}: {e}")
            return 0.0

    def calculate_hedge_metrics(self, data1, data2):
        """Calculate advanced hedging metrics with funding rate consideration"""
        try:
            # Calculate returns
            returns1 = data1['close'].pct_change().dropna()
            returns2 = data2['close'].pct_change().dropna()

            # Align data
            df = pd.DataFrame({
                'returns1': returns1,
                'returns2': returns2
            }).dropna()

            # Calculate dynamic hedge ratio using rolling regression
            hedge_window = self.window_size
            rolling_cov = df['returns1'].rolling(window=hedge_window).cov(df['returns2'])
            rolling_var = df['returns2'].rolling(window=hedge_window).var()
            dynamic_hedge_ratio = -(rolling_cov / rolling_var)

            # Get funding rates and calculate adjustment
            funding_rate1 = "**********"
            funding_rate2 = "**********"
            funding_adjustment = 1 + (funding_rate1 - funding_rate2)

            # Apply funding rate adjustment to hedge ratio
            adjusted_hedge_ratio = dynamic_hedge_ratio * funding_adjustment

            # Calculate hedge effectiveness
            hedged_returns = df['returns1'] + (adjusted_hedge_ratio * df['returns2'])
            hedge_effectiveness = 1 - (hedged_returns.std() / df['returns1'].std())

            # Calculate volatility regimes
            returns_vol = hedged_returns.rolling(window=hedge_window).std()
            high_vol_regime = returns_vol > returns_vol.quantile(0.75)
            low_vol_regime = returns_vol < returns_vol.quantile(0.25)

            # Create DataFrame for visualization
            hedge_df = pd.DataFrame({
                'timestamp': df.index,
                'base_hedge_ratio': dynamic_hedge_ratio,
                'adjusted_hedge_ratio': adjusted_hedge_ratio,
                'volatility_regime': np.where(high_vol_regime, 'High',
                                              np.where(low_vol_regime, 'Low', 'Medium')),
                'rolling_volatility': returns_vol
            })

            # Calculate regime-specific metrics
            regime_stats = {
                'high_vol_ratio': adjusted_hedge_ratio[high_vol_regime].mean(),
                'low_vol_ratio': adjusted_hedge_ratio[low_vol_regime].mean(),
                'normal_ratio': adjusted_hedge_ratio[~(high_vol_regime | low_vol_regime)].mean()
            }

            return {
                'hedge_ratios': hedge_df,
                'effectiveness': hedge_effectiveness,
                'funding_impact': funding_adjustment - 1,
                'volatility1': returns1.std() * np.sqrt(252),  # Annualized volatility
                'volatility2': returns2.std() * np.sqrt(252),
                'recent_hedge_ratio': adjusted_hedge_ratio.iloc[-1] if not adjusted_hedge_ratio.empty else 0,
                'regime_stats': regime_stats
            }
        except Exception as e:
            print(f"Error in calculate_hedge_metrics: {e}")
            return None

    def create_hedge_visualization(self, hedge_metrics):
        """Create enhanced visualization for hedge analysis"""
        if hedge_metrics is None or 'hedge_ratios' not in hedge_metrics:
            return None

        try:
            hedge_df = hedge_metrics['hedge_ratios']

            # Create base hedge ratio plot
            base_plot = hv.Curve(
                hedge_df, 'timestamp', 'adjusted_hedge_ratio',
                label='Adjusted Hedge Ratio'
            ).opts(
                color='blue',
                line_width=2
            )

            # Create scatter plot for different volatility regimes
            regime_colors = {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}
            regime_scatter = hv.Scatter(
                hedge_df, 'timestamp', 'adjusted_hedge_ratio'
            ).opts(
                color=hv.dim('volatility_regime').categorize(regime_colors),
                size=8,
                marker='circle',
                alpha=0.6
            )

            # Combine plots with proper formatting
            combined_plot = (base_plot * regime_scatter).opts(
                width=800,
                height=400,
                title='Dynamic Hedge Ratio Analysis',
                xlabel='Time',
                ylabel='Hedge Ratio',
                legend_position='top_right',
                tools=['hover'],
                toolbar='above',
                show_grid=True
            )

            return combined_plot

        except Exception as e:
            print(f"Error in create_hedge_visualization: {e}")
            return None

        # Add volatility regime overlay with corrected hover options
        regime_colors = {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}
        scatter = hv.Scatter(
            hedge_df, 'timestamp', 'adjusted_hedge_ratio',
            label='Volatility Regime'
        ).opts(
            color=hv.dim('volatility_regime').categorize(regime_colors),
            size=5,
            alpha=0.6,
            tools=['hover'],
            hover_tooltips=[
                ('Time', '@timestamp{%F %T}'),
                ('Ratio', '@adjusted_hedge_ratio{0.00}'),
                ('Regime', '@volatility_regime'),
                ('Volatility', '@rolling_volatility{0.000}')
            ],
            hover_formatters={'@timestamp': 'datetime'}
        )

        combined_plot = (hedge_plot * scatter).opts(
            legend_position='top_right'
        )

        return combined_plot

    def generate_hedge_recommendations(self, hedge_metrics):
        """Generate actionable hedging recommendations"""
        if hedge_metrics is None:
            return "Unable to generate hedge recommendations due to insufficient data"

        regime_stats = hedge_metrics['regime_stats']
        recent_ratio = hedge_metrics['recent_hedge_ratio']
        effectiveness = hedge_metrics['effectiveness']
        funding_impact = hedge_metrics['funding_impact']

        recommendations = f"""
        ### Hedging Strategy Recommendations

        #### Current Hedge Parameters:
        - Optimal Hedge Ratio: {recent_ratio:.4f}
        - Hedge Effectiveness: {effectiveness:.2%}
        - Funding Rate Impact: {funding_impact:.2%}

        #### Regime-Based Hedge Ratios:
        - High Volatility: {regime_stats['high_vol_ratio']:.4f}
        - Normal Volatility: {regime_stats['normal_ratio']:.4f}
        - Low Volatility: {regime_stats['low_vol_ratio']:.4f}

        #### Implementation Strategy:
        1. Base Position: Implement a {abs(recent_ratio):.4f}x hedge ratio
        2. Volatility Adjustments:
           - Increase hedge to {abs(regime_stats['high_vol_ratio']):.4f}x during high volatility
           - Reduce hedge to {abs(regime_stats['low_vol_ratio']):.4f}x during low volatility
        3. Funding Rate Consideration:
           - Current funding rate impact suggests a {abs(funding_impact * 100):.2f}% adjustment

        #### Risk Metrics:
        - Asset 1 Volatility: {hedge_metrics['volatility1']:.2%}
        - Asset 2 Volatility: {hedge_metrics['volatility2']:.2%}

        #### Monitoring Points:
        - Track hedge effectiveness daily
        - Adjust positions at volatility regime changes
        - Monitor funding rates for significant changes
        """

        return recommendations

    def update_volatility_adjusted_correlation(data1, data2, window_size=30):
        """Calculate volatility-adjusted correlation between two price series"""
        try:
            # Calculate returns
            returns1 = data1['close'].pct_change().dropna()
            returns2 = data2['close'].pct_change().dropna()

            # Calculate rolling volatilities
            vol1 = returns1.rolling(window=window_size).std()
            vol2 = returns2.rolling(window=window_size).std()
            combined_vol = np.sqrt(vol1 ** 2 + vol2 ** 2)

            # Calculate rolling correlation
            vol_adj_corr = returns1.rolling(window=window_size).corr(returns2)

            # Define high volatility periods (above 75th percentile)
            vol_threshold = combined_vol.quantile(0.75)
            high_vol_mask = combined_vol > vol_threshold

            # Create DataFrame for plotting
            df = pd.DataFrame({
                'timestamp': returns1.index,
                'Standard Correlation': vol_adj_corr,
                'Volatility': combined_vol,
                'High Volatility': high_vol_mask
            })

            # Create correlation plot
            corr_plot = hv.Curve(
                df, 'timestamp', 'Standard Correlation', label='Correlation'
            ).opts(
                color='blue',
                line_width=2,
                height=300,
                tools=['hover'],
                title='Volatility-Adjusted Correlation',
                ylabel='Correlation'
            )

            # Add volatility overlay
            vol_plot = hv.Curve(
                df, 'timestamp', 'Volatility', label='Volatility'
            ).opts(
                color='red',
                line_width=1,
                tools=['hover'],
                ylabel='Volatility'
            )

            # Highlight high volatility periods
            high_vol_periods = df[high_vol_mask]
            high_vol_scatter = hv.Scatter(
                high_vol_periods, 'timestamp', 'Standard Correlation'
            ).opts(
                color='yellow',
                size=5,
                marker='circle'
            )

            # Calculate summary statistics
            high_vol_corr = vol_adj_corr[high_vol_mask].mean()
            low_vol_corr = vol_adj_corr[~high_vol_mask].mean()

            # Combine plots
            combined_plot = (corr_plot * high_vol_scatter).opts(
                width=800,
                height=300,
                legend_position='top_right',
                title='Volatility-Adjusted Correlation Analysis'
            )

            return {
                'plot': combined_plot,
                'metrics': {
                    'high_vol_corr': high_vol_corr,
                    'low_vol_corr': low_vol_corr,
                    'vol_threshold': vol_threshold
                }
            }

        except Exception as e:
            print(f"Error in volatility adjusted correlation: {str(e)}")
            return None

    @param.depends('token1', 'token2', 'timeframe', 'window_size',
                   'show_beta', 'show_granger', 'show_cointegration', 'show_dtw',
                   'show_risk_analysis', 'risk_free_rate', 'show_vol_adjusted',
                   'show_hedge_analysis', 'min_hedge_effectiveness',
                   watch=True)
    def update_data(self):
        try:
            data1 = "**********"
            data2 = "**********"

            if data1.empty or data2.empty:
                self.correlation_info.object = "Error: "**********"
                self.plot_pane.object = None
                return

            # Initialize plots list and analysis results
            plots = []
            analysis_results = ""

            # Standard correlation analysis
            static_corr, rolling_corr, df = calculate_correlations(data1, data2, self.window_size)

            # Create normalized price comparison plot first
            normalized_df = pd.DataFrame({
                'timestamp': df.index,
                self.token1: "**********"
                self.token2: "**********"
            }).reset_index(drop=True)

            price_plot = (
                    hv.Curve(normalized_df, 'timestamp', self.token1, label= "**********"
                        color='blue', line_width=2
                    ) *
                    hv.Curve(normalized_df, 'timestamp', self.token2, label= "**********"
                        color='red', line_width=2
                    )
            ).opts(
                height=300,
                responsive=True,
                title='Normalized Price Comparison'
            )
            plots.append(price_plot)

            if self.show_hedge_analysis:
                print("Hedge analysis is enabled")
                try:
                    hedge_metrics = self.calculate_hedge_metrics(data1, data2)
                    if hedge_metrics is not None:
                        print("Successfully calculated hedge metrics")
                        hedge_plot = self.create_hedge_visualization(hedge_metrics)
                        if hedge_plot is not None:
                            print("Successfully created hedge plot")
                            plots.append(hedge_plot.opts(height=300))
                        hedge_recommendations = self.generate_hedge_recommendations(hedge_metrics)
                        analysis_results += f"\n{hedge_recommendations}\n"
                except Exception as e:
                    print(f"Error in hedge analysis section: {str(e)}")
                    import traceback
                    traceback.print_exc()
            # Create rolling correlation plot with volatility adjustment if enabled
            rolling_corr_df = pd.DataFrame({
                'timestamp': df.index,
                'Correlation': rolling_corr
            })

            if self.show_vol_adjusted:
                # Calculate volatility metrics
                returns1 = data1['close'].pct_change().dropna()
                returns2 = data2['close'].pct_change().dropna()

                vol1 = returns1.rolling(window=self.window_size).std()
                vol2 = returns2.rolling(window=self.window_size).std()
                combined_vol = np.sqrt(vol1 ** 2 + vol2 ** 2)

                # Align indexes
                combined_vol = combined_vol.reindex(rolling_corr_df.index)

                # Define high volatility periods (above 75th percentile)
                vol_threshold = combined_vol.quantile(0.75)
                high_vol_mask = combined_vol > vol_threshold

                # Add volatility information to the DataFrame
                rolling_corr_df['Volatility'] = combined_vol
                rolling_corr_df['High_Volatility'] = high_vol_mask

                # Create correlation line plot with larger dimensions
                corr_plot = hv.Curve(
                    rolling_corr_df, 'timestamp', 'Correlation', label='Rolling Correlation'
                ).opts(
                    color='green',
                    line_width=2,
                    ylim=(-1, 1),
                    height=600,  # Increased height
                    width=1000,  # Increased width
                    tools=['hover'],
                    fontsize={'title': 16, 'labels': 14, 'ticks': 12},  # Larger fonts
                    title='Rolling Correlation with Volatility',
                    xlabel='Time',
                    ylabel='Correlation'
                )

                # Add high volatility points
                high_vol_df = rolling_corr_df[rolling_corr_df['High_Volatility']]
                high_vol_scatter = hv.Scatter(
                    high_vol_df, 'timestamp', 'Correlation', label='High Volatility Periods'
                ).opts(
                    color='yellow',
                    size=10,  # Larger points
                    marker='circle'
                )

                # Calculate statistics for analysis results
                high_vol_corr = rolling_corr[high_vol_mask].mean()
                low_vol_corr = rolling_corr[~high_vol_mask].mean()

                # Add volatility analysis to results
                vol_info = f"""
                ### Volatility-Adjusted Analysis

                • Average Correlation (High Volatility): {high_vol_corr:.4f}

                • Average Correlation (Low Volatility): {low_vol_corr:.4f}

                • Volatility Impact: {(high_vol_corr - low_vol_corr):.4f}
                (Positive value indicates stronger correlation during high volatility)

                • Volatility Threshold: {vol_threshold:.6f}

                • High Volatility Periods: {high_vol_mask.sum()} out of {len(high_vol_mask)}

                Note: Yellow dots on correlation plot indicate high volatility periods
                """
                analysis_results += vol_info

                # Combine correlation plot with high volatility markers
                corr_plot = (corr_plot * high_vol_scatter).opts(
                    title='Volatility-Adjusted Rolling Correlation',
                    legend_position='top_right'
                )
            else:
                # Standard correlation plot without volatility adjustment
                corr_plot = hv.Curve(rolling_corr_df, 'timestamp', 'Correlation').opts(
                    color='green',
                    line_width=2,
                    ylim=(-1, 1),
                    width=1000,
                    height=500,
                    tools=['hover'],
                    title='Rolling Correlation'
                )

            # Initialize list to store all plots
            plots = [price_plot, corr_plot]

            # Add Risk Analysis if enabled
            # Update the risk analysis section in the update_data method:
            if self.show_risk_analysis:
                try:
                    data1.name = "**********"
                    data2.name = "**********"
                    ef_results = perform_risk_analysis(data1, data2, self.risk_free_rate)

                    if ef_results[0] is not None:
                        ef_plot, min_vol_port, max_sharpe_port = ef_results
                        plots.append(ef_plot.opts(height=300))

                        risk_info = """### Portfolio Risk Analysis

            Minimum Volatility Portfolio:

            Expected Return: {:.4f}

            Risk (Std Dev): {:.4f}

            Weights for {}: {:.2%}

            Weights for {}: {:.2%}


            Maximum Sharpe Ratio Portfolio:

            Expected Return: {:.4f}

            Risk (Std Dev): {:.4f}

            Sharpe Ratio: {:.4f}

            Weights for {}: {:.2%}

            Weights for {}: {:.2%}
            """.format(
                            min_vol_port['return'],
                            min_vol_port['std_dev'],
                            self.token1, min_vol_port[self.token1],
                            self.token2, min_vol_port[self.token2],
                            max_sharpe_port['return'],
                            max_sharpe_port['std_dev'],
                            max_sharpe_port['sharpe_ratio'],
                            self.token1, max_sharpe_port[self.token1],
                            self.token2, max_sharpe_port[self.token2]
                        )
                    else:
                        risk_info = """
            ### Portfolio Risk Analysis

            Unable to calculate risk metrics.
            """

                    analysis_results += f"\n{risk_info}\n"

                except Exception as e:
                    print(f"Risk analysis error: {str(e)}")
                    analysis_results += """
            ### Portfolio Risk Analysis

            Error calculating risk metrics.
            """

                # Add other analyses results
            if self.show_beta:
                beta_metrics = calculate_beta_metrics(data1, data2)
                beta_info = f"""
                        ### Beta Analysis

                        • Beta: {beta_metrics['beta']:.4f}

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"• "**********"  "**********"{ "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********"} "**********"  "**********"M "**********"e "**********"t "**********"r "**********"i "**********"c "**********"s "**********": "**********"
                          - Volatility: "**********":.4f}
                          - Sharpe Ratio: "**********":.4f}

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"• "**********"  "**********"{ "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"2 "**********"} "**********"  "**********"M "**********"e "**********"t "**********"r "**********"i "**********"c "**********"s "**********": "**********"
                          - Volatility: "**********":.4f}
                          - Sharpe Ratio: "**********":.4f}
                        """
                analysis_results += f"\n{beta_info}\n"

            if self.show_granger:
                results_1_2, results_2_1 = analyze_granger_causality(data1, data2)
                p_value_1_2 = results_1_2[1][0]['ssr_chi2test'][1]
                p_value_2_1 = results_2_1[1][0]['ssr_chi2test'][1]
                granger_info = f"""
                        ### Granger Causality Analysis

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"• "**********"  "**********"{ "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********"} "**********"  "**********"→ "**********"  "**********"{ "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"2 "**********"} "**********": "**********"
                          - p-value: {p_value_1_2:.4f}

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"• "**********"  "**********"{ "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"2 "**********"} "**********"  "**********"→ "**********"  "**********"{ "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"1 "**********"} "**********": "**********"
                          - p-value: {p_value_2_1:.4f}
                        """
                analysis_results += f"\n{granger_info}\n"

            if self.show_cointegration:
                coint_result = analyze_cointegration(data1, data2)
                coint_info = f"""
                        ### Cointegration Analysis

                        • Status: {'Cointegrated' if coint_result['cointegrated'] else 'Not Cointegrated'}

                        • P-value: {coint_result['p_value']:.4f}
                        """
                analysis_results += f"\n{coint_info}\n"

                # Combine all plots vertically using HoloViews Layout
            combined_plot = hv.Layout(plots).cols(1).opts(
                shared_axes=False,
                sizing_mode='stretch_width'
            )

            # Update display
            self.plot_pane.object = combined_plot
            self.correlation_info.object = f"""
                    ## Correlation Analysis Results

                    **Static Correlation:** {static_corr:.4f}

                    **Analysis Period:** {df.index[0]} to {df.index[-1]}

                    **Window Size:** {self.window_size} periods

                    **Latest Rolling Correlation:** {rolling_corr.iloc[-1]:.4f}

                    {analysis_results}
                    """

        except Exception as e:
            print(f"Debug - Error details: {str(e)}")
            self.correlation_info.object = f"Error during analysis: {str(e)}"
            self.plot_pane.object = None

            # Add other analyses results as needed
            if self.show_beta:
                beta_metrics = calculate_beta_metrics(data1, data2)
                beta_info = f"""
                ### Beta Analysis
                - Beta: {beta_metrics['beta']:.4f}
                - {self.token1} Volatility: "**********":.4f}
                - {self.token2} Volatility: "**********":.4f}
                - {self.token1} Sharpe Ratio: "**********":.4f}
                - {self.token2} Sharpe Ratio: "**********":.4f}
                """
                analysis_results += beta_info

            if self.show_granger:
                results_1_2, results_2_1 = analyze_granger_causality(data1, data2)
                p_value_1_2 = results_1_2[1][0]['ssr_chi2test'][1]
                p_value_2_1 = results_2_1[1][0]['ssr_chi2test'][1]
                granger_info = f"""
                ### Granger Causality
                - {self.token1} → {self.token2} p-value: "**********":.4f}
                - {self.token2} → {self.token1} p-value: "**********":.4f}
                """
                analysis_results += granger_info

            if self.show_cointegration:
                coint_result = analyze_cointegration(data1, data2)
                coint_info = f"""
                ### Cointegration Analysis
                - Cointegrated: {coint_result['cointegrated']}
                - P-value: {coint_result['p_value']:.4f}
                """
                analysis_results += coint_info

            # Combine all plots vertically using HoloViews Layout
            combined_plot = hv.Layout(plots).cols(1).opts(
                shared_axes=False,
                sizing_mode='stretch_width'
            )

            # Update display
            self.plot_pane.object = combined_plot
            self.correlation_info.object = f"""
            ## Correlation Analysis Results

            **Static Correlation:** {static_corr:.4f}

            **Analysis Period:** {df.index[0]} to {df.index[-1]}

            **Window Size:** {self.window_size} periods

            **Latest Rolling Correlation:** {rolling_corr.iloc[-1]:.4f}

            {analysis_results}
            """

        except Exception as e:
            print(f"Debug - Error details: {str(e)}")
            self.correlation_info.object = f"Error during analysis: {str(e)}"
            self.plot_pane.object = None

            if self.show_dtw:
                dtw_results = calculate_dtw(data1, data2)

                if dtw_results['success']:
                    # Add DTW info to analysis results
                    analysis_results += f"""
                                ### Dynamic Time Warping Analysis
                                - DTW Distance: {dtw_results['distance']:.4f}
                                - Mean Distance per Point: {dtw_results['mean_dist']:.4f}
                                - Data Points Compared: {dtw_results['length']}
                                - Correlation of Normalized Series: {dtw_results['correlation']:.4f}
                                - Interpretation: 
                                    * Lower DTW distance indicates more similar price movements
                                    * Mean distance per point helps compare series of different lengths
                                    * Values are calculated on normalized prices for fair comparison
                                """

                    # Create DTW path visualization
                    path = dtw_results['path']
                    warping_df = pd.DataFrame({
                        'index1': [p[0] for p in path],
                        'index2': [p[1] for p in path]
                    })

                    path_plot = hv.Curve(warping_df, 'index1', 'index2').opts(
                        title='DTW Warping Path',
                        width=400,
                        height=400,
                        tools=['hover'],
                        color='purple',
                        line_width=1,
                        xlabel= "**********"
                        ylabel= "**********"
                        show_grid=True
                    )

                    # Add path plot to combined plot
                    combined_plot = pn.Column(
                        combined_plot,
                        path_plot,
                        sizing_mode='stretch_width'
                    )
                else:
                    analysis_results += """
                                ### Dynamic Time Warping
                                Unable to calculate DTW distance. Please check data validity.
                                """

            # Perform analyses based on toggles
            if self.show_beta:
                beta_metrics = calculate_beta_metrics(data1, data2)
                beta_info = f"""
                ### Beta Analysis
                - Beta: {beta_metrics['beta']:.4f}
                - {self.token1} Volatility: "**********":.4f}
                - {self.token2} Volatility: "**********":.4f}
                - {self.token1} Sharpe Ratio: "**********":.4f}
                - {self.token2} Sharpe Ratio: "**********":.4f}
                """
                analysis_results += beta_info

            # Inside the update_data method, add this section where other analysis methods are called:
            if self.show_vol_adjusted:
                # Create correlation line plot with larger dimensions
                corr_plot = hv.Curve(
                    rolling_corr_df, 'timestamp', 'Correlation', label='Rolling Correlation'
                ).opts(
                    color='green',
                    line_width=2,
                    ylim=(-1, 1),
                    height=700,  # Increased height
                    width=2000,  # Increased width
                    tools=['hover'],
                    fontsize={'title': 16, 'labels': 14, 'ticks': 12},  # Larger fonts
                    title='Rolling Correlation with Volatility',
                    xlabel='Time',
                    ylabel='Correlation'
                )

                # Add high volatility points
                high_vol_df = rolling_corr_df[rolling_corr_df['High_Volatility']]
                high_vol_scatter = hv.Scatter(
                    high_vol_df, 'timestamp', 'Correlation', label='High Volatility Periods'
                ).opts(
                    color='yellow',
                    size=10,  # Larger points
                    marker='circle'
                )

                # Combine correlation plot with high volatility markers
                corr_plot = (corr_plot * high_vol_scatter).opts(
                    title='Volatility-Adjusted Rolling Correlation',
                    legend_position='top_right'
                )
            else:
                # Standard correlation plot without volatility adjustment
                corr_plot = hv.Curve(rolling_corr_df, 'timestamp', 'Correlation').opts(
                    color='green',
                    line_width=2,
                    ylim=(-1, 1),
                    height=1000,  # Increased height
                    width=3000,  # Increased width
                    tools=['hover'],
                    fontsize={'title': 16, 'labels': 14, 'ticks': 12},  # Larger fonts
                    title='Rolling Correlation',
                    xlabel='Time',
                    ylabel='Correlation',
                    legend_position='top_right'
                )

            if self.show_granger:
                results_1_2, results_2_1 = analyze_granger_causality(data1, data2)
                p_value_1_2 = results_1_2[1][0]['ssr_chi2test'][1]
                p_value_2_1 = results_2_1[1][0]['ssr_chi2test'][1]
                granger_info = f"""
                ### Granger Causality
                - {self.token1} → {self.token2} p-value: "**********":.4f}
                - {self.token2} → {self.token1} p-value: "**********":.4f}
                """
                analysis_results += granger_info

            if self.show_cointegration:
                coint_result = analyze_cointegration(data1, data2)
                coint_info = f"""
                ### Cointegration Analysis
                - Cointegrated: {coint_result['cointegrated']}
                - P-value: {coint_result['p_value']:.4f}
                """
                analysis_results += coint_info

                # Add DTW Analysis
                if self.show_dtw:
                    distance, path, data_length, mean_dist, correlation = calculate_dtw(data1, data2)

                    if distance is not None:
                        dtw_info = f"""
                                ### Dynamic Time Warping Analysis
                                - DTW Distance: {distance:.4f}
                                - Mean Distance per Point: {mean_dist:.4f}
                                - Data Points Compared: {data_length}
                                - Correlation of Normalized Series: {correlation:.4f}
                                - Interpretation: 
                                    * Lower DTW distance indicates more similar price movements
                                    * Mean distance per point helps compare series of different lengths
                                    * Values are calculated on normalized prices for fair comparison
                                """

                        try:
                            warping_df = pd.DataFrame({
                                'index1': [p[0] for p in path],
                                'index2': [p[1] for p in path]
                            })

                            path_plot = hv.Curve(warping_df, 'index1', 'index2').opts(
                                title='DTW Warping Path',
                                width=300,
                                height=300,
                                tools=['hover'],
                                color='purple',
                                line_width=1,
                                xlabel= "**********"
                                ylabel= "**********"
                            )

                            # Add path_plot to combined_plot
                            combined_plot = pn.Column(
                                combined_plot,
                                path_plot.opts(width=400, height=400),
                                sizing_mode='stretch_width'
                            )

                        except Exception as e:
                            print(f"DTW visualization error: {e}")
                    else:
                        dtw_info = """
                                ### Dynamic Time Warping
                                Unable to calculate DTW distance. Please check data validity.
                                """

                    analysis_results += dtw_info

                # Add DTW visualization if available


            # Standard correlation analysis
            static_corr, rolling_corr, df = calculate_correlations(data1, data2, self.window_size)

            # Create normalized price comparison plot
            normalized_df = pd.DataFrame({
                'timestamp': df.index,
                self.token1: "**********"
                self.token2: "**********"
            }).reset_index(drop=True)

            price_plot = (
                hv.Curve(normalized_df, 'timestamp', self.token1, label= "**********"
                    color='blue', line_width=2, height=600, responsive=True
                ) * hv.Curve(normalized_df, 'timestamp', self.token2, label= "**********"
                    color='red', line_width=2, height=600, responsive=True
                )
            )

            # Create rolling correlation plot
            rolling_corr_df = pd.DataFrame({
                'timestamp': df.index,
                'Correlation': rolling_corr
            }).reset_index(drop=True)

            corr_plot = hv.Curve(rolling_corr_df, 'timestamp', 'Correlation').opts(
                color='green', line_width=2,
                ylim=(-1, 1),
                height=300,
                tools=['hover'],
                responsive=True
            )

            # Combine plots
            combined_plot = (price_plot + corr_plot).cols(1).opts(
                title=f'Price Comparison and Rolling Correlation ({self.window_size} periods)',
                shared_axes=False,
                legend_position='top_right'
            )

            # Add DTW plot if available


            # Add Risk Analysis if enabled
            if self.show_risk_analysis:
                try:
                    # Set names for the datasets
                    data1.name = "**********"
                    data2.name = "**********"

                    ef_results = perform_risk_analysis(data1, data2, self.risk_free_rate)

                    if ef_results[0] is not None:
                        ef_plot, min_vol_port, max_sharpe_port = ef_results

                        # Update combined plot to include efficient frontier
                        combined_plot = pn.Column(
                            combined_plot,
                            ef_plot,
                            sizing_mode='stretch_width'
                        )

                        risk_info = f"""
                        ### Portfolio Risk Analysis

                        **Minimum Volatility Portfolio:**
                        - Expected Return: {min_vol_port['return']:.4f}
                        - Risk (Std Dev): {min_vol_port['std_dev']:.4f}
                        - Weights: "**********": {min_vol_port[self.token1]:.2%}, {self.token2}: {min_vol_port[self.token2]:.2%}

                        **Maximum Sharpe Ratio Portfolio:**
                        - Expected Return: {max_sharpe_port['return']:.4f}
                        - Risk (Std Dev): {max_sharpe_port['std_dev']:.4f}
                        - Sharpe Ratio: {max_sharpe_port['sharpe_ratio']:.4f}
                        - Weights: "**********": {max_sharpe_port[self.token1]:.2%}, {self.token2}: {max_sharpe_port[self.token2]:.2%}
                        """
                    else:
                        risk_info = "\n### Portfolio Risk Analysis\nUnable to calculate risk metrics."

                    analysis_results += risk_info

                except Exception as e:
                    print(f"Risk analysis error: {str(e)}")
                    analysis_results += "\n### Portfolio Risk Analysis\nError calculating risk metrics."


            # Update display
            self.plot_pane.object = combined_plot
            self.correlation_info.object = f"""
            ## Correlation Analysis Results

            **Static Correlation:** {static_corr:.4f}

            **Analysis Period:** {df.index[0]} to {df.index[-1]}

            **Window Size:** {self.window_size} periods

            **Latest Rolling Correlation:** {rolling_corr.iloc[-1]:.4f}

            {analysis_results}
            """
        except Exception as e:
            print(f"Debug - Error details: {str(e)}")  # For debugging
            self.correlation_info.object = f"Error during analysis: {str(e)}"
            self.plot_pane.object = None

    @param.depends('update_frequency', watch=True)
    def update_periodic_callback(self):
        if self.periodic_callback:
            self.periodic_callback.stop()
            self.periodic_callback = None
        if self.update_frequency > 0:
            self.periodic_callback = pn.state.add_periodic_callback(
                self.update_data, period=self.update_frequency * 1000
            )

    def view(self):
        # Create basic input widgets
        token1_input = "**********"
            name= "**********"
            value= "**********"
            sizing_mode='stretch_width'
        )
        token1_input.link(self, value= "**********"

        token2_input = "**********"
            name= "**********"
            value= "**********"
            sizing_mode='stretch_width'
        )
        token2_input.link(self, value= "**********"

        timeframe_select = pn.widgets.Select(
            name='Timeframe',
            options=self.param.timeframe.objects,
            value=self.timeframe,
            sizing_mode='stretch_width'
        )
        timeframe_select.link(self, value='timeframe')

        window_slider = pn.widgets.IntSlider(
            name='Rolling Window Size',
            start=5,
            end=100,
            step=1,
            value=self.window_size,
            sizing_mode='stretch_width'
        )
        window_slider.link(self, value='window_size')

        update_freq = pn.widgets.IntInput(
            name='Update Frequency (seconds)',
            value=self.update_frequency,
            step=10,
            sizing_mode='stretch_width'
        )
        update_freq.link(self, value='update_frequency')

        # Create analysis toggle widgets
        show_beta = pn.widgets.Checkbox(
            name='Show Beta Analysis',
            value=self.show_beta
        )
        show_beta.link(self, value='show_beta')

        show_granger = pn.widgets.Checkbox(
            name='Show Granger Causality',
            value=self.show_granger
        )
        show_granger.link(self, value='show_granger')

        show_cointegration = pn.widgets.Checkbox(
            name='Show Cointegration',
            value=self.show_cointegration
        )
        show_cointegration.link(self, value='show_cointegration')

        show_dtw = pn.widgets.Checkbox(
            name='Show DTW Analysis',
            value=self.show_dtw
        )
        show_dtw.link(self, value='show_dtw')
        # Create hedge analysis widgets
        show_hedge = pn.widgets.Checkbox(
            name='Show Hedge Analysis',
            value=self.show_hedge_analysis
        )
        show_hedge.link(self, value='show_hedge_analysis')

        min_hedge_effect = pn.widgets.FloatSlider(
            name='Minimum Hedge Effectiveness',
            start=0.3,
            end=0.9,
            step=0.05,
            value=self.min_hedge_effectiveness
        )
        min_hedge_effect.link(self, value='min_hedge_effectiveness')

        # Group hedge settings
        show_hedge = pn.widgets.Checkbox(
            name='Show Hedge Analysis',
            value=self.show_hedge_analysis
        )
        show_hedge.link(self, value='show_hedge_analysis')

        min_hedge_effect = pn.widgets.FloatSlider(
            name='Minimum Hedge Effectiveness',
            start=0.3,
            end=0.9,
            step=0.05,
            value=self.min_hedge_effectiveness
        )
        min_hedge_effect.link(self, value='min_hedge_effectiveness')

        # Group hedge settings
        hedge_settings = pn.Column(
            pn.pane.Markdown("### Hedging Strategy Settings"),
            show_hedge,
            min_hedge_effect,
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px'},
            margin=(0, 0, 20, 0)
        )

        show_vol_adjusted = pn.widgets.Checkbox(
            name='Show Volatility-Adjusted Correlation',
            value=self.show_vol_adjusted
        )
        show_vol_adjusted.link(self, value='show_vol_adjusted')

        show_risk_analysis = pn.widgets.Checkbox(
            name='Show Risk Analysis',
            value=self.show_risk_analysis
        )
        show_risk_analysis.link(self, value='show_risk_analysis')

        risk_free_input = pn.widgets.FloatInput(
            name='Risk-Free Rate',
            value=self.risk_free_rate,
            step=0.005,
            start=0.0,
            end=0.1,
            sizing_mode='stretch_width'
        )
        risk_free_input.link(self, value='risk_free_rate')

        basic_settings = pn.Column(
            pn.pane.Markdown("### Basic Settings"),
            token1_input,
            token2_input,
            timeframe_select,
            window_slider,
            update_freq,
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px'},
            margin=(0, 0, 20, 0)
        )

        advanced_analysis = pn.Column(
            pn.pane.Markdown("### Advanced Analysis"),
            show_beta,
            show_granger,
            show_cointegration,
            show_dtw,
            show_vol_adjusted,
            show_risk_analysis,
            risk_free_input,
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px'},
            margin=(0, 0, 20, 0)
        )

        sidebar = pn.Column(
            basic_settings,
            advanced_analysis,
            hedge_settings,
            sizing_mode='stretch_width',
            margin=(10, 5)
        )

        # Create main content
        main_content = pn.Column(
            self.correlation_info,
            self.plot_pane,
            sizing_mode='stretch_both'
        )

        # Create navigation
        nav_links = pn.Row(
            pn.pane.HTML('<a href="/" style="font-size:15px; margin:10px;">Main Dashboard</a>'),
            align='end'
        )

        # Apply custom CSS
        pn.extension(raw_css=["""
            .bk.panel-main {
                padding-left: 10px;
                padding-right: 10px;
            }
            """])

        return pn.template.FastListTemplate(
            title='Correlation Analysis',
            header=[nav_links],
            sidebar=[sidebar],
            main=[main_content],
            accent_base_color='#0A5A9C',
            header_background='#0A5A9C',
            theme_toggle=False,
            main_max_width='80%'
        )

dashboard = CryptoDashboard()
correlation_analysis = CorrelationAnalysis()
beta_analysis = BetaDistributionAnalysis()

# Create the navigation links
nav_links = pn.Row(
    pn.pane.HTML('<a href="/correlation" style="font-size:15px; margin:10px;">Correlation Analysis</a>'),
    pn.pane.HTML('<a href="/beta" style="font-size:15px; margin:10px;">Beta Distribution</a>'),
    align='end'
)

# Serve the dashboard and analysis pages
pn.serve({
   '/': dashboard.view(),
            '/correlation': correlation_analysis.view(),
            '/beta': beta_analysis.view()
        }, show=True, port=5006)
