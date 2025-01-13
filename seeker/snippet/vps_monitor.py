#date: 2025-01-13T16:43:25Z
#url: https://api.github.com/gists/00efbb36c8b866212815646eb1ab9ce2
#owner: https://api.github.com/users/vanbumi

import requests
import psutil
import time
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import json
import threading

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "vps_monitor.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Load environment variables
logging.info("Loading environment variables...")
load_dotenv()

# Verify Telegram credentials
bot_token = "**********"
chat_id = os.getenv('TELEGRAM_CHAT_ID')
 "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"c "**********"h "**********"a "**********"t "**********"_ "**********"i "**********"d "**********": "**********"
    logging.error("Missing Telegram credentials in .env file!")
    exit(1)
logging.info("Telegram credentials loaded successfully")

# MT5 credentials
MT5_LOGIN = os.getenv('MT5_LOGIN')
MT5_PASSWORD = "**********"
MT5_SERVER = os.getenv('MT5_SERVER')

logging.info(f"Using MT5 Server: {MT5_SERVER}")

# Trading pairs and schedules (NY and WIB)
TRADING_SCHEDULES = {
    'BTCUSD': {
        'asia': {'ny': '18:30', 'wib': '06:30'},
        'london': {'ny': '02:00', 'wib': '14:00'},
        'ny': {'ny': '07:00', 'wib': '19:00'}
    },
    'EURUSD': {
        'asia': {'ny': '18:30', 'wib': '06:30'},
        'london': {'ny': '02:00', 'wib': '14:00'},
        'ny': {'ny': '07:00', 'wib': '19:00'}
    },
    'XAUUSD': {
        'asia': {'ny': '18:30', 'wib': '06:30'},
        'london': {'ny': '02:00', 'wib': '14:00'},
        'ny': {'ny': '07:00', 'wib': '19:00'}
    }
}

# High-Impact News Times (NY) - Update these based on your forex calendar
HIGH_IMPACT_NEWS = {
    'USD': {
        'NFP': {'day': 'first friday', 'time': '08:30 AM'},  # Non-Farm Payroll
        'CPI': {'day': '12', 'time': '08:30 AM'},           # Consumer Price Index
        'FOMC': {'day': 'varies', 'time': '02:00 PM'}       # Federal Reserve
    },
    'EUR': {
        'ECB': {'day': 'varies', 'time': '08:45 AM'}        # European Central Bank
    }
}

def check_upcoming_news():
    """Check for upcoming high-impact news events"""
    now = datetime.now(timezone.utc)
    alerts = []
    
    # Example for NFP (first Friday)
    first_friday = get_first_friday_of_month(now)
    if first_friday and first_friday.date() == now.date():
        nfp_time = datetime.strptime(HIGH_IMPACT_NEWS['USD']['NFP']['time'], '%I:%M %p').time()
        nfp_datetime = datetime.combine(first_friday.date(), nfp_time)
        nfp_datetime = nfp_datetime.astimezone(timezone(timedelta(hours=-5)))  # Convert to NY time
        time_diff = (nfp_datetime - now.astimezone(timezone(timedelta(hours=-5)))).total_seconds()
        
        if 0 < time_diff <= 1800:  # 30 minutes before
            alerts.append(f" HIGH IMPACT NEWS in 30 minutes: NFP (Non-Farm Payroll)")
            # Take safety measures for USD pairs
            for symbol in ['BTCUSD', 'EURUSD']:
                handle_high_impact_news(symbol)
    
    # Check CPI
    if now.day == 12:  # CPI usually on 12th
        cpi_time = datetime.strptime(HIGH_IMPACT_NEWS['USD']['CPI']['time'], '%I:%M %p').time()
        cpi_datetime = datetime.combine(now.date(), cpi_time)
        cpi_datetime = cpi_datetime.astimezone(timezone(timedelta(hours=-5)))  # Convert to NY time
        time_diff = (cpi_datetime - now.astimezone(timezone(timedelta(hours=-5)))).total_seconds()
        
        if 0 < time_diff <= 1800:
            alerts.append(f" HIGH IMPACT NEWS in 30 minutes: CPI (Consumer Price Index)")
            # Take safety measures for USD pairs
            for symbol in ['BTCUSD', 'EURUSD']:
                handle_high_impact_news(symbol)
    
    return alerts

def get_first_friday_of_month(dt):
    """Get the first Friday of the current month"""
    first_day = dt.replace(day=1)
    while first_day.weekday() != 4:  # 4 is Friday
        first_day += timedelta(days=1)
    return first_day

def handle_high_impact_news(symbol):
    """Handle high-impact news by closing positions and canceling orders"""
    try:
        if not mt5.initialize():
            return False
            
        success = True
        message = f" NEWS SAFETY: Taking precautions for {symbol}:\n"
        
        # 1. Close existing positions
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "comment": "Closed before high-impact news"
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    message += f" Closed position #{pos.ticket}\n"
                else:
                    message += f" Failed to close position #{pos.ticket}\n"
                    success = False
        
        # 2. Cancel pending orders
        orders = mt5.orders_get(symbol=symbol)
        if orders:
            for order in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "comment": "Cancelled before high-impact news"
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    message += f" Cancelled pending order #{order.ticket}\n"
                else:
                    message += f" Failed to cancel order #{order.ticket}\n"
                    success = False
        
        # 3. Update bot status to disable trading
        try:
            status_file = Path("bot_status.json")
            if status_file.exists():
                with open(status_file, "r") as f:
                    status = json.load(f)
            else:
                status = {}
            
            status[symbol] = {
                "enabled": False,
                "reason": "High-impact news period",
                "disabled_until": (datetime.now() + timedelta(minutes=45)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(status_file, "w") as f:
                json.dump(status, f, indent=4)
            
            message += f" Disabled trading for {symbol} for 45 minutes\n"
            
        except Exception as e:
            message += f" Failed to update bot status: {str(e)}\n"
            success = False
        
        # Send notification
        send_telegram_message(message)
        return success
        
    except Exception as e:
        logging.error(f"Error handling news safety for {symbol}: {str(e)}")
        return False

def connect_mt5():
    """Connect to MT5"""
    try:
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            return False
            
        # Login to MT5
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"m "**********"t "**********"5 "**********". "**********"l "**********"o "**********"g "**********"i "**********"n "**********"( "**********"i "**********"n "**********"t "**********"( "**********"M "**********"T "**********"5 "**********"_ "**********"L "**********"O "**********"G "**********"I "**********"N "**********") "**********", "**********"  "**********"M "**********"T "**********"5 "**********"_ "**********"P "**********"A "**********"S "**********"S "**********"W "**********"O "**********"R "**********"D "**********", "**********"  "**********"M "**********"T "**********"5 "**********"_ "**********"S "**********"E "**********"R "**********"V "**********"E "**********"R "**********") "**********": "**********"
            logging.error("Failed to login to MT5")
            return False
            
        logging.info("Connected to MT5 successfully")
        return True
    except Exception as e:
        logging.error(f"Error connecting to MT5: {str(e)}")
        return False

def get_trading_status():
    """Get trading status for all pairs"""
    try:
        if not mt5.initialize():
            return None
            
        status = {
            'pairs': {}, 
            'total_profit': 0.0, 
            'pending_orders': {},
            'positions': {},
            'history': {}
        }
        
        # Get all current positions
        positions = mt5.positions_get()
        if positions is None:
            positions = ()
            
        # Get all pending orders
        pending_orders = mt5.orders_get()
        if pending_orders is None:
            pending_orders = ()
            
        # Get today's history
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Process positions and history by symbol
        for symbol in TRADING_SCHEDULES.keys():
            symbol_positions = [pos for pos in positions if pos.symbol == symbol]
            symbol_profit = sum(pos.profit for pos in symbol_positions)
            symbol_orders = [order for order in pending_orders if order.symbol == symbol]
            
            status['pairs'][symbol] = {
                'positions': len(symbol_positions),
                'profit': symbol_profit,
                'pending_orders': len(symbol_orders)
            }
            
            # Store pending order details
            if symbol_orders:
                status['pending_orders'][symbol] = []
                for order in symbol_orders:
                    order_type = "Buy Stop" if order.type == mt5.ORDER_TYPE_BUY_STOP else "Sell Stop"
                    status['pending_orders'][symbol].append({
                        'type': order_type,
                        'price': order.price_open,
                        'sl': order.sl,
                        'tp': order.tp,
                        'ticket': order.ticket
                    })
            
            # Store position details
            if symbol_positions:
                status['positions'][symbol] = []
                for pos in symbol_positions:
                    pos_type = "Buy" if pos.type == mt5.POSITION_TYPE_BUY else "Sell"
                    status['positions'][symbol].append({
                        'type': pos_type,
                        'price': pos.price_open,
                        'profit': pos.profit,
                        'ticket': pos.ticket
                    })
        
        return status
        
    except Exception as e:
        logging.error(f"Error getting trading status: {str(e)}")
        return None

def format_pending_orders_message(symbol, orders):
    """Format pending orders message"""
    msg = f" New Pending Orders for {symbol}:\n"
    for order in orders:
        msg += f"Type: {order['type']}\n"
        msg += f"Entry: ${order['price']:.2f}\n"
        msg += f"SL: ${order['sl']:.2f}\n"
        msg += f"TP: ${order['tp']:.2f}\n"
        msg += "-------------------\n"
    return msg

def format_position_opened_message(symbol, position):
    """Format position opened message"""
    msg = f" Position Opened for {symbol}:\n"
    msg += f"Type: {position['type']}\n"
    msg += f"Entry: ${position['price']:.2f}\n"
    msg += "-------------------\n"
    return msg

def format_position_closed_message(symbol, deal):
    """Format position closed message"""
    emoji = "" if deal['profit'] > 0 else ""
    msg = f"{emoji} Position Closed for {symbol}:\n"
    msg += f"Type: {deal['type']}\n"
    msg += f"Exit: ${deal['price']:.2f}\n"
    msg += f"Profit/Loss: ${deal['profit']:.2f}\n"
    msg += "-------------------\n"
    return msg

def send_telegram_message(message):
    """Send message via Telegram"""
    try:
        bot_token = "**********"
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
         "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"c "**********"h "**********"a "**********"t "**********"_ "**********"i "**********"d "**********": "**********"
            logging.error("Telegram credentials not found in environment variables")
            return False
            
        url = f"https: "**********"
        data = {
            "chat_id": chat_id,
            "text": message
        }
        
        response = requests.post(url, json=data)
        logging.info(f"Telegram API Response: {response.status_code}")
        logging.info(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            return True
        else:
            logging.error(f"Failed to send Telegram message: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Error sending Telegram message: {str(e)}")
        return False

def get_system_metrics():
    """Get system resource usage"""
    logging.info("\nChecking system metrics...")
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024 ** 3),
            'memory_total_gb': memory.total / (1024 ** 3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024 ** 3),
            'disk_total_gb': disk.total / (1024 ** 3)
        }
        logging.info(f"Current metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error getting metrics: {str(e)}")
        return None

def check_mt5_process():
    """Check if MT5 is running"""
    logging.info("\nChecking MT5 process...")
    try:
        for proc in psutil.process_iter(['name']):
            if 'terminal64.exe' in proc.info['name'].lower():
                logging.info("MT5 is running")
                return True
        logging.info("MT5 is not running")
        return False
    except Exception as e:
        logging.error(f"Error checking MT5: {str(e)}")
        return False

def check_bot_process():
    """Check if trading bot is running"""
    logging.info("\nChecking bot process...")
    try:
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if any('multi_pair_bot.py' in cmd for cmd in cmdline):
                        logging.info("Trading bot is running")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        logging.info("Trading bot is not running")
        return False
    except Exception as e:
        logging.error(f"Error checking bot: {str(e)}")
        return False

def get_next_trading_times():
    """Get next trading times for each pair"""
    now = datetime.now(timezone.utc)
    
    next_times = {}
    for pair, schedules in TRADING_SCHEDULES.items():
        pair_times = []
        for session, times in schedules.items():
            wib_time = datetime.strptime(times['wib'], '%H:%M').time()
            wib_datetime = datetime.combine(now.date(), wib_time)
            wib_datetime = wib_datetime.astimezone(timezone(timedelta(hours=7)))  # Convert to WIB
            
            if wib_datetime.time() < now.astimezone(timezone(timedelta(hours=7))).time():
                pair_times.append(f"{session.capitalize()}: Tomorrow {times['wib']} WIB")
            else:
                pair_times.append(f"{session.capitalize()}: Today {times['wib']} WIB")
        
        next_times[pair] = "\n".join(pair_times)
    
    return next_times

def check_bot_status():
    """Check if trading bot is running"""
    try:
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if any('multi_pair_bot.py' in cmd for cmd in cmdline):
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    except Exception as e:
        logging.error(f"Error checking bot status: {str(e)}")
        return False

def check_momentum(timeframe=mt5.TIMEFRAME_M15):
    """Check last 3 candles for momentum confirmation"""
    try:
        candles = mt5.copy_rates_from_pos("BTCUSD", timeframe, 0, 3)
        if candles is None:
            return "NEUTRAL"
            
        # Convert to list for easier indexing (newest to oldest)
        candles = list(reversed(candles))
        
        # Function to check if candle is doji
        def is_doji(candle, threshold=5):  # threshold in points for BTCUSD
            return abs(candle['close'] - candle['open']) <= threshold
            
        # Function to get candle type
        def get_candle_type(candle):
            if is_doji(candle):
                return "DOJI"
            return "BULL" if candle['close'] > candle['open'] else "BEAR"
        
        # Get candle types
        candle_types = [get_candle_type(candle) for candle in candles]
        
        # Count consecutive non-doji candles
        consecutive_bullish = 0
        consecutive_bearish = 0
        
        # Check most recent candles first
        for i, candle_type in enumerate(candle_types):
            if candle_type == "DOJI":
                continue
                
            if i == 0:  # First non-doji candle sets direction
                if candle_type == "BULL":
                    consecutive_bullish = 1
                else:
                    consecutive_bearish = 1
            else:
                if (candle_type == "BULL" and consecutive_bullish > 0):
                    consecutive_bullish += 1
                elif (candle_type == "BEAR" and consecutive_bearish > 0):
                    consecutive_bearish += 1
                else:
                    break
                    
        if consecutive_bullish >= 2:
            return "UP"
        elif consecutive_bearish >= 2:
            return "DOWN"
        return "NEUTRAL"
        
    except Exception as e:
        logging.error(f"Error in momentum check: {str(e)}")
        return "NEUTRAL"

def check_trend_strength(timeframe=mt5.TIMEFRAME_M15):
    """Check trend strength using multiple factors"""
    try:
        # Get last 10 candles for trend analysis
        candles = mt5.copy_rates_from_pos("BTCUSD", timeframe, 0, 10)
        if candles is None:
            return "NEUTRAL", 0
            
        candles = list(reversed(candles))  # Newest first
        
        # 1. Calculate average body size
        body_sizes = [abs(c['close'] - c['open']) for c in candles]
        avg_body = sum(body_sizes) / len(body_sizes)
        
        # 2. Check higher highs/lower lows
        higher_highs = 0
        lower_lows = 0
        for i in range(1, len(candles)):
            if candles[i-1]['high'] > candles[i]['high']:
                higher_highs += 1
            if candles[i-1]['low'] < candles[i]['low']:
                lower_lows += 1
                
        # 3. Calculate volume trend
        volumes = [c['tick_volume'] for c in candles]
        avg_volume = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[:3]) / 3
        volume_increasing = recent_volume > avg_volume
        
        # Calculate trend score
        score = 0
        
        # Body size factor
        recent_bodies = body_sizes[:3]
        for i, body in enumerate(recent_bodies):
            if body > avg_body:
                if candles[i]['close'] > candles[i]['open']:
                    score += 10
                else:
                    score -= 10
                    
        # Higher highs/lower lows factor
        score += (higher_highs - lower_lows) * 8
        
        # Volume factor
        if volume_increasing:
            last_3_trend = sum(1 for i in range(3) if candles[i]['close'] > candles[i]['open'])
            if last_3_trend >= 2:
                score += 30
            elif last_3_trend <= 1:
                score -= 30
                
        # Get trend category
        if score > 60:
            return "STRONG_BULL", score
        elif score > 30:
            return "BULL", score
        elif score < -60:
            return "STRONG_BEAR", score
        elif score < -30:
            return "BEAR", score
        return "NEUTRAL", score
        
    except Exception as e:
        logging.error(f"Error checking trend strength: {str(e)}")
        return "NEUTRAL", 0

def get_market_update():
    """Get current market prices and analysis"""
    try:
        if not connect_mt5():
            return "❌ Failed to connect to MT5"
            
        # Get bot status
        bot_status = "🟢 RUNNING" if check_bot_status() else "🔴 STOPPED"
            
        # Get positions
        positions = mt5.positions_get(symbol="BTCUSD")
        if positions is None:
            positions = []
            
        # Get current market data
        tick = mt5.symbol_info_tick("BTCUSD")
        if tick is None:
            return "❌ Failed to get market data"
            
        daily_open = get_daily_open()
        if daily_open is None:
            return "❌ Failed to get daily open"
            
        current_price = (tick.bid + tick.ask) / 2
        distance = current_price - daily_open
        momentum = check_momentum()
        trend_direction, trend_score = check_trend_strength()
        
        # Format the message
        message = f"""Market Update

BTCUSD Trading Conditions:
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Daily Open: {daily_open:.2f}
Current Price: {current_price:.2f}
Distance: {distance:.2f} points (Need ±1000)

Analysis:
• Momentum: {momentum}
• Trend: {trend_direction} (Score: {trend_score})
• Volatility: {'Normal' if is_volatility_normal() else 'High'}

Current Prices:
• Bid: {tick.bid:.2f}
• Ask: {tick.ask:.2f}
• Spread: {(tick.ask - tick.bid):.1f}

Bot Status:
{bot_status}"""

        # Add position info if exists
        if positions:
            pos = positions[0]  # Get first position
            pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            message += f"""

Active Position:
• Type: {pos_type}
• Entry: {pos.price_open:.2f}
• SL: {pos.sl:.2f}
• Profit: {pos.profit:.2f}$ ({(pos.price_current - pos.price_open if pos_type == 'BUY' else pos.price_open - pos.price_current):.0f} pts)"""
            
        return message
        
    except Exception as e:
        logging.error(f"Error getting market update: {str(e)}")
        return f"❌ Error: {str(e)}"

def get_daily_open():
    """Get daily open price"""
    try:
        rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_D1, 0, 1)
        if rates is None or len(rates) == 0:
            return None
        return rates[0]['open']
    except Exception as e:
        logging.error(f"Error getting daily open: {str(e)}")
        return None

def is_volatility_normal():
    """Check if current volatility is within acceptable range"""
    try:
        # Get last 20 candles
        candles = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M15, 0, 20)
        if candles is None:
            return False
            
        # Calculate average range
        ranges = [candle['high'] - candle['low'] for candle in candles]
        avg_range = sum(ranges) / len(ranges)
        current_range = ranges[0]
        
        # If current range is more than 2x average, volatility is high
        return current_range <= (2 * avg_range)
        
    except Exception as e:
        logging.error(f"Error checking volatility: {str(e)}")
        return False

def monitor_system():
    """Main monitoring function"""
    try:
        last_market_update = 0
        update_interval = 300  # 5 minutes in seconds
        
        while True:
            current_time = time.time()
            
            # System metrics check (every minute)
            metrics = get_system_metrics()
            if metrics["cpu"] > 90 or metrics["memory_percent"] > 90 or metrics["disk_percent"] > 90:
                alert_message = " System Resource Alert:\n"
                alert_message += f"CPU: {metrics['cpu']}%\n"
                alert_message += f"Memory: {metrics['memory_percent']}%\n"
                alert_message += f"Disk: {metrics['disk_percent']}%"
                send_telegram_message(alert_message)
            
            # Check MT5 and bot processes
            mt5_running = check_mt5_process()
            bot_running = check_bot_process()
            
            if not mt5_running or not bot_running:
                status_message = " Process Status Alert:\n"
                status_message += f"MT5: {'' if mt5_running else ''}\n"
                status_message += f"Trading Bot: {'' if bot_running else ''}"
                send_telegram_message(status_message)
            
            # Market update every 5 minutes
            if current_time - last_market_update >= update_interval:
                market_update = get_market_update()
                if market_update:
                    send_telegram_message(market_update)
                last_market_update = current_time
            
            # Sleep for 1 minute
            time.sleep(60)
            
    except Exception as e:
        logging.error(f"Monitor system error: {str(e)}")
        send_telegram_message(f" VPS Monitor Error: {str(e)}")
        # Try to restart monitoring after error
        time.sleep(60)
        monitor_system()

def stop_bot():
    """Stop the trading bot"""
    try:
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if any('multi_pair_bot.py' in cmd for cmd in cmdline):
                        proc.terminate()
                        logging.info("Trading bot stopped")
                        send_telegram_message("🛑 Trading bot has been stopped")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        logging.info("Trading bot not found")
        send_telegram_message("❌ Trading bot was not running")
        return False
    except Exception as e:
        logging.error(f"Error stopping bot: {str(e)}")
        send_telegram_message(f"❌ Error stopping bot: {str(e)}")
        return False

def start_bot():
    """Start the trading bot"""
    try:
        # Check if bot is already running
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if any('multi_pair_bot.py' in cmd for cmd in cmdline):
                        logging.info("Trading bot is already running")
                        send_telegram_message("⚠️ Trading bot is already running")
                        return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # Start the bot
        import subprocess
        bot_path = os.path.join(os.path.dirname(__file__), 'multi_pair_bot.py')
        subprocess.Popen(['python', bot_path], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        logging.info("Trading bot started")
        send_telegram_message("🟢 Trading bot has been started")
        return True
        
    except Exception as e:
        logging.error(f"Error starting bot: {str(e)}")
        send_telegram_message(f"❌ Error starting bot: {str(e)}")
        return False

def handle_telegram_updates():
    offset = None
    while True:
        try:
            url = f"https: "**********"
            if offset:
                url += f"?offset={offset}"
            response = requests.get(url)
            updates = response.json()
            
            if updates.get("ok") and updates.get("result"):
                for update in updates["result"]:
                    offset = update["update_id"] + 1
                    if "message" in update and "text" in update["message"]:
                        command = update["message"]["text"].lower()
                        
                        if command == "/stop":
                            stop_bot()
                        elif command == "/start":
                            start_bot()
                        elif command == "/status":
                            market_update = get_market_update()
                            send_telegram_message(market_update)
                        elif command == "/help":
                            help_message = """
🤖 *Trading Bot Commands*
/start - Start the trading bot
/stop - Stop the trading bot
/status - Check market conditions
/help - Show this help message

*Bot Settings:*
• Entry: ±1000 points from daily open
• Stop Loss: 2000 points
• Risk: 1% per trade
• Trailing: Starts at +2000 points profit
"""
                            send_telegram_message(help_message)
                                    
        except Exception as e:
            logging.error(f"Error handling Telegram updates: {str(e)}")
        time.sleep(5)

if __name__ == "__main__":
    try:
        logging.info("=== Initializing VPS Monitor ===")
        
        # Setup Telegram command handler
        telegram_thread = threading.Thread(target=handle_telegram_updates)
        telegram_thread.daemon = True
        telegram_thread.start()
        
        # Send test market update
        logging.info("Sending test market update to Telegram...")
        market_update = get_market_update()
        send_telegram_message(market_update)
        logging.info("Test update sent, starting monitor...")
        
        # Start normal monitoring
        monitor_system()
    except KeyboardInterrupt:
        logging.info("Monitor stopped by user")
    except Exception as e:
        logging.error(f"Monitor error: {str(e)}")
        send_telegram_message(f" Monitor error: {str(e)}")
