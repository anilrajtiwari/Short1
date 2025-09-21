"""
crypto_screener_telegram.py - GitHub Actions Compatible Pump-Dump Screener
--------------------------------------------------------------------------
- Designed to run on GitHub Actions
- Sends results to Telegram
- Optimized for CI/CD environment
- No local file dependencies

Dependencies: pip install ccxt pandas numpy requests
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import requests
import sys
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
COIN_LIST = [
    'DOGE/USDT','SHIB/USDT','PEPE/USDT','WIF/USDT','BONK/USDT','FLOKI/USDT','MEME/USDT',
    'KOMA/USDT','DOGS/USDT','NEIROETH/USDT','1000RATS/USDT','ORDI/USDT','PIPPIN/USDT',
    'BAN/USDT','1000SHIB/USDT','OM/USDT','CHILLGUY/USDT','PONKE/USDT','BOME/USDT',
    'MYRO/USDT','PEOPLE/USDT','PENGU/USDT','SPX/USDT','1000BONK/USDT','PNUT/USDT',
    'FARTCOIN/USDT','HIPPO/USDT','AIXBT/USDT','BRETT/USDT','VINE/USDT','MOODENG/USDT',
    'MUBARAK/USDT','MEW/USDT','POPCAT/USDT','1000FLOKI/USDT','1000CAT/USDT','ACT/USDT',
    'SLERF/USDT','DEGEN/USDT','1000PEPE/USDT'
]

EXCHANGE_LIST = ['binance', 'bybit', 'kucoin']

# Trading parameters
ROLL_1H = 100
ROLL_15M = 200
WATCH_HOURS = 1
RECENT_PUMP_HOURS = 12

# Thresholds
Z_RET_BASE = 1.6  # Slightly lower for more signals
Z_VOL_BASE = 1.8
VOL_MULT_15 = 2.5  # Lower for more sensitivity
BETA_CLOSE_NEAR_LOW = 0.35

# Technical indicators
EMA_FAST = 21
EMA_SLOW = 50

# Scoring
MIN_SCORE = 0.20  # Lower threshold for GitHub environment

# ---------------- TELEGRAM FUNCTIONS ----------------
def send_telegram_message(message):
    """Send message to Telegram"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("ERROR: Telegram credentials not found in environment variables")
        print("Required: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("Message sent to Telegram successfully")
            return True
        else:
            print(f"Failed to send Telegram message: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def format_telegram_message(signals):
    """Format signals for Telegram"""
    if not signals:
        return """🔍 *CRYPTO PUMP-DUMP SCREENER*
        
❌ No short signals detected
Market conditions not favorable

⏰ Next scan in 30 minutes
🤖 Automated via GitHub Actions"""

    message = f"🎯 *CRYPTO SHORT SIGNALS* ({len(signals)} found)\n"
    message += "=" * 30 + "\n\n"
    
    for i, s in enumerate(signals, 1):
        pump_time = s['pump_time'].strftime('%H:%M UTC')
        confirm_time = s['confirm_time'].strftime('%H:%M UTC')
        
        message += f"*#{i} {s['ticker']}*\n"
        message += f"📊 Score: `{s['score']:.3f}`\n"
        message += f"⬆️ Pump: {pump_time}\n"
        message += f"⬇️ Dump: {confirm_time}\n"
        message += f"🏛️ Exchange: {s.get('exchange', 'N/A')}\n"
        message += f"📉 Action: *SHORT {s['ticker'].replace('/USDT', '')}*\n"
        message += "—" * 25 + "\n\n"
    
    message += "⚠️ *TRADING NOTES:*\n"
    message += "• Use 2-3% stop losses\n"
    message += "• Target 5-10% profit\n"
    message += "• Monitor 15m charts\n"
    message += "• Risk management essential\n\n"
    
    message += f"⏰ Scanned at: {datetime.now().strftime('%H:%M UTC')}\n"
    message += "🤖 Automated via GitHub Actions"
    
    return message

# ---------------- CORE FUNCTIONS ----------------
def get_exchange(exchange_id):
    """Get exchange instance with error handling"""
    try:
        ex_class = getattr(ccxt, exchange_id)
        exchange = ex_class({
            'enableRateLimit': True,
            'timeout': 30000,
            'rateLimit': 1200
        })
        return exchange
    except Exception as e:
        print(f"Error creating {exchange_id} exchange: {e}")
        return None

def fetch_ohlcv_data(symbol, timeframe, limit=500):
    """Fetch OHLCV data with fallback exchanges"""
    for ex_id in EXCHANGE_LIST:
        ex = get_exchange(ex_id)
        if not ex:
            continue
            
        try:
            ex.load_markets()
            if not ex.has.get('fetchOHLCV', False):
                continue
                
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                continue
                
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
            
            # Check data freshness (24 hours for GitHub environment)
            latest_time = df['datetime'].max()
            current_time = pd.Timestamp.now(tz='UTC')
            hours_old = (current_time - latest_time).total_seconds() / 3600
            
            if hours_old > 24:
                continue
                
            return df.dropna().sort_values('datetime').reset_index(drop=True), ex_id
            
        except Exception as e:
            print(f"Error fetching {symbol} from {ex_id}: {e}")
            continue
    
    return None, None

def add_technical_indicators(df):
    """Add EMA indicators"""
    if df is None or df.empty or len(df) < EMA_SLOW:
        return df
    
    df = df.copy()
    df['EMA_21'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    return df

def calculate_z_scores(df):
    """Calculate Z-scores for returns and volume"""
    if len(df) < ROLL_1H:
        return df
    
    df = df.copy()
    
    # Return Z-scores
    df['ret_pct'] = df['close'].pct_change() * 100
    df['ret_mean'] = df['ret_pct'].rolling(ROLL_1H, min_periods=20).mean()
    df['ret_std'] = df['ret_pct'].rolling(ROLL_1H, min_periods=20).std().replace(0, np.nan)
    df['z_ret'] = (df['ret_pct'] - df['ret_mean']) / df['ret_std']
    
    # Volume Z-scores
    df['vol_mean'] = df['volume'].rolling(ROLL_1H, min_periods=20).mean()
    df['vol_std'] = df['volume'].rolling(ROLL_1H, min_periods=20).std().replace(0, np.nan)
    df['z_vol'] = (df['volume'] - df['vol_mean']) / df['vol_std']
    
    return df

def calculate_15m_volume_stats(df):
    """Calculate 15m volume statistics"""
    if len(df) < 40:
        return df
    
    df = df.copy()
    df['vol_mean_15m'] = df['volume'].rolling(ROLL_15M, min_periods=20).mean()
    return df

def get_dynamic_thresholds(df):
    """Calculate dynamic thresholds based on volatility"""
    if df is None or df.empty:
        return Z_RET_BASE, Z_VOL_BASE
    
    recent_returns = df['close'].pct_change().dropna()[-50:]
    if len(recent_returns) < 10:
        return Z_RET_BASE, Z_VOL_BASE
    
    volatility = recent_returns.std()
    
    # Adjust thresholds
    if volatility > 0.05:  # High volatility
        z_ret = Z_RET_BASE + 0.3
        z_vol = Z_VOL_BASE + 0.2
    elif volatility < 0.02:  # Low volatility
        z_ret = Z_RET_BASE - 0.2
        z_vol = Z_VOL_BASE - 0.2
    else:
        z_ret = Z_RET_BASE
        z_vol = Z_VOL_BASE
    
    return max(1.2, z_ret), max(1.5, z_vol)

def detect_pumps(df, z_ret_thresh, z_vol_thresh):
    """Detect recent pump signals"""
    if df.empty:
        return []
    
    current_time = pd.Timestamp.now(tz='UTC')
    cutoff_time = current_time - timedelta(hours=RECENT_PUMP_HOURS)
    
    recent_df = df[df['datetime'] >= cutoff_time].copy()
    if recent_df.empty:
        return []
    
    # Basic pump conditions
    pump_mask = (recent_df['z_ret'] >= z_ret_thresh) & (recent_df['z_vol'] >= z_vol_thresh)
    
    # EMA trend filter
    if 'EMA_21' in recent_df.columns:
        ema_filter = recent_df['close'] > recent_df['EMA_21']
        pump_mask = pump_mask & ema_filter
    
    return recent_df.index[pump_mask].tolist()

def is_bearish_dump_signal(bar):
    """Check if 15m bar shows bearish dump characteristics"""
    if pd.isna(bar['vol_mean_15m']) or bar['vol_mean_15m'] == 0:
        return False
    
    # Price conditions
    range_size = bar['high'] - bar['low']
    if range_size <= 0:
        return False
    
    close_near_low = bar['close'] <= bar['low'] + BETA_CLOSE_NEAR_LOW * range_size
    bearish_candle = bar['close'] < bar['open']
    
    # Volume spike
    vol_spike = bar['volume'] >= VOL_MULT_15 * bar['vol_mean_15m']
    
    # Significant price drop
    price_drop = (bar['open'] - bar['close']) / bar['open']
    significant_drop = price_drop > 0.01  # At least 1% drop
    
    # EMA break (if available)
    ema_break = True
    if 'EMA_21' in bar.index and not pd.isna(bar.get('EMA_21')):
        ema_break = bar['close'] < bar['EMA_21']
    
    return bearish_candle and close_near_low and vol_spike and significant_drop and ema_break

def calculate_signal_score(pump_bar, dump_bar):
    """Calculate signal strength score"""
    # Z-score components
    z_ret = float(pump_bar.get('z_ret', 0))
    z_vol = float(pump_bar.get('z_vol', 0))
    
    s1 = min(1.0, max(0.0, z_ret / 4.0))
    s2 = min(1.0, max(0.0, z_vol / 5.0))
    
    # Volume ratio
    vol_ratio = dump_bar['volume'] / max(dump_bar.get('vol_mean_15m', 1), 1)
    s3 = min(1.0, max(0.0, vol_ratio / (VOL_MULT_15 * 1.5)))
    
    # Time decay
    time_diff = (dump_bar['datetime'] - pump_bar['datetime']).total_seconds() / 60.0
    s4 = max(0.0, 1.0 - (time_diff / (WATCH_HOURS * 60.0)))
    
    # Price drop magnitude
    price_drop = (dump_bar['open'] - dump_bar['close']) / dump_bar['open']
    s5 = min(1.0, max(0.0, price_drop / 0.04))
    
    return 0.3*s1 + 0.25*s2 + 0.2*s3 + 0.1*s4 + 0.15*s5

# ---------------- MAIN SCANNER ----------------
def scan_crypto_signals():
    """Main scanning function"""
    print(f"Starting crypto scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    signals = []
    processed = 0
    
    for coin in COIN_LIST:
        try:
            processed += 1
            print(f"Processing {coin} ({processed}/{len(COIN_LIST)})")
            
            # Fetch hourly data
            df_1h, ex_1h = fetch_ohlcv_data(coin, '1h', ROLL_1H + 50)
            if df_1h is None or len(df_1h) < ROLL_1H:
                continue
            
            # Fetch 15m data
            df_15m, ex_15m = fetch_ohlcv_data(coin, '15m', ROLL_15M + 100)
            if df_15m is None or len(df_15m) < 40:
                continue
            
            # Add technical indicators
            df_1h = add_technical_indicators(calculate_z_scores(df_1h))
            df_15m = add_technical_indicators(calculate_15m_volume_stats(df_15m))
            
            # Get dynamic thresholds
            z_ret_thresh, z_vol_thresh = get_dynamic_thresholds(df_1h)
            
            # Detect pumps
            pump_indices = detect_pumps(df_1h, z_ret_thresh, z_vol_thresh)
            if not pump_indices:
                continue
            
            print(f"  Found {len(pump_indices)} potential pumps in {coin}")
            
            # Look for confirmations
            for pump_idx in pump_indices:
                pump_bar = df_1h.iloc[pump_idx]
                pump_time = pump_bar['datetime']
                watch_end = pump_time + timedelta(hours=WATCH_HOURS)
                
                # Get 15m confirmation window
                confirmation_window = df_15m[
                    (df_15m['datetime'] > pump_time) & 
                    (df_15m['datetime'] <= watch_end)
                ]
                
                if confirmation_window.empty:
                    continue
                
                # Look for bearish dump signal
                for _, bar in confirmation_window.iterrows():
                    if is_bearish_dump_signal(bar):
                        score = calculate_signal_score(pump_bar, bar)
                        
                        if score >= MIN_SCORE:
                            signals.append({
                                'ticker': coin,
                                'score': score,
                                'pump_time': pump_time,
                                'confirm_time': bar['datetime'],
                                'exchange': ex_1h or ex_15m
                            })
                            print(f"  SIGNAL: {coin} - Score: {score:.3f}")
                            break
                
                if signals and signals[-1]['ticker'] == coin:
                    break  # Only one signal per coin
            
        except Exception as e:
            print(f"Error processing {coin}: {e}")
            continue
    
    # Sort by score
    signals.sort(key=lambda x: x['score'], reverse=True)
    print(f"Scan complete. Found {len(signals)} signals.")
    
    return signals

# ---------------- MAIN EXECUTION ----------------
def main():
    """Main function for GitHub Actions"""
    try:
        print("=== CRYPTO PUMP-DUMP SCREENER ===")
        print("Running on GitHub Actions")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("-" * 50)
        
        # Run the scan
        signals = scan_crypto_signals()
        
        # Format and send to Telegram
        message = format_telegram_message(signals)
        success = send_telegram_message(message)
        
        if success:
            print("Results sent to Telegram successfully")
        else:
            print("Failed to send results to Telegram")
            
        # Also print to console for GitHub Actions logs
        print("\n" + "=" * 50)
        print("SCAN RESULTS:")
        if signals:
            for i, s in enumerate(signals, 1):
                print(f"{i}. {s['ticker']} - Score: {s['score']:.3f}")
        else:
            print("No signals detected")
        
        print("=" * 50)
        return 0 if success else 1
        
    except Exception as e:
        error_msg = f"🚨 SCREENER ERROR\n\nFailed to run crypto scan:\n`{str(e)}`\n\nTime: {datetime.now().strftime('%H:%M UTC')}"
        send_telegram_message(error_msg)
        print(f"Critical error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
