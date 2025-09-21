"""
rt_github_improved.py — Enhanced Pump→Dump Screener
-------------------------------------------------
- Fixed static data issues with proper timestamp filtering
- Added signal cooldown to prevent repeated signals
- Dynamic thresholds based on coin volatility
- EMA trend confirmation
- Better logging and deduplication

Dependencies: pip install ccxt pandas numpy tqdm
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------------- USER CONFIG ----------------
COIN_LIST = [
    'DOGE/USDT','SHIB/USDT','PEPE/USDT','WIF/USDT','BONK/USDT','FLOKI/USDT','MEME/USDT',
    'KOMA/USDT','DOGS/USDT','NEIROETH/USDT','1000RATS/USDT','ORDI/USDT','PIPPIN/USDT',
    'BAN/USDT','1000SHIB/USDT','OM/USDT','CHILLGUY/USDT','PONKE/USDT','BOME/USDT',
    'MYRO/USDT','PEOPLE/USDT','PENGU/USDT','SPX/USDT','1000BONK/USDT','PNUT/USDT',
    'FARTCOIN/USDT','HIPPO/USDT','AIXBT/USDT','BRETT/USDT','VINE/USDT','MOODENG/USDT',
    'MUBARAK/USDT','MEW/USDT','POPCAT/USDT','1000FLOKI/USDT','1000CAT/USDT','ACT/USDT',
    'SLERF/USDT','DEGEN/USDT','1000PEPE/USDT'
]

EXCHANGE_LIST = ['binance', 'bybit', 'kucoin']  # Removed 'okx' due to API issues

ROLL_1H = 100
ROLL_15M = 200
WATCH_HOURS = 1
SIGNAL_COOLDOWN_HOURS = 6  # Don't re-signal same coin for 6 hours
RECENT_PUMP_HOURS = 12      # Only look for pumps in last 12 hours (increased for testing)

# Base thresholds (will be adjusted dynamically)
Z_RET_BASE = 1.8
Z_VOL_BASE = 2.0
VOL_MULT_15 = 3.0
BETA_CLOSE_NEAR_LOW = 0.30

EMA_FAST = 21
EMA_SLOW = 50

# Files for persistence
SIGNALS_LOG_FILE = "signals_log.json"
LAST_SIGNALS_FILE = "last_signals.json"

# Enable debug logging
DEBUG = False  # Set to False for clean output

# ---------------- HELPERS ----------------
def log_debug(message):
    if DEBUG:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def load_signal_history():
    """Load previous signals to implement cooldown"""
    if os.path.exists(LAST_SIGNALS_FILE):
        try:
            with open(LAST_SIGNALS_FILE, 'r') as f:
                data = json.load(f)
                # Convert string timestamps back to datetime
                for coin in data:
                    if data[coin]:
                        data[coin] = datetime.fromisoformat(data[coin])
                return data
        except Exception as e:
            log_debug(f"Error loading signal history: {e}")
    return {}

def save_signal_history(history):
    """Save signal history for cooldown tracking"""
    try:
        # Convert datetime to string for JSON serialization
        save_data = {}
        for coin, last_time in history.items():
            save_data[coin] = last_time.isoformat() if last_time else None
        
        with open(LAST_SIGNALS_FILE, 'w') as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        log_debug(f"Error saving signal history: {e}")

def is_in_cooldown(coin, signal_history, current_time):
    """Check if coin is in cooldown period"""
    last_signal_time = signal_history.get(coin)
    if last_signal_time:
        hours_since = (current_time - last_signal_time).total_seconds() / 3600
        return hours_since < SIGNAL_COOLDOWN_HOURS
    return False

def log_signal(coin, pump_time, confirm_time, score, exchange):
    """Log detected signals"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'coin': coin,
            'pump_time': pump_time.isoformat(),
            'confirm_time': confirm_time.isoformat(),
            'score': score,
            'exchange': exchange
        }
        
        # Load existing log
        log_data = []
        if os.path.exists(SIGNALS_LOG_FILE):
            with open(SIGNALS_LOG_FILE, 'r') as f:
                log_data = json.load(f)
        
        log_data.append(log_entry)
        
        # Keep only last 1000 entries
        if len(log_data) > 1000:
            log_data = log_data[-1000:]
        
        with open(SIGNALS_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)
            
    except Exception as e:
        log_debug(f"Error logging signal: {e}")

def get_exchange(exchange_id):
    try:
        ex_class = getattr(ccxt, exchange_id)
        return ex_class({'enableRateLimit': True})
    except Exception:
        return None

def fetch_ohlcv_fallback(symbol: str, timeframe: str, limit: int = 500):
    """Fetch OHLCV with fallback across multiple exchanges (simple & reliable)"""
    for ex_id in EXCHANGE_LIST:
        ex = get_exchange(ex_id)
        if ex is None:
            continue
        try:
            ex.load_markets()
            if not getattr(ex, 'has', {}).get('fetchOHLCV', True):
                continue

            ohl = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohl:
                continue

            df = pd.DataFrame(ohl, columns=['ts', 'open','high','low','close','volume'])
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True)

            # Clean & sort
            df = df.dropna().sort_values('datetime').reset_index(drop=True)
            log_debug(f"Fetched {len(df)} candles for {symbol} from {ex_id}, latest: {df['datetime'].max()}")
            return df, ex_id

        except Exception as e:
            log_debug(f"Error fetching {symbol} from {ex_id}: {e}")
            continue
    return None, None

def add_emas(df):
    """Add EMA indicators"""
    if df is None or df.empty or len(df) < EMA_SLOW: 
        return df
    df = df.copy()
    df['EMA_21'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    return df

def calculate_dynamic_thresholds(df):
    """Calculate dynamic thresholds based on coin's volatility"""
    if df is None or df.empty:
        return Z_RET_BASE, Z_VOL_BASE
    
    # Calculate recent volatility
    recent_returns = df['close'].pct_change().dropna()[-50:]  # Last 50 periods
    if len(recent_returns) < 10:
        return Z_RET_BASE, Z_VOL_BASE
    
    volatility = recent_returns.std()
    
    # Adjust thresholds based on volatility
    if volatility > 0.05:  # Very volatile (>5% moves)
        z_ret_threshold = Z_RET_BASE + 0.5
        z_vol_threshold = Z_VOL_BASE + 0.3
    elif volatility < 0.02:  # Low volatility (<2% moves)
        z_ret_threshold = Z_RET_BASE - 0.3
        z_vol_threshold = Z_VOL_BASE - 0.2
    else:
        z_ret_threshold = Z_RET_BASE
        z_vol_threshold = Z_VOL_BASE
    
    return max(1.2, z_ret_threshold), max(1.5, z_vol_threshold)

# ---------------- DETECTION ----------------
def compute_1h_stats(df_1h):
    if len(df_1h) < ROLL_1H:
        return df_1h
    
    df = df_1h.copy()
    df['ret_pct'] = df['close'].pct_change() * 100
    df['ret_mean'] = df['ret_pct'].rolling(ROLL_1H, min_periods=20).mean()
    df['ret_std'] = df['ret_pct'].rolling(ROLL_1H, min_periods=20).std().replace(0, np.nan)
    df['z_ret'] = (df['ret_pct'] - df['ret_mean']) / df['ret_std']
    
    df['vol_mean_1h'] = df['volume'].rolling(ROLL_1H, min_periods=20).mean()
    df['vol_std_1h'] = df['volume'].rolling(ROLL_1H, min_periods=20).std().replace(0, np.nan)
    df['z_vol'] = (df['volume'] - df['vol_mean_1h']) / df['vol_std_1h']
    
    return df

def compute_15m_stats(df_15m):
    if len(df_15m) < 40:
        return df_15m
    
    df = df_15m.copy()
    df['vol_mean_15m'] = df['volume'].rolling(ROLL_15M, min_periods=20).mean()
    return df

def detect_recent_pumps_1h(df_1h_stats, z_ret_threshold, z_vol_threshold):
    """Only detect pumps in the last few hours, not just last 24 data points"""
    if df_1h_stats.empty:
        return []
    
    current_time = pd.Timestamp.now(tz='UTC')
    recent_cutoff = current_time - timedelta(hours=RECENT_PUMP_HOURS)
    
    recent_df = df_1h_stats[df_1h_stats['datetime'] >= recent_cutoff].copy()
    if recent_df.empty:
        return []
    
    mask = (recent_df['z_ret'] >= z_ret_threshold) & (recent_df['z_vol'] >= z_vol_threshold)
    
    if 'EMA_21' in recent_df.columns:
        ema_filter = recent_df['close'] > recent_df['EMA_21']
        mask = mask & ema_filter
    
    pump_indices = recent_df.index[mask].tolist()
    
    if pump_indices:
        log_debug(f"Found {len(pump_indices)} recent pumps in last {RECENT_PUMP_HOURS} hours")
    
    return pump_indices

def is_15m_bearish_vol_spike(bar_15m):
    """Improved bearish volume spike detection"""
    rng = bar_15m['high'] - bar_15m['low']
    if rng <= 0: 
        return False
    
    close_near_low = bar_15m['close'] <= bar_15m['low'] + BETA_CLOSE_NEAR_LOW * rng
    bearish = bar_15m['close'] < bar_15m['open']
    
    mean15 = bar_15m.get('vol_mean_15m', np.nan)
    if pd.isna(mean15) or mean15 == 0: 
        return False
    vol_spike = bar_15m['volume'] >= VOL_MULT_15 * mean15
    
    price_drop = (bar_15m['open'] - bar_15m['close']) / bar_15m['open']
    significant_drop = price_drop > 0.015  
    
    ema_break = True
    if 'EMA_21' in bar_15m.index and not pd.isna(bar_15m.get('EMA_21')):
        ema_break = bar_15m['close'] < bar_15m['EMA_21']
    
    return bearish and close_near_low and vol_spike and significant_drop and ema_break

# ---------------- SCORING ----------------
def compute_signal_score(pump_row, confirm_row):
    """Enhanced scoring with more factors"""
    zret = float(pump_row.get('z_ret', 0.0))
    zvol = float(pump_row.get('z_vol', 0.0))
    
    s1 = min(1.0, max(0.0, zret / 5.0))
    s2 = min(1.0, max(0.0, zvol / 6.0))
    
    vol_ratio = confirm_row['volume'] / max(confirm_row.get('vol_mean_15m', 1.0), 1.0)
    s3 = min(1.0, max(0.0, vol_ratio / (VOL_MULT_15 * 2)))
    
    dt = (confirm_row['datetime'] - pump_row['datetime']).total_seconds() / 60.0
    s4 = max(0.0, 1.0 - (dt / (WATCH_HOURS * 60.0)))
    
    price_drop = (confirm_row['open'] - confirm_row['close']) / confirm_row['open']
    s5 = min(1.0, max(0.0, price_drop / 0.05))
    
    s6 = 1.0
    if 'EMA_21' in pump_row.index and 'EMA_21' in confirm_row.index:
        pump_above_ema = pump_row['close'] > pump_row.get('EMA_21', pump_row['close'])
        confirm_below_ema = confirm_row['close'] < confirm_row.get('EMA_21', confirm_row['close'])
        s6 = 1.0 if (pump_above_ema and confirm_below_ema) else 0.7
    
    final_score = 0.25*s1 + 0.20*s2 + 0.20*s3 + 0.10*s4 + 0.15*s5 + 0.10*s6
    return float(final_score)

# ---------------- MAIN ----------------
def scan_once():
    """Enhanced scan with all improvements"""
    signals = []
    signal_history = load_signal_history()
    current_time = datetime.now()
    
    log_debug(f"Starting scan of {len(COIN_LIST)} coins...")
    
    for coin in tqdm(COIN_LIST, desc="Scanning coins", unit="coin"):
        try:
            if is_in_cooldown(coin, signal_history, current_time):
                log_debug(f"Skipping {coin} - in cooldown for {SIGNAL_COOLDOWN_HOURS}h")
                continue
            
            df_1h, ex1 = fetch_ohlcv_fallback(coin, '1h', limit=ROLL_1H+50)
            if df_1h is None or len(df_1h) < ROLL_1H:
                log_debug(f"Insufficient 1h data for {coin}")
                continue
                
            df_15m, ex15 = fetch_ohlcv_fallback(coin, '15m', limit=ROLL_15M+100)
            if df_15m is None or len(df_15m) < 40:
                log_debug(f"Insufficient 15m data for {coin}")
                continue
            
            z_ret_thresh, z_vol_thresh = calculate_dynamic_thresholds(df_1h)
            log_debug(f"{coin} thresholds: z_ret={z_ret_thresh:.2f}, z_vol={z_vol_thresh:.2f}")
            
            df_1h_stats = add_emas(compute_1h_stats(df_1h))
            df_15m_stats = add_emas(compute_15m_stats(df_15m))
            
            if df_1h_stats.empty or df_15m_stats.empty:
                continue
            
            pump_idx_list = detect_recent_pumps_1h(df_1h_stats, z_ret_thresh, z_vol_thresh)
            
            if not pump_idx_list:
                continue
                
            log_debug(f"{coin}: Found {len(pump_idx_list)} potential pumps")
            
            for pidx in pump_idx_list:
                pump_row = df_1h_stats.iloc[pidx]
                pump_time = pump_row['datetime']
                watch_end = pump_time + timedelta(hours=WATCH_HOURS)
                
                win15 = df_15m_stats[
                    (df_15m_stats['datetime'] > pump_time) & 
                    (df_15m_stats['datetime'] <= watch_end)
                ].copy()
                
                if win15.empty:
                    continue
                
                confirmed = None
                for _, bar in win15.iterrows():
                    if is_15m_bearish_vol_spike(bar):
                        confirmed = bar
                        break
                
                if confirmed is not None:
                    score = compute_signal_score(pump_row, confirmed)
                    
                    if score >= 0.25:
                        signals.append({
                            'ticker': coin,
                            'score': score,
                            'pump_time': pump_time,
                            'confirm_time': confirmed['datetime'],
                            'exchange': ex1 or ex15
                        })
                        
                        signal_history[coin] = current_time
                        log_signal(coin, pump_time, confirmed['datetime'], score, ex1 or ex15)
                        log_debug(f"SIGNAL: {coin} - Score: {score:.3f}")
                        break  
        
        except Exception as e:
            log_debug(f"Error processing {coin}: {e}")
            continue
    
    save_signal_history(signal_history)
    signals.sort(key=lambda x: x['score'], reverse=True)
    log_debug(f"Scan complete. Found {len(signals)} signals.")
    return signals

if __name__ == "__main__":
    print("=== CRYPTO PUMP-DUMP SCREENER ===")
    print("Scanning for short opportunities...\n")
    
    results = scan_once()
    
    if not results:
        print("NO SIGNALS DETECTED")
        print("Market conditions not favorable for shorts")
    else:
        print(f"FOUND {len(results)} SHORT SIGNALS")
        print("=" * 50)
        
        for i, s in enumerate(results, 1):
            pump_time = s['pump_time'].strftime('%H:%M UTC')
            confirm_time = s['confirm_time'].strftime('%H:%M UTC')
            
            print(f"#{i} {s['ticker']}")
            print(f"   Score: {s['score']:.3f}")
            print(f"   Pump: {pump_time} -> Dump: {confirm_time}")
            print(f"   Exchange: {s.get('exchange', 'N/A')}")
            print(f"   Action: SHORT {s['ticker'].replace('/USDT', '')}")
            print("-" * 30)
        
        print("\
