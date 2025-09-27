# atm_github.py - Full optimized ATm scanner (tuned params + full detectors + resampling + skip summary + Telegram + CI friendly)
# Copy to your repo and run as a single-run (GitHub Actions friendly). Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as secrets.

import os
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
import argparse

import requests
import ccxt
import pandas as pd
from tabulate import tabulate

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ================== CONFIG (tuned) ==================
COIN_LIST = [
    'DOGE/USDT','SHIB/USDT','PEPE/USDT','WIF/USDT','BONK/USDT','FLOKI/USDT','MEME/USDT',
    'KOMA/USDT','DOGS/USDT','NEIROETH/USDT','1000RATS/USDT','ORDI/USDT','PIPPIN/USDT',
    'BAN/USDT','1000SHIB/USDT','OM/USDT','CHILLGUY/USDT','PONKE/USDT','BOME/USDT',
    'MYRO/USDT','PEOPLE/USDT','PENGU/USDT','SPX/USDT','1000BONK/USDT','PNUT/USDT',
    'FARTCOIN/USDT','HIPPO/USDT','AIXBT/USDT','BRETT/USDT','VINE/USDT','MOODENG/USDT',
    'MUBARAK/USDT','MEW/USDT','POPCAT/USDT','1000FLOKI/USDT','1000CAT/USDT','ACT/USDT',
    'SLERF/USDT','DEGEN/USDT','1000PEPE/USDT'
]

EXCHANGE_LIST = [
    'binance','bybit','kucoin','mexc','gateio','okx','bitget','huobi'
]

SYMBOL_MAP = {
    '1000PEPE/USDT': {'mexc':'PEPE1000/USDT','gateio':'PEPE1000/USDT','bitget':'PEPE1000/USDT'},
    '1000BONK/USDT': {'mexc':'BONK1000/USDT'},
    '1000FLOKI/USDT': {'mexc':'FLOKI1000/USDT'},
    '1000SHIB/USDT': {'mexc':'SHIB1000/USDT'},
    '1000CAT/USDT': {'mexc':'CAT1000/USDT'},
}

# ===== Optimized Parameters (from your backtest results) =====
TIMEFRAME_TARGET = '15m'
LOOKBACK_LIMIT = 400
RECENT_WINDOW_HOURS = 96
RATE_LIMIT_SLEEP_DEFAULT = 0.35

# Pump rules
IMPULSE_WINDOW = 6
IMPULSE_PCT_FLOOR = 0.20
GRADUAL_WINDOW = 24
GRADUAL_PCT_FLOOR = 0.08

# Drop rules (base values)
DUMP_WINDOW = 30
BASE_SAFE_DROP_MIN = 0.08
BASE_AGGRESSIVE_DROP_MIN = 0.18
BASE_AGGRESSIVE_DROP_ALT = 0.12
POTENTIAL_DROP_FLOOR = 0.03

# Standalone dump detectors
SUDDEN_BARS = 4
SUDDEN_PCT_FLOOR = 0.05
SUDDEN_ATR_MULT = 1.2

# Tuned Gradual (EARLY)
GRADUAL_DUMP_WINDOW = 40
GRADUAL_DUMP_PCT_FLOOR = 0.04
GRADUAL_DUMP_EMA_FRAC = 0.7
GRADUAL_DUMP_LHLL_MIN = 3

# Confirmation filters
REQUIRE_VOL_FOR_CONFIRMED = False
REQUIRE_STRUCTURE_FOR_CONFIRMED = True

# Volume multiplier for requiring volume spike on dump bar
VOLUME_SPIKE_MULT = 1.2

# Post-signal validation parameters
POST_VALIDATION_CANDLES = 2
POST_VALIDATION_RECOVERY_FRAC = 0.5  # disallow >50% recovery of the drop

# Multi-timeframe alignment settings
HIGHER_TF = '1h'
HIGHER_EMA_SLOPE_LOOKBACK = 2
HIGHER_RSI_THRESHOLD = 50

# Momentum thresholds
RSI_LEN = 14
RSI_SHORT_THRESHOLD = 40

# ================== LOGGING ==================
logger = logging.getLogger("scan")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("scan.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(fmt)
logger.addHandler(handler)

# ----------------- helpers -----------------

def utcnow():
    return datetime.now(timezone.utc)


def seconds_to_next_15m(now=None):
    if now is None:
        now = utcnow()
    minute = now.minute
    next_q = ((minute//15)+1)*15
    if next_q >= 60:
        next_dt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_dt = now.replace(minute=next_q, second=0, microsecond=0)
    return max(0, int((next_dt - now).total_seconds()))


def load_exchange(eid):
    return getattr(ccxt, eid)({'enableRateLimit': True})


def normalize_base(s):
    b, q = s.split('/')
    return b.upper(), q.upper()


def base_candidates(base):
    out = {base}
    if base.startswith("1000"):
        out.add(base[4:])
    else:
        out.add("1000" + base)
    out.add(base.replace("-", "").replace("_", ""))
    return list(out)


def is_perp_market(m):
    try:
        if getattr(m, 'contract', False) or m.get('contract', False):
            sym = m.get('symbol', '')
            quote = m.get('quote', '')
            return (':USDT' in sym) or (quote == 'USDT')
    except Exception:
        return False
    return False


def resolve_symbol(eid, canonical, markets):
    mapped = SYMBOL_MAP.get(canonical, {}).get(eid)
    if mapped and mapped in markets:
        return mapped, "MAPPED"
    if canonical in markets:
        return canonical, "EXACT"
    base, _ = normalize_base(canonical)
    perp = f"{base}/USDT:USDT"
    if perp in markets:
        return perp, "PERP"
    cands = base_candidates(base)
    best_perp = None
    best_spot = None
    for sym, m in markets.items():
        b = (m.get('base') or '').upper()
        q = (m.get('quote') or '').upper()
        if b in cands and (q == 'USDT' or ':USDT' in sym):
            if is_perp_market(m):
                if best_perp is None:
                    best_perp = sym
            else:
                if best_spot is None:
                    best_spot = sym
    if best_perp:
        return best_perp, "FUZZY_PERP"
    if best_spot:
        return best_spot, "FUZZY_SPOT"
    for sym, m in markets.items():
        b = (m.get('base') or '').upper()
        if b in cands:
            return sym, "FALLBACK"
    return None, "NOT_FOUND"


def pick_timeframe(ex):
    tfs = getattr(ex, 'timeframes', None) or {}
    if TIMEFRAME_TARGET in tfs:
        return TIMEFRAME_TARGET, None
    for tf in ['1m','5m','30m','1h']:
        if tf in tfs:
            return tf, TIMEFRAME_TARGET
    return TIMEFRAME_TARGET, None


def safe_fetch_ohlcv(ex, symbol, tf_req, limit):
    # robust fetch with retries
    for attempt in range(3):
        try:
            return ex.fetch_ohlcv(symbol, tf_req, limit=limit)
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"RATE_LIMIT {ex.id} {symbol} {tf_req} attempt={attempt+1} {e}")
            time.sleep(max(getattr(ex, 'rateLimit', 350)/1000.0, RATE_LIMIT_SLEEP_DEFAULT) * (attempt+1))
        except (ccxt.DDoSProtection, ccxt.NetworkError) as e:
            logger.warning(f"NETWORK {ex.id} {symbol} {tf_req} attempt={attempt+1} {e}")
            time.sleep(1.0 * (attempt+1))
        except ccxt.BadSymbol as e:
            logger.info(f"BAD_SYMBOL {ex.id} {symbol} {tf_req} {e}")
            raise
        except Exception as e:
            logger.error(f"UNKNOWN_FETCH {ex.id} {symbol} {tf_req} {e}")
            time.sleep(0.5)
    # final try
    try:
        return ex.fetch_ohlcv(symbol, tf_req, limit=limit)
    except Exception as e:
        logger.error(f"FINAL_FETCH_FAIL {ex.id} {symbol} {tf_req}: {e}")
        return []


def to_df(ohlcv):
    if not ohlcv:
        return pd.DataFrame(columns=['open','high','low','close','volume']).set_index(pd.DatetimeIndex([], tz='UTC'))
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('dt').drop(columns=['ts']).sort_index()


def resample_to_target(df, target_tf):
    if target_tf is None or df.empty:
        return df
    rule = {'1m':'1T','5m':'5T','15m':'15T','30m':'30T','1h':'1h'}.get(target_tf, '15T')
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    return df.resample(rule).apply(agg).dropna(how='any')


def resample_higher_tf(df, rule='1h'):
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    return df.resample(rule).apply(agg).dropna(how='any')


def add_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    high, low, close, vol = df['high'], df['low'], df['close'], df['volume']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14, min_periods=1).mean()
    df['vol_ma'] = vol.rolling(20, min_periods=1).mean()
    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(RSI_LEN, min_periods=1).mean()
    ma_down = down.rolling(RSI_LEN, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, 1e-9))
    df['rsi'] = 100 - (100 / (1 + rs))
    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

# ----------------- Pump & dump detectors -----------------

def detect_impulse(df):
    if len(df) < IMPULSE_WINDOW + 2:
        return None
    w = df.iloc[-(IMPULSE_WINDOW+2):]
    local_min_idx = w['close'].idxmin()
    local_max_idx = w['close'].idxmax()
    if local_max_idx <= local_min_idx:
        return None
    lowp = float(w.loc[local_min_idx, 'close'])
    highp = float(w.loc[local_max_idx, 'close'])
    pct = (highp - lowp) / lowp if lowp > 0 else 0.0
    if pct >= IMPULSE_PCT_FLOOR:
        peak = local_max_idx
        atr = float(w.loc[peak, 'atr14']) if 'atr14' in w.columns else 0.0
        return {
            'category': 'IMPULSE',
            'pump_pct': float(pct),
            'pump_peak_time': peak,
            'pump_peak_price': float(highp),
            'atr_pct': float(atr / highp if highp > 0 else 0.0),
        }
    return None


def detect_gradual(df):
    if len(df) < GRADUAL_WINDOW + 1:
        return None
    w = df.iloc[-(GRADUAL_WINDOW+1):]
    start = float(w['close'].iloc[0])
    end = float(w['close'].iloc[-1])
    pct = (end - start) / start if start > 0 else 0.0
    if pct >= GRADUAL_PCT_FLOOR:
        peak = w.index[-1]
        atr = float(w['atr14'].iloc[-1]) if 'atr14' in w.columns else 0.0
        return {
            'category': 'GRADUAL',
            'pump_pct': float(pct),
            'pump_peak_time': peak,
            'pump_peak_price': float(end),
            'atr_pct': float(atr / end if end > 0 else 0.0),
        }
    return None


def detect_sudden_dump(df, bars=SUDDEN_BARS, pct_floor=SUDDEN_PCT_FLOOR, atr_mult=SUDDEN_ATR_MULT):
    if len(df) < bars + 1:
        return None
    w = df.iloc[-(bars+1):]
    ref_time = w.index[0]
    ref = float(w['close'].iloc[0])
    low_min_idx = w['low'].idxmin()
    lowp = float(w.loc[low_min_idx, 'low'])
    pct_drop = (ref - lowp) / ref if ref > 0 else 0.0
    atr = float(w['atr14'].iloc[-1]) if 'atr14' in w.columns else 0.0
    wide = False
    if atr > 0:
        wide = (float(w['high'].iloc[-1]) - float(w['low'].iloc[-1])) >= atr_mult * atr
    if pct_drop >= pct_floor or wide:
        return {
            'category': 'SUDDEN_DUMP',
            'dump_pct': float(pct_drop),
            'dump_time': low_min_idx,
            'dump_price': float(lowp),
            'ref_time': ref_time,
            'ref_price': float(ref),
            'atr_pct': float((atr / ref) if ref > 0 else 0.0),
        }
    return None


def structure_metrics_since(df, start_time):
    if start_time not in df.index:
        return 0, 0.0
    i0 = df.index.get_loc(start_time)
    seg = df.iloc[i0:]
    if len(seg) < 2:
        return 0, 0.0
    lhll = 0
    for i in range(1, len(seg)):
        lh = float(seg['high'].iloc[i]) < float(seg['high'].iloc[i-1])
        ll = float(seg['low'].iloc[i]) < float(seg['low'].iloc[i-1])
        if lh and ll:
            lhll += 1
    below = (seg['close'] < seg['ema20']).sum() if 'ema20' in seg.columns else 0
    frac = float(below) / float(len(seg)) if len(seg) > 0 else 0.0
    return lhll, frac


def detect_gradual_dump(df, window=GRADUAL_DUMP_WINDOW, pct_floor=GRADUAL_DUMP_PCT_FLOOR):
    if len(df) < window + 1:
        return None
    w = df.iloc[-(window+1):]
    peak_time = w['close'].idxmax()
    peak_price = float(df.loc[peak_time, 'close'])
    end_time = w.index[-1]
    end_price = float(w['close'].iloc[-1])
    if end_time <= peak_time:
        return None
    drop = (peak_price - end_price) / peak_price if peak_price > 0 else 0.0
    if drop < pct_floor:
        return None
    lhll_cnt, below_frac = structure_metrics_since(df, peak_time)
    if (below_frac < GRADUAL_DUMP_EMA_FRAC) or (lhll_cnt < GRADUAL_DUMP_LHLL_MIN):
        return None
    atr = float(df['atr14'].loc[end_time]) if 'atr14' in df.columns else 0.0
    return {
        'category': 'GRADUAL_DUMP',
        'dump_pct': float(drop),
        'dump_time': end_time,
        'dump_price': float(end_price),
        'ref_time': peak_time,
        'ref_price': float(peak_price),
        'atr_pct': float((atr / peak_price) if peak_price > 0 else 0.0),
        'lhll_count': int(lhll_cnt),
        'below_ema_frac': float(below_frac),
    }


def detect_drop(df, peak_time, peak_price):
    if peak_time not in df.index:
        return 0.0, None
    pidx = df.index.get_loc(peak_time)
    end = min(len(df) - 1, pidx + DUMP_WINDOW)
    max_drop = 0.0
    t_at = None
    for j in range(pidx + 1, end + 1):
        lowj = float(df['low'].iloc[j])
        drop = (peak_price - lowj) / peak_price if peak_price > 0 else 0.0
        if drop > max_drop:
            max_drop = drop
            t_at = df.index[j]
    return float(max_drop), t_at


def structural_confirm(df, idx_time):
    if idx_time not in df.index:
        return False
    i = df.index.get_loc(idx_time)
    if i < 1:
        return False
    lh_ll = (float(df['high'].iloc[i]) < float(df['high'].iloc[i-1])) and (float(df['low'].iloc[i]) < float(df['low'].iloc[i-1]))
    ema_break = ('ema20' in df.columns) and (float(df['close'].iloc[i]) < float(df['ema20'].iloc[i]))
    return lh_ll or ema_break

# ----------------- Multi-timeframe & Dynamic helpers -----------------

def higher_tf_alignment(df):
    """Return True if higher timeframe trend aligns for shorting."""
    try:
        h = resample_higher_tf(df, rule=HIGHER_TF)
        if h.empty:
            return False
        h = add_indicators(h)
        if 'ema20' in h.columns and len(h) >= HIGHER_EMA_SLOPE_LOOKBACK + 1:
            last = h['ema20'].iloc[-1]
            prev = h['ema20'].iloc[-(HIGHER_EMA_SLOPE_LOOKBACK+1)]
            if last < prev:
                return True
        if 'rsi' in h.columns and h['rsi'].iloc[-1] < HIGHER_RSI_THRESHOLD:
            return True
    except Exception:
        return False
    return False


def rsi_macd_checks(df, at_time):
    """Return True if momentum supports short: RSI < RSI_SHORT_THRESHOLD or MACD histogram falling"""
    try:
        if at_time not in df.index:
            return False
        r = df.loc[at_time, 'rsi'] if 'rsi' in df.columns else None
        macdh = df.loc[at_time, 'macd_hist'] if 'macd_hist' in df.columns else None
        macd_falling = False
        if 'macd_hist' in df.columns:
            i = df.index.get_loc(at_time)
            if i >= 1:
                macd_falling = float(df['macd_hist'].iloc[i]) < float(df['macd_hist'].iloc[i-1])
        if r is not None and r < RSI_SHORT_THRESHOLD:
            return True
        if macd_falling:
            return True
    except Exception:
        return False
    return False


def dynamic_drop_thresholds(peak_price, atr):
    """Scale base thresholds by atr_pct. For low-ATR assets, lower the percent requirement to catch early moves."""
    if peak_price <= 0:
        return BASE_SAFE_DROP_MIN, BASE_AGGRESSIVE_DROP_MIN, BASE_AGGRESSIVE_DROP_ALT
    atr_pct = (atr / peak_price) if peak_price > 0 else 0.0
    scale = max(0.5, min(3.0, atr_pct / 0.01))
    safe = BASE_SAFE_DROP_MIN * scale
    aggr = BASE_AGGRESSIVE_DROP_MIN * scale
    aggr_alt = BASE_AGGRESSIVE_DROP_ALT * scale
    return safe, aggr, aggr_alt


def volume_spike_ok(df, at_time):
    try:
        if at_time not in df.index or 'vol_ma' not in df.columns:
            return False
        return float(df.loc[at_time, 'volume']) >= VOLUME_SPIKE_MULT * float(df.loc[at_time, 'vol_ma'])
    except Exception:
        return False


def post_signal_validation(df, peak_price, drop_price, drop_time):
    """Check next 1-2 candles do not recover beyond POST_VALIDATION_RECOVERY_FRAC of the drop."""
    try:
        if drop_time not in df.index:
            return False
        pidx = df.index.get_loc(drop_time)
        drop_range = peak_price - drop_price
        if drop_range <= 0:
            return False
        end_idx = min(len(df)-1, pidx + POST_VALIDATION_CANDLES)
        for j in range(pidx+1, end_idx+1):
            closej = float(df['close'].iloc[j])
            recovery = (closej - drop_price) / drop_range
            if recovery >= POST_VALIDATION_RECOVERY_FRAC:
                return False
        return True
    except Exception:
        return False

# ----------------- Decision logic -----------------

def decide_short_signal(cat, df, peak_time, peak_price, drop_frac, drop_time, ev_dump=None):
    short_signal = None
    entry_time = None
    reason = None

    atr = 0.0
    if peak_time is not None and peak_time in df.index and 'atr14' in df.columns:
        atr = float(df['atr14'].loc[peak_time])

    safe_req, aggr_req, aggr_alt_req = dynamic_drop_thresholds(peak_price, atr)

    vol_ok = volume_spike_ok(df, drop_time)

    lhll_cnt, below_frac = (0, 0.0)
    if peak_time is not None and peak_time in df.index:
        lhll_cnt, below_frac = structure_metrics_since(df, peak_time)

    high_tf_ok = higher_tf_alignment(df)

    mom_ok = rsi_macd_checks(df, drop_time)

    validated = post_signal_validation(df, peak_price, float(df.loc[drop_time,'low']) if drop_time in df.index else peak_price, drop_time)

    # SAFE SHORT
    if (cat in ('CONFIRMED_GRADUAL_DUMP', 'CONFIRMED_DUMP') or (cat in ('POTENTIAL_DUMP','POTENTIAL_GRADUAL_DUMP') and drop_frac >= safe_req)):
        if lhll_cnt >= 1 and ('ema20' in df.columns and drop_time in df.index and float(df.loc[drop_time,'close']) < float(df.loc[drop_time,'ema20'])):
            if high_tf_ok or mom_ok or vol_ok:
                if validated:
                    short_signal = 'SHORT_SAFE'
                    entry_time = drop_time
                    reason = f"Safe: drop {round(drop_frac*100,2)}%, LHLL={lhll_cnt}, close<EMA, high_tf={high_tf_ok}, mom={mom_ok}, vol={vol_ok}"
                    return short_signal, entry_time, reason

    # AGGRESSIVE SHORT
    if cat == 'SUDDEN_DUMP' or (ev_dump is not None and ev_dump.get('category') == 'SUDDEN_DUMP'):
        if drop_frac >= aggr_req or drop_frac >= aggr_alt_req:
            if mom_ok or vol_ok or high_tf_ok:
                if validated:
                    short_signal = 'SHORT_AGGRESSIVE'
                    entry_time = drop_time
                    reason = f"Aggressive sudden: drop {round(drop_frac*100,2)}%, mom={mom_ok}, vol={vol_ok}, high_tf={high_tf_ok}"
                    return short_signal, entry_time, reason
            else:
                if validated:
                    short_signal = 'SHORT_AGGRESSIVE'
                    entry_time = drop_time
                    reason = f"Aggressive wick: drop {round(drop_frac*100,2)}% (no mom/vol alignment)"
                    return short_signal, entry_time, reason

    if drop_frac >= aggr_alt_req:
        atr_pct = (atr / peak_price) if peak_price > 0 else 0.0
        if atr_pct >= 0.02 or vol_ok:
            if validated:
                short_signal = 'SHORT_AGGRESSIVE'
                entry_time = drop_time
                reason = f"Aggressive alt: drop {round(drop_frac*100,2)}%, atr_pct={round(atr_pct*100,2)}%, vol={vol_ok}"
                return short_signal, entry_time, reason

    return short_signal, entry_time, reason

# ----------------- Main cycle -----------------

def run_cycle(save_dir=None):
    now0 = utcnow()
    print(f"\n[{now0.isoformat()}] Running coin-first scan (enhanced short signals)...")
    logger.info("CYCLE_START")

    ex_objs = {}
    ex_markets = {}
    init_fail = set()

    for eid in EXCHANGE_LIST:
        try:
            ex = load_exchange(eid)
            ex_objs[eid] = ex
            logger.info(f"INIT_OK {eid}")
        except Exception as e:
            logger.error(f"INIT_FAIL {eid} {e}")
            init_fail.add(eid)

    for eid, ex in list(ex_objs.items()):
        try:
            ex_markets[eid] = ex.load_markets()
            logger.info(f"MARKETS_OK {eid} {len(ex_markets[eid])}")
        except Exception as e:
            logger.error(f"MARKETS_FAIL {eid} {e}")
            ex_markets.pop(eid, None)

    skipped = {}
    def bucket(eid, reason, coin):
        skipped.setdefault(eid, {}).setdefault(reason, []).append(coin)

    rows_impulse = []
    rows_gradual = []
    rows_potential_dump = []
    rows_confirmed_dump = []
    rows_gradual_dump_potential = []
    rows_gradual_dump_confirmed = []
    rows_short_signals = []

    recent_cut = utcnow() - timedelta(hours=RECENT_WINDOW_HOURS)
    progress = tqdm(total=len(COIN_LIST), desc="Scanning coins", ncols=80) if tqdm else None

    for coin in COIN_LIST:
        processed = False
        for eid in EXCHANGE_LIST:
            if processed:
                break
            if eid in init_fail or eid not in ex_markets:
                bucket(eid, "MARKETS_FAIL", coin)
                continue

            ex = ex_objs[eid]
            markets = ex_markets[eid]
            tf_req, tf_target = pick_timeframe(ex)

            try:
                resolved, how = resolve_symbol(eid, coin, markets)
                if resolved is None:
                    bucket(eid, "SYMBOL_NOT_FOUND", coin)
                    logger.info(f"SYMBOL_NOT_FOUND {eid} {coin}")
                    continue

                try:
                    raw = safe_fetch_ohlcv(ex, resolved, tf_req, LOOKBACK_LIMIT)
                except ccxt.BadSymbol:
                    bucket(eid, "BAD_SYMBOL", coin)
                    logger.info(f"BAD_SYMBOL {eid} {resolved}")
                    continue
                except ccxt.BadRequest as e:
                    bucket(eid, "TIMEFRAME_UNSUPPORTED", coin)
                    logger.info(f"TIMEFRAME_UNSUPPORTED {eid} {resolved} {tf_req} {e}")
                    continue
                except Exception as e:
                    bucket(eid, "FETCH_FAIL", coin)
                    logger.error(f"FETCH_FAIL {eid} {resolved} {e}")
                    continue

                df = to_df(raw)
                if tf_target:
                    df = resample_to_target(df, tf_target)
                if df.empty:
                    bucket(eid, "OHLCV_EMPTY", coin)
                    logger.info(f"OHLCV_EMPTY {eid} {resolved}")
                    continue
                df = add_indicators(df)

                processed = True

                # Standalone dump check (FIRST)
                sd = detect_sudden_dump(df)
                gd = detect_gradual_dump(df)
                ev_dump = None
                if sd and gd:
                    ev_dump = sd if sd['dump_pct'] >= gd['dump_pct'] else gd
                elif sd:
                    ev_dump = sd
                elif gd:
                    ev_dump = gd

                if ev_dump is not None:
                    dump_time = ev_dump['dump_time']
                    if pd.notna(dump_time) and dump_time >= recent_cut:
                        if 'vol_ma' in df.columns and dump_time in df.index:
                            vol_gate = float(df.loc[dump_time, 'volume']) >= float(df.loc[dump_time, 'vol_ma'])
                        else:
                            vol_gate = True
                        structure_gate = structural_confirm(df, dump_time) if ev_dump['category'] == 'SUDDEN_DUMP' else True
                        dump_frac = float(ev_dump['dump_pct'])
                        atr_pct = float(ev_dump.get('atr_pct', 0.0))

                        # categorize potential/confirmed
                        confirmed_gate = dump_frac >= max(POTENTIAL_DROP_FLOOR, atr_pct)
                        if ev_dump['category'] == 'GRADUAL_DUMP':
                            if confirmed_gate and (not REQUIRE_VOL_FOR_CONFIRMED or vol_gate) and (not REQUIRE_STRUCTURE_FOR_CONFIRMED or True):
                                cat = 'CONFIRMED_GRADUAL_DUMP'
                            elif dump_frac >= POTENTIAL_DROP_FLOOR:
                                cat = 'POTENTIAL_GRADUAL_DUMP'
                            else:
                                cat = None

                            if cat is not None:
                                rec_h = (utcnow() - dump_time).total_seconds() / 3600.0
                                action = "Avoid chasing; wait stabilization" if 'CONFIRMED' in cat else "Watch for reversal; avoid breakouts"
                                row = {
                                    'symbol': coin,
                                    'exchange': eid,
                                    'category': cat,
                                    'dump_pct': round(dump_frac*100,3),
                                    'dump_time': str(dump_time),
                                    'action': action
                                }
                                if 'CONFIRMED' in cat:
                                    rows_gradual_dump_confirmed.append(row)
                                else:
                                    rows_gradual_dump_potential.append(row)

                        else:
                            # Sudden dump handling
                            if dump_frac >= POTENTIAL_DROP_FLOOR:
                                cat = 'SUDDEN_POTENTIAL_DUMP'
                                rec_h = (utcnow() - dump_time).total_seconds() / 3600.0
                                row = {'symbol': coin, 'exchange': eid, 'category': cat, 'dump_pct': round(dump_frac*100,3), 'dump_time': str(dump_time)}
                                rows_potential_dump.append(row)

                        # Now attempt to produce short signals using enhanced logic
                        peak_time = ev_dump.get('ref_time') if ev_dump.get('ref_time') is not None else None
                        peak_price = float(ev_dump.get('ref_price', 0.0)) if ev_dump.get('ref_price') is not None else (float(df['high'].max()) if not df.empty else 0.0)

                        short_signal, entry_time, reason = decide_short_signal(ev_dump.get('category', None), df, peak_time, peak_price, dump_frac, dump_time, ev_dump=ev_dump)
                        if short_signal is not None:
                            rows_short_signals.append({
                                'symbol': coin,
                                'exchange': eid,
                                'signal': short_signal,
                                'entry_time': str(entry_time),
                                'reason': reason
                            })

                # Also keep original pump detectors
                imp = detect_impulse(df)
                gr = detect_gradual(df)
                if imp is not None:
                    rows_impulse.append({'symbol': coin, 'exchange': eid, 'pump_pct': round(imp['pump_pct']*100,3)})
                if gr is not None:
                    rows_gradual.append({'symbol': coin, 'exchange': eid, 'pump_pct': round(gr['pump_pct']*100,3)})

            except Exception as e:
                logger.error(f"CYCLE_COIN_FAIL {eid} {coin} {e}")
                bucket(eid, "CYCLE_ERROR", coin)
                continue

        if progress:
            progress.update(1)

    if progress:
        progress.close()

    # Print summary tables
    print("\nScan summary:")
    print(f"Short signals: {len(rows_short_signals)}")
    print(f"Confirmed gradual dumps: {len(rows_gradual_dump_confirmed)}")
    print(f"Potential gradual dumps: {len(rows_gradual_dump_potential)}")
    print(f"Potential sudden dumps: {len(rows_potential_dump)}")
    print(f"Impulse pumps: {len(rows_impulse)}")
    print(f"Gradual pumps: {len(rows_gradual)}")

    if rows_short_signals:
        print("\nShort signals detail:\n", tabulate(rows_short_signals, headers="keys"))
    if rows_gradual_dump_confirmed:
        print("\nConfirmed gradual dumps:\n", tabulate(rows_gradual_dump_confirmed, headers="keys"))
    if rows_gradual_dump_potential:
        print("\nPotential gradual dumps:\n", tabulate(rows_gradual_dump_potential, headers="keys"))
    if rows_potential_dump:
        print("\nPotential sudden dumps:\n", tabulate(rows_potential_dump, headers="keys"))
    if rows_impulse:
        print("\nImpulse pumps:\n", tabulate(rows_impulse, headers="keys"))
    if rows_gradual:
        print("\nGradual pumps:\n", tabulate(rows_gradual, headers="keys"))

    # Skipped coins summary
    if skipped:
        print("\nSkipped coins:")
        for eid, reasons in skipped.items():
            for reason, coins in reasons.items():
                sample = ', '.join(coins[:5])
                more = f"... ({len(coins)} total)" if len(coins) > 5 else ""
                print(f"{eid} - {reason}: {sample}{more}")

    logger.info("CYCLE_END")

    # Save CSV logs if requested
    saved = {}
    if save_dir:
        ts = utcnow().strftime('%Y%m%dT%H%M%SZ')
        try:
            os.makedirs(save_dir, exist_ok=True)
            def save(name, rows):
                if not rows:
                    return None
                df = pd.DataFrame(rows)
                fname = os.path.join(save_dir, f"{ts}_{name}.csv")
                df.to_csv(fname, index=False)
                return fname
            saved['short_signals'] = save('short_signals', rows_short_signals)
            saved['gradual_confirmed'] = save('gradual_confirmed', rows_gradual_dump_confirmed)
            saved['gradual_potential'] = save('gradual_potential', rows_gradual_dump_potential)
            saved['sudden_potential'] = save('sudden_potential', rows_potential_dump)
            saved['impulse'] = save('impulse', rows_impulse)
            saved['gradual'] = save('gradual', rows_gradual)
            summary = {
                'ts': [ts],
                'short_signals': [len(rows_short_signals)],
                'gradual_confirmed': [len(rows_gradual_dump_confirmed)],
                'gradual_potential': [len(rows_gradual_dump_potential)],
                'sudden_potential': [len(rows_potential_dump)],
                'impulse': [len(rows_impulse)],
                'gradual': [len(rows_gradual)]
            }
            pd.DataFrame(summary).to_csv(os.path.join(save_dir, f"{ts}_summary.csv"), index=False)
            logger.info(f"SAVED: {saved}")
        except Exception as e:
            logger.error(f"SAVE_FAIL {e}")

    # Return structured results for callers (telegram summary / CI)
    return {
        "short_signals": rows_short_signals,
        "gradual_confirmed": rows_gradual_dump_confirmed,
        "gradual_potential": rows_gradual_dump_potential,
        "sudden_potential": rows_potential_dump,
        "impulse": rows_impulse,
        "gradual": rows_gradual,
        "skipped": skipped,
        "saved": saved,
    }

# ----------------- Telegram + summary -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def tg_send(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured (set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID). Skipping send.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        r = requests.post(url, data=data, timeout=10)
        if not r.ok:
            print("Telegram send failed:", r.status_code, r.text)
    except Exception as e:
        print("Telegram send exception:", e)

def telegram_summary(results):
    def summarize(title, rows, key="symbol"):
        if not rows:
            return f"{title}: None"
        lines = [f"{title}:"]
        for r in rows:
            lines.append(f"- {r.get(key,'?')} | {r.get('action', r.get('reason',''))}")
        return "\n".join(lines)

    parts = [
        summarize("Short signals", results.get("short_signals", [])),
        summarize("IMPULSE pumps", results.get("impulse", [])),
        summarize("GRADUAL pumps", results.get("gradual", [])),
        summarize("Potential sudden dumps", results.get("sudden_potential", [])),
        summarize("Confirmed gradual dumps", results.get("gradual_confirmed", [])),
        summarize("Potential gradual dumps", results.get("gradual_potential", [])),
    ]
    message = "📊 Crypto Pump-Dump Screener\n\n" + "\n\n".join(parts)
    tg_send(message)

# ==================================================
# ================== MAIN ENTRY =====================
# ==================================================

def main():
    # Run one scan; CI / GitHub Actions should run this script on a schedule (e.g. every 15 minutes)
    results = run_cycle(save_dir="scan_logs")
    # Send telegram summary (if configured)
    telegram_summary(results)
    # Optionally, write a short status file for CI
    try:
        with open("last_scan_status.txt", "w", encoding="utf-8") as f:
            f.write(f"ts={datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"short_signals={len(results.get('short_signals', []))}\n")
            f.write(f"saved={results.get('saved', {})}\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()
