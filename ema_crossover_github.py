# planmod_coin_first.py
#
# CI/GitHub-friendly scan:
# - Single run (no sleep loop, no banners, no progress)
# - Tables show only: symbol, suggested_action
# - Skipped-by-reason section removed from stdout (still logged)
# - Sends a Telegram summary every run (confirmed or not)

import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
import os
import json
import urllib.parse
import urllib.request

import ccxt
import pandas as pd
from tabulate import tabulate

# ================== TELEGRAM ==================
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def tg_send(text, retries=2, timeout=15):
    if not BOT_TOKEN or not CHAT_ID or not text:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
                return bool(resp.get("ok"))
        except Exception:
            if i < retries:
                time.sleep(1.5 * (i + 1))
            else:
                return False

# ================== CONFIG ==================

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

# Exchange-specific symbol hints
SYMBOL_MAP = {
    '1000PEPE/USDT': {'mexc':'PEPE1000/USDT','gateio':'PEPE1000/USDT','bitget':'PEPE1000/USDT'},
    '1000BONK/USDT': {'mexc':'BONK1000/USDT'},
    '1000FLOKI/USDT': {'mexc':'FLOKI1000/USDT'},
    '1000SHIB/USDT': {'mexc':'SHIB1000/USDT'},
    '1000CAT/USDT': {'mexc':'CAT1000/USDT'},
}

TIMEFRAME_TARGET = '15m'
LOOKBACK_LIMIT = 400
RECENT_WINDOW_HOURS = 48
RATE_LIMIT_SLEEP_DEFAULT = 0.35

# Pump rules
IMPULSE_WINDOW = 8            # bars
IMPULSE_PCT_FLOOR = 0.18      # 18%
GRADUAL_WINDOW = 24           # bars
GRADUAL_PCT_FLOOR = 0.10      # 10%

# Drop rules (used for pump→drop and standalone dumps)
DUMP_WINDOW = 36              # bars after peak
DUMP_PCT_FLOOR = 0.12         # 12%
POTENTIAL_DROP_FLOOR = 0.03   # 3%

# Standalone dump detectors (do not require a pump)
# Sudden
SUDDEN_BARS = 3
SUDDEN_PCT_FLOOR = 0.06
SUDDEN_ATR_MULT = 2.0

# Tuned Gradual (EARLY) — drop-from-recent-peak
GRADUAL_DUMP_WINDOW = 16          # shorter window for earlier catch
GRADUAL_DUMP_PCT_FLOOR = 0.07     # lower threshold for sensitivity
GRADUAL_DUMP_EMA_FRAC = 0.60      # ≥60% closes below EMA20 since peak
GRADUAL_DUMP_LHLL_MIN = 2         # ≥2 LH+LL steps since peak for structure

# Filters for confirmation
REQUIRE_VOL_FOR_CONFIRMED = True       # volume gate
REQUIRE_STRUCTURE_FOR_CONFIRMED = True # structure gate

# ================== LOGGING ==================
logger = logging.getLogger("scan")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("scan.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(fmt)
logger.addHandler(handler)

def utcnow():
    return datetime.now(timezone.utc)

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
    for tf in ['15m','5m','1m','30m','1h']:
        if tf in tfs:
            return tf, TIMEFRAME_TARGET
    return TIMEFRAME_TARGET, None

def safe_fetch_ohlcv(ex, symbol, tf_req, limit):
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
    return ex.fetch_ohlcv(symbol, tf_req, limit=limit)

def to_df(ohlcv):
    if not ohlcv:
        return pd.DataFrame(columns=['open','high','low','close','volume']).set_index(pd.DatetimeIndex([], tz='UTC'))
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('dt').drop(columns=['ts']).sort_index()

def resample_to_target(df, target_tf):
    if target_tf is None or df.empty:
        return df
    rule = {'1m':'1T','5m':'5T','15m':'15T','30m':'30T','1h':'1H'}.get(target_tf, '15T')
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
    return df

# ----------------- Pump detectors -----------------

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

# ----------------- Standalone dump detectors -----------------

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
    # Returns (lhll_count, below_ema_fraction)
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
    # Early, sensitive: drop-from-recent-peak inside window plus structure metrics
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
        'category': 'GRADUAL_DUMP',           # standalone gradual
        'dump_pct': float(drop),
        'dump_time': end_time,
        'dump_price': float(end_price),
        'ref_time': peak_time,
        'ref_price': float(peak_price),
        'atr_pct': float((atr / peak_price) if peak_price > 0 else 0.0),
        'lhll_count': int(lhll_cnt),
        'below_ema_frac': float(below_frac),
    }

# ----------------- Drop and confirmation helpers -----------------

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

def suggest_action(cat, recency_h, drop_frac):
    if cat == 'CONFIRMED_DUMP' or cat == 'CONFIRMED_GRADUAL_DUMP':
        return "Avoid chasing; wait stabilization"
    if cat == 'POTENTIAL_DUMP' or cat == 'POTENTIAL_GRADUAL_DUMP':
        return "Watch for reversal; avoid breakouts"
    if cat == 'IMPULSE':
        return "Wait pullback; no breakout buys"
    if cat == 'GRADUAL':
        return "Trend monitor; buy pullbacks"
    return "Neutral; observe"

# ----------------- Main cycle -----------------

def run_cycle():
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

    # Keep skipped internally for logs; do not print to stdout
    skipped = {}
    def bucket(eid, reason, coin):
        skipped.setdefault(eid, {}).setdefault(reason, []).append(coin)

    rows_impulse = []
    rows_gradual = []
    rows_potential_dump = []
    rows_confirmed_dump = []
    rows_gradual_dump_potential = []  # standalone gradual only
    rows_gradual_dump_confirmed = []  # standalone gradual only

    recent_cut = utcnow() - timedelta(hours=RECENT_WINDOW_HOURS)

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

                # ----------------- Standalone dump check (FIRST) -----------------
                sd = detect_sudden_dump(df)
                gd = detect_gradual_dump(df)  # tuned early gradual (drop-from-peak)
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
                        # volume gate at dump bar
                        if 'vol_ma' in df.columns and dump_time in df.index:
                            vol_gate = float(df.loc[dump_time, 'volume']) >= float(df.loc[dump_time, 'vol_ma'])
                        else:
                            vol_gate = True
                        # structure gate (single-bar) for sudden; multi-bar already applied for gradual
                        structure_gate = structural_confirm(df, dump_time) if ev_dump['category'] == 'SUDDEN_DUMP' else True

                        dump_frac = float(ev_dump['dump_pct'])
                        atr_pct = float(ev_dump.get('atr_pct', 0.0))
                        confirmed_gate = dump_frac >= max(DUMP_PCT_FLOOR, atr_pct)

                        if ev_dump['category'] == 'GRADUAL_DUMP':
                            # Separate tables for standalone gradual
                            if confirmed_gate and (not REQUIRE_VOL_FOR_CONFIRMED or vol_gate) and (not REQUIRE_STRUCTURE_FOR_CONFIRMED or True):
                                cat = 'CONFIRMED_GRADUAL_DUMP'
                            elif dump_frac >= POTENTIAL_DROP_FLOOR:
                                cat = 'POTENTIAL_GRADUAL_DUMP'
                            else:
                                cat = None

                            if cat is not None:
                                rec_h = (utcnow() - dump_time).total_seconds() / 3600.0
                                action = suggest_action(cat, rec_h, dump_frac)
                                row = {
                                    'symbol': coin,
                                    'exchange': eid,
                                    'category': cat,
                                    'pump_pct': None,
                                    'pump_peak_time': None,
                                    'dump_time': dump_time,
                                    'dump_pct_from_peak': round(dump_frac*100, 2),
                                    'below_ema_frac': round(ev_dump.get('below_ema_frac', 0.0), 2),
                                    'lhll_count': ev_dump.get('lhll_count', 0),
                                    'suggested_action': action,
                                }
                                if cat == 'CONFIRMED_GRADUAL_DUMP':
                                    rows_gradual_dump_confirmed.append(row)
                                else:
                                    rows_gradual_dump_potential.append(row)
                                logger.info(f"EVENT {eid} {coin} {cat} standalone_gradual={row['dump_pct_from_peak']}%")
                                time.sleep(getattr(ex, 'rateLimit', 350)/1000.0)
                                break  # done with this coin
                        else:
                            # Sudden stays in generic dumps tables
                            if confirmed_gate and (not REQUIRE_VOL_FOR_CONFIRMED or vol_gate) and (not REQUIRE_STRUCTURE_FOR_CONFIRMED or structure_gate):
                                cat = 'CONFIRMED_DUMP'
                            elif dump_frac >= POTENTIAL_DROP_FLOOR:
                                cat = 'POTENTIAL_DUMP'
                            else:
                                cat = None

                            if cat is not None:
                                rec_h = (utcnow() - dump_time).total_seconds() / 3600.0
                                action = suggest_action(cat, rec_h, dump_frac)
                                row = {
                                    'symbol': coin,
                                    'exchange': eid,
                                    'category': cat,
                                    'pump_pct': None,
                                    'pump_peak_time': None,
                                    'dump_time': dump_time,
                                    'dump_pct_from_peak': round(dump_frac*100, 2),
                                    'suggested_action': action,
                                }
                                if cat == 'CONFIRMED_DUMP':
                                    rows_confirmed_dump.append(row)
                                else:
                                    rows_potential_dump.append(row)
                                logger.info(f"EVENT {eid} {coin} {cat} standalone_sudden={row['dump_pct_from_peak']}%")
                                time.sleep(getattr(ex, 'rateLimit', 350)/1000.0)
                                break  # done with this coin

                # ----------------- Pump detection (FALLBACK) -----------------
                imp = detect_impulse(df)
                gra = detect_gradual(df)
                ev = imp if imp else gra

                if ev is None:
                    logger.info(f"NO_EVENT {eid} {coin}")
                    time.sleep(getattr(ex, 'rateLimit', 350)/1000.0)
                    break

                peak = ev['pump_peak_time']
                if (not pd.notna(peak)) or (peak < recent_cut):
                    logger.info(f"STALE_EVENT {eid} {coin} peak={peak}")
                    time.sleep(getattr(ex, 'rateLimit', 350)/1000.0)
                    break

                peak_price = float(ev['pump_peak_price'])
                atr_pct = float(ev['atr_pct'])
                drop_frac, drop_time = detect_drop(df, peak, peak_price)

                if drop_time is not None and 'vol_ma' in df.columns and peak in df.index:
                    vol_gate = (float(df.loc[peak, 'volume']) >= float(df.loc[peak, 'vol_ma'])) or \
                               (float(df.loc[drop_time, 'volume']) >= float(df.loc[drop_time, 'vol_ma']))
                else:
                    vol_gate = True

                confirmed_gate = drop_frac >= max(DUMP_PCT_FLOOR, atr_pct)
                structure_gate = (drop_time is not None) and structural_confirm(df, drop_time)

                if confirmed_gate and (not REQUIRE_VOL_FOR_CONFIRMED or vol_gate) and (not REQUIRE_STRUCTURE_FOR_CONFIRMED or structure_gate):
                    cat = 'CONFIRMED_DUMP'
                elif drop_frac >= POTENTIAL_DROP_FLOOR:
                    cat = 'POTENTIAL_DUMP'
                else:
                    cat = ev['category']

                rec_h = (utcnow() - peak).total_seconds()/3600.0
                action = suggest_action(cat, rec_h, drop_frac)

                row = {
                    'symbol': coin,
                    'exchange': eid,
                    'category': cat,
                    'pump_pct': round(ev['pump_pct']*100, 2),
                    'pump_peak_time': peak,
                    'dump_time': drop_time if cat in ('POTENTIAL_DUMP','CONFIRMED_DUMP') else None,
                    'dump_pct_from_peak': round(drop_frac*100, 2) if drop_frac > 0 else None,
                    'suggested_action': action,
                }

                if cat == 'CONFIRMED_DUMP':
                    rows_confirmed_dump.append(row)
                elif cat == 'POTENTIAL_DUMP':
                    rows_potential_dump.append(row)
                elif cat == 'IMPULSE':
                    rows_impulse.append(row)
                elif cat == 'GRADUAL':
                    rows_gradual.append(row)

                logger.info(f"EVENT {eid} {coin} {cat} pump={row.get('pump_pct')}% drop={row.get('dump_pct_from_peak')}")
                time.sleep(getattr(ex, 'rateLimit', 350)/1000.0)
                break

            except Exception as e:
                bucket(eid, "UNKNOWN", coin)
                logger.exception(f"UNKNOWN {eid} {coin} {e}")

    # --------- Print tables (symbol + suggested_action only) ---------

    def print_table(title, rows):
        print(f"\n=== {title} ===")
        if not rows:
            print("None")
            return
        dfp = pd.DataFrame(rows)
        cols = ['symbol', 'suggested_action']
        for c in cols:
            if c not in dfp.columns:
                dfp[c] = None
        print(tabulate(dfp[cols], headers="keys", tablefmt="pretty", showindex=False))

    print_table("IMPULSE pumps", rows_impulse)
    print_table("GRADUAL pumps", rows_gradual)
    print_table("Potential dumps", rows_potential_dump)
    print_table("Confirmed dumps", rows_confirmed_dump)
    print_table("Standalone Gradual dumps (Potential)", rows_gradual_dump_potential)
    print_table("Standalone Gradual dumps (Confirmed)", rows_gradual_dump_confirmed)

    # --------- Telegram summary (always send) ---------

    def summarize(title, rows):
        if not rows:
            return f"{title}: None"
        lines = [f"{title}:"]
        for r in rows:
            lines.append(f"- {r['symbol']} | {r.get('suggested_action','')}")
        return "\n".join(lines)

    summary_parts = [
        summarize("IMPULSE pumps", rows_impulse),
        summarize("GRADUAL pumps", rows_gradual),
        summarize("Potential dumps", rows_potential_dump),
        summarize("Confirmed dumps", rows_confirmed_dump),
        summarize("Standalone Gradual dumps (Potential)", rows_gradual_dump_potential),
        summarize("Standalone Gradual dumps (Confirmed)", rows_gradual_dump_confirmed),
    ]
    message = "Crypto Pump-Dump Screener\n" + "\n\n".join(summary_parts)
    tg_send(message)

    logger.info("CYCLE_END")

def main():
    # Single CI run; scheduling managed by GitHub Actions
    run_cycle()

if __name__ == "__main__":
    main()
