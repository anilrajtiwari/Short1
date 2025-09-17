# short_only_strategy.py (with fast Supertrend + corrected symbols)
# Short-only strategy with multi-timeframe EMA filter (15m + 1h) + Supertrend
# Relaxed EMA slope rule, broader pullback, ATR-based exits, closed-candle enforcement

import os
import ccxt
import pandas as pd
import requests
from datetime import datetime, timezone
from tqdm import tqdm
import time

# === Config ===
symbols = [
    "KOMA/USDT","DOGS/USDT","NEIROETH/USDT","1000RATS/USDT","ORDI/USDT","MEME/USDT",
    "PIPPIN/USDT","BAN/USDT","1000SHIB/USDT","OM/USDT","CHILLGUY/USDT","PONKE/USDT",
    "BOME/USDT","MYRO/USDT","PEOPLE/USDT","PENGU/USDT","SPX/USDT","1000BONK/USDT",
    "PNUT/USDT","FARTCOIN/USDT","HIPPO/USDT","AIXBT/USDT","BRETT/USDT","VINE/USDT",
    "MOODENG/USDT","MUBARAK/USDT","MEW/USDT","POPCAT/USDT","1000FLOKI/USDT",
    "DOGE/USDT","1000CAT/USDT","ACT/USDT","SLERF/USDT","DEGEN/USDT","WIF/USDT",
    "1000PEPE/USDT"
]
primary_exchanges = ['binanceusdm','bybit','okx']
secondary_exchanges = ['gateio','kucoin','huobi','coinex','kraken','phemex','bitget']

rsi_len = 14
max_age_minutes = 30  # tightened freshness: ~≤ 2 bars on 15m

# ATR-based exits
atr_len = 14
atr_tp_mult = 1.5  # TP = 1.5 * ATR(14)
atr_sl_mult = 1.0  # SL = 1.0 * ATR(14)

# Supertrend params (fast responding)
st_atr_period = 7
st_mult = 2.0

# Telegram settings
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" if TELEGRAM_TOKEN else None
TG_MAX = 4000

# Track open positions
# { "symbol": {"exchange": ex_id, "entry": price, "time": timestamp, "tp": tp_price, "sl": sl_price} }
open_positions = {}

# === Helpers ===
def get_ema(df, length):
    return df['close'].ewm(span=length, adjust=False).mean()

def get_rsi(df, length=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def get_atr(df, length=14):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def get_supertrend(df, atr_period=7, multiplier=2.0):
    hl2 = (df['high'] + df['low']) / 2
    atr = get_atr(df, atr_period)

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(df)):
        if df['close'].iloc[i-1] > final_upperband.iloc[i-1]:
            final_upperband.iloc[i] = upperband.iloc[i]
        else:
            final_upperband.iloc[i] = min(upperband.iloc[i], final_upperband.iloc[i-1])

        if df['close'].iloc[i-1] < final_lowerband.iloc[i-1]:
            final_lowerband.iloc[i] = lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = max(lowerband.iloc[i], final_lowerband.iloc[i-1])

    supertrend = pd.Series(index=df.index, dtype='float64')
    in_uptrend = pd.Series(index=df.index, dtype='bool')

    for i in range(len(df)):
        if df['close'].iloc[i] > final_upperband.iloc[i]:
            in_uptrend.iloc[i] = True
        elif df['close'].iloc[i] < final_lowerband.iloc[i]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i-1] if i > 0 else True

        supertrend.iloc[i] = final_lowerband.iloc[i] if in_uptrend.iloc[i] else final_upperband.iloc[i]

    df['supertrend'] = supertrend
    df['in_uptrend'] = in_uptrend
    return df

def fetch_candles(exchange, symbol, tf="15m", limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df
    except Exception:
        return pd.DataFrame()

def minutes_ago(ts):
    now = datetime.now(timezone.utc)
    return int((now - ts).total_seconds() // 60)

def send_telegram_text_chunks(text):
    if not TELEGRAM_API or not TELEGRAM_CHAT:
        print("Telegram not configured.")
        return
    lines = text.splitlines()
    chunks, cur = [], ""
    for line in lines:
        if len(cur) + len(line) + 1 <= TG_MAX:
            cur += line + "\n"
        else:
            if cur: chunks.append(cur)
            cur = line + "\n"
    if cur: chunks.append(cur)
    for chunk in chunks:
        try:
            requests.post(TELEGRAM_API, data={"chat_id": TELEGRAM_CHAT, "text": chunk}, timeout=15)
            time.sleep(0.4)
        except Exception as e:
            print("Telegram send failed:", e)

# === Strategy ===
def evaluate(symbol):
    exchanges_to_check = primary_exchanges + secondary_exchanges
    for ex_id in exchanges_to_check:
        try:
            ex_class = getattr(ccxt, ex_id)
            exchange = ex_class({'options': {'defaultType':'future'}})
            if not exchange.has.get("fetchOHLCV", False):
                continue

            # --- 15m timeframe ---
            df15 = fetch_candles(exchange, symbol, tf="15m", limit=200)
            if df15.empty or len(df15) < 60:
                continue

            df15["ema9"] = get_ema(df15, 9)
            df15["ema21"] = get_ema(df15, 21)
            df15["ema50"] = get_ema(df15, 50)
            df15["rsi"] = get_rsi(df15, rsi_len)
            df15["atr14"] = get_atr(df15, atr_len)
            df15 = get_supertrend(df15, atr_period=st_atr_period, multiplier=st_mult)

            last15 = df15.iloc[-2]  # closed
            prev15 = df15.iloc[-3]

            if minutes_ago(last15["time"]) > max_age_minutes:
                continue

            # --- 1h timeframe filter ---
            df1h = fetch_candles(exchange, symbol, tf="1h", limit=200)
            if df1h.empty or len(df1h) < 60:
                continue
            df1h["ema9"] = get_ema(df1h, 9)
            df1h["ema21"] = get_ema(df1h, 21)
            last1h = df1h.iloc[-2]

            higher_tf_down = last1h["ema9"] < last1h["ema21"]

            # === ENTRY ===
            if symbol not in open_positions:
                trend_down = (
                    last15["ema9"] < last15["ema21"]
                    and last15["close"] < last15["ema9"]
                    and last15["close"] < last15["ema21"]
                )

                rsi_pullback = last15["rsi"] > 50

                def near_dynamic_ma(df, idx, pct_tol=0.003):
                    row = df.iloc[idx]
                    close = row["close"]
                    d21 = abs(close - row["ema21"]) / close
                    d50 = abs(close - row["ema50"]) / close
                    return (d21 <= pct_tol) or (d50 <= pct_tol)

                recent_near = (
                    near_dynamic_ma(df15, -2) or near_dynamic_ma(df15, -3) or near_dynamic_ma(df15, -4)
                )
                pullback_ok = rsi_pullback or recent_near

                supertrend_short = (not last15["in_uptrend"]) and (last15["close"] < last15["supertrend"])

                if higher_tf_down and trend_down and pullback_ok and supertrend_short:
                    entry_price = last15["close"]
                    entry_atr = df15["atr14"].iloc[-2]
                    if pd.isna(entry_atr) or entry_atr <= 0:
                        continue

                    tp_price = entry_price - atr_tp_mult * entry_atr
                    sl_price = entry_price + atr_sl_mult * entry_atr

                    open_positions[symbol] = {
                        "exchange": ex_id.upper(),
                        "entry": entry_price,
                        "time": last15["time"],
                        "tp": tp_price,
                        "sl": sl_price
                    }
                    return {
                        "type": "entry",
                        "symbol": symbol,
                        "exchange": ex_id.upper(),
                        "price": entry_price,
                        "tp": tp_price,
                        "sl": sl_price
                    }

            # === EXIT ===
            else:
                pos = open_positions[symbol]
                entry_price = pos["entry"]
                tp_price = pos["tp"]
                sl_price = pos["sl"]
                price = last15["close"]

                tp_hit = price <= tp_price
                sl_hit = price >= sl_price
                trend_flip = last15["ema9"] >= last15["ema21"] or last15["in_uptrend"]

                if tp_hit or sl_hit or trend_flip:
                    reason = "TP" if tp_hit else ("SL" if sl_hit else "TrendFlip")
                    del open_positions[symbol]
                    return {
                        "type": "exit",
                        "symbol": symbol,
                        "exchange": pos["exchange"],
                        "price": price,
                        "reason": reason
                    }

        except Exception:
            continue
    return None

# === Main ===
def run_once():
    signals, missing = [], []
    for sym in tqdm(symbols, desc="Checking coins", unit="coin"):
        result = evaluate(sym)
        if result:
            signals.append(result)
        else:
            missing.append(sym)

    lines = [f"=== SHORT STRATEGY (15m + 1h EMA filter + Supertrend; Exits: TP={atr_tp_mult}xATR, SL={atr_sl_mult}xATR) ==="]
    if signals:
        for s in signals:
            if s["type"] == "entry":
                lines.append(
                    f"📉 ENTRY SHORT {s['symbol']} ({s['exchange']}) @ {s['price']:.6f} | TP={s['tp']:.6f} SL={s['sl']:.6f}"
                )
            elif s["type"] == "exit":
                lines.append(
                    f"✅ EXIT {s['symbol']} ({s['exchange']}) @ {s['price']:.6f} | Reason={s['reason']}"
                )
    else:
        lines.append("No signals this run.")

    lines.append("\n=== Open Positions ===")
    if open_positions:
        for sym, pos in open_positions.items():
            lines.append(f"{sym} @ {pos['entry']:.6f} ({pos['exchange']}) | TP={pos['tp']:.6f} SL={pos['sl']:.6f}")
    else:
        lines.append("None")

    final_text = "\n".join(lines)
    send_telegram_text_chunks(final_text)
    print(final_text)

if __name__ == "__main__":
    while True:
        print("\n===== New Scan at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "=====")
        run_once()
        print("Sleeping for 20 minutes...\n")
        time.sleep(1200)  # wait 20 min
