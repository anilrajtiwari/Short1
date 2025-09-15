# rsi_macd_atr_short.py
# Single timeframe RSI + MACD + ATR (15m)
# Short-only plan:
#   - Price < EMA-200 (trend filter)
#   - RSI < 50 and falling OR MACD histogram just flipped negative
#   - ATR-based stop loss (1.5x ATR above entry)

import os
import ccxt
import pandas as pd
import requests
from datetime import datetime, timezone
from tqdm import tqdm
import time

# === Config ===
symbols = [
    "KOMA/USDT:USDT","DOGS/USDT:USDT","NEIRO/USDT:USDT",
    "ORDI/USDT:USDT","MEME/USDT:USDT","PIPPIN/USDT:USDT","BAN/USDT:USDT",
    "OM/USDT:USDT","CHILLGUY/USDT:USDT","PONKE/USDT:USDT",
    "BOME/USDT:USDT","MYRO/USDT:USDT","PEOPLE/USDT:USDT","PENGU/USDT:USDT",
    "SPX/USDT:USDT","PNUT/USDT:USDT","FARTCOIN/USDT:USDT",
    "HIPPO/USDT:USDT","AIXBT/USDT:USDT","BRETT/USDT:USDT","VINE/USDT:USDT",
    "MOODENG/USDT:USDT","MUBARAK/USDT:USDT","MEW/USDT:USDT","POPCAT/USDT:USDT",
    "DOGE/USDT:USDT","ACT/USDT:USDT",
    "SLERF/USDT:USDT","DEGEN/USDT:USDT","WIF/USDT:USDT"
]

primary_exchanges = ['binanceusdm','bybit','okx']
secondary_exchanges = ['gateio','kucoin','huobi','coinex','kraken','phemex','bitget']

# Strategy settings
ema_long = 200
rsi_len = 14
macd_fast, macd_slow, macd_signal = 12, 26, 9
atr_len, atr_mult = 14, 1.5
max_age_minutes = 150

# Telegram settings
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" if TELEGRAM_TOKEN else None
TG_MAX = 4000

# === Helpers ===
def get_ema(df, length):
    return df['close'].ewm(span=length, adjust=False).mean()

def get_rsi(df, length=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def get_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def get_atr(df, length=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

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

# === Evaluation ===
def evaluate(symbol):
    details = []
    exchanges_to_check = primary_exchanges + secondary_exchanges
    for ex_id in exchanges_to_check:
        try:
            ex_class = getattr(ccxt, ex_id)
            exchange = ex_class({'options': {'defaultType':'future'}})
            if not exchange.has.get("fetchOHLCV", False):
                continue

            df = fetch_candles(exchange, symbol, tf="15m", limit=200)
            if df.empty: 
                continue

            df["ema200"] = get_ema(df, ema_long)
            df["rsi"] = get_rsi(df, rsi_len)
            df["macd"], df["macd_signal"], df["macd_hist"] = get_macd(df, macd_fast, macd_slow, macd_signal)
            df["atr"] = get_atr(df, atr_len)

            last = df.iloc[-1]
            prev = df.iloc[-2]

            if minutes_ago(last["time"]) > max_age_minutes:
                continue

            # Trend filter
            trend_ok = last["close"] < last["ema200"]

            # Momentum filters
            rsi_bear = last["rsi"] < 50 and last["rsi"] < prev["rsi"]
            macd_bear = last["macd_hist"] < 0 and prev["macd_hist"] > 0

            momentum_bear = rsi_bear or macd_bear

            bearish_signal = trend_ok and momentum_bear
            tag = "✅ SHORT" if bearish_signal else ""

            signal = None
            if bearish_signal:
                stop_loss = last["close"] + last["atr"] * atr_mult
                signal = (f"SHORT SIGNAL: {symbol} ({ex_id.upper()}) @ {last['close']:.6f} | "
                          f"StopLoss={stop_loss:.6f} | {tag}")

            details.append(
                f"{symbol} ({ex_id.upper()}) @ {last['close']:.6f} | "
                f"EMA200={last['ema200']:.4f}, RSI={last['rsi']:.2f}, MACD_hist={last['macd_hist']:.4f}, ATR={last['atr']:.4f} "
                f"| rsi_bear={'YES' if rsi_bear else 'NO'}, macd_bear={'YES' if macd_bear else 'NO'}, trend_ok={'YES' if trend_ok else 'NO'} {tag}"
            )

            return signal, details, False

        except Exception:
            continue

    return None, [f"{symbol} not available on any exchange"], True

# === Main ===
def main():
    signals, details_all, missing = [], [], []
    total, processed, strong_count = len(symbols), 0, 0

    for sym in tqdm(symbols, desc="Checking coins", unit="coin"):
        sig, det, is_missing = evaluate(sym)
        details_all.extend(det)
        if is_missing:
            missing.append(sym)
        else:
            processed += 1
        if sig:
            signals.append(sig)
            strong_count += 1

    lines = ["=== SHORT SIGNALS (RSI+MACD+ATR, 15m) ==="]
    lines.extend(signals if signals else ["No valid short signals this run."])
    lines.append("\n=== MISSING COINS ===")
    lines.extend(missing if missing else ["None"])
    lines.append(f"\n=== SUMMARY ===\nChecked: {total} | Processed: {processed} | Missing: {len(missing)} | "
                 f"Signals: {len(signals)} (Valid: {strong_count})")

    final_text = "\n".join(lines)
    send_telegram_text_chunks(final_text)
    print(final_text)

if __name__ == "__main__":
    main()
