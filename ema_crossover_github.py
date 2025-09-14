# ema_crossover_github.py
# Full copy-paste script for GitHub Actions:
# - runs the multi-TF short-only checks (2h boss chart relaxed, 15m execution)
# - stops at first working exchange per coin
# - collects missing coins
# - sends results to Telegram (splits into <=4000-char chunks)
# - uses env vars TELEGRAM_TOKEN and TELEGRAM_CHAT_ID (set as GitHub Secrets)

import os
import ccxt
import pandas as pd
import requests
from datetime import datetime, timezone
from tqdm import tqdm
import math
import time

# === Config ===
symbols = [
    "KOMA/USDT:USDT","DOGS/USDT:USDT","NEIRO/USDT:USDT","1000RATS/USDT:USDT",
    "ORDI/USDT:USDT","MEME/USDT:USDT","PIPPIN/USDT:USDT","BAN/USDT:USDT",
    "1000SHIB/USDT:USDT","OM/USDT:USDT","CHILLGUY/USDT:USDT","PONKE/USDT:USDT",
    "BOME/USDT:USDT","MYRO/USDT:USDT","PEOPLE/USDT:USDT","PENGU/USDT:USDT",
    "SPX/USDT:USDT","1000BONK/USDT:USDT","PNUT/USDT:USDT","FARTCOIN/USDT:USDT",
    "HIPPO/USDT:USDT","AIXBT/USDT:USDT","BRETT/USDT:USDT","VINE/USDT:USDT",
    "MOODENG/USDT:USDT","MUBARAK/USDT:USDT","MEW/USDT:USDT","POPCAT/USDT:USDT",
    "1000FLOKI/USDT:USDT","DOGE/USDT:USDT","1000CAT/USDT:USDT","ACT/USDT:USDT",
    "SLERF/USDT:USDT","DEGEN/USDT:USDT","WIF/USDT:USDT","1000PEPE/USDT:USDT"
]

exchanges_to_check = [
    'bybit','okx','gateio','kucoin','huobi','coinex',
    'binanceusdm','kraken','phemex','bitget'
]

ema_fast, ema_slow, ema_long = 12, 50, 200
max_age_minutes = 150  # skip candles older than this
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")  # set this as a GitHub secret
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" if TELEGRAM_TOKEN else None
TG_MAX = 4000  # safe split size (Telegram limit ~4096 chars; keep margin)

# === Helpers ===
def get_ema(df, length):
    return df['close'].ewm(span=length, adjust=False).mean()

def fetch_candles(exchange, symbol, tf="2h", limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df
    except Exception:
        return pd.DataFrame()

def minutes_ago(ts):
    now = datetime.now(timezone.utc)
    diff = now - ts
    return int(diff.total_seconds() // 60)

def send_telegram_text_chunks(text):
    """Split by newline boundaries into chunks <= TG_MAX and send sequentially."""
    if not TELEGRAM_API or not TELEGRAM_CHAT:
        print("Telegram not configured (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID). Skipping send.")
        return

    # Split intelligently on newline to keep messages readable
    lines = text.splitlines()
    chunks = []
    cur = ""
    for line in lines:
        if len(cur) + len(line) + 1 <= TG_MAX:
            cur += (line + "\n")
        else:
            if cur:
                chunks.append(cur)
            # if single line is too long, chunk it
            if len(line) > TG_MAX:
                for i in range(0, len(line), TG_MAX):
                    chunks.append(line[i:i+TG_MAX])
                cur = ""
            else:
                cur = line + "\n"
    if cur:
        chunks.append(cur)

    for idx, chunk in enumerate(chunks):
        payload = {"chat_id": TELEGRAM_CHAT, "text": chunk}
        try:
            resp = requests.post(TELEGRAM_API, data=payload, timeout=15)
            if not resp.ok:
                print(f"Telegram send failed (part {idx+1}/{len(chunks)}):", resp.text)
            time.sleep(0.4)  # slight pause to avoid spamming
        except Exception as e:
            print("Telegram send exception:", e)

# === Strategy Evaluation ===
def evaluate(symbol):
    """
    Returns: (signal_str_or_None, details_list, missing_flag)
    - signal_str_or_None: formatted short signal text (or None)
    - details_list: list of lines describing what happened (for logs and telegram)
    - missing_flag: True if coin not on any exchange
    """
    details = []
    for ex_id in exchanges_to_check:
        try:
            ex_class = getattr(ccxt, ex_id)
            exchange = ex_class({'options': {'defaultType':'future'}})
            if not exchange.has.get("fetchOHLCV", False):
                continue

            # 2h boss chart
            df_2h = fetch_candles(exchange, symbol, tf="2h", limit=200)
            if df_2h.empty:
                continue

            df_2h["ema12"] = get_ema(df_2h, ema_fast)
            df_2h["ema50"] = get_ema(df_2h, ema_slow)
            df_2h["ema200"] = get_ema(df_2h, ema_long)

            last_2h = df_2h.iloc[-1]
            mins = minutes_ago(last_2h["time"])
            if mins > max_age_minutes:
                continue

            # Relaxed filter: require EMA12 < EMA50; EMA50 < EMA200 is optional (strong)
            boss_ok = last_2h["ema12"] < last_2h["ema50"]
            strong = boss_ok and (last_2h["ema50"] < last_2h["ema200"])

            if not boss_ok:
                details.append(f"{symbol} ({ex_id.upper()}) skipped - no 2h EMA12<EMA50")
                # stop at first exchange that returns data but fails boss filter:
                return None, details, False

            # 15m execution chart
            df_15m = fetch_candles(exchange, symbol, tf="15m", limit=200)
            if df_15m.empty:
                # This exchange gave 2h data but no 15m data -> treat as unavailable for entry; try next exchange
                continue

            df_15m["ema12"] = get_ema(df_15m, ema_fast)
            df_15m["ema50"] = get_ema(df_15m, ema_slow)

            last_15m = df_15m.iloc[-1]
            prev_15m = df_15m.iloc[-2]

            # fresh bearish cross on 15m (prefer fresh cross)
            bearish_cross = (prev_15m["ema12"] >= prev_15m["ema50"]) and (last_15m["ema12"] < last_15m["ema50"])

            tag = "✅ Strong" if strong else "⚠ Weak"
            if bearish_cross:
                signal = (f"SHORT SIGNAL: {symbol} ({ex_id.upper()}) @ {last_15m['close']:.6f} | {tag}")
            else:
                signal = None

            details.append(
                f"{symbol} ({ex_id.upper()}) @ {last_15m['close']:.6f} | "
                f"2h: EMA12={last_2h['ema12']:.4f}, EMA50={last_2h['ema50']:.4f}, EMA200={last_2h['ema200']:.4f} {tag} | "
                f"15m: EMA12={last_15m['ema12']:.4f}, EMA50={last_15m['ema50']:.4f} | 15m_cross={'YES' if bearish_cross else 'NO'}"
            )

            # Stop at first usable exchange (either gives signal or not)
            return signal, details, False

        except Exception:
            # ignore and try next exchange
            continue

    # if we finished all exchanges with no usable data
    return None, [f"{symbol} not available on any exchange"], True

# === Main ===
def main():
    signals = []
    details_all = []
    missing = []
    total = len(symbols)
    processed = 0
    strong_count = 0
    weak_count = 0

    for sym in tqdm(symbols, desc="Checking coins", unit="coin"):
        sig, det, is_missing = evaluate(sym)
        details_all.extend(det)
        if is_missing:
            missing.append(sym)
        else:
            processed += 1
        if sig:
            signals.append(sig)
            if "✅ Strong" in sig:
                strong_count += 1
            else:
                weak_count += 1

    # Build telegram text
    lines = []
    lines.append("=== SHORT SIGNALS ===")
    if signals:
        for s in signals:
            lines.append(s)
    else:
        lines.append("No valid short signals this run.")

    lines.append("")
    lines.append("=== MISSING COINS ===")
    if missing:
        for m in missing:
            lines.append(m)
    else:
        lines.append("None")

    lines.append("")
    lines.append("=== SUMMARY ===")
    lines.append(f"Checked: {total} | Processed: {processed} | Missing: {len(missing)} | Signals: {len(signals)} (Strong: {strong_count}, Weak: {weak_count})")

    # also append last few detailed lines (limit to e.g. last 40 lines to keep message reasonable)
    lines.append("")
    lines.append("=== LAST DETAILS (sample) ===")
    SAMPLE_LINES = 40
    for d in details_all[-SAMPLE_LINES:]:
        lines.append(d)

    final_text = "\n".join(lines)

    # send to Telegram (split into safe chunks)
    send_telegram_text_chunks(final_text)

    # print to console as well
    print(final_text)

if __name__ == "__main__":
    main()
