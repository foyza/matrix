# main.py
import os
import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -------------------------
# CONFIG / ENV
# -------------------------
# suppress TF spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
# Optional: BINANCE API or other for orderbook/OI (not mandatory)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

ASSETS = ['BTC/USD', 'XAU/USD', 'ETH/USD']

if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN (or BOT_TOKEN) not set in .env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

# -------------------------
# BOT & DISPATCHER
# -------------------------
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

user_settings = {}  # {uid: {"asset": ... , "muted": False}}

# -------------------------
# ML + LSTM (placeholders)
# -------------------------
model_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
scaler = StandardScaler()
model_lstm = None
ml_trained = False

# -------------------------
# NLP (VADER)
# -------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# -------------------------
# UI: Keyboard (restore original buttons)
# -------------------------
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Получить сигнал")],
            [KeyboardButton(text="BTC/USD"), KeyboardButton(text="XAU/USD"), KeyboardButton(text="ETH/USD")],
            [KeyboardButton(text="🔕 Mute"), KeyboardButton(text="🔔 Unmute")],
            [KeyboardButton(text="ℹ️ Помощь"), KeyboardButton(text="📰 Новости")]
        ],
        resize_keyboard=True
    )

# -------------------------
# DATA: TwelveData wrapper
# -------------------------
TD_INTERVAL_MAP = {"D1": "1day", "H4": "4h", "H1": "1h", "M15": "15min"}

async def get_twelvedata(asset, interval="1h", count=150):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": asset, "interval": interval, "outputsize": count, "apikey": TWELVEDATA_API_KEY}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, timeout=20) as response:
                data = await response.json()
        except Exception as e:
            logger.exception("TwelveData request failed")
            return None
    if not data or "values" not in data:
        logger.warning("No values from TwelveData: %s %s -> %s", asset, interval, data)
        return None
    df = pd.DataFrame(data["values"])
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df["datetime"] = pd.to_datetime(df.index)
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0
    return df

# -------------------------
# NEWS sentiment
# -------------------------
async def get_news_sentiment(asset):
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "ethereum"
    if not NEWSAPI_KEY:
        return 0
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "sortBy": "publishedAt", "apiKey": NEWSAPI_KEY, "language": "en", "pageSize": 8}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, timeout=15) as r:
                data = await r.json()
        except Exception:
            return 0
    if "articles" not in data:
        return 0
    scores = []
    for art in data["articles"][:6]:
        title = (art.get("title") or "").lower()
        description = (art.get("description") or "").lower()
        if any(word in title+description for word in ["fed","cpi","interest","regulation","etf","halving"]):
            text = (art.get("title") or "") + " " + (art.get("description") or "")
            scores.append(sia.polarity_scores(text)["compound"])
    return float(np.mean(scores)) if scores else 0

# -------------------------
# INDICATORS (all preserved)
# -------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def compute_atr(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(close, volume):
    obv = [0]
    for i in range(1,len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + (volume.iloc[i] if not np.isnan(volume.iloc[i]) else 0))
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - (volume.iloc[i] if not np.isnan(volume.iloc[i]) else 0))
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def compute_bollinger(series, period=20, dev=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + dev*std, ma - dev*std

def add_indicators(df):
    df = df.copy()
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df["high"], df["low"], df["close"])
    df["bb_upper"], df["bb_lower"] = compute_bollinger(df["close"])
    if "volume" in df.columns:
        df["obv"] = compute_obv(df["close"], df["volume"])
    else:
        df["obv"] = 0
    df = df.dropna()
    return df

# -------------------------
# Volume Profile (approx) & order-flow proxy
# -------------------------
def compute_volume_profile(df, bins=30, period_bars=200):
    recent = df.tail(period_bars)
    if recent.empty or recent["volume"].sum() == 0:
        return {"poc": None, "hv_ratio": 0}
    prices = (recent["high"] + recent["low"] + recent["close"]) / 3
    hist = np.histogram(prices, bins=bins, weights=recent["volume"])
    bin_edges = hist[1]
    bin_vols = hist[0]
    max_idx = np.argmax(bin_vols)
    poc = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
    hv_ratio = bin_vols[max_idx] / (bin_vols.sum() / len(bin_vols))
    return {"poc": float(poc), "hv_ratio": float(hv_ratio)}

def compute_order_flow_proxy(df, lookback=50):
    recent = df.tail(lookback).copy()
    if recent.empty or recent["volume"].sum() == 0:
        return {"of_delta": 0, "of_cum": 0}
    recent["prev_close"] = recent["close"].shift(1)
    recent["signed_vol"] = np.where(recent["close"] > recent["prev_close"], recent["volume"],
                                    np.where(recent["close"] < recent["prev_close"], -recent["volume"], 0))
    of_delta = recent["signed_vol"].iloc[-1]
    of_cum = recent["signed_vol"].sum()
    return {"of_delta": float(of_delta), "of_cum": float(of_cum)}

# -------------------------
# Market structure detection & Stop-Hunt (SMC)
# -------------------------
def detect_market_structure(df, lookback=50, swing_window=3):
    if df is None or df.empty:
        return "flat", {}
    close = df["close"]
    highs = []
    lows = []
    for i in range(swing_window, len(close)-swing_window):
        window = close.iloc[i-swing_window:i+swing_window+1]
        if close.iloc[i] == window.max():
            highs.append((df["datetime"].iloc[i], float(close.iloc[i]), i))
        if close.iloc[i] == window.min():
            lows.append((df["datetime"].iloc[i], float(close.iloc[i]), i))
    last_highs = highs[-3:]
    last_lows = lows[-3:]
    info = {"last_highs": last_highs, "last_lows": last_lows}
    try:
        if len(last_highs) >= 2 and len(last_lows) >= 2:
            h_vals = [h[1] for h in last_highs[-2:]]
            l_vals = [l[1] for l in last_lows[-2:]]
            if h_vals[-1] > h_vals[-2] and l_vals[-1] > l_vals[-2]:
                return "up", info
            if h_vals[-1] < h_vals[-2] and l_vals[-1] < l_vals[-2]:
                return "down", info
    except Exception:
        pass
    try:
        if df["ema10"].iloc[-1] > df["ema50"].iloc[-1]:
            return "up", info
        if df["ema10"].iloc[-1] < df["ema50"].iloc[-1]:
            return "down", info
    except Exception:
        pass
    return "flat", info

def detect_stop_hunt(df, lookback=30, wick_ratio_threshold=1.5):
    recent = df.tail(lookback)
    if recent.empty: return False, {}
    idx = recent["volume"].idxmax()
    bar = recent.loc[idx]
    atr = recent["atr"].iloc[-1] if "atr" in recent.columns and not recent["atr"].isna().all() else None
    wick_up = bar["high"] - max(bar["open"], bar["close"])
    wick_down = min(bar["open"], bar["close"]) - bar["low"]
    is_spike = (bar["volume"] > recent["volume"].mean() * 2)
    is_long_wick = (atr is not None and (wick_up > atr * wick_ratio_threshold or wick_down > atr * wick_ratio_threshold))
    rev = False
    info = {"bar_idx": int(idx), "vol": float(bar["volume"]), "wick_up": float(wick_up), "wick_down": float(wick_down)}
    try:
        nxt = df.loc[idx+1]
        if wick_up > wick_down and nxt["close"] < bar["close"]:
            rev = True
        if wick_down > wick_up and nxt["close"] > bar["close"]:
            rev = True
    except Exception:
        pass
    detected = is_spike and is_long_wick and rev
    return bool(detected), info

# -------------------------
# Higher timeframe trend helper
# -------------------------
async def get_higher_tf_trend(asset, interval="4h", count=200):
    df = await get_twelvedata(asset, interval=interval, count=count)
    if df is None or len(df) < 50:
        return "neutral"
    df = add_indicators(df)
    last = df.iloc[-1]
    if last["ema10"] > last["ema50"]:
        return "bullish"
    if last["ema10"] < last["ema50"]:
        return "bearish"
    return "neutral"

# -------------------------
# ML TRAINING (kept)
# -------------------------
async def train_models(asset="BTC/USD"):
    global ml_trained, model_gb, scaler, model_lstm
    df = await get_twelvedata(asset, interval=TD_INTERVAL_MAP["H1"], count=500)
    if df is None:
        logger.warning("No data for training")
        return
    df = add_indicators(df)
    if len(df) < 60:
        logger.warning("Not enough data for training")
        return
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    features = df[["ema10","ema50","rsi","macd","atr","obv"]].iloc[:-3]
    labels = df["target"].iloc[:-3]
    X = scaler.fit_transform(features)
    y = labels
    model_gb.fit(X, y)
    X_lstm = np.expand_dims(X, axis=1)
    model_lstm = Sequential([
        LSTM(32, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
        Dense(1, activation="sigmoid")
    ])
    model_lstm.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    model_lstm.fit(X_lstm, y, epochs=3, verbose=0)
    ml_trained = True
    logger.info("✅ ML + LSTM обучены")

# -------------------------
# SIGNAL building (kept + extended)
# -------------------------
async def send_signal(uid, asset):
    # fetch multi TFs
    dfs = {}
    for tf in ["D1","H4","H1","M15"]:
        df_tf = await get_twelvedata(asset, interval=TD_INTERVAL_MAP[tf], count=500)
        dfs[tf] = add_indicators(df_tf) if df_tf is not None else None

    main_df = dfs.get("H1") or dfs.get("M15")
    if main_df is None or len(main_df) < 50:
        await bot.send_message(uid, f"⚠️ Нет данных для {asset}")
        return

    dir_ml, acc_ml = "neutral", 50
    if ml_trained:
        latest = main_df[["ema10","ema50","rsi","macd","atr","obv"]].iloc[-1]
        X = scaler.transform([latest])
        prob_gb = model_gb.predict_proba(X)[0][1]
        prob_lstm = float(model_lstm.predict(np.expand_dims(X, axis=1))[0][0])
        prob = (prob_gb + prob_lstm) / 2
        if prob > 0.55:
            dir_ml = "buy"
        elif prob < 0.45:
            dir_ml = "sell"
        acc_ml = int(prob * 100)

    news_score = await get_news_sentiment(asset)

    # SMC: stop-hunt + order-flow proxy + volume profile
    vp = compute_volume_profile(main_df, bins=40, period_bars=200)
    of = compute_order_flow_proxy(main_df, lookback=100)
    stop_hunt_detected, sh_info = detect_stop_hunt(main_df.tail(200))

    primary_struct, p_info = detect_market_structure(dfs.get("D1") or main_df, lookback=200)
    secondary_struct, s_info = detect_market_structure(dfs.get("H1") or main_df, lookback=100)

    direction = dir_ml
    accuracy = acc_ml

    # news influence
    if news_score > 0.15 and direction != "sell":
        direction = "buy"; accuracy = min(100, accuracy + 10)
    elif news_score < -0.15 and direction != "buy":
        direction = "sell"; accuracy = min(100, accuracy + 10)

    # market structure influence
    if primary_struct == "up" and secondary_struct == "up":
        if direction == "sell":
            direction = "neutral"; accuracy = max(30, accuracy - 10)
        else:
            direction = "buy"; accuracy = min(95, accuracy + 8)
    elif primary_struct == "down" and secondary_struct == "down":
        if direction == "buy":
            direction = "neutral"; accuracy = max(30, accuracy - 10)
        else:
            direction = "sell"; accuracy = min(95, accuracy + 8)
    else:
        if direction in ("buy","sell"):
            accuracy = max(40, accuracy - 12)

    # volume profile influence
    price = main_df["close"].iloc[-1]
    if vp.get("poc") is not None:
        dist = abs(price - vp["poc"]) / max(1e-8, price)
        if dist < 0.002 and vp.get("hv_ratio",0) > 1.5:
            accuracy = max(30, accuracy - 15)

    # order-flow proxy
    if of["of_cum"] > main_df["volume"].tail(50).mean() * 10:
        direction = "buy"; accuracy = min(98, accuracy + 7)
    if of["of_cum"] < -main_df["volume"].tail(50).mean() * 10:
        direction = "sell"; accuracy = min(98, accuracy + 7)

    # stop-hunt handling
    if stop_hunt_detected:
        accuracy = max(25, int(accuracy * 0.6))
        direction = "neutral" if accuracy < 45 else direction

    # higher timeframe (H4/D1) confirmation
    higher_tf_h4 = await get_higher_tf_trend(asset, interval="4h")
    higher_tf_d1 = await get_higher_tf_trend(asset, interval="1day")

    # require that 'kiti' (stop-hunt) + trend confirm
    smc_reason = ""
    if stop_hunt_detected:
        # if stop-hunt direction aligns with direction and HTF aligns -> confirm
        sh_dir = "buy" if sh_info.get("wick_down",0) > sh_info.get("wick_up",0) else "sell"
        # map h4/d1 to buy/sell bias
        h4_bias = "buy" if higher_tf_h4 == "bullish" else "sell" if higher_tf_h4 == "bearish" else None
        d1_bias = "buy" if higher_tf_d1 == "bullish" else "sell" if higher_tf_d1 == "bearish" else None
        # require at least one HTF aligning
        if (h4_bias == sh_dir) or (d1_bias == sh_dir):
            # accept SMC override
            direction = sh_dir
            accuracy = max(accuracy, 70)
            smc_reason = f"Киты вынесли стопы ({'вверх' if sh_dir=='buy' else 'вниз'}) + HTF подтверждение"
        else:
            smc_reason = "Киты заметны, но HTF не совпадает"
            # we may reduce confidence
            accuracy = max(30, accuracy - 15)
    else:
        smc_reason = "Киты не подтвердили движение"

    # final TP/SL using ATR
    atr = main_df["atr"].iloc[-1] if "atr" in main_df.columns and not np.isnan(main_df["atr"].iloc[-1]) else None
    if not atr or atr <= 0:
        atr = (main_df["high"].iloc[-20:].max() - main_df["low"].iloc[-20:].min()) / 50
    tp_price = round(price + atr*2 if direction=="buy" else price - atr*2, 2) if direction in ("buy","sell") else None
    sl_price = round(price - atr*1 if direction=="buy" else price + atr*1, 2) if direction in ("buy","sell") else None

    # prepare message (include reason + all info)
    lines = [
        f"📢 Сигнал для <b>{asset}</b>",
        f"Направление: <b>{direction.upper()}</b>",
        f"Цена: {price}",
    ]
    if tp_price and sl_price:
        lines += [f"🟢 TP: {tp_price}", f"🔴 SL: {sl_price}"]
    lines += [
        f"📊 Точность (прибл): {accuracy}%",
        f"📰 Новости: {'позитив' if news_score>0 else 'негатив' if news_score<0 else 'нейтрально'}",
        f"📈 Структура (Primary D1): {primary_struct.upper()}  |  (Secondary H1): {secondary_struct.upper()}",
    ]
    if vp.get("poc") is not None:
        lines += [f"🔎 Volume Profile POC: {vp['poc']:.2f}  | HV_ratio: {vp['hv_ratio']:.2f}"]
    lines += [f"📦 OrderFlow proxy Δ: {int(of['of_delta'])}  cum: {int(of['of_cum'])}"]
    lines += [f"📌 Stop-Hunt: {'YES' if stop_hunt_detected else 'no'}  {smc_reason}"]
    msg = "\n".join(lines)
    muted = user_settings.get(uid,{}).get("muted",False)
    await bot.send_message(uid, msg, disable_notification=muted)

# -------------------------
# HANDLERS (commands + buttons) — restore original UI
# -------------------------
@dp.message(CommandStart())
async def cmd_start(message: Message):
    user_settings[message.from_user.id] = {"asset":"BTC/USD","muted":False}
    await message.answer("Escape the matrix.", reply_markup=get_main_keyboard())

@dp.message()
async def generic_handler(message: Message):
    uid = message.from_user.id
    text = (message.text or "").strip()
    if uid not in user_settings:
        user_settings[uid] = {"asset":"BTC/USD","muted":False}

    # Buttons
    if text == "🔄 Получить сигнал":
        await send_signal(uid, user_settings[uid]["asset"])
        return
    if text in ASSETS:
        user_settings[uid]["asset"] = text
        await message.answer(f"✅ Актив установлен: {text}")
        return
    if text == "🔕 Mute":
        user_settings[uid]["muted"] = True
        await message.answer("🔕 Уведомления отключены")
        return
    if text == "🔔 Unmute":
        user_settings[uid]["muted"] = False
        await message.answer("🔔 Уведомления включены")
        return
    if text == "ℹ️ Помощь" or text.lower() == "/help":
        help_text = (
            "Команды:\n"
            "/start - запустить бота\n"
            "/signal - получить сигнал для BTC/USD\n\n"
            "Кнопки:\n"
            "🔄 Получить сигнал - отправляет текущий сигнал\n"
            "BTC/USD / XAU/USD / ETH/USD - сменить актив\n"
            "🔕 Mute / 🔔 Unmute - отключить/включить уведомления\n"
            "📰 Новости - показать краткие новости (тональность)"
        )
        await message.answer(help_text)
        return
    if text == "📰 Новости":
        ns = await get_news_sentiment(user_settings[uid]["asset"])
        await message.answer(f"Новости (тональность): {ns:.3f}")
        return

    # Fallback: support slash commands too
    if text.lower() == "/signal":
        await send_signal(uid, user_settings[uid]["asset"])
        return
    if text.lower() == "/start":
        await cmd_start(message)
        return
    # otherwise echo or ignore
    await message.answer("Неизвестная команда. Нажми ℹ️ Помощь чтобы узнать команды.")

# -------------------------
# AUTO LOOP (preserve)
# -------------------------
async def auto_signal_loop():
    while True:
        for uid, settings in list(user_settings.items()):
            try:
                await send_signal(uid, settings["asset"])
            except Exception:
                logger.exception("Error in auto_signal_loop sending signal")
        await asyncio.sleep(900)  # 15 min

# -------------------------
# STARTUP
# -------------------------
async def main():
    # train models at startup (best-effort)
    try:
        await train_models("BTC/USD")
    except Exception:
        logger.exception("Training failed")
    # start auto loop and polling
    loop = asyncio.get_event_loop()
    loop.create_task(auto_signal_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
