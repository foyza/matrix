import os
import logging
import asyncio
import numpy as np
import pandas as pd
import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message
from aiogram.filters import CommandStart
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# === suppress TF spam ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# === load .env ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not TOKEN:
    raise ValueError("❌ BOT_TOKEN не найден в .env")

# === logging ===
logging.basicConfig(level=logging.INFO)

# === aiogram bot ===
bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

# === NLTK sentiment ===
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# === ML models ===
scaler = StandardScaler()
model_gb = GradientBoostingClassifier()
model_lstm = Sequential([
    LSTM(32, input_shape=(1, 6)),
    Dense(1, activation="sigmoid")
])
ml_trained = False

# === helpers ===
async def get_twelvedata(symbol="BTC/USD", interval="1h", n=100):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={n}&apikey={TWELVEDATA_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if "values" not in data:
                return None
            df = pd.DataFrame(data["values"])
            df = df.rename(columns={"datetime": "time"}).astype(float, errors="ignore")
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)
            return df

def add_indicators(df):
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = 100 - (100 / (1 + df["close"].pct_change().rolling(14).mean()))
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    return df.fillna(0)

def detect_smc(df, volume_threshold=1.5):
    recent_high = df["high"].iloc[-20:-1].max()
    recent_low = df["low"].iloc[-20:-1].min()
    last_close = df["close"].iloc[-1]
    last_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].iloc[-20:].mean()

    signal, reason = "neutral", "нет сигнала"
    if last_close > recent_high and last_volume > avg_volume * volume_threshold:
        signal, reason = "buy", "Киты вынесли стопы сверху"
    elif last_close < recent_low and last_volume > avg_volume * volume_threshold:
        signal, reason = "sell", "Киты вынесли стопы снизу"
    return signal, reason

async def get_news_sentiment(asset="bitcoin"):
    url = f"https://newsapi.org/v2/everything?q={asset}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if "articles" not in data:
                return 0
            texts = [a["title"] + " " + a.get("description", "") for a in data["articles"][:10]]
            scores = [sia.polarity_scores(t)["compound"] for t in texts]
            return np.mean(scores) if scores else 0

async def send_signal(uid, asset="BTC/USD"):
    df = await get_twelvedata(asset)
    if df is None or len(df) < 50:
        await bot.send_message(uid, f"⚠️ Нет данных для {asset}")
        return
    df = add_indicators(df)

    dir_ml, acc_ml = "neutral", 50
    if ml_trained:
        latest = df[["ema10","ema50","rsi","macd","atr","obv"]].iloc[-1]
        X = scaler.transform([latest])
        prob_gb = model_gb.predict_proba(X)[0]
        prob_lstm = model_lstm.predict(np.expand_dims(X, axis=1))[0][0]
        prob = (prob_gb[1] + prob_lstm) / 2
        if prob > 0.55: dir_ml = "buy"
        elif prob < 0.45: dir_ml = "sell"
        acc_ml = int(prob * 100)

    smc_signal, smc_reason = detect_smc(df)
    if smc_signal != "neutral":
        direction = smc_signal
        accuracy = max(acc_ml, 70)
    else:
        direction, accuracy = dir_ml, acc_ml

    news_score = await get_news_sentiment(asset)
    if news_score > 0.15 and direction != "sell":
        direction, accuracy = "buy", min(100, accuracy + 10)
    elif news_score < -0.15 and direction != "buy":
        direction, accuracy = "sell", min(100, accuracy + 10)

    price = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1]
    tp = round(price + atr*2 if direction=="buy" else price - atr*2, 2)
    sl = round(price - atr if direction=="buy" else price + atr, 2)

    msg = (
        f"📢 Сигнал для <b>{asset}</b>\n"
        f"Направление: <b>{direction.upper()}</b>\n"
        f"Цена: {price}\n"
        f"🟢 TP: {tp}\n"
        f"🔴 SL: {sl}\n"
        f"📊 Точность: {accuracy}%\n"
        f"🐋 SMC: {smc_reason}\n"
        f"📰 Новости: {'позитив' if news_score>0 else 'негатив' if news_score<0 else 'нейтрально'}"
    )
    await bot.send_message(uid, msg)

# === handlers ===
@dp.message(CommandStart())
async def start_cmd(message: Message):
    await message.answer("👋 Привет! Я трейдинг-бот с ML + Smart Money Concepts.\nНапиши /signal для BTC/USD")

@dp.message(lambda m: m.text and m.text.lower() == "/signal")
async def signal_cmd(message: Message):
    await send_signal(message.from_user.id, "BTC/USD")

# === main ===
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
