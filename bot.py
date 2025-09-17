#!/usr/bin/env python3
"""
bot_project.py
High-precision SMC+ML pipeline + Telegram bot.
Commands:
 - /train    -> запускает оффлайн training (скачивает историю, тренирует модели)
 - /backtest -> запускает vectorbt backtest и возвращает краткий отчет
 - /start    -> приветствие
 - "🔄 Получить сигнал" -> сгенерировать онлайн-сигнал (использует сохранённые модели)
Notes:
 - Создай .env с TELEGRAM_TOKEN, TWELVEDATA_API_KEY, NEWSAPI_KEY (NEWSAPI optional)
 - Модели сохраняются в ./models
"""
import os
import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import vectorbt as vbt  # backtest engine

# ---------------- CONFIG ----------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ASSET = os.getenv("ASSET", "BTC/USD")
INTERVAL = os.getenv("INTERVAL", "15m")  # 5m/15m/1h
HIST_PAGES = int(os.getenv("HIST_PAGES", "6"))  # how many pages/requests to pull (TwelveData paging)
MODEL_DIR = os.path.abspath(os.getenv("MODEL_DIR", "./models"))
os.makedirs(MODEL_DIR, exist_ok=True)

PROB_THRESH = float(os.getenv("PROB_THRESH", "0.75"))
SMC_CONFIRM_REQUIRED = os.getenv("SMC_CONFIRM_REQUIRED", "True").lower() in ("1","true","yes")
MIN_VOLUME_FOR_SIGNAL = float(os.getenv("MIN_VOLUME_FOR_SIGNAL","1.2"))

logging.basicConfig(level=logging.INFO)
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ---------------- Telegram ----------------
if not TELEGRAM_TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN in .env")
bot = Bot(token=TELEGRAM_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot)
user_settings = {}

# ---------------- Helpers: TwelveData ----------------
async def fetch_twelvedata(symbol: str, interval: str = "15m", outputsize: int = 5000):
    """
    Fetch time_series from TwelveData. outputsize default 5000.
    For very long history call multiple times (this script uses one call).
    """
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "apikey": TWELVEDATA_API_KEY, "outputsize": outputsize, "format":"JSON"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=60) as resp:
            data = await resp.json()
    if "values" not in data:
        logging.warning("TwelveData response: %s", data)
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ---------------- Indicators & SMC ----------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + (up.rolling(14).mean() / down.rolling(14).mean())))
    ema12 = df["close"].ewm(span=12, adjust=False).mean(); ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["volume_avg20"] = df["volume"].rolling(20).mean().fillna(0)
    df["volume_spike"] = (df["volume"] > df["volume_avg20"] * MIN_VOLUME_FOR_SIGNAL).astype(int)
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iat[i] > df["close"].iat[i-1]:
            obv.append(obv[-1] + df["volume"].iat[i])
        elif df["close"].iat[i] < df["close"].iat[i-1]:
            obv.append(obv[-1] - df["volume"].iat[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    return df.dropna().reset_index(drop=True)

def detect_order_blocks(df: pd.DataFrame, lookback=50, vol_mult=2.0):
    blocks = []
    for i in range(lookback, len(df)):
        vol = df["volume"].iat[i]
        vol_avg = df["volume_avg20"].iat[i] if df["volume_avg20"].iat[i] > 0 else 1
        body = abs(df["close"].iat[i] - df["open"].iat[i])
        atr = df["atr"].iat[i] if df["atr"].iat[i] > 0 else 0.0001
        if vol > vol_mult * vol_avg and body > 0.6 * atr:
            direction = "buy" if df["close"].iat[i] > df["open"].iat[i] else "sell"
            blocks.append({"index": i, "time": df["datetime"].iat[i], "price": float(df["close"].iat[i]), "dir": direction})
    return blocks

def detect_liquidity_grab(df: pd.DataFrame, lookback=20, vol_mult=1.5):
    grabs = []
    highs = df["high"].rolling(lookback).max()
    lows = df["low"].rolling(lookback).min()
    for i in range(lookback+1, len(df)):
        prev_high = highs.iat[i-1]
        prev_low = lows.iat[i-1]
        if prev_high and df["high"].iat[i] > prev_high and df["close"].iat[i] < prev_high and df["volume"].iat[i] > vol_mult * df["volume_avg20"].iat[i]:
            grabs.append({"index": i, "time": df["datetime"].iat[i], "dir":"sell"})
        elif prev_low and df["low"].iat[i] < prev_low and df["close"].iat[i] > prev_low and df["volume"].iat[i] > vol_mult * df["volume_avg20"].iat[i]:
            grabs.append({"index": i, "time": df["datetime"].iat[i], "dir":"buy"})
    return grabs

def detect_fvg(df: pd.DataFrame):
    fvg = []
    for i in range(2, len(df)):
        # simplistic heuristics for FVG
        if df["low"].iat[i] > df["high"].iat[i-2]:
            fvg.append({"index": i, "dir":"buy"})
        if df["high"].iat[i] < df["low"].iat[i-2]:
            fvg.append({"index": i, "dir":"sell"})
    return fvg

# ---------------- Features & Labels ----------------
def build_features_labels(df: pd.DataFrame, horizon=3):
    df = df.copy()
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    feats = ["ema10","ema50","rsi","macd","atr","obv","volume","volume_spike"]
    X = df[feats].iloc[:-horizon].fillna(0)
    y = df["target"].iloc[:-horizon].astype(int)
    return X.reset_index(drop=True), y.reset_index(drop=True)

# ---------------- Training ----------------
def train_lightgbm(X: pd.DataFrame, y: pd.Series):
    model = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=64)
    model.fit(X, y)
    return model

def train_lstm(X: pd.DataFrame, y: pd.Series, epochs=6):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X3 = np.expand_dims(Xs, axis=1)  # timesteps=1
    model = Sequential([
        LSTM(64, input_shape=(X3.shape[1], X3.shape[2])),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X3, y.values, epochs=epochs, batch_size=256, verbose=0)
    return model, scaler

def walkforward_validate(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    precisions = []; recalls = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1]
        preds = (probs >= PROB_THRESH).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        precisions.append(prec); recalls.append(rec)
    return float(np.mean(precisions)), float(np.mean(recalls))

# ---------------- Pipeline: build & train ----------------
async def build_and_train(symbol=ASSET, interval=INTERVAL):
    logging.info("Fetching data...")
    df = await fetch_twelvedata(symbol, interval=interval)
    if df is None or len(df) < 500:
        raise RuntimeError("Not enough data to train.")
    df = compute_indicators(df)
    X, y = build_features_labels(df, horizon=3)
    logging.info("Shapes X=%s y=%s", X.shape, y.shape)

    logging.info("Walk-forward validation (estimate precision)...")
    prec, rec = walkforward_validate(X, y, n_splits=5)
    logging.info("WF precision=%.3f recall=%.3f", prec, rec)

    logging.info("Training LightGBM final...")
    lgbm = train_lightgbm(X, y)
    joblib.dump(lgbm, os.path.join(MODEL_DIR, "lgbm.pkl"))

    logging.info("Training LSTM...")
    lstm, lstm_scaler = train_lstm(X, y, epochs=6)
    lstm.save(os.path.join(MODEL_DIR, "lstm.h5"))
    joblib.dump(lstm_scaler, os.path.join(MODEL_DIR, "lstm_scaler.pkl"))

    feat_scaler = StandardScaler().fit(X)
    joblib.dump(feat_scaler, os.path.join(MODEL_DIR, "feat_scaler.pkl"))

    # save last training timestamp
    joblib.dump({"trained_at": datetime.utcnow().isoformat(), "wf_prec": prec, "wf_rec": rec}, os.path.join(MODEL_DIR, "meta.pkl"))
    logging.info("Models saved to %s", MODEL_DIR)
    return prec, rec

# ---------------- Load models ----------------
def load_models():
    lgbm = joblib.load(os.path.join(MODEL_DIR, "lgbm.pkl"))
    feat_scaler = joblib.load(os.path.join(MODEL_DIR, "feat_scaler.pkl"))
    lstm_scaler = joblib.load(os.path.join(MODEL_DIR, "lstm_scaler.pkl"))
    lstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm.h5"))
    meta = joblib.load(os.path.join(MODEL_DIR, "meta.pkl"))
    return lgbm, feat_scaler, lstm, lstm_scaler, meta

# ---------------- Online signal generation ----------------
async def generate_signal(symbol=ASSET, interval=INTERVAL):
    df = await fetch_twelvedata(symbol, interval=interval)
    if df is None or len(df) < 80:
        return {"ok":False, "reason":"no data"}
    df = compute_indicators(df)
    blocks = detect_order_blocks(df)
    grabs = detect_liquidity_grab(df)
    fvg = detect_fvg(df)
    try:
        lgbm, feat_scaler, lstm, lstm_scaler, meta = load_models()
    except Exception as e:
        logging.exception("Load models: %s", e)
        return {"ok":False, "reason":"models missing"}

    feats = ["ema10","ema50","rsi","macd","atr","obv","volume","volume_spike"]
    latest = df[feats].iloc[-1:].fillna(0)
    Xs = feat_scaler.transform(latest)
    prob = float(lgbm.predict_proba(Xs)[0][1])
    Xs_l = lstm_scaler.transform(latest)
    X3 = np.expand_dims(Xs_l, axis=1)
    prob_lstm = float(lstm.predict(X3)[0][0])
    prob_ens = (prob + prob_lstm) / 2.0

    # SMC confirm
    last_idx = len(df)-1
    smc_confirm = False; smc_dir = None
    if grabs and grabs[-1]["index"] >= last_idx - 12:
        smc_confirm = True; smc_dir = grabs[-1]["dir"]
    elif blocks and blocks[-1]["index"] >= last_idx - 24:
        smc_confirm = True; smc_dir = blocks[-1]["dir"]
    elif fvg and fvg[-1]["index"] >= last_idx - 24:
        smc_confirm = True; smc_dir = fvg[-1]["dir"]

    accepted = False; direction = "neutral"; reason = ""
    if prob_ens >= PROB_THRESH and (smc_confirm or not SMC_CONFIRM_REQUIRED):
        direction = "buy" if prob_ens >= 0.5 else "sell"
        accepted = True
        reason = "High prob ensemble + SMC"
    else:
        if prob >= PROB_THRESH and prob_lstm >= PROB_THRESH:
            direction = "buy"; accepted = True; reason = "Both ML high prob buy"
        elif prob <= (1-PROB_THRESH) and prob_lstm <= (1-PROB_THRESH):
            direction = "sell"; accepted = True; reason = "Both ML high prob sell"
        else:
            reason = "No consensus / low prob"

    # volume check
    vol_ok = df["volume"].iat[-1] >= df["volume_avg20"].iat[-1] * MIN_VOLUME_FOR_SIGNAL
    if accepted and not vol_ok:
        accepted = False
        reason += " (volume low)"

    price = float(df["close"].iat[-1])
    atr = float(df["atr"].iat[-1]) if df["atr"].iat[-1] > 0 else price * 0.005
    tp = round(price + 2*atr if direction=="buy" else price - 2*atr, 2)
    sl = round(price - 1*atr if direction=="buy" else price + 1*atr, 2)
    news_sent = 0.0  # optionally compute via NEWSAPI

    return {
        "ok": True,
        "accepted": accepted,
        "direction": direction,
        "prob_lgbm": prob,
        "prob_lstm": prob_lstm,
        "prob_ens": prob_ens,
        "smc_confirm": smc_confirm,
        "smc_dir": smc_dir,
        "price": price,
        "tp": tp,
        "sl": sl,
        "reason": reason
    }

# ---------------- Backtest (vectorbt) ----------------
def run_backtest_local(symbol_df: pd.DataFrame, signals: pd.Series, price_col="close"):
    """
    Quick vectorbt backtest:
    - signals: 1 buy, -1 sell, 0 no-op (we'll keep it simple: market entries, fixed TP/SL not simulated)
    - We'll compute returns from entering at next-bar open and exiting after horizon or opposite signal.
    This is illustrative; for production use more careful trade logic.
    """
    price = symbol_df[price_col]
    entries = signals == 1
    # assume exit after fixed horizon (e.g., 3 bars)
    pf = vbt.Portfolio.from_signals(close=price, entries=entries, exits=None, init_cash=10000, fees=0.0005, slippage=0.0005)
    stats = pf.stats()
    return stats

# ---------------- Telegram handlers ----------------
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Получить сигнал")],
            [KeyboardButton(text="BTC/USD"), KeyboardButton(text="XAU/USD"), KeyboardButton(text="ETH/USD")],
            [KeyboardButton(text="🔕 Mute"), KeyboardButton(text="🔔 Unmute")]
        ],
        resize_keyboard=True
    )

@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    user_settings[message.from_user.id] = {"asset": ASSET, "muted": False}
    await message.answer("High-precision SMC+ML бот запущен. Нажми кнопку, чтобы получить сигнал.", reply_markup=get_main_keyboard())

@dp.message()
async def msg_handler(message: types.Message):
    uid = message.from_user.id
    text = message.text
    if uid not in user_settings:
        user_settings[uid] = {"asset": ASSET, "muted": False}
    if text == "🔄 Получить сигнал":
        await message.answer("Генерируем сигнал...")
        out = await generate_signal(user_settings[uid]["asset"], INTERVAL)
        if not out.get("ok"):
            await message.answer(f"Ошибка: {out.get('reason')}")
            return
        if out["accepted"]:
            msg = (f"📢 <b>{user_settings[uid]['asset']}</b>\n"
                   f"Direction: <b>{out['direction'].upper()}</b>\n"
                   f"ProbLGBM: {out['prob_lgbm']:.2f}, ProbLSTM: {out['prob_lstm']:.2f}\n"
                   f"Price: {out['price']}\nTP: {out['tp']}  SL: {out['sl']}\nReason: {out['reason']}")
        else:
            msg = f"Нет надёжного сигнала ({out['reason']})."
        await message.answer(msg)
    elif text == "/train":
        await message.answer("Начинаю обучение (это может занять несколько минут)...")
        try:
            prec, rec = await build_and_train(symbol=user_settings[uid].get("asset", ASSET), interval=INTERVAL)
            await message.answer(f"Обучение завершено. WF precision={prec:.3f}, recall={rec:.3f}")
        except Exception as e:
            logging.exception("Train failed: %s", e)
            await message.answer(f"Ошибка обучения: {e}")
    elif text == "/backtest":
        await message.answer("Запуск бэктеста (локально)...")
        try:
            df = await fetch_twelvedata(user_settings[uid].get("asset", ASSET), interval=INTERVAL)
            df = compute_indicators(df)
            X, y = build_features_labels(df)
            lgbm, feat_scaler, lstm, lstm_scaler, meta = load_models()
            latest_feats = feat_scaler.transform(X)
            probs = lgbm.predict_proba(latest_feats)[:,1]
            # strict threshold orchestration -> generate signals series
            sigs = (probs >= PROB_THRESH).astype(int)
            stats = run_backtest_local(df, pd.Series(sigs, index=df.index[:len(sigs)]))
            await message.answer(f"Backtest done. stats summary:\n{stats[['Total Return','Max Drawdown','Sharpe Ratio']].to_string()}")
        except Exception as e:
            logging.exception("Backtest failed: %s", e)
            await message.answer(f"Backtest error: {e}")
    elif text in ["BTC/USD","XAU/USD","ETH/USD"]:
        user_settings[uid]["asset"] = text
        await message.answer(f"Актив изменён: {text}")
    elif text == "🔕 Mute":
        user_settings[uid]["muted"] = True
        await message.answer("Оповещения отключены")
    elif text == "🔔 Unmute":
        user_settings[uid]["muted"] = False
        await message.answer("Оповещения включены")
    else:
        await message.answer("Используй клавиатуру.", reply_markup=get_main_keyboard())

# ---------------- Main ----------------
async def main():
    # if models missing, inform user (but don't auto-train)
    models_present = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in ["lgbm.pkl","lstm.h5","lstm_scaler.pkl","feat_scaler.pkl","meta.pkl"])
    if not models_present:
        logging.warning("Models not found in %s. Use /train to build models.", MODEL_DIR)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
