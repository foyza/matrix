"""
bot_smc_high_precision.py
High-precision signal generator + Telegram interface.
Design: offline training -> model persistence -> online signal generation with strict filters.
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

# ========== CONFIG ==========
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ASSET = "BTC/USD"   # можно заменить
INTERVAL = "15m"    # таймфрейм; 5m/15m/1h — тестируй
HIST_DAYS = 365*2   # сколько дней истории скачать (рекомендуется 1-3 года)
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# thresholds — строгие, чтобы увеличить precision
PROB_THRESH = 0.75   # требуемая вероятность LightGBM
SMC_CONFIRM_REQUIRED = True  # требуем подтверждение SMC (order block / liquidity grab)
MIN_VOLUME_FOR_SIGNAL = 1.2  # объём должен быть >= avg20 * factor

logging.basicConfig(level=logging.INFO)
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Telegram bot
bot = Bot(token=TELEGRAM_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot)
user_settings = {}  # {uid: {"asset":ASSET, "muted":False}}

# ========== DATA FETCH ==========
async def fetch_twelvedata(symbol, interval="15m", start=None, end=None, outputsize=10000):
    """
    Uses TwelveData time_series endpoint. For large historical windows, call in loops by pages.
    """
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "apikey": TWELVEDATA_API_KEY, "outputsize": 5000, "format":"JSON"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=30) as resp:
            data = await resp.json()
    if "values" not in data:
        logging.warning("TwelveData returned: %s", data)
        return None
    df = pd.DataFrame(data["values"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    numeric = ["open","high","low","close","volume"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ========== INDICATORS & SMC ==========
def compute_indicators(df):
    df = df.copy()
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + (up.rolling(14).mean() / down.rolling(14).mean())))
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean(); ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    # ATR
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["volume_avg20"] = df["volume"].rolling(20).mean().fillna(0)
    df["volume_spike"] = (df["volume"] > df["volume_avg20"] * MIN_VOLUME_FOR_SIGNAL).astype(int)
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    return df.dropna().reset_index(drop=True)

def detect_order_blocks(df, lookback=50, vol_mult=2.0):
    blocks = []
    for i in range(lookback, len(df)):
        vol = df["volume"].iloc[i]
        vol_avg = df["volume_avg20"].iloc[i] if df["volume_avg20"].iloc[i] > 0 else 1
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        atr = df["atr"].iloc[i] if df["atr"].iloc[i] > 0 else 0.0001
        if vol > vol_mult * vol_avg and body > 0.6 * atr:
            dirc = "buy" if df["close"].iloc[i] > df["open"].iloc[i] else "sell"
            blocks.append({"index": i, "time": df["datetime"].iloc[i], "price": float(df["close"].iloc[i]), "dir": dirc})
    return blocks

def detect_liquidity_grabs(df, lookback=20, vol_mult=1.5):
    grabs = []
    highs = df["high"].rolling(lookback).max()
    lows = df["low"].rolling(lookback).min()
    for i in range(lookback+1, len(df)):
        prev_high = highs.iloc[i-1]
        prev_low = lows.iloc[i-1]
        if prev_high and df["high"].iloc[i] > prev_high and df["close"].iloc[i] < prev_high and df["volume"].iloc[i] > vol_mult * df["volume_avg20"].iloc[i]:
            grabs.append({"index": i, "time": df["datetime"].iloc[i], "dir":"sell"})
        elif prev_low and df["low"].iloc[i] < prev_low and df["close"].iloc[i] > prev_low and df["volume"].iloc[i] > vol_mult * df["volume_avg20"].iloc[i]:
            grabs.append({"index": i, "time": df["datetime"].iloc[i], "dir":"buy"})
    return grabs

def detect_fvg(df):
    # simple 3-candle FVG: gap between candle1 and candle3 bodies
    fvg = []
    for i in range(2, len(df)):
        c1_h, c1_l = df["high"].iloc[i-2], df["low"].iloc[i-2]
        c3_h, c3_l = df["high"].iloc[i], df["low"].iloc[i]
        # bullish gap
        if df["close"].iloc[i-2] < df["open"].iloc[i-2] and df["close"].iloc[i] > df["open"].iloc[i] and (df["low"].iloc[i] > df["high"].iloc[i-2]):
            fvg.append({"index": i, "dir": "buy"})
        # bearish gap
        if df["close"].iloc[i-2] > df["open"].iloc[i-2] and df["close"].iloc[i] < df["open"].iloc[i] and (df["high"].iloc[i] < df["low"].iloc[i-2]):
            fvg.append({"index": i, "dir": "sell"})
    return fvg

# ========== FEATURES & LABELS ==========
def build_features_labels(df, horizon=3):
    """
    horizon: сколько свечей вперед считать движение (например, 3 candles)
    label: 1 если close.shift(-horizon) > close -> buy, else 0
    """
    df = df.copy()
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    # features
    feats = ["ema10","ema50","rsi","macd","atr","obv","volume","volume_spike"]
    X = df[feats].iloc[:-horizon].fillna(0)
    y = df["target"].iloc[:-horizon].astype(int)
    return X.reset_index(drop=True), y.reset_index(drop=True)

# ========== TRAINING ==========
def train_lightgbm(X, y):
    params = {"objective":"binary","metric":"binary_logloss","verbosity":-1,"boosting":"gbdt","num_leaves":64,"learning_rate":0.03,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5}
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain], early_stopping_rounds=50, verbose_eval=False)
    return model

def train_lstm(X, y, epochs=10):
    # scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X3 = np.expand_dims(Xs, axis=1)  # timesteps=1
    model = Sequential([
        LSTM(64, input_shape=(X3.shape[1], X3.shape[2])),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X3, y.values, epochs=epochs, batch_size=128, verbose=0)
    return model, scaler

def walkforward_validate(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    precisions = []
    recalls = []
    fold = 0
    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # train simple model
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1]
        # apply strict threshold to get high precision
        preds = (probs >= PROB_THRESH).astype(int)
        if preds.sum() == 0:
            prec = 0.0
        else:
            prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        logging.info(f"WF fold {fold}: prec={prec:.3f}, rec={rec:.3f}, signals={preds.sum()}")
        precisions.append(prec); recalls.append(rec)
    return np.mean(precisions), np.mean(recalls)

# ========== FULL PIPELINE ==========
async def build_and_train(symbol=ASSET, interval=INTERVAL, days=HIST_DAYS):
    # fetch
    logging.info("Fetching historical data...")
    df = await fetch_twelvedata(symbol, interval=interval)
    if df is None or len(df) < 500:
        raise RuntimeError("Not enough history")
    df = compute_indicators(df)
    X, y = build_features_labels(df, horizon=3)
    logging.info("Data shapes X=%s y=%s", X.shape, y.shape)

    # quick walkforward to estimate precision
    logging.info("Running walk-forward validation (strict threshold)...")
    prec, rec = walkforward_validate(X, y, n_splits=5)
    logging.info("Estimated precision (WF) = %.3f, recall=%.3f", prec, rec)

    # full train LGB on all data
    logging.info("Training final LightGBM...")
    lgbm_final = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03)
    lgbm_final.fit(X, y)
    joblib.dump(lgbm_final, os.path.join(MODEL_DIR, "lgbm_final.pkl"))

    # LSTM (smaller)
    logging.info("Training LSTM (may be slower)...")
    lstm_model, scaler = train_lstm(X, y, epochs=5)
    lstm_model.save(os.path.join(MODEL_DIR, "lstm_final.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler.pkl"))

    # store feature scaler for online use
    feat_scaler = StandardScaler().fit(X)
    joblib.dump(feat_scaler, os.path.join(MODEL_DIR, "feat_scaler.pkl"))

    logging.info("Training complete. Stored models to %s", MODEL_DIR)
    return prec, rec

# ========== ONLINE SIGNAL (uses persisted models) ==========
def load_models():
    lgbm = joblib.load(os.path.join(MODEL_DIR, "lgbm_final.pkl"))
    feat_scaler = joblib.load(os.path.join(MODEL_DIR, "feat_scaler.pkl"))
    lstm_scaler = joblib.load(os.path.join(MODEL_DIR, "lstm_scaler.pkl"))
    lstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_final.h5"))
    return lgbm, feat_scaler, lstm, lstm_scaler

async def generate_signal_for_asset(symbol=ASSET):
    df = await fetch_twelvedata(symbol, interval=INTERVAL)
    if df is None or len(df) < 60:
        return {"ok":False, "reason":"no data"}
    df = compute_indicators(df)
    # SMC detections
    blocks = detect_order_blocks(df)
    grabs = detect_liquidity_grabs(df)
    fvg = detect_fvg(df)
    # load models
    try:
        lgbm, feat_scaler, lstm, lstm_scaler = load_models()
    except Exception as e:
        logging.exception("load models failed: %s", e)
        return {"ok":False, "reason":"models missing"}

    # features last row
    feats = ["ema10","ema50","rsi","macd","atr","obv","volume","volume_spike"]
    latest = df[feats].iloc[-1:].fillna(0)
    Xs = feat_scaler.transform(latest)
    prob = float(lgbm.predict_proba(Xs)[0][1])
    # LSTM predict
    Xs_lstm = lstm_scaler.transform(latest)
    X3 = np.expand_dims(Xs_lstm, axis=1)
    prob_lstm = float(lstm.predict(X3)[0][0])
    # ensemble
    prob_ens = (prob + prob_lstm) / 2.0

    # SMC confirmation logic
    smc_confirm = False
    # last significant grab or block within N candles
    last_idx = len(df) - 1
    if grabs and grabs[-1]["index"] >= last_idx - 12:
        smc_confirm = True
        smc_dir = grabs[-1]["dir"]
    elif blocks and blocks[-1]["index"] >= last_idx - 24:
        smc_confirm = True
        smc_dir = blocks[-1]["dir"]
    elif fvg and fvg[-1]["index"] >= last_idx - 24:
        smc_confirm = True
        smc_dir = fvg[-1]["dir"]
    else:
        smc_dir = None

    # decision rules: require high probability + SMC confirmation OR both models agree with high prob
    direction = "neutral"
    accepted = False
    if prob_ens >= PROB_THRESH and smc_confirm:
        direction = "buy" if prob_ens >= 0.5 else "sell"
        accepted = True
        reason = "LGBM+LSTM high prob + SMC confirm"
    else:
        # both ML agree strongly
        if prob >= PROB_THRESH and prob_lstm >= PROB_THRESH:
            direction = "buy"
            accepted = True
            reason = "both ML high prob buy"
        elif prob <= (1-PROB_THRESH) and prob_lstm <= (1-PROB_THRESH):
            direction = "sell"
            accepted = True
            reason = "both ML high prob sell"
        else:
            reason = "no strong consensus"

    # final volume check
    vol_ok = df["volume"].iloc[-1] >= df["volume_avg20"].iloc[-1] * MIN_VOLUME_FOR_SIGNAL
    if accepted and not vol_ok:
        accepted = False
        reason += " (volume low) "

    # compute TP/SL using ATR
    price = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if df["atr"].iloc[-1] > 0 else price*0.005
    tp = round(price + 2*atr if direction=="buy" else price - 2*atr, 2)
    sl = round(price - 1*atr if direction=="buy" else price + 1*atr, 2)

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

# ========== Telegram handlers ==========
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
async def cmd_start(message: types.Message):
    user_settings[message.from_user.id] = {"asset": ASSET, "muted": False}
    await message.answer("Запущен high-precision SMC+ML бот. Нажми кнопку, чтобы получить сигнал.", reply_markup=get_main_keyboard())

@dp.message()
async def handlers(message: types.Message):
    uid = message.from_user.id
    text = message.text
    if uid not in user_settings:
        user_settings[uid] = {"asset": ASSET, "muted": False}
    if text == "🔄 Получить сигнал":
        await message.answer("Генерируем сигнал...")
        out = await generate_signal_for_asset(user_settings[uid]["asset"])
        if not out.get("ok"):
            await message.answer(f"Ошибка: {out.get('reason')}")
            return
        if out["accepted"]:
            msg = (f"📢 <b>{user_settings[uid]['asset']}</b>\n"
                   f"Direction: <b>{out['direction'].upper()}</b>\n"
                   f"Prob(LGBM)={out['prob_lgbm']:.2f}, Prob(LSTM)={out['prob_lstm']:.2f}\n"
                   f"Price: {out['price']}\nTP: {out['tp']}  SL: {out['sl']}\n"
                   f"Reason: {out['reason']}")
        else:
            msg = f"Нет надёжного сигнала ({out['reason']})."
        await message.answer(msg)
    elif text in ["BTC/USD","XAU/USD","ETH/USD"]:
        user_settings[uid]["asset"] = text
        await message.answer(f"Актив изменён на {text}")
    elif text == "🔕 Mute":
        user_settings[uid]["muted"] = True
        await message.answer("Отключены уведомления")
    elif text == "🔔 Unmute":
        user_settings[uid]["muted"] = False
        await message.answer("Включены уведомления")
    else:
        await message.answer("Используй кнопки.", reply_markup=get_main_keyboard())

# ========== MAIN ==========
async def main():
    # If no models, build and train
    if not (os.path.exists(os.path.join(MODEL_DIR,"lgbm_final.pkl")) and os.path.exists(os.path.join(MODEL_DIR,"lstm_final.h5"))):
        logging.info("Models not found — training from scratch (this can take several minutes)...")
        prec, rec = await build_and_train(symbol=ASSET, interval=INTERVAL, days=HIST_DAYS)
        logging.info("Train estimated precision %.3f recall %.3f", prec, rec)
    else:
        logging.info("Models found. Skipping training.")

    # start bot
    loop = asyncio.get_event_loop()
    # optional: auto-send to subscribers, implement as needed
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
