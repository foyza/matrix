import os
import logging
import asyncio
import httpx
import numpy as np
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.client.default import DefaultBotProperties
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
import joblib

# ==================== НАСТРОЙКИ ====================
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("TWELVEDATA_API_KEY")

logging.basicConfig(level=logging.INFO)

bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()
user_settings = {}  # {uid: {"asset": ..., "muted": False}}

# ==================== УТИЛИТЫ ====================
def ensure_user(uid: int):
    if uid not in user_settings:
        user_settings[uid] = {"asset": "AAPL", "muted": False}

async def fetch_data(symbol: str = "AAPL", interval="1h", outputsize=100):
    url = f"https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "apikey": API_KEY, "outputsize": outputsize}
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params)
        data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["close"] = df["close"].astype(float)
    df = df[::-1].reset_index(drop=True)
    return df

def apply_smc_logic(df: pd.DataFrame):
    """
    Упрощённая Smart Money Concepts логика:
    - ищем ликвидность (сильные экстремумы)
    - смотрим, где были захваты ликвидности
    - если цена выносила стопы и вернулась — сигнал в сторону возврата
    """
    if df is None or len(df) < 5:
        return "Нет данных"
    closes = df["close"].values
    last = closes[-1]
    prev = closes[-5:]

    high = np.max(prev)
    low = np.min(prev)

    if last > high:
        return "BUY (SMC: захват ликвидности сверху)"
    elif last < low:
        return "SELL (SMC: захват ликвидности снизу)"
    else:
        return "WAIT (SMC: нет сигнала)"

def get_ml_signal(df: pd.DataFrame):
    """
    Заглушка ML — можно обучать на исторических данных.
    Сейчас просто LogisticRegression на простых фичах.
    """
    closes = df["close"].values
    X = np.array([[closes[i] - closes[i-1]] for i in range(1, len(closes))])
    y = np.array([1 if closes[i] > closes[i-1] else 0 for i in range(1, len(closes))])

    if len(set(y)) < 2:
        return "WAIT (ML: данных мало)"

    model = LogisticRegression()
    model.fit(X, y)
    pred = model.predict([[closes[-1] - closes[-2]]])[0]
    return "BUY (ML)" if pred == 1 else "SELL (ML)"

def get_signal(df: pd.DataFrame):
    smc = apply_smc_logic(df)
    ml = get_ml_signal(df)
    return f"{smc} | {ml}"

# ==================== ХЕНДЛЕРЫ ====================
@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    uid = message.from_user.id
    ensure_user(uid)

    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("📈 Сигнал"))
    kb.add(KeyboardButton("⚙️ Настройки"))
    kb.add(KeyboardButton("🔕 Вкл/Выкл уведомления"))

    await message.answer("Привет! Я бот для сигналов 📊\nВыберите действие:", reply_markup=kb)

@dp.message(lambda m: m.text == "📈 Сигнал")
async def signal_cmd(message: types.Message):
    uid = message.from_user.id
    ensure_user(uid)

    asset = user_settings[uid]["asset"]
    df = await fetch_data(asset)
    sig = get_signal(df)
    await message.answer(f"Сигнал по {asset}: {sig}")

@dp.message(lambda m: m.text == "⚙️ Настройки")
async def settings_cmd(message: types.Message):
    uid = message.from_user.id
    ensure_user(uid)

    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("AAPL"), KeyboardButton("BTC/USD"))
    kb.add(KeyboardButton("Назад"))

    await message.answer("Выберите актив:", reply_markup=kb)

@dp.message(lambda m: m.text in ["AAPL", "BTC/USD"])
async def set_asset(message: types.Message):
    uid = message.from_user.id
    ensure_user(uid)
    user_settings[uid]["asset"] = message.text
    await message.answer(f"Актив изменён на {message.text}")

@dp.message(lambda m: m.text == "Назад")
async def back_cmd(message: types.Message):
    uid = message.from_user.id
    ensure_user(uid)

    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("📈 Сигнал"))
    kb.add(KeyboardButton("⚙️ Настройки"))
    kb.add(KeyboardButton("🔕 Вкл/Выкл уведомления"))

    await message.answer("Вы вернулись в меню", reply_markup=kb)

@dp.message(lambda m: m.text == "🔕 Вкл/Выкл уведомления")
async def mute_cmd(message: types.Message):
    uid = message.from_user.id
    ensure_user(uid)

    user_settings[uid]["muted"] = not user_settings[uid]["muted"]
    status = "🔔 Включены" if not user_settings[uid]["muted"] else "🔕 Выключены"
    await message.answer(f"Уведомления: {status}")

# ==================== РАССЫЛКА ====================
async def broadcast_signal():
    for uid, settings in list(user_settings.items()):
        if settings.get("muted"):
            continue
        asset = settings.get("asset", "AAPL")
        df = await fetch_data(asset)
        sig = get_signal(df)
        try:
            await bot.send_message(uid, f"Автосигнал по {asset}: {sig}")
        except Exception as e:
            logging.error(f"Не удалось отправить {uid}: {e}")

# ==================== MAIN ====================
async def main():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(lambda: asyncio.create_task(broadcast_signal()), "interval", hours=1)
    scheduler.start()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
