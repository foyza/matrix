import os
import logging
import asyncio
import httpx
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ta

# ===============================
# Загрузка окружения и логирование
# ===============================
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not TOKEN or not API_KEY:
    raise ValueError("TELEGRAM_TOKEN или TWELVEDATA_API_KEY не найдены в .env")

logging.basicConfig(level=logging.INFO)

# ===============================
# Инициализация бота и диспетчера
# ===============================
bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

# ===============================
# Глобальные структуры
# ===============================
user_settings = {}  # {uid: {"asset": str, "muted": bool}}
model = None
scaler = None

# ===============================
# Клавиатура
# ===============================
main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📈 Сигнал")],
        [KeyboardButton(text="⚙️ Настройки"), KeyboardButton(text="🔕 Вкл/Выкл уведомления")],
        [KeyboardButton(text="💾 Экспорт сигналов")]
    ],
    resize_keyboard=True
)

# ===============================
# Функции ML и данных
# ===============================
async def fetch_data(symbol: str, interval="1h", outputsize=100):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize={outputsize}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df["close"] = df["close"].astype(float)
    return df

def prepare_features(df: pd.DataFrame):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema"] = ta.trend.EMAIndicator(df["close"], window=14).ema_indicator()
    df = df.dropna()
    X = df[["rsi", "ema"]].values
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    y = y[:-1]
    X = X[:-1]
    return X, y

def train_model(df: pd.DataFrame):
    global model, scaler
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)

def smc_analysis(df: pd.DataFrame):
    """
    Упрощённая логика Smart Money Concepts:
    - Определяем ликвидность: max/min за последние свечи
    - Сравниваем текущую цену с уровнями
    """
    recent = df.tail(20)
    high = recent["close"].max()
    low = recent["close"].min()
    last = recent["close"].iloc[-1]
    if last >= high:
        return "🟢 Ликвидность выбита сверху (бычий сценарий)"
    elif last <= low:
        return "🔴 Ликвидность выбита снизу (медвежий сценарий)"
    else:
        return "⚪ Цена внутри диапазона (наблюдаем)"

async def send_signal(user_id: int):
    asset = user_settings.get(user_id, {}).get("asset", "AAPL")
    df = await fetch_data(asset)
    if df is None or len(df) < 20:
        await bot.send_message(user_id, f"Не удалось получить данные по {asset}")
        return

    if model is None:
        train_model(df)

    X, _ = prepare_features(df)
    X_scaled = scaler.transform([X[-1]])
    pred = model.predict(X_scaled)[0]

    smc_text = smc_analysis(df)
    direction = "🟢 BUY" if pred == 1 else "🔴 SELL"

    signal = (
        f"📊 Сигнал по {asset}\n"
        f"{direction}\n"
        f"{smc_text}\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    await bot.send_message(user_id, signal)

    # Экспорт в CSV
    with open("signals.csv", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()},{asset},{direction},{smc_text}\n")

# ===============================
# Хэндлеры
# ===============================
@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    user_settings[message.from_user.id] = {"asset": "AAPL", "muted": False}
    await message.answer("Привет! Я бот для трейдинга с SMC и ML 📈", reply_markup=main_kb)

@dp.message(lambda m: m.text == "📈 Сигнал")
async def signal_cmd(message: types.Message):
    await send_signal(message.from_user.id)

@dp.message(lambda m: m.text == "⚙️ Настройки")
async def settings_cmd(message: types.Message):
    await message.answer("Напиши тикер актива (например, AAPL, BTC/USD):")

@dp.message(lambda m: m.text == "🔕 Вкл/Выкл уведомления")
async def mute_cmd(message: types.Message):
    uid = message.from_user.id
    user_settings[uid]["muted"] = not user_settings[uid]["muted"]
    status = "🔔 Включены" if not user_settings[uid]["muted"] else "🔕 Выключены"
    await message.answer(f"Уведомления: {status}")

@dp.message(lambda m: m.text == "💾 Экспорт сигналов")
async def export_cmd(message: types.Message):
    if os.path.exists("signals.csv"):
        await message.answer_document(types.FSInputFile("signals.csv"))
    else:
        await message.answer("Сигналы пока не сохранены.")

@dp.message()
async def custom_asset(message: types.Message):
    uid = message.from_user.id
    if uid in user_settings:
        user_settings[uid]["asset"] = message.text.strip().upper()
        await message.answer(f"Ассет изменён на {user_settings[uid]['asset']}")

# ===============================
# Планировщик
# ===============================
async def scheduler_task():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        lambda: [asyncio.create_task(send_signal(uid)) for uid, s in user_settings.items() if not s["muted"]],
        "interval",
        minutes=60
    )
    scheduler.start()

# ===============================
# Точка входа
# ===============================
async def main():
    await scheduler_task()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
