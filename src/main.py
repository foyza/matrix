#!/usr/bin/env python3
"""
Whale Signal Bot - Обнаружение китовых сделок через Twelvedata
Только сигналы, без торговли
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import os

import pandas as pd
import numpy as np

from whale_detector import WhaleDetector
from signal_generator import SignalGenerator
from notifier import NotificationManager
from config import load_config, setup_logging
from data_provider import TwelveDataProvider

class WhaleSignalBot:
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logging()
        
        # Инициализация компонентов
        self.whale_detector = WhaleDetector(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.notifier = NotificationManager(self.config)
        
        # Инициализация провайдера данных
        api_key = os.getenv('TWELVEDATA_API_KEY')
        if not api_key:
            self.logger.error("❌ TWELVEDATA_API_KEY не установлен в .env файле")
            raise ValueError("TWELVEDATA_API_KEY required")
            
        self.data_provider = TwelveDataProvider(api_key)
        
        # Состояние бота
        self.last_signals = {}
        self.symbols = self.config['symbols']
        
    async def get_market_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Получение рыночных данных"""
        try:
            # Используем USD вместо USDT для Twelvedata
            twelvedata_symbol = symbol.replace('/USDT', '/USD')
            
            df = await self.data_provider.get_ohlcv(
                symbol=twelvedata_symbol,
                interval='1min',
                output_size=limit
            )
            
            if not df.empty:
                self.logger.debug(f"📊 Получено {len(df)} свечей для {symbol}")
            else:
                self.logger.warning(f"⚠️ Нет данных для {symbol}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_recent_quote(self, symbol: str) -> Optional[Dict]:
        """Получение текущей котировки"""
        try:
            twelvedata_symbol = symbol.replace('/USDT', '/USD')
            quote = await self.data_provider.get_quote(twelvedata_symbol)
            return quote
        except Exception as e:
            self.logger.error(f"Ошибка получения котировки {symbol}: {e}")
            return None
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Анализ символа на наличие китовых активностей"""
        try:
            # Получаем рыночные данные
            df = await self.get_market_data(symbol)
            if df.empty:
                return None
            
            # Получаем текущую котировку для дополнительной информации
            current_quote = await self.get_recent_quote(symbol)
            
            # Детектор китов
            whale_activity = self.whale_detector.detect_whales(df, current_quote)
            if not whale_activity['whale_detected']:
                return None
            
            # Генератор сигналов
            signal = self.signal_generator.generate_signal(symbol, whale_activity, df)
            
            # Фильтрация слабых сигналов
            if signal['confidence'] < self.config['whale_detection']['confidence_threshold']:
                return None
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа {symbol}: {e}")
            return None
    
    async def process_signals(self):
        """Обработка сигналов по всем символам"""
        current_time = datetime.now()
        signals_found = []
        
        for symbol in self.symbols:
            try:
                signal = await self.analyze_symbol(symbol)
                if signal:
                    # Проверяем, не был ли уже отправлен похожий сигнал
                    signal_id = f"{symbol}_{signal['direction']}"
                    last_signal_time = self.last_signals.get(signal_id)
                    
                    # Отправляем не чаще чем раз в 5 минут для того же символа/направления
                    if (last_signal_time is None or 
                        (current_time - last_signal_time).total_seconds() > 300):
                        
                        signals_found.append(signal)
                        self.last_signals[signal_id] = current_time
                        self.logger.info(f"🐋 Обнаружен сигнал: {symbol} {signal['direction']} (уверенность: {signal['confidence']:.0%})")
                        
            except Exception as e:
                self.logger.error(f"Ошибка обработки {symbol}: {e}")
                continue
        
        return signals_found
    
    async def run(self):
        """Основной цикл бота"""
        self.logger.info("🚀 Запуск Whale Signal Bot...")
        self.logger.info(f"📊 Мониторинг символов: {', '.join(self.symbols)}")
        self.logger.info("🏦 Используется провайдер: Twelvedata")
        
        # Тестируем подключение
        async with self.data_provider as provider:
            self.data_provider = provider
            
            if not await self.data_provider.test_connection():
                self.logger.error("Не удалось подключиться к Twelvedata. Проверьте API ключ.")
                return
            
            try:
                while True:
                    # Анализ рынка
                    signals = await self.process_signals()
                    
                    # Отправка уведомлений
                    for signal in signals:
                        await self.notifier.send_signal(signal)
                        self.logger.info(f"📨 Отправлен сигнал: {signal['symbol']} {signal['direction']}")
                    
                    # Пауза между анализами
                    check_interval = self.config.get('CHECK_INTERVAL', 10)
                    await asyncio.sleep(check_interval)
                    
            except KeyboardInterrupt:
                self.logger.info("🛑 Остановка бота...")
            except Exception as e:
                self.logger.error(f"💥 Критическая ошибка: {e}")
            finally:
                self.logger.info("✅ Бот остановлен")

if __name__ == "__main__":
    bot = WhaleSignalBot()
    asyncio.run(bot.run())
