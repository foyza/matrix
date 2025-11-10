#!/usr/bin/env python3
"""
Whale Signal Bot - Обнаружение китовых сделок через Bybit
Только сигналы, без торговли
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import ccxt

from whale_detector import WhaleDetector
from signal_generator import SignalGenerator
from notifier import NotificationManager
from config import load_config, setup_logging

class WhaleSignalBot:
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logging()
        
        # Инициализация компонентов
        self.whale_detector = WhaleDetector(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.notifier = NotificationManager(self.config)
        
        # Инициализация биржи (Bybit вместо Binance)
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'  # или 'future' для фьючерсов
            }
        })
        
        # Состояние бота
        self.last_signals = {}
        self.symbols = self.config['symbols']
        
    async def get_market_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Получение рыночных данных с биржи"""
        try:
            # Получаем свечи (1 минута)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['time'] = df['timestamp']  # для совместимости
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Получение последних сделок"""
        try:
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            self.logger.error(f"Ошибка получения сделок {symbol}: {e}")
            return []
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Анализ символа на наличие китовых активностей"""
        try:
            # Получаем рыночные данные
            df = await self.get_market_data(symbol)
            if df.empty:
                return None
            
            # Получаем последние сделки для анализа объема
            recent_trades = await self.get_recent_trades(symbol)
            
            # Детектор китов
            whale_activity = self.whale_detector.detect_whales(df, recent_trades)
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
        self.logger.info(f"🏦 Используется биржа: {self.exchange.name}")
        
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
            await self.exchange.close()
            self.logger.info("✅ Бот остановлен")

if __name__ == "__main__":
    bot = WhaleSignalBot()
    asyncio.run(bot.run())
