import asyncio
import aiohttp
import pandas as pd
import logging
from typing import Dict, List, Optional
import os
from datetime import datetime, timedelta

class TwelveDataProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.logger = logging.getLogger(__name__)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_ohlcv(self, symbol: str, interval: str = '1min', output_size: int = 100) -> pd.DataFrame:
        """Получение OHLCV данных"""
        try:
            url = f"{self.base_url}/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': output_size,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'values' not in data:
                        self.logger.warning(f"Нет данных для {symbol}: {data.get('message', 'Unknown error')}")
                        return pd.DataFrame()
                    
                    # Конвертируем в DataFrame
                    df = pd.DataFrame(data['values'])
                    df = df.iloc[::-1].reset_index(drop=True)  # реверс чтобы последняя свеча была последней
                    
                    # Конвертируем типы данных
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df['open'] = pd.to_numeric(df['open'])
                    df['high'] = pd.to_numeric(df['high'])
                    df['low'] = pd.to_numeric(df['low'])
                    df['close'] = pd.to_numeric(df['close'])
                    df['volume'] = pd.to_numeric(df['volume'])
                    
                    self.logger.debug(f"Получено {len(df)} свечей для {symbol}")
                    return df
                else:
                    error_text = await response.text()
                    self.logger.error(f"Ошибка API для {symbol}: {response.status} - {error_text}")
                    return pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"Ошибка получения OHLCV для {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_realtime_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены"""
        try:
            url = f"{self.base_url}/price"
            params = {
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('price', 0))
                else:
                    error_text = await response.text()
                    self.logger.error(f"Ошибка получения цены для {symbol}: {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Получение котировки с объемом"""
        try:
            url = f"{self.base_url}/quote"
            params = {
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'symbol': data.get('symbol'),
                        'price': float(data.get('close', 0)),
                        'volume': float(data.get('volume', 0)),
                        'change': float(data.get('percent_change', 0)),
                        'high': float(data.get('high', 0)),
                        'low': float(data.get('low', 0)),
                        'open': float(data.get('open', 0))
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Ошибка получения котировки для {symbol}: {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Ошибка получения котировки для {symbol}: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """Тест подключения к API"""
        try:
            # Пробуем получить цену BTC
            price = await self.get_realtime_price('BTC/USD')
            if price and price > 0:
                self.logger.info(f"✅ Twelvedata подключение успешно. BTC цена: ${price:,.2f}")
                return True
            else:
                self.logger.error("❌ Не удалось получить данные от Twelvedata")
                return False
        except Exception as e:
            self.logger.error(f"❌ Ошибка подключения к Twelvedata: {e}")
            return False
