import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio

@dataclass
class RiskParameters:
    max_daily_loss: float = 0.02  # 2%
    max_position_size: float = 0.1  # 10% капитала
    max_drawdown: float = 0.05  # 5%
    max_open_positions: int = 3

class ProfessionalWhaleBot:
    def __init__(self, symbol: str = "BTCUSD", risk_params: RiskParameters = None):
        self.symbol = symbol
        self.risk_params = risk_params or RiskParameters()
        self.setup_logging()
        self.initialize_mt5()
        
        # Trading state
        self.daily_pnl = 0.0
        self.open_positions = []
        self.signals_cache = []
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('whale_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_mt5(self):
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            raise Exception("MT5 init failed")
        self.logger.info("MT5 initialized successfully")
    
    def get_market_data(self, timeframe=mt5.TIMEFRAME_M1, count=100) -> pd.DataFrame:
        """Получение рыночных данных"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None:
            self.logger.error("Failed to get market data")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict:
        """Расчет продвинутых индикаторов"""
        if df.empty:
            return {}
            
        # Volume analysis
        df['volume_sma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
        
        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Support/Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        
        current = df.iloc[-1]
        signals = {
            'volume_spike': current['volume_ratio'] > 2.0,
            'high_volatility': current['volatility'] > df['volatility'].quantile(0.8),
            'near_resistance': abs(current['close'] - current['resistance']) / current['close'] < 0.002,
            'near_support': abs(current['close'] - current['support']) / current['close'] < 0.002,
            'trend': 1 if current['close'] > df['close'].rolling(50).mean().iloc[-1] else -1
        }
        
        return signals
    
    def detect_whale_activity(self, df: pd.DataFrame) -> Dict:
        """Обнаружение китовой активности"""
        if len(df) < 50:
            return {}
            
        # Анализ больших свечей
        recent_candles = df.tail(5)
        large_candles = recent_candles[
            (recent_candles['high'] - recent_candles['low']) > 
            df['high'].rolling(20).mean().iloc[-1] * 1.5
        ]
        
        # Анализ объема
        volume_spike = df['tick_volume'].iloc[-1] > df['tick_volume'].rolling(20).mean().iloc[-1] * 2
        
        return {
            'whale_detected': len(large_candles) > 0 and volume_spike,
            'direction': 1 if large_candles['close'].iloc[-1] > large_candles['open'].iloc[-1] else -1,
            'confidence': min(len(large_candles) / 5.0, 1.0)
        }
    
    def calculate_position_size(self, confidence: float) -> float:
        """Расчет размера позиции на основе уверенности и риска"""
        account_info = mt5.account_info()
        if account_info is None:
            return 0.0
            
        equity = account_info.equity
        max_risk_amount = equity * self.risk_params.max_position_size
        
        # Размер основан на уверенности сигнала
        size_multiplier = min(confidence, 1.0)
        position_size = max_risk_amount * size_multiplier
        
        self.logger.info(f"Calculated position size: {position_size:.2f}")
        return position_size
    
    def execute_trade(self, direction: str, confidence: float):
        """Исполнение сделки с управлением рисками"""
        if len(self.open_positions) >= self.risk_params.max_open_positions:
            self.logger.warning("Max open positions reached")
            return
            
        if self.daily_pnl < -self.risk_params.max_daily_loss * mt5.account_info().equity:
            self.logger.error("Daily loss limit exceeded")
            return
        
        # Расчет размера позиции
        lot_size = self.calculate_position_size(confidence)
        if lot_size <= 0:
            return
            
        # Определение цены и стоп-лоссов
        current_price = mt5.symbol_info_tick(self.symbol).bid
        spread = mt5.symbol_info(self.symbol).spread * mt5.symbol_info(self.symbol).point
        
        if direction == "BUY":
            sl = current_price * 0.99  # 1% стоп-лосс
            tp = current_price * 1.02  # 2% тейк-профит
        else:
            sl = current_price * 1.01
            tp = current_price * 0.98
        
        # Отправка ордера
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": current_price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 12345,
            "comment": f"WhaleBot_{direction}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Trade execution failed: {result.retcode}")
        else:
            self.logger.info(f"Trade executed: {direction} {lot_size} lots")
            self.open_positions.append(result.order)
    
    def monitor_positions(self):
        """Мониторинг открытых позиций"""
        positions = mt5.positions_get(symbol=self.symbol)
        self.open_positions = [pos.ticket for pos in positions] if positions else []
        
        # Обновление дневного PnL
        total_pnl = sum(pos.profit for pos in positions) if positions else 0
        self.daily_pnl = total_pnl
    
    def trading_decision(self) -> Optional[str]:
        """Принятие торгового решения"""
        df = self.get_market_data()
        if df.empty:
            return None
        
        indicators = self.calculate_advanced_indicators(df)
        whale_signals = self.detect_whale_activity(df)
        
        if not whale_signals.get('whale_detected', False):
            return None
        
        direction = "BUY" if whale_signals['direction'] > 0 else "SELL"
        confidence = whale_signals['confidence']
        
        # Фильтры ложных сигналов
        if indicators.get('near_resistance', False) and direction == "BUY":
            self.logger.info("Filtered: Buy signal near resistance")
            return None
            
        if indicators.get('near_support', False) and direction == "SELL":
            self.logger.info("Filtered: Sell signal near support")
            return None
        
        # Только высококачественные сигналы
        if confidence > 0.7 and indicators.get('volume_spike', False):
            return direction
        elif confidence > 0.8:  # Очень уверенные сигналы
            return direction
            
        return None
    
    def run(self):
        """Основной цикл торговли"""
        self.logger.info("Starting Professional Whale Bot")
        
        try:
            while True:
                self.monitor_positions()
                
                decision = self.trading_decision()
                if decision:
                    self.execute_trade(decision, 0.7)  # Базовая уверенность
                
                # Пауза между анализами
                mt5.sleep(5000)  # 5 секунд
                
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
        finally:
            mt5.shutdown()

# Использование
if __name__ == "__main__":
    risk_params = RiskParameters(
        max_daily_loss=0.01,  # 1%
        max_position_size=0.05,  # 5%
        max_drawdown=0.03,  # 3%
        max_open_positions=2
    )
    
    bot = ProfessionalWhaleBot(symbol="BTCUSD", risk_params=risk_params)
    bot.run()
