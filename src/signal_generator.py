import pandas as pd
from typing import Dict, Any
from datetime import datetime

class SignalGenerator:
    def __init__(self, config):
        self.config = config
        
    def generate_signal(self, symbol: str, whale_activity: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация торгового сигнала"""
        
        current_price = df['close'].iloc[-1]
        
        # Расчет уровней для сделки
        stop_loss, take_profit = self._calculate_levels(
            current_price, 
            whale_activity.direction,
            df
        )
        
        # Расчет силы сигнала
        signal_strength = self._calculate_signal_strength(whale_activity)
        
        # Формирование сообщения
        message = self._create_signal_message(symbol, whale_activity, current_price)
        
        return {
            'symbol': symbol,
            'direction': whale_activity.direction,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': whale_activity.confidence,
            'signal_strength': signal_strength,
            'volume_ratio': whale_activity.volume_ratio,
            'cluster_size': whale_activity.cluster_size,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'type': 'WHALE_SIGNAL'
        }
    
    def _calculate_levels(self, current_price: float, direction: str, df: pd.DataFrame) -> tuple:
        """Расчет стоп-лосса и тейк-профита"""
        # Волатильность для расчета риска
        atr = self._calculate_atr(df, period=14)
        
        if direction == 'LONG':
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3.0)
        else:  # SHORT
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3.0)
            
        return round(stop_loss, 5), round(take_profit, 5)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Расчет Average True Range"""
        if len(df) < period:
            return 0.0
            
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1] if not atr.empty else 0.0
    
    def _calculate_signal_strength(self, whale_activity) -> str:
        """Определение силы сигнала"""
        confidence = whale_activity.confidence
        
        if confidence >= 0.8:
            return "STRONG"
        elif confidence >= 0.6:
            return "MEDIUM"
        else:
            return "WEAK"
    
    def _create_signal_message(self, symbol: str, whale_activity, price: float) -> str:
        """Создание текстового сообщения сигнала"""
        
        direction_emoji = "🟢" if whale_activity.direction == 'LONG' else "🔴"
        strength_emoji = "🔥" if whale_activity.confidence > 0.8 else "⚡"
        
        # Определение силы сигнала
        if whale_activity.confidence >= 0.8:
            strength = "СИЛЬНЫЙ"
        elif whale_activity.confidence >= 0.6:
            strength = "СРЕДНИЙ"
        else:
            strength = "СЛАБЫЙ"
        
        message = f"""
{direction_emoji} *WHALE SIGNAL* {strength_emoji}

*Тикер:* `{symbol}`
*Направление:* {whale_activity.direction}
*Сила сигнала:* {strength}

*Цена:* `${price:,.4f}`
*Уверенность:* `{whale_activity.confidence:.0%}`

*Детали:*
• Объем: `x{whale_activity.volume_ratio:.1f}` от среднего
• Кластер: `{whale_activity.cluster_size}` крупных свеч
• Волатильность: `{whale_activity.price_impact:.2f}%`

*Рекомендация:*
Рассмотреть вход в направлении *{whale_activity.direction}*

⏰ *Время:* `{datetime.now().strftime('%H:%M:%S')}`
"""
        return message
