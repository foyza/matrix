import pandas as pd
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class WhaleActivity:
    whale_detected: bool
    direction: str  # 'LONG' or 'SHORT'
    confidence: float
    volume_ratio: float
    cluster_size: int
    price_impact: float

class WhaleDetector:
    def __init__(self, config):
        self.config = config
        self.volume_multiplier = config['whale_detection']['volume_multiplier']
        self.cluster_window = config['whale_detection']['cluster_window']
        
    def detect_whales(self, df: pd.DataFrame, recent_trades: List[Dict]) -> Dict[str, Any]:
        """Обнаружение китовых активностей"""
        
        if df.empty:
            return WhaleActivity(
                whale_detected=False,
                direction='NEUTRAL',
                confidence=0.0,
                volume_ratio=0.0,
                cluster_size=0,
                price_impact=0.0
            )
        
        # Анализ объема
        volume_analysis = self._analyze_volume(df)
        
        # Анализ кластеров
        cluster_analysis = self._analyze_clusters(df)
        
        # Анализ цены
        price_analysis = self._analyze_price_movement(df)
        
        # Совокупный анализ
        whale_detected = (volume_analysis['volume_spike'] or 
                         cluster_analysis['cluster_detected'])
        
        direction = self._determine_direction(volume_analysis, cluster_analysis, price_analysis)
        confidence = self._calculate_confidence(volume_analysis, cluster_analysis, price_analysis)
        
        return WhaleActivity(
            whale_detected=whale_detected,
            direction=direction,
            confidence=confidence,
            volume_ratio=volume_analysis['volume_ratio'],
            cluster_size=cluster_analysis['cluster_size'],
            price_impact=price_analysis['price_change_pct']
        )
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ объемов"""
        if len(df) < 20:
            return {'volume_spike': False, 'volume_ratio': 0.0}
        
        # Текущий объем
        current_volume = df['volume'].iloc[-1]
        
        # Средний объем за последние 20 свечей
        avg_volume = df['volume'].tail(20).mean()
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        volume_spike = volume_ratio >= self.volume_multiplier
        
        return {
            'volume_spike': volume_spike,
            'volume_ratio': volume_ratio,
            'current_volume': current_volume,
            'avg_volume': avg_volume
        }
    
    def _analyze_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ кластеров крупных свечей"""
        if len(df) < 10:
            return {'cluster_detected': False, 'cluster_size': 0}
        
        # Определяем крупные свечи (выше среднего диапазона и объема)
        avg_range = (df['high'] - df['low']).tail(20).mean()
        avg_volume = df['volume'].tail(20).mean()
        
        large_candles = df[
            ((df['high'] - df['low']) > avg_range * 1.5) &
            (df['volume'] > avg_volume * 2.0)
        ].tail(5)
        
        cluster_detected = len(large_candles) >= 2
        cluster_size = len(large_candles)
        
        return {
            'cluster_detected': cluster_detected,
            'cluster_size': cluster_size,
            'large_candles': large_candles
        }
    
    def _analyze_price_movement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ движения цены"""
        if len(df) < 5:
            return {'price_change_pct': 0.0, 'trend': 'NEUTRAL'}
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-5]
        price_change_pct = (current_price - prev_price) / prev_price * 100
        
        # Определение тренда по EMA
        ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
        ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
        
        if ema_9 > ema_21 and price_change_pct > 0:
            trend = 'LONG'
        elif ema_9 < ema_21 and price_change_pct < 0:
            trend = 'SHORT'
        else:
            trend = 'NEUTRAL'
            
        return {
            'price_change_pct': price_change_pct,
            'trend': trend,
            'momentum': price_change_pct
        }
    
    def _determine_direction(self, volume_analysis, cluster_analysis, price_analysis) -> str:
        """Определение направления сделки"""
        return price_analysis['trend']
    
    def _calculate_confidence(self, volume_analysis, cluster_analysis, price_analysis) -> float:
        """Расчет уверенности в сигнале"""
        confidence = 0.0
        
        # Вес объема (40%)
        if volume_analysis['volume_spike']:
            confidence += 0.4
        else:
            confidence += 0.2 * min(volume_analysis['volume_ratio'] / self.volume_multiplier, 1.0)
            
        # Вес кластера (30%)
        if cluster_analysis['cluster_detected']:
            confidence += 0.3 * min(cluster_analysis['cluster_size'] / 3.0, 1.0)
            
        # Вес движения цены (30%)
        price_strength = min(abs(price_analysis['price_change_pct']) / 2.0, 1.0)
        confidence += 0.3 * price_strength
        
        return min(confidence, 1.0)
