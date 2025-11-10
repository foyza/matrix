import asyncio
import logging
from typing import Dict, Any
from telegram import Bot
from telegram.error import TelegramError

class NotificationManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Инициализация Telegram бота
        self.telegram_bot = None
        if config['notifications']['telegram']['enabled']:
            token = config.get('TELEGRAM_BOT_TOKEN')
            if token and token != 'your_telegram_bot_token_here':
                self.telegram_bot = Bot(token=token)
            else:
                self.logger.warning("Telegram token не настроен")
        
    async def send_signal(self, signal: Dict[str, Any]):
        """Отправка сигнала в Telegram"""
        if not self.telegram_bot:
            self.logger.warning("Telegram бот не инициализирован")
            return
            
        try:
            chat_id = self.config.get('TELEGRAM_CHAT_ID')
            if not chat_id or chat_id == 'your_chat_id_here':
                self.logger.warning("TELEGRAM_CHAT_ID не настроен")
                return
                
            message = self._format_signal_message(signal)
            
            await self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            self.logger.info(f"📨 Сигнал отправлен в Telegram: {signal['symbol']}")
            
        except TelegramError as e:
            self.logger.error(f"Ошибка отправки в Telegram: {e}")
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка Telegram: {e}")
    
    def _format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Форматирование сообщения для Telegram"""
        direction_emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
        strength_emoji = "🔥" if signal['confidence'] > 0.8 else "⚡"
        
        # Определение силы сигнала
        if signal['confidence'] >= 0.8:
            strength = "СИЛЬНЫЙ"
        elif signal['confidence'] >= 0.6:
            strength = "СРЕДНИЙ"
        else:
            strength = "СЛАБЫЙ"
        
        message = f"""
{direction_emoji} *WHALE SIGNAL* {strength_emoji}

*Тикер:* `{signal['symbol']}`
*Направление:* {signal['direction']}
*Сила сигнала:* {strength}

*Цена:* `${signal['current_price']:,.4f}`
*Уверенность:* `{signal['confidence']:.0%}`

*Детали:*
• Объем: `x{signal['volume_ratio']:.1f}` от среднего
• Кластер: `{signal['cluster_size']}` крупных свеч
• Волатильность: `{signal['price_impact']:.2f}%`

*Уровни:*
• Стоп-лосс: `${signal['stop_loss']:,.4f}`
• Тейк-профит: `${signal['take_profit']:,.4f}`

*Рекомендация:*
Рассмотреть вход в направлении *{signal['direction']}*

⏰ *Время:* `{signal['timestamp'][11:19]}`
"""
        return message
