import os
import logging
import yaml
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Загрузка конфигурации"""
    
    # Базовые настройки
    config = {
        'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT'],
        'exchange': os.getenv('EXCHANGE', 'bybit'),  # bybit, kucoin, okx
        'whale_detection': {
            'volume_multiplier': float(os.getenv('WHALE_VOLUME_MULTIPLIER', '5.0')),
            'cluster_window': int(os.getenv('CLUSTER_WINDOW_SECONDS', '30')),
            'min_cluster_trades': 2,
            'confidence_threshold': float(os.getenv('MIN_CONFIDENCE', '0.7'))
        },
        'notifications': {
            'telegram': {
                'enabled': bool(os.getenv('TELEGRAM_BOT_TOKEN')),
                'format': 'detailed'
            }
        },
        'CHECK_INTERVAL': int(os.getenv('CHECK_INTERVAL', '10'))
    }
    
    # Загрузка из YAML если есть
    config_path = 'config/signals.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
    
    # Переменные окружения
    env_vars = [
        'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'EXCHANGE'
    ]
    
    for var in env_vars:
        env_value = os.getenv(var)
        if env_value:
            config[var] = env_value
    
    return config

def setup_logging() -> logging.Logger:
    """Настройка логирования"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/whale_signals.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('WhaleSignalBot')
