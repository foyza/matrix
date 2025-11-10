def detect_whales(self, df: pd.DataFrame, current_quote: Dict = None) -> Dict[str, Any]:
    """Обнаружение китовых активностей для Twelvedata"""
    
    if df.empty:
        return {
            'whale_detected': False,
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'volume_ratio': 0.0,
            'cluster_size': 0,
            'price_impact': 0.0
        }
    
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
    
    return {
        'whale_detected': whale_detected,
        'direction': direction,
        'confidence': confidence,
        'volume_ratio': volume_analysis['volume_ratio'],
        'cluster_size': cluster_analysis['cluster_size'],
        'price_impact': price_analysis['price_change_pct']
    }
