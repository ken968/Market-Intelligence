"""
Forecast Analyzer - Automated Insights Generation
Analyzes forecast predictions and generates human-readable insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class ForecastAnalyzer:
    """Generate automated insights from price forecasts"""
    
    def __init__(self):
        pass
    
    def analyze_forecast(self, current_price: float, forecast_prices: List[float], 
                        asset_name: str, timeframes: List[str] = None) -> Dict:
        """
        Generate comprehensive insights from forecast
        
        Args:
            current_price: Current asset price
            forecast_prices: List of predicted prices
            asset_name: Name of asset (e.g., 'BTC', 'SPY')
            timeframes: List of timeframe labels (e.g., ['1 Day', '1 Week', ...])
        
        Returns:
            dict: {
                'trend': 'bullish' | 'bearish' | 'neutral',
                'strength': 'strong' | 'moderate' | 'weak',
                'volatility': 'high' | 'medium' | 'low',
                'key_levels': [price1, price2, ...],
                'summary': "Detailed text summary",
                'recommendation': "Trading strategy suggestion",
                'risk_level': 'high' | 'medium' | 'low'
            }
        """
        
        forecast_array = np.array(forecast_prices)
        
        # Calculate overall change
        final_price = forecast_array[-1]
        total_change_pct = ((final_price - current_price) / current_price) * 100
        
        # Determine trend direction
        if total_change_pct > 2:
            trend = 'bullish'
        elif total_change_pct < -2:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Determine strength
        abs_change = abs(total_change_pct)
        if abs_change > 15:
            strength = 'strong'
        elif abs_change > 5:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        # Calculate volatility (standard deviation of daily returns)
        returns = np.diff(forecast_array) / forecast_array[:-1]
        volatility_score = np.std(returns)
        
        if volatility_score > 0.03:  # 3% daily std
            volatility = 'high'
        elif volatility_score > 0.015:  #  1.5%
            volatility = 'medium'
        else:
            volatility = 'low'
        
        # Identify key price levels (support/resistance)
        key_levels = [
            current_price,
            current_price * 1.05,  # +5% resistance
            current_price * 0.95,  # -5% support
            final_price
        ]
        key_levels = sorted(list(set([round(x, 2) for x in key_levels])))
        
        # Generate summary
        direction = "upward" if trend == 'bullish' else ("downward" if trend == 'bearish' else "sideways")
        summary = f"{asset_name} forecast shows {strength} {direction} momentum "
        summary += f"with an expected {abs(total_change_pct):.1f}% {'gain' if trend == 'bullish' else 'decline' if trend == 'bearish' else 'movement'} "
        summary += f"over the forecast period. "
        summary += f"Market volatility is assessed as {volatility}."
        
        # Risk assessment
        if volatility == 'high' or abs_change > 20:
            risk_level = 'high'
        elif volatility == 'medium' or abs_change > 10:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Generate recommendation
        if trend == 'bullish' and strength in ['strong', 'moderate']:
            recommendation = f"Consider accumulating on dips. Target: ${final_price:,.2f}. "
            recommendation += f"Set stop-loss below ${key_levels[1]:,.2f} support."
        elif trend == 'bearish' and strength in ['strong', 'moderate']:
            recommendation = f"Consider taking profits or reducing exposure. "
            recommendation += f"Watch for support at ${key_levels[1]:,.2f}."
        else:
            recommendation = f"Hold current positions. Wait for clearer directional signals. "
            recommendation += f"Range: ${key_levels[0]:,.2f} - ${key_levels[-1]:,.2f}."
        
        return {
            'trend': trend,
            'strength': strength,
            'volatility': volatility,
            'key_levels': key_levels,
            'summary': summary,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'change_pct': total_change_pct,
            'final_price': final_price
        }
    
    def compare_forecasts(self, forecasts_dict: Dict[str, List[float]], 
                         current_prices: Dict[str, float]) -> Dict:
        """
        Compare forecasts across multiple assets
        
        Args:
            forecasts_dict: {asset: [prices]}
            current_prices: {asset: current_price}
        
        Returns:
            dict: Comparative analysis
        """
        
        comparisons = {}
        
        for asset, forecast in forecasts_dict.items():
            current = current_prices.get(asset, forecast[0])
            change_pct = ((forecast[-1] - current) / current) * 100
            
            comparisons[asset] = {
                'change_pct': change_pct,
                'final_price': forecast[-1],
                'current_price': current
            }
        
        # Rank by performance
        ranked = sorted(comparisons.items(), key=lambda x: x[1]['change_pct'], reverse=True)
        
        best_performer = ranked[0][0]
        worst_performer = ranked[-1][0]
        
        summary = f"Best outlook: {best_performer} ({ranked[0][1]['change_pct']:+.1f}%). "
        summary += f"Weakest: {worst_performer} ({ranked[-1][1]['change_pct']:+.1f}%)."
        
        return {
            'ranked': ranked,
            'best': best_performer,
            'worst': worst_performer,
            'summary': summary
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*60)
    print("FORECAST ANALYZER - TESTING")
    print("="*60)
    
    analyzer = ForecastAnalyzer()
    
    # Test 1: Bullish forecast
    print("\n--- Test 1: Bullish Forecast ---")
    insights = analyzer.analyze_forecast(
        current_price=100,
        forecast_prices=[102, 105, 108, 112, 115, 118, 120],
        asset_name='SPY',
        timeframes=['1D', '1W', '2W', '1M', '3M', '6M', '1Y']
    )
    
    print(f"Trend: {insights['trend']}")
    print(f"Strength: {insights['strength']}")
    print(f"Volatility: {insights['volatility']}")
    print(f"Risk: {insights['risk_level']}")
    print(f"\nSummary: {insights['summary']}")
    print(f"\nRecommendation: {insights['recommendation']}")
    
    # Test 2: Bearish forecast
    print("\n--- Test 2: Bearish Forecast ---")
    insights2 = analyzer.analyze_forecast(
        current_price=500,
        forecast_prices=[495, 485, 470, 460, 450, 445, 440],
        asset_name='QQQ'
    )
    
    print(f"Trend: {insights2['trend']}")
    print(f"Summary: {insights2['summary']}")
    
    # Test 3: Multi-asset comparison
    print("\n--- Test 3: Multi-Asset Comparison ---")
    comparison = analyzer.compare_forecasts(
        forecasts_dict={
            'SPY': [100, 105, 110, 112, 115, 118, 120],
            'QQQ': [500, 495, 490, 485, 480,475, 470],
            'AAPL': [200, 202, 205, 208, 210, 212, 215]
        },
        current_prices={'SPY': 100, 'QQQ': 500, 'AAPL': 200}
    )
    
    print(f"\n{comparison['summary']}")
    print(f"\nRanked Performance:")
    for asset, data in comparison['ranked']:
        print(f"  {asset}: {data['change_pct']:+.1f}%")
