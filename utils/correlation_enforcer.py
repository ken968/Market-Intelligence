"""
Correlation Enforcer - Post-Processing System for Multi-Asset Predictions

Enforces historical correlation relationships between assets to prevent
impossible divergences (e.g., SPY +10% while QQQ -50%).

Uses SPY as anchor and applies beta-adjusted corrections to maintain
realistic inter-asset relationships.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List


class CorrelationEnforcer:
    """
    Enforces historical correlations in multi-asset forecasts
    
    Uses beta coefficients and historical correlations to adjust
    predictions so they remain consistent with historical relationships.
    """
    
    def __init__(self, reference_ticker='SPY', lookback_days=252):
        """
        Initialize correlation enforcer
        
        Args:
            reference_ticker (str): Anchor asset (default: SPY - most stable)
            lookback_days (int): Days to calculate historical relationships (default: 252 = 1 year)
        """
        self.reference_ticker = reference_ticker
        self.lookback_days = lookback_days
        self.betas = {}
        self.correlations = {}
        
        # Calculate historical relationships
        self._calculate_historical_relationships()
    
    def _calculate_historical_relationships(self):
        """Calculate beta and correlation coefficients from historical data"""
        
        # Load reference asset data
        ref_file = f'data/{self.reference_ticker}_global_insights.csv'
        if not os.path.exists(ref_file):
            print(f"Warning: {ref_file} not found. Cannot calculate correlations.")
            return
        
        ref_df = pd.read_csv(ref_file)
        ref_df['Date'] = pd.to_datetime(ref_df['Date'])
        
        # Use last N days for calculation
        ref_df = ref_df.tail(self.lookback_days)
        ref_returns = ref_df[self.reference_ticker].pct_change().dropna()
        
        # Stock tickers to analyze
        stock_tickers = ['QQQ', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'TSM']
        
        for ticker in stock_tickers:
            try:
                # Load asset data
                asset_file = f'data/{ticker}_global_insights.csv'
                if not os.path.exists(asset_file):
                    continue
                
                asset_df = pd.read_csv(asset_file)
                asset_df['Date'] = pd.to_datetime(asset_df['Date'])
                
                # Merge on date
                merged = pd.merge(
                    ref_df[['Date', self.reference_ticker]], 
                    asset_df[['Date', ticker]], 
                    on='Date'
                )
                
                if len(merged) < 50:  # Need enough data
                    continue
                
                # Calculate returns
                merged = merged.tail(self.lookback_days)
                ref_ret = merged[self.reference_ticker].pct_change().dropna()
                asset_ret = merged[ticker].pct_change().dropna()
                
                # Calculate correlation
                correlation = ref_ret.corr(asset_ret)
                
                # Calculate beta: β = Cov(asset, ref) / Var(ref)
                covariance = np.cov(asset_ret, ref_ret)[0, 1]
                variance_ref = np.var(ref_ret)
                beta = covariance / variance_ref if variance_ref > 0 else 1.0
                
                # Store
                self.betas[ticker] = beta
                self.correlations[ticker] = correlation
                
                print(f"  {ticker:6s}: β={beta:.3f}, ρ={correlation:.3f}")
                
            except Exception as e:
                print(f"  {ticker:6s}: Error - {e}")
                continue
        
        print(f"\nCalculated relationships for {len(self.betas)} assets vs {self.reference_ticker}")
    
    def enforce_predictions(self, predictions_dict: Dict[str, List[float]], 
                          adjustment_strength: float = 0.7) -> Dict[str, List[float]]:
        """
        Enforce correlations across multi-asset predictions
        
        Args:
            predictions_dict (dict): {ticker: [p1, p2, p3, ...]} raw predictions
            adjustment_strength (float): How much to enforce (0=none, 1=full). 
                                        Default 0.7 allows some divergence.
        
        Returns:
            dict: Adjusted predictions maintaining historical relationships
        """
        
        # Validate input
        if self.reference_ticker not in predictions_dict:
            print(f"Warning: Reference ticker {self.reference_ticker} not in predictions. No adjustment applied.")
            return predictions_dict
        
        if len(self.betas) == 0:
            print("Warning: No historical relationships calculated. No adjustment applied.")
            return predictions_dict
        
        # Get reference predictions (anchor)
        ref_predictions = np.array(predictions_dict[self.reference_ticker])
        
        # Calculate reference returns
        ref_prices = ref_predictions
        ref_returns = np.diff(ref_prices) / ref_prices[:-1]
        
        adjusted_predictions = {self.reference_ticker: predictions_dict[self.reference_ticker].copy()}
        
        # Adjust each asset based on reference
        for ticker, raw_predictions in predictions_dict.items():
            if ticker == self.reference_ticker:
                continue  # Skip reference (already set)
            
            if ticker not in self.betas:
                # No historical relationship known - keep original
                adjusted_predictions[ticker] = raw_predictions.copy()
                print(f"  {ticker}: No beta data, keeping original prediction")
                continue
            
            beta = self.betas[ticker]
            correlation = self.correlations[ticker]
            
            # Convert to numpy array
            raw_predictions = np.array(raw_predictions)
            
            # Calculate raw returns
            raw_returns = np.diff(raw_predictions) / raw_predictions[:-1]
            
            # Calculate expected returns based on reference
            # Expected return = beta * reference_return
            expected_returns = beta * ref_returns
            
            # Blend raw and expected returns
            # Higher adjustment_strength = more enforcement
            blended_returns = (1 - adjustment_strength) * raw_returns + adjustment_strength * expected_returns
            
            # Reconstruct prices from blended returns
            start_price = raw_predictions[0]  # Keep first prediction as-is
            adjusted_prices = [start_price]
            
            for i, ret in enumerate(blended_returns):
                next_price = adjusted_prices[-1] * (1 + ret)
                adjusted_prices.append(next_price)
            
            adjusted_predictions[ticker] = adjusted_prices
            
            # Calculate adjustment magnitude
            original_change_pct = ((raw_predictions[-1] - raw_predictions[0]) / raw_predictions[0]) * 100
            adjusted_change_pct = ((adjusted_prices[-1] - adjusted_prices[0]) / adjusted_prices[0]) * 100
            
            print(f"  {ticker}: {original_change_pct:+.1f}% → {adjusted_change_pct:+.1f}% (β={beta:.2f})")
        
        return adjusted_predictions
    
    def validate_enforcement(self, adjusted_predictions: Dict[str, List[float]]) -> Dict:
        """
        Validate that enforcement worked correctly
        
        Args:
            adjusted_predictions: Results from enforce_predictions()
        
        Returns:
            dict: Validation metrics
        """
        
        if self.reference_ticker not in adjusted_predictions:
            return {'valid': False, 'reason': 'No reference ticker in predictions'}
        
        ref_prices = np.array(adjusted_predictions[self.reference_ticker])
        ref_returns = np.diff(ref_prices) / ref_prices[:-1]
        
        metrics = {
            'valid': True,
            'assets': {},
            'reference_ticker': self.reference_ticker,
            'reference_change_pct': ((ref_prices[-1] - ref_prices[0]) / ref_prices[0]) * 100
        }
        
        for ticker, prices in adjusted_predictions.items():
            if ticker == self.reference_ticker:
                continue
            
            prices = np.array(prices)
            asset_returns = np.diff(prices) / prices[:-1]
            
            # Calculate realized correlation in predictions
            pred_correlation = np.corrcoef(ref_returns, asset_returns)[0, 1]
            
            # Calculate change percentages
            change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
            
            # Check if signs match
            same_sign = (metrics['reference_change_pct'] * change_pct) > 0
            
            metrics['assets'][ticker] = {
                'change_pct': change_pct,
                'prediction_correlation': pred_correlation,
                'historical_correlation': self.correlations.get(ticker, None),
                'same_sign_as_reference': same_sign,
                'beta': self.betas.get(ticker, None)
            }
        
        return metrics


# ==================== TESTING / EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("="*60)
    print("CORRELATION ENFORCER - TESTING")
    print("="*60)
    
    # Initialize enforcer
    print("\nCalculating historical relationships...")
    enforcer = CorrelationEnforcer(reference_ticker='SPY', lookback_days=252)
    
    # Simulate problematic predictions (like the bug we saw)
    print("\n" + "="*60)
    print("SIMULATING DIVERGENT PREDICTIONS")
    print("="*60)
    
    test_predictions = {
        'SPY': [690, 695, 705, 715, 730, 742],  # +7.5%
        'QQQ': [597, 550, 480, 420, 350, 289],  # -51.6% (BUG!)
        'AAPL': [278, 280, 282, 283, 283, 283],  # +1.8%
    }
    
    print("\nOriginal (Buggy) Predictions:")
    for ticker, prices in test_predictions.items():
        change = ((prices[-1] - prices[0]) / prices[0]) * 100
        print(f"  {ticker}: ${prices[0]:.2f} → ${prices[-1]:.2f} ({change:+.1f}%)")
    
    # Apply enforcement
    print("\n" + "="*60)
    print("APPLYING CORRELATION ENFORCEMENT")
    print("="*60)
    
    adjusted = enforcer.enforce_predictions(test_predictions, adjustment_strength=0.7)
    
    print("\nAdjusted Predictions:")
    for ticker, prices in adjusted.items():
        change = ((prices[-1] - prices[0]) / prices[0]) * 100
        print(f"  {ticker}: ${prices[0]:.2f} → ${prices[-1]:.2f} ({change:+.1f}%)")
    
    # Validate
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    validation = enforcer.validate_enforcement(adjusted)
    
    for ticker, metrics in validation['assets'].items():
        print(f"\n{ticker}:")
        print(f"  Change: {metrics['change_pct']:+.1f}%")
        print(f"  Correlation (predicted): {metrics['prediction_correlation']:.3f}")
        print(f"  Correlation (historical): {metrics['historical_correlation']:.3f}")
        print(f"  Same sign as {validation['reference_ticker']}: {metrics['same_sign_as_reference']}")
        print(f"  Beta: {metrics['beta']:.3f}")
