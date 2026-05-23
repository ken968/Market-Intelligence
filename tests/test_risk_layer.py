"""
Unit Tests for Phase 6: Risk Layer & Dynamic Circuit Breaker
Run: pytest tests/test_risk_layer.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.signal_generator import SignalGenerator

class TestRiskLayerCalculation:
    """Test pure risk layer calculations in _calculate_signal"""

    def setup_method(self):
        # We can instantiate SignalGenerator for gold or any asset.
        # Since _calculate_signal is pure math and config lookups, we can test it directly.
        self.generator = SignalGenerator('gold')

    def test_kelly_criterion_buy_signal(self):
        """Test BUY signal Kelly sizing with valid risk-reward ratio"""
        # P = 0.6, B = 2.5
        factors = {
            'forecast': {'bullish': True, 'bearish': False, 'score': 0.8, 'weight': 0.4, 'detail': 'Forecast: +3.0%'},
            'macro': {'bullish': True, 'bearish': False, 'score': 0.7, 'weight': 0.3, 'detail': 'Macro: bullish'},
            'sentiment': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.2, 'detail': 'Sentiment: neutral'},
            'technical': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.15, 'detail': 'Technical: neutral'}
        }
        current_price = 100.0
        forecast = {
            'predicted': 110.0,
            'pct_change': 3.0,
            'direction_prob': 0.6
        }
        ood_data = {
            'active': False,
            'cbm': 0,
            'vix': 15.0,
            'gk_vol': 0.16  # GK Vol of 16%
        }
        
        # Call calculate signal
        res = self.generator._calculate_signal(factors, current_price, forecast, ood_data)
        
        # Vol calculation check:
        # vol_7d = 0.16 * np.sqrt(7 / 252) = 0.16 * 0.1666666... = 0.0266666...
        # stop_loss_pct = 1.5 * vol_7d = 1.5 * 0.0266666... = 0.04
        # stop_loss = 100.0 * (1 - 0.04) = 96.0
        # target_price = 110.0
        # risk_reward = (110 - 100) / (100 - 96) = 10 / 4 = 2.5
        # f_star = P - Q/B = 0.6 - (0.4 / 2.5) = 0.6 - 0.16 = 0.44
        # recommended_allocation = 0.44 * 0.5 = 0.22
        
        assert res['signal'] == 'BUY'
        assert res['entry_price'] == 100.0
        assert res['target_price'] == 110.0
        assert res['stop_loss'] == pytest.approx(96.0, rel=1e-5)
        assert res['risk_reward'] == pytest.approx(2.5, rel=1e-5)
        assert res['kelly_fraction'] == pytest.approx(0.44, rel=1e-5)
        assert res['recommended_allocation'] == pytest.approx(0.22, rel=1e-5)
        assert res['ood_active'] is False
        assert res['cbm'] == 0

    def test_kelly_criterion_sell_signal(self):
        """Test SELL signal Kelly sizing with valid risk-reward ratio"""
        # P = 0.65, B = 2.0
        factors = {
            'forecast': {'bullish': False, 'bearish': True, 'score': 0.8, 'weight': 0.4, 'detail': 'Forecast: -3.0%'},
            'macro': {'bullish': False, 'bearish': True, 'score': 0.7, 'weight': 0.3, 'detail': 'Macro: bearish'},
            'sentiment': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.2, 'detail': 'Sentiment: neutral'},
            'technical': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.15, 'detail': 'Technical: neutral'}
        }
        current_price = 100.0
        forecast = {
            'predicted': 92.0,
            'pct_change': -3.0,
            'direction_prob': 0.65
        }
        ood_data = {
            'active': False,
            'cbm': 0,
            'vix': 15.0,
            'gk_vol': 0.16  # GK Vol of 16%
        }
        
        # Call calculate signal
        res = self.generator._calculate_signal(factors, current_price, forecast, ood_data)
        
        # Vol calculation check:
        # vol_7d = 0.16 * np.sqrt(7 / 252) = 0.0266666...
        # stop_loss_pct = 1.5 * vol_7d = 0.04
        # stop_loss = 100.0 * (1 + 0.04) = 104.0
        # target_price = 92.0
        # risk_reward = |92 - 100| / |104 - 100| = 8 / 4 = 2.0
        # f_star = P - Q/B = 0.65 - (0.35 / 2.0) = 0.65 - 0.175 = 0.475
        # recommended_allocation = 0.475 * 0.5 = 0.2375
        
        assert res['signal'] == 'SELL'
        assert res['entry_price'] == 100.0
        assert res['target_price'] == 92.0
        assert res['stop_loss'] == pytest.approx(104.0, rel=1e-5)
        assert res['risk_reward'] == pytest.approx(2.0, rel=1e-5)
        assert res['kelly_fraction'] == pytest.approx(0.475, rel=1e-5)
        assert res['recommended_allocation'] == pytest.approx(0.2375, rel=1e-5)
        assert res['ood_active'] is False

    def test_kelly_criterion_hold_signal(self):
        """Test HOLD signal logic: Kelly sizing must be 0.0"""
        factors = {
            'forecast': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.4, 'detail': 'Forecast: neutral'},
            'macro': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.3, 'detail': 'Macro: neutral'},
            'sentiment': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.2, 'detail': 'Sentiment: neutral'},
            'technical': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.15, 'detail': 'Technical: neutral'}
        }
        current_price = 100.0
        forecast = {
            'predicted': 100.0,
            'pct_change': 0.0,
            'direction_prob': 0.5
        }
        ood_data = {
            'active': False,
            'cbm': 0,
            'vix': 15.0,
            'gk_vol': 0.16
        }
        
        res = self.generator._calculate_signal(factors, current_price, forecast, ood_data)
        
        assert res['signal'] == 'HOLD'
        assert res['kelly_fraction'] == 0.0
        assert res['recommended_allocation'] == 0.0
        assert res['risk_reward'] == 0.0
        assert res['ood_active'] is False

    def test_ood_circuit_breaker_active(self):
        """Test that OOD Circuit Breaker forces HOLD and zero Kelly allocation"""
        factors = {
            'forecast': {'bullish': True, 'bearish': False, 'score': 0.8, 'weight': 0.4, 'detail': 'Forecast: +5.0%'},
            'macro': {'bullish': True, 'bearish': False, 'score': 0.8, 'weight': 0.3, 'detail': 'Macro: bullish'},
            'sentiment': {'bullish': True, 'bearish': False, 'score': 0.8, 'weight': 0.2, 'detail': 'Sentiment: bullish'},
            'technical': {'bullish': True, 'bearish': False, 'score': 0.8, 'weight': 0.15, 'detail': 'Technical: bullish'}
        }
        current_price = 100.0
        forecast = {
            'predicted': 105.0,
            'pct_change': 5.0,
            'direction_prob': 0.8
        }
        # Even with extremely bullish scores, active OOD forces HOLD
        ood_data = {
            'active': True,
            'cbm': 3,
            'vix': 36.0,
            'gk_vol': 0.16
        }
        
        res = self.generator._calculate_signal(factors, current_price, forecast, ood_data)
        
        assert res['signal'] == 'HOLD'
        assert res['kelly_fraction'] == 0.0
        assert res['recommended_allocation'] == 0.0
        assert res['risk_reward'] == 0.0
        assert res['ood_active'] is True
        assert res['cbm'] == 3
        assert any("OOD CIRCUIT BREAKER ACTIVE" in reason for reason in res['reasons'])

    def test_negative_kelly_clamped_to_zero(self):
        """Test that negative Kelly values are clamped to 0.0"""
        # P = 0.3 (low winning probability), B = 1.0 (low risk reward)
        # f* = 0.3 - 0.7/1.0 = -0.4 -> clamped to 0.0
        factors = {
            'forecast': {'bullish': True, 'bearish': False, 'score': 0.8, 'weight': 0.4, 'detail': 'Forecast: +1.0%'},
            'macro': {'bullish': True, 'bearish': False, 'score': 0.8, 'weight': 0.3, 'detail': 'Macro: bullish'},
            'sentiment': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.2, 'detail': 'Sentiment: neutral'},
            'technical': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.15, 'detail': 'Technical: neutral'}
        }
        current_price = 100.0
        forecast = {
            'predicted': 101.0,
            'pct_change': 1.0,
            'direction_prob': 0.3
        }
        ood_data = {
            'active': False,
            'cbm': 0,
            'vix': 15.0,
            'gk_vol': 0.16
        }
        
        res = self.generator._calculate_signal(factors, current_price, forecast, ood_data)
        
        assert res['signal'] == 'BUY'
        assert res['kelly_fraction'] == 0.0
        assert res['recommended_allocation'] == 0.0

    def test_kelly_sizing_clamping(self):
        """Test that Risk/Reward (B) is clamped to min 1.0 to prevent irrational Kelly"""
        # P = 0.6 (60% win prob)
        factors = {
            'forecast': {'bullish': True, 'bearish': False, 'score': 0.8, 'weight': 0.4, 'detail': 'Forecast: +3.0%'},
            'macro': {'bullish': True, 'bearish': False, 'score': 0.7, 'weight': 0.3, 'detail': 'Macro: bullish'},
            'sentiment': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.2, 'detail': 'Sentiment: neutral'},
            'technical': {'bullish': False, 'bearish': False, 'score': 0.5, 'weight': 0.15, 'detail': 'Technical: neutral'}
        }
        current_price = 100.0
        forecast = {
            'predicted': 100.5,  # Target is only 0.5 away
            'pct_change': 0.5,
            'direction_prob': 0.6
        }
        ood_data = {
            'active': False,
            'cbm': 0,
            'vix': 15.0,
            'gk_vol': 0.06  # GK Vol of 6% -> vol_7d = 0.01, stop loss = 100 * (1 - 1.5*0.01) = 98.5
            # Reward = 0.5, Risk = 1.5. B = 0.333...
        }
        
        res = self.generator._calculate_signal(factors, current_price, forecast, ood_data)
        
        # Since B is clamped to 1.0: f* = 0.6 - (0.4 / 1.0) = 0.2
        assert res['signal'] == 'BUY'
        assert res['kelly_fraction'] == pytest.approx(0.2, rel=1e-5)
        assert res['recommended_allocation'] == pytest.approx(0.1, rel=1e-5)


class TestSignalGeneratorIntegration:
    """Test full generate_signal flow by mocking data reads and external dependencies"""

    @patch('utils.xai_explainer.get_top_macro_drivers')
    @patch('utils.data_store.MarketDataStore.read_table')
    @patch('utils.signal_generator.AssetPredictor.predict_week')
    def test_generate_signal_ood_by_vix(self, mock_predict_week, mock_read_table, mock_get_top_macro_drivers):
        """Test generate_signal triggers OOD when VIX is extremely high (> 35)"""
        # Mock macro drivers
        mock_get_top_macro_drivers.return_value = [
            {'feature': 'CPI_MoM', 'z_score': 0.5},
            {'feature': 'YieldCurve_10Y2Y', 'z_score': -1.2}
        ]
        
        # Mock latest data row
        mock_df = pd.DataFrame([{
            'Gold': 2000.0,
            'VIX': 38.0,  # VIX > 35 triggers OOD
            'GK_Vol_21d': 0.12,
            'EMA_90': 1980.0
        }])
        mock_read_table.return_value = mock_df
        
        # Mock forecast
        mock_predict_week.return_value = {
            'current': 2000.0,
            'predicted': 2050.0,
            'change': 50.0,
            'pct_change': 2.5,
            'direction_prob': 0.65
        }
        
        generator = SignalGenerator('gold')
        res = generator.generate_signal()
        
        assert res['ood_active'] is True
        assert res['signal'] == 'HOLD'
        assert res['kelly_fraction'] == 0.0
        assert res['recommended_allocation'] == 0.0
        assert any("OOD CIRCUIT BREAKER ACTIVE" in reason for reason in res['reasons'])

    @patch('utils.xai_explainer.get_top_macro_drivers')
    @patch('utils.data_store.MarketDataStore.read_table')
    @patch('utils.signal_generator.AssetPredictor.predict_week')
    def test_generate_signal_data_integrity_error(self, mock_predict_week, mock_read_table, mock_get_top_macro_drivers):
        """Test generate_signal triggers Data Integrity Error when NaN is present"""
        mock_get_top_macro_drivers.return_value = []
        
        # Mock latest data row with NaN
        mock_df = pd.DataFrame([{
            'Gold': 2000.0,
            'VIX': np.nan,  # NaN value to trigger Data Integrity Error
            'GK_Vol_21d': 0.12,
            'EMA_90': 1980.0
        }])
        mock_read_table.return_value = mock_df
        
        # Mock forecast
        mock_predict_week.return_value = {
            'current': 2000.0,
            'predicted': 2050.0,
            'change': 50.0,
            'pct_change': 2.5,
            'direction_prob': 0.65
        }
        
        generator = SignalGenerator('gold')
        res = generator.generate_signal()
        
        assert res['signal'] == 'HOLD'
        assert res['kelly_fraction'] == 0.0
        assert any("DATA INTEGRITY ERROR" in reason for reason in res['reasons'])

    @patch('utils.xai_explainer.get_top_macro_drivers')
    @patch('utils.data_store.MarketDataStore.read_table')
    @patch('utils.signal_generator.AssetPredictor.predict_week')
    def test_generate_signal_ood_by_cbm(self, mock_predict_week, mock_read_table, mock_get_top_macro_drivers):
        """Test generate_signal triggers OOD when CBM >= 3 (3 or more macro variables deviate by > 3 sigma)"""
        # Mock macro drivers: 3 features with z-score >= 3.0
        mock_get_top_macro_drivers.return_value = [
            {'feature': 'CPI_MoM', 'z_score': 3.2},
            {'feature': 'YieldCurve_10Y2Y', 'z_score': -3.1},
            {'feature': 'Credit_Spread', 'z_score': 3.5},
            {'feature': 'DXY', 'z_score': 0.8}
        ]
        
        # Mock latest data row
        mock_df = pd.DataFrame([{
            'Gold': 2000.0,
            'VIX': 15.0,  # VIX is normal
            'GK_Vol_21d': 0.12,
            'EMA_90': 1980.0
        }])
        mock_read_table.return_value = mock_df
        
        # Mock forecast
        mock_predict_week.return_value = {
            'current': 2000.0,
            'predicted': 2050.0,
            'change': 50.0,
            'pct_change': 2.5,
            'direction_prob': 0.65
        }
        
        generator = SignalGenerator('gold')
        res = generator.generate_signal()
        
        assert res['ood_active'] is True
        assert res['cbm'] == 3
        assert res['signal'] == 'HOLD'
        assert any("OOD CIRCUIT BREAKER ACTIVE" in reason for reason in res['reasons'])

    @patch('utils.xai_explainer.get_top_macro_drivers')
    @patch('utils.data_store.MarketDataStore.read_table')
    @patch('utils.signal_generator.AssetPredictor.predict_week')
    def test_generate_signal_ood_by_extreme_btc_gk_vol(self, mock_predict_week, mock_read_table, mock_get_top_macro_drivers):
        """Test generate_signal triggers OOD when BTC Garman-Klass Volatility is extreme (> 80%)"""
        mock_get_top_macro_drivers.return_value = []
        
        # Mock latest data row for BTC
        mock_df = pd.DataFrame([{
            'BTC': 80000.0,
            'VIX': 15.0,
            'GK_Vol_21d': 0.85,  # BTC Vol > 80% triggers OOD
            'EMA_90': 75000.0
        }])
        mock_read_table.return_value = mock_df
        
        # Mock forecast
        mock_predict_week.return_value = {
            'current': 80000.0,
            'predicted': 85000.0,
            'change': 5000.0,
            'pct_change': 6.25,
            'direction_prob': 0.70
        }
        
        generator = SignalGenerator('btc')
        res = generator.generate_signal()
        
        assert res['ood_active'] is True
        assert res['signal'] == 'HOLD'
        assert res['kelly_fraction'] == 0.0

    @patch('utils.xai_explainer.get_top_macro_drivers')
    @patch('utils.data_store.MarketDataStore.read_table')
    @patch('utils.signal_generator.AssetPredictor.predict_week')
    def test_generate_signal_ood_by_extreme_gold_gk_vol(self, mock_predict_week, mock_read_table, mock_get_top_macro_drivers):
        """Test generate_signal triggers OOD when Gold Garman-Klass Volatility is extreme (> 30%)"""
        mock_get_top_macro_drivers.return_value = []
        
        # Mock latest data row for Gold
        mock_df = pd.DataFrame([{
            'Gold': 2000.0,
            'VIX': 15.0,
            'GK_Vol_21d': 0.32,  # Gold Vol > 30% triggers OOD
            'EMA_90': 1980.0
        }])
        mock_read_table.return_value = mock_df
        
        # Mock forecast
        mock_predict_week.return_value = {
            'current': 2000.0,
            'predicted': 2050.0,
            'change': 50.0,
            'pct_change': 2.5,
            'direction_prob': 0.65
        }
        
        generator = SignalGenerator('gold')
        res = generator.generate_signal()
        
        assert res['ood_active'] is True
        assert res['signal'] == 'HOLD'
        assert res['kelly_fraction'] == 0.0

    @patch('utils.xai_explainer.get_top_macro_drivers')
    @patch('utils.data_store.MarketDataStore.read_table')
    @patch('utils.signal_generator.AssetPredictor.predict_week')
    def test_generate_signal_ood_by_extreme_stock_gk_vol(self, mock_predict_week, mock_read_table, mock_get_top_macro_drivers):
        """Test generate_signal triggers OOD when Stock Garman-Klass Volatility is extreme (> 40%)"""
        mock_get_top_macro_drivers.return_value = []
        
        # Mock latest data row for SPY (stocks)
        mock_df = pd.DataFrame([{
            'SPY': 500.0,
            'VIX': 15.0,
            'GK_Vol_21d': 0.42,  # Stock Vol > 40% triggers OOD
            'EMA_90': 490.0
        }])
        mock_read_table.return_value = mock_df
        
        # Mock forecast
        mock_predict_week.return_value = {
            'current': 500.0,
            'predicted': 520.0,
            'change': 20.0,
            'pct_change': 4.0,
            'direction_prob': 0.68
        }
        
        generator = SignalGenerator('spy')
        res = generator.generate_signal()
        
        assert res['ood_active'] is True
        assert res['signal'] == 'HOLD'
        assert res['kelly_fraction'] == 0.0
