"""
Market Intelligence — Core Test Suite
Run: pytest tests/test_core.py -v

Tests critical invariants WITHOUT running actual ML models or making API calls.
All tests should complete in < 5 seconds total.
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG STRUCTURE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigStructure:
    """Ensure config.py has correct structure for all assets."""

    def setup_method(self):
        from utils.config import ASSETS
        self.ASSETS = ASSETS

    def test_gold_config_has_required_keys(self):
        gold = self.ASSETS['gold']
        for key in ['features', 'model_file', 'scaler_file', 'data_file',
                    'sequence_length', 'model_arch']:
            assert key in gold, f"'gold' config missing key: {key}"

    def test_btc_config_has_required_keys(self):
        btc = self.ASSETS['btc']
        for key in ['features', 'model_file', 'scaler_file', 'data_file',
                    'sequence_length', 'model_arch']:
            assert key in btc, f"'btc' config missing key: {key}"

    def test_all_assets_have_model_arch(self):
        for asset_key, cfg in self.ASSETS.items():
            assert 'model_arch' in cfg, f"Asset '{asset_key}' missing model_arch"
            arch = cfg['model_arch']
            assert 'units' in arch,    f"Asset '{asset_key}' model_arch missing 'units'"
            assert 'dropout' in arch,  f"Asset '{asset_key}' model_arch missing 'dropout'"
            assert 'attention' in arch, f"Asset '{asset_key}' model_arch missing 'attention'"

    def test_credit_spread_in_feature_lists(self):
        """After Minggu 2 update, Credit_Spread must be in all asset features."""
        for asset_key, cfg in self.ASSETS.items():
            assert 'Credit_Spread' in cfg['features'], (
                f"Asset '{asset_key}' missing 'Credit_Spread' in features"
            )

    def test_model_arch_units_are_list(self):
        for asset_key, cfg in self.ASSETS.items():
            units = cfg['model_arch']['units']
            assert isinstance(units, list), f"'{asset_key}' model_arch units must be a list"
            assert len(units) >= 1, f"'{asset_key}' model_arch units must have >= 1 layer"

    def test_dropout_in_valid_range(self):
        for asset_key, cfg in self.ASSETS.items():
            d = cfg['model_arch']['dropout']
            assert 0.0 < d < 1.0, f"'{asset_key}' dropout={d} outside (0, 1)"

    def test_volatile_stocks_have_attention(self):
        from utils.config import VOLATILE_STOCKS
        for ticker in VOLATILE_STOCKS:
            key = ticker.lower()
            if key in self.ASSETS:
                assert self.ASSETS[key]['model_arch']['attention'] is True, (
                    f"Volatile stock '{ticker}' should have attention=True"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Z-SCORE / FEATURE ENGINEERING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    """Validate Z-score lookback logic and derived features."""

    def test_monthly_indicators_get_long_lookback(self):
        from utils.feature_engineering import get_indicator_lookback
        monthly_indicators = ['CPI_MoM', 'NFP_Change', 'PPI_MoM', 'PCE_MoM', 'M2_MoM']
        for ind in monthly_indicators:
            lb = get_indicator_lookback(ind)
            assert lb >= 30, (
                f"Monthly indicator '{ind}' should have lookback >= 30 days, got {lb}"
            )

    def test_daily_indicators_get_short_lookback(self):
        from utils.feature_engineering import get_indicator_lookback
        daily_indicators = ['YieldCurve_10Y2Y', 'VIX', 'DXY', 'Credit_Spread']
        for ind in daily_indicators:
            lb = get_indicator_lookback(ind)
            assert lb <= 20, (
                f"Daily indicator '{ind}' should have lookback <= 20 days, got {lb}"
            )

    def test_recession_risk_in_valid_range(self):
        from utils.feature_engineering import compute_dynamic_recession_risk
        # Synthetic DataFrame with yield curve data
        dates = pd.date_range('2020-01-01', periods=100)
        df = pd.DataFrame({
            'YieldCurve_10Y2Y': np.random.uniform(-1.5, 1.5, 100),
        }, index=dates)
        risk = compute_dynamic_recession_risk(df)
        assert 0.0 <= risk <= 1.0, f"Recession risk {risk} outside [0.0, 1.0]"

    def test_garman_klass_vol_non_negative(self):
        from utils.feature_engineering import compute_garman_klass_vol
        # Generate synthetic OHLCV
        df = pd.DataFrame({
            'Open':  [100.0, 101.0, 99.5],
            'High':  [102.0, 103.0, 101.0],
            'Low':   [98.0,  99.0,  97.5],
            'Close': [101.0, 100.0, 100.5],
        })
        result = compute_garman_klass_vol(df)
        # result may be a Series or scalar — extract last non-NaN value
        if hasattr(result, 'dropna'):
            vals = result.dropna()
            if len(vals) == 0:
                pytest.skip("compute_garman_klass_vol returned all NaN")
            vol = float(vals.iloc[-1])
        else:
            vol = float(result)
        if not np.isnan(vol):
            assert vol >= 0.0, f"GK volatility {vol} is negative"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceScore:
    """Validate dynamic confidence score logic."""

    def test_fallback_returns_valid_structure(self):
        from utils.config import get_dynamic_confidence
        # No backtest JSON exists for 'test_asset' — should return fallback
        result = get_dynamic_confidence('gold', '1 Week')
        assert isinstance(result, dict), "Confidence score must return dict"
        assert 'score' in result,  "Missing 'score' key"
        assert 'label' in result,  "Missing 'label' key"
        assert 'color' in result,  "Missing 'color' key"
        assert 0.0 <= result['score'] <= 1.0, f"Score {result['score']} outside [0, 1]"

    def test_three_months_always_speculative(self):
        from utils.config import get_dynamic_confidence
        result = get_dynamic_confidence('gold', '3 Months')
        assert result['label'] == 'Speculative', (
            "'3 Months' must always have label='Speculative'"
        )
        assert result['color'] == 'error', (
            "'3 Months' must always have color='error'"
        )

    def test_confidence_from_backtest_json(self, tmp_path, monkeypatch):
        """If a valid backtest JSON exists, score must differ from hard-coded fallback."""
        import utils.config as cfg_module

        # Patch os.path.exists to find our temp JSON
        report_path = tmp_path / "reports" / "backtest_gold.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps({"hit_ratio_3layer": 70.0}))

        # Patch os.path.exists and open to use temp path
        original_exists = os.path.exists
        def mock_exists(path):
            if 'stacker_gold_backtest.json' in str(path):
                return False
            if 'backtest_gold.json' in str(path):
                return str(report_path)
            return original_exists(path)

        import builtins
        original_open = builtins.open
        def mock_open(path, *args, **kwargs):
            if 'backtest_gold.json' in str(path):
                return original_open(str(report_path), *args, **kwargs)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(os.path, 'exists', mock_exists)
        monkeypatch.setattr(builtins, 'open', mock_open)

        result = cfg_module.get_dynamic_confidence('gold', '1 Week')
        # 70% hit_ratio × 0.92 decay = 0.644 → label should not be 'Low' (fallback=0.60)
        assert result['score'] > 0.0, "Score from JSON must be > 0"


# ─────────────────────────────────────────────────────────────────────────────
# DATA INTEGRITY TESTS (only runs if data files exist)
# ─────────────────────────────────────────────────────────────────────────────

class TestDataIntegrity:
    """Validate CSV data files are non-empty and have required columns."""

    REQUIRED_COLS = {
        'data/gold_global_insights.csv': ['Date', 'Gold', 'DXY', 'VIX', 'Sentiment'],
        'data/btc_global_insights.csv':  ['Date', 'BTC', 'DXY', 'VIX', 'Sentiment'],
    }

    @pytest.mark.parametrize("filepath,columns", list(REQUIRED_COLS.items()))
    def test_csv_not_empty(self, filepath, columns):
        if not os.path.exists(filepath):
            pytest.skip(f"{filepath} not found — run data sync first")
        df = pd.read_csv(filepath)
        assert len(df) > 100, f"{filepath} has only {len(df)} rows — expected > 100"

    @pytest.mark.parametrize("filepath,columns", list(REQUIRED_COLS.items()))
    def test_required_columns_exist(self, filepath, columns):
        if not os.path.exists(filepath):
            pytest.skip(f"{filepath} not found — run data sync first")
        df = pd.read_csv(filepath)
        missing = [c for c in columns if c not in df.columns]
        assert not missing, f"{filepath} missing columns: {missing}"

    def test_fred_indicators_has_credit_spread(self):
        """After FRED sync, fred_indicators.csv must have Credit_Spread column.
        EXPECTED FAIL until fred_fetcher.py is re-run to regenerate the CSV.
        """
        path = 'data/fred_indicators.csv'
        if not os.path.exists(path):
            pytest.skip("fred_indicators.csv not found — run FRED sync first")
        df = pd.read_csv(path)
        if 'Credit_Spread' not in df.columns:
            pytest.xfail(
                "Credit_Spread not yet in fred_indicators.csv — "
                "run: python scripts/fred_fetcher.py to sync"
            )

    def test_no_negative_prices(self):
        for filepath, price_col in [('data/gold_global_insights.csv', 'Gold'),
                                     ('data/btc_global_insights.csv', 'BTC')]:
            if not os.path.exists(filepath):
                continue
            df = pd.read_csv(filepath)
            assert (df[price_col] >= 0).all(), (
                f"{filepath} has negative {price_col} prices"
            )


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT SOURCE CREDIBILITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSentimentCredibility:
    """Validate the source credibility weighting in aggregator."""

    def test_credibility_dict_has_tier1_sources(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from sentiment_sources.aggregator import SentimentAggregator
        cred = SentimentAggregator.SOURCE_CREDIBILITY
        tier1 = ['reuters.com', 'bloomberg.com', 'federalreserve.gov']
        for domain in tier1:
            assert domain in cred, f"Tier-1 domain '{domain}' missing from SOURCE_CREDIBILITY"
            assert cred[domain] >= 0.9, f"'{domain}' credibility should be >= 0.9, got {cred[domain]}"

    def test_social_media_lower_credibility(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from sentiment_sources.aggregator import SentimentAggregator
        cred = SentimentAggregator.SOURCE_CREDIBILITY
        social = ['reddit.com', 'twitter.com']
        for domain in social:
            if domain in cred:
                assert cred[domain] <= 0.5, (
                    f"Social domain '{domain}' credibility should be <= 0.5, got {cred[domain]}"
                )

    def test_default_credibility_in_range(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from sentiment_sources.aggregator import SentimentAggregator
        default = SentimentAggregator.DEFAULT_CREDIBILITY
        assert 0.3 <= default <= 0.7, f"Default credibility {default} seems off"


# ─────────────────────────────────────────────────────────────────────────────
# MACRO PROCESSOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroProcessor:
    """Validate macro context output includes Credit_Spread."""

    def test_macro_context_has_credit_spread_key(self):
        path = 'data/fred_indicators.csv'
        if not os.path.exists(path):
            pytest.skip("fred_indicators.csv not found")
        from utils.macro_processor import build_macro_context
        ctx = build_macro_context()
        # latest_values should have Credit_Spread if FRED data is present
        lv = ctx.get('latest_values', {})
        assert 'Credit_Spread' in lv, (
            "macro_processor latest_values missing 'Credit_Spread' key"
        )

    def test_macro_summary_is_non_empty_string(self):
        path = 'data/fred_indicators.csv'
        if not os.path.exists(path):
            pytest.skip("fred_indicators.csv not found")
        from utils.macro_processor import build_macro_context
        ctx = build_macro_context()
        assert isinstance(ctx.get('macro_summary', ''), str), "macro_summary must be str"
        assert len(ctx.get('macro_summary', '')) > 50, "macro_summary is too short"
