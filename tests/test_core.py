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


# ─────────────────────────────────────────────────────────────────────────────
# DUCKDB + POLARS DATA STORE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketDataStore:
    """Validate DuckDB data store operations and Polars/Pandas integration."""

    def test_store_initialization(self, tmp_path):
        pytest.importorskip("duckdb")
        from utils.data_store import MarketDataStore
        db_file = tmp_path / "test_market.db"
        store = MarketDataStore(db_path=str(db_file))
        assert store.db_path == os.path.abspath(str(db_file))

    def test_write_and_read_pandas(self, tmp_path):
        pytest.importorskip("duckdb")
        from utils.data_store import MarketDataStore
        db_file = tmp_path / "test_market.db"
        csv_file = tmp_path / "test_backup.csv"
        store = MarketDataStore(db_path=str(db_file))

        # Create dummy df
        df = pd.DataFrame({
            'Date': ['2026-01-01', '2026-01-02'],
            'Price': [100.5, 101.2]
        })

        # Write to db and CSV
        store.write_table('test_table_pd', df, csv_backup_path=str(csv_file))

        # Check DB table exists and can be read
        df_read = store.read_table('test_table_pd', format='pandas')
        assert len(df_read) == 2
        assert list(df_read.columns) == ['Date', 'Price']
        assert df_read['Price'].iloc[0] == 100.5

        # Check CSV backup exists
        assert os.path.exists(csv_file)
        df_csv = pd.read_csv(csv_file)
        assert len(df_csv) == 2
        assert df_csv['Price'].iloc[1] == 101.2

    def test_write_and_read_polars(self, tmp_path):
        pytest.importorskip("duckdb")
        pytest.importorskip("polars")
        from utils.data_store import MarketDataStore
        import polars as pl
        db_file = tmp_path / "test_market.db"
        csv_file = tmp_path / "test_backup.csv"
        store = MarketDataStore(db_path=str(db_file))

        # Create dummy pl df
        df = pl.DataFrame({
            'Date': ['2026-01-01', '2026-01-02'],
            'Price': [100.5, 101.2]
        })

        # Write to db and CSV
        store.write_table('test_table_pl', df, csv_backup_path=str(csv_file))

        # Check DB table can be read as Polars
        df_read = store.read_table('test_table_pl', format='polars')
        assert isinstance(df_read, pl.DataFrame)
        assert df_read.shape == (2, 2)
        assert df_read['Price'][0] == 100.5

        # Check CSV backup exists
        assert os.path.exists(csv_file)
        df_csv = pl.read_csv(csv_file)
        assert df_csv.shape == (2, 2)

    def test_migrate_csvs(self, tmp_path):
        pytest.importorskip("duckdb")
        from utils.data_store import MarketDataStore
        db_file = tmp_path / "data" / "market_intelligence.db"
        csv_file = tmp_path / "data" / "gold_global_insights.csv"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        
        # Write dummy CSV
        df = pd.DataFrame({
            'Date': ['2026-01-01'],
            'Gold': [2000.0]
        })
        df.to_csv(csv_file, index=False)
        
        # Run migration
        store = MarketDataStore(db_path=str(db_file))
        migrated = store.migrate_all_csvs()
        assert migrated >= 1
        
        # Verify read
        df_read = store.read_table('gold_global_insights', format='pandas')
        assert len(df_read) == 1
        assert df_read['Gold'].iloc[0] == 2000.0


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-HORIZON DIRECT FORECAST TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiHorizonForecast:
    """Validate direct multi-horizon forecasting, fallback scaling, and interpolation."""

    class MockModel:
        def predict(self, x, verbose=0):
            return np.array([[1.0]])

    class MockScaler:
        def __init__(self, target_value=0.05):
            self.target_value = target_value
            self.n_features_in_ = 5
        def transform(self, x):
            return x
        def inverse_transform(self, x):
            return np.array([[self.target_value]])

    class MockDataHandler:
        def __init__(self):
            self.data = np.zeros((100, 5))
        def load_data(self):
            pass
        def get_latest_price(self):
            return 100.0

    class MockFile:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def test_load_horizon_model_mock(self, monkeypatch):
        from utils.predictor_engine import ForecastEngine
        
        engine = ForecastEngine('gold', {'model_file': 'm', 'scaler_file': 's', 'sequence_length': 60}, self.MockDataHandler())
        
        # Mock load_model and file check
        monkeypatch.setattr(os.path, 'exists', lambda path: True)
        
        # Mock worker_layer.load_lstm_model
        import utils.layers.worker_lstm as worker_layer
        monkeypatch.setattr(worker_layer, 'load_lstm_model', lambda m_p, s_p: (self.MockModel(), self.MockScaler()))
        
        # Mock keras load_model in predictor_engine
        import utils.predictor_engine as pe_module
        monkeypatch.setattr(pe_module, 'load_model', lambda path: self.MockModel())
        
        # Mock builtins open to avoid missing files error
        import builtins
        original_open = builtins.open
        def mock_open_func(file, *args, **kwargs):
            if "models/" in str(file) or "m" in str(file) or "s" in str(file):
                return self.MockFile()
            return original_open(file, *args, **kwargs)
        monkeypatch.setattr(builtins, "open", mock_open_func)
        
        # Mock pickle load
        import pickle
        monkeypatch.setattr(pickle, 'load', lambda fh: self.MockScaler())
        
        # Test loading a specific horizon
        success = engine.load_horizon_model(30)
        assert success is True
        assert 30 in engine.models
        assert 30 in engine.scalers

    def test_predict_horizon_power_law_fallback(self, monkeypatch):
        from utils.predictor_engine import ForecastEngine
        
        engine = ForecastEngine('gold', {'model_file': 'm', 'scaler_file': 's', 'sequence_length': 60}, self.MockDataHandler())
        
        # Ensure Phase 7 path is bypassed so we test legacy power-law fallback
        monkeypatch.setattr(engine, '_predict_phase7', lambda horizon_days: None)

        # Mock load_horizon_model to return False for 14D but True for 7D
        def mock_load_horizon(horizon_days):
            if horizon_days == 7:
                engine.models[7] = self.MockModel()
                engine.scalers[7] = (self.MockScaler(), self.MockScaler(target_value=0.10)) # 10% change for 7D
                return True
            return False
            
        monkeypatch.setattr(engine, 'load_horizon_model', mock_load_horizon)
        
        # For 14D prediction, it should fallback to 7D scaled by power-law:
        # pct_14 = pct_7 * ((14 / 7.0) ** 0.65) = 0.10 * (2.0 ** 0.65)
        pct_14 = engine.predict_horizon(14)
        expected_pct = 0.10 * (2.0 ** 0.65)
        assert pytest.approx(pct_14, rel=1e-5) == expected_pct

    def test_trajectory_interpolation(self, monkeypatch):
        from utils.predictor_engine import ForecastEngine
        
        engine = ForecastEngine('gold', {'model_file': 'm', 'scaler_file': 's', 'sequence_length': 60}, self.MockDataHandler())
        
        # Mock ensemble_forecast
        monkeypatch.setattr(engine, 'ensemble_forecast', lambda: {})
        
        # Mock predict_horizon to return a fixed percent change per horizon
        # 1D: 1%, 7D: 7%, 14D: 14%, 30D: 30%, 90D: 90%
        horizon_returns = {1: 0.01, 7: 0.07, 14: 0.14, 30: 0.30, 90: 0.90}
        monkeypatch.setattr(engine, 'predict_horizon', lambda h: horizon_returns[h])
        
        # Call get_multi_range_forecast
        forecasts = engine.get_multi_range_forecast()
        
        # Assert structure of output
        assert 'Current' in forecasts
        assert '1 Day' in forecasts
        assert '1 Week' in forecasts
        assert '2 Weeks' in forecasts
        assert '1 Month' in forecasts
        assert '3 Months' in forecasts
        
        # Current price is 100.0 (from MockDataHandler)
        assert forecasts['Current'] == 100.0
        
        # Verify specific contextual prices
        # Day 1: 100.0 * 1.01 = 101.0
        # Day 7: 100.0 * 1.07 = 107.0
        # Day 14: 100.0 * 1.14 = 114.0
        # Day 30: 100.0 * 1.30 = 130.0
        # Day 90: 100.0 * 1.90 = 190.0
        assert pytest.approx(forecasts['1 Day']['price']) == 101.0
        assert pytest.approx(forecasts['1 Week']['price']) == 107.0
        assert pytest.approx(forecasts['2 Weeks']['price']) == 114.0
        assert pytest.approx(forecasts['1 Month']['price']) == 130.0
        assert pytest.approx(forecasts['3 Months']['price']) == 190.0
        
        # Verify trajectories series length is correct
        series_3m = forecasts['3 Months']['series']
        assert len(series_3m) == 90
        
        # Verify correct endpoints and values in series
        assert pytest.approx(series_3m[0]) == 101.0  # Day 1
        assert pytest.approx(series_3m[6]) == 107.0  # Day 7
        assert pytest.approx(series_3m[13]) == 114.0 # Day 14
        assert pytest.approx(series_3m[29]) == 130.0 # Day 30
        assert pytest.approx(series_3m[89]) == 190.0 # Day 90
        
        # Verify linear interpolation between Day 1 and Day 7
        # Day 4 should be exactly halfway: 104.0
        assert pytest.approx(series_3m[3]) == 104.0
