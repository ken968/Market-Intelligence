"""
FRED Macro Indicators Fetcher
Uses direct FRED REST API via requests (no fredapi package needed).
Fetches: CPI, PPI, PCE, NFP, GDP, Yield Curve (10Y-2Y), M2 Money Supply
"""

import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# API KEY ROTATION: tries keys in order until one succeeds
# ---------------------------------------------------------------------------
FRED_API_KEYS = [
    "08429f7112a912c06e4e57e8f56da2c1",
    "78880e0a8138a8fdc568ddc64a4abbac",
    "88775e668ea046de41b1e288c7ce6d13",
]

# ---------------------------------------------------------------------------
# FRED Series to fetch
# ---------------------------------------------------------------------------
FRED_SERIES = {
    'CPI':         'CPIAUCSL',   # CPI All Urban (monthly)
    'PPI':         'PPIACO',     # PPI All Commodities (monthly)
    'PCE':         'PCEPI',      # PCE Price Index (monthly)
    'NFP':         'PAYEMS',     # Non-Farm Payrolls (monthly)
    'GDP':         'GDP',        # Nominal GDP (quarterly)
    'YieldCurve':  'T10Y2Y',     # 10Y minus 2Y spread (daily)
    'M2':          'M2SL',       # M2 Money Supply (monthly)
}

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_series(series_id, start_date='2009-01-01', max_retries=3):
    """
    Fetch a FRED series using REST API directly (no fredapi package needed).
    Tries each API key in rotation until one works.
    """
    last_error = None
    for api_key in FRED_API_KEYS:
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(FRED_BASE, params={
                    'series_id':          series_id,
                    'api_key':            api_key,
                    'file_type':          'json',
                    'observation_start':  start_date,
                    'output_type':        1,   # observations
                }, timeout=20)
                resp.raise_for_status()
                data = resp.json()

                if 'observations' not in data:
                    raise ValueError(f"No observations in response: {data}")

                records = {
                    obs['date']: float(obs['value'])
                    for obs in data['observations']
                    if obs['value'] != '.'
                }
                s = pd.Series(records)
                s.index = pd.to_datetime(s.index)
                s.name = series_id
                return s

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"    Retry {attempt}/{max_retries} for {series_id} (key ...{api_key[-4:]})...")
                    time.sleep(2 * attempt)
                continue

    raise RuntimeError(f"All API keys failed for {series_id}. Last error: {last_error}")


def fetch_fred_data(start_date='2009-01-01'):
    """
    Fetch all FRED series and return:
      daily_df    : daily DataFrame (forward-filled from monthly/quarterly)
      gdp_df      : daily GDP
      calendar_df : last 24 release dates per indicator
    """
    print("System: Fetching from FRED REST API (no fredapi package needed)...")
    raw = {}
    release_dates = {}

    for name, series_id in FRED_SERIES.items():
        try:
            s = _fetch_series(series_id, start_date)
            raw[name] = s
            release_dates[name] = s.index.tolist()
            print(f"  [{name}] {series_id}: {len(s)} obs "
                  f"({str(s.index[0].date())} to {str(s.index[-1].date())})")
        except Exception as e:
            print(f"  Warning: Could not fetch {name} ({series_id}): {e}")
            raw[name] = None

    # -- Transform monthly to MoM % change ----------------------------------
    monthly = pd.DataFrame()
    for name in ['CPI', 'PPI', 'PCE']:
        if raw[name] is not None:
            monthly[f'{name}_MoM'] = raw[name].pct_change() * 100
        else:
            monthly[f'{name}_MoM'] = np.nan

    if raw['NFP'] is not None:
        monthly['NFP_Change'] = raw['NFP'].diff()
    else:
        monthly['NFP_Change'] = np.nan

    if raw['M2'] is not None:
        monthly['M2_MoM'] = raw['M2'].pct_change() * 100
    else:
        monthly['M2_MoM'] = np.nan

    monthly.dropna(how='all', inplace=True)

    # -- Build MacroEvent_Flag -----------------------------------------------
    all_release_dates = set()
    for name in ['CPI', 'PPI', 'PCE', 'NFP']:
        if raw[name] is not None:
            all_release_dates.update(release_dates[name])

    # -- Resample to daily using daily_index ---------------------------------
    # Use the earliest date we have across all series
    min_dates = [s.index.min() for s in raw.values() if s is not None]
    start_dt = min(min_dates) if min_dates else pd.Timestamp(start_date)

    daily_index = pd.date_range(start=start_dt, end=datetime.today(), freq='D')

    daily_df = monthly.reindex(daily_index).ffill()
    daily_df.index.name = 'Date'

    # Add Yield Curve (T10Y2Y) — already daily, just reindex
    if raw['YieldCurve'] is not None:
        yc_daily = raw['YieldCurve'].reindex(daily_index).ffill()
        daily_df['YieldCurve_10Y2Y'] = yc_daily

    # MacroEvent_Flag
    daily_df['MacroEvent_Flag'] = daily_df.index.isin(all_release_dates).astype(int)
    daily_df.fillna(0, inplace=True)

    # -- GDP for Buffett Indicator -------------------------------------------
    gdp_df = None
    if raw['GDP'] is not None:
        gdp_daily = raw['GDP'].reindex(daily_index).ffill()
        gdp_df = pd.DataFrame({'GDP': gdp_daily}, index=daily_index)
        gdp_df.index.name = 'Date'

    # -- Release calendar ----------------------------------------------------
    calendar_records = []
    for name in ['CPI', 'PPI', 'PCE', 'NFP']:
        if raw[name] is not None:
            for dt in release_dates[name][-24:]:
                calendar_records.append({'Indicator': name, 'Release_Date': dt})

    calendar_df = pd.DataFrame(calendar_records).sort_values('Release_Date')

    return daily_df, gdp_df, calendar_df


def save_fred_data():
    """Main entry: fetch and save all FRED data to CSVs."""
    print("=" * 55)
    print(" FRED MACRO DATA SYNC")
    print("=" * 55)

    os.makedirs('data', exist_ok=True)

    try:
        daily_df, gdp_df, calendar_df = fetch_fred_data('2009-01-01')
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return False

    daily_df.to_csv('data/fred_indicators.csv')
    print(f"\nSystem: fred_indicators.csv saved - {len(daily_df)} daily rows.")
    print(f"  Columns: {list(daily_df.columns)}")

    if gdp_df is not None:
        gdp_df.to_csv('data/gdp_series.csv')
        print(f"System: gdp_series.csv saved - {len(gdp_df)} daily rows.")

    if not calendar_df.empty:
        calendar_df.to_csv('data/macro_calendar.csv', index=False)
        print(f"System: macro_calendar.csv saved - {len(calendar_df)} events.")

    print("\nSystem: FRED sync complete.")
    return True


if __name__ == '__main__':
    import sys
    success = save_fred_data()
    if not success:
        sys.exit(1)
