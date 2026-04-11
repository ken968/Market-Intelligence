import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# FRED API Key
FRED_API_KEY = "08429f7112a912c06e4e57e8f56da2c1"

# FRED Series IDs
FRED_SERIES = {
    'CPI':  'CPIAUCSL',   # Consumer Price Index (All Urban Consumers)
    'PPI':  'PPIACO',     # Producer Price Index (All Commodities)
    'PCE':  'PCEPI',      # Personal Consumption Expenditures Price Index
    'NFP':  'PAYEMS',     # Non-Farm Payrolls (Total, thousands)
    'GDP':  'GDP',        # Nominal GDP (quarterly, for Buffett Indicator)
}


def _fetch_series_with_retry(fred, series_id, start_date, max_retries=3):
    """Fetch a single FRED series with retry on server errors."""
    for attempt in range(1, max_retries + 1):
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            return s
        except Exception as e:
            err_str = str(e)
            if 'Internal Server Error' in err_str and attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} for {series_id}...")
                time.sleep(3 * attempt)
            else:
                raise


def fetch_fred_data(start_date='2009-01-01'):
    """
    Fetch FRED macro series and transform to daily time-series.

    Returns:
        daily_df   : daily DataFrame with CPI_MoM, PPI_MoM, PCE_MoM,
                     NFP_Change, MacroEvent_Flag columns
        gdp_df     : daily GDP series (for Buffett Indicator)
        calendar_df: DataFrame of exact release dates per indicator
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi package not installed. Run: pip install fredapi"
        )

    fred = Fred(api_key=FRED_API_KEY)
    print("System: Connected to FRED API.")

    # -- 1. Pull raw monthly series -------------------------------------------
    raw = {}
    release_dates = {}

    for name, series_id in FRED_SERIES.items():
        try:
            s = _fetch_series_with_retry(fred, series_id, start_date)
            raw[name] = s
            release_dates[name] = s.index.tolist()
            print(f"  [{name}] {series_id}: {len(s)} obs "
                  f"({str(s.index[0].date())} to {str(s.index[-1].date())})")
        except Exception as e:
            print(f"  Warning: Could not fetch {name} ({series_id}): {e}")
            raw[name] = None


    # -- 2. Transform to MoM % change -------------------------------------------
    monthly = pd.DataFrame()

    for name in ['CPI', 'PPI', 'PCE']:
        if raw[name] is not None:
            mom = raw[name].pct_change() * 100        # % change vs prior month
            monthly[f'{name}_MoM'] = mom
        else:
            monthly[f'{name}_MoM'] = np.nan

    # NFP: report the month-over-month absolute change (thousands of jobs)
    if raw['NFP'] is not None:
        monthly['NFP_Change'] = raw['NFP'].diff()
    else:
        monthly['NFP_Change'] = np.nan

    monthly = monthly.dropna(how='all')

    # -- 3. Build MacroEvent_Flag ------------------------------------------------
    # Flag = 1 on the day a major indicator is published, 0 otherwise.
    # We combine release dates across CPI / PPI / PCE / NFP.
    all_release_dates = set()
    for name in ['CPI', 'PPI', 'PCE', 'NFP']:
        if raw[name] is not None:
            all_release_dates.update(release_dates[name])

    # -- 4. Resample to daily (forward-fill) ------------------------------------
    daily_index = pd.date_range(
        start=monthly.index.min(),
        end=datetime.today(),
        freq='D'
    )

    daily_df = monthly.reindex(daily_index).ffill()
    daily_df.index.name = 'Date'

    # Add event flag (1 = release day, 0 = other)
    daily_df['MacroEvent_Flag'] = daily_df.index.isin(all_release_dates).astype(int)

    # Fill any remaining NaN with 0
    daily_df = daily_df.fillna(0)

    # -- 5. GDP series for Buffett Indicator ------------------------------------
    gdp_df = None
    if raw['GDP'] is not None:
        gdp_daily = raw['GDP'].reindex(daily_index).ffill()
        gdp_df = pd.DataFrame({'GDP': gdp_daily}, index=daily_index)
        gdp_df.index.name = 'Date'

    # -- 6. Build macro calendar DataFrame -------------------------------------
    calendar_records = []
    for name in ['CPI', 'PPI', 'PCE', 'NFP']:
        if raw[name] is not None:
            for dt in release_dates[name][-24:]:   # last 24 releases
                calendar_records.append({'Indicator': name, 'Release_Date': dt})

    calendar_df = pd.DataFrame(calendar_records).sort_values('Release_Date')

    return daily_df, gdp_df, calendar_df


def save_fred_data():
    """Main entry point: fetch and save to CSV files."""
    print("=" * 55)
    print(" FRED MACRO DATA SYNC")
    print("=" * 55)

    os.makedirs('data', exist_ok=True)

    try:
        daily_df, gdp_df, calendar_df = fetch_fred_data(start_date='2009-01-01')
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return False

    # Save daily FRED indicators
    daily_df.to_csv('data/fred_indicators.csv')
    print(f"\nSystem: fred_indicators.csv saved - {len(daily_df)} daily rows.")
    print(f"  Columns: {list(daily_df.columns)}")

    # Save GDP for Buffett Indicator
    if gdp_df is not None:
        gdp_df.to_csv('data/gdp_series.csv')
        print(f"System: gdp_series.csv saved - {len(gdp_df)} daily rows.")

    # Save release calendar
    if not calendar_df.empty:
        calendar_df.to_csv('data/macro_calendar.csv', index=False)
        print(f"System: macro_calendar.csv saved — {len(calendar_df)} events.")

    print("\nSystem: FRED sync complete.")
    return True


if __name__ == '__main__':
    success = save_fred_data()
    if not success:
        import sys
        sys.exit(1)
