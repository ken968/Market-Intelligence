"""
US Stocks Analysis Page
Multi-stock tracking for indices, Mag7, and semiconductors
"""

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from utils.config import ASSETS, STOCK_TICKERS, get_all_stock_tickers
from utils.ui_components import (
    inject_custom_css, render_page_header, render_metric_card,
    render_news_section, create_price_chart,
    show_loading_message, show_error_message, render_prediction_table
)
from utils.predictor import AssetPredictor, batch_predict_tomorrow, batch_multi_range_forecast

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Stocks Analysis | Market Intelligence",
    layout="wide"
)

inject_custom_css()

# ==================== MAIN CONTENT ====================

render_page_header(
    icon="",
    title="US Equities Intelligence",
    subtitle="Comprehensive analysis of market indices, Magnificent 7, and semiconductor leaders"
)

# ==================== STOCK SELECTOR ====================

all_tickers = get_all_stock_tickers()
available_stocks = [t for t in all_tickers if os.path.exists(ASSETS[t.lower()]['data_file'])]

if not available_stocks:
    show_error_message("No stock data available. Please sync data from Settings page.")
    st.stop()

st.markdown("###  Stock Selection")

# Group stocks by category
indices = [t for t in available_stocks if t in ['SPY', 'QQQ', 'DIA']]
mag7 = [t for t in available_stocks if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']]
semi = [t for t in available_stocks if t == 'TSM']

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("####  Market Indices")
    selected_indices = st.multiselect("Select indices", indices, default=indices)

with col2:
    st.markdown("####  Magnificent 7")
    selected_mag7 = st.multiselect("Select Mag7", mag7, default=mag7)

with col3:
    st.markdown("####  Semiconductors")
    selected_semi = st.multiselect("Select semis", semi, default=semi)

selected_stocks = selected_indices + selected_mag7 + selected_semi

if not selected_stocks:
    st.warning(" Please select at least one stock to analyze.")
    st.stop()

st.markdown("---")

# ==================== MARKET OVERVIEW ====================

st.markdown("###  Current Market Snapshot")

# Load latest data for all selected stocks
latest_data = {}
for ticker in selected_stocks:
    try:
        config = ASSETS[ticker.lower()]
        df = pd.read_csv(config['data_file'])
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        price_col = config['features'][0]
        latest_data[ticker] = {
            'price': latest[price_col],
            'change': latest[price_col] - prev[price_col],
            'pct_change': ((latest[price_col] - prev[price_col]) / prev[price_col]) * 100
        }
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")

# Display metrics
cols = st.columns(min(len(selected_stocks), 4))

for i, ticker in enumerate(selected_stocks):
    with cols[i % 4]:
        data = latest_data[ticker]
        render_metric_card(
            label=f"{ticker} - {STOCK_TICKERS[ticker]['name']}",
            value=data['price'],
            delta=data['change']
        )
        
        if data['pct_change'] > 0:
            st.success(f" +{data['pct_change']:.2f}%")
        else:
            st.error(f" {data['pct_change']:.2f}%")
        
        # Show Oil Price reference for stocks
        st.caption(f"Oil: ${latest['Oil_Price']:.2f}")

st.markdown("---")

# ==================== COMPARISON CHART ====================

st.markdown("###  Price Performance Comparison")

# Timeframe selector
timeframe = st.selectbox(
    "Select timeframe",
    ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "All Time"],
    index=3
)

days_map = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "5 Years": 1825,
    "All Time": 99999
}

days = days_map[timeframe]

# Create comparison chart (normalized to 100 at start)
fig = go.Figure()

for ticker in selected_stocks:
    try:
        config = ASSETS[ticker.lower()]
        df = pd.read_csv(config['data_file'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Filter by timeframe
        if days < 99999:
            df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=days)]
        
        # Normalize to 100
        price_col = config['features'][0]
        normalized = (df[price_col] / df[price_col].iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=normalized,
            name=ticker,
            line=dict(color=config['color'], width=2)
        ))
    
    except Exception as e:
        st.error(f"Error plotting {ticker}: {e}")

fig.update_layout(
    template="plotly_dark",
    height=600,
    yaxis_title="Normalized Price (Base 100)",
    xaxis_title="Date",
    margin=dict(l=40, r=100, t=40, b=20),
    hovermode='x unified',
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02,
        bgcolor="rgba(0,0,0,0)"
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==================== INDIVIDUAL STOCK ANALYSIS ====================

st.markdown("###  Individual Stock Deep Dive")

selected_focus = st.selectbox("Select stock for detailed analysis", selected_stocks)

if selected_focus:
    config = ASSETS[selected_focus.lower()]
    df = pd.read_csv(config['data_file'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"#### {selected_focus} - {STOCK_TICKERS[selected_focus]['name']}")
        
        fig = create_price_chart(df, selected_focus, f"{selected_focus} Price History", config['color'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Stock Info")
        st.info(f"""
        **Ticker:** {selected_focus}  
        **Company:** {STOCK_TICKERS[selected_focus]['name']}  
        **Sector:** {STOCK_TICKERS[selected_focus]['sector']}  
        **Current Price:** ${latest_data[selected_focus]['price']:,.2f}  
        **24H Change:** {latest_data[selected_focus]['pct_change']:+.2f}%
        """)

st.markdown("---")

# ==================== AI PREDICTIONS ====================

st.markdown("###  AI-Powered Predictions")

# Add disclaimer
from utils.config import FORECAST_DISCLAIMER
st.info(FORECAST_DISCLAIMER)

# Check which stocks have trained models
stocks_with_models = [t for t in selected_stocks if os.path.exists(ASSETS[t.lower()]['model_file'])]

if not stocks_with_models:
    st.warning(" No trained models for selected stocks. Please train models from Settings page.")
else:
    st.markdown(f"**Models available for:** {', '.join(stocks_with_models)}")
    
    if st.button(" Generate Multi-Range Forecast (All Selected)", use_container_width=True):
        with show_loading_message("Generating all forecasts..."):
            try:
                all_forecasts = batch_multi_range_forecast([s.lower() for s in stocks_with_models])
                
                # NEW: Apply correlation enforcement to prevent impossible divergences
                from utils.correlation_enforcer import CorrelationEnforcer
                
                # Check if we have SPY in forecasts (needed as anchor)
                if 'spy' in [s.lower() for s in stocks_with_models]:
                    st.info("🔗 Applying correlation enforcement...")
                    
                    enforcer = CorrelationEnforcer(reference_ticker='SPY')
                    
                    # Convert forecasts to enforcer format
                    raw_predictions = {}
                    for ticker in stocks_with_models:
                        forecast = all_forecasts[ticker.lower()]
                        if 'error' not in forecast:
                            # Extract prices from new format (handle both dict and float)
                            price_list = []
                            for key in ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']:
                                value = forecast.get(key, 0)
                                if isinstance(value, dict):
                                    price_list.append(value['price'])
                                else:
                                    price_list.append(value)
                            raw_predictions[ticker] = price_list
                    
                    # Apply enforcement
                    adjusted = enforcer.enforce_predictions(raw_predictions, adjustment_strength=0.7)
                    
                    # Update forecasts
                    for ticker in stocks_with_models:
                        if ticker in adjusted:
                            range_keys = ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']
                            for i, key in enumerate(range_keys):
                                # Update the price in the dict format
                                if isinstance(all_forecasts[ticker.lower()][key], dict):
                                    all_forecasts[ticker.lower()][key]['price'] = adjusted[ticker][i]
                                else:
                                    all_forecasts[ticker.lower()][key] = adjusted[ticker][i]
                
                # Valid timeframe keys only (exclude 'ceo_context', 'Current', etc.)
                VALID_TIMEFRAMES = ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']

                # Create results dataframe
                results = []
                chart_data = {}  # For grouped bar chart
                for ticker in stocks_with_models:
                    forecast = all_forecasts[ticker.lower()]
                    if 'error' not in forecast:
                        current_price = forecast.get('Current', 0)
                        row = {
                            'Stock': ticker,
                            'Current': f"${current_price:,.2f}",
                        }
                        pct_changes = []
                        for range_name in VALID_TIMEFRAMES:
                            value = forecast.get(range_name)
                            if value is None:
                                continue
                            price = value['price'] if isinstance(value, dict) else value
                            row[range_name] = f"${price:,.2f}"
                            if current_price > 0:
                                pct = ((price - current_price) / current_price) * 100
                                pct_changes.append(pct)
                            else:
                                pct_changes.append(0)
                        chart_data[ticker] = pct_changes
                        results.append(row)

                st.markdown("####  Multi-Range Forecast Summary")

                # Dynamic height calculation to avoid scrolling
                row_height = 35
                table_height = (len(results) + 1) * row_height + 3

                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True,
                    height=table_height
                )

                # ── Grouped Bar Chart: % Change per Timeframe ──
                if chart_data:
                    st.markdown("####  Forecast % Change — Visual Comparison")
                    fig_bar = go.Figure()
                    colors_map = {t: ASSETS[t.lower()]['color'] for t in stocks_with_models if t.lower() in ASSETS}
                    for ticker, pcts in chart_data.items():
                        fig_bar.add_trace(go.Bar(
                            name=ticker,
                            x=VALID_TIMEFRAMES[:len(pcts)],
                            y=[round(p, 2) for p in pcts],
                            marker_color=colors_map.get(ticker, '#00A8E8'),
                            text=[f"{p:+.1f}%" for p in pcts],
                            textposition='outside',
                        ))
                    fig_bar.update_layout(
                        template='plotly_dark',
                        barmode='group',
                        height=450,
                        yaxis_title="Projected Change (%)",
                        xaxis_title="Timeframe",
                        hovermode='x unified',
                        legend=dict(orientation='h', y=-0.2),
                        margin=dict(t=40, b=80),
                    )
                    fig_bar.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)')
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # NEW: Add deep analysis for batch forecasts
                st.markdown("---")
                st.markdown("###  AI Deep Analysis - Market Overview")
                
                from utils.forecast_analyzer import ForecastAnalyzer
                
                analyzer = ForecastAnalyzer()
                analyses = {}
                
                # Analyze each stock
                for ticker in stocks_with_models:
                    forecast = all_forecasts[ticker.lower()]
                    if 'error' not in forecast:
                        current = forecast['Current']
                        
                        # Extract prices from new format with safety check
                        predictions = []
                        for key in ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']:
                            if isinstance(forecast, dict):
                                value = forecast.get(key, 0)
                                if isinstance(value, dict):
                                    predictions.append(value['price'])
                                else:
                                    predictions.append(value)
                            else:
                                predictions.append(0)
                        
                        insights = analyzer.analyze_forecast(
                            current_price=current,
                            forecast_prices=predictions,
                            asset_name=ticker
                        )
                        analyses[ticker] = insights
                
                # Show top 3 best/worst
                sorted_stocks = sorted(analyses.items(), key=lambda x: x[1]['change_pct'], reverse=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("####  Best Opportunities")
                    for ticker, insights in sorted_stocks[:3]:
                        with st.expander(f"**{ticker}** ({insights['trend'].title()} {insights['change_pct']:+.1f}%)"):
                            st.info(insights['summary'])
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Trend", insights['trend'].title())
                            with col_b:
                                st.metric("Strength", insights['strength'].title())
                            with col_c:
                                st.metric("Risk", insights['risk_level'].title())
                            st.success(f" {insights['recommendation']}")
                
                with col2:
                    st.markdown("####  Watch List")
                    for ticker, insights in sorted_stocks[-3:]:
                        with st.expander(f"**{ticker}** ({insights['trend'].title()} {insights['change_pct']:.1f}%)"):
                            st.info(insights['summary'])
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Trend", insights['trend'].title())
                            with col_b:
                                st.metric("Strength", insights['strength'].title())
                            with col_c:
                                st.metric("Risk", insights['risk_level'].title())
                            st.warning(f" {insights['recommendation']}")

                # ── Sector-Level XAI (one Gemini call for all stocks) ──
                st.markdown("---")
                st.markdown("### 🔍 Sector Intelligence — *Why are markets moving this way?*")
                try:
                    from utils.xai_explainer import explain_sector_forecast, get_top_macro_drivers
                    from utils.macro_processor import build_macro_context

                    # Build compact ticker forecast summary for Gemini
                    ticker_forecasts_xai = {}
                    for ticker in stocks_with_models:
                        fc = all_forecasts.get(ticker.lower(), {})
                        if 'error' not in fc:
                            cur = fc.get('Current', 0)
                            week_val = fc.get('1 Week', {})
                            week_price = week_val['price'] if isinstance(week_val, dict) else week_val
                            pct = ((week_price - cur) / cur * 100) if cur > 0 else 0
                            ticker_forecasts_xai[ticker] = {
                                'direction': 'up' if pct > 0 else ('down' if pct < 0 else 'sideways'),
                                'pct_change': round(pct, 2),
                            }

                    # Get top macro drivers using SPY as reference
                    top_drivers_xai = get_top_macro_drivers('spy', lookback_days=14, top_n=3)
                    macro_ctx_xai = build_macro_context()
                    macro_summary_xai = macro_ctx_xai.get('macro_summary', '')

                    # Show driver table
                    if top_drivers_xai:
                        import pandas as _pd_xai
                        st.markdown("**📊 Top Macro Drivers (14-Day Movement vs Historical):**")
                        driver_rows_xai = [{
                            'Indicator': d['label'],
                            'Recent Avg': d['recent_mean'],
                            'Hist Avg': d['hist_mean'],
                            'Z-Score': d['z_score'],
                            'Direction': '▲ Rising' if d['direction'] == 'rising' else '▼ Falling',
                        } for d in top_drivers_xai]
                        st.dataframe(_pd_xai.DataFrame(driver_rows_xai), use_container_width=True, hide_index=True)

                    with st.spinner("🧠 Gemini analyzing sector dynamics..."):
                        sector_narrative = explain_sector_forecast(
                            ticker_forecasts=ticker_forecasts_xai,
                            macro_summary=macro_summary_xai,
                            top_drivers=top_drivers_xai,
                        )
                    st.info(f"🧠 **Gemini Sector Analysis:** {sector_narrative}")
                    st.caption("⚠️ PROBABILISTIC FORECAST — Not a trading signal. Confidence intervals apply.")
                except Exception as _xe:
                    st.caption(f"XAI sector analysis unavailable: {_xe}")

            except Exception as e:
                show_error_message(f"Prediction error: {e}")
    
    # Individual detailed forecast
    st.markdown("---")
    st.markdown("####  Multi-Range Forecast (Individual Stock)")
    
    forecast_stock = st.selectbox("Select stock for detailed forecast", stocks_with_models)
    
    if st.button(f"Generate Forecast for {forecast_stock}"):
        with show_loading_message(f"Analyzing {forecast_stock}..."):
            try:
                predictor = AssetPredictor(forecast_stock.lower())
                forecasts = predictor.get_multi_range_forecast()
                
                # CRITICAL FIX: Apply correlation enforcement for individual forecasts too!
                # This ensures QQQ shows same trend individual vs batch mode
                from utils.correlation_enforcer import CorrelationEnforcer
                
                # Check if this stock needs correlation enforcement
                # (i.e., if it's highly correlated with SPY and not SPY itself)
                if forecast_stock.upper() != 'SPY':
                    st.info(f"🔗 Checking correlation with SPY for realistic forecast...")
                    
                    enforcer = CorrelationEnforcer(reference_ticker='SPY')
                    
                    # Get SPY forecast for comparison
                    predictor = AssetPredictor('spy')
                    fetched_forecasts = predictor.get_multi_range_forecast()
                    
                    if isinstance(fetched_forecasts, dict):
                        spy_forecasts = fetched_forecasts
                    else:
                        spy_forecasts = {'Current': 0, 'error': 'Invalid data format'}
                    
                    # Prepare data for enforcement
                    raw_predictions = {
                        'SPY': [],
                        forecast_stock.upper(): []
                    }
                    
                    # Extract prices from new format
                    for key in ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']:
                        spy_val = spy_forecasts.get(key, 0)
                        stock_val = forecasts.get(key, 0)
                        
                        if isinstance(spy_val, dict):
                            raw_predictions['SPY'].append(spy_val['price'])
                        else:
                            raw_predictions['SPY'].append(spy_val)
                        
                        if isinstance(stock_val, dict):
                            raw_predictions[forecast_stock.upper()].append(stock_val['price'])
                        else:
                            raw_predictions[forecast_stock.upper()].append(stock_val)
                    
                    # Apply enforcement (70% strength like batch mode)
                    adjusted = enforcer.enforce_predictions(raw_predictions, adjustment_strength=0.7)
                    
                    # Update forecasts if adjustment was applied
                    if forecast_stock.upper() in adjusted:
                        range_keys = ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']
                        for i, key in enumerate(range_keys):
                            # Update the price in the dict format
                            if isinstance(forecasts[key], dict):
                                forecasts[key]['price'] = adjusted[forecast_stock.upper()][i]
                            else:
                                forecasts[key] = adjusted[forecast_stock.upper()][i]
                            
                        st.success(" Correlation enforcement applied - forecast aligned with SPY")
                
                render_prediction_table(forecasts, forecast_stock)
                
                # NEW: Add automated analysis for individual forecast
                from utils.forecast_analyzer import ForecastAnalyzer
                
                analyzer = ForecastAnalyzer()
                
                # Final guard for analysis
                if isinstance(forecasts, dict) and 'Current' in forecasts:
                    # Extract prices from new format
                    forecast_prices = []
                    for key in ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']:
                        value = forecasts.get(key, 0)
                        if isinstance(value, dict):
                            forecast_prices.append(value['price'])
                        else:
                            forecast_prices.append(value)
                    
                    insights = analyzer.analyze_forecast(
                        current_price=forecasts['Current'],
                        forecast_prices=forecast_prices,
                        asset_name=forecast_stock
                    )
                    
                    st.markdown("### AI Deep Analysis")
                    st.info(insights['summary'])
                    
                    col_i1, col_i2, col_i3 = st.columns(3)
                    with col_i1:
                        st.metric("Trend", insights['trend'].title())
                    with col_i2:
                        st.metric("Strength", insights['strength'].title())
                    with col_i3:
                        st.metric("Risk", insights['risk_level'].title())
                    
                    st.success(f" **Recommendation**: {insights['recommendation']}")
                else:
                    st.warning(" Detailed AI analysis unavailable due to incomplete forecast data.")
                
                # Show forecast chart (Fan Chart)
                st.markdown("####  30-Day Probability Cloud (Fan Chart)")
                df_stock = pd.read_csv(ASSETS[forecast_stock.lower()]['data_file'])
                df_stock['Date'] = pd.to_datetime(df_stock['Date'])
                
                predictor = AssetPredictor(forecast_stock.lower())
                fetched_forecasts = predictor.get_multi_range_forecast()
                
                if isinstance(fetched_forecasts, dict):
                    forecasts = fetched_forecasts
                else:
                    forecasts = {'Current': 0, 'error': 'Invalid data format'}
                
                month_data = forecasts.get('1 Month', {})
                forecast_30d = month_data.get('series', [])
                fan_p10 = month_data.get('fan_p10')
                fan_p90 = month_data.get('fan_p90')
                
                if not forecast_30d:
                    forecast_30d = predictor.recursive_forecast(30)
                    
                from utils.ui_components import create_forecast_chart
                fig = create_forecast_chart(df_stock.tail(90), forecast_30d, forecast_stock, len(forecast_30d), fan_p10=fan_p10, fan_p90=fan_p90)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                show_error_message(f"Error: {e}")

st.markdown("---")

# ==================== SECTOR ANALYSIS ====================

st.markdown("###  Sector Performance")

# Group by sector
sector_performance = {}
for ticker in selected_stocks:
    sector = STOCK_TICKERS[ticker]['sector']
    if sector not in sector_performance:
        sector_performance[sector] = []
    sector_performance[sector].append({
        'ticker': ticker,
        'change': latest_data[ticker]['pct_change']
    })

cols = st.columns(len(sector_performance))

for i, (sector, stocks) in enumerate(sector_performance.items()):
    with cols[i]:
        st.markdown(f"#### {sector}")
        
        avg_change = sum(s['change'] for s in stocks) / len(stocks)
        
        for stock in stocks:
            st.text(f"{stock['ticker']}: {stock['change']:+.2f}%")
        
        st.markdown("---")
        if avg_change > 0:
            st.success(f"Avg: +{avg_change:.2f}%")
        else:
            st.error(f"Avg: {avg_change:.2f}%")

# ==================== NEWS (Optional) ====================

st.markdown("---")
st.markdown("###  Latest Market News")

# Show news for selected stock (ui_components handles file check and empty state)
if selected_focus:
    render_news_section(selected_focus.lower(), max_items=20)

# ==================== DISCLAIMER ====================

st.markdown("---")
st.warning("""
 **Investment Risk**: Stock market investments carry risk. Past performance does not guarantee future results.
This analysis is for educational purposes only. Conduct your own research and consult a financial advisor.
""")
