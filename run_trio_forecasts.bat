@echo off
echo ========================================================
echo Market Intelligence - Trio Forecast (SPY, GOLD, BTC)
echo ========================================================
.\.venv\Scripts\python.exe scripts\generate_forecasts.py spy,gold,btc
echo.
echo Forecast Generation Complete!
pause
