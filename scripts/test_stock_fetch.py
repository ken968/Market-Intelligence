import yfinance as yf
from datetime import datetime

def test_stock_fetch():
    print(f"Current Time: {datetime.now()}")
    
    # Check SPY
    print(f"\nChecking SPY (period='10y')")
    data = yf.download('SPY', period='10y', interval='1d', progress=False)
    if not data.empty:
        print(f"Latest Date: {data.index[-1]}")
        print(f"Latest Close: {data['Close'].iloc[-1]}")
    else:
        print("No data returned")

if __name__ == "__main__":
    test_stock_fetch()
