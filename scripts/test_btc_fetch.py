import yfinance as yf
from datetime import datetime, timedelta

def test_btc_fetch():
    print(f"Current Time: {datetime.now()}")
    
    # Method 1: Current implementation
    end_date = datetime.now().strftime('%Y-%m-%d')
    print(f"\nMethod 1 (Current): Start=2024-01-01, End={end_date}")
    data1 = yf.download('BTC-USD', start='2024-01-01', end=end_date, interval='1d', progress=False)
    if not data1.empty:
        print(f"Latest Date: {data1.index[-1]}")
        print(f"Latest Close: {data1['Close'].iloc[-1]}")
    else:
        print("No data returned")

    # Method 2: No end date
    print(f"\nMethod 2 (No End): Start=2024-01-01")
    data2 = yf.download('BTC-USD', start='2024-01-01', interval='1d', progress=False)
    if not data2.empty:
        print(f"Latest Date: {data2.index[-1]}")
        print(f"Latest Close: {data2['Close'].iloc[-1]}")
    else:
        print("No data returned")

    # Method 3: Period max
    print(f"\nMethod 3 (Period=1mo)")
    data3 = yf.download('BTC-USD', period='1mo', interval='1d', progress=False)
    if not data3.empty:
        print(f"Latest Date: {data3.index[-1]}")
        print(f"Latest Close: {data3['Close'].iloc[-1]}")
    else:
        print("No data returned")

if __name__ == "__main__":
    test_btc_fetch()
