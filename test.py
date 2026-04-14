import yfinance as yf
df = yf.download("AAPL", start="2018-01-01", end="2024-01-01", progress=False, auto_adjust=True)
print(df)
print(df.columns)