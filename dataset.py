import pandas as pd

data = {
    "Date": pd.date_range(start="2024-01-01", periods=100, freq="B"),
    "Open": [130 + (i % 5) for i in range(100)],
    "High": [132 + (i % 5) for i in range(100)],
    "Low": [128 + (i % 5) for i in range(100)],
    "Close": [131 + (i % 5) for i in range(100)],
    "Volume": [80000000 + (i * 10000) for i in range(100)],
}

df = pd.DataFrame(data)
df.to_csv("AAPL.csv", index=False)
print("✅ CSV generated with 100 rows.")
