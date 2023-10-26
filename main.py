import pandas as pd

stock_data = pd.read_csv("data/daily_MSFT.csv", index_col="timestamp")

print(stock_data)