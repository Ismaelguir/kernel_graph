import pandas as pd
import numpy as np
import json

DATE = "2020-03-16"
L = 60

# 1) tickers (ordre des colonnes attendu)
meta = json.load(open("data/processed/tickers.json"))
tickers = [meta["index_to_ticker"][str(i)] for i in range(len(meta["index_to_ticker"]))]

# 2) prix ajustés et alignement
prices = pd.read_csv("data/raw/adj_close_2014-01-01_2024-12-31.csv", index_col=0, parse_dates=True)
prices = prices[tickers].dropna(axis=0, how="any")  # intersection stricte, comme dans le pipeline

# 3) rendements log
rets = np.log(prices).diff().dropna()

# 4) fenêtre de corrélation finissant à DATE
t = pd.to_datetime(DATE)
window = rets.loc[:t].tail(L)  # 60 derniers jours de rendements jusqu’à DATE
rho = window.corr().to_numpy() # (60,60)

print(rho.shape, rho.min(), rho.max())