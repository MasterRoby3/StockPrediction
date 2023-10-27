import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import yfinance as yf
from datetime import datetime

import os

#bodacious colors
colors=sns.color_palette("rocket", 8)
#Ram's colors, if desired
seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
#            0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry

#stock_data = pd.read_csv("data/daily_MSFT.csv").iloc[::-1].reset_index(drop=True)

if os.path.isfile("data/MSFT_data.pkl"):
    stock_data = pd.read_pickle("data/MSFT_data.pkl")
elif os.path.isfile("data/MSFT_data.csv"):
    stock_data = pd.read_csv("data/MSFT_data.csv")
else:
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2023, 10, 26)
    stock_data = yf.download('MSFT', start=start_date, end=end_date)
    stock_data.to_pickle("data/MSFT_data.pkl")
    stock_data.to_csv("data/MSFT_data.csv")




daily_returns = stock_data["Close"] - stock_data["Open"]
win_lose = np.zeros(daily_returns.size - 1)

for index, return_ in enumerate(daily_returns[:-1]):
    if (return_ > 0 and daily_returns[index + 1] > 0) or (return_ < 0 and daily_returns[index + 1] < 0):
        win_lose[index] = 1
    else:
        win_lose[index] = 0


win_rate = np.count_nonzero(win_lose == 1) / win_lose.size

print(win_rate)

percent_returns = daily_returns / stock_data["Open"] * 100

fig = plt.figure(1, figsize=(15,10))
plt.hist(percent_returns, bins = 120, range=(-12,12), facecolor=seshadri[0], alpha=0.8, edgecolor="white", label="Percentage daily returns occurrences")

#plt.plot(stock_data.index, stock_data["Close"], linestyle="-", color=seshadri[0])
#plt.plot(stock_data.index, stock_data["Adj Close"], linestyle="-", color=seshadri[1])
#plt.plot(stock_data.index, stock_data["close"] - 20, linestyle="-", color=seshadri[2])
#plt.plot(stock_data.index, stock_data["close"] - 30, linestyle="-", color=seshadri[3])
#plt.plot(stock_data.index, stock_data["close"] - 40, linestyle="-", color=seshadri[4])
#plt.plot(stock_data.index, stock_data["close"] - 50, linestyle="-", color=seshadri[5])
#plt.plot(stock_data.index, stock_data["close"] - 60, linestyle="-", color=seshadri[6])
#plt.plot(stock_data.index, stock_data["close"] - 70, linestyle="-", color=seshadri[7])
#plt.show()


#plot params
plt.xlim([-12,12])
#plt.ylim([-0.5,16])
plt.minorticks_on()
plt.tick_params(labelsize=14)
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
#xticks = np.arange(0, 1e4,10)
#yticks = np.arange(0,16.1,4)

plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
#plt.xticks(xticks)
#plt.yticks(yticks)


#plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)


plt.xlabel(r'Percentage daily return', fontsize=14) 
plt.ylabel(r'Occurrences',fontsize=14)  # label the y axis


plt.legend(fontsize=14, loc="upper right", bbox_to_anchor=(0.99, 0.99))  # add the legend (will default to 'best' location)

plt.savefig("plots/MSFT_daily_occs.png", dpi=300)

plt.show()

