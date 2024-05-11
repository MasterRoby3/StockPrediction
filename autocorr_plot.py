import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import yfinance as yf
from datetime import datetime
import os, sys

from sklearn import preprocessing

#bodacious colors
colors=sns.color_palette("rocket", 8)
#Ram's colors, if desired
seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
#            0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry

stock_data = pd.read_pickle("data/MSFT_data.pkl")

daily_returns = ((stock_data["Close"] - stock_data["Open"]) / stock_data["Open"]).to_numpy() * 100
prices = stock_data[["Open", "High", "Low", "Close"]].to_numpy()
volume = stock_data["Volume"].to_numpy()

minmax_scaler = preprocessing.MinMaxScaler()
std_scaler = preprocessing.StandardScaler()

features = np.vstack((daily_returns, volume)).T

# Scale volume data to obtain better results
#minmax_scaler = preprocessing.MinMaxScaler()
#norm_ret = std_scaler.fit_transform(daily_returns.reshape(-1,1)).flatten()
#norm_vol = minmax_scaler.fit_transform(volume.reshape(-1,1)).flatten()
#norm_features = np.vstack((norm_ret, norm_vol)).T

# Solo volumi e ritorni
#norm_features = std_scaler.fit_transform(features)

# Aggiunta di prezzi
#norm_prices = minmax_scaler.fit_transform(prices.reshape(-1, 1)).reshape(-1, 4)
#norm_ret_and_vol = std_scaler.fit_transform(features)
#norm_features = np.hstack((norm_ret_and_vol, norm_prices))

# Necessary for MAs
part_features = std_scaler.fit_transform(features)

# Aggiunta SMA
#SMA_20 = stock_data["Close"].rolling(20).mean().to_numpy()
#SMA_50 = stock_data["Close"].rolling(50).mean().to_numpy()
#SMA_200 = stock_data["Close"].rolling(200).mean().to_numpy()
#SMAs = np.vstack((SMA_20, SMA_50)).T
#norm_SMAs = minmax_scaler.fit_transform(SMAs[49:, ].reshape(-1, 1)).reshape(-1, 2)
#norm_features = np.hstack((part_features[49:, ], norm_SMAs))

#SMAs = np.vstack((SMA_20, SMA_50, SMA_200)).T
#norm_SMAs = minmax_scaler.fit_transform(SMAs[199:, ].reshape(-1, 1)).reshape(-1, 3)
#norm_features = np.hstack((part_features[199:, ], norm_SMAs))

# Aggiunta EMA
EMA_20 = stock_data["Close"].ewm(span=20, adjust=False).mean()
EMA_50 = stock_data["Close"].ewm(span=50, adjust=False).mean()
EMAs = np.vstack((EMA_20, EMA_50)).T
norm_EMAs = minmax_scaler.fit_transform(EMAs.reshape(-1, 1)).reshape(-1, 2)

#EMA_200 = stock_data["Close"].ewm(span=200, adjust=False).mean()
#EMAs = np.vstack((EMA_20, EMA_50, EMA_200)).T
#norm_EMAs = minmax_scaler.fit_transform(EMAs.reshape(-1, 1)).reshape(-1, 3)
norm_features = np.hstack((part_features, norm_EMAs))

dfeat = {"Daily Returns" : norm_features[:,0],
"Volume" : norm_features[:,1],
"EMA20" : norm_features[:,2],
"EMA50" : norm_features[:,3] 
}

corr = pd.DataFrame(dfeat).corr()
fig = plt.figure(1, (11, 10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="mako")
plt.tick_params(labelsize=14)

plt.savefig("plots/Correlation_EMAs.png", dpi=300)

# merge data into 2d numpy array
Y = np.zeros(features.shape[0] - 1)


for i in range(Y.size):
    if daily_returns[i+1] >= 0:
        Y[i] = 1
    else:
        Y[i] = 0

# per quando su usano ma fino a 200
#Y = Y[49:]
#Y = Y[199:]

print(norm_features.shape, Y.shape)

fig, ax = plt.subplots(figsize=(15,10))


#plot params
#plt.xlim([-12,12])
#plt.ylim([-0.5,16])
ax.minorticks_on()
ax.tick_params(labelsize=14)
ax.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
#xticks = np.arange(0, 1e4,10)
#yticks = np.arange(0,16.1,4)

ax.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
ax.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
#plt.xticks(xticks)
#plt.yticks(yticks)


#plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)

ax.set_xlim([0, 500])
#ax.set_ylim([-0.5, 0.5])




pd.plotting.autocorrelation_plot(daily_returns, ax=ax, color=seshadri[0], label="Daily Returns")
pd.plotting.autocorrelation_plot(np.abs(daily_returns), ax=ax, color=seshadri[1], label="Absolute Daily Returns")
pd.plotting.autocorrelation_plot(volume, ax=ax, color=seshadri[2], label="Volume")

ax.grid(False)
ax.set_xlabel(r'Lag', fontsize=14) 
ax.set_ylabel(r'Autocorrelation',fontsize=14)  # label the y axis


ax.legend(fontsize=14, loc="upper right", bbox_to_anchor=(0.99, 0.99))  # add the legend (will default to 'best' location)
plt.savefig("plots/Autocorrelation_returns_volume_abs.png", dpi=300)
