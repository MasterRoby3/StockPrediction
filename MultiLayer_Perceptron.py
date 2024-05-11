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

train_quota = 0.8

def enlarge_lag(to_enlarge, time_window=1):
    # to_enlarge is the data already present, should be a numpy array
    enlarged = []
    for i in range(to_enlarge.shape[0] - time_window + 1):
        new_element = []
        for j in range(time_window):
            new_element.extend(to_enlarge[i + time_window - 1 - j, :])
        enlarged.append(new_element)

    return np.array(enlarged)


if len(sys.argv) > 1:
    time_window = int(sys.argv[1])
else:
    time_window = 1

#time_window = 10

stock_data = pd.read_pickle("data/MSFT_data.pkl")

daily_returns = ((stock_data["Close"] - stock_data["Open"]) / stock_data["Open"]).to_numpy()
prices = stock_data[["Open", "High", "Low", "Close"]].to_numpy()
volume = stock_data["Volume"].to_numpy()

minmax_scaler = preprocessing.MinMaxScaler()
std_scaler = preprocessing.StandardScaler()

features = np.vstack((daily_returns, volume)).T

# Necessary for MAs
part_features = std_scaler.fit_transform(features)

# Aggiunta EMA
EMA_20 = stock_data["Close"].ewm(span=20, adjust=False).mean()
EMA_50 = stock_data["Close"].ewm(span=50, adjust=False).mean()
EMAs = np.vstack((EMA_20, EMA_50)).T
norm_EMAs = minmax_scaler.fit_transform(EMAs.reshape(-1, 1)).reshape(-1, 2)

#EMA_200 = stock_data["Close"].ewm(span=200, adjust=False).mean()
#EMAs = np.vstack((EMA_20, EMA_50, EMA_200)).T
#norm_EMAs = minmax_scaler.fit_transform(EMAs.reshape(-1, 1)).reshape(-1, 3)
norm_features = np.hstack((part_features, norm_EMAs))


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

if time_window > 1:
    norm_features = enlarge_lag(norm_features, time_window)
    Y = Y[time_window-1:]

train_size = int(norm_features.shape[0] * 0.8)
X_train = norm_features[:train_size, ]
Y_train = Y[:train_size]

X_test = norm_features[train_size:-1, ]
Y_test = Y[train_size:]



# Iterations vs Accuracy plot
#plt.figure()
#plt.plot(np.arange(0, len(acc_array)) * 100, acc_array)
#plt.xlabel("Iterations")
#plt.ylabel("Accuracy")
#
## Iterations vs Loss plot
#plt.figure()
#plt.plot(np.arange(0, len(acc_array)) * 100, losses)
#plt.xlabel("Iterations")
#plt.ylabel("Losses")
#
#plt.show()



#lets try sklearn
from sklearn.neural_network import MLPClassifier
#classifier = LogisticRegression(random_state=0, solver="saga").fit(X_train, Y_train)
clf = MLPClassifier(hidden_layer_sizes=(20,10,5,2), max_iter=30000, verbose=True).fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
score = clf.score(X_test, Y_test)
print("sklearn score, all default: ", score, " train ", train_score)

with open("plots/data/MLP_20_10_5_2.csv", "a") as f:
    f.write(f"{time_window};{train_score};{score};\n")