import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import yfinance as yf
from datetime import datetime
import os, sys

from sklearn import preprocessing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#bodacious colors
colors=sns.color_palette("rocket", 8)
#Ram's colors, if desired
seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
#            0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry

np.set_printoptions(threshold=100)

def enlarge_lag(to_enlarge, time_window=1):
    # to_enlarge is the data already present, should be a numpy array
    enlarged = []
    for i in range(to_enlarge.shape[0] - time_window + 1):
        new_element = []
        for j in range(time_window):
            new_element.extend(to_enlarge[i + time_window - 1 - j, :])
        enlarged.append(new_element)

    return np.array(enlarged)

#### Calculate the metrics RMSE and MAPE ####
def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

train_quota = 0.8

if len(sys.argv) > 1:
    time_window = int(sys.argv[1])
else:
    time_window = 1

#time_window = 10

stock_data = pd.read_pickle("data/MSFT_data.pkl")

price = stock_data["Close"].to_numpy()
volume = stock_data["Volume"].to_numpy()
daily_returns = ((stock_data["Close"] - stock_data["Open"]) / stock_data["Open"]).to_numpy()

minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
sec_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

#features = np.vstack((price, volume)).T

# Necessary for MAs
#norm_features = np.hstack((minmax_scaler.fit_transform(price.reshape(-1, 1)), sec_scaler.fit_transform(volume.reshape(-1, 1))))
norm_features = minmax_scaler.fit_transform(price.reshape(-1, 1))

rets = np.diff(price)
bin_rets = np.zeros(len(rets))
for i, r in enumerate(rets):
    if r >= 0:
        bin_rets[i] = 1
    else:
        bin_rets[i] = 0

bin_rets_np = np.array(bin_rets)


#norm_rets = sec_scaler.fit_transform(rets.reshape(-1, 1))

print("occai")

print(rets)
print(bin_rets)

print("ocai")

# merge data into 2d numpy array
#Y = np.zeros(norm_features.shape[0] - 1)
#for i in range(Y.size):
#    Y[i] = norm_features[i+1, 0]

Y = bin_rets

time_window = 20

if time_window > 1:
    norm_features = enlarge_lag(norm_features, time_window)
    Y = Y[time_window-1:]


train_size = int(norm_features.shape[0] * 0.8)
X_train = norm_features[:train_size, ]
Y_train = Y[:train_size]

X_test = norm_features[train_size:-1, ]
Y_test = Y[train_size:]

def LSTM_model():
    model = Sequential()

    model.add(LSTM(units = 20, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(LSTM(units=20))
    model.add(Dropout(0.2))

    model.add(Dense(units=5))
    model.add(Dropout(0.3))

    model.add(Dense(units=1, activation="sigmoid"))

    return model

model = LSTM_model()
model.summary()
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

#if os.path.exists("./checkpoints/checkpoint"):
#    model.load_weights("./checkpoints/my_checkpoint")
#else:
model.fit(
    X_train, 
    Y_train,
    shuffle=True,
    epochs=20,
    batch_size=20
)
    
    #model.save_weights("./checkpoints/my_checkpoint")

prediction = model.predict(X_test)
print(prediction)
print(model.evaluate(X_test, Y_test))
#predicted_prices = minmax_scaler.inverse_transform(prediction).flatten()
#predicted_rets = sec_scaler.inverse_transform(prediction).flatten()
#print(predicted_rets)
#counter = 0
#for i in range(prediction.shape[0]-1):
#    if (prediction[i+1,] - prediction[i,] > 0 and predicted_prices[i+1,] - predicted_prices[i,] > 0) or (prediction[i+1,] - prediction[i,] < 0 and predicted_prices[i+1,] - predicted_prices[i,] < 0):
#        counter = counter + 1

#print("acc: ", counter/prediction.shape[0])



#test_prices = price[time_window - 1 + train_size:]
#pred_ret = []
#actual_ret = []
#for j in range(len(test_prices) - 1):
#    # il predicted price è il prezzo di domani, lo voglio confrontare con il ritorno effettivo di domani
#    pred_ret.append((predicted_prices[j] - test_prices[j])/test_prices[j])
#    actual_ret.append((test_prices[j+1] - test_prices[j])/test_prices[j])
#
#pred_ret_np = np.array(pred_ret)
#actual_ret_np = np.array(actual_ret)
#
#sign_comp = np.sum(np.sign(pred_ret_np) == np.sign(actual_ret_np))/len(pred_ret_np)
#sign_comp_red_nottoomuch = np.sum(np.sign(pred_ret_np[:200]) == np.sign(actual_ret_np[:200]))/len(pred_ret_np[:200])
#sign_comp_red = np.sum(np.sign(pred_ret_np[:100]) == np.sign(actual_ret_np[:100]))/len(pred_ret_np[:100])
#sign_comp_red_alot = np.sum(np.sign(pred_ret_np[:50]) == np.sign(actual_ret_np[:50]))/len(pred_ret_np[:50])
#print(sign_comp)
#print(sign_comp_red_nottoomuch)
#print(sign_comp_red)
#print(sign_comp_red_alot)

#rmse = calculate_rmse(test_prices[1:], predicted_prices)
#mape = calculate_mape(test_prices[1:], predicted_prices)
#
#print("RMSE: ", rmse)
#print("MAPE: ", mape)
#
#rmse = calculate_rmse(test_prices[1:301], predicted_prices[:300])
#mape = calculate_mape(test_prices[1:301], predicted_prices[:300])
#
#print("RMSE su 300 gg: ", rmse)
#print("MAPE su 300 gg: ", mape)

#plt.plot(pred_ret, color=seshadri[0])
#plt.plot(daily_returns[1:], color=seshadri[1])

fig = plt.figure(1, figsize=(12,10))
plt.plot(Y_test, color=seshadri[0], label="Registered Closing Price")
plt.plot(prediction, color=seshadri[1], label="Prediction")

#plot params
#plt.xlim([0,450])
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


plt.xlabel(r'Days (from last training)', fontsize=14) 
plt.ylabel(r'Price (USD)',fontsize=14)  # label the y axis


plt.legend(fontsize=14, loc="upper right", bbox_to_anchor=(0.99, 0.99))  # add the legend (will default to 'best' location)

plt.savefig("plots/LSTM_advanced_rets_1.png", dpi=300)

plt.show()
#with open("plots/data/MLP_20_10_5_2.csv", "a") as f:
#    f.write(f"{time_window};{train_score};{score};\n")