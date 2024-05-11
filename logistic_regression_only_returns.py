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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logreg_inference(x, w, b):
    z = (x @ w) + b
    p = sigmoid(z)
    return p


def cross_entropy(P, Y):
    return (-Y * np.log(P) - (1 - Y) * np.log(1 - P)).mean()


def logreg_train(X, Y, lambda_, lr = 1e-4, steps=100000):
    # The training samples are defined as such (each row of X is a sample):
    # X[0, :] -> Y[0]
    # X[1, :] -> Y[1]

    m, n = X.shape

    # Initial values for the parameters
    w = np.zeros(n)
    b = 0

    # Initial values for the "precedent loss" and "convergence" variables, used to check convergence
    prec_loss = 0
    convergence = 0

    for step in range(steps):
        P = logreg_inference(X, w, b)
        loss = cross_entropy(P, Y)
        

        if step % 1000 == 0:
            print(step, loss)

            # Difference between "precedent loss" and "current loss"
            diff = np.absolute(prec_loss - loss)
            prec_loss = loss
            if diff < 0.00001:
                # If convergence is reached, the algorithm is stopped
                convergence = step
                break

        # Derivative of the loss function with respect to bias
        grad_b = (P - Y).mean()

        # Gradient of the loss function with respect to weights
        grad_w = (X.T @ (P - Y)) / m

        w -= lr * grad_w
        b -= lr * grad_b

        # Every 100 iteration the values of accuracy and loss are saved for plotting
        if step%100 == 0:
            Yhat = (P > 0.5)
            acc_array.append((Y == Yhat).mean() * 100)
            losses.append(loss)

    # Print the iterations needed for convergence before returning
    print("Convergence = ", convergence)

    return w, b


if len(sys.argv) > 1:
    time_window = int(sys.argv[1])
else:
    time_window = 1

#time_window = 10

stock_data = pd.read_pickle("data/MSFT_data.pkl")

daily_returns = ((stock_data["Close"] - stock_data["Open"]) / stock_data["Open"]).to_numpy().reshape(-1,1)


# merge data into 2d numpy array
Y = np.zeros(daily_returns.shape[0] - 1)

print(daily_returns.shape, Y.shape)

for i in range(Y.size):
    if daily_returns[i+1] >= 0:
        Y[i] = 1
    else:
        Y[i] = 0
import copy
norm_features = copy.deepcopy(daily_returns)
if time_window > 1:
    norm_features = enlarge_lag(norm_features, time_window)
    Y = Y[time_window-1:]

train_size = int(norm_features.shape[0] * 0.8)
X_train = norm_features[:train_size, ]
Y_train = Y[:train_size]

X_test = norm_features[train_size:-1, ]
Y_test = Y[train_size:]


# Lists to save accuracy and loss
acc_array = []
losses = []

w, b = logreg_train(X_train, Y_train, 0.0, 1e-3, 1000000)
print("Weights: ", w)
print("Bias: ", b)

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
# Training accuracy of the model, is the last value recorded in the array
print("Training Acc: ", acc_array[-1])

P_test = logreg_inference(X_test, w, b)
Yhat_test = (P_test > 0.5)
accuracy_test = (Y_test == Yhat_test).mean()
print("Test accuracy: ", 100*accuracy_test)


#lets try sklearn
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state=0, solver="saga").fit(X_train, Y_train)
#score = classifier.score(X_test, Y_test)
#print("sklearn score, all default: ", score)

with open("plots/data/logistic_regression_only_rets.csv", "a") as f:
    f.write(f"{time_window};{acc_array[-1]};{accuracy_test};\n")