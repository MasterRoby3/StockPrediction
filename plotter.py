import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

#bodacious colors
colors=sns.color_palette("rocket", 8)
#Ram's colors, if desired
seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
#            0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry

data = pd.read_csv("plots/data/MLP_20_10_5_2.csv", sep=";")
#data = pd.read_csv("plots/data/logistic_regression.csv", sep=";")
#data_SMA = pd.read_csv("plots/data/logistic_regression_SMA.csv", sep=";")
#data_SMA_20_50 = pd.read_csv("plots/data/logistic_regression_SMA_20_50.csv", sep=";")
#data_EMA = pd.read_csv("plots/data/logistic_regression_EMA.csv", sep=";")
#data_EMA_20_50 = pd.read_csv("plots/data/logistic_regression_EMA_20_50.csv", sep=";")

print(data)

fig = plt.figure(1, figsize=(15,10))
plt.plot(data["time_window"], data["training_accuracy"]*100, color=seshadri[0], label="Training Accuracy", linewidth=2)
plt.plot(data["time_window"], data["testing_accuracy"]*100, color=seshadri[1], label="Testing Accuracy", linewidth=2)



#plt.plot(data["time_window"], data["testing_accuracy"]*100, color=seshadri[0], label="Returns and Volume", linewidth=2)
#plt.plot(data_SMA_20_50["time_window"], data_SMA_20_50["testing_accuracy"]*100, color=seshadri[1], label="With SMA 20 and 50 candles", linewidth=2)
#plt.plot(data_SMA["time_window"], data_SMA["testing_accuracy"]*100, color=seshadri[2], label="With SMA 20, 50 and 200 candles", linewidth=2)
#plt.plot(data_EMA_20_50["time_window"], data_EMA_20_50["testing_accuracy"]*100, color=seshadri[3], label="With EMA 20 and 50 candles", linewidth=2)
#plt.plot(data_EMA["time_window"], data_EMA["testing_accuracy"]*100, color=seshadri[4], label="With EMA 20, 50 and 200 candles", linewidth=2)


#plot params
plt.xlim([0, 50])
#plt.ylim([50, 60])
plt.minorticks_on()
plt.tick_params(labelsize=14)
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
#xticks = np.arange(0, 1e4,10)
#yticks = np.arange(0,16.1,4)

plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
#plt.xticks(xticks)
#plt.yticks(yticks)
#plt.grid(True)

#plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)


plt.xlabel(r'Lag (Days)', fontsize=14) 
plt.ylabel(r'Accuracy (%)',fontsize=14)  # label the y axis

plt.legend(fontsize=14, loc="upper right", bbox_to_anchor=(0.99, 0.99))  # add the legend (will default to 'best' location)

plt.savefig("plots/MLP_20_10_5_2.png", dpi=300)

plt.show()