from matplotlib.pyplot import figure
import pandas as pd
import pmdarima as pm
from pmdarima.arima import ARIMA as arima
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

train_set = pd.read_csv("./hw3_Data2/train.csv")
test_set = pd.read_csv("./hw3_Data2/test.csv")

train_y = train_set.loc[:, "Close"].to_numpy()
train_x = list(train_set.loc[:, "Date"])

test_y = test_set.loc[:, "Close"].to_numpy()
test_x = list(test_set.loc[:, "Date"])
train_x.extend(test_x)
for i in range(len(train_x)):
    if i % 10 != 0:
        train_x[i] = ""
    else:
        train_x[i] = train_x[i].replace("2021-", "")
        train_x[i] = train_x[i].replace("2022-", "")

Lowest_mse = 10000000
best_m = 12

order = (1, 2, 1)
seasonal_order = (3, 1, 5, 12)
model = arima(order=order, seasonal_order=seasonal_order)
 
model.fit(train_y)
forecasts = model.predict(test_y.shape[0])  # predict N steps into the future

mse = mean_squared_error(test_y, forecasts)  

print("Lowest MSE = ", mse)

figure(figsize=(300, 60), dpi=150)
plt.title(f"MSE is {mse}")
plt.ylabel("Close Value", fontsize=16)
plt.xlabel("Date", fontsize=16)
plt.xticks(range(0, len(train_x)*3, 3), train_x, size='small')
plt.plot(range(0, len(train_y)*3, 3), train_y, c="blue", label="Train_set")
plt.plot(range(len(train_y)*3, len(train_x)*3, 3), test_y, c="green", label="Answer")
plt.plot(range(len(train_y)*3, len(train_x)*3, 3), forecasts, c="red", label="Forecast")
plt.legend(loc="best")
plt.show()
