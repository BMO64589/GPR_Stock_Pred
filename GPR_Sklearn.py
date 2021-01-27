import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 10)

data = yf.download("MSFT AAPL TSLA CRM GOOGL TWTR", start="2019-01-01", end="2019-04-30")
print(data)

apple = yf.Ticker("aapl")

DF = apple.history(start="2021-1-1", end="2021-1-6", interval="1m")
print(DF)

y = DF["High"].values
x = DF[["Low"]].values

nsplit = int(y.shape[0] * .8)
xtrain,ytrain = x[:nsplit,:], y[:nsplit]
xtest,ytest = x[nsplit:,:], y[nsplit:]

kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=None,
        random_state=0).fit(xtrain, ytrain)
score = gpr.score(xtrain, ytrain)  # Score GPR
print(score)
y_bar_hat, y_std = gpr.predict(xtest, return_std=True)
reward_data = {"t": np.arange(ytest.shape[0]), "y": y_bar_hat}

# Specify the intervals
interval_plus = np.add(y_bar_hat, y_std)
interval_minus = np.subtract(y_bar_hat, y_std)

# Create the plot with CI
sns.lineplot(data=reward_data, x="t", y="y")
plt.fill_between(reward_data["t"], interval_plus, interval_minus,
                 color='gray', alpha=0.2)
plt.xlabel("Time")
plt.plot(ytest, color = "r")
plt.ylabel("Price")
plt.title("Apple Stock Price over Time")
plt.show()












