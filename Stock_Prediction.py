import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import tslearn as ts

pd.set_option('display.max_columns', 10)

def tslr(algo="svr"):
    DF = yf.download("MSFT AAPL TSLA CRM GOOGL TWTR", start="2019-01-01", end="2019-04-30")
    STOCK_NAMES = ["MSFT","AAPL","TSLA", "CRM", "GOOGL","TWTR"]
    metric = dict()
    for stock in STOCK_NAMES:
        x = DF[("Open", stock)].values.reshape((-1, 1))
        y = DF[("Close", "AAPL")].values

        nsplit = int(y.shape[0] * .8)
        xtrain,ytrain = x[:nsplit], y[:nsplit]
        xtest,ytest = x[nsplit:], y[nsplit:]

        if algo=="svr":
            clf = TimeSeriesSVR(C=1.0, kernel="gak")
            clf.fit(xtrain, ytrain)
            model=clf
            y_bar_hat = clf.predict(xtest)

        if algo=="gpr":
            kernel = DotProduct() + WhiteKernel()
            gpr = GaussianProcessRegressor(kernel=kernel,
                    random_state=0).fit(xtrain, ytrain)
            score = gpr.score(xtrain, ytrain)  # Score GPR
            model = gpr

            # Predict
            y_bar_hat, y_std = gpr.predict(xtest, return_std=True)

            # Specify the intervals
            interval_plus = np.add(y_bar_hat, y_std)
            interval_minus = np.subtract(y_bar_hat, y_std)

        reward_data = {"t": np.arange(ytest.shape[0]), "y": y_bar_hat}


        # Create the plot with CI
        plt.plot(reward_data['y'], color = 'b', label = 'Predicted')
        if algo == "gpr":  # Plot only if we have std_dev available
            plt.fill_between(reward_data["t"], interval_plus, interval_minus,
                             color='gray', alpha=0.2)
        plt.xlabel("Time")
        plt.plot(ytest, color = "r", label = 'Ground Truth')
        plt.ylabel("Price")
        plt.title("Apple Closing Price from {} Opening Price, Algorithm = {}".format(stock, algo))
        plt.legend()
        plt.show()

        xs = np.arange(x.size)
        if algo == "svr":
            y_r_Pred = clf.predict(x)
        if algo == "gpr":
            y_r_Pred, y_r_std = gpr.predict(x, return_std=True)

            # Specify the intervals
            interval_plus = np.add(y_r_Pred, y_r_std)
            interval_minus = np.subtract(y_r_Pred, y_r_std)

        # Create the plot with CI
        plt.plot(y_r_Pred, color = 'b', label = 'Predicted')
        if algo == "gpr":
            plt.fill_between(xs, interval_plus, interval_minus,
                             color='gray', alpha=0.2)
        plt.xlabel("Time")
        plt.plot(y, color = "r", label = 'Ground Truth')
        plt.ylabel("Price")
        plt.title("Apple Closing Price from {} Opening Price, Algorithm = {}".format(stock, algo))
        plt.legend()
        plt.show()

        train_rmse = np.sqrt(mse(ytrain, y_r_Pred[:nsplit]))
        test_rmse = np.sqrt(mse(ytest, y_bar_hat))
        metric[stock] = {"train":train_rmse, "test":test_rmse}

    return metric

svr = tslr(algo="svr")
gpr = tslr(algo="gpr")












