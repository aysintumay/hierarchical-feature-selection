import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
import math
from sklearn.model_selection import train_test_split
def ARIMA(phi, theta, d, t, mu, sigma, n, burn=5, init=None, MA=None):
    """ Simulate data from ARMA model (eq. 1.2.4):
    y_t = phi_1*y_{t-1} + ... + phi_p*y_{t-p} + theta_0*epsilon_t + theta_1*epsilon_{t-1} + ... + theta_q*epsilon_{t-q}
    with d unit roots for ARIMA model.
    Arguments:
    phi -- array of shape (p,) or (p, 1) containing phi_1, phi2, ... for AR model
    theta -- array of shape (q) or (q, 1) containing theta_1, theta_2, ... for MA model
    d -- number of unit roots for non-stationary time series
    t -- value deterministic linear trend
    mu -- mean value for normal distribution error term
    sigma -- standard deviation for normal distribution error term
    n -- length time series
    burn -- number of discarded values because series beginns without lagged terms
    Return:
    x -- simulated ARMA process of shape (n, 1)
    """
    # add theta_0 = 1 to theta
    theta = np.append(1, theta)

    # phi -- array of shape (p,)
    p = phi.shape[0]

    # theta -- array of shape (q,1)
    q = theta.shape[0]

    # add error terms (n+q)
    epsilon = np.random.normal(mu, sigma, (n + max(n, q) + burn, 1))

    # create array for returned values
    x = np.zeros((n + max(n, q) + burn, 1))

    # initialize first time series value y_{t-p}=theta_0*epsilon_{t-q}
    x[0] = epsilon[0]

    for i in range(1, x.shape[0]):
        AR = np.dot(phi[0: min(i, p)], np.flip(x[i - min(i, p): i], 0))
        MA = np.dot(theta[0: min(i + 1, q)], np.flip(epsilon[i - min(i, q - 1): i + 1], 0))
        if MA:
            x[i] = AR + MA + t
        else:
            x[i] = AR + t


    # add unit roots
    if d != 0:
        ARMA = x[-n:]
        m = ARMA.shape[0]
        y = np.zeros((m + 1, 1))  # create temp array

        for i in range(d):
            for j in range(m):
                y[j + 1] = ARMA[j] + y[j]
            ARMA = y[1:]
        x[-n:] = y[1:]


    return  x[-n:]


def synthetic_dataset(phi, theta, mu, sigma, dataset_length=1001):
    def take_orhogonal(k):
        y = np.zeros(4)
        y[0] = k[3]
        y[1] = -k[1]
        y[2] = -k[2]
        y[3] = k[4]

        return y

    def calculate_region(line, points):
        if np.dot(line, points) > 0:
            return 1
        else:
            return -1

    line1 = np.random.rand(4)
    line2 = take_orhogonal(line1)

    y = np.zeros(dataset_length)
    e = np.random.normal(mu, sigma, dataset_length)

    w1 = 0.2
    w2 = -0.3
    w3 = 0.8
    w4 = -0.6

    w11 = 0.15
    w22 = -0.35
    w33 = 0.75
    w44 = -0.8

    w111 = -0.4
    w222 = 0.25
    w333 = 0.65
    w444 = 0.70

    w1111 = -0.28
    w2222 = 0.38
    w3333 = -0.70
    w4444 = -0.85

    # y_t+1 = a*y_t + b*y_t-1 + c*e_t + d*e_t-1 + e_t+1

    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0

    for i in range(dataset_length):
        et_1 = np.random.normal(mu, sigma, 1)
        if i - 1 < 0:
            y[0] = 0 + 0 + 0 + 0 + et_1
        elif i - 2 < 0:
            y[1] = y[0] + 0 + 0 + 0 + et_1
        else:
            if (
                    calculate_region(line1, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == 1
                    and calculate_region(line2, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == 1
            ):
                counter1 += 1
                y[i] = w1 * y[i - 1] + w2 * y[i - 2] + w3 * e[i - 1] + w4 * e[i - 2] + et_1

            elif (
                    calculate_region(line1, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == 1
                    and calculate_region(line2, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
            ):
                counter2 += 1
                y[i] = w11 * y[i - 1] + w22 * y[i - 2] + w33 * e[i - 1] + w44 * e[i - 2] + et_1

            elif (
                    calculate_region(line1, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
                    and calculate_region(line2, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
            ):
                counter3 += 1
                y[i] = w111 * y[i - 1] + w222 * y[i - 2] + w333 * e[i - 1] + w444 * e[i - 2] + et_1
            else:
                counter4 += 1
                y[i] = w1111 * y[i - 1] + w2222 * y[i - 2] + w3333 * e[i - 1] + w4444 * e[i - 2] + et_1

    df = pd.DataFrame(y[200:])
    for i in range(phi):
        df[f"y-{i}"] = df["y"].shift(i)
    for j in range(theta):
        df[f"e-{j}"] = df["e"].shift(j)

    return df


def create_ARIMA_data(phi, theta, mu, sigma, t, n, test_size, ):
    y = ARIMA(phi=phi, theta=theta, mu=mu, sigma=sigma, n=n, t=t)
    y_train_new = y[:-test_size]
    y_test_new = y[-test_size:]
    return y_train_new, y_test_new


# def create_dataset_lgb(phi, theta, d, t,mu,sigma, n, lags, start, end, val_ratio=0.2,MA=None, autogenerated = True):
#     if autogenerated:
#         data = pd.read_csv("tsgen-2023-05-03 10_59_32.csv")
#         data.index = pd.date_range(start=start, end=end,tz = None, freq="1H")[:-1]
#         data.columns = ["y"]
#     else:
#         data = pd.DataFrame(ARIMA(phi, theta, d, t,mu,sigma, n,MA))
#
#         data.columns = ["y"]
#         data.index = pd.date_range(start=start, end=end,tz = None, freq="1H")[:-1]
#     data = data.diff()
#     for lag in lags:
#         #data["y" + '_lag_' + str(lag)] = data["y"].transform(lambda x: x.shift(lag, fill_value=0))
#         #data["y" + '_lag_' + str(lag)+"_std"]= data["y" + '_lag_' + str(lag)].std()
#         #data["y" + '_lag_' + str(lag) + "_mean"] = data["y" + '_lag_' + str(lag)].mean()
#         data["y" + '_lag_' + str(lag) + "_rolling_mean"] = (data["y"].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag+1).mean()
#         data["y" + '_lag_' + str(lag) + "_rolling_std"] = (data["y"].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag+1).std()
#     data = data.dropna()
#
#     plt.figure(figsize=(10,10))
#     corrmat = data.corr()
#     hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10}, cmap="Spectral_r")
#     plt.show()
#     X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], test_size=val_ratio,
#                                                         random_state=42, shuffle=False)
#     return data,X_train, X_test, y_train, y_test

def creeate_lgb_dataset_v2(data_path, start, end,lags, val_ratio=0.2):
    data = pd.read_csv(data_path)
    data.index = pd.date_range(start=start, end=end, tz=None, freq="1H")[:-1]
    data.columns = ["y"]
    dataset =make_classification(data.shape[0], n_features = 10, n_informative = 2, n_redundant=7, weights = [0.8],)

    dataset_x = pd.DataFrame(dataset[0]).set_index(data.index)
    dataset_y = pd.DataFrame(dataset[1]).set_index(data.index)

    X_train, X_test, _, _ = train_test_split(dataset_x, dataset_y, test_size=val_ratio, random_state=42, shuffle=False)

    data[dataset_y == 1] = data[dataset_y == 1] * 1.33
    data[dataset_y == 0] = data[dataset_y == 0] * 0.66
    data = data.diff()
    for lag in lags:
        # data["y" + '_lag_' + str(lag)] = data["y"].transform(lambda x: x.shift(lag, fill_value=0))
        # data["y" + '_lag_' + str(lag)+"_std"]= data["y" + '_lag_' + str(lag)].std()
        # data["y" + '_lag_' + str(lag) + "_mean"] = data["y" + '_lag_' + str(lag)].mean()
        data["y" + '_lag_' + str(lag) + "_rolling_mean"] = (
            data["y"].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag + 1).mean()
        data["y" + '_lag_' + str(lag) + "_rolling_std"] = (
            data["y"].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag + 1).std()
    data = data.dropna()
    data_all = dataset_x.merge(data, left_index =True, right_index = True)
    plt.figure(figsize=(10, 10))
    corrmat = data_all.corr()
    hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10}, cmap="Spectral_r")
    plt.show()
    second_set = data_all.iloc[:,:10]
    first_set = data_all.iloc[:,10:]
    X_train, X_test, y_train, y_test = train_test_split(first_set.iloc[:,1:], first_set.iloc[:, 0], test_size=val_ratio,
                                                        random_state=42, shuffle=False)
    X_train2, X_test2, = train_test_split(second_set, test_size=val_ratio,
                                                        random_state=42, shuffle=False)
    return data, X_train, X_test, y_train, y_test, X_train2, X_test2


# def creeate_lgb_dataset_v2(data, val_ratio=0.2):
#     """
#     :return: time related feature columns for the 2nd hierarchial layer
#     """
#     def sc_transform(c):
#         max_val = c.max()
#         sin_values = [math.sin((2 * math.pi * x) / max_val) for x in list(c)]
#         cos_values = [math.cos((2 * math.pi * x) / max_val) for x in list(c)]
#         return sin_values, cos_values
#
#     data_new =data.iloc[:,0]
#     data_new.index = pd.to_datetime(data.index )
#     data_new.columns = ["y"]
#     data_new['sin_hour'], data_new['cos_hour'] = sc_transform(pd.DatetimeIndex(data_new.index).hour)
#     data_new['sin_dayofweek'], data_new['cos_dayofweek'] = sc_transform(pd.DatetimeIndex(data_new.index).dayofweek)
#     data_new['sin_dayofyear'], data_new['cos_dayofyear'] = sc_transform(pd.DatetimeIndex(data_new.index).dayofyear)
#     data_new['sin_dayofmonth'], data_new['cos_dayofmonth'] = sc_transform(pd.DatetimeIndex(data_new.index).day)
#     data_new['sin_weekofyear'], data_new['cos_weekofyear'] = sc_transform(pd.DatetimeIndex(data_new.index).isocalendar().week)
#     data_new = data_new.drop(columns =["y"])
#     X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:, 1:], data_new.iloc[:, 0], test_size=val_ratio,
#                                                         random_state=42, shuffle=False)
#     return  X_train, X_test, y_train, y_test
# phi = np.array([0.4, 0.3, 0.2, 0.1])
# theta = np.array([0.65, 0.35, 0.3, -0.15, -0.3, ])
# mu = 0
# sigma = 1
# d = 0
# t = 0
# n = 2184
# start_date, end_date = "09/01/2022","12/01/2022"
# start_date, end_date = "09/01/2022 00:00:00", "09/21/2022 20:00:00"
# #
# data, X_train, X_test, y_train, y_test, X_train2, X_test2 = creeate_lgb_dataset_v2("tsgen-2023-05-03 10_59_32.csv",start_date, end_date, [2, 3, 4, 5])
# # X_train, X_test, y_train, y_test,data = creeate_essential_lgb_dataset(dataset_y, "tsgen-2023-05-03 10_59_32.csv", start_date, end_date, [2, 3, 4, 5])

print()


