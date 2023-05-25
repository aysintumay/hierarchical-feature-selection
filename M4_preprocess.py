import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from itertools import product
from plotly.offline import plot
from sklearn.metrics import mean_absolute_error as mape
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import pickle as pickle
from sklearn.datasets import make_classification
import math
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from alpha_optimization import  scaler_F,ordinary_ensembele,train_lgb,l2loss,alpha_calculation

import random
random.seed(42)
def plot_acf_pacf(series):
    plt.subplots(figsize=(10, 10))
    plot_pacf(series, lags=20)
    plot_acf(series, lags=20)
    plt.show()
    return None
def create_M4(path_train, path_test):
    data_train = pd.read_csv(path_train, index_col=0).T
    data_test = pd.read_csv(path_test, index_col=0).T

    data_train = (data_train.sample(axis='columns')).dropna()
    data_test = (data_test[data_train.columns]).dropna()

    data = pd.concat([data_train, data_test], axis=0)
    data = data.reset_index()
    data = data.drop(columns=["index"])

    if data.shape[0] <100:
        n_f = 5
    elif 100< data.shape[0] <500:
        n_f = 14
    else:
        n_f = 20

    k = 0.15
    lags  = [2,4]
    dataset = make_classification(data.shape[0], n_features=n_f, n_informative=n_f, n_redundant=0, weights=[0.9], )
    col_names = []
    for i in range(n_f):
        col_names.append("x" + str(i))
    dataset_x = pd.DataFrame(dataset[0]).set_index(data.index)
    dataset_y = pd.DataFrame(dataset[1]).set_index(data.index)
    dataset_x.columns = col_names
    dataset_y.columns = [data.columns]

    X_train, X_test, = dataset_x.iloc[: len(data_train)], dataset_x.iloc[len(data_train):]

    data[dataset_y == 1] = data[dataset_y == 1] * (1 + k)
    data[dataset_y == 0] = data[dataset_y == 0] * (1 - k)
    noise = np.random.normal(0, 1.5, data.shape[0])
    #data = data.diff().add(noise, axis=0)
    #data = (data.add(noise.T)).dropna()
    data = data.diff(periods=2).add(noise, axis=0)
    data = data.dropna()
    for lag in lags:
        # data["y" + '_lag_' + str(lag)] = data["y"].transform(lambda x: x.shift(lag, fill_value=0))
        # data["y" + '_lag_' + str(lag)+"_std"]= data["y" + '_lag_' + str(lag)].std()
        # data["y" + '_lag_' + str(lag) + "_mean"] = data["y" + '_lag_' + str(lag)].mean()
        data["y" + '_lag_' + str(lag) + "_rolling_mean"] = (
            data[data.columns[0]].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag + 1).mean()
        data["y" + '_lag_' + str(lag) + "_rolling_std"] = (
            data[data.columns[0]].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag + 1).std()
    data = data.dropna()
    data_all = dataset_x.merge(data, left_index =True, right_index = True)
    plt.figure(figsize=(10, 10))
    # corrmat = data_all.corr()
    # hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10}, cmap="Spectral_r")
    # plt.show()
    second_set = data_all.iloc[:,:n_f]
    first_set = data_all.iloc[:,n_f:]
    X_train, X_test, y_train, y_test =  first_set.iloc[:-len(data_test), 1:], first_set.iloc[-len(data_test):, 1:], \
        first_set.iloc[:-len(data_test), 0], first_set.iloc[-len(data_test):, 0]


    X_train2, X_test2, = second_set.iloc[:-len(data_test) ], second_set.iloc[-len(data_test):]

    y_train = pd.DataFrame(y_train, index = X_train.index, columns=[data.columns[0]])
    y_test = pd.DataFrame(y_test, index = X_test.index, columns=[data.columns[0]])


    return data_all, X_train, X_test, y_train, y_test, X_train2, X_test2

if __name__ == '__main__':

    mape_first = []
    mape_second = []
    mape_ensemble = []
    mape_all = []
    for i in range(200):
        print("======= ITERATION ========", i)
        data, X_train, X_test, y_train, y_test, X_train2, X_test2 = create_M4("Quarterly-train.csv",  "Quarterly-test.csv")
        #plot_acf_pacf(data[data.columns[0])
        # result = adfuller(data[data.columns[0]])
        # print('Test Statistic: %f' % result[0])
        # print('p-value: %f' % result[1])
        # print('Critical values:')
        # for key, value in result[4].items():
        #     print('\t%s: %.3f' % (key, value))

        X_train3 = pd.concat([X_train, X_train2], axis=1)
        X_test3 = pd.concat([X_test, X_test2], axis=1)

        scaler = MinMaxScaler()
        X_train, X_test = scaler_F(X_train, X_test, scaler)
        X_train2, X_test2 = scaler_F(X_train2, X_test2, scaler)
        X_train3, X_test3 = scaler_F(X_train3, X_test3, scaler)
        y_train, y_test = scaler_F(y_train, y_test, scaler)
        y_train.columns, y_test.columns = ["y"], ["y"]
        param = [{'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_bin': 15, 'max_depth': 7, 'n_estimators': 250,
        'num_leaves': 255, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.9, 'subsample_freq': 1}]
        param = [{k: [v] for k, v in d.items()} for d in param]
        first_preds, first_preds_tra, mape_score_first  = train_lgb(X_train, X_test, y_train,y_test, param, grid=True)
        first_preds.columns = ["preds"]

        mape_first.append(mape_score_first[0])
        param = [{'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_bin': 63, 'max_depth': 10, 'n_estimators': 500,
                 'num_leaves': 63, 'reg_alpha': 0.1, 'reg_lambda': 0, 'subsample': 0.9, 'subsample_freq': 5}]
        param = [{k: [v] for k, v in d.items()} for d in param]
        all_preds, all_preds_tra, mape_score_all = train_lgb(X_train3, X_test3, y_train, y_test, param, grid=True)
        all_preds.columns = ["preds"]

        mape_all.append(mape_score_all[0])
        y = pd.concat([y_train, y_test], axis=0)
        y_hat = pd.concat([first_preds_tra, first_preds], axis=0)
        all_alphas = alpha_calculation(y, y_hat)
        #
        # with open("alpha_pkl", "wb") as f:
        #     pickle.dump(all_alphas, f)
        # all_alphas  = pd.read_pickle("alpha_pkl")
        yy_test = (all_alphas.merge(y_test, left_index=True, right_index=True)).iloc[:, 0]
        yy_train = all_alphas.merge(y_train, left_index=True, right_index=True).iloc[:, 0]
        X_train2 = pd.concat([X_train2, first_preds_tra], axis=1)
        X_test2 = pd.concat([X_test2, first_preds], axis=1)

        param = [
            {'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_bin': 255, 'max_depth': 7, 'n_estimators': 250,
             'num_leaves': 31, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 0.9, 'subsample_freq': 5}]
        param = [{k: [v] for k, v in d.items()} for d in param]
        alpha_preds, alpha_preds_tr, _ = train_lgb(X_train2, X_test2, yy_train, yy_test, param, grid=True)

        y_new_test = (1 + alpha_preds.values) * (first_preds.values)
        y_new_train = (1 + alpha_preds_tr.values) * (first_preds_tra.values)
        mape_score_second = mape(y_test, y_new_test)
        first = mape(y_test, first_preds)
        all = mape(y_test, all_preds)
        mse_pointwise = [((y_test.values - y_new_test) ** 2)]

        mape_second.append(mse_pointwise[0])
        if first - mape_score_second > 0:
            print("second layer is better")
        print(" second is: ", mape_score_second, "\n first is: ", first, "\n all is: ", all)
        param = [
            {'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_bin': 31, 'max_depth': 4, 'n_estimators': 250,
             'num_leaves': 255, 'reg_alpha': 0.1, 'reg_lambda': 0, 'subsample': 0.6, 'subsample_freq': 1}]

        param = [{k: [v] for k, v in d.items()} for d in param]
        second_preds, second_preds_tra, _ = train_lgb(X_train2, X_test2, y_train, y_test, param, grid=True)
        ensemble_train, ensemble_val, ensemble_test, mape_score_ensemble = ordinary_ensembele(first_preds,
                                                                                              first_preds_tra,
                                                                                              second_preds,
                                                                                              second_preds_tra,
                                                                                              y_train, y_test)
        mape_ensemble.append(mape_score_ensemble[0])
    mape_first_mean = pd.DataFrame([np.mean(mape_first, axis=0)][0])

    mape_all_mean = pd.DataFrame([np.mean(mape_all, axis=0)][0])

    mape_second_mean = pd.DataFrame([np.mean(mape_second, axis=0)][0])
    mape_ensemble_mean = pd.DataFrame([np.mean(mape_ensemble, axis=0)][0])
    mape_first_mean_expanding = mape_first_mean.expanding().mean()
    mape_all_mean_expanding = mape_all_mean.expanding().mean()
    mape_second_mean_expanding = mape_second_mean.expanding().mean()
    mape_ensemble_mean_expanding = mape_ensemble_mean.expanding().mean()

    plt.plot(np.arange(mape_first_mean_expanding.shape[0]), mape_first_mean_expanding)
    plt.plot(np.arange(mape_all_mean_expanding.shape[0]), mape_all_mean_expanding)
    plt.plot(np.arange(mape_second_mean_expanding.shape[0]), mape_second_mean_expanding)
    plt.plot(np.arange(mape_ensemble_mean_expanding.shape[0]), mape_ensemble_mean_expanding)
    plt.legend(["y-related Features", "ALl Features Mixed ", "Hierarchical Layer", "Ensemble"])
    plt.title("M4 Dataset Experiment Results")
    plt.show()

print()



