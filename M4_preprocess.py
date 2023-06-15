import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

from sklearn.model_selection import ParameterSampler
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
def train_sarimax(X_train,X_test, y_train, y_test, ):
    model_autoarima = auto_arima(y_train, exog=X_train,
                                 start_p=0, start_q=0, start_Q=0, max_P=5, max_Q=5,
                                 test='kpss',
                                 max_p=5, max_q=5, m=7,
                                 start_P=0, seasonal=True,
                                 d=None, D=None,
                                 trace=False,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)
    # m = 7 as data contains daily observations

    sarimax_model = SARIMAX(y_train, exog=X_train, order=model_autoarima.get_params().get("order"),
                            seasonal_order=model_autoarima.get_params().get("seasonal_order"),
                        enforce_invertibility=False).fit()
    start = len(X_train)
    end = len(X_train) + len(X_test) - 1
    predictions = sarimax_model.predict(start=start, end=end, exog=X_test, dynamic=False).rename('SARIMAX_preds')
    predictions_tra = sarimax_model.predict(start=0, end=start-1, exog=X_train, dynamic=False).rename('SARIMAX_preds')


    mape_score = mape(y_test, predictions)
    return predictions, predictions_tra, mape_score

def add_date_features(df):
    #add date index to the dataframe with the same length as th dataframe and add the following features
    #day of the week, day of the month, month of the year, week of the year, quarter of the year
    #create a date column
    date = pd.date_range(end='1/1/2023', periods=len(df), freq='D')
    #create a dataframe with the date column

    df = pd.DataFrame(df)
    df['day_of_week'] = pd.DatetimeIndex(date).dayofweek
    df['day_of_month'] = pd.DatetimeIndex(date).day
    df['week_of_year'] = pd.DatetimeIndex(date).weekofyear
    df['quarter_of_year'] = pd.DatetimeIndex(date).quarter


    def sc_transform(c):
        max_val = c.max()
        sin_values = [math.sin((2 * math.pi * x) / max_val) for x in list(c)]
        cos_values = [math.cos((2 * math.pi * x) / max_val) for x in list(c)]
        return sin_values, cos_values

    df["month"] = pd.DatetimeIndex(date).month-1

    df["month_sin"], df["month_cos"] = sc_transform(pd.DatetimeIndex(date).month -1)

    df["day_of_month_sin"], df["day_of_month_cos"] = sc_transform(pd.DatetimeIndex(date).day )
    df["weekday_sin"], df["weekday_cos"] = sc_transform(pd.DatetimeIndex(date).weekday)
    df["week_of_the_year_sin"], df["week_of_the_year_cos"] = sc_transform(pd.DatetimeIndex(date).isocalendar().week -1)

    df["season_of_the_year_sin"], df["season_of_the_year_cos"] = sc_transform(pd.DatetimeIndex(date).month % 12 // 3)
    df.index= pd.DatetimeIndex(date)
    #drop the first and second columns
    df = df.drop(columns=["y"])
    return df


def create_M4(path_train, path_test):
    data_train = pd.read_csv(path_train, index_col=0).T
    data_test = pd.read_csv(path_test, index_col=0).T

    data_train = (data_train.sample(axis='columns')).dropna()
    data_test = (data_test[data_train.columns]).dropna()

    data = pd.concat([data_train, data_test], axis=0)
    data = data.reset_index()
    data = data.drop(columns=["index"])
    data.columns = ["y"]

    lags  = [2,4,6]

    data = data.diff().dropna()

    for lag in lags:
        # data["y" + '_lag_' + str(lag)] = data["y"].transform(lambda x: x.shift(lag, fill_value=0))
        # data["y" + '_lag_' + str(lag)+"_std"]= data["y" + '_lag_' + str(lag)].std()
        # data["y" + '_lag_' + str(lag) + "_mean"] = data["y" + '_lag_' + str(lag)].mean()
        data["y" + '_lag_' + str(lag) + "_rolling_mean"] = (
            data[data.columns[0]].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag + 1).mean()
        data["y" + '_lag_' + str(lag) + "_rolling_std"] = (
            data[data.columns[0]].transform(lambda x: x.shift(lag, fill_value=0))).rolling(lag + 1).std()
    data = data.dropna()

    dataset_2 = add_date_features(data["y"])
    data.index = dataset_2.index

    X_train, X_test, y_train, y_test =  data.iloc[:-len(data_test), 1:], data.iloc[-len(data_test):, 1:], \
        data.iloc[:-len(data_test), 0], data.iloc[-len(data_test):, 0]
    y_train,y_test = pd.DataFrame(y_train,columns=["y"]),pd.DataFrame(y_test,columns=["y"])
    param = [{'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_bin': 15, 'max_depth': 7, 'n_estimators': 250,
              'num_leaves': 255, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.9, 'subsample_freq': 1}]
    param = [{k: [v] for k, v in d.items()} for d in param]
    first_preds, first_preds_tra, mape_score_first = train_lgb(X_train, X_test, y_train, y_test, param, grid=True)
    first_preds.columns = ["LGB_preds"]
    first_preds_tra.columns = ["LGB_preds"]
    first_preds.index = X_test.index
    first_preds_tra.index = X_train.index

    #train a SARIMAX model

    predictions, predictions_tra, mape_score_second = train_sarimax(X_train, X_test, y_train, y_test)
    predictions = pd.DataFrame(predictions)
    predictions_tra = pd.DataFrame(predictions_tra)

    X_train = pd.concat([X_train, first_preds_tra,predictions_tra], axis=1)
    X_test = pd.concat([X_test, first_preds,predictions], axis=1)

    y_train = pd.DataFrame(y_train, index = X_train.index, columns=[data.columns[0]])
    y_test = pd.DataFrame(y_test, index = X_test.index, columns=[data.columns[0]])


    X_train2, X_test2, = dataset_2.iloc[:-len(data_test), :], dataset_2.iloc[-len(data_test):, :]
    #concatenate train and test data
    data_all_y = pd.concat([y_train, y_test], axis=0)
    data_all_x = pd.concat([X_train, X_test], axis=0)
    data_all_x2 = pd.concat([X_train2, X_test2], axis=0)
    data_all = pd.concat([data_all_y, data_all_x,data_all_x2], axis=1)
    # plt.figure(figsize=(8, 8))
    # corrmat = data_all.corr()
    # hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 7}, cmap="Spectral_r")
    # plt.title("Correlation Matrix", fontsize=10)
    # plt.show()
    # plt.savefig("corr_matrix_m4.png")

    return data_all, X_train, X_test, y_train, y_test, X_train2, X_test2

def grid_param_create():
    param_grid = {"n_estimators": [100, 250, 500, 750, 1000],
                  "learning_rate": [0.001, 0.01, 0.1, 0.5],
                  "num_leaves": [31, 63, 127, 255],
                  "max_depth": [4, 6, 7, 8, 10],
                  "subsample": [0.6, 0.8, 0.9],
                  "subsample_freq": [1, 5],
                  "colsample_bytree": [0.6, 0.8, 0.9],
                  "reg_alpha": [0, 0.1, 1, 10],
                  "reg_lambda": [0, 0.1, 1, 10],
                  "max_bin": [15, 31, 63, 127, 255], },

    entire_grid = [*ParameterSampler(param_grid, 999, random_state=42)]
    entire_grid = [{k: [v] for k, v in d.items()} for d in entire_grid]

    return entire_grid


if __name__ == '__main__':

    mape_first = []
    mape_second = []
    mape_ensemble = []
    mape_all = []
    for i in range(100):
        print("======= ITERATION ========", i)
        data, X_train, X_test, y_train, y_test, X_train2, X_test2 = create_M4("Daily-train.csv",  "Daily-test.csv")
        # plot_acf_pacf(data[data.columns[0]])
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

        #FIRST PREDICTORS
        param = [{'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_bin': 31, 'max_depth': 4, 'n_estimators': 250,
                  'num_leaves': 255, 'reg_alpha': 0.1, 'reg_lambda': 0, 'subsample': 0.6, 'subsample_freq': 1}]
        param = [{k: [v] for k, v in d.items()} for d in param]
        first_preds, first_preds_tra, mape_score_first  = train_lgb(X_train, X_test, y_train,y_test, param, grid=True)
        first_preds.columns = ["preds"]
        #ALL PREDICTORS
        mape_first.append(mape_score_first[0])
        param = [{'colsample_bytree': 0.1, 'learning_rate': 0.1, 'max_bin': 63, 'max_depth': 10, 'n_estimators': 120,
                 'num_leaves': 63, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 0.9, 'subsample_freq': 5}]
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
        # ALPHA PREDICTION
        param = [
            {'colsample_bytree': 0.3, 'learning_rate': 0.01, 'max_bin': 63, 'max_depth': 10, 'n_estimators': 255,
             'num_leaves': 63, 'reg_alpha': 0.1, 'reg_lambda': 0, 'subsample': 0.9, 'subsample_freq': 5}]
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
            {'colsample_bytree': 0.1, 'learning_rate': 0.1, 'max_bin': 31, 'max_depth': 4, 'n_estimators': 125,
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

    plt.plot(np.arange(mape_first_mean_expanding.shape[0]), mape_first_mean_expanding,"--", color="red")
    plt.plot(np.arange(mape_all_mean_expanding.shape[0]), mape_all_mean_expanding,"--", color="green")
    plt.plot(np.arange(mape_second_mean_expanding.shape[0]), mape_second_mean_expanding, color="black")
    plt.plot(np.arange(mape_ensemble_mean_expanding.shape[0]), mape_ensemble_mean_expanding, "-.", color="blue")
    plt.legend(["y-related Features", "All Features", "Hierarchical", "Ensemble"])
    plt.title("M4 Dataset Experiment Results")
    plt.ylabel("Mean Square Error")
    plt.xlabel("Data Points")
    plt.grid("on")
    plt.show()
    plt.savefig("M4_Dataset_Experiment_Results.png")
    plt.close()
    plt.plot(np.arange(len(all_alphas)), np.array(all_alphas))
    plt.plot(np.arange(len(all_alphas)), np.concatenate([alpha_preds_tr, alpha_preds], axis=0))
    plt.legend(["Ground Truth", "Alpha Prediction"])
    plt.title("Alpha Prediction and Ground Truth")
    plt.ylabel("Alpha Values")
    plt.xlabel("Data Points")
    plt.grid("on")
    plt.show()
    plt.savefig("M4_Dataset_ALPHA.png")
    plt.close()
    plt.plot(y_test.index, y_test)
    plt.plot(y_test.index, y_new_test.reshape(-1,1))
    plt.plot(y_test.index, first_preds)
    plt.plot(y_test.index, all_preds)
    plt.plot(y_test.index, ensemble_test)
    plt.legend(["Ground Truth", "Hierarchical", "y-related","All Features","Ensemble"])
    plt.title("All Predicted Layers and Ground Truth")
    plt.ylabel("Values")
    plt.xlabel("Date")
    plt.grid("on")
    plt.savefig("M4_all_predictions.png")

    plt.show()
print()



