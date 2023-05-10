import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pickle as pickle

from plotly.offline import plot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import pickle as pickle
from statsmodels.tsa.stattools import adfuller
import numpy as np
import itertools as itertools
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
import tqdm as tqdm
import time as time
from sklearn.metrics import mean_absolute_error as mape
from Syntethic_data_prep import *

from sklearn.model_selection import ParameterSampler
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
def alpha_calculation(y, y_hat):
    loss_Arr = []
    alpha_range = np.linspace(-0.3, 0.3, num=67)
    alpha_Arr = []
    alfa_y = np.zeros((1, y.shape[0]))
    for j in tqdm.tqdm(y.index):
        temp_alfa = alpha_range[0]

        y_temp = pd.DataFrame([y.loc[j]])
        y_temp.index = [str(j)]
        y_temp_hat = pd.DataFrame([y_hat.loc[j]])
        y_temp_hat.index = [str(j)]

        temp_loss = l2loss(y_temp, y_temp_hat, [temp_alfa])
        for i in range(len(alpha_range)):

            alpha = alpha_range[i + 1]
            loss = l2loss(y_temp, y_temp_hat, [alpha])

            if loss < temp_loss:
                temp_loss = loss
                temp_alfa = alpha
            else:
                temp_loss = temp_loss
                temp_alfa = temp_alfa
            if i == 65:
                break
        loss_Arr.append(temp_loss)
        alpha_Arr.append(temp_alfa)
    alpha_df = pd.DataFrame(alpha_Arr, index = y.index)
    return alpha_df



def l2loss(y, y_pred, alpha):

    alpha= pd.DataFrame(alpha)
    alpha.index = y.index
    y.columns = ['0']
    ind = y.index
    y_pred_new = ((1+alpha.values)*(y_pred.values))[0][0]
    cost = (y.values-y_pred_new)**2
    return cost


def train_lgb( X_train, X_test, y_train,y_test,param, grid=None):
    if grid:
        print("grid search started")
        START = time.time()
        model = lgb.LGBMRegressor(objective ="mse")
        grid = GridSearchCV(estimator=model, param_grid=param,
                            cv=5, verbose=-1, refit=True)
        grid.fit(X_train, y_train)
        END = time.time()
        print(END - START)
        best_parameters = grid.best_params_
        model = grid.best_estimator_
        best_score = grid.best_score_
        print("Val MAE:",best_score)
        print("grid search ended")
        print("best parameters:", best_parameters)
    else:
        model = lgb.LGBMRegressor(param)
        model.fit(X_train, y_train)
    pred = pd.DataFrame([model.predict(X_test)]).T
    pred = pred.set_index(y_test.index)
    pred_tr = pd.DataFrame([model.predict(X_train)]).T
    pred_tr = pred_tr.set_index(y_train.index)
    mape_score = mape(y_test,pred)
    mape_score_tr = mape(y_train,pred_tr)

    print("Test MAPE:",mape_score)
    print("Train MAPE:",mape_score_tr)
    feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    f, ax = plt.subplots(figsize=(20, 10))
    ax = sns.barplot(x=feature_scores, y=feature_scores.index)
    ax.set_title("Visualize feature scores")
    ax.set_yticklabels(feature_scores.index)
    ax.set_xlabel("Feature importance score")
    ax.set_ylabel("Features")

    plt.show()
    return pred, pred_tr

def scaler_F(train1, test1, scaler):
    scaler = scaler.fit(train1)
    sc_train = scaler.transform(train1)
    sc_test = scaler.transform(test1)
    sc_train = pd.DataFrame(sc_train, index=train1.index, columns=train1.columns)
    sc_test = pd.DataFrame(sc_test, index=test1.index, columns=test1.columns)

    return sc_train, sc_test

if __name__ == "__main__":

    phi = np.array([0.4, 0.3, 0.2, 0.1])
    theta = np.array([0.65, 0.35, 0.3, -0.15, -0.3, ])
    mu = 0
    sigma = 1
    d = 0
    t = 0
    n = 2184
    start_date, end_date = "09/01/2022","12/01/2022"
    #start_date, end_date = "09/01/2022 00:00:00", "09/21/2022 20:00:00"
    dataset1, X_train, X_test, y_train, y_test, X_train2, X_test2 = creeate_lgb_dataset_v2(phi, theta, d, t,mu,sigma, n,
                    "tsgen-2023-05-03 10_59_32.csv",start_date, end_date, [2, 4, 6, 8], autogenerated = False, MA = False)
    #plot_acf_pacf(dataset1['y'])
    result = adfuller(dataset1['y'])
    print('Test Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

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

    entire_grid = [*ParameterSampler(param_grid, 1, random_state=42)]
    entire_grid = [{k: [v] for k, v in d.items()} for d in entire_grid]

    # GR = None
    # with open("lgb_grid_0.pkl", 'rb') as f:
    #     GR = pickle.load(f)
    #concatenate X_train and X_train2, X_test and X_test2
    X_train3 = pd.concat([X_train, X_train2], axis=1)
    X_test3 = pd.concat([X_test, X_test2], axis=1)

    scaler = MinMaxScaler()
    X_train, X_test = scaler_F(X_train, X_test, scaler)
    X_train2, X_test2 = scaler_F(X_train2, X_test2, scaler)
    X_train3, X_test3 = scaler_F(X_train3, X_test3, scaler)
    y_train, y_test = scaler_F(y_train, y_test, scaler)

    first_preds, first_preds_tra  = train_lgb( X_train, X_test, y_train,y_test, entire_grid, grid=True)
    first_preds.columns = ["preds"]

    all_preds, all_preds_tra = train_lgb(X_train3, X_test3, y_train, y_test, entire_grid, grid=True)
    all_preds.columns = ["preds"]


    y = pd.concat([y_train, y_test], axis=0)
    y_hat = pd.concat([first_preds_tra, first_preds], axis=0)
    all_alphas = alpha_calculation(y, y_hat)

    with open("alpha_pkl", "wb") as f:
        pickle.dump(all_alphas, f)

    #all_alphas  = pd.read_pickle("alpha_pkl")
    yy_test = (all_alphas.merge(y_test, left_index=True, right_index=True)).iloc[:,0]
    yy_train = all_alphas.merge(y_train, left_index=True, right_index=True).iloc[:,0]
    #concatenate first_preds to X_train2 and X_test2
    X_train2 = pd.concat([X_train2, first_preds_tra], axis=1)
    X_test2 = pd.concat([X_test2, first_preds], axis=1)
    alpha_preds, alpha_preds_tr  = train_lgb( X_train2, X_test2, yy_train,yy_test, entire_grid, grid=True)

    y_new_test = (1 + alpha_preds.values) * (first_preds.values)
    y_new_train = (1 + alpha_preds_tr.values) * (first_preds_tra.values)

    # y_new_test = (1 + yy_test.values) * (first_preds.values)
    # y_new_train = (1 + yy_train.values) * (first_preds_tra.values)

    plt.plot(y_test.index, yy_test)
    plt.plot(y_test.index, alpha_preds)
    plt.legend(["Ground Truth", "Alpha Prediction"])
    plt.show()

    plt.plot(y_test.index, y_test)
    plt.plot(y_test.index, y_new_test)
    plt.plot(y_test.index, first_preds)
    plt.plot(y_test.index, all_preds)
    plt.legend(["Ground Truth", "Second Layer Prediction", "First Layer Prediction", "All Features Prediction"])
    plt.show()

    # plt.show()
    # fig =  px.line()
    # fig.add_scatter(x=y_test.index, y=y_test, mode='lines', name="Ground Truth")
    # fig.add_scatter(x=all_preds.index, y=all_preds, mode='lines', name="All prediction")
    # fig.add_scatter(x=all_preds.index, y=y_new_test, mode='lines', name="Second Layer Prediction")
    # fig.add_scatter(x=first_preds.index, y=first_preds, mode='lines', name="Prediction (1st Layer)")
    #
    # fig.update_layout(title="ALl Features Pred, Fist layer pred, and Ground Truth", width=1800, showlegend=True)
    # # Show plot 
    # fig.update_traces(line={'width': 2})
    # fig.write_html(file="All_first_and_truth.html")
    #fig.show()
    #
    # trace1 = go.Scatter(
    #     x=y_test.index,
    #     y=y_test,
    #     mode='lines',
    #     name="Ground Truth"
    # )
    #
    # trace2 = go.Scatter(
    #     x=all_preds.index,
    #     y=all_preds,
    #     mode='lines',
    #     name='All prediction'
    # )
    # trace3 = go.Scatter(
    #     x=all_preds.index,
    #     y=y_new_test,
    #     mode='lines',
    #     name="Second Layer Prediction"
    # )
    # trace4 = go.Scatter(
    #     x=all_preds.index,
    #     y=first_preds,
    #     mode='lines',
    #     name="Prediction (1st Layer"
    # )
    # # Layout setting
    # layout = go.Layout(
    #     title=dict(text="Automobile Dataset", x=0.5, y=0.95),
    #     yaxis=dict(title='Mpg', showgrid=False, showline=False),  # y-axis label
    #     paper_bgcolor='#FFFDE7',
    #     plot_bgcolor='#FFFDE7'
    # )
    #
    # data = [trace1, trace2,trace3,trace4]
    # fig = go.Figure(data=data, layout=layout)
    # plot(fig)

print()
