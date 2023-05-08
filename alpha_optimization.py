import pandas as pd
import math
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot

from functools import partial
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import pickle as pickle
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import numpy as np
import itertools as itertools
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
import tqdm as tqdm
import time as time
from sklearn.metrics import mean_absolute_error as mape
from Syntethic_data_prep import creeate_lgb_dataset_v2

from sklearn.model_selection import ParameterSampler
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
def alpha_calculation(y, y_hat, x, smf_ptf):
    loss_Arr = []
    alpha_range = np.linspace(-0.33, 0.33, num=67)
    alpha_Arr = []
    alfa_y = np.zeros((1, y.shape[0]))
    for j in tqdm.tqdm(y.index):
        temp_alfa = alpha_range[0]

        y_temp = pd.DataFrame([y.loc[j]])
        y_temp.index = [str(j)]
        y_temp_hat = pd.DataFrame([y_hat.loc[j]])
        y_temp_hat.index = [str(j)]

        temp_loss = l1loss(y_temp, y_temp_hat, [temp_alfa], smf_ptf) ** 2
        for i in range(len(alpha_range)):

            alpha = alpha_range[i + 1]
            loss = l1loss(y_temp, y_temp_hat, [alpha], smf_ptf) ** 2

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
    return alpha_Arr


def l1loss(y, y_pred, alpha, df_temp):
    alpha= pd.DataFrame(alpha)

    alpha.index = y.index
    y = pd.DataFrame(y)
    y.columns = ['0']
    ind = y.index
    df_temp = df_temp.loc[ind]
    df_temp["undershoot_cost"] = df_temp[["smf", "ptf"]].apply(min, axis=1) * 0.97 - df_temp["ptf"]
    df_temp["overshoot_cost"] = (df_temp[["smf", "ptf"]].apply(max, axis=1) * 1.03 - df_temp["ptf"])
    df_temp["q_t"]=y.values
    df_temp["(1+alf)*f_t"]=((1+alpha)*pd.DataFrame(y_pred).values).values
    df_temp["q_t-f_t"] = (df_temp["q_t"]-df_temp["(1+alf)*f_t"]).values
    df_temp["q_t-f_t_boolean"]=(df_temp["q_t-f_t"]>0).astype(int)
    df_temp["cost_under"]=df_temp["q_t-f_t"].loc[df_temp["q_t-f_t_boolean"]!=0]*df_temp["undershoot_cost"].loc[df_temp["q_t-f_t_boolean"]!=0]

    df_temp["cost_over"]=df_temp["q_t-f_t"].loc[df_temp["q_t-f_t_boolean"]==0]*df_temp["overshoot_cost"].loc[df_temp["q_t-f_t_boolean"]==0]
    df_temp=df_temp.fillna(0)

    unitcost=(df_temp["cost_under"].astype("float")+df_temp["cost_over"].astype("float")).sum()/(df_temp["q_t"].astype("float")).sum()
    return unitcost


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
        model = lgb.LGBMRegressor(param, objective ="mse")
        model.fit(X_train, y_train)
    pred = pd.DataFrame([model.predict(X_test)]).T
    pred = pred.set_index(y_test.index)

    mape_score = mape(y_test,pred)
    print("Test MAPE:",mape_score)

    return pred

if __name__ == "__main__":

    # TODO: Put grid search into modular mode
    phi = np.array([0.4, 0.3, 0.2, 0.1])
    theta = np.array([0.65, 0.35, 0.3, -0.15, -0.3, ])
    mu = 0
    sigma = 1
    d = 0
    t = 0
    n = 2184
    #start_date, end_date = "09/01/2022","12/01/2022"
    start_date, end_date = "09/01/2022 00:00:00", "09/21/2022 20:00:00"
    dataset1, X_train, X_test, y_train, y_test, X_train2, X_test2 = creeate_lgb_dataset_v2("tsgen-2023-05-03 10_59_32.csv",start_date, end_date, [2, 3, 4, 5])

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


    first_preds = train_lgb( X_train, X_test, y_train,y_test, entire_grid, grid=True)
    fig = px.line(first_preds, x=first_preds.index, y="preds")
    fig.add_scatter(x=y_test.index, y=y_test['preds'], mode='lines', name="Prediction (1st Layer)",
                    text=f"Test results Unit Cost=-73.292", )

    fig.update_layout(title="First Layer Predictons,  and Ground Truth", width=1800, showlegend=True)
    # Show plot 
    fig.update_traces(line={'width': 2})
    fig.write_html(file="First_preds_and_truth.html")

print()
