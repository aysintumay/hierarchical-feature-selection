import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from itertools import product
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
    alpha_range_pos = np.linspace(0, 0.33, num=30)
    alpha_range_neg = np.linspace(-0.33, 0, num=30)

    alpha_Arr = []
    for j in tqdm.tqdm(y.index):
        y_temp = pd.DataFrame([y["y"].loc[j]])
        y_temp.index = [str(j)]
        y_temp_hat = pd.DataFrame([y_hat["y"].loc[j]])
        y_temp_hat.index = [str(j)]
        if y_temp.values < y_temp_hat.values:
            alpha_range = alpha_range_neg
            temp_alfa = alpha_range[0]
        else:
            alpha_range = alpha_range_pos
            temp_alfa = alpha_range[0]
        temp_loss = l2loss(y_temp, y_temp_hat, [temp_alfa])
        for i in range(len(alpha_range)):

            alpha = alpha_range[i + 1]
            loss = l2loss(y_temp, y_temp_hat, [alpha])

            if loss < temp_loss:
                temp_loss = loss
                temp_alfa = alpha

            if i == 28:
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
    cost = abs(y.values-y_pred_new)
    return cost


def train_lgb( X_t, X_te, y_t,y_te,param, grid=None):
    if grid:
        print("grid search started")
        START = time.time()
        model = lgb.LGBMRegressor(objective ="mse")
        grid = GridSearchCV(estimator=model, param_grid=param,
                            cv=5, verbose=-1, refit=True)
        grid.fit(X_t, y_t)
        END = time.time()
        print(END - START)
        best_parameters = grid.best_params_
        model = grid.best_estimator_
        best_score = grid.best_score_
        print("Val MAE:",best_score)
        print("grid search ended")
        print("best parameters:", best_parameters)
    else:
        model = lgb.LGBMRegressor(*param)
        model.fit(X_t, y_t)
    pred = pd.DataFrame([model.predict(X_te)]).T
    pred = pred.set_index(y_te.index)

    pred_tr = pd.DataFrame([model.predict(X_t)]).T
    pred_tr = pred_tr.set_index(y_t.index)

    pred.columns, pred_tr.columns = ["y"], ["y"]
    mape_score = mape(y_te,pred)
    mape_score_tr = mape(y_t,pred_tr)

    print("\t Test MAPE:",mape_score)
    print("Train MAPE:",mape_score_tr)
    mse_pointwise = [((y_te.values - pred.values) ** 2)]
    feature_scores = pd.Series(model.feature_importances_, index=X_t.columns).sort_values(ascending=False)
    # f, ax = plt.subplots(figsize=(20, 10))
    # ax = sns.barplot(x=feature_scores, y=feature_scores.index)
    # ax.set_title("Visualize feature scores")
    # ax.set_yticklabels(feature_scores.index)
    # ax.set_xlabel("Feature importance score")
    # ax.set_ylabel("Features")
    #
    # plt.show()
    return pred, pred_tr, mse_pointwise
def ordinary_ensembele(first_p, first_p_tr, second_p, second_p_tr, y_train, y_test):

    first_p_val = first_p_tr
    first_p_tr = first_p_tr.iloc[:int(len(first_p_tr)*0.8)]
    first_p_val = first_p_val.iloc[int(len(first_p_val)*0.8):]
    val = pd.merge(first_p_val, second_p_tr,left_index=True, right_index=True).merge(y_train,left_index=True, right_index=True)
    train = pd.merge(first_p_tr, second_p_tr,left_index=True, right_index=True).merge(y_train,left_index=True, right_index=True)
    second_p_val = pd.DataFrame(val.iloc[:, 1])
    second_p_tr = pd.DataFrame(train.iloc[:, 1])
    second_p.columns = ["preds"]

    y_val = pd.DataFrame(val.iloc[:, 2])
    y_train = pd.DataFrame(train.iloc[:, 2])
    second_p_val.columns, second_p_tr.columns, y_val.columns, y_train.columns = ["y"], ["y"], ["y"], ["y"]
    alphas = np.linspace(0, 1, num=30)
    best_alpha = 0
    best_mape = 0
    for a in alphas:
        all_pr_val = a * first_p_val + (1-a) * second_p_val
        mape_score_val = mape(y_val, all_pr_val)
        if best_mape>mape_score_val:
            best_mape = mape_score_val
            best_alpha = a
    all_pr_test = a * first_p + (1 - a) * second_p
    all_pr_tr = a * first_p_tr + (1 - a) * second_p_tr
    all_pr_val = a * first_p_val + (1 - a) * second_p_val
    mape_score = mape(y_test, all_pr_test)
    mape_score_tr = mape(y_train, all_pr_tr)
    mse_pointwise = [((y_test.values - all_pr_test.values) ** 2)]
    print("CHOSEN ALPHA IS ", best_alpha)
    print("\t Test MAPE:", mape_score)
    print("\t Train MAPE:", mape_score_tr)
    print("\t Validation MAPE:", best_mape)
    return all_pr_tr,all_pr_val, all_pr_test,mse_pointwise
def scaler_F(train1, test1, scaler):
    scaler = scaler.fit(train1)
    sc_train = scaler.transform(train1)
    sc_test = scaler.transform(test1)
    sc_train = pd.DataFrame(sc_train, index=train1.index, columns=train1.columns)
    sc_test = pd.DataFrame(sc_test, index=test1.index, columns=train1.columns)

    return sc_train, sc_test

if __name__ == "__main__":
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
    phi = np.array([0.4, 0.3, 0.2, 0.1])
    theta = np.array([0.65, 0.35, 0.3, -0.15, -0.3, ])
    mu = 0
    sigma = 1
    d = 0
    t = 0
    n = 2184
    #start_date, end_date = "09/01/2022","12/01/2022"
    start_date, end_date = "09/01/2022 00:00:00", "09/21/2022 20:00:00"
    tunable_params = dict(n_f = [26, 28,30], k =[0.05, 0.1,0.15,0.2, 0.25])
    param_values = [v for v in tunable_params.values()]

    #for n_f, k in product(*param_values):
    mape_first = []
    mape_second = []
    mape_ensemble = []
    mape_all = []
    for i in range(100):
        print("======= ITERATION ========", i)
        dataset1, X_train, X_test, y_train, y_test, X_train2, X_test2 = creeate_lgb_dataset_v2(phi, theta, d, t,mu,sigma, n,
                        "tsgen-2023-05-03 10_59_32.csv",start_date, end_date, [2, 4, 6, 8], 26, 0.05, autogenerated = True, MA = False)
        #plot_acf_pacf(dataset1['y'])
        # result = adfuller(dataset1['y'])
        # print('Test Statistic: %f' % result[0])
        # print('p-value: %f' % result[1])
        # print('Critical values:')
        # for key, value in result[4].items():
        #     print('\t%s: %.3f' % (key, value))
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
        y_train.columns, y_test.columns = ["y"], ["y"]
        param = [{'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_bin': 15, 'max_depth': 7, 'n_estimators': 250,
        'num_leaves': 255, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.9, 'subsample_freq': 1}]
        param = [{k: [v] for k, v in d.items()} for d in param]
        first_preds, first_preds_tra, mape_score_first  = train_lgb( X_train, X_test, y_train,y_test, param, grid=True)
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

        with open("alpha_pkl", "wb") as f:
            pickle.dump(all_alphas, f)

        #all_alphas  = pd.read_pickle("alpha_pkl")
        yy_test = (all_alphas.merge(y_test, left_index=True, right_index=True)).iloc[:,0]
        yy_train = all_alphas.merge(y_train, left_index=True, right_index=True).iloc[:,0]
        #concatenate first_preds to X_train2 and X_test2
        X_train2 = pd.concat([X_train2, first_preds_tra], axis=1)
        X_test2 = pd.concat([X_test2, first_preds], axis=1)

        param =  [{'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_bin': 255, 'max_depth': 7, 'n_estimators': 250,
                  'num_leaves': 31, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 0.9, 'subsample_freq': 5}]
        param = [{k: [v] for k, v in d.items()} for d in param]
        alpha_preds, alpha_preds_tr, _  = train_lgb( X_train2, X_test2, yy_train,yy_test, param, grid=True)

        y_new_test = (1 + alpha_preds.values) * (first_preds.values)
        y_new_train = (1 + alpha_preds_tr.values) * (first_preds_tra.values)
        mape_score_second = mape(y_test, y_new_test)
        first = mape(y_test, first_preds)
        all = mape(y_test,  all_preds)
        mse_pointwise = [((y_test.values - y_new_test) ** 2)]

        mape_second.append(mse_pointwise[0])
        if first -mape_score_second>0:
            print("second layer is better")
        print(" second is: ",mape_score_second,"\n first is: ", first, "\n all is: ", all)
        param = [ {'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_bin': 31, 'max_depth': 4, 'n_estimators': 250,
                   'num_leaves': 255, 'reg_alpha': 0.1, 'reg_lambda': 0, 'subsample': 0.6, 'subsample_freq': 1}]

        param = [{k: [v] for k, v in d.items()} for d in param]
        second_preds, second_preds_tra, _ = train_lgb(X_train2, X_test2, y_train, y_test, param, grid=True)
        ensemble_train, ensemble_val, ensemble_test, mape_score_ensemble = ordinary_ensembele(first_preds, first_preds_tra, second_preds,
                                                                         second_preds_tra, y_train, y_test)
        mape_ensemble.append(mape_score_ensemble[0])
    mape_first_mean = pd.DataFrame([np.mean(mape_first, axis = 0)][0])

    mape_all_mean = pd.DataFrame([np.mean(mape_all, axis = 0)][0])

    mape_second_mean = pd.DataFrame([np.mean(mape_second, axis = 0)][0])
    mape_ensemble_mean = pd.DataFrame([np.mean(mape_ensemble, axis = 0)][0])
    mape_first_mean_expanding = mape_first_mean.expanding().mean()
    mape_all_mean_expanding = mape_all_mean.expanding().mean()
    mape_second_mean_expanding = mape_second_mean.expanding().mean()
    mape_ensemble_mean_expanding = mape_ensemble_mean.expanding().mean()

    plt.plot(np.arange(mape_first_mean_expanding.shape[0]), mape_first_mean_expanding)
    plt.plot(np.arange(mape_first_mean_expanding.shape[0]), mape_all_mean_expanding)
    plt.plot(np.arange(mape_first_mean_expanding.shape[0]), mape_second_mean_expanding)
    plt.plot(np.arange(mape_first_mean_expanding.shape[0]), mape_ensemble_mean_expanding)
    plt.legend(["y-related Features", "ALl Features Mixed ", "Hierarchical Layer", "Ensemble"])
    plt.title("Syntethic Dataset Experiment Results")
    plt.show()

    plt.plot(np.arange(len(all_alphas)), all_alphas)
    plt.plot(np.arange(len(all_alphas)), np.concatenate([alpha_preds_tr, alpha_preds], axis=0))
    plt.legend(["Ground Truth", "Alpha Prediction"])
    plt.show()

    plt.plot(y_test.index, y_test)
    plt.plot(y_test.index, y_new_test)
    plt.plot(y_test.index, first_preds)
    #plt.plot(y_test.index, all_preds)
    plt.legend(["Ground Truth", "Second Layer Prediction", "First Layer Prediction",])
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
