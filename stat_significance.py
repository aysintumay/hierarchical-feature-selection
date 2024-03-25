import numpy as np
from scipy import stats
import pandas as pd



def paired_t_test(mse_predictions, mse_ground_truth):
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(mse_ground_truth, mse_predictions,alternative = "greater")
    return t_statistic, p_value

# Example MSE values for predictions and ground truth
file_names = [
                # "mape_all_mean_expanding_r.csv",
                  "mape_all_mean_expanding_s.csv",
                  # "mape_ensemble_mean_expanding_r.csv",
                  "mape_ensemble_mean_expanding_s.csv",
                  # "mape_first_mean_expanding_r.csv",
                  "mape_first_mean_expanding_s.csv",
                  # "mape_second_mean_expanding_r.csv", "mape_second_mean_expanding_s.csv",
                  # "mape_wrapper_mean_expanding_r.csv",
                  "mape_wrapper_mean_expanding_s.csv"
              ]
for file in file_names:
    mse_ground_truth = pd.read_csv(file)
    mse_predictions = pd.read_csv("mape_second_mean_expanding_s.csv")

    # Perform paired t-test
    t_statistic, p_value = paired_t_test(mse_predictions, mse_ground_truth)

    # Print results
    print(file)
    print("\tPaired t-test results:")
    print("\tT-statistic:", t_statistic)
    print("\tP-value:", p_value)
