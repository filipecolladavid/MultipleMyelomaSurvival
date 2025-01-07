import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the parent directory to the Python path if not already added
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# %%
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer

from scripts.gradient_descent import CustomLinearRegression

from scripts.evaluation import evaluate_model, compare_models, cMSE_error
from scripts.models import (
    evaluate_knn_imputation, 
    evaluate_iterative_imputation, 
    create_polynomial_model, 
    cross_val_polynomial, 
    cross_val_knn, 
    create_KNN_model, 
    run_imputation_baseline_model,
    create_baseline_model
)
from scripts.plots import (
    plot_mse_per_degree,
    plot_best_degrees,
    plot_feature_complexity,
    plot_mse_metric_weight,
    plot_y_yhat
)
from scripts.utils import (
    save_model,
    load_model,
    load_kaggle_df,
    load_starting_df,
    load_train_test,
    create_submission_csv,
    missing_columns
)


# Load data
startingDF = load_starting_df()
kaggleDF = load_kaggle_df()
kf = KFold(n_splits=10, shuffle=True)
task3_results = []


# %% [markdown]
# ## Task 3 - Handling missing Data
# We now add to the data used in Task 2 the data points where the features have missing data. We still cannot take advantage of the unlabeled data, as our ML task is regression, a supervised learning task.

# %%
X_train, X_test, y_train, y_test = load_train_test()

X_train_reduced = X_train.drop(columns=missing_columns)
X_test_reduced = X_test.drop(columns=missing_columns)

X_train_reduced = X_train_reduced[y_train["SurvivalTime"].notna()]
y_train_reduced = y_train[y_train["SurvivalTime"].notna()]

X_test_reduced = X_test_reduced[y_test["SurvivalTime"].notna()]
y_test_reduced = y_test[y_test["SurvivalTime"].notna()]

# Filter out rows where Censored is 1
X_test_no_censored = X_test_reduced[y_test_reduced["Censored"] == 0]
y_test_no_censored = y_test_reduced[y_test_reduced["Censored"] == 0]

# %%
X_train = X_train[y_train["SurvivalTime"].notna()]
y_train = y_train[y_train["SurvivalTime"].notna()]

X_test = X_test[y_test["SurvivalTime"].notna()]
y_test = y_test[y_test["SurvivalTime"].notna()]

# %%
y_pred_baseline = load_model("baseline_reduced_features").predict(X_test_no_censored)
y_pred_poly = load_model("poly_reduced_features").predict(X_test_reduced)
y_pred_GD = load_model("Gradient_Descent_model").predict(X_test_reduced)
y_pred_knn = load_model("knn_reduced_features").predict(X_test_reduced)

df = compare_models([("Polynomal Regression", y_test_reduced["SurvivalTime"], y_pred_poly, y_test_reduced["Censored"]), ("Gradient Descent", y_test_reduced["SurvivalTime"], y_pred_GD, y_test_reduced["Censored"]),  ("Baseline Model", y_test_no_censored["SurvivalTime"], y_pred_baseline,  y_test_no_censored["Censored"]), ("KNN (k = 20)", y_test_reduced["SurvivalTime"], y_pred_knn, y_test_reduced["Censored"])])
df.T.sort_values("cMSE")

# %% [markdown]
# ### Task 3.1 Missing data imputation
# 
# - Experiment with completing missing data using imputation techniques in [Scikit-Learn](https://scikit-learn.org/stable/modules/impute.html) and [here](https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html), using the baseline model.
# - Compare the results with Task 1.2  in the slides, using a table with the error statistics and the y-y hat plot. Present evidence of your analysis.
# - Choose the best imputation strategies obtained with the baseline and apply them to the best models of Task 2. Analyze your results and report them in the slides, with evidence from your experiments.

# %% [markdown]
# #### Simple Imputer
# [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#simpleimputer)    
# Provides basic strategies for imputing missing values. Missing values can be imputed with a provided constant value, or using the statistics (mean, median or most frequent) of each column in which the missing values are located.   
# Univariate feature Imputation:   
# - If "mean", then replace missing values using the mean along each column. Can only be used with numeric data.
# - If "median", then replace missing values using the median along each column. Can only be used with numeric data.
# - If "most_frequent", then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
# - If "constant", then replace missing values with fill_value. Can be used with strings or numeric data.
# - If an instance of Callable, then replace missing values using the scalar statistic returned by running the callable over a dense 1d array containing non-missing values of each column.

# %%
results, best_strat = run_imputation_baseline_model(X_train, y_train)
df_simple = pd.DataFrame(results, columns=["Model", "MSE"])
df_simple.sort_values(by="MSE", ascending=True)

# %%
imp = SimpleImputer(strategy=best_strat)

X_train_imp = pd.DataFrame(imp.fit_transform(X_train))
X_test_imp = pd.DataFrame(imp.fit_transform(X_test))
mean_imp_model = create_baseline_model(X_train_imp, y_train["SurvivalTime"])
y_pred_mean = mean_imp_model.predict(X_test_imp)
print(f"{best_strat} Imputation MSE: {cMSE_error(y_test["SurvivalTime"], y_pred_mean, y_test["Censored"])}")
task3_results.append(("Mean Imputation", y_test["SurvivalTime"], y_pred_mean, y_test["Censored"]))

plot_y_yhat(y_test["SurvivalTime"], y_pred_mean, "y vs. y-hat (Baseline Simple Imputer - Mean)")

save_model(mean_imp_model, "simple_imp_model_best")

# %% [markdown]
# #### KNN Imputer
# [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#knnimputer)   
# The KNNImputer class provides imputation for filling in missing values using the k-Nearest Neighbors approach.   
# Each missing feature is imputed using values from n_neighbors nearest neighbors that have a value for the feature.

# %%
k_values = list(range(1, 51))
evaluations, bestK = evaluate_knn_imputation(X_train, y_train, k_values)
df_knn = pd.DataFrame(evaluations, columns=["Model", "MSE"])
df_knn.sort_values(by="MSE", ascending=True)

# %%
mean_errors = df_knn['MSE']
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_errors, marker='o')
plt.xlabel('k value')
plt.ylabel('cMSE')
plt.title('cMSE for different k values in KNN Imputation')
plt.grid(True)
plt.show()

# %%
from sklearn.impute import KNNImputer

imp = KNNImputer(n_neighbors=bestK)

X_train_imp = pd.DataFrame(imp.fit_transform(X_train))
X_test_imp = pd.DataFrame(imp.fit_transform(X_test))
kNN_imp_model = create_baseline_model(X_train_imp, y_train["SurvivalTime"])
y_pred = kNN_imp_model.predict(X_test_imp)
results = evaluate_model(y_test["SurvivalTime"], y_pred, y_test["Censored"])
print(f"cMSE: {cMSE_error(y_test['SurvivalTime'], y_pred, y_test['Censored'])}")
task3_results.append(("KNN Imput Model", y_test["SurvivalTime"], y_pred, y_test["Censored"]))

save_model(kNN_imp_model, "knn_imp_model_best")
results

# %%
plot_y_yhat(y_test["SurvivalTime"], y_pred, "y vs. y-hat (Baseline KNN Imputer)")

# %% [markdown]
# #### Iteractive Imputer (Experimental)

# %%
max_iterations = list(range(1, 51))
evaluations, bestIter = evaluate_iterative_imputation(X_train, y_train, max_iterations)
df_II = pd.DataFrame(evaluations, columns=["Model", "MSE"])
df_II.sort_values(by="MSE", ascending=True)

# %%
mean_errors = df_II['MSE']
plt.figure(figsize=(10, 6))
plt.plot(max_iterations, mean_errors, marker='o')
plt.xlabel('Max Iterations')
plt.ylabel('cMSE')
plt.title('cMSE With Max Iterations')
plt.grid(True)

# Disable scientific notation on y-axis
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.gca().yaxis.get_major_formatter().set_useOffset(False)

plt.show()

# %%
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=bestIter)

X_train_imp = pd.DataFrame(imp.fit_transform(X_train))
X_test_imp = pd.DataFrame(imp.fit_transform(X_test))
iterative_imp_model = create_baseline_model(X_train_imp, y_train["SurvivalTime"])
y_pred = iterative_imp_model.predict(X_test_imp)
print(f"cMSE: {cMSE_error(y_test['SurvivalTime'], y_pred, y_test['Censored'])}")
task3_results.append(("Iterative", y_test["SurvivalTime"], y_pred, y_test["Censored"]))

save_model(iterative_imp_model, "iterative_imp_model_best")

# %%
plot_y_yhat(y_test["SurvivalTime"], y_pred, "y vs. y-hat (Baseline Interactive Imputer)")

# %% [markdown]
# #### Overview

# %%
merged_df = pd.concat([df_II.T, df_knn.T, df_simple.T], axis=1)
merged_df.T.sort_values("MSE")

# %%
compare_models(task3_results).T.sort_values(by="cMSE", ascending=True)
task3_results

# %% [markdown]
# ### Task 3.2 Train Models that do not require imputation
# - Develop code to apply models and techniques that can directly handle missing data, such as tree-based methods, like decision trees.
# - Experiment with the Scikit-Learn model [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html), and CatBoost’s [CatBoostRegressor](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor). For installation instructions of the CatBoost Library check [here](https://catboost.ai/en/docs/concepts/python-installation). You can use conda or pip.
# - There is a tutorial on using CatBoost for censored data [here](https://github.com/catboost/tutorials/blob/master/regression/survival.ipynb). Try the Accelerated Failure Time (AFT) CatBoost applied to the assignment data.

# %% [markdown]
# ##### Custom Decision tree using cMSE

# %%
# from scripts.decision_tree import DecisionTreeRegressor
# from scripts.models import cross_val_DecisionTree


# dt = DecisionTreeRegressor()
# cMSEs, y_preds = cross_val_DecisionTree(X_train, y_train, dt)
# np.average(cMSEs)

# %% [markdown]
# #### Decision Tree

# %%
from scripts.models import evaluate_decision_tree
y_train["SurvivalTime"]

best_max_depth, dt_regressor, rmse_dt, cMSE_err = evaluate_decision_tree(X_train, y_train, X_test, y_test)

# %%
save_model(dt_regressor, "best_dt_model")
y_pred = dt_regressor.predict(X_test)
task3_results.append(("DecisionTree", y_test["SurvivalTime"], y_pred, y_test["Censored"]))

# %% [markdown]
# #### Random Forests

# %%
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 10, 30]
}

rf = RandomForestRegressor()

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search_rf.fit(X_train, y_train["SurvivalTime"])

best_params_rf = grid_search_rf.best_params_
best_score_rf = -grid_search_rf.best_score_

print(f"Best parameters: {best_params_rf}")
print(f"Best MSE: {best_score_rf}")

# Fit the model with the best parameters on the entire training data
best_rf = grid_search_rf.best_estimator_
best_rf.fit(X_train, y_train["SurvivalTime"])

y_pred_rf = best_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test["SurvivalTime"], y_pred_rf))

print(f"Test MSE: {rmse_rf}")

# %%
from scripts.models import evaluate_and_plot_rf


n_estimators = [10, 50, 100, 200]
max_depths = [None, 10, 20, 30]
min_samples_split = [2, 5, 10, 20, 30]
min_samples_leafs = [1, 2, 4, 10, 30, 40]

# Call the function
best_params, best_model, rmse_test, results_df = evaluate_and_plot_rf(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    n_estimators=n_estimators, 
    max_depths=max_depths, 
    min_samples_leafs=min_samples_leafs
)

# %%
random_forest_best = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=10
)

random_forest_best.fit(X_train, y_train["SurvivalTime"])
y_pred = random_forest_best.predict(X_test)
task3_results.append(("Random Forest", y_test["SurvivalTime"], y_pred, y_test["Censored"]))

save_model(random_forest_best, "random_forest_best")

# %% [markdown]
# #### HistGradientBoostingRegressor
# Experiment with the Scikit-Learn model [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_depth': [None,2, 3, 5, 7, 10],
    'min_samples_leaf': [20, 50, 100]
}

hist_gbr = HistGradientBoostingRegressor()

grid_search = GridSearchCV(estimator=hist_gbr, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X_train, y_train["SurvivalTime"])

best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best MSE: {best_score}")

# Fit the model with the best parameters on the entire training data
best_hist_gbr = grid_search.best_estimator_
best_hist_gbr.fit(X_train, y_train["SurvivalTime"])

y_pred_hist_gbr = best_hist_gbr.predict(X_test)

rmse_hist_gbr = np.sqrt(mean_squared_error(y_test["SurvivalTime"], y_pred_hist_gbr))

print(f"Test MSE: {rmse_hist_gbr}")

# %%
from scripts.models import evaluate_HistGradientBoostingRegressor
from scripts.plots import visualize_parameter_search

learning_rates=[0.01, 0.1, 0.2]
max_iters=[5, 10, 50, 100, 200, 300]
max_depths=[None, 2, 3, 5, 7, 10]
min_samples_leafs=[5, 10 ,20, 50, 100]

best_params, best_model, rmse_hgb, cMSE_err, results = evaluate_HistGradientBoostingRegressor(X_train,
                                                                                              y_train,
                                                                                              X_test,
                                                                                              y_test,
                                                                                              learning_rates,
                                                                                              max_iters,
                                                                                              max_depths,
                                                                                              min_samples_leafs
                                                                                              )

# %%
y_pred = best_model.predict(X_test)
print(f"cMSE Error: {cMSE_error(y_test['SurvivalTime'], y_pred, y_test['Censored'])}")
task3_results.append(("HistGradientBoosting", y_test["SurvivalTime"], y_pred, y_test["Censored"]))
save_model(best_model, "best_hist_boost_rgr")

# %%
y_pred_missing = load_model("best_hist_boost_rgr").predict(kaggleDF)

# %%
create_submission_csv(best_model.predict(kaggleDF), "handle-missing-submission-07.csv")

# %%
_ = visualize_parameter_search(results)

# %% [markdown]
# #### CatBoostRegressor
# - There is a tutorial on using CatBoost for censored data [here](https://github.com/catboost/tutorials/blob/master/regression/survival.ipynb). Try the Accelerated Failure Time (AFT) CatBoost applied to the assignment data.
# This is wrong

# %%
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from scripts.utils import X_cols_universal


features = X_train.columns.difference(['SurvivalTime', 'Censored'], sort=False)
cat_features = ["Gender", "Stage", "TreatmentType", "TreatmentResponse"]
features

# %%
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from scripts.utils import X_cols_universal
y_train_cat = y_train
y_test_cat = y_test
X_train_cat = X_train
X_test_cat = X_test

# Since we already have the dataset splitted, I'm doing the interval target both on y_train and y_test
# Right Censored: [A, +inf]

y_train_cat['y_lower'] = y_train_cat['SurvivalTime']
y_train_cat['y_upper'] = np.where(y_train['Censored'] == 1, y_train["SurvivalTime"], -1)

y_test_cat['y_lower'] = y_test_cat['SurvivalTime']
y_test_cat['y_upper'] = np.where(y_test_cat['Censored'] == 1, y_test_cat["SurvivalTime"], -1)

# Convert categorical features to strings
cat_features = ["Gender", "Stage", "TreatmentType", "TreatmentResponse"]
X_train_cat[cat_features] = X_train_cat[cat_features].astype(str)
X_test_cat[cat_features] = X_test_cat[cat_features].astype(str)

# Create CatBoost Pool
y_test_cat = y_test_cat.drop(['SurvivalTime', 'Censored'], axis=1)
y_train_cat = y_train_cat.drop(['SurvivalTime', 'Censored'], axis=1)

train_pool = Pool(X_train_cat, label=y_train_cat[['y_lower', 'y_upper']], cat_features=cat_features)
test_pool = Pool(X_test_cat, label=y_test_cat[['y_lower','y_upper']], cat_features=cat_features)

# %%
# Train models with different distributions
model_normal = CatBoostRegressor(
    iterations=500,
    loss_function='SurvivalAft:dist=Normal',
    eval_metric='SurvivalAft',
    verbose=0
)
model_logistic = CatBoostRegressor(
    iterations=500,
    loss_function='SurvivalAft:dist=Logistic;scale=1.2',
    eval_metric='SurvivalAft',
    verbose=0
)
model_extreme = CatBoostRegressor(
    iterations=500,
    loss_function='SurvivalAft:dist=Extreme;scale=2',
    eval_metric='SurvivalAft',
    verbose=0
)

# Fit models
_ = model_normal.fit(train_pool, eval_set=test_pool)
_ = model_logistic.fit(train_pool, eval_set=test_pool)
_ = model_extreme.fit(train_pool, eval_set=test_pool)

# %%
train_predictions = pd.DataFrame({'y_lower': y_train_cat['y_lower'],
                                  'y_upper': y_train_cat['y_upper'],
                                  'preds_normal': model_normal.predict(train_pool, prediction_type='Exponent'),
                                  'preds_logistic': model_logistic.predict(train_pool, prediction_type='Exponent'),
                                  'preds_extreme': model_extreme.predict(train_pool, prediction_type='Exponent')})
train_predictions['y_upper'] = np.where(train_predictions['y_upper']==-1, np.inf, train_predictions['y_upper'])

test_predictions = pd.DataFrame({'y_lower': y_test_cat['y_lower'],
                                  'y_upper': y_test_cat['y_upper'],
                                  'preds_normal': model_normal.predict(test_pool, prediction_type='Exponent'),
                                  'preds_logistic': model_logistic.predict(test_pool, prediction_type='Exponent'),
                                  'preds_extreme': model_extreme.predict(test_pool, prediction_type='Exponent')})
test_predictions['y_upper'] = np.where(test_predictions['y_upper']==-1, np.inf, test_predictions['y_upper'])

# %%
def interval_mae(y_true_lower, y_true_upper, y_pred):
    mae = np.where((y_true_lower <= y_pred) & (y_pred <= y_true_upper),
                   0,
                   np.minimum(np.abs(y_true_lower-y_pred),
                              np.abs(y_true_upper-y_pred))) 
    return mae.mean()

distributions = ['normal', 'logistic', 'extreme']
print('Interval MAE')
for dist in distributions:
    train_metric = interval_mae(train_predictions['y_lower'], train_predictions['y_upper'], train_predictions[f'preds_{dist}'])
    test_metric = interval_mae(test_predictions['y_lower'], test_predictions['y_upper'], test_predictions[f'preds_{dist}'])
    print(f'Train set. dist:{dist}: {train_metric:0.2f}')
    print(f'Test set. dist:{dist}: {test_metric:0.2f}')
    print('---------------------------')
    
train_predictions

# %%
y_pred_test = model_normal.predict(test_pool, prediction_type='Exponent')
print(f"cMSE Error: {cMSE_error(y_test['SurvivalTime'], y_pred_test, y_test['Censored'])}")
df = load_kaggle_df()
df[cat_features] = df[cat_features].astype(str)
kaggle_pool = Pool(df, cat_features=cat_features)
pred = model_normal.predict(kaggle_pool, prediction_type='Exponent')
task3_results.append(("Catboost", y_test["SurvivalTime"], y_pred_test, y_test["Censored"]))

create_submission_csv(pred, "handle-missing-submission-08.csv.csv")

# %%
plot_y_yhat(y_test["SurvivalTime"], y_pred_test, "y vs. y_hat (Catboost)")

# %% [markdown]
# ### Task 3.3 Evaluation
# 
# - Compare the results of the strategies developed in Task 3.1 and 3.2 with the baseline model in the slides, using a table with the error statistics and the y-y hat plot. Present evidence of your analysis.
# - Submit the best predictions from Task 3 to Kaggle with the file name `handle-missing-submission-xx.csv` where `xx` is an natural number. The submission used for grading is the one with the larger value.

# %%
compare_models(task3_results).T.sort_values(by="cMSE", ascending=True)

# %% [markdown]
# - Try the best imputation strategies of Task 3.1, impute the data, run the best model of task 3.2 and compare with the baseline in the slides.

# %%
imp = KNNImputer(n_neighbors=8)

X_train_imp = pd.DataFrame(imp.fit_transform(X_train))
X_test_imp = pd.DataFrame(imp.fit_transform(X_test))

y_pred = load_model("best_hist_boost_rgr").predict(X_test_imp)

task3_results.append(("KNN Imputation with HGB", y_test["SurvivalTime"], y_pred, y_test["Censored"]))
compare_models(task3_results).T.sort_values(by="cMSE", ascending=True)

# %%



