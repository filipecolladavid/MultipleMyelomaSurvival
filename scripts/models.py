# Models
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from scripts.evaluation import cMSE_error, gradient_cMSE_error
from scripts.plots import plot_y_yhat
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

def create_baseline_model(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('regressor', LinearRegression())  
    ])
    pipeline.fit(X, y)    
    return pipeline

def create_polynomial_model(X, y, degree):
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('linear_regression', LinearRegression())
    ])
    pipeline.fit(X,y)
    return pipeline



def cross_val_polynomial(X, y, degrees, k=10, printInfo = True):
    """
    Perform k-fold cross-validation for polynomial regression models of varying degrees.

    Parameters:
    X (pd.DataFrame): The input features.
    y (pd.Series): The target variable.
    degrees (list): A list of polynomial degrees to evaluate.
    k (int, optional): The number of folds for cross-validation. Default is 10.
    printInfo (bool, optional): Whether to print detailed information during the process. Default is True.

    Returns:
    tuple: A tuple containing:
        - best_degree (int): The polynomial degree with the lowest average validation MSE.
        - train_errors (dict): A dictionary with degrees as keys and lists of average training MSEs for each fold as values.
        - val_errors (dict): A dictionary with degrees as keys and lists of average validation MSEs for each fold as values.
        - feature_counts (dict): A dictionary with degrees as keys and the number of features for each polynomial degree as values.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    train_errors = {degree: [] for degree in degrees}
    val_errors = {degree: [] for degree in degrees}
    feature_counts = {}

    for degree in degrees:
        num_features = PolynomialFeatures(degree=degree).fit(X).n_output_features_
        feature_counts[degree] = num_features
        if printInfo:
            print(f"Degree: {degree}, Feature Complexity (Number of Features): {num_features}")

        fold_train_errors = []
        fold_val_errors = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = create_polynomial_model(X_train, y_train["SurvivalTime"], degree)
            model.fit(X_train, y_train["SurvivalTime"])

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_mse = mean_squared_error(y_train["SurvivalTime"], y_train_pred)
            val_mse = mean_squared_error(y_val["SurvivalTime"], y_val_pred)

            fold_train_errors.append(train_mse)
            fold_val_errors.append(val_mse)

        train_errors[degree].append(np.mean(fold_train_errors))
        val_errors[degree].append(np.mean(fold_val_errors))

        if printInfo:
            print(f"Degree: {degree}, Average Training MSE: {np.mean(fold_train_errors)}")
            print(f"Degree: {degree}, Average Validation MSE: {np.mean(fold_val_errors)}\n")

    best_degree = min(val_errors, key=lambda d: np.mean(val_errors[d]))
    if printInfo:
        print(f"\nBest Degree: {best_degree} with Average Validation MSE: {np.mean(val_errors[best_degree])}")
        print(f"Feature Complexity for Best Degree: {feature_counts[best_degree]} features")

    return best_degree, train_errors, val_errors, feature_counts

def create_KNN_model(X, y, neighbors, metric, weight):
    pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsRegressor(n_neighbors=neighbors, metric=metric, weights=weight))
                ])
    pipeline.fit(X, y)
    return pipeline

def cross_val_knn(X, y, k_values, weights=["uniform", "distance"], metrics=["euclidean", "manhattan"], k=10, printInfo=True):
    """
    Perform k-fold cross-validation for K-Nearest Neighbors (KNN) regression with various hyperparameters.
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    k_values : list of int
        List of k values (number of neighbors) to evaluate.
    weights : list of str, optional
        List of weight functions to use in prediction. Default is ["uniform", "distance"].
    metrics : list of str, optional
        List of distance metrics to use. Default is ["euclidean", "manhattan"].
    k : int, optional
        Number of folds for cross-validation. Default is 10.
    printInfo : bool, optional
        If True, print information about each combination of hyperparameters and the best parameters. Default is True.
    Returns:
    --------
    best_params : tuple
        The best combination of (k, metric, weight) based on the lowest average validation MSE.
    train_errors : dict
        Dictionary with k values as keys and lists of average training MSEs for each fold as values.
    val_errors : dict
        Dictionary with k values as keys and lists of average validation MSEs for each fold as values.
    cv_results : dict
        Dictionary with (k, metric, weight) tuples as keys and average validation MSEs as values.
    metric_weight_mse : dict
        Nested dictionary with metrics as keys, weights as sub-keys, and lists of average validation MSEs as values.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    cv_results = {}
    train_errors = {k: [] for k in k_values}
    val_errors = {k: [] for k in k_values}

    metric_weight_mse = {metric: {weight: [] for weight in weights} for metric in metrics}

    for neighbors in k_values:
        for weight in weights:
            for metric in metrics:
                model = create_KNN_model(X, y["SurvivalTime"], neighbors, metric, weight)
                
                fold_train_errors = []
                fold_val_errors = []

                for train_index, val_index in kf.split(X):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    
                    model.fit(X_train, y_train["SurvivalTime"])

                    y_train_pred = model.predict(X_train)
                    y_val_pred = model.predict(X_val)

                    train_mse = mean_squared_error(y_train["SurvivalTime"], y_train_pred)
                    val_mse = mean_squared_error(y_val["SurvivalTime"], y_val_pred)

                    fold_train_errors.append(train_mse)
                    fold_val_errors.append(val_mse)

                train_errors[neighbors].append(np.mean(fold_train_errors))
                val_errors[neighbors].append(np.mean(fold_val_errors))
                
                avg_mse = np.mean(fold_val_errors)
                cv_results[(neighbors, metric, weight)] = avg_mse

                metric_weight_mse[metric][weight].append(avg_mse)

                if printInfo:
                    print(f"k: {neighbors}, Metric: {metric}, Weights: {weight}, Average Validation MSE: {avg_mse}")

    best_params = min(cv_results, key=cv_results.get)
    if printInfo:
        print(f"\nBest Parameters: k={best_params[0]}, Metric={best_params[1]}, Weights={best_params[2]}")
        print(f"Best Average Validation MSE: {cv_results[best_params]}")

    return best_params, train_errors, val_errors, cv_results, metric_weight_mse

def cross_val_DecisionTree(X,y, pipeline, k=10):
    """
    Perform k-fold cross-validation using a baseline model and calculate the Censored Mean Squared Error (cMSE) for each fold.
    
    Parameters:
    X (pd.DataFrame): The input features for the dataset.
    y (pd.Series): The target variable for the dataset.
    c (int): A binary to be used as a parameter in the cMSE_error, 1 for censored 0, for non censored
    k (int, optional): The number of folds for cross-validation. Default is 10.
    
    Returns:
    tuple: A tuple containing:
        - cMSE_errors (list): A list of cumulative Mean Squared Error (cMSE) values for each fold.
        - y_preds (list): A list of predicted values for each fold.
    """
    kf = KFold(n_splits=10, shuffle=True)
    cMSE_errors = []
    y_preds = []

    for train_index, test_index in kf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_censored, y_test_censored = y["Censored"].iloc[train_index], y["Censored"].iloc[test_index]
        y_train_fold, y_test_fold = y.drop(columns=["Censored"]).iloc[train_index], y.drop(columns=["Censored"]).iloc[test_index]
        pipeline.fit(X_train_fold, y_train_fold)
        y_pred_fold = pipeline.predict(X_test_fold)
        y_preds.append(y_pred_fold)
        cMSE_errors.append(cMSE_error(y_test_fold, y_pred_fold, y_test_censored))
    
    return cMSE_errors, y_preds

# Imputation

def run_imputation_baseline_model(X, y, imputers=["mean", "median", "most_frequent"], k=10):
    """
    Runs a baseline linear regression model with optional imputation using K-Fold Cross-Validation.
    
    Parameters:
    ----------
    X (pd.DataFrame): The input features for training.
    y (pd.DataFrame): The target variable for training with "SurvivalTime".
    imputers (list): A list of imputation strategies. Default is ["mean", "median", "most_frequent"].
    k (int): Number of K-Folds. Default is 10.
    
    Returns:
    --------
    results (list): A list of tuples with evaluation results for each imputation strategy.
    avg_MSEs (dict): A dictionary with imputation strategies as keys and average MSEs as values.
    best_strat (str): The imputation strategy with the lowest average MSE.
    """
    results = []
    avg_MSEs = {}
    best_strat = None
    best_MSE = float('inf')
    kf = KFold(n_splits=k, shuffle=True)

    for strat in imputers:
        print(f"Evaluating imputation strategy: {strat}")

        # Create the pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=strat)),
            ('scaler', StandardScaler()),  
            ('regressor', LinearRegression())  
        ])

        fold_MSE = []

        for train_idx, val_idx in kf.split(X):
            # Split the data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit the pipeline and predict
            pipeline.fit(X_train, y_train["SurvivalTime"])
            y_val_pred = pipeline.predict(X_val)

            # Calculate the MSE error
            current_MSE = mean_squared_error(y_val["SurvivalTime"], y_val_pred)
            fold_MSE.append(current_MSE)

        avg_MSE = np.mean(fold_MSE)
        avg_MSEs[strat] = avg_MSE

        print(f"Average MSE for {strat} imputation: {avg_MSE:.4f}")

        # Update the best strategy
        if avg_MSE < best_MSE:
            best_MSE = avg_MSE
            best_strat = strat

        results.append((f"{strat} model", avg_MSE))


    return results, best_strat


from sklearn.model_selection import KFold

def evaluate_knn_imputation(X_train, y_train, k_values, k_folds=10):
    """
    Evaluate KNN imputation using K-Fold cross-validation.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training target variable.
        k_values (list): List of K values to evaluate.
        k_folds (int): Number of K-Folds for cross-validation.

    Returns:
        list: A list of evaluation results in the format expected by compare_models.
        int: The best K value with the lowest average cMSE.
    """
    kf = KFold(n_splits=k_folds, shuffle=True)
    best_k = None
    best_MSE = float('inf')
    evaluations = []

    for k in k_values:
        imp = KNNImputer(n_neighbors=k)
        fold_MSE = []

        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Impute missing values for both train and validation sets
            X_fold_train_imputed = pd.DataFrame(imp.fit_transform(X_fold_train), columns=X_train.columns)
            X_fold_val_imputed = pd.DataFrame(imp.transform(X_fold_val), columns=X_train.columns)
            
            # Create baseline model and fit on imputed training data
            baseline_model = create_baseline_model(X_fold_train_imputed, y_fold_train["SurvivalTime"])

            # Predict on validation set
            y_val_pred = baseline_model.predict(X_fold_val_imputed)
            
            # Compute cMSE
            current_MSE = mean_squared_error(y_fold_val["SurvivalTime"], y_val_pred)
            fold_MSE.append(current_MSE)
        
        # Calculate average cMSE across folds
        avg_MSE = np.mean(fold_MSE)

        # Track the best k value
        if avg_MSE < best_MSE:
            best_k = k
            best_MSE = avg_MSE
        
        # Append evaluation result for compare_models compatibility
        evaluations.append((f"KNN (k={k})", avg_MSE))

    return evaluations, best_k




from sklearn.model_selection import KFold

def evaluate_iterative_imputation(X_train, y_train, max_iterations, k_folds=10):
    """
    Evaluate Iterative Imputation using K-Fold cross-validation.

    Args:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.DataFrame): Training target variable.
    - max_iterations (list): List of max_iter values to evaluate.
    - k_folds (int): Number of folds for cross-validation.

    Returns:
    - evaluations (list): A list of evaluation results in the format expected by compare_models.
    - best_iter (int): The best max_iter with the lowest average cMSE.
    """
    kf = KFold(n_splits=k_folds, shuffle=True)
    best_iter = None
    best_MSE = float('inf')
    evaluations = []

    for iter_num in max_iterations:
        print(f"Evaluating Iterative Imputer with max_iter={iter_num}")
        imp = IterativeImputer(max_iter=iter_num)
        fold_MSE = []

        for train_idx, val_idx in kf.split(X_train):
            # Split data into train/validation sets
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Impute missing values for both train and validation sets
            X_fold_train_imputed = pd.DataFrame(imp.fit_transform(X_fold_train), columns=X_train.columns)
            X_fold_val_imputed = pd.DataFrame(imp.transform(X_fold_val), columns=X_train.columns)

            # Create baseline model and fit on imputed training data
            baseline_model = create_baseline_model(X_fold_train_imputed, y_fold_train["SurvivalTime"])

            # Predict on validation set
            y_val_pred = baseline_model.predict(X_fold_val_imputed)

            # Compute cMSE
            current_MSE = mean_squared_error(y_fold_val["SurvivalTime"], y_val_pred)
            fold_MSE.append(current_MSE)

        # Calculate average cMSE across folds
        avg_MSE = np.mean(fold_MSE)
        print(f"Average MSE for max_iter={iter_num}: {avg_MSE:.4f}")

        # Track the best max_iter
        if avg_MSE < best_MSE:
            best_iter = iter_num
            best_MSE = avg_MSE

        # Append evaluation result for compare_models compatibility
        evaluations.append(( f"Iterative Imputer (max_iter={iter_num})", avg_MSE))

    return evaluations, best_iter


def evaluate_decision_tree(X_train, y_train, X_test, y_test, max_depths=range(1, 21)):
    """
    Evaluate DecisionTreeRegressor with different max depths using cross-validation.
    
    Parameters:
    X_train (pd.DataFrame): The input features for training.
    y_train (pd.Series or np.ndarray): The target variable for training.
    X_test (pd.DataFrame): The input features for testing.
    y_test (pd.Series or np.ndarray): The target variable for testing.
    max_depths (range, optional): Range of max depths to evaluate. Default is range(1, 21).
    
    Returns:
    tuple: A tuple containing the best max depth, the trained DecisionTreeRegressor model, and the MSE on the test set.
    """
    kf = KFold(n_splits=10, shuffle=True)
    mean_mse_per_depth = []

    for depth in max_depths:
        dt_regressor = DecisionTreeRegressor(max_depth=depth)
        cv_scores_dt = cross_val_score(dt_regressor, X_train, y_train["SurvivalTime"], cv=kf, scoring='neg_mean_squared_error')
        mean_mse_per_depth.append(-cv_scores_dt.mean())

    # Plot the MSE for different max depths
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, mean_mse_per_depth, marker='o', linestyle='-', color='b')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean MSE')
    plt.title('Mean MSE for Different Max Depths in DecisionTreeRegressor')
    plt.grid(True)
    plt.show()

    # Find the best max depth
    best_max_depth = max_depths[np.argmin(mean_mse_per_depth)]
    print(f"Best max depth: {best_max_depth}")

    # Fit the DecisionTreeRegressor with the best max depth
    dt_regressor = DecisionTreeRegressor(max_depth=best_max_depth)
    dt_regressor.fit(X_train, y_train["SurvivalTime"])

    # Predict on the test set
    y_pred_dt = dt_regressor.predict(X_test)

    # Calculate MSE on the test set
    mse_dt = mean_squared_error(y_test["SurvivalTime"], y_pred_dt)
    cMSE_err = cMSE_error(y_test["SurvivalTime"], y_pred_dt, y_test["Censored"])
    print(f"Test MSE: {mse_dt}")
    print(f"Test cMSE: {cMSE_err}")

    # Plot y vs. y-hat for DecisionTreeRegressor
    plot_y_yhat(y_test["SurvivalTime"], y_pred_dt, plot_title="y vs. y-hat (Decision Tree Regressor)")
    plt.show()

    return best_max_depth, dt_regressor, mse_dt, cMSE_err

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

def evaluate_and_plot_rf(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    n_estimators=[10, 50, 100, 200, 300], 
    max_depths=[None, 10, 20, 30], 
    min_samples_leafs=[1, 2, 4, 10]
):
    """
    Evaluate RandomForestRegressor with different hyperparameters using cross-validation and plot MSE vs hyperparameters.

    Parameters:
    - X_train: Training features (pd.DataFrame or np.ndarray).
    - y_train: Training target variable (pd.Series or np.ndarray).
    - X_test: Testing features (pd.DataFrame or np.ndarray).
    - y_test: Testing target variable (pd.Series or np.ndarray).
    - n_estimators: List of n_estimators values to evaluate.
    - max_depths: List of max_depth values to evaluate.
    - min_samples_leafs: List of min_samples_leaf values to evaluate.

    Returns:
    - best_params: Best hyperparameter combination.
    - best_model: Trained model with best hyperparameters.
    - mse_test: MSE on the test set.
    - results: DataFrame with all evaluated combinations and corresponding MSEs.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_mse = float('inf')
    best_params = None
    best_model = None

    # Grid search over hyperparameters
    for n in n_estimators:
        for depth in max_depths:
            for min_samples in min_samples_leafs:
                rf = RandomForestRegressor(
                    n_estimators=n, 
                    max_depth=depth, 
                    min_samples_leaf=min_samples,
                )
                # Cross-validation MSE
                cv_scores = cross_val_score(
                    rf, X_train, y_train["SurvivalTime"], cv=kf, scoring="neg_mean_squared_error"
                )
                mean_mse = -cv_scores.mean()
                results.append((n, depth, min_samples, mean_mse))

                # Update best model
                if mean_mse < best_mse:
                    best_mse = mean_mse
                    best_params = (n, depth, min_samples)
                    best_model = rf

    # Train the best model on the entire training set
    best_model.fit(X_train, y_train["SurvivalTime"])
    y_pred = best_model.predict(X_test)

    # Test MSE
    mse_test = mean_squared_error(y_test["SurvivalTime"], y_pred)

    print(f"Best Parameters: n_estimators={best_params[0]}, max_depth={best_params[1]}, min_samples_leaf={best_params[2]}")
    print(f"Validation MSE: {best_mse}")
    print(f"Test MSE: {mse_test}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=["n_estimators", "max_depth", "min_samples_leaf", "MSE"])

    # Plot MSE evolution
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MSE vs Hyperparameters", fontsize=16)

    for i, param in enumerate(["n_estimators", "max_depth", "min_samples_leaf"]):
        mean_mse = results_df.groupby(param)["MSE"].mean()
        std_mse = results_df.groupby(param)["MSE"].std()

        axs[i].plot(mean_mse.index, mean_mse.values, 'o-', linewidth=2)
        axs[i].fill_between(mean_mse.index, 
                            mean_mse.values - std_mse.values, 
                            mean_mse.values + std_mse.values, 
                            alpha=0.2)
        axs[i].set_title(f"MSE vs {param}")
        axs[i].set_xlabel(param)
        axs[i].set_ylabel("MSE")
        axs[i].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    return best_params, best_model, mse_test, results_df


def evaluate_HistGradientBoostingRegressor(X_train, y_train, X_test, y_test, learning_rates=[0.01, 0.1, 0.2], max_iters=[10, 50, 100, 200, 300], max_depths=[None, 2, 3, 5, 7, 10], min_samples_leafs=[10 ,20, 50, 100]):
    """
    Evaluate HistGradientBoostingRegressor with different hyperparameters using cross-validation.
    
    Parameters:
    X_train (pd.DataFrame): The input features for training.
    y_train (pd.Series or np.ndarray): The target variable for training.
    X_test (pd.DataFrame): The input features for testing.
    y_test (pd.Series or np.ndarray): The target variable for testing.
    learning_rates (list, optional): List of learning rates to evaluate. Default is [0.01, 0.1, 0.2].
    max_iters (list, optional): List of max iterations to evaluate. Default is [100, 200, 300].
    max_depths (list, optional): List of max depths to evaluate. Default is [None, 2, 3, 5, 7, 10].
    min_samples_leafs (list, optional): List of min samples leaf to evaluate. Default is [20, 50, 100].
    
    Returns:
    tuple: A tuple containing the best hyperparameters, the trained HistGradientBoostingRegressor model, and the MSE on the test set.
    """
    kf = KFold(n_splits=10, shuffle=True)
    best_params = None
    best_mse = float('inf')
    best_model = None
    results = []

    for lr in learning_rates:
        for iters in max_iters:
            for depth in max_depths:
                for min_samples in min_samples_leafs:
                    hgb_regressor = HistGradientBoostingRegressor(
                        learning_rate=lr, 
                        max_iter=iters, 
                        max_depth=depth, 
                        min_samples_leaf=min_samples
                    )
                    cv_scores_hgb = cross_val_score(
                        hgb_regressor, 
                        X_train, 
                        y_train["SurvivalTime"], 
                        cv=kf, 
                        scoring='neg_mean_squared_error'
                    )
                    mean_mse = -cv_scores_hgb.mean()
                    
                    results.append([lr, iters, depth, min_samples, mean_mse])
                    
                    if mean_mse < best_mse:
                        best_mse = mean_mse
                        best_params = (lr, iters, depth, min_samples)
                        best_model = hgb_regressor
    

                    # print(f"Learning Rate: {lr}, Max Iter: {iters}, Max Depth: {depth}, Min Samples Leaf: {min_samples}, Mean MSE: {mean_mse}")

    print(f"\nBest Parameters: Learning Rate={best_params[0]}, Max Iter={best_params[1]}, Max Depth={best_params[2]}, Min Samples Leaf={best_params[3]}")
    print(f"Best Mean MSE: {best_mse}")

    # Fit the HistGradientBoostingRegressor with the best hyperparameters
    best_model.fit(X_train, y_train["SurvivalTime"])

    # Predict on the test set
    y_pred_hgb = best_model.predict(X_test)

    # Calculate MSE on the test set
    mse_hgb = mean_squared_error(y_test["SurvivalTime"], y_pred_hgb)
    cMSE_err = cMSE_error(y_test["SurvivalTime"], y_pred_hgb, y_test["Censored"])
    print(f"Test MSE: {mse_hgb}")
    print(f"Test cMSE: {cMSE_err}")

    # Plot y vs. y-hat for HistGradientBoostingRegressor
    plot_y_yhat(y_test["SurvivalTime"], y_pred_hgb, plot_title="y vs. y-hat (HistGradientBoostingRegressor)")
    plt.show()

    return best_params, best_model, mse_hgb, cMSE_err, results