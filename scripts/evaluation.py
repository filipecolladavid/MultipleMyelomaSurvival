import numpy as np
import pandas as pd

def cMSE_error(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]


def gradient_cMSE_error(y, y_hat, c, X):
    residuals = (y - y_hat).values if isinstance(y, pd.Series) else y - y_hat
    c = c.values if isinstance(c, pd.Series) else c
    X = X.values if isinstance(X, pd.DataFrame) else X

    positive_mask = residuals > 0
    non_positive_mask = ~positive_mask

    gradient = np.zeros(X.shape[1])
    
    gradient += np.sum(2 * residuals[positive_mask][:, None] * X[positive_mask], axis=0)
    
    gradient += np.sum(2 * (1 - c[non_positive_mask])[:, None] * residuals[non_positive_mask][:, None] * X[non_positive_mask], axis=0)

    return gradient / len(residuals)




def evaluate_model(y_test, y_pred, c):
    y_test = y_test
    y_pred = y_pred
    errors = y_test - y_pred
    
    return {
        "Max Error": np.max(np.abs(errors)),
        "Min Error": np.min(np.abs(errors)),
        "Mean Error": np.mean(np.abs(errors)),
        "Std Dev of Error": float(np.std(errors, axis=0)),
        "cMSE": float(cMSE_error(y_test, y_pred, c)),
    }
    
def compare_models(results_list):
    """
    Compare multiple models by evaluating their performance.

    Args:
        results_list (list): A list of tuples where each tuple contains:
            - model_name (str): The name of the model.
            - y_test (array-like): The true labels.
            - y_pred (array-like): The predicted labels by the model.
            - c (any): Additional parameter(s) required by the evaluate_model function.

    Returns:
        pd.DataFrame: A DataFrame where each column represents the evaluation results of a model.
    """
    comparison_df = pd.DataFrame({})
    for model_name, y_test, y_pred, c in results_list:
        comparison_df[model_name] = evaluate_model(y_test, y_pred, c)
    return comparison_df

# TODO - make evaluation all - load all models from models folder
# Probably need to create a folder for reduced_features and normal
    