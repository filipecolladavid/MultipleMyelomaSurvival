from collections import Counter
import sys
import os

# Append the parent directory to the Python path
#sys.path.append(os.path.abspath(".."))


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

RESULTS_PATH = "results"
PREDICTIONS = RESULTS_PATH+"/predictions/"
PLOTS = RESULTS_PATH+"/plots/"


def plot_y_yhat(y_test, y_pred, plot_title="y vs. y-hat"):
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    MAX = 500
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test), MAX, replace=False)
        y_test = y_test[idx]
        y_pred = y_pred[idx]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
    plt.xlabel('True SurvivalTime')
    plt.ylabel('Predicted SurvivalTime')
    plt.title(plot_title)
    plt.axis("square")
    plt.savefig(PLOTS+plot_title.replace(" ", "_") + '.pdf')
    plt.show()
    
def plot_best_degrees(best_degrees, all_degrees, nRuns):
    degree_counts = Counter(best_degrees)
    
    frequencies = [degree_counts.get(degree, 0) for degree in all_degrees]

    plt.figure(figsize=(8, 6))
    plt.bar(all_degrees, frequencies, color='skyblue', edgecolor='black')
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Frequency of Best Polynomial Degrees Across ' + str(nRuns) + ' Runs', fontsize=14)
    plt.xticks(all_degrees)
    plt.grid(axis='y', linestyle='--', alpha=1)
    plt.tight_layout()
    plt.show()
    
def plot_mse_per_degree(train_errors, avg_val_errors):
    degrees = list(avg_val_errors.keys())
    avg_train_errors = [np.mean(train_errors[d]) for d in degrees]
    avg_val_errors_list = [avg_val_errors[d] for d in degrees]

    
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, avg_train_errors, label="Training MSE", marker='o')
    plt.plot(degrees, avg_val_errors_list, label="Validation MSE", marker='o')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title("Training vs Validation MSE")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_feature_complexity(feature_counts, avg_val_errors):
    degrees = list(avg_val_errors.keys())
    feature_counts_list = [feature_counts[d] for d in degrees]

    plt.bar(degrees, feature_counts_list, color='skyblue')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Number of Features")
    plt.title("Feature Complexity by Polynomial Degree")
    plt.xticks(ticks=degrees) 
    plt.grid(axis='y')
    
def plot_mse_metric_weight(k_values, metric_weight_mse):
    for metric, weights_dict in metric_weight_mse.items():
        plt.figure(figsize=(10, 6))
        for weight, rmses in weights_dict.items():
            plt.plot(k_values, rmses, label=f"Weight: {weight}", marker='o')
        plt.xlabel("Number of Neighbors (k)")
        plt.ylabel("Average Validation MSE")
        plt.title(f"MSE for Metric: {metric}")
        plt.legend()
        plt.grid(True)
        plt.show()        
        
def visualize_parameter_search(results_list):
    """
    Create line plot visualizations for parameter search results
    
    Args:
    results_list: List of lists containing [learning_rate, max_iter, max_depth, min_samples_leaf, mse]
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results_list, 
        columns=['learning_rate', 'max_iter', 'max_depth', 'min_samples_leaf', 'mse']
    )
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MSE vs Parameter Values', fontsize=16, y=1.02)
    
    # Plot for Learning Rate
    mean_mse = results_df.groupby('learning_rate')['mse'].mean()
    std_mse = results_df.groupby('learning_rate')['mse'].std()
    
    axs[0, 0].plot(mean_mse.index, mean_mse.values, 'o-', linewidth=2)
    axs[0, 0].fill_between(mean_mse.index, 
                          mean_mse.values - std_mse.values,
                          mean_mse.values + std_mse.values,
                          alpha=0.2)
    axs[0, 0].set_title('MSE vs Learning Rate')
    axs[0, 0].set_xlabel('Learning Rate')
    axs[0, 0].set_ylabel('MSE')
    
    # Plot for Max Iterations
    mean_mse = results_df.groupby('max_iter')['mse'].mean()
    std_mse = results_df.groupby('max_iter')['mse'].std()
    
    axs[0, 1].plot(mean_mse.index, mean_mse.values, 'o-', linewidth=2)
    axs[0, 1].fill_between(mean_mse.index,
                          mean_mse.values - std_mse.values,
                          mean_mse.values + std_mse.values,
                          alpha=0.2)
    axs[0, 1].set_title('MSE vs Max Iterations')
    axs[0, 1].set_xlabel('Max Iterations')
    axs[0, 1].set_ylabel('MSE')
    
    # Plot for Max Depth
    mean_mse = results_df.groupby('max_depth')['mse'].mean()
    std_mse = results_df.groupby('max_depth')['mse'].std()
    
    axs[1, 0].plot(mean_mse.index, mean_mse.values, 'o-', linewidth=2)
    axs[1, 0].fill_between(mean_mse.index,
                          mean_mse.values - std_mse.values,
                          mean_mse.values + std_mse.values,
                          alpha=0.2)
    axs[1, 0].set_title('MSE vs Max Depth')
    axs[1, 0].set_xlabel('Max Depth')
    axs[1, 0].set_ylabel('MSE')
    
    # Plot for Min Samples Leaf
    mean_mse = results_df.groupby('min_samples_leaf')['mse'].mean()
    std_mse = results_df.groupby('min_samples_leaf')['mse'].std()
    
    axs[1, 1].plot(mean_mse.index, mean_mse.values, 'o-', linewidth=2)
    axs[1, 1].fill_between(mean_mse.index,
                          mean_mse.values - std_mse.values,
                          mean_mse.values + std_mse.values,
                          alpha=0.2)
    axs[1, 1].set_title('MSE vs Min Samples Leaf')
    axs[1, 1].set_xlabel('Min Samples Leaf')
    axs[1, 1].set_ylabel('MSE')
    
    # Adjust layout
    plt.tight_layout()
    
    # Add grid to all subplots
    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig