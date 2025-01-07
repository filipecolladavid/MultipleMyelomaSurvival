import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from scripts.evaluation import cMSE_error

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, error_red=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.error_red = error_red
        self.value = value # Leaf nodes


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def get_params(self, deep=True):
        return {
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def node_cMSE(self, y):
        """
        Compute cMSE for a node using mean prediction
        
        Parameters:
        y: array of target values
        
        Returns:
        float: cMSE error for the node
        """
        y_hat = np.full_like(y, np.mean(y))
        return cMSE_error(y["SurvivalTime"], y_hat["SurvivalTime"], y["Censored"])
    
    def error_reduction(self, parent, l_child, r_child):
        """
        Compute error reduction using cMSE
        
        Parameters:
        parent: array of parent node target values
        l_child: array of left child target values
        r_child: array of right child target values
        
        Returns:
        float: reduction in cMSE error after split
        """
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        
        error_parent = self.node_cMSE(parent)
        error_l = self.node_cMSE(l_child)
        error_r = self.node_cMSE(r_child)
        
        reduction = error_parent - (weight_l * error_l + weight_r * error_r)
        return reduction
    
    def get_best_split(self, X, y, num_samples, num_features):
        ''' 
        Function to find the best split point using cMSE error reduction
        '''
        best_split = {}
        max_error_red = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold)
                
                # check if both children have samples
                if len(y_left) > 0 and len(y_right) > 0:
                    # compute error reduction using cMSE
                    curr_error_red = self.error_reduction(y, y_left, y_right)
                    
                    # update the best split if needed
                    if curr_error_red > max_error_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["X_left"] = X_left
                        best_split["y_left"] = y_left
                        best_split["X_right"] = X_right
                        best_split["y_right"] = y_right
                        best_split["error_red"] = curr_error_red
                        max_error_red = curr_error_red
                        
        return best_split
    
    def calculate_leaf_value(self, Y):
        """
        Censored Mean Squared error so still mean
        """
        return np.mean(Y)
    
    def build_tree(self, X, y, curr_depth=0):
        num_samples, num_features = np.shape(X)
        X = np.array(X)
        y = np.array(y)
        
        # Check stopping criteria
        if num_samples < self.min_samples_split or curr_depth >= self.max_depth:
            return Node(value=self.calculate_leaf_value(y))
        
        best_split = self.get_best_split(X, y, num_samples, num_features)
        
        # If no improvement, make a leaf node
        if not best_split or best_split.get("error_red", 0) <= 0:
            return Node(value=self.calculate_leaf_value(y))
            
        # Recursive building of subtrees
        left_subtree = self.build_tree(
            best_split["X_left"], 
            best_split["y_left"], 
            curr_depth + 1
        )
        right_subtree = self.build_tree(
            best_split["X_right"], 
            best_split["y_right"], 
            curr_depth + 1
        )
        
        return Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_subtree,
            right=right_subtree,
            error_red=best_split["error_red"]
        )
    
    def split(self, X, y, feature_index, threshold):
        ''' 
        Function to split the data 
        '''
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        
        X_left = X[left_mask]
        y_left = y[left_mask]
        X_right = X[right_mask]
        y_right = y[right_mask]
        
        return X_left, y_left, X_right, y_right
        
    def fit(self, X, y):
        """Train the decision tree"""
        self.root = self.build_tree(X, y)
        return self
    
    def make_prediction(self, X, tree):
        """Make a prediction for a single sample"""
        if tree.value is not None:
            return tree.value
            
        feature_val = X[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(X, tree.left)
        else:
            return self.make_prediction(X, tree.right)
        
        
    def predict(self, X):
        """Predict using the decision tree"""
        X = np.array(X)
        return np.array([self.make_prediction(x, self.root) for x in X])