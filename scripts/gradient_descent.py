import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scripts.evaluation import cMSE_error, gradient_cMSE_error
import pandas as pd


class CustomLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, iterations=1000, learning_rate=0.01, 
                 regularization=None, alpha=0.1, threshold=1e-6):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.alpha = alpha
        self.threshold = threshold
        self.weights = None
        self.losses = []

    def gradient_descent(self, X, y, c, iterations, learning_rate, regularization, alpha, threshold):
        weights = np.random.rand(X.shape[1]) * 0.01  
        losses = []

        for i in range(iterations):
            # Compute predictions
            y_hat = X @ weights

            # Calculate cMSE loss
            loss = cMSE_error(y, y_hat, c)  
            losses.append(loss)

            # Check for convergence
            if i > 0 and abs(losses[-2] - losses[-1]) < threshold:  
                print(f"Convergence reached at iteration {i+1} with loss change below threshold.")
                break

            # Compute gradient
            grad_err = gradient_cMSE_error(y, y_hat, c, X)
            grad = -grad_err

            # Apply regularization
            if regularization == 'ridge':
                grad += 2 * alpha * weights  # Ridge (L2)
            elif regularization == 'lasso':
                grad += alpha * np.sign(weights)  # Lasso (L1)

            # Update weights
            weights -= learning_rate * grad

            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i+1}: Loss = {loss}")

        return weights, losses

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y_survival = y["SurvivalTime"]
            y_censored = y["Censored"]
        else:
            raise ValueError("Target 'y' must be a DataFrame with 'SurvivalTime' and 'Censored' columns.")

        # Train the model using gradient descent
        self.weights, self.losses = self.gradient_descent(
            X, y_survival, y_censored, 
            iterations=self.iterations, 
            learning_rate=self.learning_rate, 
            regularization=self.regularization, 
            alpha=self.alpha, 
            threshold=self.threshold
        )
        return self

    def predict(self, X):
        """ Predict using the learned weights """
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call `fit` before prediction.")
        return X @ self.weights