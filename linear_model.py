import numpy as np

class LinearRegressor():
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):

        """
        Train the linear regression model using the normal equation.

        Parameters:
        X: numpy array, shape (n_samples, n_features)
           Input features.
        y: numpy array, shape (n_samples,)
           Target values.
        """
        # Add a column of ones to X for the bias term
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute weights using the normal equation: (X^T * X)^(-1) * X^T * y
        weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        # Separate bias and weights
        self.bias = weights[0]
        self.weights = weights[1:]
    
    def predict(self, X):
        """
        Predict target values using the trained model.

        Parameters:
        X: numpy array, shape (n_samples, n_features)
           Input features.

        Returns:
        numpy array, shape (n_samples,)
           Predicted target values.
        """
        return X.dot(self.weights) + self.bias