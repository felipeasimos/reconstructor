from copy import copy
import numpy as np
from numpy.random import default_rng
rng = default_rng() # Class to generate random numbers of numpy

'''
`points_param`:     Minimum number of data points to estimate parameters
`max_iterations`:   Maximum iterations allowed
`threshold`:        Threshold value to determine if points are fit well
`points_fit`:       Number of close data points required to assert model fits well
`model`:            class implementing `fit` and `predict`
`loss`:             function of `y_true` and `y_pred` that returns a vector
`metric`:           function of `y_true` and `y_pred` and returns a float
'''
class RANSAC:
    def __init__(self, points_param=10, max_iterations=100, threshold=0.05, points_fit=10, model=None, loss=None, metric=None):
        self.points_param = points_param       
        self.max_iterations = max_iterations   
        self.threshold = threshold             
        self.points_fit = points_fit               
        self.model = model                     
        self.loss = loss                       
        self.metric = metric                   
        self.best_fit = None
        self.best_error = np.inf

    def fit(self, X, y):
        for _ in range(self.k):
            ids = rng.permutation(X.shape[0]) # Permutation on the number of points

            maybe_inliers = ids[: self.points_param] # Get the first 10 points
            maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

            # Test if the loss it's smaller than the threshold
            # [ids][self.points_param] --> Label 
            thresholded = (
                self.loss(y[ids][self.points_param :], maybe_model.predict(X[ids][self.points_param :]))
                < self.threshold
            )

            inlier_ids = ids[self.points_param :][np.flatnonzero(thresholded).flatten()]

            # If the number of inliners gets bigger than the mininum required 
            if inlier_ids.size > self.points_fit:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])

                this_error = self.metric(
                    y[inlier_points], better_model.predict(X[inlier_points])
                )

                # Update the error and model if it's better 
                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = better_model

        return self

    def predict(self, X):
        return self.best_fit.predict(X)


# Loss 
def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

# Metric 
def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]

# Model
# TODO[David]: Buscar outros possíveis modelos que possam ser utilizados
class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params