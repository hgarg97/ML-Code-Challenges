# https://www.youtube.com/watch?v=ltXSoduiVwY&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=3
# https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch

import numpy as np

class LinearRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Predicting Results
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculating Error (Gradients)
            dw = (1/n_samples) * np.dot(X.T, y_pred-y)
            db = (1/n_samples) * np.sum(y_pred-y)

            # Updating weights
            self.weights = self.weights - (self.lr*dw)
            self.bias = self.bias - (self.lr*db)
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred