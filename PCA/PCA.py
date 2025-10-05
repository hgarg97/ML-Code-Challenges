# https://www.youtube.com/watch?v=Rjr62b_h7S4&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=8
# https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):

        # Mean Centering
        self.mean = np.mean(X, axis=0)
        X = X-self.mean

        # Covariance (needs samples as column: that is why transpose)
        cov = np.cov(X.T)

        # Eigenvectors, Eignenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Choosing first n components
        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        # Project data
        X = X-self.mean

        return np.dot(X, self.components.T)
