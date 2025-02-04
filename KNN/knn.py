import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predictions(x) for x in X]
        return predictions
    
    def _predictions(self, x):
        # Compute the Distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the Closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority Vote
        most_common = Counter(k_nearest_labels).most_common()

        return most_common[0][0]