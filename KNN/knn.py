# https://www.youtube.com/watch?v=rTEtEy5o3X0&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=2
# https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch

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

        return most_common[0][0] # Returns only the name of the label .. [[label_name1, count1], [], .....]

#  CODE RUN ONE FUNCTION 

# def _ed(x1, x2):
#     return np.sqrt(np.sum((x2-x1)**2))

# def _knn(X_train, y_train, X_test, k):

#     res = []
#     for x1 in X_test:
#         distances = []
#         for x2 in X_train:
#             distances.append(_ed(x1, x2))

#         k_indices = np.argsort(distances)[:k]

#         k_nearest_labels = []
#         for i in k_indices:
#             k_nearest_labels.append(y_train[i])

#         most_common = Counter(k_nearest_labels).most_common()

#         res.append(most_common[0][0])

#     return res


# import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split

# iris = datasets.load_iris()
# X, y = iris.data, iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# predictions = _knn(X_train, y_train, X_test, k=2)

# print(predictions)

# acc = np.sum(predictions == y_test) / len(y_test)

# print(acc)