from sklearn.neighbors import KNeighborsRegressor

from .model import Model


class KNNModel(Model):
    def __init__(self, k=5):
        self.knn = KNeighborsRegressor(n_neighbors=k)

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        return self.knn.predict(X)
