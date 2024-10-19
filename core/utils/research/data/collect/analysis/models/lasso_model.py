from sklearn.linear_model import Lasso
from .model import Model


class LassoModel(Model):
    def __init__(self, alpha=1.0):
        self.model = Lasso(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
