from sklearn.tree import DecisionTreeRegressor

from .model import Model


class DecisionTreeModel(Model):

    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)