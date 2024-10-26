from sklearn.svm import SVR, LinearSVR

from .model import Model


class SVRModel(Model):

    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        if kernel == 'linear':
            print("Using LinearSVR")
            self.model = LinearSVR(C=C)
        else:
            self.model = SVR(kernel=kernel, C=C, gamma=gamma)

    def fit(self, X, y):
        self.model.fit(X, y,)

    def predict(self, X):
        return self.model.predict(X)
