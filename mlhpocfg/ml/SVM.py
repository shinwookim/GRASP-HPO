from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

class SVM():
    def __init__(self):
        self.name = 'SVM'
        self.hp = {
            "penalty": {
                "type": "categorical",
                "range": ["l1", "l2"]
            },
            "loss": {
                "type": "categorical",
                "range": ["hinge", "squared_hinge"]
            },
            "dual": {
                "type": "categorical",
                "range": ["auto", True, False]
            },
            "tol": {
                "type": "float",
                "range": [0.0, None]
            },
            "C": {
                "type": "float",
                "range": [0.0, None]
            },
            "multi_class": {
                "type": "categorical",
                "range": ["ovr", "crammer_singer"]
            },
        }
        self.model = None
    
    def get_hps(self):
        return self.hp

    def train(self, X, y, hps):
        self.model = LinearSVC(**hps)
        self.model.fit(X, y)
        return self.f1_score(y, self.model.predict(X))

    def predict(self, X):
        return self.model.predict(X)
    
    def f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred)