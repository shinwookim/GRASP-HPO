from sklearn.metrics import f1_score
import xgboost as xgb
import sys

class XGBoost():
    def __init__(self):
        self.name = 'XGBoost'
        self.hp = {
            "max_depth": {
                "type": "int",
                "range": [0, sys.maxsize]
            },
            "max_leaves": {
                "type": "int",
                "range": [0, sys.maxsize]
            },
            "max_bin": {
                "type": "int",
                "range": [0, sys.maxsize]
            },
            "grow_policy": {
                "type": "categorical",
                "range": [0, 1]
            },
            "learning_rate": {
                "type": "float",
                "range": [0.0, sys.maxsize]
            },
            "verbosity": {
                "type": "int",
                "range": [0, 3]
            },
            "objective": {
                "type": "categorical",
                "range": ["reg:squarederror", "reg:squaredlogerror", "reg:logistic"]
            },
            "booster": {
                "type": "categorical",
                "range": ["gbtree", "gblinear", "dart"]
            },
            "tree_method": {
                "type": "categorical",
                "range": ["auto", "exact", "approx", "hist", "gpu_hist"]
            },
            "n_jobs": {
                "type": "int",
                "range": [0, sys.maxsize]
            },
            "gamma": {
                "type": "float",
                "range": [0.0, 1e3]
            },
            "min_child_weight": {
                "type": "float",
                "range": [0.0, 1e3]
            },
            "max_delta_step": {
                "type": "float",
                "range": [0.0, 1e3]
            },
            "subsample": {
                "type": "float",
                "range": [0.0, 1.0]
            },
            "sampling_method": {
                "type": "categorical",
                "range": ["uniform", "gradient_based"]
            },
            "colsample_bytree": {
                "type": "float",
                "range": [0.0, 1.0]
            },
            "colsample_bylevel": {
                "type": "float",
                "range": [0.0, 1.0]
            },
            "colsample_bynode": {
                "type": "float",
                "range": [0.0, 1.0]
            },
            "reg_alpha": {
                "type": "float",
                "range": [0.0, 1e3]
            },
            "reg_lambda": {
                "type": "float",
                "range": [0.0, 1e3]
            },
            "base_score": {
                "type": "float",
                "range": [0.0, 1e3]
            },
            "random_state": {
                "type": "int",
                "range": [0, 1e3]
            },
            "eta": {
                "type": "float",
                "range": [0.0, 1e3]
            },
        }
        self.model = None

    def get_hps(self):
        return self.hp
    
    def train(self, X, y, hps):
        self.model = xgb.XGBClassifier(**hps)
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred)