import time

import numpy as np
import xgboost
from sklearn.metrics import f1_score

from src.hpo.hpo_strategy import HPOStrategy


class Default(HPOStrategy):
    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):

        def evaluate_f1_score(predt: np.ndarray, dtrain: xgboost.DMatrix) -> np.ndarray:
            """Compute the f1 score"""
            y = dtrain.get_label()
            if len(np.unique(y)) == 2:
                threshold = 0.5
                binary_preds = [1 if p > threshold else 0 for p in predt]
                f1 = f1_score(y, binary_preds, average="weighted")
            else:
                f1 = f1_score(y, np.argmax(predt, axis=1), average="weighted") if predt.ndim > 1 else f1_score(y, predt,
                                                                                                               average="weighted")
            return "f1_score", f1

        train_set = xgboost.DMatrix(data=x_train, label=y_train)
        val_set = xgboost.DMatrix(data=x_val, label=y_val)

        params = {}
        class_quantity = len(np.unique(y_train))
        if class_quantity > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = class_quantity

        trained_model = xgboost.train(
            params,
            train_set,
            evals=[(val_set, "eval")],
            verbose_eval=False,
            custom_metric=evaluate_f1_score,
            num_boost_round=100,
        )

        return trained_model, None, None
