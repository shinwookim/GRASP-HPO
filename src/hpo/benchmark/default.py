from sklearn.metrics import f1_score
import numpy as np
from xgboost import XGBClassifier

from src.hpo.hpo_strategy import HPOStrategy


class Default(HPOStrategy):
    def hyperparameter_optimization(self,  x_train, x_test, y_train, y_test, search_space):
        params = {}
        class_quantity = len(np.unique(y_train))
        if class_quantity > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = str(class_quantity)

        xgboost_classifier = XGBClassifier(**params)
        xgboost_classifier.fit(x_train, y_train)
        y_pred = xgboost_classifier.predict(x_test)
        return params, f1_score(y_test, y_pred, average='weighted')
