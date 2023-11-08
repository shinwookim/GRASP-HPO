import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.hpo.hpo_strategy import HPOStrategy
from src.hpo.grasp.phase1grasp import Construction
from src.hpo.grasp.phase2grasp import LocalSearch


LOCAL_SEARCH_ITERATIONS = 10
BUILDING_PHASE_ITERATIONS = 10


class GraspHpo(HPOStrategy):

    def __init__(self) -> None:
        self.phase1 = Construction(self.evaluate_solution, BUILDING_PHASE_ITERATIONS)
        self.phase2 = LocalSearch(self.evaluate_solution, LOCAL_SEARCH_ITERATIONS)

    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test, search_space):

        best_intermediate_combinations = self.phase1.building_phase(x_train, x_test, y_train, y_test, search_space)

        return self.phase2.local_search(best_intermediate_combinations, x_train, x_test, y_train, y_test, search_space)

    @staticmethod
    def evaluate_solution(params, x_train, x_test, y_train, y_test):
        class_quantity = len(np.unique(y_train))
        if class_quantity > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = str(class_quantity)

        xgboost_classifier = XGBClassifier(**params)
        xgboost_classifier.fit(x_train, y_train)
        y_pred = xgboost_classifier.predict(x_test)
        return f1_score(y_test, y_pred, average='weighted')
