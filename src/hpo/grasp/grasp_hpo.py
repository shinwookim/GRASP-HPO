import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.hpo.hpo_strategy import HPOStrategy
from src.hpo.grasp.phase1grasp import Construction
from src.hpo.grasp.phase2grasp import LocalSearch


HILL_CLIMB_ITERATIONS = 1
LOCAL_SEARCH_ITERATIONS = 160
LOCAL_SEARCH_TIMELIMIT = 20
BUILDING_PHASE_ITERATIONS = 40
BUILDING_PHASE_TIMELIMIT = 40
INTERMEDIATE_RESULTS_SIZE = 40


class GraspHpo(HPOStrategy):

    def __init__(self) -> None:
        self.phase1 = Construction(self.evaluate_solution, BUILDING_PHASE_ITERATIONS, INTERMEDIATE_RESULTS_SIZE, BUILDING_PHASE_TIMELIMIT)
        self.phase2 = LocalSearch(self.evaluate_solution, HILL_CLIMB_ITERATIONS, LOCAL_SEARCH_ITERATIONS, LOCAL_SEARCH_TIMELIMIT)

    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test, search_space):
        start_time = time.time()
        best_intermediate_combinations, f1_scores_evolution, time_evolution = self.phase1.building_phase(x_train, x_test, y_train, y_test, search_space, start_time)
        local_search_start_time = time.time()
        local_best_sol, local_best_score, f1_scores_evolution2, time_evolution2 = self.phase2.local_search(best_intermediate_combinations, x_train, x_test, y_train, y_test, search_space, start_time, local_search_start_time)
        f1_scores_evolution.extend(f1_scores_evolution2)
        time_evolution.extend(time_evolution2)
        #remove the first element of the time_evolution list
        time_evolution.pop(0)
        #remove the first element of the f1_scores_evolution list
        f1_scores_evolution.pop(0)

        return local_best_sol, local_best_score, (f1_scores_evolution, time_evolution)

    @staticmethod
    def evaluate_solution(params, x_train, x_test, y_train, y_test):
        class_quantity = len(np.unique(y_train))
        if class_quantity > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = str(class_quantity)
        params['n_estimators'] = 100

        xgboost_classifier = XGBClassifier(**params)
        xgboost_classifier.fit(x_train, y_train)
        y_pred = xgboost_classifier.predict(x_test)
        return f1_score(y_test, y_pred, average='weighted')