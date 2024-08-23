import time
from sklearn.svm import LinearSVC
import numpy as np

from sklearn.metrics import f1_score
from src.hpo.hpo_strategy import HPOStrategy
from src.ml_mod.phase1grasp_svm import Construction
from src.ml_mod.phase2grasp_svm import LocalSearch

LOCAL_SEARCH_ITERATIONS = 5
LOCAL_SEARCH_TIMELIMIT = 100000
BUILDING_PHASE_ITERATIONS = 2
BUILDING_PHASE_TIMELIMIT = 100000
INTERMEDIATE_RESULTS_SIZE = 3

class GraspHpoSVC(HPOStrategy):

    def __init__(self) -> None:
        self.phase1 = Construction(self.evaluate_solution, BUILDING_PHASE_ITERATIONS, INTERMEDIATE_RESULTS_SIZE, BUILDING_PHASE_TIMELIMIT)
        self.phase2 = LocalSearch(self.evaluate_solution, LOCAL_SEARCH_ITERATIONS, LOCAL_SEARCH_TIMELIMIT)

    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        best_results, f1_scores, cumulative_time = self.phase1.building_phase(x_train, y_train, x_val, y_val, start_time)
        phase2_start_time = time.time()
        final_model = self.phase2.local_search(best_results, x_train, y_train, x_val, y_val, start_time, f1_scores, cumulative_time, phase2_start_time)

        return final_model, f1_scores, cumulative_time

    @staticmethod
    def evaluate_solution(params, x_train, y_train, x_val, y_val, start_time):
        trained_model = LinearSVC(**params)
        trained_model.fit(x_train, y_train)
        y_pred = trained_model.predict(x_val)
        f1 = f1_score(y_val, y_pred, average="weighted")

        return trained_model, f1, time.time() - start_time
