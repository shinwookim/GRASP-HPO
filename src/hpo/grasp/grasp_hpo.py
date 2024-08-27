import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.hpo.hpo_strategy import HPOStrategy
from src.hpo.grasp.phase1grasp import Construction
from src.hpo.grasp.phase2grasp import LocalSearch

LOCAL_SEARCH_ITERATIONS = 5
LOCAL_SEARCH_TIMELIMIT = 100000
BUILDING_PHASE_ITERATIONS = 2
BUILDING_PHASE_TIMELIMIT = 100000
INTERMEDIATE_RESULTS_SIZE = 3


class GraspHpo(HPOStrategy):

    def __init__(self) -> None:
        self.phase1 = Construction(self.evaluate_solution, BUILDING_PHASE_ITERATIONS, INTERMEDIATE_RESULTS_SIZE,
                                   BUILDING_PHASE_TIMELIMIT)
        self.phase2 = LocalSearch(self.evaluate_solution, LOCAL_SEARCH_ITERATIONS, LOCAL_SEARCH_TIMELIMIT)

    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        best_results, f1_scores, cumulative_time = self.phase1.building_phase(x_train, y_train, x_val, y_val,
                                                                              start_time)
        phase2_start_time = time.time()
        final_model = self.phase2.local_search(best_results, x_train, y_train, x_val, y_val, start_time, f1_scores,
                                               cumulative_time, phase2_start_time)

        return final_model, f1_scores, cumulative_time

    @staticmethod
    def evaluate_solution(params, x_train, y_train, x_val, y_val, start_time):
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            random_state=42
        )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        f1 = f1_score(y_val, y_pred, average="weighted")

        elapsed_time = time.time() - start_time
        return model, f1, elapsed_time
