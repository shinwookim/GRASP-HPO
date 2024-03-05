import time
import xgboost
import numpy as np

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
        def evaluate_f1_score(predt: np.ndarray, dtrain: xgboost.DMatrix) -> np.ndarray:
            """Compute the f1 score"""
            y = dtrain.get_label()
            if len(np.unique(y)) == 2:
                threshold = 0.5
                binary_preds = [1 if p > threshold else 0 for p in predt]
                f1 = f1_score(y, binary_preds, average="weighted")
            else:
                f1 = f1_score(y, np.argmax(predt, axis=1), average="weighted") if predt.ndim > 1 else f1_score(y, predt, average="weighted")
            return "f1_score", f1

        class_quantity = len(np.unique(y_train))
        if class_quantity > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = class_quantity

        train_set = xgboost.DMatrix(data=x_train, label=y_train)
        val_set = xgboost.DMatrix(data=x_val, label=y_val)

        evals_result = {}
        trained_model = xgboost.train(
            params,
            train_set,
            evals=[(val_set, "eval")],
            evals_result=evals_result,
            verbose_eval=False,
            custom_metric=evaluate_f1_score,
            num_boost_round=100
        )

        best_f1_score = max(evals_result['eval']['f1_score']) if 'eval' in evals_result and 'f1_score' in evals_result['eval'] else None
        return trained_model, best_f1_score, time.time() - start_time
