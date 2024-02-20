import time
import xgboost
import numpy as np

from sklearn.metrics import f1_score
from src.hpo.hpo_strategy import HPOStrategy
from src.hpo.grasp.phase1grasp import Construction
from src.hpo.grasp.phase2grasp import LocalSearch


LOCAL_SEARCH_ITERATIONS = 2
LOCAL_SEARCH_TIMELIMIT = 20
BUILDING_PHASE_ITERATIONS = 2
BUILDING_PHASE_TIMELIMIT = 20
INTERMEDIATE_RESULTS_SIZE = 5


class GraspHpo(HPOStrategy):

    def __init__(self) -> None:
        self.phase1 = Construction(self.evaluate_solution, BUILDING_PHASE_ITERATIONS, INTERMEDIATE_RESULTS_SIZE, BUILDING_PHASE_TIMELIMIT)
        self.phase2 = LocalSearch(self.evaluate_solution, LOCAL_SEARCH_ITERATIONS, LOCAL_SEARCH_TIMELIMIT)

    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test, search_space):
        start_time = time.time()
        best_intermediate_combinations, f1_scores_evolution, time_evolution = self.phase1.building_phase(x_train, x_test, y_train, y_test, search_space, start_time)
        local_search_start_time = time.time()
        local_best_sol, local_best_score, f1_scores_evolution2, time_evolution2 = self.phase2.local_search(best_intermediate_combinations, x_train, x_test, y_train, y_test, search_space, start_time, local_search_start_time)
        f1_scores_evolution.extend(f1_scores_evolution2)
        time_evolution.extend(time_evolution2)

        return local_best_sol, local_best_score, (f1_scores_evolution, time_evolution)

    @staticmethod
    def evaluate_solution(params, x_train, x_test, y_train, y_test, start_time):
        # Define the custom evaluation function inside evaluate_solution to record time of each iteration
        class TimeEvaluationCallback(xgboost.callback.TrainingCallback):
            def __init__(self):
                super().__init__()
                self.times = []

            def after_iteration(self, model, epoch, evals_log):
                current_time = time.time()
                self.times.append(current_time - start_time)
                return False  # Return False to continue training

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
        test_set = xgboost.DMatrix(data=x_test, label=y_test)

        evals_result = {}

        time_callback = TimeEvaluationCallback()

        xgboost.train(
            params,
            train_set,
            evals=[(test_set, "eval")],
            verbose_eval=False,
            custom_metric=evaluate_f1_score,
            num_boost_round=100,
            evals_result=evals_result,
            callbacks=[time_callback]
        )
        round_times = time_callback.times

        f1_scores_per_round = evals_result['eval']['f1_score']

        return f1_scores_per_round, round_times
