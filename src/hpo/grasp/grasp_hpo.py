import time
import xgboost
import numpy as np

from sklearn.metrics import f1_score
from src.hpo.hpo_strategy import HPOStrategy
from src.hpo.grasp.phase1grasp import Construction
from src.hpo.grasp.phase2grasp import LocalSearch


LOCAL_SEARCH_ITERATIONS = 5
LOCAL_SEARCH_TIMELIMIT = 20
BUILDING_PHASE_ITERATIONS = 2
BUILDING_PHASE_TIMELIMIT = 20
INTERMEDIATE_RESULTS_SIZE = 3


class GraspHpo(HPOStrategy):

    def __init__(self) -> None:
        self.phase1 = Construction(self.evaluate_solution, BUILDING_PHASE_ITERATIONS, INTERMEDIATE_RESULTS_SIZE, BUILDING_PHASE_TIMELIMIT)
        self.phase2 = LocalSearch(self.evaluate_solution, LOCAL_SEARCH_ITERATIONS, LOCAL_SEARCH_TIMELIMIT)

    def hyperparameter_optimization(self, train_set, validation_set, test_set, search_space):
        start_time = time.time()
        best_intermediate_combinations, f1_scores_evolution, time_evolution = self.phase1.building_phase(train_set, validation_set, test_set, search_space, start_time)
        local_search_start_time = time.time()
        local_best_sol, local_best_score, f1_scores_evolution2, time_evolution2 = self.phase2.local_search(best_intermediate_combinations, train_set, validation_set, test_set, search_space, start_time, local_search_start_time)
        f1_scores_evolution.extend(f1_scores_evolution2)
        time_evolution.extend(time_evolution2)

        return local_best_sol, local_best_score, (f1_scores_evolution, time_evolution)

    @staticmethod
    def evaluate_solution(params, train_set, validation_set, test_set, start_time):
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

        test_labels = test_set.get_label()
        class_quantity = len(np.unique(test_labels))
        if class_quantity > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = class_quantity

        trained_model = xgboost.train(
            params,
            train_set,
            evals=[(validation_set, "eval")],
            verbose_eval=False,
            custom_metric=evaluate_f1_score,
            num_boost_round=100,
        )

        predictions = trained_model.predict(test_set)
        y_pred = predictions
        if class_quantity == 2 and predictions.ndim == 1:
            threshold = 0.5
            y_pred = np.array([1 if pred > threshold else 0 for pred in predictions])

        # For multiclass classification, ensure predictions are class labels
        # If your model outputs probabilities (for example, using softmax), you need to convert these to class labels
        if predictions.ndim > 1:
            y_pred = np.argmax(predictions, axis=1)

        f1 = f1_score(test_labels, y_pred, average='weighted')

        return f1, time.time() - start_time
