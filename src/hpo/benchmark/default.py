from sklearn.metrics import f1_score
import numpy as np
import xgboost

from src.hpo.hpo_strategy import HPOStrategy
import time


class Default(HPOStrategy):
    def hyperparameter_optimization(self,  x_train, x_test, y_train, y_test, search_space):
        start_time = time.time()

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
                f1 = f1_score(y, np.argmax(predt, axis=1), average="weighted") if predt.ndim > 1 else f1_score(y, predt,
                                                                                                               average="weighted")
            return "f1_score", f1

        params = {}
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

        return params, max(f1_scores_per_round), (f1_scores_per_round, round_times)
