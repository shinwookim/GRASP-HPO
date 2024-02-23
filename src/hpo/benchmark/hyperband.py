from ray.tune.integration import xgboost
from sklearn.metrics import f1_score
import numpy as np
import xgboost
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from src.hpo.hpo_strategy import HPOStrategy
from ray.train import RunConfig
from ray.train import CheckpointConfig
import time


class Hyperband(HPOStrategy):
    def hyperparameter_optimization(self,  x_train, x_test, y_train, y_test, search_space):
        def evaluate_f1_score(predt: np.ndarray, dtrain: xgboost.DMatrix) -> tuple[str, float]:
            """Compute the f1 score"""
            y = dtrain.get_label()
            if len(np.unique(y)) == 2:
                threshold = 0.5
                binary_preds = [1 if p > threshold else 0 for p in predt]
                f1 = f1_score(y, binary_preds, average="weighted")
            else:
                f1 = f1_score(y, predt, average="weighted")
            return "f1_score", f1

        def train_xgboost(config: dict):
            train_set = xgboost.DMatrix(data=x_train, label=y_train)
            test_set = xgboost.DMatrix(data=x_test, label=y_test)

            xgboost.train(
                config,
                train_set,
                evals=[(test_set, "eval")],
                verbose_eval=False,
                custom_metric=evaluate_f1_score,
                callbacks=[TuneReportCheckpointCallback({"f1_score": "eval-f1_score"})],
                num_boost_round=100,
            )

        start_time = time.time()

        # Define the hyperparameter search space
        tuner_search_space = {
            "max_depth": tune.randint(search_space['max_depth'][0], search_space['max_depth'][1]),
            "subsample": tune.uniform(search_space['subsample'][0], search_space['subsample'][1]),
            "colsample_bytree": tune.uniform(search_space['colsample_bytree'][0], search_space['colsample_bytree'][1]),
            "reg_lambda": tune.uniform(search_space['reg_lambda'][0], search_space['reg_lambda'][1]),
            "min_child_weight": tune.uniform(search_space['min_child_weight'][0], search_space['min_child_weight'][1]),
            "learning_rate": tune.loguniform(search_space['learning_rate'][0], search_space['learning_rate'][1]),
            "gamma": tune.uniform(search_space['gamma'][0], search_space['gamma'][1]),
        }
        # Change objective for multi-class
        if len(np.unique(y_train)) > 2:
            tuner_search_space["objective"] = "multi:softmax"
            tuner_search_space["num_class"] = str(len(np.unique(y_train)))

        # Define the ASHA scheduler for hyperparameter optimization
        scheduler = ASHAScheduler(
            time_attr="training_iteration",  # Each iteration is a training step
            max_t=100,  # Maximum number of training iterations
        )

        # Config to reduce verbosity
        run_config = RunConfig(verbose=0)

        # Run the hyperparameter search
        tuner = tune.Tuner(
            train_xgboost,
            tune_config=tune.TuneConfig(
                mode="max", metric="f1_score", scheduler=scheduler, num_samples=10
            ),
            param_space=tuner_search_space,
            run_config=run_config,
        )
        results = tuner.fit()
        best_param = results.get_best_result().config
        best_result = results.get_best_result().metrics["f1_score"]
        evo_time = []
        for i in range(len(results.get_dataframe()["time_total_s"])):
            if i == 0:
                evo_time.append(results.get_dataframe()["time_total_s"][i])
            else:
                evo_time.append(results.get_dataframe()["time_total_s"][i] + evo_time[i-1])
        f1_evo = results.get_dataframe()["f1_score"]
        return best_param, best_result, (f1_evo.tolist(), evo_time)
