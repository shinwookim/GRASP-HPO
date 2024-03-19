import numpy as np
import xgboost
from ray import tune
from ray.train import RunConfig
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
import ray
from sklearn.metrics import f1_score

from src.hpo.hpo_strategy import HPOStrategy
from ..hyperparameters import get_hyperparameters


class Hyperband(HPOStrategy):
    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
        if not ray.is_initialized():
            ray.init(log_to_driver=True, ignore_reinit_error=True)

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
            val_set = xgboost.DMatrix(data=x_val, label=y_val)

            trained_model = xgboost.train(
                config,
                train_set,
                evals=[(val_set, "eval")],
                verbose_eval=False,
                custom_metric=evaluate_f1_score,
                callbacks=[TuneReportCheckpointCallback({"f1_score": "eval-f1_score"})],
                num_boost_round=100,
            )

            return trained_model

        # Define the hyperparameter search space
        search_space = get_hyperparameters('search_space')
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
        num_class = len(np.unique(y_train))
        if num_class > 2:
            tuner_search_space["objective"] = "multi:softmax"
            tuner_search_space["num_class"] = str(num_class)

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
        best_params = results.get_best_result().config

        # Extract F1 score evolution and cumulative time
        df = results.get_dataframe()
        f1_scores = df["f1_score"].tolist()
        cumulative_time = df["time_total_s"].cumsum().tolist()

        if num_class > 2:
            best_params["objective"] = "multi:softmax"
            best_params["num_class"] = num_class

        final_model = train_xgboost(best_params)
        return final_model, f1_scores, cumulative_time
