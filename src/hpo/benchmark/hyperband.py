from ray.tune.integration import xgboost
from sklearn.metrics import f1_score
import sklearn.datasets
import numpy as np
import sklearn.metrics
import xgboost
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from src.hpo.hpo_strategy import HPOStrategy
from ray.train import RunConfig


class Hyperband(HPOStrategy):
    def hyperparameter_optimization(self,  x_train, x_test, y_train, y_test, search_space):
        def evaluate_f1_score(predt: np.ndarray, dtrain: xgboost.DMatrix) -> np.ndarray:
            """Compute the f1 score"""
            y = dtrain.get_label()
            if len(np.unique(y)) == 2:
                threshold = 0.5
                binary_preds = [1 if p > threshold else 0 for p in predt]
                f1 = sklearn.metrics.f1_score(y, binary_preds, average="weighted")
            else:
                f1 = sklearn.metrics.f1_score(y, predt, average="weighted")
            return ("f1_score", f1)

        def train_xgboost(config: dict):
            train_set = xgboost.DMatrix(data=x_train, label=y_train)
            test_set = xgboost.DMatrix(data=x_test, label=y_test)
            num_boost_round = config.pop("num_boost_round")
            xgboost.train(
                config,
                train_set,
                num_boost_round,
                evals=[(test_set, "eval")],
                verbose_eval=True,
                custom_metric=evaluate_f1_score,
                callbacks=[TuneReportCheckpointCallback({"f1_score": "eval-f1_score"})],
            )

        # Define the hyperparameter search space
        search_space = {
            "max_depth": tune.randint(search_space['max_depth'][0], search_space['max_depth'][1]),
            "subsample": tune.uniform(search_space['subsample'][0], search_space['subsample'][1]),
            "colsample_bytree": tune.uniform(search_space['colsample_bytree'][0], search_space['colsample_bytree'][1]),
            "n_estimators": tune.randint(search_space['n_estimators'][0], search_space['n_estimators'][1]),
            "reg_lambda": tune.uniform(search_space['reg_lambda'][0], search_space['reg_lambda'][1]),
            "log_level": "ERROR",
            "num_boost_round": tune.choice([10, 50, 100])
        }
        # Change objective for multi-class
        if len(np.unique(y_train)) > 2:
            search_space["objective"] = "multi:softmax"
            search_space["num_class"] = str(len(np.unique(y_train)))

        # Define the ASHA scheduler for hyperparameter optimization
        scheduler = ASHAScheduler(
            max_t=100,  # Maximum number of training iterations
            grace_period=20,  # Minimum number of iterations for each trial
            reduction_factor=2,  # Factor by which trials are pruned
        )

        # Config to reduce verbosity
        run_config = RunConfig(verbose=0)

        # Run the hyperparameter search
        tuner = tune.Tuner(
            train_xgboost,
            tune_config=tune.TuneConfig(
                mode="max", metric="f1_score", scheduler=scheduler, num_samples=10
            ),
            param_space=search_space,
            run_config=run_config,
        )
        results = tuner.fit()
        best_param = results.get_best_result().config
        best_result = results.get_best_result().metrics["f1_score"]
        return best_param, best_result
