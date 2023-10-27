import sklearn.datasets
import numpy as np
import sklearn.metrics
import xgboost
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from src.hpo.hpo_strategy import HPOStrategy


class Hyperband(HPOStrategy):
    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test):
        def evaluate_f1_score(predt: np.ndarray, dtrain: xgboost.DMatrix) -> np.ndarray:
            """Compute the f1 score"""
            y = dtrain.get_label()
            threshold = 0.5  # You can adjust this threshold as needed
            binary_preds = [1 if p > threshold else 0 for p in predt]
            f1 = sklearn.metrics.f1_score(y, binary_preds)
            return ("f1_score", f1)

        def train_xgboost(config: dict):
            train_set = xgboost.DMatrix(data=x_train, label=y_train)
            test_set = xgboost.DMatrix(data=x_test, label=y_test)

            xgboost.train(
                config,
                train_set,
                config["num_boost_round"],
                evals=[(test_set, "eval")],
                verbose_eval=True,
                custom_metric=evaluate_f1_score,
                callbacks=[TuneReportCheckpointCallback({"f1_score": "eval-f1_score"})],
            )

        # Define the hyperparameter search space
        search_space = {
            "disable_default_eval_metric": 1,
            "objective": "binary:logistic",
            "max_depth": tune.randint(3, 10),
            "min_child_weight": tune.uniform(1, 10),
            "subsample": tune.uniform(0.5, 1.0),
            "colsample_bytree": tune.uniform(0.5, 1.0),
            "learning_rate": tune.loguniform(1e-3, 1.0),
            "gamma": tune.uniform(0, 1),
            "n_estimators": tune.choice([50, 500]),
            "num_boost_round": tune.choice([10, 50, 100]),
        }

        # Define the ASHA scheduler for hyperparameter optimization
        scheduler = ASHAScheduler(
            max_t=100,  # Maximum number of training iterations
            grace_period=10,  # Minimum number of iterations for each trial
            reduction_factor=2,  # Factor by which trials are pruned
        )

        # Run the hyperparameter search
        tuner = tune.Tuner(
            train_xgboost,
            tune_config=tune.TuneConfig(
                mode="max", metric="f1_score", scheduler=scheduler, num_samples=10
            ),
            param_space=search_space,
        )
        results = tuner.fit()
        best_param = results.get_best_result().config
        best_result = results.get_best_result().metrics["f1_score"]
        return best_param, best_result
