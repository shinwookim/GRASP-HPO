import sklearn.datasets
import numpy as np
import sklearn.metrics
import xgboost
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from src.hpo.hpo_strategy import HPOStrategy
from ray.train import RunConfig
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp


class HyperOpt(HPOStrategy):
    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test, search_space):
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
            xgboost.train(
                config,
                train_set,
                evals=[(test_set, "eval")],
                verbose_eval=False,
                custom_metric=evaluate_f1_score,
                callbacks=[TuneReportCheckpointCallback({"f1_score": "eval-f1_score"})],
            )

        # Define the hyperparameter search space
        tuner_search_space = {
            "max_depth": tune.randint(search_space['max_depth'][0], search_space['max_depth'][1]),
            "subsample": tune.uniform(search_space['subsample'][0], search_space['subsample'][1]),
            "colsample_bytree": tune.uniform(search_space['colsample_bytree'][0], search_space['colsample_bytree'][1]),
            "n_estimators": tune.choice(search_space['n_estimators']),
            "reg_lambda": tune.uniform(search_space['reg_lambda'][0], search_space['reg_lambda'][1]),
            "min_child_weight": tune.uniform(search_space['min_child_weight'][0], search_space['min_child_weight'][1]),
            "learning_rate": tune.loguniform(search_space['learning_rate'][0], search_space['learning_rate'][1]),
            "gamma": tune.uniform(search_space['gamma'][0], search_space['gamma'][1]),
        }

        # Change objective for multi-class
        if len(np.unique(y_train)) > 2:
            tuner_search_space["objective"] = "multi:softmax"
            tuner_search_space["num_class"] = str(len(np.unique(y_train)))

        default_config = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "silent": True,
            "booster": 'gbtree',
            "n_jobs": 1,
            "nthread": None,
            "gamma": 0,
            "min_child_weight": 1,
            "max_delta_step": 0,
            "subsample": 1,
            "colsample_bytree": 1,
            "colsample_bylevel": 1,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "base_score": 0.5,
            "random_state":0,
            "seed": None,
            "missing": None,
        }

        # Define the HyperOpt search algorithm
        algo = HyperOptSearch(
            metric="f1_score",
            mode="max",
            # points_to_evaluate=[default_config],
            n_initial_points=4,
        )

        algo = ConcurrencyLimiter(algo, max_concurrent=2)

        # Config to reduce verbosity
        run_config = RunConfig(verbose=0)

        # Run the hyperparameter search
        tuner = tune.Tuner(
            train_xgboost,
            tune_config=tune.TuneConfig(
                mode="max", metric="f1_score", num_samples=10, search_alg=algo
            ),
            param_space=tuner_search_space,
            run_config=run_config,
        )
        results = tuner.fit()
        best_param = results.get_best_result().config
        best_result = results.get_best_result().metrics["f1_score"]
        return best_param, best_result