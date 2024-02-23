import numpy as np
import xgboost
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from sklearn.metrics import f1_score
from ray.train import CheckpointConfig
from src.hpo.hpo_strategy import HPOStrategy
from ray.train import RunConfig
from ray.tune.search.bohb import TuneBOHB
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
import time


class BOHB(HPOStrategy):
    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test, search_space):
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
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(UniformIntegerHyperparameter("max_depth", search_space['max_depth'][0], search_space['max_depth'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("subsample", search_space['subsample'][0], search_space['subsample'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("colsample_bytree", search_space['colsample_bytree'][0], search_space['colsample_bytree'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("reg_lambda", search_space['reg_lambda'][0], search_space['reg_lambda'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("min_child_weight", search_space['min_child_weight'][0], search_space['min_child_weight'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("learning_rate", search_space['learning_rate'][0], search_space['learning_rate'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("gamma", search_space['gamma'][0], search_space['gamma'][1]))

        # Change objective for multi-class
        if len(np.unique(y_train)) > 2:
            config_space.add_hyperparameter(CategoricalHyperparameter("objective", ["multi:softmax"]))
            config_space.add_hyperparameter(CategoricalHyperparameter("num_class", [str(len(np.unique(y_train)))]))

        # Define the BOHB search algorithm
        bohb_search = TuneBOHB(space=config_space, metric="f1_score", mode="max")

        # Define the HyperBandForBOHB scheduler
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=100,
            reduction_factor=4,
            stop_last_trials=True,
        )

        # Config to reduce verbosity
        run_config = RunConfig(verbose=0)

        # Run the hyperparameter search
        tuner = tune.Tuner(
            train_xgboost,
            tune_config=tune.TuneConfig(
                mode="max", metric="f1_score", scheduler=scheduler, num_samples=10, search_alg=bohb_search
            ),
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
