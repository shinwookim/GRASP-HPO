import numpy as np
import xgboost
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from sklearn.metrics import f1_score
from src.hpo.hpo_strategy import HPOStrategy
from ray.train import RunConfig
from ray.tune.search.bohb import TuneBOHB
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ..hyperparameters import get_hyperparameters


class BOHB(HPOStrategy):
    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
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

        search_space = get_hyperparameters('search_space')
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(UniformIntegerHyperparameter("max_depth", search_space['max_depth'][0], search_space['max_depth'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("subsample", search_space['subsample'][0], search_space['subsample'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("colsample_bytree", search_space['colsample_bytree'][0], search_space['colsample_bytree'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("reg_lambda", search_space['reg_lambda'][0], search_space['reg_lambda'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("min_child_weight", search_space['min_child_weight'][0], search_space['min_child_weight'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("learning_rate", search_space['learning_rate'][0], search_space['learning_rate'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("gamma", search_space['gamma'][0], search_space['gamma'][1]))

        num_class = len(np.unique(y_train))
        # Change objective for multi-class
        if num_class > 2:
            config_space.add_hyperparameter(CategoricalHyperparameter("objective", ["multi:softmax"]))
            config_space.add_hyperparameter(CategoricalHyperparameter("num_class", [str(num_class)]))

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
