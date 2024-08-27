import numpy as np
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.hpo.hpo_strategy import HPOStrategy
from ray.train import RunConfig
from ray.tune.search.bohb import TuneBOHB
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from ..hyperparameters import get_hyperparameters


class BOHB(HPOStrategy):
    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
        def evaluate_f1_score(y_pred, y_true):
            """Compute the f1 score"""
            f1 = f1_score(y_true, y_pred, average="weighted")
            return f1

        def train_random_forest(config: dict):
            model = RandomForestClassifier(
                n_estimators=int(config.get("n_estimators", 100)),
                max_depth=int(config.get("max_depth", None)),
                min_samples_split=int(config.get("min_samples_split", 2)),
                min_samples_leaf=int(config.get("min_samples_leaf", 1)),
                max_features=config.get("max_features", 'sqrt'),
                random_state=42
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            f1 = evaluate_f1_score(y_pred, y_val)
            tune.report(f1_score=f1)
            return model

        search_space = get_hyperparameters('search_space')
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(UniformIntegerHyperparameter("n_estimators", search_space['n_estimators'][0], search_space['n_estimators'][1]))
        config_space.add_hyperparameter(UniformIntegerHyperparameter("max_depth", search_space['max_depth'][0], search_space['max_depth'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("min_samples_split", search_space['min_samples_split'][0], search_space['min_samples_split'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("min_samples_leaf", search_space['min_samples_leaf'][0], search_space['min_samples_leaf'][1]))
        config_space.add_hyperparameter(CategoricalHyperparameter("max_features", ["sqrt", "log2"]))

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
            train_random_forest,
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

        final_model = train_random_forest(best_params)
        return final_model, f1_scores, cumulative_time
