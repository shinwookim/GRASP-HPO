import numpy as np
from ray import tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.hpo.hpo_strategy import HPOStrategy
from ..hyperparameters import get_hyperparameters

class Hyperband(HPOStrategy):
    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
        def train_random_forest(config: dict):
            model = RandomForestClassifier(
                n_estimators=int(config.get("n_estimators", 100)),
                max_depth=int(config.get("max_depth", None)),
                min_samples_split=int(config.get("min_samples_split", 2)),
                min_samples_leaf=int(config.get("min_samples_leaf", 1)),
                max_features=config.get("max_features", "sqrt"),
                random_state=42
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            f1 = f1_score(y_val, y_pred, average="weighted")
            tune.report(f1_score=f1)  # Reporting the F1 score to Tune
            return model

        # Define the hyperparameter search space
        search_space = get_hyperparameters('search_space')
        tuner_search_space = {
            "n_estimators": tune.randint(search_space['n_estimators'][0], search_space['n_estimators'][1]),
            "max_depth": tune.randint(search_space['max_depth'][0], search_space['max_depth'][1] + 1),
            "min_samples_split": tune.randint(search_space['min_samples_split'][0], search_space['min_samples_split'][1]),
            "min_samples_leaf": tune.randint(search_space['min_samples_leaf'][0], search_space['min_samples_leaf'][1]),
            "max_features": tune.choice(['sqrt', 'log2'])
        }

        # Define the ASHA scheduler for hyperparameter optimization
        scheduler = ASHAScheduler(
            time_attr="training_iteration",  # Each iteration is a training step
            max_t=100,  # Maximum number of training iterations
        )

        # Config to reduce verbosity
        run_config = RunConfig(verbose=0)

        # Run the hyperparameter search
        tuner = tune.Tuner(
            train_random_forest,
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

        final_model = train_random_forest(best_params)
        return final_model, f1_scores, cumulative_time
