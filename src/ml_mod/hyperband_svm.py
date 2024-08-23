import numpy as np
from sklearn.svm import LinearSVC
from ray import tune
from ray.train import RunConfig
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import f1_score

from src.hpo.hpo_strategy import HPOStrategy
from src.ml_mod.hyperparameters_svm import get_hyperparameters


class HyperbandSVC(HPOStrategy):
    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
        def train_svm_no_model(config: dict):
            model = LinearSVC(**config)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            f1 = f1_score(y_val, y_pred, average="weighted")

            return {"f1_score": f1}
        
        def train_svm(config: dict):
            model = LinearSVC(**config)
            model.fit(x_train, y_train)
            return model
        
        # Define the hyperparameter search space
        search_space = get_hyperparameters('search_space')
        tuner_search_space = {
            'tol': tune.uniform(*search_space['tol']),
            'C': tune.uniform(*search_space['C']),
            'intercept_scaling': tune.uniform(*search_space['intercept_scaling']),
            'max_iter': tune.randint(*search_space['max_iter'])
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
            train_svm_no_model,
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

        final_model = train_svm(best_params)
        return final_model, f1_scores, cumulative_time
