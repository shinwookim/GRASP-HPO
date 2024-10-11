import numpy as np
from sklearn.svm import LinearSVC
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from sklearn.metrics import f1_score
from src.hpo.hpo_strategy import HPOStrategy
from ray.train import RunConfig
from ray.tune.search.bohb import TuneBOHB
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from src.ml_mod.hyperparameters_svm import get_hyperparameters

class BOHBSVC(HPOStrategy):
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

        search_space = get_hyperparameters('search_space')
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(UniformFloatHyperparameter("tol", search_space['tol'][0], search_space['tol'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("C", search_space['C'][0], search_space['C'][1]))
        config_space.add_hyperparameter(UniformFloatHyperparameter("intercept_scaling", search_space['intercept_scaling'][0], search_space['intercept_scaling'][1]))
        config_space.add_hyperparameter(UniformIntegerHyperparameter("max_iter", search_space['max_iter'][0], search_space['max_iter'][1]))
        num_class = len(np.unique(y_train))

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
            train_svm_no_model,
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

        final_model = train_svm(best_params)
        return final_model, f1_scores, cumulative_time
