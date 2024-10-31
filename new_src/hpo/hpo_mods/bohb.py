import numpy as np
from ray import tune
from ray.train import RunConfig
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter


class bohb():
    def __init__(self):
        self.hps = {}
        self.results = None
        self.best_params = None
        self.best_scores = None
        self.time = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.ml = None
        self.name = "bohb"
        
    def set_hps(self, hps):
        '''
        Load the hyperparameters of the model
        It is a dictionary with the following fields:
        {
            "parameter_name": {
                "type": "categorical" or "float" or "int",
                "range": [min, max]
            }
        }
        '''
        for k, v in hps.items():
            if v["type"] == "categorical":
                self.hps[k] = tune.choice(v["range"])
            elif v["type"] == "float":
                self.hps[k] = tune.uniform(*v["range"])
            elif v["type"] == "int":
                self.hps[k] = tune.randint(*v["range"])
    
    def set_data(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
    def set_ml(self, ml):
        self.ml = ml
        
    def hyperparameter_optimization(self, num_samples=10):
        def train_model(config: dict):
            f1 = self.ml.train(self.x_train, self.y_train, self.x_val, self.y_val, config)
            return {"f1_score": f1}
        tuner_search_space = self.hps

        algo = TuneBOHB(
            metric="f1_score",
            mode="max",
        )
        
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",  # Each iteration is a training step
            max_t=100,
            reduction_factor=4,
            stop_last_trials=True,# Maximum number of training iterations
        )
        
        run_config = RunConfig(verbose=0)
        
        tuner = tune.Tuner(
            train_model,
            tune_config=tune.TuneConfig(
                mode="max", metric="f1_score", scheduler=scheduler, num_samples=num_samples, search_alg=algo
            ),
            param_space=tuner_search_space,
            run_config=run_config,
        )
        
        self.results = tuner.fit()
        
        self.best_params = self.results.get_best_result().config
        
        df = self.results.get_dataframe()
        self.best_scores = df["f1_score"].tolist()
        self.time = df["time_total_s"].cumsum().tolist()
        
        return (self.best_params, self.best_scores, self.time)