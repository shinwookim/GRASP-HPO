import numpy as np
from ray import tune
from ray.train import RunConfig
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

class hyperband():
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
        self.name = "hyperband"
        
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
        # Import ray at the top of your file if not already imported
        import ray
        
        # Put large datasets in Ray's object store
        x_train_ref = ray.put(self.x_train)
        y_train_ref = ray.put(self.y_train)
        x_val_ref = ray.put(self.x_val)
        y_val_ref = ray.put(self.y_val)
        
        # Reference to the ML model
        ml_ref = ray.put(self.ml)
        
        def train_model(config: dict):
            # Get data from object store
            x_train = ray.get(x_train_ref)
            y_train = ray.get(y_train_ref)
            x_val = ray.get(x_val_ref)
            y_val = ray.get(y_val_ref)
            ml = ray.get(ml_ref)
            
            # Train using retrieved data
            f1 = ml.train(x_train, y_train, x_val, y_val, config)
            return {"f1_score": f1}
        
        tuner_search_space = self.hps
        
        scheduler = ASHAScheduler(
            time_attr="training_iteration",  # Each iteration is a training step
            max_t=100,  # Maximum number of training iterations
        )
        
        run_config = RunConfig(verbose=0)
        
        tuner = tune.Tuner(
            train_model,
            tune_config=tune.TuneConfig(
                mode="max", metric="f1_score", scheduler=scheduler, num_samples=num_samples
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