import numpy as np
import random

class grasp():
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
        self.name = "grasp"
        
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
        #for grasp, we need to ignore the categorical type
        for k, v in hps.items():
            if v["type"] == "float":
                self.hps[k] = random.uniform(*v["range"])
            elif v["type"] == "int":
                self.hps[k] = random.randint(*v["range"])
    
    def set_data(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
    def set_ml(self, ml):
        self.ml = ml
        
    def hyperparameter_optimization(self, num_samples=10):
        pass
    
    def get_random_hyperparameter_value(hyperparameter, hyperparameter_range):
        if hyperparameter in ['max_depth']:
            return random.randint(hyperparameter_range[0], hyperparameter_range[1])
        elif hyperparameter in ['learning_rate']:
            return random.lognormvariate(hyperparameter_range[0], hyperparameter_range[1])
        else:
            return random.uniform(hyperparameter_range[0], hyperparameter_range[1])