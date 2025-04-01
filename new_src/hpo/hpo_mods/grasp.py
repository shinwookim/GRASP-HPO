import numpy as np
import random
import uuid
import time
from queue import PriorityQueue

class grasp():
    def __init__(self, inter_size = 5, ratio = 0.2, margins = 0.3):
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
        self.margins = margins
        self.intermediate_size = inter_size
        self.building_local_ratio = ratio
        self.priority_queue = PriorityQueue()
        
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
        # for k, v in hps.items():
        #     if v["type"] == "float":
        #         self.hps[k] = random.uniform(*v["range"])
        #     elif v["type"] == "int":
        #         self.hps[k] = random.randint(*v["range"])
    
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
        
    def evaluate(self, hyperparameters, x_train, y_train, x_val, y_val):
        f1 = self.ml.train(x_train, y_train, x_val, y_val, hyperparameters)
        return {'f1_score': f1}

    def building_phase(self, iterations):
        search_hps = {}
        f1_scores = []
        cumulative_time = []
        phase_start_time = time.time()

        for i in range(iterations):
            for k, v in self.hps.items():
                search_hps[k] = self.get_random_hyperparameter_value(k, v)
        
            f1_score = self.evaluate(self.x_train, self.y_train, self.x_val, self.y_val, search_hps)

            f1_scores.append(f1_score)
            cumulative_time.append(time.time() - phase_start_time)
            self.priority_queue.put((f1_score, uuid.uuid4(), search_hps))

            if self.priority_queue.qsize() > self.intermediate_size:
                self.priority_queue.get()

        return cumulative_time, f1_scores
    
    def hillclimb():
        pass

    def gen_neighbors(self, cur_solution):
        neighbors = {}

        for k, v in cur_solution.items():
            #Get the range of the hyperparameter
            hp_range = self.hps[k]
            #Get the type of the hyperparameter
            hp_type = hp_range['type']

            plus_minus = hp_range[1] - hp_range[0] * (self.margins / 2.0)

            if hp_type == 'int':
                neighbors[k] = random.randint(v - plus_minus, v + plus_minus)