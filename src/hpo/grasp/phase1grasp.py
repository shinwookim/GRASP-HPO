import uuid
import random
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue
import time


DEFAULT_HYPERPARAMETERS = {
    'max_depth': 6,
    'colsample_bytree': 1,
    'reg_lambda': 1,
    'subsample': 1,
    "min_child_weight": 1,
    "learning_rate": 0.3,
    "gamma": 0
}


class Construction:
    
    def __init__(self, evaluate, iterations_quantity, intermediate_results_size, timelimit) -> None:
        self.evaluate = evaluate
        self.max_iter = iterations_quantity
        self.intermediate_results_size = intermediate_results_size
        self.timelimit = timelimit

    @staticmethod
    def get_random_hyperparameter_value(hyperparameter, hyperparameter_range):
        if hyperparameter in ['max_depth']:
            return random.randint(hyperparameter_range[0], hyperparameter_range[1])
        elif hyperparameter in ['learning_rate']:
            return random.lognormvariate(hyperparameter_range[0], hyperparameter_range[1])
        else:
            return random.uniform(hyperparameter_range[0], hyperparameter_range[1])

    def building_phase(self, x_train, x_test, y_train, y_test, search_space, start_time):
        # print('\nStarting building phase...')
        best_intermediate_combinations = PriorityQueue()

        f1_score = self.evaluate(DEFAULT_HYPERPARAMETERS, x_train, x_test, y_train, y_test)
        best_intermediate_combinations.put((f1_score, uuid.uuid4(), DEFAULT_HYPERPARAMETERS))
        f1_scores_evolution = []
        time_evolution = []
        for i in range(self.max_iter):
            if time.time() - start_time > self.timelimit:
                break

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            selected_hyperparameters = {
                'max_depth': self.get_random_hyperparameter_value('max_depth', search_space['max_depth']),
                'colsample_bytree': self.get_random_hyperparameter_value('colsample_bytree', search_space['colsample_bytree']),
                'reg_lambda': self.get_random_hyperparameter_value('reg_lambda', search_space['reg_lambda']),
                'subsample': self.get_random_hyperparameter_value('subsample', search_space['subsample']),
                "min_child_weight":  self.get_random_hyperparameter_value('min_child_weight', search_space['min_child_weight']),
                "learning_rate":  self.get_random_hyperparameter_value('learning_rate', search_space['learning_rate']),
                "gamma":  self.get_random_hyperparameter_value('gamma', search_space['gamma'])
            }

            f1_score = self.evaluate(selected_hyperparameters, x_train, x_test, y_train, y_test)

            if f1_scores_evolution.__len__() == 0 or f1_score > f1_scores_evolution[-1]:
                f1_scores_evolution.append(f1_score)
                time_evolution.append(time.time() - start_time)

            best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
            if best_intermediate_combinations.qsize() > self.intermediate_results_size:
                best_intermediate_combinations.get()

        return best_intermediate_combinations, f1_scores_evolution, time_evolution
