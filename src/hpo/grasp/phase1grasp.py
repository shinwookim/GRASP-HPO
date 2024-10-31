import random
import time
import uuid
from queue import PriorityQueue
from ..hyperparameters import get_hyperparameters


class Construction:
    
    def __init__(self, evaluate, iterations_quantity, intermediate_results_size, timelimit) -> None:
        self.evaluate = evaluate
        self.max_iter = iterations_quantity
        self.intermediate_results_size = intermediate_results_size
        self.timelimit = timelimit

    @staticmethod
    r

    def building_phase(self, x_train, y_train, x_val, y_val, start_time):
        # print('\nStarting building phase...')
        best_intermediate_combinations = PriorityQueue()

        default_hps = get_hyperparameters('default')
        model, f1_score, spent_time = self.evaluate(default_hps, x_train, y_train, x_val, y_val, start_time)
        f1_scores = [f1_score]
        cumulative_time = [spent_time]

        best_intermediate_combinations.put((f1_score, uuid.uuid4(), default_hps, model))
        search_space = get_hyperparameters('search_space')
        for i in range(self.max_iter):
            if time.time() - start_time > self.timelimit:
                break

            selected_hyperparameters = {
                'max_depth': self.get_random_hyperparameter_value('max_depth', search_space['max_depth']),
                'colsample_bytree': self.get_random_hyperparameter_value('colsample_bytree', search_space['colsample_bytree']),
                'reg_lambda': self.get_random_hyperparameter_value('reg_lambda', search_space['reg_lambda']),
                'subsample': self.get_random_hyperparameter_value('subsample', search_space['subsample']),
                "min_child_weight":  self.get_random_hyperparameter_value('min_child_weight', search_space['min_child_weight']),
                "learning_rate":  self.get_random_hyperparameter_value('learning_rate', search_space['learning_rate']),
                "gamma":  self.get_random_hyperparameter_value('gamma', search_space['gamma'])
            }

            model, f1_score, spent_time = self.evaluate(selected_hyperparameters, x_train, y_train, x_val, y_val, start_time)

            f1_scores.append(f1_score)
            cumulative_time.append(spent_time)

            best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters, model))
            if best_intermediate_combinations.qsize() > self.intermediate_results_size:
                best_intermediate_combinations.get()

        return best_intermediate_combinations, f1_scores, cumulative_time
