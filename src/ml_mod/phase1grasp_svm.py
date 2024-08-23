import random
import time
import uuid
from queue import PriorityQueue
from src.ml_mod.hyperparameters_svm import get_hyperparameters


class Construction:
    
    def __init__(self, evaluate, iterations_quantity, intermediate_results_size, timelimit) -> None:
        self.evaluate = evaluate
        self.max_iter = iterations_quantity
        self.intermediate_results_size = intermediate_results_size
        self.timelimit = timelimit

    @staticmethod
    def get_random_hyperparameter_value(hyperparameter, hyperparameter_range):
        if hyperparameter in ['max_iter']:
            return random.randint(hyperparameter_range[0], hyperparameter_range[1])
        else:
            return random.uniform(hyperparameter_range[0], hyperparameter_range[1])

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
                'tol': self.get_random_hyperparameter_value('tol', search_space['tol']),
                'C': self.get_random_hyperparameter_value('C', search_space['C']),
                'intercept_scaling': self.get_random_hyperparameter_value('intercept_scaling', search_space['intercept_scaling']),
                'max_iter': self.get_random_hyperparameter_value('max_iter', search_space['max_iter']),
            }

            model, f1_score, spent_time = self.evaluate(selected_hyperparameters, x_train, y_train, x_val, y_val, start_time)

            f1_scores.append(f1_score)
            cumulative_time.append(spent_time)

            best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters, model))
            if best_intermediate_combinations.qsize() > self.intermediate_results_size:
                best_intermediate_combinations.get()

        return best_intermediate_combinations, f1_scores, cumulative_time
