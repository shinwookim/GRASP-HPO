import uuid
import random

from queue import PriorityQueue


class Construction:

    
    def __init__(self, evaluate, MAX) -> None:
        self.evaluate = evaluate
        self.max_iter = MAX


    def get_random_hyperparameter_value(self, hyperparameter, hyperparameter_range):
        if hyperparameter in ['n_estimators', 'max_depth']:
            return random.randint(hyperparameter_range[0], hyperparameter_range[1])
        else:
            return random.uniform(hyperparameter_range[0], hyperparameter_range[1])


    def building_phase(self, x_train, x_test, y_train, y_test, search_space):

        best_intermediate_combinations = PriorityQueue()
        intermediate_results_size = 2
        for _ in range(self.max_iter):

            selected_hyperparameters = {
                'n_estimators': self.get_random_hyperparameter_value('n_estimators', search_space['n_estimators']),
                'max_depth': self.get_random_hyperparameter_value('max_depth', search_space['max_depth']),
                'colsample_bytree': self.get_random_hyperparameter_value('colsample_bytree', search_space['colsample_bytree']),
                'reg_lambda': self.get_random_hyperparameter_value('reg_lambda', search_space['reg_lambda']),
                'subsample': self.get_random_hyperparameter_value('subsample', search_space['subsample'])
            }

            f1_score = self.evaluate(selected_hyperparameters, x_train, x_test, y_train, y_test)

            best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
            if best_intermediate_combinations.qsize() > intermediate_results_size:
                best_intermediate_combinations.get()

        return best_intermediate_combinations