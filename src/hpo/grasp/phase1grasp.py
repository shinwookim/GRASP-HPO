import uuid
import random
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue


class Construction:
    
    def __init__(self, evaluate, iterations_quantity, intermediate_results_size) -> None:
        self.evaluate = evaluate
        self.max_iter = iterations_quantity
        self.intermediate_results_size = intermediate_results_size

    @staticmethod
    def get_random_hyperparameter_value(hyperparameter, hyperparameter_range):
        if hyperparameter in ['max_depth']:
            return random.randint(hyperparameter_range[0], hyperparameter_range[1])
        elif hyperparameter in ['n_estimators']:
            return random.choice(hyperparameter_range)
        elif hyperparameter in ['learning_rate']:
            return random.lognormvariate(hyperparameter_range[0], hyperparameter_range[1])
        else:
            return random.uniform(hyperparameter_range[0], hyperparameter_range[1])

    def building_phase(self, x_train, x_test, y_train, y_test, search_space):
        # print('\nStarting building phase...')
        best_intermediate_combinations = PriorityQueue()
        for i in range(self.max_iter):

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            selected_hyperparameters = {
                'n_estimators': self.get_random_hyperparameter_value('n_estimators', search_space['n_estimators']),
                'max_depth': self.get_random_hyperparameter_value('max_depth', search_space['max_depth']),
                'colsample_bytree': self.get_random_hyperparameter_value('colsample_bytree', search_space['colsample_bytree']),
                'reg_lambda': self.get_random_hyperparameter_value('reg_lambda', search_space['reg_lambda']),
                'subsample': self.get_random_hyperparameter_value('subsample', search_space['subsample']),
                "min_child_weight":  self.get_random_hyperparameter_value('min_child_weight', search_space['min_child_weight']),
                "learning_rate":  self.get_random_hyperparameter_value('learning_rate', search_space['learning_rate']),
                "gamma":  self.get_random_hyperparameter_value('gamma', search_space['gamma'])
            }

            f1_score = self.evaluate(selected_hyperparameters, x_train, x_test, y_train, y_test)

            best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
            if best_intermediate_combinations.qsize() > self.intermediate_results_size:
                best_intermediate_combinations.get()

        return best_intermediate_combinations
