from src.hpo.hpo_factory import HPOStrategy
import uuid

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from queue import PriorityQueue
import random

HYPERPARAMETER_RANGES = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0)
}

LOCAL_SEARCH_ITERATIONS = 10
BUILDING_PHASE_ITERATIONS = 10


class GraspHpo(HPOStrategy):
    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test):
        best_intermediate_combinations = building_phase(x_train, x_test, y_train, y_test)

        local_best_score, local_best_solution = local_search_phase(
            best_intermediate_combinations.get()[2],
            x_train, x_test, y_train, y_test
        )
        while not best_intermediate_combinations.empty():
            temporary_score, temporary_solution = local_search_phase(
                best_intermediate_combinations.get()[2],
                x_train, x_test, y_train, y_test
            )
            if local_best_score < temporary_score:
                local_best_score = temporary_score
                local_best_solution = temporary_solution

        return local_best_solution, local_best_score


def building_phase(x_train, x_test, y_train, y_test):
    print('\nStarting building phase...')
    best_intermediate_combinations = PriorityQueue()
    intermediate_results_size = 3
    for i in range(0, BUILDING_PHASE_ITERATIONS):

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        selected_hyperparameters = {
            'n_estimators': get_random_hyperparameter_value('n_estimators'),
            'max_depth': get_random_hyperparameter_value('max_depth'),
            'colsample_bytree': get_random_hyperparameter_value('colsample_bytree'),
            'reg_lambda': get_random_hyperparameter_value('reg_lambda'),
            'subsample': get_random_hyperparameter_value('subsample')
        }

        f1_score = evaluate_solution(selected_hyperparameters, x_train, x_test, y_train, y_test)

        best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
        if best_intermediate_combinations.qsize() > intermediate_results_size:
            best_intermediate_combinations.get()

    print('Finished building phase.')
    print()
    return best_intermediate_combinations


def get_random_hyperparameter_value(hyperparameter):
    if hyperparameter in ['n_estimators', 'max_depth']:
        return random.randint(HYPERPARAMETER_RANGES[hyperparameter][0], HYPERPARAMETER_RANGES[hyperparameter][1])
    else:
        return random.uniform(HYPERPARAMETER_RANGES[hyperparameter][0], HYPERPARAMETER_RANGES[hyperparameter][1])


def evaluate_solution(params, x_train, x_test, y_train, y_test):
    xgboost_classifier = XGBClassifier(**params)
    xgboost_classifier.fit(x_train, y_train)
    y_pred = xgboost_classifier.predict(x_test)
    return f1_score(y_test, y_pred, average='weighted')


def local_search_phase(current_solution, x_train, x_test, y_train, y_test):
    print('Starting local search phase for combination: ', current_solution)

    best_solution = current_solution
    best_score = evaluate_solution(current_solution, x_train, x_test, y_train, y_test)

    for i in range(LOCAL_SEARCH_ITERATIONS):
        neighbor_solution = generate_neighbor(current_solution)
        neighbor_score = evaluate_solution(neighbor_solution, x_train, x_test, y_train, y_test)

        if neighbor_score > best_score:
            best_solution = neighbor_solution
            best_score = neighbor_score
        current_solution = neighbor_solution

    print('Best result for this combination: ', best_score)
    print()
    return best_score, best_solution


def generate_neighbor(current_solution):
    neighbor_solution = current_solution.copy()
    param_to_perturb = random.choice(list(neighbor_solution.keys()))
    neighbor_solution[param_to_perturb] = get_random_hyperparameter_value(param_to_perturb)
    return neighbor_solution
