import uuid

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from queue import PriorityQueue
import random

from src.hpo.hpo_strategy import HPOStrategy


LOCAL_SEARCH_ITERATIONS = 100
BUILDING_PHASE_ITERATIONS = 500


class GraspHpo(HPOStrategy):
    def hyperparameter_optimization(self, data, labels, search_space):
        x_train, x_test, y_train, y_test = prepare_dataset(data, labels)
        best_intermediate_combinations = building_phase(x_train, x_test, y_train, y_test, search_space)

        local_best_score, local_best_solution = local_search_phase(
            best_intermediate_combinations.get()[2],
            x_train, x_test, y_train, y_test,
            search_space
        )
        while not best_intermediate_combinations.empty():
            temporary_score, temporary_solution = local_search_phase(
                best_intermediate_combinations.get()[2],
                x_train, x_test, y_train, y_test,
                search_space
            )
            if local_best_score < temporary_score:
                local_best_score = temporary_score
                local_best_solution = temporary_solution

        return local_best_solution, local_best_score


def prepare_dataset(data, labels):
    return train_test_split(data, labels, test_size=0.2, random_state=1)


def building_phase(x_train, x_test, y_train, y_test, search_space):
    # print('\nStarting building phase...')
    best_intermediate_combinations = PriorityQueue()
    intermediate_results_size = 20
    for i in range(0, BUILDING_PHASE_ITERATIONS):

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        selected_hyperparameters = {
            'n_estimators': get_random_hyperparameter_value('n_estimators', search_space['n_estimators']),
            'max_depth': get_random_hyperparameter_value('max_depth', search_space['max_depth']),
            'colsample_bytree': get_random_hyperparameter_value('colsample_bytree', search_space['colsample_bytree']),
            'reg_lambda': get_random_hyperparameter_value('reg_lambda', search_space['reg_lambda']),
            'subsample': get_random_hyperparameter_value('subsample', search_space['subsample'])
        }

        f1_score = evaluate_solution(selected_hyperparameters, x_train, x_test, y_train, y_test)

        best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
        if best_intermediate_combinations.qsize() > intermediate_results_size:
            best_intermediate_combinations.get()

    # print('Finished building phase.')
    print()
    return best_intermediate_combinations


def get_random_hyperparameter_value(hyperparameter, hyperparameter_range):
    if hyperparameter in ['n_estimators', 'max_depth']:
        return random.randint(hyperparameter_range[0], hyperparameter_range[1])
    else:
        return random.uniform(hyperparameter_range[0], hyperparameter_range[1])


def evaluate_solution(params, x_train, x_test, y_train, y_test):
    xgboost_classifier = XGBClassifier(**params)
    xgboost_classifier.fit(x_train, y_train)
    y_pred = xgboost_classifier.predict(x_test)
    return f1_score(y_test, y_pred, average='weighted')


def local_search_phase(current_solution, x_train, x_test, y_train, y_test, search_space):
    # print('Starting local search phase for combination: ', current_solution)

    best_solution = current_solution
    best_score = evaluate_solution(current_solution, x_train, x_test, y_train, y_test)

    for i in range(LOCAL_SEARCH_ITERATIONS):
        neighbor_solution = generate_neighbor(current_solution, search_space)
        neighbor_score = evaluate_solution(neighbor_solution, x_train, x_test, y_train, y_test)

        if neighbor_score > best_score:
            best_solution = neighbor_solution
            best_score = neighbor_score
        current_solution = neighbor_solution

    # print('Best result for this combination: ', best_score)
    # print()
    return best_score, best_solution


def generate_neighbor(current_solution, search_space):
    neighbor_solution = current_solution.copy()
    param_to_perturb = random.choice(list(neighbor_solution.keys()))
    neighbor_solution[param_to_perturb] = get_random_hyperparameter_value(param_to_perturb, search_space[param_to_perturb])
    return neighbor_solution
