import uuid

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from queue import PriorityQueue
import random
from phase2grasp import LocalSearch


def prepare_dataset(dataset):
    x = dataset.data
    y = dataset.target
    return train_test_split(x, y, test_size=0.2,random_state=1)


x_train, x_test, y_train, y_test = prepare_dataset(load_digits())


def evaluate_solution(params):
    xgboost_classifier = XGBClassifier(**params)
    xgboost_classifier.fit(x_train, y_train)
    y_pred = xgboost_classifier.predict(x_test)
    return f1_score(y_test, y_pred, average='weighted')


hyperparameter_ranges = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0)
}


def get_random_hyperparameter_value(hyperparameter):
    if hyperparameter in ['n_estimators', 'max_depth']:
        return random.randint(hyperparameter_ranges[hyperparameter][0], hyperparameter_ranges[hyperparameter][1])
    else:
        return random.uniform(hyperparameter_ranges[hyperparameter][0], hyperparameter_ranges[hyperparameter][1])


def building_phase():
    global x_train, x_test
    number_of_iterations = 20
    best_intermediate_combinations = PriorityQueue()
    intermediate_results_size = 3
    for i in range(0, number_of_iterations):

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

        print(selected_hyperparameters)

        f1_score = evaluate_solution(selected_hyperparameters)

        best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
        if best_intermediate_combinations.qsize() > intermediate_results_size:
            best_intermediate_combinations.get()
        print(f1_score)

    return best_intermediate_combinations


best_intermediate_combinations = building_phase()


def hill_climb(current_solution):
    max_iterations = 100
    best_solution = current_solution
    best_score = evaluate_solution(current_solution)

    for _ in range(max_iterations):

        neighbor_solution = ls.generate_neighbor(current_solution)
        neighbor_score = evaluate_solution(neighbor_solution)

        if neighbor_score > best_score:
            best_solution = neighbor_solution
            best_score = neighbor_score
        current_solution = neighbor_solution

    return best_score, best_solution


ls = LocalSearch(hyperparameter_ranges, ['n_estimators', 'max_depth'])
ls.set_fn(0)

outer_counter = 1
print(str(outer_counter) + "---------")
print()
local_best_score, local_best_solution = hill_climb(best_intermediate_combinations.get()[2])
while not best_intermediate_combinations.empty():
    outer_counter = outer_counter + 1
    print(str(outer_counter) + "---------")
    print()
    temporary_score, temporary_solution = hill_climb(best_intermediate_combinations.get()[2])
    if local_best_score < temporary_score:
        local_best_score = temporary_score
        local_best_solution = temporary_solution


print("Hyperparameters: " + str(local_best_solution))
print("Achieved best score: " + str(local_best_score))
xgboost_classifier = XGBClassifier(**local_best_solution)
scores = cross_val_score(xgboost_classifier,x_train,y_train,cv=5,error_score='raise')
print(scores)
print(scores.mean(),scores.std())