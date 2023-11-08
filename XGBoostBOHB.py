from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

import ray
from ray import tune
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
import random


HYPERPARAMETER_RANGES = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0)
}

def get_random_hyperparameter_value(hyperparameter):
    if hyperparameter in ['n_estimators', 'max_depth']:
        return tune.randint(HYPERPARAMETER_RANGES[hyperparameter][0], HYPERPARAMETER_RANGES[hyperparameter][1])
    else:
        return tune.uniform(HYPERPARAMETER_RANGES[hyperparameter][0], HYPERPARAMETER_RANGES[hyperparameter][1])


def prepare_dataset(dataset):
    x = dataset.data
    y = dataset.target
    return train_test_split(x, y, test_size=0.2, random_state=1)


def evaluate_solution(params, x_train, x_test, y_train, y_test):
    xgboost_classifier = XGBClassifier(**params)
    xgboost_classifier.fit(x_train, y_train)
    y_pred = xgboost_classifier.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    ray.train.report({'f1_score':f1})



if __name__ == "__main__":
    x_train, x_test, y_train, y_test = prepare_dataset(load_breast_cancer())
    config_space = {
        "learning_rate": tune.uniform(0.01, 0.1),
        "max_depth": tune.randint(3, 10),
        "n_estimators": tune.choice([100, 200, 300]),
        "subsample": tune.quniform(0.5, 1.0, 0.1),
        "min_child_weight": tune.randint(1, 10)
    }
    selected_hyperparameters = {
        'n_estimators': get_random_hyperparameter_value('n_estimators'),
        'max_depth': get_random_hyperparameter_value('max_depth'),
        'colsample_bytree': get_random_hyperparameter_value('colsample_bytree'),
        'reg_lambda': get_random_hyperparameter_value('reg_lambda'),
        'subsample': get_random_hyperparameter_value('subsample')
    }

    analysis = tune.run(
        tune.with_parameters(evaluate_solution, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test),
        name="bohb_optimization",
        scheduler = HyperBandForBOHB(),
        search_alg=TuneBOHB(),
        config = selected_hyperparameters,
        num_samples=100,
        metric="f1_score",
        mode="max"
    )

    best_trial = analysis.best_trial
    best_config = best_trial.config
    best_score = best_trial.last_result["f1_score"]

    print("Optimized hyperparameters:", best_config)
    print("Achieved best score:", best_score)

    xgboost_classifier = XGBClassifier(**best_config)
    scores = cross_val_score(xgboost_classifier, x_train, y_train, cv=5, error_score='raise')
    print("Cross validation scores:", scores)
    print("Scores' standard deviation:", scores.mean(), "", scores.std())
