from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier

from src.hpo.grasp_hpo import GraspHpo


def prepare_dataset(dataset):
    x = dataset.data
    y = dataset.target
    return train_test_split(x, y, test_size=0.2,random_state=1)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = prepare_dataset(load_digits())
    hpo_strategy = GraspHpo()
    local_best_solution, local_best_score = hpo_strategy.hyperparameter_optimization(x_train, x_test, y_train, y_test)
    print("Optimized hyperparameters: " + str(local_best_solution))
    print("Achieved best score: " + str(local_best_score))
    xgboost_classifier = XGBClassifier(**local_best_solution)
    scores = cross_val_score(xgboost_classifier, x_train, y_train, cv=5, error_score='raise')
    print("Cross validation scores: ", scores)
    print("Scores' standard deviation: ", scores.mean(), "", scores.std())
