from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_diabetes, load_linnerud, load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
from src.hpo.hpo_factory import HPOFactory
import time

HYPERPARAMETER_RANGES = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0)
}


def prepare_dataset(dataset):
    x = dataset.data
    y = dataset.target
    return train_test_split(x, y, test_size=0.2, random_state=1)


def evaluate_hpo(strategy_name, dataset):
    x_train, x_test, y_train, y_test = prepare_dataset(dataset)
    hpo = HPOFactory.create_hpo_strategy(strategy_name)
    start_time = time.time()
    best_trial_config, best_trial_score = hpo.hyperparameter_optimization(
        x_train, x_test, y_train, y_test, HYPERPARAMETER_RANGES
    )
    end_time = time.time()
    evaluation_time = end_time - start_time
    return best_trial_score


def plot(data):
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index="dataset", columns="hpo_strategy", values="f1_score")
    ax = pivot_df.plot(kind="bar", width=0.7, figsize=(10, 6))
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Dataset")
    ax.set_title("F1 Score Comparison: GRASP vs. Hyperband")
    plt.xticks(rotation=0)
    plt.legend(title="HPO Strategy results")
    plt.show()


if __name__ == "__main__":

    breast_cancer_dataset = load_breast_cancer()
    lbgrasp = evaluate_hpo('GraspHpo', breast_cancer_dataset)
    lbhyperband = evaluate_hpo('Hyperband', breast_cancer_dataset)

    digits_dataset = load_digits()
    dgrasp = evaluate_hpo('GraspHpo', breast_cancer_dataset)
    dhyperband = evaluate_hpo('Hyperband', breast_cancer_dataset)

    iris_dataset = load_iris()
    igrasp = evaluate_hpo('GraspHpo', iris_dataset)
    ihyperband = evaluate_hpo('Hyperband', iris_dataset)

    winedataset = load_wine()
    winegrasp = evaluate_hpo('GraspHpo', winedataset)
    winehyperband = evaluate_hpo('Hyperband', winedataset)

    data = [
        {"dataset": "Breast Cancer", "hpo_strategy": "GRASP", "f1_score": lbgrasp},
        {"dataset": "Breast Cancer", "hpo_strategy": "Hyperband", "f1_score": lbhyperband},
        {"dataset": "Digits", "hpo_strategy": "GRASP", "f1_score": dgrasp},
        {"dataset": "Digits", "hpo_strategy": "Hyperband", "f1_score": dhyperband},
        {"dataset": "Iris", "hpo_strategy": "GRASP", "f1_score": igrasp},
        {"dataset": "Iris", "hpo_strategy": "Hyperband", "f1_score": ihyperband},
        {"dataset": "Wine", "hpo_strategy": "GRASP", "f1_score": winegrasp},
        {"dataset": "Wine", "hpo_strategy": "Hyperband", "f1_score": winehyperband}
    ]

    plot(data)
