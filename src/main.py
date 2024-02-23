from sklearn.model_selection import train_test_split

from src.input.dataset_factory import DatasetFactory
from src.hpo.hpo_factory import HPOFactory
import time
import xgboost
from src.output.chart_builder import plot_final_metrics, plot_evolution_through_time

HYPERPARAMETER_RANGES = {
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0),
    "min_child_weight": (1, 10),
    "learning_rate": (1e-3, 0.3),
    "gamma": (0, 1)
}


class Main:
    @staticmethod
    def prepare_dataset(dataset):
        x_temp, x_test, y_temp, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=1)

        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

        train_set = xgboost.DMatrix(data=x_train, label=y_train)
        validation_set = xgboost.DMatrix(data=x_val, label=y_val)
        test_set = xgboost.DMatrix(data=x_test, label=y_test)

        return train_set, validation_set, test_set

    @staticmethod
    def evaluate_hpo(strategy_name, dataset):
        train_set, validation_set, test_set = Main.prepare_dataset(dataset)
        hpo = HPOFactory.create_hpo_strategy(strategy_name)
        start_time = time.time()
        best_trial_config, best_trial_score, evolution_through_time = hpo.hyperparameter_optimization(
            train_set, validation_set, test_set, HYPERPARAMETER_RANGES
        )
        end_time = time.time()
        evaluation_time = end_time - start_time
        return best_trial_score, evaluation_time, evolution_through_time

    @staticmethod
    def main():
        dataset_names = ['Breast Cancer', 'Digits', 'Iris', 'Wine', 'Ereno']
        # dataset_names = ['Breast Cancer', 'Digits', 'Iris', 'Wine']
        # dataset_names = ['Ereno']
        # strategies = ['HyperOpt', 'Hyperband', 'GraspHpo', 'BOHB', 'Default HPs']
        strategies = ['GraspHpo']
        # strategies = ['BOHB']
        # strategies = ['Default HPs']

        data_final_metrics = []
        data_evolution = []

        for dataset_name in dataset_names:
            dataset = DatasetFactory.load_dataset(dataset_name)

            for strategy in strategies:
                print('Dataset: ', dataset_name, '\nStrategy: ', strategy)
                f1_score, evaluation_time, evolution_through_time = Main.evaluate_hpo(strategy, dataset)
                data_final_metrics.append({
                    "input": dataset_name,
                    "hpo_strategy": strategy,
                    "f1_score": f1_score,
                    "evaluation_time": evaluation_time
                })
                data_evolution.append({
                    "input": dataset_name,
                    "hpo_strategy": strategy,
                    "evolution_through_time": evolution_through_time
                })
        plot_evolution_through_time(data_evolution, dataset_names, "output/results")
        plot_final_metrics(data_final_metrics, "output/results")


if __name__ == "__main__":
    Main.main()
