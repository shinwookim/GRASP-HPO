from sklearn.model_selection import train_test_split

from src.input.dataset_factory import DatasetFactory
from src.hpo.hpo_factory import HPOFactory
from src.output.chart_builder import plot_final_metrics, plot_evolution_through_time
import numpy as np
import xgboost
from sklearn.metrics import f1_score


class Main:
    @staticmethod
    def prepare_dataset(dataset):
        x_temp, x_test, y_temp, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def evaluate_hpo(strategy_name, dataset):
        x_train, y_train, x_val, y_val, x_test, y_test = Main.prepare_dataset(dataset)
        hpo = HPOFactory.create_hpo_strategy(strategy_name)
        best_model, f1_scores_cumulative, cumulative_time = hpo.hyperparameter_optimization(x_train, y_train, x_val, y_val)

        final_score = Main.get_final_model_results(best_model, x_test, y_test)
        return final_score, f1_scores_cumulative, cumulative_time

    @staticmethod
    def get_final_model_results(trained_model, x_test, y_test):
        test_set = xgboost.DMatrix(data=x_test, label=y_test)

        y_pred_prob = trained_model.predict(test_set)

        class_quantity = len(np.unique(y_test))
        if class_quantity == 2:
            threshold = 0.5
            y_pred = (y_pred_prob > threshold).astype(int)
        else:
            y_pred = np.argmax(y_pred_prob, axis=1) if y_pred_prob.ndim > 1 else y_pred_prob

        final_f1_score = f1_score(y_test, y_pred, average='weighted')

        return final_f1_score

    @staticmethod
    def main():
        dataset_names = ['BreastCancer', 'Digits', 'Iris', 'Wine', 'Ereno', 'PowerSystemAttacks']
        # dataset_names = ['Breast Cancer']
        strategies = ['HyperOpt', 'GraspHpo', 'BOHB', 'Default HPs']
        # strategies = ['Hyperband']

        data_final_metrics = []
        data_evolution = []

        for dataset_name in dataset_names:
            dataset = DatasetFactory.load_dataset(dataset_name)

            for strategy in strategies:
                print('Dataset: ', dataset_name, '\nStrategy: ', strategy)
                final_score, f1_scores, cumulative_time = Main.evaluate_hpo(strategy, dataset)
                data_final_metrics.append({
                    "input": dataset_name,
                    "hpo_strategy": strategy,
                    "f1_score": final_score
                })
                if strategy == "Default HPs":
                    continue
                data_evolution.append({
                    "input": dataset_name,
                    "hpo_strategy": strategy,
                    "f1_scores": f1_scores,
                    "cumulative_time": cumulative_time
                })
        plot_evolution_through_time(data_evolution, dataset_names, "output/results")
        plot_final_metrics(data_final_metrics, "output/results")


if __name__ == "__main__":
    Main.main()
