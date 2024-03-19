from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, load_diabetes

from src.input.processor.ereno_processor import ErenoProcessor
from src.input.processor.power_system_attack_processor import PowerSystemAttackProcessor


class DatasetFactory:
    @staticmethod
    def load_dataset(dataset_name):
        if dataset_name == 'Ereno':
            return ErenoProcessor.load_data()
        elif dataset_name == 'BreastCancer':
            return load_breast_cancer()
        elif dataset_name == 'Digits':
            return load_digits()
        elif dataset_name == 'Iris':
            return load_iris()
        elif dataset_name == 'Wine':
            return load_wine()
        elif dataset_name == 'PowerSystemAttacks':
            return PowerSystemAttackProcessor.load_data()
        else:
            raise ValueError("Invalid dataset name")
