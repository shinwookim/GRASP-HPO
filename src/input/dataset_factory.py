from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, load_diabetes

from src.input.processor.ereno_processor import ErenoProcessor
from src.input.processor.canids_processor import CanidsProcessor


class DatasetFactory:
    @staticmethod
    def load_dataset(dataset_name):
        if dataset_name == 'Ereno':
            return ErenoProcessor.load_data()
        elif dataset_name == 'Breast Cancer':
            return load_breast_cancer()
        elif dataset_name == 'Digits':
            return load_digits()
        elif dataset_name == 'Iris':
            return load_iris()
        elif dataset_name == 'Wine':
            return load_wine()
        elif dataset_name == "Canids":
            return CanidsProcessor.load_data()
        else:
            raise ValueError("Invalid HPO strategy name")
