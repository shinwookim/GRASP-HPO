from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, load_diabetes

from src.input.processor.ereno_processor import erenoProcessor
from src.input.processor.canids_processor import canidsProcessor


class DatasetFactory:
    @staticmethod
    def load_dataset(dataset_name):
        if dataset_name.lower() == 'ereno':
            return erenoProcessor.load_data()
        elif dataset_name.lower() == 'breast cancer':
            return load_breast_cancer()
        elif dataset_name.lower() == 'digits':
            return load_digits()
        elif dataset_name.lower() == 'iris':
            return load_iris()
        elif dataset_name.lower() == 'wine':
            return load_wine()
        elif dataset_name.lower() == 'canids':
            processor = canidsProcessor()
            return processor.load_data()
        else:
            raise ValueError("Invalid HPO strategy name")
