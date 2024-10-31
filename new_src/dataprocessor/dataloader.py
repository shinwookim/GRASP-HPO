import pandas as pd
import os
import pathlib
import json
import time
import sys

from sklearn.preprocessing import LabelEncoder

class Dataload():
    def __init__(self):
        self.data = None
        self.training_data = None
        self.testing_data = None
        self.validation_data = None

    def check_extension(self, path):
        if path.endswith('.csv'):
            return 'csv'
        elif path.endswith('.parquet'):
            return 'parquet'
        else:
            return None
    
    def load_data(self, path):
        extension = self.check_extension(path)
        if extension == 'csv':
            self.data = pd.read_csv(path)
        elif extension == 'parquet':
            self.data = pd.read_parquet(path)
        else:
            raise Exception('File extension not supported')
    
    def split_data(self, training_size, testing_size, validation_size):
        sum = training_size + testing_size + validation_size
        #account for floating point errors
        sum = round(sum, 10)
        if sum != 1:
            raise Exception('Split sizes do not add up to 1')
        self.training_data = self.data.sample(frac=training_size)
        self.testing_data = self.data.drop(self.training_data.index).sample(frac=testing_size)
        self.validation_data = self.data.drop(self.training_data.index).drop(self.testing_data.index)

    def export_data(self, output_dir):
        self.training_data.to_csv(output_dir + 'training_data_' + str(time.time()) + '.csv')
        self.testing_data.to_csv(output_dir + 'testing_data_' + str(time.time()) + '.csv')
        self.validation_data.to_csv(output_dir + 'validation_data_' + str(time.time()) + '.csv')

    def clean_data(self):
        self.data = self.data.dropna(axis=1)

    def set_label_column(self, column_name):
        label_encoder = LabelEncoder()
        self.data[column_name] = label_encoder.fit_transform(self.data[column_name])
        self.data["label"] = self.data[column_name]
        self.data = self.data.drop(columns=[column_name])

    def load_data_config(self, filename, data_path):
        '''
        Reads a json file with the following structure:
        {
            "training_size": 0.7,
            "testing_size": 0.2,
            "validation_size": 0.1,
            "label_column": "column_name",
            "columns_to_drop": ["column1", "column2", ...]
            "column_types": {
                "column1": "type1",
                "column2": "type2",
                ...
            }
        }
        '''
        with open(filename, 'r') as f:
            data = json.load(f)
            self.load_data(data_path)
            self.set_label_column(data['label_column'])
            self.data = self.data.drop(columns=data['columns_to_drop'])
            if 'column_types' in data:
                for column, dtype in data['column_types'].items():
                    self.data[column] = self.data[column].astype(dtype)
            #coerce to numeric
            self.data = self.data.apply(pd.to_numeric, errors='coerce')
            self.clean_data()
            self.split_data(data['training_size'], data['testing_size'], data['validation_size'])
            directory = os.path.dirname(os.path.realpath(__file__))
            self.export_data(directory + '/outputs/')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python dataloader.py config_file data_file')
        sys.exit(1)
    dataload = Dataload()
    dataload.load_data_config(sys.argv[1], sys.argv[2])