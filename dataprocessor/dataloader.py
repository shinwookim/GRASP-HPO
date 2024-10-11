import pandas as pd
import os
import pathlib
import json
import sys

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
        if training_size + testing_size + validation_size != 1:
            raise Exception('Training, testing and validation sizes should sum to 1')
        self.training_data = self.data.sample(frac=training_size)
        self.testing_data = self.data.drop(self.training_data.index).sample(frac=testing_size)
        self.validation_data = self.data.drop(self.training_data.index).drop(self.testing_data.index)

    def export_data(self, output_dir):
        self.training_data.to_csv(output_dir + 'training_data.csv')
        self.testing_data.to_csv(output_dir + 'testing_data.csv')
        self.validation_data.to_csv(output_dir + 'validation_data.csv')

    def clean_data(self):
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()
        self.data = self.data.reset_index(drop=True)

    def set_label_column(self, column_name):
        self.data['label'] = self.data[column_name]
        self.data = self.data.drop(columns=[column_name])

    def load_data_config(self, filename):
        '''
        Reads a json file with the following structure:
        {
            "data_path": "path/to/data",
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
            self.load_data(data['data_path'])
            self.split_data(data['training_size'], data['testing_size'], data['validation_size'])
            self.clean_data()
            self.set_label_column(data['label_column'])
            self.data = self.data.drop(columns=data['columns_to_drop'])
            for column, dtype in data['column_types'].items():
                self.data[column] = self.data[column].astype(dtype)
            #find directory of this file and export to ./outputs
            directory = os.path.dirname(os.path.realpath(__file__))
            self.export_data(directory + '/outputs/')


if __name__ == '__main__':
    #load json from arguments
    data_config = sys.argv[1]
    dataload = Dataload()
    dataload.load_data_config(data_config)
    print(dataload.data.head())