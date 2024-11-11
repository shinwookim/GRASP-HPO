from . import hpo_mods
import json
import pandas
import time
import os
import sys
from ..mlhpocfg import ml
import importlib
from pkgutil import iter_modules

class loaderHPO():
    def __init__(self):
        self.hpo = []
        self.num_samples = None
        self.hps = {}
        self.train = None
        self.test = None
        self.val = None
        self.ml = None
        self.results = {}
    
    def load_config(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.ml_import(data['ml'])
            self.hps = data['hps']
            self.num_samples = data['hpo']['iterations']
            for hpo in data['hpo']['hpo_name']:
                self.load_hpo(hpo)
    
    def load_data(self, filename):
        #loads a csv file
        with open(filename, 'r') as f:
            reader = pandas.read_csv(f)
            #split data into features and target in the column called 'label'
            data = reader.drop(columns=['label'])
            target = reader['label']
            return [data, target]

        
    def load_train(self, filename):
        self.train = self.load_data(filename)
    
    def load_test(self, filename):
        self.test = self.load_data(filename)
    
    def load_val(self, filename):
        self.val = self.load_data(filename)
    
    def load_hpo(self, hpo_name):
        if self.hpo_search(hpo_name):
            print(hpo_name)
            obj_module = getattr(hpo_mods, hpo_name)
            obj_class = getattr(obj_module, hpo_name)
            obj = obj_class()
            self.add_hpo(obj)
        else:
            raise Exception('HPO not found')
        
    def add_hpo(self, hpo):
        hpo.set_data(self.train[0], self.train[1], self.test[0], self.test[1])
        hpo.set_hps(self.hps)
        hpo.set_ml(self.ml)
        self.hpo.append(hpo)
    
    def ml_search(self, search_name):
        for importer, modname, ispkg in iter_modules(ml.__path__):
            if modname == search_name:
                return True
        return False
    
    def ml_import(self, ml_name):
        if self.ml_search(ml_name):
            obj_module = getattr(ml, ml_name)
            obj_class = getattr(obj_module, ml_name)
            obj = obj_class()
            self.set_ml(obj)
            return True
        else:
            raise Exception('ML not found')
        
    def set_ml(self, ml):
        self.ml = ml
        
    def hpo_search(self, search_name):
        for importer, modname, ispkg in iter_modules(hpo_mods.__path__):
            if modname == search_name:
                return True
        return False

    def run_hpo(self):
        for hpo in self.hpo:
            print(hpo.name)
            results = hpo.hyperparameter_optimization(self.num_samples)
            best_params = results[0]
            best_scores = results[1]
            time = results[2]
            print(hpo.name)
            self.results[hpo.name] = {
                "best_params": best_params,
                "best_scores": best_scores,
                "time": time
            }
        
    def export_results(self, directory = None):
        filename = 'results_' + str(time.time()) + '.json'
        if directory is None:
            directory = os.path.dirname(os.path.realpath(__file__))
            directory = directory + '/outputs/'
        if os.path.exists(directory + filename):
            print('File already exists')
            raise Exception('File already exists')
        else:
            with open(directory + filename, 'w+') as f:
                f.write(json.dumps(self.results, indent=4))
                
                
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python hpoloader.py config_file train_file test_file val_file')
        sys.exit(1)
    hpoload = loaderHPO()
    hpoload.load_train(sys.argv[2])
    hpoload.load_test(sys.argv[3])
    hpoload.load_val(sys.argv[4])
    hpoload.load_config(sys.argv[1])
    hpoload.run_hpo()
    hpoload.export_results()