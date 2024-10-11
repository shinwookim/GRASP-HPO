import hpo_mods
import json
import pandas
import os
import mlhpocfg
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
            module = importlib.import_module('hpo.hpo_mods.' + hpo_name)
            obj_class = getattr(module, hpo_name)
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
            module = importlib.import_module('ml.' + ml_name)
            obj_class = getattr(module, ml_name)
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
            results = hpo.hyperparameter_optimization(self.num_samples)
            best_params = results.best_params
            best_scores = results.best_scores
            time = results.time
        
            self.results[hpo] = {
                "best_params": best_params,
                "best_scores": best_scores,
                "time": time
            }
        
    def export_results(self, output_dir):
        filename = 'results.json'
        if os.path.exists(output_dir + filename):
            print('File already exists')
            raise Exception('File already exists')
        else:
            with open(output_dir + filename, 'w+') as f:
                f.write(json.dumps(self.results, indent=4))
                
                
if __name__ == '__main__':
    hpo = loaderHPO()
    hpo.load_config('SVM_cfg.json')
    hpo.load_train('train.csv')
    hpo.load_test('test.csv')
    hpo.load_val('val.csv')
    hpo.run_hpo()
    hpo.export_results('results/')