from . import ml
from ..hpo import hpo_mods as hpo
import json
import os
import sys
import time
import importlib
from pkgutil import iter_modules

class MLConfig():
    def __init__(self):
        self.ml = None
        self.hps = {}
        self.hpocfg = {}
    
    def set_ml(self, ml):
        self.ml = ml

    def set_hps(self, hps):
        self.hps = hps

    def set_hpocfg(self, hpocfg):
        self.hpocfg = hpocfg
    
    def get_hpocfg(self):
        return self.hpocfg

    def get_ml(self):
        return self.ml
    
    def get_hps(self):
        return self.hps
    
    def export_config(self):
        return json.dumps({'ml': self.ml.name, 'hps': self.hps, 'hpo': self.hpocfg}, indent=4)
    
    def export_file(self, output_dir):
        #set filename
        filename = self.ml.name + '_' + str(time.time()) + '_cfg.json'
        #check if filename already exists
        if os.path.exists(output_dir + filename):
            print('File already exists')
            raise Exception('File already exists')
        else:
            with open(output_dir + filename, 'w+') as f:
                f.write(self.export_config())

    def import_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.ml_import(data['ml'])
            self.hps = data['hps']

    def ml_search(self, search_name):
        for importer, modname, ispkg in iter_modules(ml.__path__):
            if modname == search_name:
                return True
        return False
    
    def ml_import(self, ml_name):
        if self.ml_search(ml_name):
            #create class object dynamically, module is already imported
            obj_module = getattr(ml, ml_name)
            obj_class = getattr(obj_module, ml_name)
            obj = obj_class()
            self.set_ml(obj)
            self.set_hps(obj.get_hps())
            return True
        else:
            raise Exception('ML not found')
        
    def read_input_config(self, filename, output_dir = None):
        '''
        Opens a json file with the following structure:
        {
            "ml": "ml_name",
            "whitelist": ["hp1", "hp2", "hp3"]
            "hpo": {
                "hpo_name": ["hpo1", "hpo2", "hpo3"],
                "iterations": 100
            }
        }
        '''
        #open json file
        with open(filename, 'r') as f:
            data = json.load(f)
            self.ml_import(data['ml'])
            #check if whitelist and blacklist are defined
            whitelist = data['whitelist'] if 'whitelist' in data else []
            #get hyperparameters
            hps = self.ml.get_hps()
            #filter hyperparameters
            if len(whitelist) > 0:
                hps = {k: v for k, v in hps.items() if k in whitelist}
            else:
                hps = {k: v for k, v in hps.items()}
            self.set_hps(hps)
            if 'hpo' in data:
                self.set_hpocfg(data['hpo'])
            else:
                #if not, look through the hpo folder and get the number of modules
                hpo_list = []
                for importer, modname, ispkg in iter_modules(hpo.__path__):
                    hpo_list.append(modname)
                self.set_hpocfg({'hpo_name': hpo_list, 'iterations': 100})

            #find directory of this file
            if output_dir is None:
                output_dir = os.path.dirname(os.path.realpath(__file__)) + '/outputs/'
            #export config
            self.export_file(output_dir)
            
    def read_input_config_string(self, inputStr, output_dir = None):
        '''
        Reads a json string with the following structure:
        {
            "ml": "ml_name",
            "whitelist": ["hp1", "hp2", "hp3"]
            "hpo": {
                "hpo_name": ["hpo1", "hpo2", "hpo3"],
                "iterations": 100
            }
        }
        '''
        data = json.loads(inputStr)
        self.ml_import(data['ml'])
        #check if whitelist and blacklist are defined
        whitelist = data['whitelist'] if 'whitelist' in data else []
        #get hyperparameters
        hps = self.ml.get_hps()
        #filter hyperparameters
        if len(whitelist) > 0:
            hps = {k: v for k, v in hps.items() if k in whitelist}
        else:
            hps = {k: v for k, v in hps.items()}
        self.set_hps(hps)
        if 'hpo' in data:
            self.set_hpocfg(data['hpo'])
        else:
            #if not, look through the hpo folder and get the number of modules
            hpo_list = []
            for importer, modname, ispkg in iter_modules(hpo.__path__):
                hpo_list.append(modname)
            self.set_hpocfg({'hpo_name': hpo_list, 'iterations': 100})

        #find directory of this file
        if output_dir is None:
            output_dir = os.path.dirname(os.path.realpath(__file__))
        #export config
        self.export_file(output_dir)


if __name__ == '__main__':
    #check if input file is provided
    if len(sys.argv) < 2:
        print('Usage: python cfg.py input_file')
        sys.exit(1)
    #create object
    mlcfg = MLConfig()
    #read input file
    mlcfg.read_input_config(sys.argv[1])
    #print output
    print(mlcfg.export_config())