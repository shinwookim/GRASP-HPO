import ml
import hpo.hpo_mods as hpo
import json
import os
import sys
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
        filename = self.ml.name + '_cfg.json'
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
            module = importlib.import_module('ml.' + ml_name)
            obj_class = getattr(module, ml_name)
            obj = obj_class()
            self.set_ml(obj)
            self.set_hps(obj.get_hps())
            return True
        else:
            raise Exception('ML not found')
        
    def read_input_config(self, filename):
        '''
        Opens a json file with the following structure:
        {
            "ml": "ml_name",
            "whitelist": ["hp1", "hp2", "hp3"]
            "blacklist": ["hp4", "hp5", "hp6"]
            "hpo": {
                "hpo_name": ["hpo1", "hpo2", "hpo3"],
                "iterations": 100
            }
        }
        '''
        #open json file
        with open(filename, 'r') as f:
            data = json.load(f)
            whitelist = data['whitelist']
            blacklist = data['blacklist']
            #import ml
            self.ml_import(data['ml'])
            #if whitelist is empty, keep all hps
            if not whitelist:
                whitelist = self.hps.keys()
            #if blacklist is empty, keep all hps
            if not blacklist:
                blacklist = []
            #filter hps
            self.hps = {k: v for k, v in self.hps.items() if k in whitelist}
            self.hps = {k: v for k, v in self.hps.items() if k not in blacklist}
            #check if hpo is defined
            if 'hpo' in data:
                self.set_hpocfg(data['hpo'])
            else:
                #if not, look through the hpo folder and get the number of modules
                hpo_list = []
                for importer, modname, ispkg in iter_modules(hpo.__path__):
                    hpo_list.append(modname)
                self.set_hpocfg({'hpo_name': hpo_list, 'iterations': 100})

            #find directory of this file
            directory = os.path.dirname(os.path.realpath(__file__))
            #export config
            self.export_file(directory + '/outputs/')


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