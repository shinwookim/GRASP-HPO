import os
import importlib

#iterate over all files in the directory
for file in os.listdir(os.path.dirname(__file__)):
    #check if the file is a python file
    if file.endswith(".py") and file != "__init__.py":
        #import the module
        importlib.import_module(f".{file[:-3]}", __package__)