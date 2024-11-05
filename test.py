#enumerate files in a directory
import os

#iterate over all files in the directory
for file in os.listdir(os.path.dirname(__file__)):
    print(file)