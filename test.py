#find a .csv file in the input directory
import os

#use a for loop to iterate through the files in the directory
for filename in os.listdir('input/'):
    #check if the file is a .csv file
    if filename.endswith('.csv'):
        #print the name of the file
        print(filename)