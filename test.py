#find a .csv file in the input directory
import os

#use a for loop to iterate through the files in the directory
for filename in os.listdir('input/'):
    #check if the file is a .csv file
    if filename.endswith('.csv'):
        #print the name of the file
        print(filename)
        #print the absolute path of the file
        print(os.path.abspath('input/' + filename))
        if "testing_data" in filename:
            print("This is the testing data")
        elif "training_data" in filename:
            print("This is the training data")
        elif "validation_data" in filename:
            print("This is the validation data")