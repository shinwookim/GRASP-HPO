# Hyperparameter Optimization Using GRASP-Based Techniques (Applied to Machine-Learning Based Intrusion Detection Systems)
This project aims to adapt [GRASP](https://en.wikipedia.org/wiki/Greedy_randomized_adaptive_search_procedure) (Greedy Randomized Adaptive Search Procedure) in optimizing hyperparameter that are used to train machine learning models in the context of Intrusion Detection Systems (IDS).

## Background 

This project is built on top of [existing work](https://ieeexplore.ieee.org/document/9452077)[^1] by *S. E. Quincozes, et al.* which used a GRASP-based technique for feature selection in IDSs built using machine learning.

[^1]: S. E. Quincozes, D. Mossé, D. Passos, C. Albuquerque, L. S. Ochi and V. F. dos Santos, "*On the Performance of GRASP-Based Feature Selection for CPS Intrusion Detection*," in IEEE Transactions on Network and Service Management, vol. 19, no. 1, pp. 614-626, March 2022, doi: 10.1109/TNSM.2021.3088763.

## Setup

Python Modules Required:

To install Python modules, run `pip install _module_name_here_` in either a Windows or Linux terminal.
If you don't already have `pip` installed, see [here](https://pip.pypa.io/en/stable/installation/) for instructions on how to do so.
If you don't already have it installed, you'll need Jupyter Notebook to run the notebook in the main directory, which can be installed with `pip install notebook` and run from a terminal or command line with `jupyter notebook`.
- `sklearn`
- `xgboost`
- `numpy`
- `ray`
- `hyperopt`
- `pandas`
- `ConfigSpace`
- `matplotlib`

Once all the required modules are installed, you can go to Cell/Run All to start testing.

To add a new HPO algorithm to the setup for testing, you'll need to write a new Class that inherits HPOStrategy, and then overwrite the `hyperparameter_optimization()` method with whatever you want the HPO algorithm to be, and have it return `best_parameters_found, f1_score_of_best_parameters`. To then have it run against the other HPO algorithms, first add a new elif statement to the `create_hpo_strategy()` method below with the string you want to use to identify the class and return the class function. Then add the HPO id string to the list in the main method with the other HPO algorithm id strings, and it should now call your new HPO algorithm method.

To add a new dataset to the setup for testing, have whatever method your use to load/create the new dataset return an instance of the `Dataset()` class shown, with a data matrix (number of samples (rows) x number of features (cols)) implemented as a Pandas DataFrame and a target array (length equal to the number of samples) implemented as a Pandas Series. Then add a new elif statement to the `load_dataset()` method in the DatasetFactory class with the string you want to use to identify the class and return the class function. Then add the dataset id string to the list in the main method with the other dataset id strings, and it should now use your new dataset in the testing routine.

## How to use
To use the framework for HPO tuning on specific datasets, you will need the following:
- Ensure you have the python modules installed as listed above
- Ensure you have cloned the repo to your local filesystem
- Ensure you are in the directory GRASP-HPO

You will first need to create a config JSON file for your desired ML algorithm and HPO modules.

Here is an example of tuning all hyperparameters of SVM using grid search (note the iteration number does not apply in this case)
```
{
    "ml": "SVM",
    "whitelist": [],
    "hpo": {
        "hpo_name": ["grid"],
        "iterations": 100
    }
}
```

You can then run the python module to generate the hyperparameters, their types and ranges with the following command:
```
python -m new_src.ml.cfg filename.json
```

This will create a corresponding json file inside the 'new_src/ml/outputs' folder, which you can either use straight away or customize ranges and values.

Following this, you can also divide your own dataset and clean it in a similar approach.

You can provide a JSON file with the following format:
```
{
    "training_size": 0.6,
    "testing_size": 0.3,
    "validation_size": 0.1,
    "label_column": " Label",
    "columns_to_drop": ["Flow ID", " Source IP", " Destination IP", " Timestamp"]
}
```

This allows you to control what columns you want to keep, what columns you want to remove and the label/class column.

Note this will coerce all values into either ints or floats.

You run this with the following:
```
python -m new_src.dataprocessor.dataloader config_file.json dataset.csv
```

This will generate three files or two if you set validation size to 0 in the 'new_src/dataprocessor/outputs' folder

You can perform the data cleaning yourself if you would like.

To finally run the HPO on your datasets, you do the following:
```
python -m new_src.hpo.hpoloader hpo_cfg.json training.csv testing.csv valiate.csv
```

This will run the HPO algorithms on the ML algorithm as specified in the `hpo_cfg.json` file on the csv files.

The outputs will be found in the 'new_src/hpo/outputs' folder. It will contain the best hyperparametes, the time taken for each iteration, and the f1 score evolution.

## Authors
This project was developed a part of the CS 1980 Capstone course at the [University of Pittsburgh](https://pitt.edu). The main contributors are: **Shinwoo Kim**, **Enoch Li**, **Jack Bellamy**, **Zi Han Ding**, **Zane Kissel**, **Gabriel Otsuka**.

This project was sponsored and directed by **Dr. Daniel Mossé**[^mosse] and **Dr. Silvio E. Quincozes**[^quincozes].

[^mosse]: Professor of Computer Science, [University of Pittsburgh](https://cs.pitt.edu), USA

[^quincozes]: Professor, Federal University of Pampa [[UNIPAMPA]](https://unipampa.edu.br), Brazil
