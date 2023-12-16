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

## Authors
This project was developed a part of the CS 1980 Capstone course at the [University of Pittsburgh](https://pitt.edu). The main contributors are: **Shinwoo Kim**, **Enoch Li**, **Jack Bellamy**, **Zi Han Ding**, **Zane Kissel**, **Gabriel Otsuka**.

This project was sponsored and directed by **Dr. Daniel Mossé**[^mosse] and **Dr. Silvio E. Quincozes**[^quincozes].

[^mosse]: Professor of Computer Science, [University of Pittsburgh](https://cs.pitt.edu), USA

[^quincozes]: Professor, Federal University of Pampa [[UNIPAMPA]](https://unipampa.edu.br), Brazil
