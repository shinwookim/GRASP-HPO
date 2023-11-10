import uuid
import random
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue


class Construction:

    
    def __init__(self, evaluate, MAX) -> None:
        self.evaluate = evaluate
        self.max_iter = MAX


    def get_random_hyperparameter_value(self, hyperparameter, hyperparameter_range):
        if hyperparameter in ['n_estimators', 'max_depth']:
            return random.randint(hyperparameter_range[0], hyperparameter_range[1])
        else:
            return random.uniform(hyperparameter_range[0], hyperparameter_range[1])

    hp_indexes = {
        0:'n_estimators',
        1:'max_depth',
        2:'colsample_bytree',
        3:'reg_lambda',
        4:'subsample'
    }

    def grid_search(self, split_size, x_train, x_test, y_train, y_test, search_space):
        best_combination = PriorityQueue()

        grid_split_size = split_size
        hp_val_list = self.grid_search_recur(0,grid_split_size,None,x_train, x_test, y_train, y_test, search_space)[1]
        in_dict = {}
        for i in range(len(hp_val_list)):
            if i<2:#hardcoded
                in_dict[self.hp_indexes[i]]=int(hp_val_list[i])
            else:
                in_dict[self.hp_indexes[i]]=hp_val_list[i]

        f1_score = self.evaluate(in_dict, x_train, x_test, y_train, y_test)
        best_combination.put((f1_score, uuid.uuid4(), in_dict))
        return best_combination

    def grid_search_recur(self, hp_index,split_size,hp_val_list, x_train, x_test, y_train, y_test, search_space):
        if hp_val_list is None:
            hp_val_list = []
        #print(hp_index,hp_val_list, type(hp_val_list))
        if hp_index>=len(search_space):
            in_dict = {}
            for i in range(len(hp_val_list)):
                if i<2:#hardcoded
                    in_dict[self.hp_indexes[i]]=int(hp_val_list[i])
                else:
                    in_dict[self.hp_indexes[i]]=hp_val_list[i]
            return (self.evaluate(in_dict, x_train, x_test, y_train, y_test),hp_val_list)

        min_val = search_space[self.hp_indexes[hp_index]][0]
        max_val = search_space[self.hp_indexes[hp_index]][1]
        delta_val = (max_val-min_val)/(split_size-1)

        best_eval=0.0
        best_eval_hps = []
        for i in range(split_size):
            temp = hp_val_list.copy()
            temp.append(i*(delta_val)+min_val)
            ret_val, ret_hp_val_list = self.grid_search_recur(hp_index+1,split_size,temp,x_train, x_test, y_train, y_test, search_space)
            if ret_val>best_eval:
                best_eval=ret_val
                best_eval_hps=ret_hp_val_list
        return (best_eval,best_eval_hps)
    
    def random_search(self, x_train, x_test, y_test, search_space):
        best_intermediate_combinations = PriorityQueue()
        intermediate_results_size = 20
        for i in range(self.max_iter):

            #scaler = StandardScaler()
            #x_train = scaler.fit_transform(x_train)
            #x_test = scaler.transform(x_test)

            selected_hyperparameters = {
                'n_estimators': self.get_random_hyperparameter_value('n_estimators', search_space['n_estimators']),
                'max_depth': self.get_random_hyperparameter_value('max_depth', search_space['max_depth']),
                'colsample_bytree': self.get_random_hyperparameter_value('colsample_bytree', search_space['colsample_bytree']),
                'reg_lambda': self.get_random_hyperparameter_value('reg_lambda', search_space['reg_lambda']),
                'subsample': self.get_random_hyperparameter_value('subsample', search_space['subsample'])
            }

            f1_score = self.evaluate(selected_hyperparameters, x_train, x_test, y_test)

            best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
            if best_intermediate_combinations.qsize() > intermediate_results_size:
                best_intermediate_combinations.get()

        # print('Finished building phase.')
        #print()
        return best_intermediate_combinations


    def building_phase(self, x_train, x_test, y_test, search_space):
        # print('\nStarting building phase...')
        return self.random_search(x_train, x_test, y_test, search_space)
        #return self.grid_search(2,x_train, x_test, y_train, y_test, search_space)