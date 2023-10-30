import uuid
import random
from queue import PriorityQueue


class Construction:

    hp_indexes = {
        0:'n_estimators',
        1:'max_depth',
        2:'colsample_bytree',
        3:'reg_lambda',
        4:'subsample'
    }

    def __init__(self, evaluate) -> None:
        self.evaluate = evaluate


    def get_random_hyperparameter_value(self, hyperparameter):
        if hyperparameter in ['n_estimators', 'max_depth']:
            return random.randint(self.hp_range[hyperparameter][0], self.hp_range[hyperparameter][1])
        else:
            return random.uniform(self.hp_range[hyperparameter][0], self.hp_range[hyperparameter][1])

    def building_phase(self, hp_ranges: dict):
        self.hp_range = hp_ranges

        number_of_iterations = 5
        best_intermediate_combinations = PriorityQueue()
        intermediate_results_size = 2
        for _ in range(number_of_iterations):

            selected_hyperparameters = {
                'n_estimators': self.get_random_hyperparameter_value('n_estimators'),
                'max_depth': self.get_random_hyperparameter_value('max_depth'),
                'colsample_bytree': self.get_random_hyperparameter_value('colsample_bytree'),
                'reg_lambda': self.get_random_hyperparameter_value('reg_lambda'),
                'subsample': self.get_random_hyperparameter_value('subsample')
            }

            #selected_hyperparameters = grid_search()
            #print(selected_hyperparameters)

            f1_score = self.evaluate(selected_hyperparameters)

            best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
            if best_intermediate_combinations.qsize() > intermediate_results_size:
                best_intermediate_combinations.get()
            #print(f1_score)

        return best_intermediate_combinations

    
    grid_search = '''
    def grid_search(self):
        grid_split_size = 3
        hp_val_list = self.grid_search_recur(0,grid_split_size,None)[1]
        in_dict = {}
        for i in range(len(hp_val_list)):
            if i<2:#hardcoded
                in_dict[self.hp_indexes[i]]=int(hp_val_list[i])
            else:
                in_dict[self.hp_indexes[i]]=hp_val_list[i]
        return in_dict
        
    def grid_search_recur(self, hp_index,split_size,hp_val_list):
        if hp_val_list is None:
            hp_val_list = []
        #print(hp_index,hp_val_list, type(hp_val_list))
        if hp_index>=len(self.hp_ranges):
            in_dict = {}
            for i in range(len(hp_val_list)):
                if i<2:#hardcoded
                    in_dict[self.hp_indexes[i]]=int(hp_val_list[i])
                else:
                    in_dict[self.hp_indexes[i]]=hp_val_list[i]
            return (self.evaluate(in_dict),hp_val_list)
        
        min_val = self.hp_ranges[self.hp_indexes[hp_index]][0]
        max_val = self.hp_ranges[self.hp_indexes[hp_index]][1]
        delta_val = (max_val-min_val)/(split_size-1)
        
        best_eval=0.0
        best_eval_hps = []
        for i in range(split_size):
            temp = hp_val_list.copy()
            temp.append(i*(delta_val)+min_val)
            ret_val, ret_hp_val_list = self.grid_search_recur(hp_index+1,split_size,temp)
            if ret_val>best_eval:
                best_eval=ret_val
                best_eval_hps=ret_hp_val_list
        return (best_eval,best_eval_hps)
        '''