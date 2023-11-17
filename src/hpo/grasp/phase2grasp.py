import random
from queue import PriorityQueue

class LocalSearch:

    # for functions that tweak: this number is the +/-% for how much the tweaked parameter is changed (/2)
    margin = .3
    gen_neighbor_def = True # tweak all HPs is default | reinitialize one random HP is alternative

    # dictionary of HPs and their ranges
    def __init__ (self, evaluate, MAX) -> None:
        self.evaluate = evaluate
        self.max_iter = MAX


    def set_param (self, ranges: dict):
        self.hp_ranges = ranges
        self.int_hps = []
        # for when we need to know whether to generate a random int vs float
        for k, v in ranges.items():
            if type(v[0]) == int: self.int_hps.append(k)

    def set_margin (self, num):
        if num >= .00 and num <= 2.0: self.margin = num
        
    def set_iter (self, num):
        if num > 0 and num < 512: self.max_iter = num



    def local_search (self, intermed_best_sols: PriorityQueue, x_train, x_test, y_test, ranges: dict, verbose=False):
        self.set_param(ranges)
        
        iter = 1
        local_best_sol, local_best_score = {}, 0

        while not intermed_best_sols.empty():
        
            cur = intermed_best_sols.get()

            if verbose: print('LS iteration {}: \nBest solution after phase 1: {}\nCorresponding score: {}'.format(iter, cur[2], cur[0]))
            tmp_score, tmp_sol = self.hill_climb(cur[2], x_train, x_test, y_test)
            if verbose: print('LS iteration {}: \nBest solution after phase 2: {}\nCorresponding score: {}\n'.format(iter, tmp_sol, tmp_score))

            if tmp_score > local_best_score:
                local_best_score = tmp_score
                local_best_sol = tmp_sol
                if verbose: print('After LS iteration {}: LS found neighbor solution w/ higher f-1 mean than phase 1'.format(iter, local_best_sol, local_best_score))
            iter += 1

        return local_best_sol, local_best_score



    def hill_climb (self, cur_sol: dict, x_train, x_test, y_test):

        best_sol = cur_sol
        best_score = self.evaluate(cur_sol, x_train, x_test, y_test)

        for _ in range(self.max_iter):

            neighbor_sol = self.generate_neighbor(cur_sol)
            neighbor_score = self.evaluate(neighbor_sol, x_train, x_test, y_test)

            if neighbor_score > best_score:
                best_sol = neighbor_sol
                best_score = neighbor_score
            cur_sol = neighbor_sol

        return best_score, best_sol
    

    def random_reinit(self, current_solution):
        neighbor_solution = current_solution.copy()
        param_to_perturb = random.choice(list(neighbor_solution.keys()))
        neighbor_solution[param_to_perturb] = self.get_random_hyperparameter_value(param_to_perturb)
        return neighbor_solution

    def get_random_hyperparameter_value(self, hyperparameter):
        if hyperparameter in ['n_estimators', 'max_depth']:
            return random.randint(self.hp_ranges[hyperparameter][0], self.hp_ranges[hyperparameter][1])
        else:
            return random.uniform(self.hp_ranges[hyperparameter][0], self.hp_ranges[hyperparameter][1])
    
    
    # change all hp's of cur_solution slightly (by self.margin)
    # do not go over or under original HP bounds
    def generate_neighbor (self, cur_solution: dict):
        if self.gen_neighbor_def:

            neighbor = cur_solution.copy()
            for hp in neighbor.keys():
                plus_minus = (self.hp_ranges[hp][1] - self.hp_ranges[hp][0]) * (self.margin / 2.0)
                param_range = (max((neighbor[hp] - plus_minus), self.hp_ranges[hp][0]), \
                                min((neighbor[hp] + plus_minus), self.hp_ranges[hp][1]))
                
                if hp in self.int_hps: neighbor[hp] = random.randint(int(param_range[0]), int(param_range[1]))
                else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

            return neighbor
        
        else: return self.random_reinit(cur_solution)