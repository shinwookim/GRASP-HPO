import random
from queue import PriorityQueue

class LocalSearch:

    # for functions that tweak: this number is the +/-% for how much the tweaked parameter is changed (/2)
    margin = .3
    # max iterations for hill climb phase
    max_iter = 25

    # dictionary of HPs and their ranges
    def __init__ (self, evaluate) -> None:
        self.evaluate = evaluate
        self.function = 0

    def set_param (self, ranges: dict):
        self.hp_ranges = ranges
        self.int_hps = []
        # for when we need to know whether to generate a random int vs float
        for k, v in ranges.items():
            if type(v[0]) == int: self.int_hps.append(k)

    def set_margin (self, num):
        if num > .05 and num < .95: self.margin = num
        
    def set_iter (self, num):
        if num > 0 and num < 512: self.max_iter = num


    def local_search (self, intermed_best_sols: PriorityQueue, ranges: dict, verbose=False):
        self.set_param(ranges)
        

        iter = 1
        local_best_score, local_best_sol = 0, {}

        while not intermed_best_sols.empty():
        
            cur = intermed_best_sols.get()

            if verbose: print('LS iteration {}: \nBest solution after phase 1: {}\nCorresponding score: {}'.format(iter, cur[2], cur[0]))
            tmp_score, tmp_sol = self.hill_climb(cur[2])
            if verbose: print('LS iteration {}: \nBest solution after phase 2: {}\nCorresponding score: {}\n'.format(iter, tmp_sol, tmp_score))

            if tmp_score > local_best_score:
                local_best_score = tmp_score
                local_best_sol = tmp_sol
                if verbose: print('After LS iteration {}: LS found neighbor solution w/ higher k-fold mean than phase 1'.format(iter, local_best_sol, local_best_score))
            iter += 1

        return local_best_score, local_best_sol



    def hill_climb (self, cur_sol: dict):

        best_sol = cur_sol
        best_score = self.evaluate(cur_sol)

        for _ in range(self.max_iter):

            neighbor_sol = self.generate_neighbor(cur_sol)
            neighbor_score = self.evaluate(neighbor_sol)

            if neighbor_score > best_score:
                best_sol = neighbor_sol
                best_score = neighbor_score
            cur_sol = neighbor_sol

        return best_score, best_sol
    
    
    def generate_neighbor (self, cur_solution: dict):
        #if self.function == 0: return self.reinit_one(cur_solution=cur_solution)
        #elif self.function == 1: 
        return self.tweak_all(cur_solution=cur_solution)
        #elif self.function == 2: return self.random_reinit(cur_solution=cur_solution)
        #else: return self.random_tweaking(cur_solution=cur_solution)
    

    # change all hp's of cur_solution slightly (by self.margin)
    # do not go over or under original HP bounds
    def tweak_all (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        for hp in neighbor.keys():
            plus_minus = (self.hp_ranges[hp][1] - self.hp_ranges[hp][0]) * (self.margin / 2.0)
            param_range = (max((neighbor[hp] - plus_minus), self.hp_ranges[hp][0]), \
                               min((neighbor[hp] + plus_minus), self.hp_ranges[hp][1]))
            
            if hp in self.int_hps: neighbor[hp] = random.randint(int(param_range[0]), int(param_range[1]))
            else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

        return neighbor


    old = '''
    def set_fn (self, fun: int):
        self.function = fun % self.num_fn
        print('local search set to use fn {}: {}'.format(fun, self.repr_fn()))


    def cycle_fn (self):
        self.function = (self.function + 1) % self.num_fn
        print('local search set to use fn {}: {}'.format(self.function, self.repr_fn()))

    def repr_fn (self) -> str:
        if self.function == 0: return 'reinitialize one random HP'
        elif self.function == 1: return 'tweak all HPs'
        elif self.function == 2: return 'reinitialize random set of HP'
        elif self.function == 3: return 'tweak random set of HP'


    # two types of modifying:
    # reinit: choose a new hp value within original range
    # tweak: slightly modify current hp value based on self.margin
    

    # fn 0
    # from Gabriel's code: randomly choose one hyperparameter to reinitialize from original range
    def reinit_one (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        param = random.choice(list(neighbor.keys()))
        param_range = (self.hp_ranges[param][0], self.hp_ranges[param][1])

        if param in self.int_hps: neighbor[param] = random.randint(param_range[0], param_range[1])
        else: neighbor[param] = random.uniform(param_range[0], param_range[1])

        return neighbor
    

    # fn 2
    # select at random a list of HP to reinit
    def random_reinit (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        keys = list(neighbor.keys())
        params = []
        for _ in range(random.randint(1, len(keys) - 1)):
            hp = random.choice(keys)
            keys.remove(hp)
            params.append(hp)

        for hp in params:
            param_range = (self.hp_ranges[hp][0], self.hp_ranges[hp][1])

            if hp in self.int_hps: neighbor[hp] = random.randint(param_range[0], param_range[1])
            else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

        return neighbor
    
    # fn 3
    # select at random a list of HP to tweak
    def random_tweaking (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        keys = list(neighbor.keys())
        params = []
        for _ in range(random.randint(1, len(keys))):
            hp = random.choice(keys)
            keys.remove(hp)
            params.append(hp)

        for hp in params:
            plus_minus = (self.hp_ranges[hp][1] - self.hp_ranges[hp][0]) * (self.margin / 2.0)
            param_range = (max((neighbor[hp] - plus_minus), self.hp_ranges[hp][0]), \
                               min((neighbor[hp] + plus_minus), self.hp_ranges[hp][1]))
            
            if hp in self.int_hps: neighbor[hp] = random.randint(int(param_range[0]), int(param_range[1]))
            else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

        return neighbor
    '''


if __name__ == '__main__':

    ls = LocalSearch(None)
    # would be set by grasp core when ranges are being optimized
    ls.set_param({'n_estimators': (50, 500), 'max_depth': (3, 10), 'colsample_bytree': (0.5, 1), 'reg_lambda': (0.01, 1.0), 'subsample': (0.5, 1.0)})

    # example HPs to test configuration
    hp = {'n_estimators': 489, 'max_depth': 8, 'colsample_bytree': 0.6828694550198424, 'reg_lambda': 0.35305738774003786, 'subsample': 0.5648795831155732}
    print('Input: {}'.format(hp))
    print('Output: {}\n'.format(ls.generate_neighbor(hp)))