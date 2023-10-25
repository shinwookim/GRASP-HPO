import random
from queue import PriorityQueue

class LocalSearch:

    # for functions that tweak: this number is the +/-% for how much the tweaked parameter is changed (/2)
    margin = .3
    # number of variations of generate_neighbor
    num_fn = 4
    # max iterations for hill climb phase
    max_iter = 100

    # dictionary of HPs and their ranges
    def __init__ (self, ranges: dict, evaluate) -> None:
        self.hp_ranges = ranges
        # for when we need to know whether to generate a random int vs float
        self.int_hps = []
        for k, v in ranges.items():
            if type(v[0]) == int: self.int_hps.append(k)
        self.evaluate = evaluate
        self.function = 0


    def local_search (self, intermed_best_sols: PriorityQueue):

        local_best_score, local_best_sol = self.hill_climb(intermed_best_sols.get()[2])

        outer_counter = 1
        while not intermed_best_sols.empty():
            outer_counter += 1

            temp_score, temp_sol = self.hill_climb(intermed_best_sols.get()[2])
            if local_best_score < temp_score:
                local_best_score = temp_score
                local_best_sol = temp_sol

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
    


    def manual_reinit (self, param: str):
        if param in self.int_hps: return random.randint(self.hp_ranges[hp][1], self.hp_ranges[hp][0])
        else: return random.uniform(self.hp_ranges[hp][1], self.hp_ranges[hp][0])



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
    def generate_neighbor (self, cur_solution: dict):
        if self.function == 0: return self.reinit_one(cur_solution=cur_solution)
        elif self.function == 1: return self.tweak_all(cur_solution=cur_solution)
        elif self.function == 2: return self.random_reinit(cur_solution=cur_solution)
        else: return self.random_tweaking(cur_solution=cur_solution)
    

    # fn 0
    # from Gabriel's code: randomly choose one hyperparameter to reinitialize from original range
    def reinit_one (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        param = random.choice(list(neighbor.keys()))
        param_range = (self.hp_ranges[param][0], self.hp_ranges[param][1])

        if param in self.int_hps: neighbor[param] = random.randint(param_range[0], param_range[1])
        else: neighbor[param] = random.uniform(param_range[0], param_range[1])

        return neighbor
    

    # fn 1
    # change all hp's of cur_solution slightly (by self.margin)
    def tweak_all (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        for hp in neighbor.keys():
            plus_minus = (self.hp_ranges[hp][1] - self.hp_ranges[hp][0]) * (self.margin / 2.0)
            param_range = (max((neighbor[hp] - plus_minus), self.hp_ranges[hp][0]), \
                               min((neighbor[hp] + plus_minus), self.hp_ranges[hp][1]))
            
            if hp in self.int_hps: neighbor[hp] = random.randint(int(param_range[0]), int(param_range[1]))
            else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

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
    


if __name__ == '__main__':

    ls = LocalSearch({'n_estimators': (50, 500), 'max_depth': (3, 10), 'colsample_bytree': (0.5, 1), 'reg_lambda': (0.01, 1.0), 'subsample': (0.5, 1.0)})

    for _ in range(ls.num_fn):
        # example HPs to test all configurations
        hp = {'n_estimators': 489, 'max_depth': 8, 'colsample_bytree': 0.6828694550198424, 'reg_lambda': 0.35305738774003786, 'subsample': 0.5648795831155732}
        print('Input: {}'.format(hp))
        print('Output: {}\n'.format(ls.generate_neighbor(hp)))

        ls.cycle_fn()