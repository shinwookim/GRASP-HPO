import random

class LocalSearch:

    margin = .3
    num_fn = 6

    # dictionary of HPs and their ranges
    # list of HPs that are int; assumed rest of HPs are float
    def __init__ (self, ranges: dict, ints: list) -> None:
        self.hp_ranges = ranges
        self.int_hps = ints
        self.function = 0


    def set_fn (self, fun: int):
        self.function = fun % self.num_fn
        print('local search set to use fn {}'.format(fun))


    def cycle_fn (self):
        self.function = (self.function + 1) % self.num_fn
        print('local search set to use fn {}'.format(self.function))


    def generate_neighbor (self, cur_solution: dict):
        if self.function == 0: return self.reinit_one_random(cur_solution=cur_solution)
        elif self.function == 1: return self.tweak_all(cur_solution=cur_solution)
        elif self.function == 2: return self.reinit_all_but_one(cur_solution=cur_solution)
        elif self.function == 3: return self.tweak_all_but_one(cur_solution=cur_solution)
        elif self.function == 4: return self.random_selection(cur_solution=cur_solution)
        elif self.function == 5: return self.random_tweaking(cur_solution=cur_solution)

    # fn 0
    # from Gabriel's code: randomly choose one hyperparameter to reinitialize from entire range
    def reinit_one_random (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        param = random.choice(list(neighbor.keys()))
        param_range = (self.hp_ranges[param][0], self.hp_ranges[param][1])

        if param in self.int_hps: neighbor[param] = random.randint(param_range[0], param_range[1])
        else: neighbor[param] = random.uniform(param_range[0], param_range[1])

        return neighbor
    

    # fn 1
    # change all hp's of cur_solution, but only by self.margin
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
    # randomly choose 1 HP to remain: reinitialize the rest
    def reinit_all_but_one (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        keys = list(neighbor.keys())
        keys.remove(random.choice(keys))

        for hp in keys:
            param_range = (self.hp_ranges[hp][0], self.hp_ranges[hp][1])

            if hp in self.int_hps: neighbor[hp] = random.randint(param_range[0], param_range[1])
            else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

        return neighbor
    

    # fn 3
    # randomly choose 1 HP to remain: tweak the rest by self.margin
    def tweak_all_but_one (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        keys = list(neighbor.keys())
        keys.remove(random.choice(keys))

        for hp in keys:
            plus_minus = (self.hp_ranges[hp][1] - self.hp_ranges[hp][0]) * (self.margin / 2.0)
            param_range = (max((neighbor[hp] - plus_minus), self.hp_ranges[hp][0]), \
                               min((neighbor[hp] + plus_minus), self.hp_ranges[hp][1]))
            
            if hp in self.int_hps: neighbor[hp] = random.randint(int(param_range[0]), int(param_range[1]))
            else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

        return neighbor
    

    # fn 4
    # select at random a list of HP to reinit
    def random_selection (self, cur_solution: dict):

        neighbor = cur_solution.copy()
        keys = list(neighbor.keys())
        params = []
        for _ in range(random.randint(1, len(keys))):
            hp = random.choice(keys)
            keys.remove(hp)
            params.append(hp)

        for hp in params:
            param_range = (self.hp_ranges[hp][0], self.hp_ranges[hp][1])

            if hp in self.int_hps: neighbor[hp] = random.randint(param_range[0], param_range[1])
            else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

        return neighbor
    
    # fn 5
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

    ls = LocalSearch({'n_estimators': (50, 500), 'max_depth': (3, 10), 'colsample_bytree': (0.5, 1), 'reg_lambda': (0.01, 1.0), 'subsample': (0.5, 1.0)}, ['n_estimators', 'max_depth'])

    for _ in range(ls.num_fn):
        hp = {'n_estimators': 489, 'max_depth': 8, 'colsample_bytree': 0.6828694550198424, 'reg_lambda': 0.35305738774003786, 'subsample': 0.5648795831155732}
        print('Input: {}'.format(hp))
        print('Output: {}\n'.format(ls.generate_neighbor(hp)))

        ls.cycle_fn()