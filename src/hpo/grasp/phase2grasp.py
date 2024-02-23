import random
from queue import PriorityQueue
import time


class LocalSearch:

    # for functions that tweak: this number is the +/-% for how much the tweaked parameter is changed (/2)
    margin = .3

    # dictionary of HPs and their ranges
    def __init__(self, evaluate, iterations, timelimit) -> None:
        self.int_hps = None
        self.hp_ranges = None
        self.evaluate = evaluate
        self.max_iter = iterations
        self.timelimit = timelimit

    def set_param(self, ranges: dict):
        self.hp_ranges = ranges
        self.int_hps = []
        # for when we need to know whether to generate a random int vs float
        for k, v in ranges.items():
            if type(v[0]) == int: self.int_hps.append(k)

    def set_margin(self, num):
        if .05 < num < .95: self.margin = num

    def set_iter(self, num):
        if 0 < num < 512: self.max_iter = num

    def local_search(self, intermed_best_sols: PriorityQueue, x_train, x_test, y_train, y_test, ranges: dict, start_time, phase_start_time, verbose=False):
        self.set_param(ranges)

        iter = 1
        local_best_sol, local_best_score = {}, 0
        f1_scores_evolution = []
        time_evolution = []

        while not intermed_best_sols.empty():
            if time.time() - phase_start_time > self.timelimit:
                break

            cur = intermed_best_sols.get()

            if verbose: print('LS iteration {}: \nBest solution after phase 1: {}\nCorresponding score: {}'.format(iter, cur[2], cur[0]))
            tmp_score, tmp_sol = self.hill_climb(cur[2], x_train, x_test, y_train, y_test, phase_start_time, start_time, f1_scores_evolution, time_evolution)
            if verbose: print('LS iteration {}: \nBest solution after phase 2: {}\nCorresponding score: {}\n'.format(iter, tmp_sol, tmp_score))

            if tmp_score > local_best_score:
                local_best_score = tmp_score
                local_best_sol = tmp_sol
                if verbose: print('After LS iteration {}: LS found neighbor solution w/ higher f-1 mean than phase 1'.format(iter, local_best_sol, local_best_score))
            iter += 1

        return local_best_sol, local_best_score, f1_scores_evolution, time_evolution

    def hill_climb(self, cur_sol: dict, x_train, x_test, y_train, y_test, phase_start_time, start_time, f1_scores_evolution, time_evolution):
        best_sol = cur_sol
        f1_scores_per_round, round_times = self.evaluate(cur_sol, x_train, x_test, y_train, y_test, start_time)
        f1_scores_evolution.extend(f1_scores_per_round)
        time_evolution.extend(round_times)
        best_score = max(f1_scores_per_round)

        for _ in range(self.max_iter):

            if time.time() - phase_start_time > self.timelimit:
                break

            neighbor_sol = self.generate_neighbor(cur_sol)
            neighbor_scores, neighbor_times = self.evaluate(neighbor_sol, x_train, x_test, y_train, y_test, start_time)
            f1_scores_evolution.extend(neighbor_scores)
            time_evolution.extend(neighbor_times)
            neighbor_score = max(neighbor_scores)
            if neighbor_score > best_score:
                best_sol = neighbor_sol
                best_score = neighbor_score
            cur_sol = neighbor_sol

        return best_score, best_sol

    # change all hp's of cur_solution slightly (by self.margin)
    # do not go over or under original HP bounds
    def generate_neighbor(self, cur_solution: dict):

        neighbor = cur_solution.copy()
        for hp in neighbor.keys():
            if hp in ['objective', 'num_class', 'n_estimators']:
                continue

            plus_minus = (self.hp_ranges[hp][1] - self.hp_ranges[hp][0]) * (self.margin / 2.0)
            param_range = (
                max((neighbor[hp] - plus_minus), self.hp_ranges[hp][0]),
                min((neighbor[hp] + plus_minus), self.hp_ranges[hp][1])
            )

            if hp in self.int_hps: neighbor[hp] = random.randint(int(param_range[0]), int(param_range[1]))
            else: neighbor[hp] = random.uniform(param_range[0], param_range[1])

        return neighbor
