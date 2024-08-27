import random
import time
from queue import PriorityQueue

from ..hyperparameters import get_hyperparameters


class LocalSearch:

    margin = .3

    def __init__(self, evaluate, iterations, timelimit) -> None:
        self.evaluate = evaluate
        self.max_iter = iterations
        self.timelimit = timelimit
        self.hp_ranges = None
        self.int_hps = []

    def set_param(self, ranges: dict):
        self.hp_ranges = ranges
        # Identify whether to generate a random int vs float
        self.int_hps = [k for k, v in ranges.items() if isinstance(v[0], int)]

    def set_margin(self, num):
        if .05 < num < .95:
            self.margin = num

    def set_iter(self, num):
        if 0 < num < 512:
            self.max_iter = num

    def local_search(self, intermed_best_sols: PriorityQueue, x_train, y_train, x_val, y_val, start_time, f1_scores, cumulative_time, phase_start_time, verbose=False):
        self.set_param(get_hyperparameters('search_space'))

        iter = 1
        local_best_sol, local_best_score, local_best_model = {}, 0, None

        while not intermed_best_sols.empty():
            if time.time() - phase_start_time > self.timelimit:
                break

            cur = intermed_best_sols.get()

            if verbose:
                print(f'LS iteration {iter}: Best solution after phase 1: {cur[2]} Corresponding score: {cur[0]}')
            tmp_score, tmp_model, tmp_sol = self.hill_climb(cur[2], x_train, y_train, x_val, y_val, phase_start_time, start_time, f1_scores, cumulative_time)
            if verbose:
                print(f'LS iteration {iter}: Best solution after phase 2: {tmp_sol} Corresponding score: {tmp_score}')

            if tmp_score > local_best_score:
                local_best_score = tmp_score
                local_best_sol = tmp_sol
                local_best_model = tmp_model
                if verbose:
                    print(f'After LS iteration {iter}: LS found neighbor solution w/ higher f-1 mean than phase 1')

            iter += 1

        return local_best_model

    def hill_climb(self, current_hps: dict, x_train, y_train, x_val, y_val, phase_start_time, start_time, f1_scores, cumulative_time):
        best_hps = current_hps
        best_model, best_score, elapsed_time = self.evaluate(current_hps, x_train, y_train, x_val, y_val, start_time)

        for _ in range(self.max_iter):
            if time.time() - phase_start_time > self.timelimit:
                break

            neighbor_hps = self.generate_neighbor(current_hps)
            neighbor_model, neighbor_score, elapsed_time = self.evaluate(neighbor_hps, x_train, y_train, x_val, y_val, start_time)
            f1_scores.append(neighbor_score)
            cumulative_time.append(time.time() - start_time)

            if neighbor_score > best_score:
                best_hps = neighbor_hps
                best_score = neighbor_score
                best_model = neighbor_model

        return best_score, best_model, best_hps

    def generate_neighbor(self, cur_solution: dict):
        neighbor = cur_solution.copy()
        for hp in neighbor:
            if hp in ['max_features', 'bootstrap']:
                continue  # Skip tweaking non-numeric or special case parameters

            plus_minus = (self.hp_ranges[hp][1] - self.hp_ranges[hp][0]) * (self.margin / 2.0)
            param_range = (
                max((neighbor[hp] - plus_minus), self.hp_ranges[hp][0]),
                min((neighbor[hp] + plus_minus), self.hp_ranges[hp][1])
            )

            neighbor[hp] = random.randint(int(param_range[0]), int(param_range[1])) if hp in self.int_hps else random.uniform(param_range[0], param_range[1])

        return neighbor
