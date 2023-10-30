from phase1grasp import Construction
from phase2grasp import LocalSearch
from queue import PriorityQueue


class GRASP_HPO:

    def __init__(self, evaluate) -> None:
        self.phase1 = Construction(evaluate)
        self.phase2 = LocalSearch(evaluate)
        self.hp_ranges = None


    def set_margin (self, num):
        self.phase2.set_margin(num)

    def set_iter (self, num):
        self.phase2.set_iter(num)



    def configure_ranges (self, ranges):
        self.hp_ranges = ranges


    def optimize (self) -> dict:
        if self.hp_ranges == None: raise Exception('initialize ranges of valid HPs first w/ GRASP_HPO.configure_ranges({HP}:(min, max))')
        intermediate = self.phase1.building_phase(self.hp_ranges)
        score, solution = self.phase2.local_search(intermediate, self.hp_ranges)
        return score, solution
    
    
    def get_intermediate (self) -> PriorityQueue:
        if self.hp_ranges == None: raise Exception('initialize ranges of valid HPs first')
        return self.phase1.building_phase(self.hp_ranges)
    
    def tune_intermediate (self, inter) -> dict:
        if self.hp_ranges == None: raise Exception('initialize ranges of valid HPs first')
        score, solution = self.phase2.local_search(inter, self.hp_ranges)
        return score, solution