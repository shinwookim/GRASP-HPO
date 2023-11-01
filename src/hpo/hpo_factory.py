from src.hpo.benchmark.hyperband import Hyperband
from src.hpo.benchmark.hyperopt import HyperOpt
from src.hpo.grasp.grasp_hpo import GraspHpo


class HPOFactory:
    @staticmethod
    def create_hpo_strategy(strategy_name):
        if strategy_name == 'GraspHpo':
            return GraspHpo()
        elif strategy_name == 'Hyperband':
            return Hyperband()
        elif strategy_name == "HyperOpt":
            return HyperOpt()
        else:
            raise ValueError("Invalid HPO strategy name")
