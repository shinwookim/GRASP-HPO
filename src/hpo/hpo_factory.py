from src.hpo.benchmark.hyperband import Hyperband
from src.hpo.grasp.grasp_hpo import GraspHpo


class HPOFactory:
    @staticmethod
    def create_hpo_strategy(strategy_name):
        if strategy_name == 'GraspHpo':
            return GraspHpo()
        elif strategy_name == 'Hyperband':
            return Hyperband()
        else:
            raise ValueError("Invalid HPO strategy name")
