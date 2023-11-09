from src.hpo.benchmark.default import Default
from src.hpo.benchmark.hyperband import Hyperband
from src.hpo.grasp.grasp_hpo import GraspHpo


class HPOFactory:
    @staticmethod
    def create_hpo_strategy(strategy_name):
        if strategy_name == 'GraspHpo':
            return GraspHpo()
        elif strategy_name == 'Hyperband':
            return Hyperband()
        elif strategy_name == 'None':
            return Default()
        else:
            raise ValueError("Invalid HPO strategy name")
