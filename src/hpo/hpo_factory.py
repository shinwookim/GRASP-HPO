from src.hpo.benchmark.default import Default
from src.hpo.benchmark.hyperband import Hyperband
from src.hpo.benchmark.hyperopt_imp import HyperOpt
from src.hpo.benchmark.bohb import BOHB
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
        elif strategy_name == "BOHB":
            return BOHB()
        elif strategy_name == 'Default HPs':
            return Default()
        else:
            raise ValueError("Invalid HPO strategy name")
