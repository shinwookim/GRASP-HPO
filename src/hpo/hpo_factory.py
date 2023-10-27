from src.hpo.grasp_hpo import GraspHpo
from src.hpo.hyperband import Hyperband


class HPOFactory:
    @staticmethod
    def create_hpo_strategy(strategy_name):
        if strategy_name == "GraspHpo":
            return GraspHpo()
        if strategy_name == "Hyperband":
            return Hyperband()
        else:
            raise ValueError("Invalid HPO strategy name")
