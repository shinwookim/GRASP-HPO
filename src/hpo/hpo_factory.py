from src.hpo.grasp_hpo import GraspHpo


class HPOFactory:
    @staticmethod
    def create_hpo_strategy(strategy_name):
        if strategy_name == 'GraspHpo':
            return GraspHpo()
        else:
            raise ValueError("Invalid HPO strategy name")
