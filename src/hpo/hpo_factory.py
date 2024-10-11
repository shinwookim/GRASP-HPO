from src.hpo.benchmark.default import Default
from src.hpo.benchmark.hyperband import Hyperband
from src.hpo.benchmark.hyperopt_imp import HyperOpt
from src.hpo.benchmark.bohb import BOHB
from src.hpo.grasp.grasp_hpo import GraspHpo
from src.ml_mod.hyperopt_imp_svm import HyperOptSVC
from src.ml_mod.bohb_svm import BOHBSVC
from src.ml_mod.hyperband_svm import HyperbandSVC
from src.ml_mod.grasp_svm import GraspHpoSVC

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
        elif strategy_name == "HyperOptSVC":
            return HyperOptSVC()
        elif strategy_name == "BOHBSVC":
            return BOHBSVC()
        elif strategy_name == "HyperbandSVC":
            return HyperbandSVC()
        elif strategy_name == "GraspHpoSVC":
            return GraspHpoSVC()
        else:
            raise ValueError("Invalid HPO strategy name")
