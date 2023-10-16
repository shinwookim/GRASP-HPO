from abc import abstractmethod, ABC


class HPOFactory:
    @staticmethod
    def create_hpo_strategy(strategy_name):
        if strategy_name == "HPOStrategy":
            return HPOStrategy()
        else:
            raise ValueError("Invalid HPO strategy name")


class HPOStrategy(ABC):
    @abstractmethod
    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test):
        pass
