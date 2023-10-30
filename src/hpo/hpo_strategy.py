from abc import abstractmethod, ABC


class HPOStrategy(ABC):
    @abstractmethod
    def hyperparameter_optimization(self, data, labels, search_space):
        pass
