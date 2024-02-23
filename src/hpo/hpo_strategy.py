from abc import abstractmethod, ABC


class HPOStrategy(ABC):
    @abstractmethod
    def hyperparameter_optimization(self, train_set, validation_set, test_set, search_space):
        pass
