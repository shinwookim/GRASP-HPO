from abc import abstractmethod, ABC


class HPOStrategy(ABC):
    @abstractmethod
    def hyperparameter_optimization(self, x_train, x_test, y_train, y_test):
        pass
