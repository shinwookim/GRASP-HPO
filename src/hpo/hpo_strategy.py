from abc import abstractmethod, ABC


class HPOStrategy(ABC):
    @abstractmethod
    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
        pass
