from abc import ABC, abstractmethod, ABCMeta

class Model(ABC):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model = None

    # Abstract methodss
    @abstractmethod
    def fit(self, X, y, params):
        pass

    @abstractmethod
    def predict(self, X, y):
        pass

    @abstractmethod
    def evaluate_acc(self, X, y):
        pass
