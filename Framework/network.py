from abc import abstractmethod

class Network:

    def __init__(self):
        pass

    @abstractmethod
    def network(self, **kwargs):
        pass

    @abstractmethod
    def optimizer(self):
        pass

    @abstractmethod
    def loss(self, **kwargs):
        pass
