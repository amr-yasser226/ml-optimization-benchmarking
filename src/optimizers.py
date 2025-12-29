from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Applies optimization rule to update parameters."""
        pass

class VanillaSGD(Optimizer):
    def __init__(self, learning_rate: float):
        self.eta = learning_rate

    def update(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return w - self.eta * grad

class MomentumSGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9):
        self.eta = learning_rate
        self.gamma = momentum
        self.velocity = 0

    def update(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.velocity = self.gamma * self.velocity + self.eta * grad
        return w - self.velocity
