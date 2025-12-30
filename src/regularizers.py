import numpy as np
from abc import ABC, abstractmethod

class Regularizer(ABC):
    """Base class for all regularizers."""
    
    @abstractmethod
    def __call__(self, w: np.ndarray) -> float:
        """Computes the penalty value."""
        pass
    
    @abstractmethod
    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Computes the gradient of the penalty with respect to weights w."""
        pass

class L2Regularizer(Regularizer):
    """L2 Regularization (Weight Decay / Ridge)."""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        
    def __call__(self, w: np.ndarray) -> float:
        return 0.5 * self.alpha * np.sum(w**2)
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        return self.alpha * w

class L1Regularizer(Regularizer):
    """L1 Regularization (Lasso)."""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        
    def __call__(self, w: np.ndarray) -> float:
        return self.alpha * np.sum(np.abs(w))
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        # Sub-gradient of L1 norm
        return self.alpha * np.sign(w)
