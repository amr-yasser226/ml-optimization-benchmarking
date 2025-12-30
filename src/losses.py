import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    """Base class for all loss functions."""
    
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the loss value."""
        pass
    
    @abstractmethod
    def gradient(self, X: np.ndarray, y_true: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Computes the gradient of the loss with respect to weights w."""
        pass

class MSELoss(Loss):
    """Mean Squared Error loss for regression tasks."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0.5 * np.mean((y_true - y_pred)**2)
    
    def gradient(self, X: np.ndarray, y_true: np.ndarray, w: np.ndarray) -> np.ndarray:
        # ∇F(w) = (1/n) * X.T @ (X @ w - y)
        n = X.shape[0]
        prediction = X @ w
        return (1/n) * X.T @ (prediction - y_true)

class LogisticLoss(Loss):
    """Binary Cross-Entropy loss for logistic regression."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def gradient(self, X: np.ndarray, y_true: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Sigmoid is assumed to be applied in the prediction
        # ∇F(w) = (1/n) * X.T @ (σ(Xw) - y)
        n = X.shape[0]
        # Note: In practice, we calculate prediction as σ(Xw) using a utility
        # For pure optimization benchmarks, we often pass probabilities directly
        # However, for gradient descent, we need the raw weights.
        z = X @ w
        sigmoid = 1 / (1 + np.exp(-z))
        return (1/n) * X.T @ (sigmoid - y_true)

class HingeLoss(Loss):
    """Hinge loss for SVM (Non-smooth)."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # y_true should be in {-1, 1}
        return np.mean(np.maximum(0, 1 - y_true * y_pred))
    
    def gradient(self, X: np.ndarray, y_true: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Sub-gradient of hinge loss
        n = X.shape[0]
        z = y_true * (X @ w)
        # mask = 1 where y*(Xw) < 1
        mask = (z < 1).astype(float)
        grad = - (1/n) * (mask * y_true)[:, np.newaxis] * X
        return np.sum(grad, axis=0)
