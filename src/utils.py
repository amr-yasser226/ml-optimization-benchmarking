def generate_synthetic_data(n: int, d: int, condition_number: float = 1.0):
    """Generates synthetic data with controlled conditioning[cite: 150]."""
    U, _ = np.linalg.qr(np.random.randn(n, d))
    V, _ = np.linalg.qr(np.random.randn(d, d))
    
    # Create singular values to match the target condition number
    singular_values = np.linspace(condition_number, 1, min(n, d))
    S = np.zeros((n, d))
    np.fill_diagonal(S, singular_values)
    
    X = U @ S @ V
    w_true = np.random.randn(d)
    y = X @ w_true + np.random.normal(0, 0.1, size=n)
    return X, y

def compute_ridge_closed_form(X: np.ndarray, y: np.ndarray, lmbda: float):
    """Analytical solution for Ridge Regression[cite: 151]."""
    n, d = X.shape
    A = X.T @ X + n * lmbda * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)

def ridge_objective(w: np.ndarray, X: np.ndarray, y: np.ndarray, lmbda: float):
    """Computes Loss and Gradient for Ridge Regression."""
    n = X.shape[0]
    error = X @ w - y
    loss = (1/(2*n)) * np.sum(error**2) + (lmbda/2) * np.sum(w**2)
    grad = (1/n) * (X.T @ error) + lmbda * w
    return loss, grad
