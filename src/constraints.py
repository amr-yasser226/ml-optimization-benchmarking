import numpy as np
# Helper functions __________________________
def get_norm(w:np.ndarray) -> float:
    """
    Get the L2 norm of vector w
    :arg w: input vector
    """
    return np.linalg.norm(w)

# Constraint Enforcing functions __________________________
def projector(w:np.ndarray, tau:float) -> np.ndarray:
    """
    Project vector w back to stay within the feasible region
    :arg w: vector to be projected
    :arg tau: maximum allowed norm
    """
    norm_w = get_norm(w)
    if norm_w > tau:
        return (tau / norm_w) * w
    return w

def penalise_violation(w:np.ndarray, original_loss: float,
                       tau:float, cons_enforcement:float, dual_multiplier:float=None) -> float:
    """
    Penalise the violation of the constraint ||w||_2 <= tau
    :arg w: vector to be penalised
    :arg original_loss: the value of the objective function before penalisation
    :arg tau: maximum allowed norm
    :arg cons_enforcement: penalty parameter
    :arg dual_multiplier: dual variable associated with the constraint (for augmented Lagrangian methods [FUTURE WORK])
    """
    norm_w = get_norm(w)
    penalty = cons_enforcement * max(0, norm_w - tau) ** 2

    if dual_multiplier is not None:
        penalty += dual_multiplier * (norm_w - tau)
    return original_loss + penalty

# def enforce_dimocraphic_parity(y_pred: np.ndarray, ) -> np.ndarray:
"""
[FUTURE WORK] needs context, what are the possible externious varaibles?
"""

