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

def kkt_diagnostic_checker(weights:np.ndarray, dual_multipliers:np.ndarray, objective_grad_func, constraint_funcs, constraint_grads,
                           tolerance=1e-5):
    """
    Evaluates how well a solution satisfies KKT conditions.

    Inputs:
        :arg weights (w*): Optimized model parameters.
        :arg dual_multipliers (mu*): Multipliers extracted from the solver (e.g., L-BFGS dual).
        :arg objective_grad_func: Function returning the gradient of the loss at w*.
        :arg constraint_funcs: List of functions g_j(w).
        :arg constraint_grads: List of functions returning gradients of g_j(w).
    """

    # 1. STATIONARITY CHECK
    # Calculate Grad(F) + sum(mu_j * Grad(g_j))
    grad_f = objective_grad_func(weights)
    weighted_constraint_grads = sum(mu * g_grad(weights)
                                    for mu, g_grad in zip(dual_multipliers, constraint_grads))
    stationarity_error = get_norm(grad_f + weighted_constraint_grads)

    # 2. PRIMAL FEASIBILITY CHECK
    # Ensure g_j(w) <= 0
    primal_violations = [max(0, g(weights)) for g in constraint_funcs]
    max_primal_error = max(primal_violations)

    # 3. DUAL FEASIBILITY CHECK
    # Ensure mu_j >= 0
    dual_violations = [max(0, -mu) for mu in dual_multipliers]
    max_dual_error = max(dual_violations)

    # 4. COMPLEMENTARY SLACKNESS CHECK
    # Ensure mu_j * g_j(w) is close to 0
    slackness_violations = [abs(mu * g(weights))
                            for mu, g in zip(dual_multipliers, constraint_funcs)]
    max_slackness_error = max(slackness_violations)

    # 5. AGGREGATE RESULTS
    is_optimal = all(err < tolerance for err in [stationarity_error, max_primal_error,
                                                 max_dual_error, max_slackness_error])

    return {
        "is_optimal": is_optimal,
        "errors": {
            "stationarity": stationarity_error,
            "primal": max_primal_error,
            "dual": max_dual_error,
            "slackness": max_slackness_error
        }
    }