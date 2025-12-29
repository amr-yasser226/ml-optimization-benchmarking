import numpy as np
import time

class OptimizationLogger:
    """
    Standardized logger for ML optimization benchmarking. 
    Tracks objective values, gradient norms, and convergence metrics.
    """
    def __init__(self, name: str, ground_truth_w: np.ndarray = None):
        self.name = name
        self.ground_truth_w = ground_truth_w
        self.start_time = time.perf_counter()
        self.logs = {
            "iteration": [],
            "objective_value": [],
            "gradient_norm": [],
            "optimality_gap": [],
            "elapsed_time": []
        }

    def record_step(self, iteration: int, w: np.ndarray, loss: float, grad: np.ndarray):
        """Captures a snapshot of the optimization state at a given iteration."""
        self.logs["iteration"].append(iteration)
        self.logs["objective_value"].append(loss)
        self.logs["gradient_norm"].append(np.linalg.norm(grad))
        self.logs["elapsed_time"].append(time.perf_counter() - self.start_time)
        
        if self.ground_truth_w is not None:
            # Measure ||w - w*|| as the iterative error [cite: 146, 153]
            gap = np.linalg.norm(w - self.ground_truth_w)
            self.logs["optimality_gap"].append(gap)

    def get_summary(self):
        return self.logs
