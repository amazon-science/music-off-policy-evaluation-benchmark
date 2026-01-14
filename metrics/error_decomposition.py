from dataclasses import dataclass

from metrics.confidence_intervals import ConfidenceInterval


@dataclass
class ErrorDecomposition:
    bias2: float
    var: float
    mse: float
    ci: ConfidenceInterval

    def __repr__(self):
        return f"bias^22={self.bias2}, var={self.var}, mse={self.mse} ({self.ci.lower_bound}, {self.ci.upper_bound})"
