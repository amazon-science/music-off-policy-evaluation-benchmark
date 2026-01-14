from dataclasses import dataclass
from typing import Optional

from metrics.confidence_intervals import ConfidenceInterval
from metrics.evaluation import EvaluationMetrics


@dataclass
class Results:
    metric: float
    ci: ConfidenceInterval
    n: int
    evaluation_metrics: Optional[EvaluationMetrics] = None

    def __repr__(self):
        return (
            f"{self.metric} ({self.ci.lower_bound}, {self.ci.upper_bound}) n={self.n}"
        )
