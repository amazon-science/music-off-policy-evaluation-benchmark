import numpy as np
from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics:
    num_observations: int = 0
    num_matches: int = 0
    reward: float = 0.0
    logging_reward: float = 0.0
    control_variates: np.ndarray = field(default_factory=lambda: np.zeros(0))
