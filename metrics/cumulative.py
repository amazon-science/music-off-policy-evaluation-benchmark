from dataclasses import dataclass, field

import numpy as np


@dataclass
class CumulativeMetrics:
    max_k: int = 0
    n: int = 0
    n_matches: int = 0
    all_rewards: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sum_rewards: float = 0.0
    sum_logging_rewards: float = 0.0
    n_per_pos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sum_weights_per_pos: np.ndarray = field(default_factory=lambda: np.zeros(0))
