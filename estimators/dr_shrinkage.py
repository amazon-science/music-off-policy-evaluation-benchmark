from dataclasses import dataclass

import numpy as np

from estimators.dr import DoublyRobust


@dataclass
class DoublyRobustShrinkage(DoublyRobust):
    """
    Doubly Robust with optimistic shrinkage (Su et al., 2020,
    "Doubly Robust Off-Policy Evaluation with Shrinkage").

    The IPS-correction weights are shrunk by

        ŵ = λ / (w² + λ) · w

    which suppresses large, high-variance importance weights while leaving
    small ones almost unchanged. The shrinkage interpolates between the two
    extremes:

        λ → 0   ⇒ ŵ → 0      (pure Direct Method)
        λ → ∞   ⇒ ŵ → w      (standard Doubly Robust)

    so λ trades a little correction-term bias for lower variance. It is the
    targeted remedy when the reward model is good but the importance weights
    are heavy-tailed.
    """

    lambda_shrinkage: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.lambda_shrinkage > 0, "lambda_shrinkage must be > 0"

    def _shrink_weights(self, w: np.ndarray) -> np.ndarray:
        lam = self.lambda_shrinkage
        return (lam / (w**2 + lam)) * w
