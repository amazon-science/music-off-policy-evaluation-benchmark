from dataclasses import dataclass

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from estimators.estimator import Estimator
from metrics.confidence_intervals import (
    compute_bootstrap_estimates,
    compute_ci,
)
from metrics.results import Results


@dataclass
class GroundTruth(Estimator):
    def __init__(self, omega: float = 1.0):
        super().__init__()

        assert omega > 0, "Omega must be > 0"
        self.omega = omega

    @staticmethod
    def _row_sums(rewards_col) -> np.ndarray:
        flat = np.asarray(pc.list_flatten(rewards_col))
        lengths = np.asarray(pc.list_value_length(rewards_col))
        offsets = np.zeros(len(lengths), dtype=np.int64)
        np.cumsum(lengths[:-1], out=offsets[1:])
        sums = np.add.reduceat(flat, offsets)
        sums[lengths == 0] = 0.0
        return sums

    def evaluate(
        self, dataset_path: str, batch_size: int = 1_000, sample_ratio: float = None
    ) -> Results:
        n = 0
        sum_rewards = 0.0
        row_sum_chunks = []

        # GroundTruth only needs the rewards column; reading it alone avoids
        # deserializing the (much larger) actions and propensities columns.
        data_iter = ds.dataset(dataset_path, format="parquet").to_batches(
            columns=["rewards"], batch_size=batch_size
        )

        for b in data_iter:
            row_sums = self._row_sums(b.column("rewards"))
            n += len(row_sums)
            sum_rewards += row_sums.sum()
            row_sum_chunks.append(row_sums)

        all_rewards = np.concatenate(row_sum_chunks) if row_sum_chunks else np.array([])

        # Compute bootstrap estimates for the logged rewards
        bootstrap_estimates = compute_bootstrap_estimates(
            values=all_rewards, func=lambda x: np.mean(x) * self.omega
        )

        return Results(
            metric=(sum_rewards / n) * self.omega,
            ci=compute_ci(
                np.mean(all_rewards).item() * self.omega, bootstrap_estimates
            ),
            n=n,
        )
