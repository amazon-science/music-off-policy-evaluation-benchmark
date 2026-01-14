from dataclasses import dataclass

import numpy as np
import pyarrow.dataset as ds

from data.batch import DataBatch
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

    def evaluate(
        self, dataset_path: str, batch_size: int = 1_000, sample_ratio: float = None
    ) -> Results:
        n = 0
        sum_rewards = 0.0
        all_rewards = []

        data_iter = ds.dataset(dataset_path, format="parquet").to_batches(
            batch_size=batch_size
        )

        for b in data_iter:
            batch = DataBatch.from_record(batch=b)
            rewards_batch = self.get_rewards(batch=batch)
            n += batch.n_rows
            sum_rewards += rewards_batch.sum()
            all_rewards.extend(rewards_batch.sum(axis=1))

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
