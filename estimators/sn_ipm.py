from dataclasses import dataclass

import numpy as np

from data.batch import DataBatch
from data.utils import array_pad_shrink
from estimators.ipm import IPM
from metrics.cumulative import CumulativeMetrics
from metrics.evaluation import EvaluationMetrics


@dataclass
class SelfNormalizedIPM(IPM):
    """
    Self-Normalized Inverse Propensity Matching (SNIPM) estimator.

    Following London et al. (2023), the estimate is a sum of per-position
    SNIPS estimators:

        V̂_SNIPM = Σⱼ [ Σᵢ wᵢⱼ rᵢⱼ  /  Σᵢ wᵢⱼ ]

    Each position j is self-normalized independently across all observations,
    then the position-level estimates are summed.
    """

    n_bootstraps: int = 1_000

    def update_cumulative_metrics(
        self, cumulative_metrics: CumulativeMetrics, batch: DataBatch
    ):
        # Reuse all the standard IPS bookkeeping (n, n_matches, all_rewards,
        # sum_rewards, sum_logging_rewards, n_per_pos, sum_weights_per_pos,
        # max_k padding).
        super().update_cumulative_metrics(cumulative_metrics, batch)

        rewards_batch = self.get_rewards(batch=batch)
        ips_weights_batch = self.compute_ips_weights(batch)
        ips_rewards_batch = rewards_batch * ips_weights_batch

        # Pad the SNIPM-specific per-position accumulator if max_k grew.
        if (
            cumulative_metrics.sum_weighted_rewards_per_pos.shape[0]
            < cumulative_metrics.max_k
        ):
            cumulative_metrics.sum_weighted_rewards_per_pos = array_pad_shrink(
                array=cumulative_metrics.sum_weighted_rewards_per_pos,
                output_length=cumulative_metrics.max_k,
                padding_value=0.0,
            )

        cumulative_metrics.sum_weighted_rewards_per_pos[: batch.max_k] += (
            ips_rewards_batch.sum(axis=0)
        )

        # Per-observation per-position arrays for the bootstrap of the
        # ratio estimator. Stored as variable-length 1D arrays; alignment
        # to cumulative_metrics.max_k happens in _compute_bootstrap_estimates.
        for i in range(batch.n_rows):
            cumulative_metrics.all_rewards_per_pos.append(ips_rewards_batch[i])
            cumulative_metrics.all_weights_per_pos.append(ips_weights_batch[i])

    @staticmethod
    def compute_evaluation_metrics(
        cumulative_metrics: CumulativeMetrics,
    ) -> EvaluationMetrics:
        # SNIPM: Σⱼ [ Σᵢ wᵢⱼ rᵢⱼ / Σᵢ wᵢⱼ ]
        sn_reward = np.sum(
            np.divide(
                cumulative_metrics.sum_weighted_rewards_per_pos,
                cumulative_metrics.sum_weights_per_pos,
                where=cumulative_metrics.sum_weights_per_pos > 0,
                out=np.zeros_like(cumulative_metrics.sum_weighted_rewards_per_pos),
            )
        )
        return EvaluationMetrics(
            num_observations=cumulative_metrics.n,
            num_matches=cumulative_metrics.n_matches,
            reward=sn_reward,
            logging_reward=cumulative_metrics.sum_logging_rewards
            / cumulative_metrics.n,
            control_variates=np.divide(
                cumulative_metrics.sum_weights_per_pos,
                cumulative_metrics.n_per_pos,
                where=cumulative_metrics.n_per_pos != 0,
                out=np.zeros_like(cumulative_metrics.sum_weights_per_pos),
            ),
        )

    def _compute_bootstrap_estimates(
        self, cumulative_metrics: CumulativeMetrics
    ) -> np.ndarray:
        # Bootstrap the ratio per position: resample observations, recompute
        # Σᵢ wᵢⱼrᵢⱼ / Σᵢ wᵢⱼ for each position j, then sum over j.
        max_k = cumulative_metrics.max_k
        n = len(cumulative_metrics.all_rewards_per_pos)

        # Align per-observation arrays to a common width; batches may have
        # had different max_k so the stored 1D arrays can vary in length.
        all_wr = np.zeros((n, max_k))
        all_w = np.zeros((n, max_k))
        for i in range(n):
            row_wr = cumulative_metrics.all_rewards_per_pos[i]
            row_w = cumulative_metrics.all_weights_per_pos[i]
            all_wr[i, : row_wr.shape[0]] = row_wr
            all_w[i, : row_w.shape[0]] = row_w

        estimates = np.empty(self.n_bootstraps)
        for b in range(self.n_bootstraps):
            idx = np.random.choice(n, n, replace=True)
            sum_wr = all_wr[idx].sum(axis=0)
            sum_w = all_w[idx].sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                estimates[b] = np.where(sum_w > 0, sum_wr / sum_w, 0.0).sum()

        return estimates
