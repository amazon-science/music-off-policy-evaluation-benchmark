from dataclasses import dataclass

import numpy as np

from data.batch import DataBatch
from data.utils import array_pad_shrink
from estimators.ipm import IPM
from estimators.reward_model import RewardModel
from metrics.cumulative import CumulativeMetrics


@dataclass
class DoublyRobust(IPM):
    """
    Doubly Robust estimator (Dudík et al., 2011) layered on IPM weights.

        V̂_DR = 1/n Σᵢ Σⱼ [ q̂(xᵢ, aᵢⱼ^π) + wᵢⱼ ( rᵢⱼ - q̂(xᵢ, aᵢⱼ^{π₀}) ) ]

    where wᵢⱼ are the IPM weights (1{a^π_j == a^{π₀}_j} / P_jj) and q̂ is a
    pre-fitted reward model over the per-action context vectors.

    Unbiased if either the propensities or the reward model is correct.
    """

    reward_model: RewardModel = None

    def __post_init__(self):
        assert self.reward_model is not None, "DoublyRobust requires a reward_model."

    def _shrink_weights(self, w: np.ndarray) -> np.ndarray:
        """
        Weights used in the IPS correction term. The base estimator uses the
        raw IPM weights; subclasses (e.g. shrinkage DR) override this to trade
        a little correction-term bias for lower variance.
        """
        return w

    def update_cumulative_metrics(
        self, cumulative_metrics: CumulativeMetrics, batch: DataBatch
    ):
        rewards_batch = self.get_rewards(batch=batch)
        ips_weights_batch = self.compute_ips_weights(batch)
        n_records = batch.n_rows

        # Gather the target- and logging-selected action vectors for every row
        # into two stacked matrices so the reward model is queried with two
        # batched predict() calls instead of two per row (per-call overhead
        # dominates for models like XGBoost). The j-th action in either ranking
        # is displayed at position j (0-indexed); positions are used by
        # position-aware reward models and ignored by position-blind ones.
        ks = [len(batch.logging_actions[i]) for i in range(n_records)]
        target_parts = []
        logging_parts = []
        pos_parts = []
        for i in range(n_records):
            k = ks[i]
            actions = np.asarray(batch.actions[i], dtype=np.float64)
            target_parts.append(
                actions[np.asarray(batch.target_actions[i][:k], dtype=int)]
            )
            logging_parts.append(
                actions[np.asarray(batch.logging_actions[i][:k], dtype=int)]
            )
            pos_parts.append(np.arange(k))

        positions = np.concatenate(pos_parts, axis=0)
        q_target_all = self.reward_model.predict(
            np.concatenate(target_parts, axis=0), positions
        )
        q_logging_all = self.reward_model.predict(
            np.concatenate(logging_parts, axis=0), positions
        )

        dr_per_row = np.zeros(n_records)
        offset = 0
        for i in range(n_records):
            k = ks[i]
            q_target = q_target_all[offset : offset + k]
            q_logging = q_logging_all[offset : offset + k]
            offset += k

            w = self._shrink_weights(ips_weights_batch[i, :k])
            r = rewards_batch[i, :k]
            dr_per_row[i] = np.sum(q_target + w * (r - q_logging))

        cumulative_metrics.n += n_records
        cumulative_metrics.n_matches += (ips_weights_batch > 0).sum()
        cumulative_metrics.all_rewards = np.append(
            cumulative_metrics.all_rewards, dr_per_row
        )
        cumulative_metrics.sum_rewards += dr_per_row.sum()
        cumulative_metrics.sum_logging_rewards += rewards_batch.sum()

        if cumulative_metrics.max_k < batch.max_k:
            cumulative_metrics.max_k = batch.max_k
            cumulative_metrics.n_per_pos = array_pad_shrink(
                array=cumulative_metrics.n_per_pos,
                output_length=batch.max_k,
                padding_value=0,
            )
            cumulative_metrics.sum_weights_per_pos = array_pad_shrink(
                array=cumulative_metrics.sum_weights_per_pos,
                output_length=batch.max_k,
                padding_value=0.0,
            )
        for selected_actions in batch.logging_actions:
            cumulative_metrics.n_per_pos[: len(selected_actions)] += 1
        cumulative_metrics.sum_weights_per_pos[: batch.max_k] += ips_weights_batch.sum(
            axis=0
        )
