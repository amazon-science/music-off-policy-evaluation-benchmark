from dataclasses import dataclass

import numpy as np

from data.batch import DataBatch
from data.utils import array_pad_shrink
from estimators.dr import DoublyRobust
from metrics.cumulative import CumulativeMetrics


@dataclass
class DirectMethod(DoublyRobust):
    """
    Direct Method (DM) estimator — the model-based term of Doubly Robust
    with no importance-sampling correction:

        V̂_DM = 1/n Σᵢ Σⱼ q̂(xᵢ, aᵢⱼ^π)

    where q̂ is a pre-fitted reward model and j runs over the same positions
    DoublyRobust uses (the first k = |logging ranking| target actions). DM is
    biased by the reward model's error but carries no IPS-weight variance, so
    it isolates how much signal q̂ alone provides; comparing DM with DR shows
    what the IPS correction adds on top of the model.
    """

    def update_cumulative_metrics(
        self, cumulative_metrics: CumulativeMetrics, batch: DataBatch
    ):
        n_records = batch.n_rows
        rewards_batch = self.get_rewards(batch=batch)

        # Gather the target-selected action vectors for every row and query the
        # reward model once (per-call overhead dominates for models like
        # XGBoost). Same positions as DoublyRobust: the first k target actions.
        # The display position of the j-th target action is j (0-indexed), used
        # by position-aware reward models and ignored by position-blind ones.
        ks = [len(batch.logging_actions[i]) for i in range(n_records)]
        target_parts = []
        pos_parts = []
        for i in range(n_records):
            k = ks[i]
            actions = np.asarray(batch.actions[i], dtype=np.float64)
            target_parts.append(
                actions[np.asarray(batch.target_actions[i][:k], dtype=int)]
            )
            pos_parts.append(np.arange(k))

        q_target_all = self.reward_model.predict(
            np.concatenate(target_parts, axis=0), np.concatenate(pos_parts, axis=0)
        )

        dm_per_row = np.zeros(n_records)
        offset = 0
        for i in range(n_records):
            k = ks[i]
            dm_per_row[i] = q_target_all[offset : offset + k].sum()
            offset += k

        cumulative_metrics.n += n_records
        cumulative_metrics.all_rewards = np.append(
            cumulative_metrics.all_rewards, dm_per_row
        )
        cumulative_metrics.sum_rewards += dm_per_row.sum()
        cumulative_metrics.sum_logging_rewards += rewards_batch.sum()

        # Per-position counts are kept for parity with the base evaluation
        # metrics; DM has no IPS weights, so sum_weights_per_pos stays zero.
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
