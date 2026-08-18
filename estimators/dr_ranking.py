from dataclasses import dataclass, field

import numpy as np

from data.batch import DataBatch
from data.utils import array_pad_shrink
from estimators.dr import DoublyRobust
from metrics.cumulative import CumulativeMetrics


@dataclass
class DoublyRobustRanking(DoublyRobust):
    """
    Doubly Robust estimator for ranking with position bias (Oosterhuis, 2023,
    "Doubly Robust Estimation for Correcting Position Bias in Click Feedback
    for Unbiased Learning to Rank", arXiv:2203.17118, Eq. 36 with trust-bias
    β = 0, which the dataset does not log).

        V̂ = 1/N Σᵢ Σ_d [ ω̂_d R̂_d + (ω̂_d / ρ̂_d) ( c_i(d) − α_{k_i(d)} R̂_d ) ]

    Unlike the match-based DoublyRobust, the weights are *examination-based*
    per rank, built from the full logged propensity matrix, so the IPS term is
    unbiased by construction on this data. Symbols mapped to the schema:

    - α_k  : examination / position-bias at rank k → the ``position_bias`` curve.
    - R̂_d  : relevance of item d, estimated as q̂(d, position 0). At the top rank
             α_0 = position_bias[0] = 1, where click prob = α_0 R_d = R_d, so the
             position-aware reward model queried at position 0 yields relevance.
    - ρ̂_d  : expected examination of d under π₀ = Σ_k π₀(k|d) α_k = (d's propensity
             row) · position_bias, floored at ``tau`` for variance control. This
             is exactly the stochastic-PBM logging weight.
    - ω̂_d  : expected weight under the (deterministic) target = α at d's target
             rank = position_bias[target_rank(d)].
    - k_i(d): d's logging rank (where its click was observed) → α_{k_i(d)} =
             position_bias[logging_rank(d)].
    """

    position_bias: np.ndarray = field(default_factory=lambda: np.ones(1))
    tau: float = 1e-3

    def __post_init__(self):
        super().__post_init__()
        assert self.tau > 0, "tau must be > 0"
        self.position_bias = np.asarray(self.position_bias, dtype=np.float64)

    def update_cumulative_metrics(
        self, cumulative_metrics: CumulativeMetrics, batch: DataBatch
    ):
        rewards_batch = self.get_rewards(batch=batch)
        n_records = batch.n_rows
        ks = [len(batch.logging_actions[i]) for i in range(n_records)]

        # R̂_d = q̂(d, position 0) for every logging-selected item, gathered into
        # one batched predict() call (position 0 ⇒ pure relevance).
        logging_parts = []
        for i in range(n_records):
            k = ks[i]
            actions = np.asarray(batch.actions[i], dtype=np.float64)
            logging_parts.append(
                actions[np.asarray(batch.logging_actions[i][:k], dtype=int)]
            )
        stacked = np.concatenate(logging_parts, axis=0)
        r_hat_all = self.reward_model.predict(stacked, np.zeros(stacked.shape[0]))

        dr_per_row = np.zeros(n_records)
        offset = 0
        for i in range(n_records):
            k = ks[i]
            r_hat = r_hat_all[offset : offset + k]  # indexed by logging rank
            offset += k

            pb = array_pad_shrink(
                array=self.position_bias,
                output_length=k,
                padding_value=self.position_bias.take(-1),
            )

            # ρ̂ per logging item: propensity row (logging rank a) · position bias.
            prop = np.asarray(batch.propensities[i], dtype=np.float64)[:k, :k]
            rho = np.maximum(prop.dot(pb), self.tau)

            # α at each item's logging rank a is just pb[a].
            alpha = pb

            # ω̂_d = pb[target_rank(d)] for items the target also ranks, else 0.
            target_rank = {a: j for j, a in enumerate(batch.target_actions[i][:k])}
            omega = np.zeros(k)
            for a, action in enumerate(batch.logging_actions[i][:k]):
                j = target_rank.get(action)
                if j is not None:
                    omega[a] = pb[j]

            c = rewards_batch[i, :k]
            dm = omega * r_hat
            correction = (omega / rho) * (c - alpha * r_hat)
            dr_per_row[i] = np.sum(dm + correction)

        cumulative_metrics.n += n_records
        cumulative_metrics.all_rewards = np.append(
            cumulative_metrics.all_rewards, dr_per_row
        )
        cumulative_metrics.sum_rewards += dr_per_row.sum()
        cumulative_metrics.sum_logging_rewards += rewards_batch.sum()
