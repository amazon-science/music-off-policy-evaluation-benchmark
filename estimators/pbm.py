from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

import numpy as np

from data.batch import DataBatch
from data.utils import array_pad_shrink
from estimators.ope import OffPolicyEstimator


class PBMType(Enum):
    DETERMINISTIC = auto()
    STOCHASTIC = auto()


@dataclass
class PBM(OffPolicyEstimator):
    type: PBMType
    n_actions: int = 50
    position_bias: np.ndarray = field(default_factory=lambda: np.ones(1))

    def __post_init__(self):
        self.position_bias = array_pad_shrink(
            array=self.position_bias,
            output_length=self.n_actions,
            padding_value=self.position_bias.take(-1),
        )

    def position_bias_batch(self, batch: DataBatch) -> List[np.ndarray]:
        # Compute position bias for the batch
        # If the specified position_bias curve is shorter than the number of actions
        # selected by the logging policy, then it gets padded with its last value.

        selected_actions_positions = [range(len(x)) for x in batch.logging_actions]

        position_bias_batch = [
            array_pad_shrink(
                array=self.position_bias,
                output_length=max(positions) + 1,
                padding_value=self.position_bias.take(-1),
            )
            for positions in selected_actions_positions
        ]

        return position_bias_batch

    def compute_ips_weights(self, batch: DataBatch) -> np.ndarray:
        ips_weights = np.zeros(shape=(batch.n_rows, batch.max_k))
        position_bias_batch = self.position_bias_batch(batch=batch)

        for i in range(batch.n_rows):
            k_logging = len(batch.logging_actions[i])
            k_target = len(batch.target_actions[i])

            match self.type:
                case PBMType.DETERMINISTIC:
                    logging_weights = position_bias_batch[i]
                case PBMType.STOCHASTIC:
                    prop_logging = np.array(batch.propensities[i])
                    prop_array = prop_logging[:k_logging, :k_logging]
                    logging_weights = prop_array.dot(position_bias_batch[i][:k_logging])

            w = array_pad_shrink(position_bias_batch[i], batch.max_n, 0)
            target_weights_full = np.zeros(batch.max_n)
            target_weights_full[batch.target_actions[i]] = w[:k_target]
            target_weights = target_weights_full[batch.logging_actions[i]][:k_logging]

            with np.errstate(divide="ignore", invalid="ignore"):
                ips_w = np.where(
                    logging_weights > 0,
                    (target_weights / logging_weights),
                    np.zeros_like(logging_weights),
                )

            ips_weights[i, :k_logging] = ips_w

        return ips_weights
