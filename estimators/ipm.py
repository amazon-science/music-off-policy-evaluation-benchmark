from dataclasses import dataclass

import numpy as np

from data.batch import DataBatch
from data.utils import array_pad_shrink
from estimators.ope import OffPolicyEstimator


@dataclass
class IPM(OffPolicyEstimator):
    def compute_ips_weights(self, batch: DataBatch) -> np.ndarray:
        ips_weights = np.zeros(shape=(batch.n_rows, batch.max_k))

        for i in range(batch.n_rows):
            k = len(batch.logging_actions[i])

            # Target policy weights (numerators of the IPS weights)
            array_selected_action_target = array_pad_shrink(
                array=np.array(batch.target_actions[i]),
                output_length=len(batch.logging_actions[i]),
                padding_value=-1,
            )

            target_weights = array_selected_action_target == np.array(
                batch.logging_actions[i]
            )

            # Logging policy weights (denominators of the IPS weights)
            logging_weights = np.diag(np.array(batch.propensities[i])[:k, :k])

            with np.errstate(divide="ignore", invalid="ignore"):
                ips_w = np.where(
                    logging_weights > 0,
                    (target_weights / logging_weights),
                    np.zeros_like(logging_weights),
                )

            ips_weights[i, :k] = ips_w

        return ips_weights
