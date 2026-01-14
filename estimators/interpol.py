from dataclasses import dataclass, field

import numpy as np

from data.batch import DataBatch
from data.utils import array_pad_shrink
from estimators.ope import OffPolicyEstimator


@dataclass
class INTERPOL(OffPolicyEstimator):
    n_actions: int = 50
    window_size: int = 1
    position_bias: np.ndarray = field(default_factory=lambda: np.ones(1))

    def __post_init__(self):
        self.position_bias = array_pad_shrink(
            array=self.position_bias,
            output_length=self.n_actions,
            padding_value=self.position_bias.take(-1),
        )

    @staticmethod
    def banded_interval_matrix(dim: int, w: int) -> np.ndarray:
        """
        Creates a banded interval matrix, i.e, a diagonal band of ones along the diagonal.
        Example for dim=4 and w=3 (interval of w = +- 1 around the diagonal):
        [[1,1,0,0],
        [1,1,1,0],
        [0,1,1,1],
        [0,0,1,1]]
        :param dim: size of the matrix
        :param w: interval to use
        :return: 2D Numpy NDArray with zeros/ones
        """
        return np.tri(dim, dim, w) - np.tri(dim, dim, -(w + 1))

    def compute_ips_weights(self, batch: DataBatch) -> np.ndarray:
        ips_weights = np.zeros(shape=(batch.n_rows, batch.max_k))

        for i in range(batch.n_rows):
            k_logging = len(batch.logging_actions[i])
            k_target = len(batch.target_actions[i])

            position_bias = array_pad_shrink(
                array=self.position_bias,
                output_length=max(k_logging, k_target),
                padding_value=0,
            )

            prop_logging = np.array(batch.propensities[i])

            # In order to check whether logging and target selected actions are within a window_size distance,
            # we can reorder the target_actions according to the ordering of the logging_actions

            # Indices of the target_actions wrt to the logging_actions ordering,
            # masked for actions that were selected by the logging policy but not by the target policy

            w = np.arange(1, k_target + 1)
            w = array_pad_shrink(array=w, output_length=batch.max_n, padding_value=0)

            eye_matrix = np.eye(batch.max_n)
            weights = (eye_matrix[batch.target_actions[i]].T * w[:k_target]).sum(axis=1)
            weights = (eye_matrix[batch.logging_actions[i]] * weights).sum(axis=1)
            weights = array_pad_shrink(
                array=weights, output_length=k_logging, padding_value=0
            )
            weights = weights.astype(int) - 1

            target_idx_logging_order = np.ma.array(weights, mask=weights < 0)

            # Re-ordering the target position bias wrt the selected_action_logging ordering
            position_bias_target = np.ma.array(
                position_bias[target_idx_logging_order],
                mask=target_idx_logging_order.mask,
            )

            # Checking which actions are displayed within window_size distance by logging and target
            intersection_indicators = (
                np.abs(np.arange(k_logging) - target_idx_logging_order)
                <= self.window_size
            )

            # Target policy weights (numerators of the IPS weights)
            target_weights = intersection_indicators * position_bias_target

            # Storing positions for selected_actions_logging that are also selected by the target policy
            valid_idx = np.nonzero(~target_idx_logging_order.mask)[0]
            valid_target_positions = np.ma.compressed(target_idx_logging_order)

            # Creating the window indicator matrix, padding for positions that should not be used, to
            # include cases where numbers of selected_action by target and logging differ
            window_matrix = np.pad(
                INTERPOL.banded_interval_matrix(dim=k_logging, w=self.window_size),
                ((0, max(0, k_target - k_logging)), (0, 0)),
                constant_values=0,
            )

            logging_weights_valid = (
                window_matrix[valid_target_positions]
                * (position_bias[:k_logging] * prop_logging[valid_idx, :k_logging])
            ).sum(axis=1)

            logging_weights = np.zeros(shape=(k_logging,))
            logging_weights[valid_idx] = logging_weights_valid

            with np.errstate(divide="ignore", invalid="ignore"):
                ips_w = np.where(
                    logging_weights > 0,
                    (target_weights / logging_weights),
                    np.zeros_like(logging_weights),
                )

            ips_weights[i, :k_logging] = ips_w

        return ips_weights
