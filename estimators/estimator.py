from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from data.batch import DataBatch
from metrics.error_decomposition import ErrorDecomposition
from metrics.results import Results


@dataclass
class Estimator(ABC):
    @staticmethod
    def bias_var_decomp(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
        """
        Error decomposition in terms of bias and variance.

        :param y: Groud truth
        :param p: Bootstrap estimates
        :return: Bias2, Variance, Mean Squared Error
        """

        bias2 = np.mean(y - p) ** 2
        var = np.var(p)
        return bias2.item(), var.item(), bias2.item() + var.item()

    def benchmark(
        self,
        name: str,
        dataset_path: str,
        result_ground_truth: Optional[Results],
        limit: Optional[int] = None,
    ) -> Tuple[str, Results, Optional[ErrorDecomposition]]:
        result, err = self.evaluate(
            dataset_path, result_ground_truth=result_ground_truth, limit=limit
        )
        return name, result, err

    @abstractmethod
    def evaluate(
        self,
        dataset_path: str,
        result_ground_truth: Optional[Results],
        limit: Optional[int] = None,
    ) -> Results:
        raise NotImplementedError

    @staticmethod
    def get_rewards(batch: DataBatch) -> np.ndarray:
        r = np.zeros(shape=(batch.n_rows, batch.max_k))
        for i, reward_selected_actions in enumerate(batch.rewards):
            r[i, : len(reward_selected_actions)] = reward_selected_actions
        return r
