from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pyarrow.dataset as ds
from data.batch import DataBatch
from data.utils import array_pad_shrink
from estimators.estimator import Estimator
from metrics.confidence_intervals import (
    compute_ci,
    compute_bootstrap_estimates,
)
from metrics.cumulative import CumulativeMetrics
from metrics.error_decomposition import ErrorDecomposition
from metrics.evaluation import EvaluationMetrics
from metrics.results import Results


@dataclass
class OffPolicyEstimator(Estimator):
    def evaluate(
        self,
        dataset_path: str,
        result_ground_truth: Optional[Results] = None,
        limit: Optional[int] = None,
        batch_size: Optional[int] = 1_000,
    ) -> Tuple[Results, Optional[ErrorDecomposition]]:
        """
        Evaluate the estimator on a dataset.

        :param dataset_path: Path of dataset.
        :param result_ground_truth: Result computed through GroundTruth estimator to get the true value of the policy.
        :param limit: Number of datapoints used to use. None to use the full dataset.
        :param batch_size: Number of datapoints in a batch.
        :return: Tuple of Results and ErrorDecomposition if true value of the policy is passed.
        """

        cumulative_metrics = CumulativeMetrics()

        dataset = ds.dataset(dataset_path, format="parquet")
        data_iter = dataset.to_batches(batch_size=batch_size)

        for b in data_iter:
            # Stop if limit of datapoints to be used is reached
            if limit and cumulative_metrics.n >= limit:
                break

            # Apply limit if specified
            batch = (
                b.slice(0, min(b.num_rows, limit - cumulative_metrics.n))
                if limit
                else b
            )

            batch = DataBatch.from_record(batch=batch)
            self.update_cumulative_metrics(cumulative_metrics, batch)

        eval_metrics = OffPolicyEstimator.compute_evaluation_metrics(cumulative_metrics)

        # Compute confidence intervals for the estimated reward
        bootstrap_estimates = compute_bootstrap_estimates(
            values=cumulative_metrics.all_rewards, func=np.mean
        )
        overall_avg = np.mean(cumulative_metrics.all_rewards).item()
        ci = compute_ci(overall_avg, bootstrap_estimates)

        # Decompose error into Bias^2, Variance and MSE
        err_dec = None
        if result_ground_truth:
            bias2, var, mse = Estimator.bias_var_decomp(
                np.array(result_ground_truth.metric), bootstrap_estimates
            )
            squared_errors = (result_ground_truth.metric - bootstrap_estimates) ** 2
            mse_ci = compute_ci(np.mean(squared_errors).item(), squared_errors)
            err_dec = ErrorDecomposition(bias2=bias2, var=var, mse=mse, ci=mse_ci)

        return Results(
            metric=eval_metrics.reward,
            ci=ci,
            n=eval_metrics.num_observations,
            evaluation_metrics=eval_metrics,
        ), err_dec

    @abstractmethod
    def compute_ips_weights(self, batch: DataBatch) -> np.ndarray:
        raise NotImplementedError

    def update_cumulative_metrics(
        self, cumulative_metrics: CumulativeMetrics, batch: DataBatch
    ):
        # Get the data from the current batch
        rewards_batch = self.get_rewards(batch=batch)
        ips_weights_batch = self.compute_ips_weights(batch)
        ips_rewards_batch = rewards_batch * ips_weights_batch
        n_records = batch.n_rows

        # Update the cumulative metrics
        cumulative_metrics.n += n_records
        cumulative_metrics.n_matches += (
            ips_weights_batch > 0
        ).sum()  # matches when ips weight is > 0
        cumulative_metrics.all_rewards = np.append(
            cumulative_metrics.all_rewards, ips_rewards_batch.sum(axis=1)
        )
        cumulative_metrics.sum_rewards += ips_rewards_batch.sum()
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

    @staticmethod
    def compute_evaluation_metrics(
        cumulative_metrics: CumulativeMetrics,
    ) -> EvaluationMetrics:
        return EvaluationMetrics(
            num_observations=cumulative_metrics.n,
            num_matches=cumulative_metrics.n_matches,
            reward=cumulative_metrics.sum_rewards / cumulative_metrics.n,
            logging_reward=cumulative_metrics.sum_logging_rewards
            / cumulative_metrics.n,
            control_variates=np.divide(
                cumulative_metrics.sum_weights_per_pos,
                cumulative_metrics.n_per_pos,
                where=cumulative_metrics.n_per_pos != 0,
            ),
        )
