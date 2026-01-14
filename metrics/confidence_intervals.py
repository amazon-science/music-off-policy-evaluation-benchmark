from collections import namedtuple
from typing import Callable, List, Union

import numpy as np

ConfidenceInterval = namedtuple("ConfidenceInterval", ["lower_bound", "upper_bound"])


def compute_bootstrap_estimates(
    values: Union[List, np.ndarray],
    bootstraps: int = 1_000,
    func: Callable[..., float] = np.mean,
) -> Union[List, np.ndarray]:
    """
    Computes M bootstrap estimates from a dataset of N values.

    :param values: Values to sample from.
    :param bootstraps: Number of bootstrap samples to use.
    :param func: Function to compute the statistic of interest.
    :return: List of all bootstrapped values.
    """

    n = len(values)

    bootstrap_vals = []
    for _ in range(bootstraps):
        bootstrap_idx = np.random.choice(n, n, replace=True)
        if isinstance(values, np.ndarray):
            bootstrap_avg = func(values[bootstrap_idx])  # faster the numpy way
        else:
            bootstrap_avg = func([values[i] for i in bootstrap_idx])
        bootstrap_vals.append(bootstrap_avg)

    return bootstrap_vals


def compute_ci(
    overall_avg: float, values: np.ndarray, confidence: float = 0.95
) -> ConfidenceInterval:
    """
    Computes a CI using Lunneborg's bootstrap method.
    https://www.uvm.edu/~statdhtx/StatPages/Randomization%20Tests/ResamplingWithR/BootstMeans/bootstrapping_means.html  # noqa

    Given a dataset of N values, Lunneborg's bootstrap goes as follows:
    1. Compute the population mean, mu_i.
    2. Sample M bootstraps of size N (with replacement).
    3. For each bootstrap i, compute its mean, mu_i, as well as its
        difference from the population mean, d_i = mu_i - mu
    4. Find the lower/upper quantiles, lq and uq, of the differences, d_i.
    5. CI = [mu - uq, mu - lq]

    :param overall_avg: Population mean.
    :param values: Bootstrap samples.
    :param confidence: Probability that the CI holds.
    :return: Tuple of floats representing the lower/upper bounds of the CI, and a list of all
        bootstrapped values.
    """

    bootstrap_deltas = np.array(values) - overall_avg

    lq, uq = np.quantile(bootstrap_deltas, [0.5 - confidence / 2, 0.5 + confidence / 2])
    ci = ConfidenceInterval(overall_avg - uq, overall_avg - lq)
    return ci
