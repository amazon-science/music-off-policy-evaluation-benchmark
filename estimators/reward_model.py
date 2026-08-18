from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pyarrow.dataset as ds

from data.batch import DataBatch


class RewardModel(ABC):
    @abstractmethod
    def fit(self, dataset_path: str, batch_size: int = 1_000) -> "RewardModel":
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, X: np.ndarray, positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict q̂ for action contexts X. ``positions`` is the 0-indexed rank
        of each row in its ranking.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, dataset_path: str, batch_size: int = 1_000) -> dict:
        raise NotImplementedError


def _stack_logged_pairs(batch: DataBatch):
    """
    Flatten a batch into stacked (X, r, pos) over the actions chosen by the
    logging policy: X has shape (Σᵢ kᵢ, d), r has shape (Σᵢ kᵢ,), and pos
    has shape (Σᵢ kᵢ,) giving the 0-indexed display position of each pair.
    """
    X_parts = []
    r_parts = []
    pos_parts = []
    for i in range(batch.n_rows):
        rewards = batch.rewards[i]
        k = len(rewards)
        if k == 0:
            continue
        actions = np.asarray(batch.actions[i], dtype=np.float64)
        X_parts.append(actions[batch.logging_actions[i][:k]])
        r_parts.append(np.asarray(rewards, dtype=np.float64))
        pos_parts.append(np.arange(k))
    if not X_parts:
        return None, None, None
    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(r_parts, axis=0),
        np.concatenate(pos_parts, axis=0),
    )


@dataclass
class PositionAwareXGBoostRewardModel(RewardModel):
    """
    Gradient-boosted reward model q̂(action, position) ≈ P(r = 1 | action,
    position), trained with XGBoost's ``binary:logistic`` objective. The
    0-indexed display position is appended to the 129-dim action context as an
    extra feature, so the model can represent position bias: trees split on the
    rank repeatedly, capturing the non-monotonic position-bias shape without
    one-hot expansion.

    XGBoost needs the training matrix in memory, so fit() draws a uniform
    reservoir sample over observations (impressions) while streaming the
    dataset: a sampled request contributes all of its logged actions, capped at
    ``max_train_requests`` requests. Sampling per request keeps the k actions of
    an impression together (they share a context and are not independent). With
    ``max_train_requests=None`` the full dataset is loaded (watch memory:
    ~n_pairs * d * 4 bytes).

    predict() returns calibrated probabilities in [0, 1], matching the reward
    range. We keep scale_pos_weight=1 by default: up-weighting the rare positive
    class improves ranking but de-calibrates the absolute probabilities, which
    biases the Doubly Robust direct-method term. Pairs with DirectMethod /
    DoublyRobust, which pass the target ranking (for the model term) and logging
    ranking (for the IPS correction) as positions.
    """

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    max_train_requests: Optional[int] = 200_000
    n_jobs: int = -1
    random_state: int = 42
    model: object = field(default=None, init=False)

    @staticmethod
    def _augment(X: np.ndarray, positions: Optional[np.ndarray]) -> np.ndarray:
        """Append the 0-indexed display position to each action context row."""
        assert positions is not None, (
            "PositionAwareXGBoostRewardModel requires positions."
        )
        X = np.asarray(X)
        pos = np.asarray(positions, dtype=X.dtype).reshape(-1, 1)
        return np.concatenate([X, pos], axis=1)

    def _stack_one(self, batch: DataBatch, i: int):
        """Augmented (X, r) for a single observation i (its logged actions)."""
        rewards = batch.rewards[i]
        k = len(rewards)
        if k == 0:
            return None, None
        actions = np.asarray(batch.actions[i], dtype=np.float64)
        X = actions[batch.logging_actions[i][:k]]
        X = self._augment(X, np.arange(k)).astype(np.float32, copy=False)
        r = np.asarray(rewards, dtype=np.float32)
        return X, r

    def _reservoir_sample_requests(self, dataset_path: str, batch_size: int):
        """
        Uniform reservoir sample of up to ``max_train_requests`` observations
        (impressions); each sampled request contributes all of its logged
        actions. Returns the concatenated (X, r) over the sampled requests.
        """
        cap = self.max_train_requests
        rng = np.random.default_rng(self.random_state)

        reservoir_X: List[np.ndarray] = []  # one (kᵢ, d) block per kept request
        reservoir_r: List[np.ndarray] = []
        seen = 0

        data_iter = ds.dataset(dataset_path, format="parquet").to_batches(
            batch_size=batch_size
        )
        for b in data_iter:
            batch = DataBatch.from_record(batch=b)
            for i in range(batch.n_rows):
                X, r = self._stack_one(batch, i)
                if X is None:
                    continue
                if cap is None:
                    reservoir_X.append(X)
                    reservoir_r.append(r)
                    seen += 1
                    continue
                if seen < cap:
                    reservoir_X.append(X)
                    reservoir_r.append(r)
                else:
                    j = rng.integers(0, seen + 1)
                    if j < cap:
                        reservoir_X[j] = X
                        reservoir_r[j] = r
                seen += 1

        if not reservoir_X:
            raise ValueError(f"No data found at {dataset_path}")
        return np.concatenate(reservoir_X, axis=0), np.concatenate(reservoir_r, axis=0)

    def fit(
        self, dataset_path: str, batch_size: int = 1_000
    ) -> "PositionAwareXGBoostRewardModel":
        import xgboost as xgb

        X, r = self._reservoir_sample_requests(dataset_path, batch_size)
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            tree_method="hist",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0,
        )
        self.model.fit(X, r.astype(int))
        return self

    def predict(
        self, X: np.ndarray, positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        assert self.model is not None, "Call fit() before predict()."
        X = self._augment(X, positions).astype(np.float32, copy=False)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, dataset_path: str, batch_size: int = 1_000) -> dict:
        """
        Streamed MSE/RMSE/MAE of q̂ against logged rewards on the actions
        chosen by the logging policy. Returns {"n", "mse", "rmse", "mae"}.
        """
        assert self.model is not None, "Call fit() before evaluate()."
        return _streamed_error_metrics(
            dataset_path=dataset_path,
            batch_size=batch_size,
            predict_fn=self.predict,
        )


def _streamed_error_metrics(dataset_path: str, batch_size: int, predict_fn) -> dict:
    n = 0
    sum_sq_err = 0.0
    sum_abs_err = 0.0

    data_iter = ds.dataset(dataset_path, format="parquet").to_batches(
        batch_size=batch_size
    )
    for b in data_iter:
        batch = DataBatch.from_record(batch=b)
        X, r, pos = _stack_logged_pairs(batch)
        if X is None:
            continue
        err = r - predict_fn(X, pos)
        sum_sq_err += float(err @ err)
        sum_abs_err += float(np.abs(err).sum())
        n += err.shape[0]

    mse = sum_sq_err / n
    return {"n": n, "mse": mse, "rmse": float(np.sqrt(mse)), "mae": sum_abs_err / n}
