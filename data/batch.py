from dataclasses import dataclass
from typing import List

from pyarrow import RecordBatch


@dataclass
class DataBatch:
    actions: List[List[List[float]]]
    rewards: List[List[float]]
    logging_actions: List[List[int]]
    target_actions: List[List[int]]
    propensities: List[List[List[float]]]
    n_rows: int
    max_k: int = 0
    max_n: int = 0

    def __post_init__(self):
        self.max_k = max(len(sa) for sa in self.logging_actions)
        self.max_n = max(len(a) for a in self.actions)

    @staticmethod
    def from_record(batch: RecordBatch) -> "DataBatch":
        batch_dict = batch.to_pydict()

        return DataBatch(
            actions=batch_dict["actions"],
            rewards=batch_dict["rewards"],
            logging_actions=batch_dict["logging_selected_actions"],
            target_actions=batch_dict["target_selected_actions"],
            propensities=batch_dict["propensities"],
            n_rows=batch.num_rows,
        )
