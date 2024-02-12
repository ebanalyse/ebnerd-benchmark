from typing import Protocol
import numpy as np


class Metric(Protocol):
    name: str

    def calculate(self, y_true: np.ndarray, y_score: np.ndarray) -> float: ...

    def __str__(self) -> str:
        return f"<Callable Metric: {self.name}>: params: {self.__dict__}"

    def __repr__(self) -> str:
        return str(self)

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self.calculate(y_true, y_score)
