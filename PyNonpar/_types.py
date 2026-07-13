"""Shared ``typing.Literal`` aliases and input coercion helpers used across PyNonpar."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

Alternative = Literal["two.sided", "less", "greater"]
Quantile = Literal["t", "normal"]
TiesMethod = Literal["average", "min", "max"]
Method = Literal["asymptotic", "exact"]
HettmanspergerAlternative = Literal["increasing", "decreasing", "custom"]
Distribution = Literal["F", "Chisq"]


def as_list(x: ArrayLike, name: str) -> list:
    """
    Coerce array-like input (list, tuple, numpy array, pandas Series, ...) to a plain list.

    Args:
        x (array_like): input to coerce \n
        name (str): parameter name, used in the error message on failure \n

    Returns:
        plain Python list (list)
    """
    try:
        arr = np.asarray(x)
    except ValueError as exc:
        raise TypeError(f"{name} must be array-like (list, tuple, numpy array, or pandas Series) of a consistent shape") from exc
    if arr.ndim != 1:
        raise TypeError(f"{name} must be one-dimensional array-like (list, tuple, numpy array, or pandas Series)")
    return arr.tolist()
