"""Shared ``typing.Literal`` aliases for the string-enum parameters used across PyNonpar."""

from typing import Literal

Alternative = Literal["two.sided", "less", "greater"]
Quantile = Literal["t", "normal"]
TiesMethod = Literal["average", "min", "max"]
Method = Literal["asymptotic", "exact"]
HettmanspergerAlternative = Literal["increasing", "decreasing", "custom"]
Distribution = Literal["F", "Chisq"]
