"""
.. py:currentmodule:: twosample_paired

.. module:: twosample
   :platform: Unix, Windows
   :synopsis: Module to calculate two sample tests for paired observations: paired ranks test,

.. moduleauthor:: Martin Happ <martin.happ@aon.at>
"""

import math
from typing import List, NamedTuple, Optional

import numpy as np
import scipy
import scipy.stats

import PyNonpar.pseudorank
from PyNonpar._types import Alternative, Quantile


class PairedRanksResult(NamedTuple):
    alternative: str
    statistic: float
    df: Optional[int]
    pHat: float
    lowerCI: float
    upperCI: float
    pvalue: float


def paired_ranks_test(
    x: List[float], y: List[float], alternative: Alternative = "two.sided", var_equal: bool = False, quantile: Quantile = "normal", alpha: float = 0.05
) -> PairedRanksResult:
    """
    Function to calculate the Paired Ranks test. This test compares the marignal distributions with each other.\n
    For var_equal = true, the null hypothesis H_0: F_1 = F_2 is assumed. For var_equal = False, the null hypothesis\n
    is H_0: p = 1/2 where p = Probablity(x_i < y_j) + 1/2 * Probablity(x_i = y_j), i != j.

    Args:
        x (list(float)): data from first group \n
        y (list(float)): data from second group \n
        alternative (str): either 'two.sided', 'less' or 'greater' \n
        var_equal (bool): False to use an adjustment for unequal distributions.\n
        quantile (str): either 't' for the small sample approximation with a t-distribution or 'normal' for the large sample procedure. \n
        alpha (float): 1-alpha confidence inverval will be calculated\\ n


    Returns:
        namedtuple('PairedRanksResult', ('alternative', 'statistic', 'df', 'pHat', 'lowerCI', 'upperCI', 'pvalue')): \n
        chosen alternative (str)\n
        test statistic (float) \n
        estimated degrees of freedom, only when quantile = 't' (int) \n
        pHat Probablity(x_i < y_j) + 1/2*Probablity(x_i = y_j), i != j (float) \n
        lower 1-alpha CI for pHat (float)\n
        upper 1-alpha CI for pHat (float) \n
        p-value (float)
    """

    # Check inputs
    if not isinstance(x, list):
        raise TypeError("x must be a list")
    if not isinstance(y, list):
        raise TypeError("y must be a list")
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if (not isinstance(alternative, str)) or (alternative not in ['two.sided', 'less', 'greater']):
        raise TypeError('alternative must be either two.sided, less or greater')
    if not isinstance(var_equal, bool):
        raise TypeError('var_equal must be a bool')
    if (not isinstance(quantile, str)) or (quantile not in ['t', 'normal']):
        raise TypeError('quantile must be either t or normal')

    data = x + y
    n = len(x)
    m = len(y)
    N = n + m
    grp = [1 for i in range(len(data))]

    ranks = PyNonpar.pseudorank.psrank(data, grp)
    xr = ranks[0:len(x)]
    yr = ranks[len(x):N]

    xr_bar = float(np.mean(xr))
    yr_bar = float(np.mean(yr))

    p_hat = 1/n*(yr_bar - (n + 1)*1/2)

    xr_arr, yr_arr = np.asarray(xr), np.asarray(yr)

    # variance estimator: F_1 = F_2
    if var_equal:
        sigma_squared = float(np.sum(( xr_arr - yr_arr - (xr_bar - yr_bar) ) ** 2))
    else:
        v = [1 for i in range(n)]
        xri = PyNonpar.pseudorank.psrank(x, v)
        yri = PyNonpar.pseudorank.psrank(y, v)
        xri_arr, yri_arr = np.asarray(xri), np.asarray(yri)
        sigma_squared = float(np.sum(( xr_arr - xri_arr - (yr_arr - yri_arr) - (xr_bar - yr_bar) ) ** 2))
    sigma_squared = sigma_squared * 1 / 4 * 1 / (n ** 2) * 1 / (n - 1)

    test = 1/math.sqrt(n)*( yr_bar - (2*n + 1)*1/2 ) * 1/math.sqrt(sigma_squared)
    p_value = 0.0
    lower = 0.0
    upper = 1.0

    if quantile == "normal":
        if alternative == "greater":
            p_value = float(scipy.stats.norm.cdf(test))
            upper = p_hat + math.sqrt(sigma_squared)*1/math.sqrt(n)*scipy.stats.norm.ppf(1-alpha)
        elif alternative == "less":
            p_value = float(1 - scipy.stats.norm.cdf(test))
            lower = p_hat - math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.norm.ppf(1 - alpha)
        elif alternative == "two.sided":
            p_value = float(2 * min(scipy.stats.norm.cdf(test), 1 - scipy.stats.norm.cdf(test)))
            lower = p_hat - math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.norm.ppf(1 - alpha/2)
            upper = p_hat + math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.norm.ppf(1 - alpha/2)

        output = PairedRanksResult(alternative, test, None, p_hat, lower, upper, p_value)
    else:
        if alternative == "greater":
            p_value = float(scipy.stats.t.cdf(test, n-1))
            upper = p_hat + math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.t.ppf(1 - alpha, n - 1)
        elif alternative == "less":
            p_value = float(1 - scipy.stats.t.cdf(test, n-1))
            lower = p_hat - math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.t.ppf(1 - alpha, n - 1)
        elif alternative == "two.sided":
            p_value = float(2 * min(scipy.stats.t.cdf(test, n-1), 1 - scipy.stats.t.cdf(test, n-1)))
            lower = p_hat - math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.t.ppf(1 - alpha/2, n - 1)
            upper = p_hat + math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.t.ppf(1 - alpha/2, n - 1)

        output = PairedRanksResult(alternative, test, n-1, p_hat, lower, upper, p_value)

    return output
