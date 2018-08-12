"""
.. py:currentmodule:: twosample_paired

.. module:: twosample
   :platform: Unix, Windows
   :synopsis: Module to calculate two sample tests for paired observations: paired ranks test,

.. moduleauthor:: Martin Happ <martin.happ@aon.at>
"""

import numpy as np
import math
import scipy
import scipy.stats
from collections import namedtuple
import PyNonpar.pseudorank

def paired_ranks_test(x, y, alternative = "two.sided", var_equal = False, quantile = "normal", alpha = 0.05):
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
        alpha (float): 1-alpha confidence inverval will be calculated\ n


    Returns:
        namedtuple('BrunnerMunzelResult', ('alternative', 'statistic', 'df', 'pHat', 'lowerCI', 'upperCI', 'pvalue')): \n
        chosen alternative (str)\n
        test statistic (float) \n
        pHat Probablity(x_i < y_j) + 1/2*Probablity(x_i = y_j), i != j (float) \n
        lower 1-alpha CI for pHat (float)\n
        upper 1-alpha CI for pHat (float) \n
        p-value (float)
    """

    data = x + y
    n = len(x)
    m = len(y)
    N = n + m
    grp = [1 for i in range(len(data))]
    if n != m:
        raise AssertionError()

    ranks = PyNonpar.pseudorank.psrank(data, grp)
    xr = ranks[0:len(x)]
    yr = ranks[len(x):N]

    xr_bar = np.mean(xr)
    yr_bar = np.mean(yr)

    p_hat = 1/n*(yr_bar - (n + 1)*1/2)

    sigma_squared = 0.0
    # variance estimator: F_1 = F_2
    if var_equal == True:
        for i in range(n):
            sigma_squared += ( xr[i] - yr[i] - (xr_bar - yr_bar) ) ** 2
    else:
        v = [1 for i in range(n)]
        xri = PyNonpar.pseudorank.psrank(x, v)
        yri = PyNonpar.pseudorank.psrank(y, v)
        for i in range(n):
            sigma_squared += ( xr[i] - xri[i] - (yr[i] - yri[i]) - (xr_bar - yr_bar) ) ** 2
    sigma_squared = sigma_squared * 1 / 4 * 1 / (n ** 2) * 1 / (n - 1)

    test = 1/math.sqrt(n)*( yr_bar - (2*n + 1)*1/2 ) * 1/math.sqrt(sigma_squared)
    p_value = 0.0
    lower = 0.0
    upper = 1.0

    if quantile == "normal":
        if alternative == "greater":
            p_value = scipy.stats.norm.cdf(test)
            upper = p_hat + math.sqrt(sigma_squared)*1/math.sqrt(n)*scipy.stats.norm.ppf(1-alpha)
        elif alternative == "less":
            p_value = 1 - scipy.stats.norm.cdf(test)
            lower = p_hat - math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.norm.ppf(1 - alpha)
        elif alternative == "two.sided":
            p_value = 2 * min(scipy.stats.norm.cdf(test), 1 - scipy.stats.norm.cdf(test))
            lower = p_hat - math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.norm.ppf(1 - alpha/2)
            upper = p_hat + math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.norm.ppf(1 - alpha/2)

        result = namedtuple('PairedRanksResult', ('alternative', 'statistic', 'pHat', 'lowerCI', 'upperCI', 'pvalue'))
        output = result(alternative, test, p_hat, lower, upper, p_value)
    else:
        if alternative == "greater":
            p_value = scipy.stats.t.cdf(test, n-1)
            upper = p_hat + math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.t.ppf(1 - alpha, n - 1)
        elif alternative == "less":
            p_value = 1 - scipy.stats.t.cdf(test, n-1)
            lower = p_hat - math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.t.ppf(1 - alpha, n - 1)
        elif alternative == "two.sided":
            p_value = 2 * min(scipy.stats.t.cdf(test, n-1), 1 - scipy.stats.t.cdf(test, n-1))
            lower = p_hat - math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.t.ppf(1 - alpha/2, n - 1)
            upper = p_hat + math.sqrt(sigma_squared) * 1 / math.sqrt(n) * scipy.stats.t.ppf(1 - alpha/2, n - 1)

        result = namedtuple('BrunnerMunzelResult', ('alternative', 'statistic', 'df', 'pHat', 'lowerCI', 'upperCI', 'pvalue'))
        output = result(alternative, test, n-1, p_hat, lower, upper, p_value)

    return output
