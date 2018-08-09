"""
.. py:currentmodule:: twosample

.. module:: twosample
   :platform: Unix, Windows
   :synopsis: Module to calculate two sample tests: Brunner-Munzel, Wilcoxon-Mann-Whitney

.. moduleauthor:: Martin Happ <martin.happ@aon.at>
"""

import numpy as np
import math
import scipy
import scipy.stats
from collections import namedtuple
import PyNonpar.pseudorank


def brunner_munzel_test(x: list, y: list, alternative="two.sided", quantile="t") -> list:
    """
    Function to calculate the Brunner-Munzel test. It is recommended to use the t-approximation for small samples.

    Args:
        x (list(float)): data from first group \n
        y (list(float)): data from second group \n
        alternative (str): either 'two.sided', 'less' or 'greater' \n
        quantile (str): either 't' for the small sample approximation with a t-distribution or 'normal' for the large sample procedure. \n

    Returns:
        str: chosen alternative \n
        list(int): trend \n
        float: test statistic \n
        float: estimated degrees of freedom for the t-approximation (only when quantile = 't') \n
        float:  p-value
    """

    # Check inputs
    if not isinstance(x, list):
        raise TypeError("x must be a list")
    if not isinstance(y, list):
        raise TypeError("y must be a list")
    if (not isinstance(alternative, str)) or (alternative not in ['two.sided', 'less', 'greater']):
        raise TypeError('alternative must be either two.sided, less or greater')
    if (not isinstance(quantile, str)) or (quantile not in ['t', 'normal']):
        raise TypeError('quantile must be either t or normal')

    n = (len(x), len(y))
    N = n[0] + n[1]
    data = x + y
    grp = [1 for i in range(len(data))]
    ranks = PyNonpar.pseudorank.psrank(data, grp)
    x = ranks[0:n[0]]
    y = ranks[n[0]:N]

    xi = PyNonpar.pseudorank.psrank(x, grp[0:n[0]])
    yi = PyNonpar.pseudorank.psrank(y, grp[0:n[1]])

    R2 = np.mean(y)
    R1 = np.mean(x)

    # calculate variance estimator
    S1_squared = 0.0
    for i in range(n[0]):
        S1_squared += (x[i] - xi[i] - R1 + (n[0] + 1) * 1 / 2) ** 2
    S1_squared = S1_squared * 1 / (n[0] - 1)

    S2_squared = 0.0
    for i in range(n[1]):
        S2_squared += (y[i] - yi[i] - R2 + (n[1] + 1) * 1 / 2) ** 2
    S2_squared = S2_squared * 1 / (n[1] - 1)

    sigma_BF_squared = N * S1_squared * 1 / (N - n[0]) + N * S2_squared * 1 / (N - n[1])

    # calculate test statistic
    W_N_BF = (R2 - R1) * 1 / math.sqrt(sigma_BF_squared) * math.sqrt(n[0] * n[1] * 1 / N)

    p_value = 0.0

    if quantile == "normal":
        if alternative == "greater":
            p_value = scipy.stats.norm.cdf(W_N_BF)
        elif alternative == "less":
            p_value = 1 - scipy.stats.norm.cdf(W_N_BF)
        elif alternative == "two.sided":
            p_value = 2 * min(scipy.stats.norm.cdf(W_N_BF), 1 - scipy.stats.norm.cdf(W_N_BF))

        result = namedtuple('BrunnerMunzelResult', ('alternative', 'statistic', 'pvalue'))
        output = result(alternative, W_N_BF, p_value)
    else:
        f_hat = ((1 / N * sigma_BF_squared) ** 2) * 1 / (
                    (S1_squared * 1 / (N - n[0])) ** 2 * 1 / (n[0] - 1) + (S2_squared * 1 / (N - n[1])) ** 2 * 1 / (
                        n[1] - 1))
        if alternative == "greater":
            p_value = scipy.stats.t.cdf(W_N_BF, f_hat)
        elif alternative == "less":
            p_value = 1 - scipy.stats.t.cdf(W_N_BF, f_hat)
        elif alternative == "two.sided":
            p_value = 2 * min(scipy.stats.t.cdf(W_N_BF, f_hat), 1 - scipy.stats.t.cdf(W_N_BF, f_hat))

        result = namedtuple('BrunnerMunzelResult', ('alternative', 'statistic', 'df', 'pvalue'))
        output = result(alternative, W_N_BF, f_hat, p_value)

    return output


def hodges_lehmann(x: list, y: list, alpha = 0.05):
    """
    Function to calculate the Hodges-Lehmann estimator.\n
    Let us assume a location shift effect, i.e., F_1(x) = F(x - mu_1), i = 1,2 and theta = mu_2 - mu_1. \n
    The Hodges-Lehmann estimator theta_hat is an asymptotically unbiased estimator for theta. \n
    Note that the calculated confidence interval is only valid if there are no ties in the data.

    Args:
        x (list(float)): data from first group \n
        y (list(float)): data from second group \n

    Returns:
        float: Hodges-Lehmann estimator theta_hat \n
        float: lower bound for the asymptotic 1-alpha/2 confidence interval for theta \n
        float: upper bound for the 1-alpha/2 confidence interval for theta \n
    """
    n = len(x)
    m = len(y)
    diff = [0 for i in range(n * m)]

    for i in range(n):
        for j in range(m):
            diff[i * m + j] = y[i] - x[j]

    hl = np.median(diff)

    # asymptotic 1-alpha/2 confidence interval
    L = 1 + n*m/2 - scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(n*m*(n + m + 1)*1/12)
    U = n*m/2 + scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(n*m*(n + m + 1)*1/12)

    diff.sort()
    lower = diff[int(round(L) - 1)]
    upper = diff[int(round(U) - 1)]

    result = namedtuple('HodgesLehmannEstimator', ('estimate', 'lowerCI', 'upperCI'))
    output = result(hl, lower, upper)
    return output


def wilcoxon_mann_whitney_test(x: list, y: list, alternative = "two.sided", alpha = 0.05):
    """
    Function to calculate the Wilcoxon-Mann-Whitney test. Exact test not yet implemented.

    Args:
        x (list(float)): data from first group \n
        y (list(float)): data from second group \n
        alternative (str): either 'two.sided', 'less' or 'greater' \n
        alpha (float): 1-alpha/2 confidence interval (only valid if there are no ties)

    Returns:
        str: chosen alternative \n
        list(int): trend \n
        float: test statistic \n
        float: hodges lehmann estimate for a location shift effect \n
        float: lower bound for 1-alpha/2 CI for location shift effect \n
        float: upper bound for 1-alpha/2 CI for location shift effect \n
        float:  p-value
    """

    n = len(x)
    m = len(y)
    data = x + y
    N = len(data)

    hl = hodges_lehmann(x, y, alpha)

    ranks = [0 for i in range(N)]
    ranks = PyNonpar.pseudorank.psrank(data, ranks)

    x = ranks[:n]
    y = ranks[n:]

    R1 = np.mean(x)
    R2 = np.mean(y)

    # variance estimator
    sigma_R_squared = 0.0
    for i in range(N):
        sigma_R_squared += (ranks[i] - (N + 1)*1/2 ) ** 2
    sigma_R_squared = sigma_R_squared*1/(N - 1)

    W_N = math.sqrt(n*m*1/N)*(R2 - R1)*1/math.sqrt(sigma_R_squared)

    if alternative == "greater":
        p_value = scipy.stats.norm.cdf(W_N)
    elif alternative == "less":
        p_value = 1 - scipy.stats.norm.cdf(W_N)
    elif alternative == "two.sided":
        p_value = 2 * min(scipy.stats.norm.cdf(W_N), 1 - scipy.stats.norm.cdf(W_N))

    result = namedtuple('WilcoxonMannWhitneyResult', ('alternative', 'statistic', 'HodgesLehmann', 'lowerCI', 'upperCI', 'pvalue'))
    output = result(alternative, W_N, hl[0], hl[1], hl[2], p_value)

    return output
