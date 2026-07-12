"""
.. py:currentmodule:: twosample

.. module:: twosample
   :platform: Unix, Windows
   :synopsis: Module to calculate two sample tests: Brunner-Munzel, Wilcoxon-Mann-Whitney

.. moduleauthor:: Martin Happ <martin.happ@aon.at>
"""

import functools
import math
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import scipy
import scipy.special
import scipy.stats

import PyNonpar.pseudorank
from PyNonpar._types import Alternative, Method, Quantile


class BrunnerMunzelResult(NamedTuple):
    alternative: str
    statistic: float
    df: Optional[float]
    pvalue: float


class HodgesLehmannEstimator(NamedTuple):
    estimate: float
    lowerCI: float
    upperCI: float


class WilcoxonMannWhitneyResult(NamedTuple):
    alternative: str
    statistic: float
    HodgesLehmann: float
    lowerCI: float
    upperCI: float
    pvalue: float


class WilcoxonMannWhitneySampleSize(NamedTuple):
    alpha: float
    power: float
    relEffect: float
    N: float
    t: float
    n1: float
    n2: float
    Nrounded: int
    n1rouned: int
    n2rouned: int


def brunner_munzel_test(x: List[float], y: List[float], alternative: Alternative = "two.sided", quantile: Quantile = "t") -> BrunnerMunzelResult:
    """
    Function to calculate the Brunner-Munzel test. It is recommended to use the t-approximation for small samples.

    Args:
        x (list(float)): data from first group \n
        y (list(float)): data from second group \n
        alternative (str): either 'two.sided', 'less' or 'greater' \n
        quantile (str): either 't' for the small sample approximation with a t-distribution or 'normal' for the large sample procedure. \n

    Returns:
        namedtuple('BrunnerMunzelResult', ('alternative', 'statistic', 'df', 'pvalue')): \n
        chosen alternative (str) \n
        test statistic (float)\n
        estimated degrees of freedom for the t-approximation, only when quantile = 't' (float) \n
        p-value (float)
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

    R2 = float(np.mean(y))
    R1 = float(np.mean(x))

    # calculate variance estimator
    x_arr, xi_arr = np.asarray(x), np.asarray(xi)
    y_arr, yi_arr = np.asarray(y), np.asarray(yi)

    S1_squared = float(np.sum((x_arr - xi_arr - R1 + (n[0] + 1) * 1 / 2) ** 2)) * 1 / (n[0] - 1)
    S2_squared = float(np.sum((y_arr - yi_arr - R2 + (n[1] + 1) * 1 / 2) ** 2)) * 1 / (n[1] - 1)

    sigma_BF_squared = N * S1_squared * 1 / (N - n[0]) + N * S2_squared * 1 / (N - n[1])

    # calculate test statistic
    W_N_BF = (R2 - R1) * 1 / math.sqrt(sigma_BF_squared) * math.sqrt(n[0] * n[1] * 1 / N)

    p_value = 0.0

    if quantile == "normal":
        if alternative == "greater":
            p_value = float(scipy.stats.norm.cdf(W_N_BF))
        elif alternative == "less":
            p_value = float(1 - scipy.stats.norm.cdf(W_N_BF))
        elif alternative == "two.sided":
            p_value = float(2 * min(scipy.stats.norm.cdf(W_N_BF), 1 - scipy.stats.norm.cdf(W_N_BF)))

        output = BrunnerMunzelResult(alternative, float(W_N_BF), None, p_value)
    else:
        f_hat = float(((1 / N * sigma_BF_squared) ** 2) * 1 / (
                    (S1_squared * 1 / (N - n[0])) ** 2 * 1 / (n[0] - 1) + (S2_squared * 1 / (N - n[1])) ** 2 * 1 / (
                        n[1] - 1)))
        if alternative == "greater":
            p_value = float(scipy.stats.t.cdf(W_N_BF, f_hat))
        elif alternative == "less":
            p_value = float(1 - scipy.stats.t.cdf(W_N_BF, f_hat))
        elif alternative == "two.sided":
            p_value = float(2 * min(scipy.stats.t.cdf(W_N_BF, f_hat), 1 - scipy.stats.t.cdf(W_N_BF, f_hat)))

        output = BrunnerMunzelResult(alternative, float(W_N_BF), f_hat, p_value)

    return output


def hodges_lehmann(x: List[float], y: List[float], alpha: float = 0.05) -> HodgesLehmannEstimator:
    """
    Function to calculate the Hodges-Lehmann estimator.\n
    Let us assume a location shift effect, i.e., F_1(x) = F(x - mu_1), i = 1,2 and theta = mu_2 - mu_1. \n
    The Hodges-Lehmann estimator theta_hat is an asymptotically unbiased estimator for theta. \n
    Note that the calculated confidence interval is only valid if there are no ties in the data.

    Args:
        x (list(float)): data from first group \n
        y (list(float)): data from second group \n

    Returns:
        namedtuple('HodgesLehmannEstimator', ('estimate', 'lowerCI', 'upperCI')):\n
        Hodges-Lehmann estimator theta_hat (float) \n
        lower bound for the asymptotic 1-alpha/2 confidence interval for theta (float)\n
        upper bound for the 1-alpha/2 confidence interval for theta (float) \n
    """
    if not isinstance(x, list):
        raise TypeError("x must be a list")
    if not isinstance(y, list):
        raise TypeError("y must be a list")

    n = len(x)
    m = len(y)
    diff: List[float] = [0.0 for i in range(n * m)]

    for i in range(n):
        for j in range(m):
            diff[i * m + j] = y[j] - x[i]

    hl = float(np.median(diff))

    # asymptotic 1-alpha/2 confidence interval
    L = 1 + n*m/2 - scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(n*m*(n + m + 1)*1/12)
    U = n*m/2 + scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(n*m*(n + m + 1)*1/12)

    diff.sort()
    lower = diff[int(round(L) - 1)]
    upper = diff[int(round(U) - 1)]

    return HodgesLehmannEstimator(hl, lower, upper)


@functools.lru_cache(maxsize=1000)
def _h(s: float, m: int, N: int, ranks: Tuple[float, ...]) -> float:
    r = 0.0
    partial_sum_ranks = 0.0
    for i in range(m):
        partial_sum_ranks += ranks[i]

    if s < 0:
        r = 0
    elif m == N:
        if s == partial_sum_ranks:
            r = 1
        else:
            r = 0
    elif m == 0:
        if s == 0:
            r = 1
        else:
            r = 0
    else:
        r = _h(s,m,N-1,ranks) + _h(s-ranks[N-1],m-1,N-1,ranks)

    return r


def wilcoxon_mann_whitney_test(
    x: List[float], y: List[float], alternative: Alternative = "two.sided", alpha: float = 0.05, method: Method = "asymptotic"
) -> WilcoxonMannWhitneyResult:
    """
    Function to calculate the Wilcoxon-Mann-Whitney test.

    Args:
        x (list(float)): data from first group \n
        y (list(float)): data from second group \n
        alternative (str): either 'two.sided', 'less' (x less y) or 'greater' (x greater y) \n
        alpha (float): 1-alpha confidence interval (only valid if there are no ties) \n
        method (str): use 'asymptotic' for the asymptotic test (normal distribution) or the exact test with 'exact'.

    Returns:
        namedtuple('WilcoxonMannWhitneyResult', ('alternative', 'statistic', 'HodgesLehmann', 'lowerCI', 'upperCI', 'pvalue')): \n
        chosen alternative (str) \n
        test statistic (float)\n
        hodges lehmann estimate for a location shift effect y - x (float) \n
        lower bound for 1-alpha CI for location shift effect (float) \n
        upper bound for 1-alpha CI for location shift effect (float) \n
        p-value
    """

    if not isinstance(x, list):
        raise TypeError("x must be a list")
    if not isinstance(y, list):
        raise TypeError("y must be a list")
    if (not isinstance(alternative, str)) or (alternative not in ['two.sided', 'less', 'greater']):
        raise TypeError('alternative must be either two.sided, less or greater')
    if (not isinstance(method, str)) or (method not in ['asymptotic', 'exact']):
        raise TypeError('method must be either asymptotic or exact')

    n = len(x)
    m = len(y)
    data = x + y
    N = len(data)

    hl = hodges_lehmann(x, y, alpha)

    group_dummy: List[int] = [0 for i in range(N)]
    ranks = PyNonpar.pseudorank.psrank(data, group_dummy)

    x = ranks[:n]
    y = ranks[n:]

    R1 = float(np.mean(x))
    R2 = float(np.mean(y))

    # case asymptotic test
    if method == "asymptotic":
        # variance estimator
        ranks_arr = np.asarray(ranks)
        sigma_R_squared = float(np.sum((ranks_arr - (N + 1)*1/2 ) ** 2))*1/(N - 1)

        W_N = math.sqrt(n*m*1/N)*(R2 - R1)*1/math.sqrt(sigma_R_squared)

        if alternative == "greater":
            p_value = float(scipy.stats.norm.cdf(W_N))
        elif alternative == "less":
            p_value = float(1 - scipy.stats.norm.cdf(W_N))
        elif alternative == "two.sided":
            p_value = float(2 * min(scipy.stats.norm.cdf(W_N), 1 - scipy.stats.norm.cdf(W_N)))

        output = WilcoxonMannWhitneyResult(alternative, W_N, hl[0], hl[1], hl[2], p_value)
    elif method == "exact":
        R_W = float(np.sum(y))
        combinations = 0.0
        total_combinations = scipy.special.binom(N, m)

        ranks.sort()
        ranks2_list = [2*ranks[i] for i in range(N)]
        ranks2: Tuple[float, ...] = tuple(ranks2_list)

        for i in range(int(2*R_W)):
            combinations += _h(i, m, N, ranks2)

        if alternative == "two.sided":
            p_value = 2*min(combinations/total_combinations + _h(2*R_W,m,N,ranks2)/total_combinations, 1 - combinations/total_combinations)
        elif alternative == "greater":
            p_value = combinations / total_combinations + _h(2*R_W,m,N,ranks2)/total_combinations
        elif alternative == "less":
            p_value = 1 - combinations / total_combinations
        output = WilcoxonMannWhitneyResult(alternative, R_W, hl[0], hl[1], hl[2], p_value)

    return output


def wilcoxon_mann_whitney_ssp(x: List[float], y: List[float], power: float = 0.8, alpha: float = 0.05, t: float = 1/2) -> WilcoxonMannWhitneySampleSize:
    """
    Function to do sample size planning for the WMW test.

    Args:
        x (list(float)): prior information from first group \n
        y (list(float)): prior information from second group \n
        power (float): probablity to detect an effect based on prior information \n
        alpha (float): type-I error probability \n
        t (float): ratio of subjects assigned to first group.

    Returns:
        namedtuple('WilcoxonMannWhitneyResult', ('alpha', 'power', 'relEffect', 'N', 't', 'n1', 'n2', 'Nrounded', 'n1rouned', 'n2rouned')): \n
        alpha (float) \n
        power(float)\n
        relEffect (float): calculated relative Effect \n
        N (float): minimal sample size required \n
        n1 (float): sample size for group 1 required; N = t*n1 \n
        n2 (float): sample size for group 2 required; N = (1-t)*n1
    """

    if not isinstance(x, list):
        raise TypeError("x must be a list")
    if not isinstance(y, list):
        raise TypeError("y must be a list")

    m1 = len(x)
    m2 = len(y)

    data = x + y
    N = len(data)

    group_dummy: List[int] = [0 for i in range(N)]
    ranks = PyNonpar.pseudorank.psrank(data, group_dummy)

    R1 = ranks[:m1]
    R2 = ranks[m1:]

    # ranks within samples:
    R11 = PyNonpar.pseudorank.psrank(x, [0 for i in range(m1)])
    R22 = PyNonpar.pseudorank.psrank(y, [0 for i in range(m2)])

    # placements
    P1: List[float] = [0.0 for i in range(m1)]
    P2: List[float] = [0.0 for i in range(m2)]

    for i in range(m1):
        P1[i] = R1[i] - R11[i]
    for i in range(m2):
        P2[i] = R2[i] - R22[i]

    # effect size:
    pStar = float((np.mean(R2) - np.mean(R1)) / (m1 + m2) + 0.5)

    # variances:
    sigmaStar = 0.0
    for i in range(m1+m2):
        sigmaStar += (ranks[i] - (m1 + m2) / 2) ** 2
    sigmaStar = float(np.sqrt( sigmaStar * 1/(m1 + m2) ** 3 ))

    P1_mean = float(np.mean(P1))
    sigma1Star = 0.0
    for i in range(m1):
        sigma1Star += (P1[i] - P1_mean) ** 2
    sigma1Star = float(np.sqrt( sigma1Star * 1/(m1 * m2 * m2)  ))

    P2_mean = float(np.mean(P2))
    sigma2Star = 0.0
    for i in range(m2):
        sigma2Star += (P2[i] - P2_mean) ** 2
    sigma2Star = float(np.sqrt( sigma2Star * 1/(m1 * m1 * m2)  ))

    # estimated sample size:
    N_required = float((sigmaStar * scipy.stats.norm.ppf(1 - alpha / 2) + scipy.stats.norm.ppf(power) *
         np.sqrt(t * sigma2Star ** 2 + (1 - t) * sigma1Star ** 2)) ** 2 * 1 / (t * (1 - t) * (pStar - 0.5) ** 2))
    n1 = N_required * t
    n2 = N_required * (1 - t)

    return WilcoxonMannWhitneySampleSize(alpha, power, pStar, N_required, t, n1, n2, math.ceil(N_required), math.ceil(n1), math.ceil(n2))
