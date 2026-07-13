"""
.. py:currentmodule:: multisample

.. module:: multisample
   :platform: Unix, Windows
   :synopsis: Module to calculate multi sample tets: Kruskal-Wallis, Hettmansperger-Norton

.. moduleauthor:: Martin Happ <martin.happ@aon.at>
"""

import math
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from numpy.typing import ArrayLike

import PyNonpar.pseudorank
import PyNonpar.pseudorank as ps
from PyNonpar._types import HettmanspergerAlternative, as_list


class KruskalWallisResult(NamedTuple):
    statistic: float
    pvalue: float


class HettmanspergerNortonResult(NamedTuple):
    alternative: str
    weight: Any
    statistic: float
    pvalue: float


def kruskal_wallis_test(data: ArrayLike, group: ArrayLike, pseudoranks: bool = True) -> KruskalWallisResult:
    """
    Function to calculate the Kruskal-Wallis test. It is recommended to use pseudo-ranks as ranks may lead to paradoxical results.\n
    Null hypothesis H_0: F_1 = ... F_a

    Args:
        data (array_like): data from all groups \n
        group (array_like): group factor \n
        pseudoranks (bool): True if pseudo-ranks instead of ranks are used \n

    Returns:
        namedtuple('KruskalWallisResult', ('statistic', 'pvalue')):\n
        test statistic (float)\n
        p-value (float)
    """

    # Check inputs
    data = as_list(data, "data")
    group = as_list(group, "group")
    if not isinstance(pseudoranks, bool):
        raise TypeError("pseudoranks must be a bool")

    N = len(data)
    if pseudoranks:
        ranks = PyNonpar.pseudorank.psrank(data, group)
    else:
        ranks = PyNonpar.pseudorank.psrank(data, [1 for i in range(N)])

    d = {'data': ranks, 'grp': group}
    df = pd.DataFrame(data=d)
    df["grp"] = df["grp"].astype('category')
    df['grp'] = df['grp'].cat.codes
    a = len(np.unique(df['grp']))

    # numerator
    grouped = df.groupby('grp')['data']
    n_series = grouped.size()
    rank_means_series = grouped.mean()
    numerator = float(np.sum(n_series.to_numpy() * ( rank_means_series.to_numpy() - (N + 1)*1/2 ) ** 2))

    denominator = float(np.sum(( df['data'].to_numpy() - (N + 1)*1/2 ) ** 2))

    Q_N = numerator*1/denominator*(N - 1)

    p_value = float(1 - scipy.stats.chi2.cdf(Q_N, a - 1))

    return KruskalWallisResult(Q_N, p_value)


def hettmansperger_norton_test(
    data: ArrayLike, group: ArrayLike, alternative: HettmanspergerAlternative = "increasing", trend: Optional[ArrayLike] = None
) -> HettmanspergerNortonResult:
    """
    Function to calculate the Hettmansperger-Norton test.

    Args:
        data (array_like): data from all groups \n
        group (array_like): group factor \n
        alternative (str): either 'increasing', 'decreasing' or 'custom' \n
        trend (array_like): a vector specifying the alternative; only used, if alternative = 'custom' \n

    Returns:
        namedtuple('HettmanspergerNortonResult', ('alternative', 'weight', 'statistic', 'pvalue')): \n
        chosen alternative (str) \n
        trend (list(float))\n
        test statistic (float)\n
        one sided p-value (float)

    References:
        Hettmansperger, T. P., & Norton, R. M. (1987). Tests for patterned alternatives in
        k-sample problems. Journal of the American Statistical Association, 82(397), 292-299.
    """

    # Check inputs
    data = as_list(data, "data")
    group = as_list(group, "group")
    if (not isinstance(alternative, str)) or (alternative not in ['increasing', 'decreasing', 'custom']):
        raise TypeError("alternative must be either 'increasing', 'decreasing' or 'custom'")
    a_groups = len(np.unique(group))
    if alternative == "custom":
        if trend is None:
            raise TypeError("trend must be array-like when alternative is 'custom'")
        trend = as_list(trend, "trend")
        if len(trend) != a_groups:
            raise ValueError("trend must have one entry per group")

    d = {'data': ps.psrank(data, group, ties_method = "average"), 'grp': group}
    df = pd.DataFrame(data=d)
    df['grp'] = df['grp'].astype(pd.api.types.CategoricalDtype(categories=np.unique(df['grp'].tolist()), ordered=True))

    df = df.sort_values(['grp'])
    # dff.assign(grp = dff['grp'][dff.index].values)
    # orig_sort = df.index
    df = df.reset_index(drop=True)
    df['codes'] = df['grp'].cat.codes

    # calculate group sizes, number of groups and unweighted relative effects
    N = len(df['data'])
    a = len(np.unique(df['codes']))
    n = [0 for x in range(a)]
    p_hat = [0.0 for x in range(a)]

    for i in range(0, a):
        tmp = df[df['codes'] == i]
        n[i] = len(tmp)
        p_hat[i] = 1/N*(float(np.mean(tmp['data'])) - 0.5)

    # define weight function for hypothesis testing
    w: Any = np.arange(1, (a + 1))

    if alternative == "decreasing":
        w = w[::-1]
    elif alternative == "custom":
        w = trend

    # defining matrices and vectors for the computing test statistic
    v1 = [1 for x in range(a)]
    mS = np.diag(n)
    mI = np.diag(v1)
    mJ = np.outer(v1, v1)

    W = np.dot( mS, ( mI - np.dot( 1/N*mJ, mS ) )  )

    # variance estimator
    sum_psranks_squared = float(np.sum((df['data'].to_numpy() - (N + 1)/2) ** 2))
    sum_psranks_squared = (1/N **2)*1/(N-1)*sum_psranks_squared

    sigma_part = np.dot(W, w)
    sigma_hat_squared = N*sum_psranks_squared*np.dot(np.dot( sigma_part.transpose(), np.linalg.inv(mS) ), sigma_part )

    # test statistic
    test_hettmansperger = float(math.sqrt(N)*np.dot(sigma_part.transpose(), p_hat)*1/math.sqrt(sigma_hat_squared))
    p_value = float(1 - scipy.stats.norm.cdf(test_hettmansperger))

    return HettmanspergerNortonResult(alternative, w, test_hettmansperger, p_value)
