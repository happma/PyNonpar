"""
.. py:currentmodule:: multisample

.. module:: multisample
   :platform: Unix, Windows
   :synopsis: Module to calculate multi sample tets: Kruskal-Wallis, Hettmansperger-Norton

.. moduleauthor:: Martin Happ <martin.happ@aon.at>
"""

import pandas as pd
import numpy as np
import PyNonpar.pseudorank as ps
import math
import scipy
from scipy.stats import norm
from scipy.stats import chi2
import collections
from collections import namedtuple
import PyNonpar

def kruskal_wallis_test(data, group, pseudoranks = True):
    """
    Function to calculate the Kruskal-Wallis test. It is recommended to use pseudo-ranks as ranks may lead to paradoxical results.\n
    Null hypothesis H_0: F_1 = ... F_a

    Args:
        data (list(float)): data from all groups \n
        group (list(int)): group factor \n
        pseudoranks (bool): True if pseudo-ranks instead of ranks are used \n

    Returns:
        float: test statistic \n
        float:  p-value
    """

    N = len(data)
    ranks = [0 for i in range(N)]
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
    rank_means = [0.0 for i in range(a)]
    n = [0 for i in range(a)]

    numerator = 0.0
    for i in range(a):
        n[i] = len(df[df['grp'] == i]['data'])
        rank_means[i] = np.mean( df[df['grp'] == i]['data'] )
        numerator += n[i] * ( rank_means[i] - (N + 1)*1/2 ) ** 2

    denominator = 0.0
    for i in range(N):
        denominator += ( df['data'][i] - (N + 1)*1/2 ) ** 2

    Q_N = numerator*1/denominator*(N - 1)

    p_value = 1 - scipy.stats.chi2.cdf(Q_N, a - 1)

    result = namedtuple('KruskalWallisResult', ('statistic', 'pvalue'))
    output = result(Q_N, p_value)

    return output


def hettmansperger_norton_test(data, group, alternative = "increasing", trend = None):
    """
    Function to calculate the Hettmansperger-Norton test.

    Args:
        data (list(float)): data from all groups \n
        group (list(int)): group factor \n
        alternative (str): either 'increasing', 'decreasing' or 'custom' \n
        trend (list(float)): a vector specifying the alternative; only used, if alternative = 'custom' \n

    Returns:
        str: chosen alternative \n
        list(int): trend \n
        float: test statistic \n
        float: one sided p-value

    References:
        Hettmansperger, T. P., & Norton, R. M. (1987). Tests for patterned alternatives in k-sample problems. Journal of the American Statistical Association, 82(397), 292-299.
    """

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
    p_hat = [0 for x in range(a)]

    for i in range(0, a):
        tmp = df[df['codes'] == i]
        n[i] = len(tmp)
        p_hat[i] = 1/N*(np.mean(tmp['data']) - 0.5)

    # define weight function for hypothesis testing
    w = np.arange(1, (a + 1))

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
    sum_psranks_squared = 0
    for i in range(N):
        sum_psranks_squared += (df['data'].tolist()[i] - (N + 1)/2) ** 2
    sum_psranks_squared = (1/N **2)*1/(N-1)*sum_psranks_squared

    sigma_part = np.dot(W, w)
    sigma_hat_squared = N*sum_psranks_squared*np.dot(np.dot( sigma_part.transpose(), np.linalg.inv(mS) ), sigma_part )

    # test statistic
    test_hettmansperger = math.sqrt(N)*np.dot(sigma_part.transpose(), p_hat)*1/math.sqrt(sigma_hat_squared)
    p_value = 1 - scipy.stats.norm.cdf(test_hettmansperger)

    result = namedtuple('HettmanspergerNortonResult', ('alternative', 'weight', 'statistic', 'pvalue'))
    output = result(alternative, w, test_hettmansperger, p_value)

    return output
