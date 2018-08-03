"""
.. py:currentmodule:: hettmansperger

.. module:: hettmansperger
   :platform: Unix, Windows
   :synopsis: Module to calculate the Hettmansperger-Norton test for patterned alternatives in k-sample problems using pseudo-ranks.

.. moduleauthor:: Martin Happ <martin.happ@aon.at>
"""

import pandas as pd
import numpy as np
import PyNonpar.pseudorank as ps
import math
import scipy
from scipy.stats import norm
import collections
from collections import namedtuple

def hettmansperger_norton_test(data, group, alternative = "increasing", trend = None):
    """
    Function to calculate the Hettmansperger-Norton test.

    Args:
        data (list(float)): values to be ranked
        group (list(int)): group factor
        alternative (str): either 'increasing', 'decreasing' or 'custom'
        trend (list(float)): a vector specifying the alternative; only used, if alternative = 'custom'

    Returns:
        str: chosen alternative
        list(int): trend
        float: test statistic
        float: one sided p-value
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

    #print("Hettmansperger-Norton test")
    #print("Alternative:", alternative)
    #print("Weight:", w)
    #print("Test Statistic:", test_hettmansperger)
    #print("p-value (one-sided):", p_value)

    result = namedtuple('HettmanspergerNortonResult', ('alternative', 'weight', 'statistic', 'pvalue'))
    output = result(alternative, w, test_hettmansperger, p_value)

    return output
