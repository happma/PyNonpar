"""
.. py:currentmodule:: pseudorank

.. module:: pseudorank
   :platform: Unix, Windows
   :synopsis: Module to calculate pseudo-ranks (mid/min/max).

.. moduleauthor:: Martin Happ <martin.happ@aon.at>


"""

import pandas as pd
import numpy as np


def _psrank_average(dff: pd.DataFrame, N: int, n: list, a: int) -> pd.DataFrame:

    # recursion start
    dff.ix[0, 'tmp'] = 1 / 2 + N / a * 1 / 2 * 1 / n[dff.ix[0, 'grp']]

    # case: no ties
    for i in range(1, N):
        dff.ix[i, 'tmp'] = dff.ix[(i - 1), 'tmp'] + N / a * 1 / 2 * (1 / n[dff['grp'][i]] + 1 / n[dff['grp'][i - 1]])

    # case: ties
    dff['psrank'] = dff['tmp'].astype('float')
    add = 0
    j = 0
    i = 0

    while i < (N - 1):
        if dff.ix[i, 'data'] == dff.ix[(i + 1), 'data']:
            add = 0
            j = i + 1
            while dff.ix[i, 'data'] == dff.ix[j, 'data']:
                add += 1 / n[dff['grp'][j]]
                j += 1
                if j == N:
                    break
            for k in range(i, (j)):
                if i > 0:
                    dff.ix[k, 'psrank'] = dff.ix[(i - 1), 'tmp'] + N / a * 1 / 2 * (
                            1 / n[dff['grp'][i]] + 1 / n[dff['grp'][i - 1]])
                    dff.ix[k, 'psrank'] += N / a * 1 / 2 * add
                else:
                    dff.ix[k, 'psrank'] = dff.ix[0, 'tmp'] + N / a * 1 / 2 * add
            i = j - 1
        i += 1

    return dff


def _psrank_max(dff: pd.DataFrame, N: int, n: list, a: int) -> pd.DataFrame:

    # recursion start
    dff.ix[0, 'tmp'] = N / a * 1 / n[dff.ix[0, 'grp']]

    # case: no ties
    for i in range(1, N):
        dff.ix[i, 'tmp'] = dff.ix[(i - 1), 'tmp'] + N / a * (1 / n[dff['grp'][i]])

    # case: ties
    dff['psrank'] = dff['tmp'].astype('float')
    add = 0
    j = 0
    i = 0

    while i < (N - 1):
        if dff.ix[i, 'data'] == dff.ix[(i + 1), 'data']:
            add = 1 / n[dff['grp'][i]]
            j = i + 1
            while dff.ix[i, 'data'] == dff.ix[j, 'data']:
                add += 1 / n[dff['grp'][j]]
                j += 1
                if j == N:
                    break
            for k in range(i, (j)):
                if i > 0:
                    dff.ix[k, 'psrank'] = dff.ix[(i - 1), 'tmp']
                    dff.ix[k, 'psrank'] += N / a * add
                else:
                    dff.ix[k, 'psrank'] = N / a * add
            i = j - 1
        i += 1

    return dff


def _psrank_min(dff: pd.DataFrame, N: int, n: list, a: int) -> pd.DataFrame:

    # recursion start
    dff.ix[0, 'tmp'] = 1

    # case: no ties
    for i in range(1, N):
        dff.ix[i, 'tmp'] = dff.ix[(i - 1), 'tmp'] + N / a * (1 / n[dff['grp'][i - 1]])

    # case: ties
    dff['psrank'] = dff['tmp'].astype('float')
    add = 0
    j = 0
    i = 0

    while i < (N - 1):
        if dff.ix[i, 'data'] == dff.ix[(i + 1), 'data']:
            add = 1 / n[dff['grp'][i]]
            j = i + 1
            while dff.ix[i, 'data'] == dff.ix[j, 'data']:
                add += 1 / n[dff['grp'][j]]
                j += 1
                if j == N:
                    break
            for k in range((i + 1), j):
                    dff.ix[k, 'psrank'] = dff.ix[i, 'tmp']
            if j < N:
                dff.ix[j, 'psrank'] = dff.ix[i, 'tmp'] + N / a * add
            i = j - 1
        i += 1

    return dff


def psrank(data, group, ties_method = "average"):
    """
    Function to calculate pseudo-ranks.

    Args:
        data (list(float)): values to be ranked
        group (list(int)): group factor
        ties_method (str): either 'average', 'max' or 'min' for mid, max or min pseudo-ranks

    Returns:
        pd.Series of pseudo-ranks.
    """

    d = {'data': data, 'grp': group}
    df = pd.DataFrame(data=d)
    df["grp"] = df["grp"].astype('category')

    dff = df.sort_values(['data'])
    # dff.assign(grp = dff['grp'][dff.index].values)
    orig_sort = dff.index
    dff = dff.reset_index(drop=True)
    dff['grp'] = dff['grp'].cat.codes

    # calculate pseudo-ranks
    dff['tmp'] = dff['data'].astype('float')
    N = len(dff['data'])
    a = len(np.unique(dff['grp']))
    n = [0 for x in range(a)]

    for i in range(a):
        n[i] = len(dff[dff['grp'] == i])

    if ties_method == "max":
        dff = _psrank_max(dff, N, n, a)
    elif ties_method == "min":
        dff = _psrank_min(dff, N, n, a)
    else:
        dff = _psrank_average(dff, N, n, a)

    # sort back
    dff.index = orig_sort
    dff.sort_index(inplace=True)
    return dff['psrank']
