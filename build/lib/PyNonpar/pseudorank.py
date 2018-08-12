"""
.. py:currentmodule:: pseudorank

.. module:: pseudorank
   :platform: Unix, Windows
   :synopsis: Module to calculate pseudo-ranks (mid/min/max).

.. moduleauthor:: Martin Happ <martin.happ@aon.at>


"""

import pandas as pd
import numpy as np
import numba as nu


@nu.jit(nopython=True)
def _psrank_average(data: list, group: list, N: int, n: list, a: int) -> list:

    tmp = [0.0 for i in range(len(data))]

    # recursion start
    tmp[0] = 1 / 2 + N / a * 1 / 2 * 1 / n[group[0]]

    # case: no ties
    for i in range(1, N):
        tmp[i] = tmp[i - 1] + N / a * 1 / 2 * (1 / n[group[i]] + 1 / n[group[i - 1]])

    # case: ties
    lpsrank = tmp[:]
    add = 0.0
    j = 0
    i = 0

    while i < (N - 1):
        if data[i] == data[i + 1]:
            add = 0.0
            j = i + 1
            while data[i] == data[j]:
                add += 1 / n[group[j]]
                j += 1
                if j == N:
                    break
            for k in range(i, j):
                if i > 0:
                    lpsrank[k] = tmp[i - 1] + N / a * 1 / 2 * (1 / n[group[i]] + 1 / n[group[i - 1]]) + N / a * 1 / 2 * add
                else:
                    lpsrank[k] = tmp[0] + N / a * 1 / 2 * add
            i = j - 1
        i += 1

    return lpsrank


@nu.jit(nopython=True)
def _psrank_max(data: list, group: list, N: int, n: list, a: int) -> list:

    tmp = [0.0 for i in range(len(data))]

    # recursion start
    tmp[0] = N / a * 1 / n[group[0]]

    # case: no ties
    for i in range(1, N):
        tmp[i] = tmp[i - 1] + N / a * (1 / n[group[i]])

    # case: ties
    lpsrank = tmp[:]
    add = 0.0
    j = 0
    i = 0

    while i < (N - 1):
        if data[i] == data[i + 1]:
            add = 1 / n[group[i]]
            j = i + 1
            while data[i] == data[j]:
                add += 1 / n[group[j]]
                j += 1
                if j == N:
                    break
            for k in range(i, (j)):
                if i > 0:
                    lpsrank[k] = tmp[i - 1] + N / a * add
                else:
                    lpsrank[k] = N / a * add
            i = j - 1
        i += 1

    return lpsrank


@nu.jit(nopython=True)
def _psrank_min(data: list, group: list, N: int, n: list, a: int) -> list:

    tmp = [0.0 for i in range(len(data))]

    # recursion start
    tmp[0] = 1

    # case: no ties
    for i in range(1, N):
        tmp[i] = tmp[i - 1] + N / a * (1 / n[group[i - 1]])

    # case: ties
    lpsrank = tmp[:]
    add = 0.0
    j = 0
    i = 0

    while i < (N - 1):
        if data[i] == data[i + 1]:
            add = 1 / n[group[i]]
            j = i + 1
            while data[i] == data[j]:
                add += 1 / n[group[j]]
                j += 1
                if j == N:
                    break
            for k in range((i + 1), j):
                    lpsrank[k] = tmp[i]
            if j < N:
                lpsrank[j] = tmp[i] + N / a * add
            i = j - 1
        i += 1

    return lpsrank


def psrank(data, group, ties_method = "average"):
    """
    Function to calculate pseudo-ranks.

    Args:
        data (list(float)): values to be ranked \n
        group (list(int)): group factor \n
        ties_method (str): either 'average', 'max' or 'min' for mid, max or min pseudo-ranks \n

    Returns:
        pseudo-ranks (list(float))
    """

    # Check inputs
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    if not isinstance(group, list):
        raise TypeError("group must be a list")
    if (not isinstance(ties_method, str)) or (ties_method not in ['average', 'min', 'max']):
        raise TypeError("ties_method must be either 'average', 'min' or 'max'")

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
        dff['psrank'] = _psrank_max(dff['data'].tolist(), dff['grp'].tolist(), N, n, a)
    elif ties_method == "min":
        dff['psrank'] = _psrank_min(dff['data'].tolist(), dff['grp'].tolist(), N, n, a)
    else:
        dff['psrank'] = _psrank_average(dff['data'].tolist(), dff['grp'].tolist(), N, n, a)

    # sort back
    dff.index = orig_sort
    dff.sort_index(inplace=True)
    return dff['psrank'].tolist()
