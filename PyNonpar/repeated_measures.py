"""
.. py:currentmodule:: repeated_measures

.. module:: repeated_measures
   :platform: Unix, Windows
   :synopsis: Module to calculate Kepner-Robinson test

.. moduleauthor:: Martin Happ <martin.happ@aon.at>


"""

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import scipy
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike

import PyNonpar.pseudorank
from PyNonpar._types import Distribution, as_list


class KepnerRobinsonResult(NamedTuple):
    statistic: float
    distribution: str
    df1: Optional[float]
    df2: Optional[float]
    relativeEffects: list[float]
    pvalue: float


def kepner_robinson_test(data: ArrayLike, time: ArrayLike, subject: ArrayLike, distribution: Distribution = "F") -> KepnerRobinsonResult:
    """
    Function to calculate the Kepner-Robinson test.

    Args:
        data (array_like): data vector \n
        time (array_like): subplot-factor variable \n
        subject (array_like): factor variable specifying subjects \n
        distribution (str): either 'F' or 'Chisq'. \n

    Returns:
        namedtuple('KepnerRobinsonResult', ('statistic', 'distribution', 'df1', 'df2', 'relativeEffects', 'pvalue')): \n
        test statistic (float)\n
        chosen distribution \n
        degrees of freedom: df1 and df2 for distribution = 'F', only df1 for distribution = 'Chisq' (df2 is None)\n
        relative effects for each level of subplot-factor \n
        p-value (float)
    """

    # Check inputs
    data = as_list(data, "data")
    time = as_list(time, "time")
    subject = as_list(subject, "subject")
    if (not isinstance(distribution, str)) or (distribution not in ['F', 'Chisq']):
        raise TypeError('distribution must be either F or Chisq')

    n = len(set(subject))
    a = len(set(time))
    N = n*a

    grp = [1 for i in range(N)]
    ranks = PyNonpar.pseudorank.psrank(data, grp, ties_method = "average")

    d = {'data': ranks, 'time': time, 'subject': subject}
    df = pd.DataFrame(data=d)
    df["time"] = df["time"].astype('category')
    df['time'] = df['time'].cat.codes
    df["subject"] = df["subject"].astype('category')
    df['subject'] = df['subject'].cat.codes

    # relative effects
    p_hat: list[float] = [0.0 for x in range(a)]
    R_dot_ind: list[float] = [0.0 for x in range(N)]

    for i in range(0, a):
        tmp = df[df['time'] == i]
        p_hat[i] = 1/N*(float(np.mean(tmp['data'])) - 0.5)

    for i in range(0, N):
        tmp = df[df['subject'] == i]
        R_dot_ind[i] = float(np.mean(tmp['data']))


    data_arr = df['data'].to_numpy()
    subject_arr = df['subject'].to_numpy()
    R_dot_ind_arr = np.asarray(R_dot_ind)
    den = float(np.sum(( data_arr - R_dot_ind_arr[subject_arr] ) ** 2))

    p_hat_arr = np.asarray(p_hat)
    test = float(np.sum(( p_hat_arr*N + 1/2 - (N+1)*1/2 ) ** 2))
    test = test*n**2*(a-1)*1/den

    if distribution == "F":
        test = test*1/(a-1)
        pValue = float(1 - scipy.stats.f.cdf(test, a-1, n*(a-1)))
        return KepnerRobinsonResult(test, "F", a-1, n*(a-1), p_hat, pValue)
    else:
        pValue = float(1 - scipy.stats.chi2.cdf(test, a-1))
        return KepnerRobinsonResult(test, "Chisq", a-1, None, p_hat, pValue)
