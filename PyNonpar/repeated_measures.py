"""
.. py:currentmodule:: repeated_measures

.. module:: repeated_measures
   :platform: Unix, Windows
   :synopsis: Module to calculate Kepner-Robinson test

.. moduleauthor:: Martin Happ <martin.happ@aon.at>


"""

import pandas as pd
import numpy as np
import math
import scipy
import scipy.stats
import scipy.special
from collections import namedtuple
import PyNonpar.pseudorank
import functools
from functools import lru_cache



def kepner_robinson_test(data, time, subject, distribution="F"):
    """
    Function to calculate the Kepner-Robinson test.

    Args:
        data (list(float)): data vector \n
        time (list(float)): subplot-factor variable \n
        subject (list(float)): factor variable specifying subjects \n
        distribution (str): either 'F' or 'Chisq'. \n

    Returns:
        namedtuple('KepnerRobinsonTest', ('statistic', 'distribution', 'df1', 'df2', 'relativeEffects', 'pvalue')): \n
        test statistic (float)\n
        chosen distribution \n
        degrees of freedom, either df1 and df2 (F) or df (Chisq)\n
        relative effects for each level of subplot-factor \n
        p-value (float)
    """

    # Check inputs
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    if not isinstance(time, list):
        raise TypeError("time must be a list")
    if not isinstance(subject, list):
        raise TypeError("subject must be a list")
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
    p_hat = [0 for x in range(a)]
    R_dot_ind = [0 for x in range(N)]

    for i in range(0, a):
        tmp = df[df['time'] == i]
        p_hat[i] = 1/N*(np.mean(tmp['data']) - 0.5)

    for i in range(0, N):
        tmp = df[df['subject'] == i]
        R_dot_ind[i] = np.mean(tmp['data'])


    den = 0
    for i in range(0, N):
        den += ( df['data'][i] - R_dot_ind[df['subject'][i]] ) ** 2

    test = 0
    for i in range(0, a):
        test += ( p_hat[i]*N + 1/2 - (N+1)*1/2 ) ** 2
    test = test*n**2*(a-1)*1/den

    pValue = 0

    if distribution == "F":
        test = test*1/(a-1)
        pValue = 1 - scipy.stats.f.cdf(test, a-1, n*(a-1))
        result = namedtuple('KepnerRobinsonTest', ('statistic', 'distribution', 'df1', 'df2', 'relativeEffects', 'pvalue'))
        output = result(test, "F", a-1, n*(a-1), p_hat, pValue)

    if distribution == "Chisq":
        pValue = 1 - scipy.stats.chi2.cdf(test, a-1)
        result = namedtuple('KepnerRobinsonTest', ('statistic', 'distribution', 'df', 'relativeEffects', 'pvalue'))
        output = result(test, "Chisq", a-1, p_hat, pValue)

    return output