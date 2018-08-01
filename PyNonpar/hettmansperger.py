"""
.. py:currentmodule:: hettmansperger

.. module:: hettmansperger
   :platform: Unix, Windows
   :synopsis: Module to calculate the Hettmansperger-Norton test for patterned alternatives in k-sample problems.

.. moduleauthor:: Martin Happ <martin.happ@aon.at>
"""

import pandas as pd
import numpy as np

def hettmansperger_norton_test(data, group, alternative, trend):
    """
    Function to calculate the Hettmansperger-Norton test.

    Args:
        data (list(float)): values to be ranked
        group (list(int)): group factor
        alternative (str): either 'increasing', 'decreasing' or 'custom'
        trend (list(float)): a vector specifying the alternative; only used, if alternative = 'custom'

    Returns:
        float test statistic
    """

    d = {'data': data, 'grp': group}
    df = pd.DataFrame(data=d)
    df["grp"] = df["grp"].astype('category')

    dff = df.sort_values(['data'])
    # dff.assign(grp = dff['grp'][dff.index].values)
    orig_sort = dff.index
    dff = dff.reset_index(drop=True)
    dff['grp'] = dff['grp'].cat.codes

    return 0
