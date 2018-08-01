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
    return 0