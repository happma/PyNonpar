import numpy as np
import pandas as pd
import pytest

import PyNonpar
import PyNonpar.repeated_measures

data = [1, 0, -2, -1, -2, 1, 0, 0, 0, -2]
time = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
subject = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

rChisq = 0.8325700312350129
rF = 0.8409167387320715


def test_kepner_robinson_test():
    tChisq = PyNonpar.repeated_measures.kepner_robinson_test(data, time, subject, distribution="Chisq")
    tF = PyNonpar.repeated_measures.kepner_robinson_test(data, time, subject, distribution="F")
    assert tChisq.pvalue == pytest.approx(rChisq, 0.0001)
    assert tF.pvalue == pytest.approx(rF, 0.0001)


def test_kepner_robinson_test_invalid_input():
    with pytest.raises(TypeError):
        PyNonpar.repeated_measures.kepner_robinson_test(data, time, subject, distribution="bogus")
    with pytest.raises(TypeError):
        PyNonpar.repeated_measures.kepner_robinson_test("not a list", time, subject)
    with pytest.raises(TypeError):
        PyNonpar.repeated_measures.kepner_robinson_test({"a": 1}, time, subject)
    with pytest.raises(TypeError):
        PyNonpar.repeated_measures.kepner_robinson_test(np.array([[1, 2], [3, 4]]), time, subject)


def test_kepner_robinson_test_array_like():
    expected = PyNonpar.repeated_measures.kepner_robinson_test(data, time, subject, distribution="F")
    assert PyNonpar.repeated_measures.kepner_robinson_test(np.array(data), np.array(time), np.array(subject), distribution="F") == expected
    assert PyNonpar.repeated_measures.kepner_robinson_test(tuple(data), tuple(time), tuple(subject), distribution="F") == expected
    assert PyNonpar.repeated_measures.kepner_robinson_test(pd.Series(data), pd.Series(time), pd.Series(subject), distribution="F") == expected
