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
