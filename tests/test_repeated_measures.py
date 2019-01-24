import pytest
import PyNonpar
import PyNonpar.repeated_measures

data = [1 ,  0 ,  -2 ,  -1 ,  -2 ,  1 ,  0 ,  0 ,  0 ,  -2]
time = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
subject = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

rChisq = 0.8325700312350129
rF = 0.8409167387320715

tChisq = PyNonpar.repeated_measures.kepner_robinson_test(data, time, subject, distribution="Chisq")
tF = PyNonpar.repeated_measures.kepner_robinson_test(data, time, subject, distribution="F")

def test_wilcoxon_mann_whitney_test_exact():
    if tChisq[-1] != pytest.approx(rChisq, 0.0001):
        raise AssertionError()
    if tF[-1] != pytest.approx(rF, 0.0001):
        raise AssertionError()