import pytest
import PyNonpar.hettmansperger

x = [1, 7, 1, 2, 3, 3, 5.5, 6, 7]
group = [1, 1, 1, 2, 2, 3, 3, 3, 3]

rI = 1.47888
rD = -1.47888
rC = 0.3335237

def test_hettmansperger_norton_test():
    assert PyNonpar.hettmansperger.hettmansperger_norton_test(x, group, alternative = "increasing")[2] == pytest.approx(rI, 0.01)
    assert PyNonpar.hettmansperger.hettmansperger_norton_test(x, group, alternative="decreasing")[2] == pytest.approx(rD,0.01)
    assert PyNonpar.hettmansperger.hettmansperger_norton_test(x, group, alternative="custom", trend = [1, 2, 3])[2] == pytest.approx(rI, 0.01)
    assert PyNonpar.hettmansperger.hettmansperger_norton_test(x, group, alternative="custom", trend = [3, 2, 1])[2] == pytest.approx(rD, 0.01)
    assert PyNonpar.hettmansperger.hettmansperger_norton_test(x, group, alternative="custom", trend=[1, 3, 2])[2] == pytest.approx(rC, 0.01)