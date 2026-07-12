import pytest

import PyNonpar
import PyNonpar.multisample

x = [1, 1, 1, 1, 2, 3, 4, 5, 6]
grp = ['A', 'A', 'B', 'B', 'B', 'D', 'D', 'D', 'D']

pF = 0.03567399334725241
pT = 0.03444397966139712


def test_kruskal_wallis_test():
    assert PyNonpar.multisample.kruskal_wallis_test(x, grp, pseudoranks=False).pvalue == pytest.approx(pF, 0.0001)
    assert PyNonpar.multisample.kruskal_wallis_test(x, grp, pseudoranks=True).pvalue == pytest.approx(pT, 0.0001)


def test_kruskal_wallis_test_invalid_input():
    with pytest.raises(TypeError):
        PyNonpar.multisample.kruskal_wallis_test("not a list", grp)
    with pytest.raises(TypeError):
        PyNonpar.multisample.kruskal_wallis_test(x, grp, pseudoranks="not a bool")


y = [1, 7, 1, 2, 3, 3, 5.5, 6, 7]
group = [1, 1, 1, 2, 2, 3, 3, 3, 3]

rI = 1.47888
rD = -1.47888
rC = 0.3335237


def test_hettmansperger_norton_test():
    assert PyNonpar.multisample.hettmansperger_norton_test(y, group, alternative="increasing").statistic == pytest.approx(rI, 0.01)
    assert PyNonpar.multisample.hettmansperger_norton_test(y, group, alternative="decreasing").statistic == pytest.approx(rD, 0.01)
    assert PyNonpar.multisample.hettmansperger_norton_test(y, group, alternative="custom", trend=[1, 2, 3]).statistic == pytest.approx(rI, 0.01)
    assert PyNonpar.multisample.hettmansperger_norton_test(y, group, alternative="custom", trend=[3, 2, 1]).statistic == pytest.approx(rD, 0.01)
    assert PyNonpar.multisample.hettmansperger_norton_test(y, group, alternative="custom", trend=[1, 3, 2]).statistic == pytest.approx(rC, 0.01)


def test_hettmansperger_norton_test_invalid_input():
    with pytest.raises(TypeError):
        PyNonpar.multisample.hettmansperger_norton_test(y, group, alternative="bogus")
    with pytest.raises(TypeError):
        PyNonpar.multisample.hettmansperger_norton_test(y, group, alternative="custom", trend="not a list")
    with pytest.raises(ValueError):
        PyNonpar.multisample.hettmansperger_norton_test(y, group, alternative="custom", trend=[1, 2])
    with pytest.raises(TypeError):
        PyNonpar.multisample.hettmansperger_norton_test("not a list", group)
