import pytest

import PyNonpar
import PyNonpar.twosample

yes = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
no = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]

p_two = 0.005786
p_less = 0.002893
p_greater = 0.9971
p_two_n = 0.001704142
p_less_n = 0.0008520709
p_greater_n = 0.9991479


def test_brunner_munzel_test():
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="less", quantile="t").pvalue == pytest.approx(p_less, 0.01)
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="greater", quantile="t").pvalue == pytest.approx(p_greater, 0.01)
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="two.sided", quantile="t").pvalue == pytest.approx(p_two, 0.01)
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="less", quantile="normal").pvalue == pytest.approx(p_less_n, 0.01)
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="greater", quantile="normal").pvalue == pytest.approx(p_greater_n, 0.01)
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="two.sided", quantile="normal").pvalue == pytest.approx(p_two_n, 0.01)


def test_brunner_munzel_test_invalid_input():
    with pytest.raises(TypeError):
        PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="bogus")
    with pytest.raises(TypeError):
        PyNonpar.twosample.brunner_munzel_test(yes, no, quantile="bogus")
    with pytest.raises(TypeError):
        PyNonpar.twosample.brunner_munzel_test("not a list", no)


x = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3]
y = [4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6]


def test_hodges_lehmann():
    assert PyNonpar.twosample.hodges_lehmann(x, y).estimate == 3


def test_hodges_lehmann_invalid_input():
    with pytest.raises(TypeError):
        PyNonpar.twosample.hodges_lehmann("not a list", y)


x_wilcox = [8, 4, 10, 4, 9, 1, 3, 3, 4, 8]
y_wilcox = [10, 5, 11, 6, 11, 2, 4, 5, 5, 10]

p_two_wilcox = 0.1586455
p_less_wilcox = 0.07932277
p_greater_wilcox = 0.9206772


def test_wilcoxon_mann_whitney_test():
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="less").pvalue == pytest.approx(p_less_wilcox, 0.001)
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="greater").pvalue == pytest.approx(p_greater_wilcox, 0.001)
    result = PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="two.sided")
    assert result.pvalue == pytest.approx(p_two_wilcox, 0.001)
    assert result.lowerCI == -2
    assert result.upperCI == 5


def test_wilcoxon_mann_whitney_test_invalid_input():
    with pytest.raises(TypeError):
        PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="bogus")
    with pytest.raises(TypeError):
        PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, method="bogus")


x_wilcox2 = [4, 6, 9, 6, 2, 2]
y_wilcox2 = [5, 8, 17, 5, 10, 10]
x_wilcox3 = [8, 4, 10, 4, 9, 1, 3, 3, 4, 8]
y_wilcox3 = [10, 5, 11, 6, 11, 2, 4, 5, 5, 10]


def test_wilcoxon_mann_whitney_test_exact():
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox2, y_wilcox2, method="exact", alternative="less").pvalue == pytest.approx(
        0.04329004329004327, 0.0001
    )
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox2, y_wilcox2, method="exact", alternative="greater").pvalue == pytest.approx(
        0.9621212121212122, 0.0001
    )
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox2, y_wilcox2, method="exact", alternative="two.sided").pvalue == pytest.approx(
        0.08658008658008653, 0.0001
    )
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(
        x_wilcox3, y_wilcox3, alternative="greater", method="exact", alpha=0.05
    ).pvalue == pytest.approx(0.9214910476520384, 0.0001)
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(
        x_wilcox3, y_wilcox3, alternative="less", method="exact", alpha=0.05
    ).pvalue == pytest.approx(0.0837915954015025, 0.0001)


x_ssp = [315, 375, 356, 374, 412, 418, 445, 403, 431, 410, 391, 475, 379]
y_ssp = [295, 355, 336, 354, 392, 398, 425, 383, 411, 390, 371, 455, 359]
result_WMWsspN = 111.3719511
result_WMWsspp = 0.3491124


def test_wilcoxon_mann_whitney_ssp():
    result = PyNonpar.twosample.wilcoxon_mann_whitney_ssp(x_ssp, y_ssp, 0.8, 0.05, 1 / 2)
    assert result.N == pytest.approx(result_WMWsspN, 0.01)
    assert result.relEffect == pytest.approx(result_WMWsspp, 0.01)


def test_wilcoxon_mann_whitney_ssp_invalid_input():
    with pytest.raises(TypeError):
        PyNonpar.twosample.wilcoxon_mann_whitney_ssp("not a list", y_ssp)
