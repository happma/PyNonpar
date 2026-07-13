import pytest

import PyNonpar
import PyNonpar.twosample_paired

x = [1, 2, 3, 4, 5, 7, 1, 1, 1]
y = [4, 6, 8, 7, 6, 5, 9, 1, 1]

p_two = 1.3009755244208776e-06
p_less = 6.504877622104388e-07
p_greater = 0.9999993495122378

pt_two = 0.001288777273009245
pt_less = 0.0006443886365046225
pt_greater = 0.9993556113634954

pt_two_varT = 0.043441693355987354
pt_less_varT = 0.021720846677993677
pt_greater_varT = 0.9782791533220063

p_two_varT = 0.016571489879396495
p_less_varT = 0.008285744939698247
p_greater_varT = 0.9917142550603018


def test_paired_ranks_test():
    assert PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="less", quantile="normal").pvalue == pytest.approx(p_less, 0.001)
    assert PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="greater", quantile="normal").pvalue == pytest.approx(p_greater, 0.001)
    assert PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="two.sided", quantile="normal").pvalue == pytest.approx(p_two, 0.001)

    assert PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="less", quantile="t").pvalue == pytest.approx(pt_less, 0.001)
    assert PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="greater", quantile="t").pvalue == pytest.approx(pt_greater, 0.001)
    assert PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="two.sided", quantile="t").pvalue == pytest.approx(pt_two, 0.001)

    assert PyNonpar.twosample_paired.paired_ranks_test(
        x, y, alternative="less", quantile="t", var_equal=True
    ).pvalue == pytest.approx(pt_less_varT, 0.001)
    assert PyNonpar.twosample_paired.paired_ranks_test(
        x, y, alternative="greater", quantile="t", var_equal=True
    ).pvalue == pytest.approx(pt_greater_varT, 0.001)
    assert PyNonpar.twosample_paired.paired_ranks_test(
        x, y, alternative="two.sided", quantile="t", var_equal=True
    ).pvalue == pytest.approx(pt_two_varT, 0.001)

    assert PyNonpar.twosample_paired.paired_ranks_test(
        x, y, alternative="less", quantile="normal", var_equal=True
    ).pvalue == pytest.approx(p_less_varT, 0.001)
    assert PyNonpar.twosample_paired.paired_ranks_test(
        x, y, alternative="greater", quantile="normal", var_equal=True
    ).pvalue == pytest.approx(p_greater_varT, 0.001)
    assert PyNonpar.twosample_paired.paired_ranks_test(
        x, y, alternative="two.sided", quantile="normal", var_equal=True
    ).pvalue == pytest.approx(p_two_varT, 0.001)


def test_paired_ranks_test_invalid_input():
    with pytest.raises(ValueError):
        PyNonpar.twosample_paired.paired_ranks_test(x, y + [1])
    with pytest.raises(TypeError):
        PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="bogus")
    with pytest.raises(TypeError):
        PyNonpar.twosample_paired.paired_ranks_test(x, y, quantile="bogus")
    with pytest.raises(TypeError):
        PyNonpar.twosample_paired.paired_ranks_test("not a list", y)
