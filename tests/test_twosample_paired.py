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

def test_paired_ranks_test():
    if PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="less", quantile = "normal")[-1] != pytest.approx(p_less, 0.001):
        raise AssertionError()
    if PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="greater", quantile="normal")[-1] != pytest.approx(p_greater, 0.001):
        raise AssertionError()
    if PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="two.sided", quantile="normal")[-1] != pytest.approx(p_two, 0.001):
        raise AssertionError()

    if PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="less", quantile = "t")[-1] != pytest.approx(pt_less, 0.001):
        raise AssertionError()
    if PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="greater", quantile="t")[-1] != pytest.approx(pt_greater, 0.001):
        raise AssertionError()
    if PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="two.sided", quantile="t")[-1] != pytest.approx(pt_two, 0.001):
        raise AssertionError()
