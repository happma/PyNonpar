import pytest
import PyNonpar.twosample

yes = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
no = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]

p_two = 0.005786
p_less = 0.002893
p_greater = 0.9971

def test_brunner_munzel_test():
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="less", quantile = "t")[-1] == pytest.approx(p_less, 0.01)
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="greater", quantile="t")[-1] == pytest.approx(p_greater, 0.01)
    assert PyNonpar.twosample.brunner_munzel_test(yes, no, alternative="two.sided", quantile="t")[-1] == pytest.approx(p_two, 0.01)


x = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3]
y = [4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6]

def test_hodges_lehmann():
    assert PyNonpar.twosample.hodges_lehmann(x, y)[0] == 3


x_wilcox = [8,4,10,4,9,1,3,3,4,8]
y_wilcox = [10,5,11,6,11,2,4,5,5,10]

p_two_wilcox = 0.1586455
p_less_wilcox = 0.07932277
p_greater_wilcox = 0.9206772

def test_wilcoxon_mann_whitney_test():
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="less")[-1] == pytest.approx(p_less_wilcox, 0.001)
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="greater")[-1] == pytest.approx(p_greater_wilcox, 0.001)
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="two.sided")[-1] == pytest.approx(p_two_wilcox, 0.001)
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="two.sided")[-3] == -2
    assert PyNonpar.twosample.wilcoxon_mann_whitney_test(x_wilcox, y_wilcox, alternative="two.sided")[-2] == 5
