import pytest
import PyNonpar.pseudorank

x = [1, 1, 1, 1, 2, 3, 4, 5, 6]
x2 = [6, 6, 6, 6, 5, 4, 3, 2, 1]
grp = [1, 1, 2, 2, 2, 3, 3, 3, 3]


def test_psrank():
    assert PyNonpar.pseudorank.psrank(x, grp, ties_method = "average") == [3.000, 3.000, 3.000, 3.000, 6.000, 6.875, 7.625, 8.375, 9.125]
    assert PyNonpar.pseudorank.psrank(x, grp, ties_method = "max") == [5.00, 5.00, 5.00, 5.00, 6.00, 6.75, 7.50, 8.25, 9.00]
    assert PyNonpar.pseudorank.psrank(x, grp, ties_method = "min") == [1.00, 1.00, 1.00, 1.00, 6.00, 7.00, 7.75, 8.50, 9.25]
    assert PyNonpar.pseudorank.psrank(x2, grp, ties_method = "average") == [7.000, 7.000, 7.000, 7.000, 4.000, 3.125, 2.375, 1.625, 0.875]
