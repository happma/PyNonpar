import pytest
import PyNonpar.pseudorank

x = [1, 1, 1, 1, 2, 3, 4, 5, 6]
grp = [1, 1, 2, 2, 2, 3, 3, 3, 3]


def test_psrank():
    assert PyNonpar.pseudorank.psrank(x, grp, ties_method = "average").tolist() == [3.000, 3.000, 3.000, 3.000, 6.000, 6.875, 7.625, 8.375, 9.125]
    assert PyNonpar.pseudorank.psrank(x, grp, ties_method = "max").tolist() == [5.00, 5.00, 5.00, 5.00, 6.00, 6.75, 7.50, 8.25, 9.00]
    assert PyNonpar.pseudorank.psrank(x, grp, ties_method = "min").tolist() == [1.00, 1.00, 1.00, 1.00, 6.00, 7.00, 7.75, 8.50, 9.25]
