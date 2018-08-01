import pytest
import PyNonpar.pseudorank

x = [1, 1, 1, 1, 2, 3, 4, 5, 6]
grp = [1, 1, 2, 2, 2, 3, 3, 3, 3]


def test_psrank():
    assert PyNonpar.pseudorank.psrank(x, grp).tolist() == [3.000, 3.000, 3.000, 3.000, 6.000, 6.875, 7.625, 8.375, 9.125]