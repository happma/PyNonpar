import PyNonpar
import PyNonpar.pseudorank

x = [1, 1, 1, 1, 2, 3, 4, 5, 6]
x2 = [6, 6, 6, 6, 5, 4, 3, 2, 1]
x3 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
x4 = [1, 2, 2, 2, 2, 2, 2, 2, 3]
x5 = [1, 1, 2, 2, 2, 3, 3, 3, 3]
grp = ['A','A','B','B','B','D','D','D','D']

r = [3.000, 3.000, 3.000, 3.000, 6.000, 6.875, 7.625, 8.375, 9.125]
rm = [1.00, 1.00, 1.00, 1.00, 6.00, 7.00, 7.75, 8.50, 9.25]
rM = [5.00, 5.00, 5.00, 5.00, 6.00, 6.75, 7.50, 8.25, 9.00]
r2 = [7.000, 7.000, 7.000, 7.000, 4.000, 3.125, 2.375, 1.625, 0.875]
r3 = [5, 5, 5, 5, 5, 5, 5, 5, 5]
r4 =[1.250, 5.375, 5.375, 5.375, 5.375, 5.375, 5.375, 5.375, 9.125]
r5 = [2, 2, 5, 5, 5, 8, 8, 8, 8]

y2 = [sum(x)*1/2 for x in zip(PyNonpar.pseudorank.psrank(x2, grp, ties_method="min") , PyNonpar.pseudorank.psrank(x2, grp, ties_method="max") )]
y3 = [sum(x)*1/2 for x in zip(PyNonpar.pseudorank.psrank(x3, grp, ties_method="min") , PyNonpar.pseudorank.psrank(x3, grp, ties_method="max") )]
y4 = [sum(x)*1/2 for x in zip(PyNonpar.pseudorank.psrank(x4, grp, ties_method="min") , PyNonpar.pseudorank.psrank(x4, grp, ties_method="max") )]
y5= [sum(x)*1/2 for x in zip(PyNonpar.pseudorank.psrank(x5, grp, ties_method="min") , PyNonpar.pseudorank.psrank(x5, grp, ties_method="max") )]

def test_psrank():
    if PyNonpar.pseudorank.psrank(x, grp, ties_method = "average") != r:
        raise AssertionError()
    if PyNonpar.pseudorank.psrank(x, grp, ties_method = "max") != rM:
        raise AssertionError()
    if PyNonpar.pseudorank.psrank(x, grp, ties_method = "min") != rm:
        raise AssertionError()
    if PyNonpar.pseudorank.psrank(x2, grp, ties_method = "average") != r2:
        raise AssertionError()
    if y2 != r2:
        raise AssertionError()

    if PyNonpar.pseudorank.psrank(x3, grp, ties_method="average") != r3:
        raise AssertionError()
    if y3 != r3:
        raise AssertionError()

    if PyNonpar.pseudorank.psrank(x4, grp, ties_method="average") != r4:
        raise AssertionError()
    if y4 != r4:
        raise AssertionError()

    if PyNonpar.pseudorank.psrank(x5, grp, ties_method="average") != r5:
        raise AssertionError()
    if y5 != r5:
        raise AssertionError()

