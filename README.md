# PyNonpar

[![PyPI version](https://badge.fury.io/py/PyNonpar.svg)](https://badge.fury.io/py/PyNonpar)
[![Travis-CI Build Status](https://travis-ci.org/happma/PyNonpar.svg?branch=master)](https://travis-ci.org/happma/PyNonpar)
[![codecov](https://codecov.io/gh/happma/PyNonpar/branch/master/graph/badge.svg)](https://codecov.io/gh/happma/PyNonpar)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d87d14eb59db450bb7e8f283ad6af7e2)](https://www.codacy.com/project/happma/PyNonpar/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=happma/PyNonpar&amp;utm_campaign=Badge_Grade_Dashboard)

Test statistics based on ranks may lead to paradoxical results. A solution are so-called pseudo-ranks.
This package provides a function to calculate pseudo-ranks which are used in nonparametric statistics for rank tests.
For a definition and discussion of pseudo-ranks, see for example [1].

To install the package from PyPi, simply type
```Python
pip install PyNonpar
```

## Table of Contents
**[Two-Sample Tests](#two-sample-tests)**<br>
**[Multi-Sample Tests](#multi-sample-tests)**<br>

## Pseudo-Ranks
If there are ties (i.e., observations with the same value) in the data, then the pseudo-ranks have to be adjusted. There
are the options 'minimum', 'maximum' and 'average'. It is recommended to use 'average' as for this adjusmtent, normalized
empirical distribution functions are used.
See the example for details on the usage of the function 'psrank'.

```Python
import PyNonpar
from PyNonpar import*

# some artificial data
x = [1, 1, 1, 1, 2, 3, 4, 5, 6]
group = ['C', 'C', 'B', 'B', 'B', 'A', 'C', 'A', 'C']

PyNonpar.pseudorank.psrank(x, group, ties_method = "average")
```

## Nonparametric Test Statistics

### Two-Sample Tests

1. Asymptotic Wilcoxon-Mann-Whitney test: wilcoxon_mann_whitney_test()
2. Brunner-Munzel test (Generalized Wilcoxon test): brunner_munzel_test()

The Hodges-Lehmann estimator can be calculated in a location shift model: hodges_lehmann(). The confidence interval for this estimator is
only asymptotic and assumes continuous distributions. 

### Multi-Sample Tests

1. The Hettmansperger-Norton Test for Patterned Alternatives: hettmansperger_norton_test()
2. Kruskal-Wallis test: kruskal_wallis_test()

####  1. The Hettmansperger-Norton Test for Patterned Alternatives
This package provides a function to calculate the Hettmansperger-Norton test for patterned alternatives
using pseudo-ranks. Originally, this test was developed for ranks but this version was adapted to pseudo-ranks.

For the alternative, it is possible to use 'increasing' (i.e., trend = [1, 2, 3, ..., g]), 'decreasing'
(i.e., trend = [g, g-1, g-2, ..., 1]) or 'custom' where the trend has to be specified manually. Note, that the trend is
a list of length g where g is the number of groups.

```Python
import PyNonpar
from PyNonpar import*

# some artificial data
x = [1, 1, 1, 1, 2, 3, 4, 5, 6]
group = ['C', 'C', 'B', 'B', 'B', 'A', 'C', 'A', 'C']

PyNonpar.hettmansperger.hettmansperger_norton_test(x, group, alternative = "custom", trend = [1,3,2])
```

## References
[1] Brunner, E., Bathke A. C. and Konietschke, F: Rank- and Pseudo-Rank Procedures in Factorial Designs - Using R and SAS, Springer Verlag, to appear.

