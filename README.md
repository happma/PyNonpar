# PyNonpar

[![PyPI version](https://badge.fury.io/py/PyNonpar.svg)](https://badge.fury.io/py/PyNonpar)
[![Travis-CI Build Status](https://travis-ci.org/happma/PyNonpar.svg?branch=master)](https://travis-ci.org/happma/PyNonpar)
[![codecov](https://codecov.io/gh/happma/PyNonpar/branch/master/graph/badge.svg)](https://codecov.io/gh/happma/PyNonpar)

Test statistics based on ranks may lead to paradoxical results. A solution are so-called pseudo-ranks.
This package provides a function to calculate pseudo-ranks which are used in nonparametric statistics for rank tests.
For a definition and discussion of pseudo-ranks, see for example [1].

To install the package from PyPi, simply type
```Python
pip install PyNonpar
```

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

### The Hettmansperger-Norton Test for Patterned Alternatives
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

