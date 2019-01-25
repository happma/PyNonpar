# PyNonpar

[![PyPI version](https://badge.fury.io/py/PyNonpar.svg)](https://badge.fury.io/py/PyNonpar)
[![Travis-CI Build Status](https://travis-ci.org/happma/PyNonpar.svg?branch=master)](https://travis-ci.org/happma/PyNonpar)
[![Build status](https://ci.appveyor.com/api/projects/status/fyui24dq7oj59554?svg=true)](https://ci.appveyor.com/project/happma/pynonpar)
[![codecov](https://codecov.io/gh/happma/PyNonpar/branch/master/graph/badge.svg)](https://codecov.io/gh/happma/PyNonpar)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d87d14eb59db450bb7e8f283ad6af7e2)](https://www.codacy.com/project/happma/PyNonpar/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=happma/PyNonpar&amp;utm_campaign=Badge_Grade_Dashboard)
[![Downloads](https://pepy.tech/badge/pynonpar/week)](https://pepy.tech/project/pynonpar)

Test statistics based on ranks may lead to paradoxical results. A solution are so-called pseudo-ranks.
This package provides a function to calculate pseudo-ranks as well as nonparametric, (pseudo)-rank statistics.
For a definition and discussion of pseudo-ranks, see for example [1].

To install the package from PyPI, simply type
```
pip install PyNonpar
```

## Table of Contents
**[Two-Sample Tests](#two-sample-tests)**<br>
**[Paired Two-Sample Tests](#paired-two-sample-tests)**<br>
**[Multi-Sample Tests](#multi-sample-tests)**<br>
**[Repeated-Measures Tests](#repeated-measures-tests)**<br>

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

1. Wilcoxon-Mann-Whitney test: wilcoxon_mann_whitney_test()
2. Brunner-Munzel test (Generalized Wilcoxon test): brunner_munzel_test()

The Hodges-Lehmann estimator can be calculated in a location shift model: hodges_lehmann(). The confidence interval for this estimator is
only asymptotic and assumes continuous distributions. 

#### 1. Wilcoxon-Mann-Whitney test

For large sample sizes is the asymptotic Wilcoxon test recommended (method = "asymptotic"). For small sample sizes,
we recommend the exact Wilcoxon test. Note that the Wilcoxon test assumes the null hypothesis of equal distributions
H0: F1 = F2.

```Python
import PyNonpar
from PyNonpar import*

x = [8,4,10,4,9,1,3,3,4,8]
y = [10,5,11,6,11,2,4,5,5,10]

PyNonpar.twosample.wilcoxon_mann_whitney_test(x, y, alternative="less", method = "asymptotic", alpha = 0.05)
PyNonpar.twosample.wilcoxon_mann_whitney_test(x, y, alternative="less", method = "exact", alpha = 0.05)

```

#### 2. Brunner-Munzel test
The Brunner-Munzel test extends the Wilcoxon test to the null hypothesis H0: p = 1/2.

```Python
import PyNonpar
from PyNonpar import*

x = [8,4,10,4,9,1,3,3,4,8]
y = [10,5,11,6,11,2,4,5,5,10]

PyNonpar.twosample.brunner_munzel_test(x, y, alternative="less", quantile = "t")
PyNonpar.twosample.brunner_munzel_test(x, y, alternative="less", quantile = "normal")

```

### Paired Two-Sample Tests

#### 1. Paired ranks test

The paired ranks test compares the marginal distributions F1 and F2. The Null hypothesis is H0: F1 = F2 (var_equal = True)
or H0: p = 1/2 (var_equal = False). The two sided alternative is for both cases p != 1/2.

p = Probability(X_i < Y_j) + 1/2 * Probability(X_i = Y_j) for i != j where (X_i, Y_i), (X_j, Y_j) are paired observations.

```Python
import PyNonpar
from PyNonpar import*

x = [1, 2, 3, 4, 5, 7, 1, 1, 1]
y = [4, 6, 8, 7, 6, 5, 9, 1, 1]

PyNonpar.twosample_paired.paired_ranks_test(x, y, alternative="two.sided", var_equal=False, quantile="normal")

```

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

#### 2. Kruskal-Wallis Test
```Python
import PyNonpar
from PyNonpar import*

# some artificial data
x = [1, 1, 1, 1, 2, 3, 4, 5, 6]
group = ['C', 'C', 'B', 'B', 'B', 'A', 'C', 'A', 'C']

# Using pseudo-ranks
PyNonpar.multisample.kruskal_wallis_test(x, group, pseudoranks = True)

# Using ranks
PyNonpar.multisample.kruskal_wallis_test(x, group, pseudoranks = False)
```

### Repeated-Measures Tests
1. The Paired-Ranks Test: paired_ranks_test()
2. The Kepner-Robinson Test Test: kepner_robinson_test()

#### 1. Paired ranks test
See Section ''Paired Twosample Tests''.

#### 2. Kepner-Robinson Test
For the Kepner-Robinson Test we have several dependent observations per subject (subplot factor). Let us denote with F_k the cdf for the k-th observation. The null hypothesis for this test is
H_0: F_1 = ... F_d where d is the number of observations per subject. This test assumes for the dependence structure a compound symmetry, that is, all variances are the same and all covariances are the same. In other words, the observations on one subject can basically be interchanged.
For more information, we refer to [2].


```Python
import PyNonpar
from PyNonpar import*

# some artificial data
data = [1, 0, -2, -1, -2, 1, 0, 0, 0, -2]
time = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
subject = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

PyNonpar.repeated_measures.kepner_robinson_test(data, time, subject, distribution="F")
```


## References
[1] Brunner, E., Bathke A. C. and Konietschke, F: Rank- and Pseudo-Rank Procedures in Factorial Designs - Using R and SAS, Springer Verlag, to appear.
[2] Kepner, J. L., & Robinson, D. H. (1988). Nonparametric methods for detecting treatment effects in repeated-measures designs. Journal of the American Statistical Association, 83(402), 456-461.

