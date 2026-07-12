# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PyNonpar is a small Python library implementing nonparametric statistical test procedures based on
ranks and pseudo-ranks (Brunner-Munzel, Wilcoxon-Mann-Whitney, Kruskal-Wallis, Hettmansperger-Norton,
Kepner-Robinson, paired-ranks tests). Pseudo-ranks are used instead of ordinary ranks to avoid paradoxical
results in unbalanced/factorial designs; see the references in [README.md](README.md).

## Setup

```
pip install -r requirements.txt
pip install -e .
```

Dependencies: numpy, pandas, scipy, numba (JIT-compiled pseudo-rank recursions), pytest/pytest-cov/codecov.

## Common commands

Run the full test suite (as CI does):
```
pytest --cov=./
```

Run a single test file or test:
```
pytest tests/test_twosample.py
pytest tests/test_twosample.py::test_brunner_munzel_test
```

Build the Sphinx docs (source in `source/`, output in `build/`):
```
make html
```

CI is configured in [.travis.yml](.travis.yml) (Linux, Python 3.6/3.7-dev) and `appveyor.yml` for Windows.

## Architecture

The package (`PyNonpar/`) is organized by test-family module, and every module's public functions are
re-exported via `PyNonpar/__init__.py` (`__all__`). Each test function returns a `collections.namedtuple`
result (e.g. `WilcoxonMannWhitneyResult`, `BrunnerMunzelResult`) bundling the statistic, degrees of freedom
(if applicable), and p-value.

- **`pseudorank.py`** — the foundational module. `psrank(data, group, ties_method)` computes pseudo-ranks
  for observations across groups; almost every test in the other modules calls into this first to convert
  raw data into (pseudo-)ranks before computing a statistic. The actual O(N) recursions for the three
  tie-handling strategies (`_psrank_average`, `_psrank_max`, `_psrank_min`) are implemented as numba
  `@nu.jit(nopython=True)`-compiled functions operating on plain Python lists (not numpy arrays) for
  speed; the public `psrank()` wrapper does the pandas-based sorting/grouping/tie-detection setup and
  restores original row order before returning results as a list.
- **`twosample.py`** — two independent samples: `brunner_munzel_test`, `wilcoxon_mann_whitney_test`
  (asymptotic and exact variants — the exact test uses a memoized (`functools.lru_cache`) recursive
  combinatorial count `_h`), `hodges_lehmann` estimator, and `wilcoxon_mann_whitney_ssp` (sample size
  planning).
- **`twosample_paired.py`** — paired two-sample test (`paired_ranks_test`) and its sample size planning
  counterpart (`paired_ranks_ssp`), for matched/paired observations.
- **`multisample.py`** — more than two independent groups: `kruskal_wallis_test` (can use ranks or
  pseudo-ranks via the `pseudoranks` flag) and `hettmansperger_norton_test` for patterned alternatives
  (trend must be a list of length = number of groups; supports `increasing`/`decreasing`/`custom`).
- **`repeated_measures.py`** — `kepner_robinson_test` for repeated/dependent measurements per subject
  under a compound-symmetry assumption (subplot factor design).

Tests in `tests/` mirror this one-file-per-module layout (`test_pseudorank.py`, `test_twosample.py`, etc.)
and mostly check numeric results against reference values (e.g. from R) rather than testing implementation
details.

## Conventions to follow when editing

- Public functions take plain Python `list` inputs (not numpy arrays) and validate types explicitly at the
  top of the function, raising `TypeError` with a descriptive message — follow this pattern for new public
  functions.
- String-valued options (`alternative`, `method`, `quantile`, `ties_method`) are validated against a fixed
  set of allowed literals rather than using enums.
- Results are returned as `namedtuple`s defined inline in the function, not shared dataclasses.
- Docstrings use Sphinx/reStructuredText-style `Args:`/`Returns:` blocks with `\n` line breaks (consumed by
  the Sphinx docs under `source/`); match this style for any new public function so `make html` keeps working.
