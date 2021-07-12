# Bootstrap.jl: Statistical Bootstrapping


## Motivation

Bootstrapping is a widely applicable technique for statistical estimation.

![img](docs/src/assets/logo.png)


## Functionality

- Bootstrapping statistics with different resampling methods:
  - Random resampling with replacement (`BasicSampling`)
  - Antithetic resampling, introducing negative correlation between samples (`AntitheticSampling`)
  - Balanced random resampling, reducing bias (`BalancedSampling`)
  - Exact resampling, iterating through all unique resamples (`ExactSampling`):
    deterministic bootstrap, suited for small samples sizes
  - Resampling of residuals in generalized linear models (`ResidualSampling`, `WildSampling`)
  - Maximum Entropy bootstrapping for dependent and non-stationary datasets (`MaximumEntropySampling`)

- Confidence intervals:
  - Basic (`BasicConfInt`)
  - Percentile (`PercentileConfInt`)
  - Normal distribution (`NormalConfInt`)
  - Studendized (`StudentConfInt`)
  - Bias-corrected and accelerated (BCa) (`BCaConfInt`)


## Installation

The `Bootstrap` package is part of the Julia ecosphere and the latest release
version can be installed with

```julia
using Pkg
Pkg.add("Bootstrap")
```

More details on packages and how to manage them can be found in the [package
section](https://docs.julialang.org/en/v1/stdlib/Pkg/) of the Julia
documentation.


## Examples

This example illustrates the basic usage and cornerstone functions of the package.
More elaborate cases are covered in the documentation notebooks.

```julia
  using Bootstrap
```

Our observations in `some_data` are sampled from a standard normal distribution.

```julia
  some_data = randn(100);
```

Let's bootstrap the standard deviation (`std`) of our data, based on 1000
resamples and with different bootstrapping approaches.

```julia
  using Statistics  # the `std` methods live here
  
  n_boot = 1000

  ## basic bootstrap
  bs1 = bootstrap(std, some_data, BasicSampling(n_boot))

  ## balanced bootstrap
  bs2 = bootstrap(std, some_data, BalancedSampling(n_boot))
```

We can explore the properties of the bootstrapped samples, for example, the
estimated bias and standard error of our statistic.

```julia
  bias(bs1)
  stderror(bs1)
```

*Note*: do not confuse function ```stderror``` provided by the`Bootstrap` package with function stderr provided by Julia base.

Furthermore, we can estimate confidence intervals (CIs) for our statistic of
interest, based on the bootstrapped samples.

```julia
  ## calculate 95% confidence intervals
  cil = 0.95;

  ## basic CI
  bci1 = confint(bs1, BasicConfInt(cil));

  ## percentile CI
  bci2 = confint(bs1, PercentileConfInt(cil));

  ## BCa CI
  bci3 = confint(bs1, BCaConfInt(cil));

  ## Normal CI
  bci4 = confint(bs1, NormalConfInt(cil));
```


## References

The [bootstrapping wikipedia article](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
is a comprehensive introduction into the topic.  An extensive description of the
bootstrap is the focus of the book *Davison and Hinkley (1997):
[Bootstrap Methods and Their Application](http://statwww.epfl.ch/davison/BMA/)*.
Most of the methodology covered in the book is implemented in the
[boot](https://cran.r-project.org/web/packages/boot/index.html) package for the
[R programming language](https://www.r-project.org/). [More references](docs/src/references.md)
are listed in the documentation for further reading.


## Contributions and Feedback

Contributions of any kind are very welcome. Please feel free to open pull
requests or issues if you have suggestions for changes, ideas or questions.


## Frequently Asked Questions

- Does it have anything to do with twitter themes, webpage frameworks,
  compiling, ...?

  No, not really. This package focuses on an interesting area in statistics, but
  the term _bootstrapping_ is also used in other contexts. You can check
  wikipedia for a longer list of
  [meanings associated with bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(disambiguation)).


## Package Status

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliangehring.github.io/Bootstrap.jl/stable)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.596080.svg)](https://doi.org/10.5281/zenodo.596080)

![Testing](https://github.com/juliangehring/Bootstrap.jl/workflows/Testing/badge.svg)

[![Coverage](https://codecov.io/gh/juliangehring/Bootstrap.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliangehring/Bootstrap.jl)

The package uses [semantic versioning](https://semver.org/).
