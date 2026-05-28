# smooth

[![PyPI version](https://img.shields.io/pypi/v/smooth.svg)](https://pypi.org/project/smooth/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/smooth.svg)](https://pypi.org/project/smooth/)
[![Python versions](https://img.shields.io/pypi/pyversions/smooth.svg)](https://pypi.org/project/smooth/)
[![Python CI](https://github.com/config-i1/smooth/actions/workflows/python_ci.yml/badge.svg)](https://github.com/config-i1/smooth/actions/workflows/python_ci.yml)
[![SLSA Build Level 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL--2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)

![hex-sticker of the smooth package for Python](https://github.com/config-i1/smooth/blob/master/python/img/smooth-python-web.png?raw=true)

Python implementation of the **smooth** package for forecasting and time series analysis using Single Source of Error (SSOE) state-space models.

Every wheel published to PyPI is signed via [Sigstore](https://www.sigstore.dev/) on the exact GitHub Actions runner that built it and ships with [PEP 740 attestations](https://peps.python.org/pep-0740/) (SLSA Build Level 3 provenance). Verify a downloaded wheel client-side with [`pypi-attestations`](https://pypi.org/project/pypi-attestations/):

```bash
pip install pypi-attestations
pypi-attestations verify pypi --repository https://github.com/config-i1/smooth smooth-*.whl
```

The package includes the following models:

- [ADAM](https://openforecast.org/adam/) - Augmented Dynamic Adaptive Model, uniting exponential smoothing, ARIMA and regression, implemented in the `ADAM` class.
- [ETS](https://github.com/config-i1/smooth/wiki/ES) - Exponential Smoothing in the SSOE state space form, implemented in the `ES` class.
- [MSARIMA](https://github.com/config-i1/smooth/wiki/MSARIMA) - Multiple seasonal ARIMA in state space form, implemented in the `MSARIMA` class (fixed orders) and `AutoMSARIMA` class (automatic order selection).
- [OM](https://github.com/config-i1/smooth/wiki/OM) - Occurrence Model for intermittent demand, implemented in the `OM` class (plus `OMG` for the general two-component model and `AutoOM` for automatic type selection).
- [SMA](https://github.com/config-i1/smooth/wiki/SMA) - Simple Moving Average in state-space form (an AR(m) model with fixed coefficients), implemented in the `SMA` class with automatic order selection.

All of these are implemented with the support of the following features:

- Automatic components selection in ETS and forecasts combination
- Explanatory variables
- Multiple seasonal models (e.g. for high frequency data)
- Advanced loss functions
- Fine tuning of any elements of ADAM/ETS/ARIMA/Regression
- A variety of prediction interval construction methods

Like the R version, the Python **smooth** depends on the [**greybox**](https://github.com/config-i1/greybox) package for distributions, information criteria, regressor selection, and the LOWESS smoother. It is installed automatically as a dependency.


## Installation

**From PyPI (recommended):**
```bash
pip install smooth
```

**From source (development):**
```bash
pip install "git+https://github.com/config-i1/smooth.git@master#subdirectory=python"
```

See the [Installation Guide](https://github.com/config-i1/smooth/wiki/Installation) for platform-specific instructions.


## System Requirements

If installing from source, this package requires compilation of C++ extensions. Before installing, ensure you have:
- **C++ compiler** (g++, clang++, or MSVC)
- **CMake** >= 3.25
- **Armadillo** linear algebra library

## Quick Example

```python
import numpy as np
from smooth import ADAM

# Sample data
y = np.array([10, 12, 15, 13, 16, 18, 20, 19, 22, 25, 28, 30,
              11, 13, 16, 14, 17, 19, 21, 20, 23, 26, 29, 31])

# Fit ADAM model with additive error, no trend, no seasonality
model = ADAM(model="ANN")
model.fit(y)

# Generate forecasts
forecasts = model.predict(h=12)

# With seasonal component (monthly data, annual seasonality)
model = ADAM(model="ANA", lags=[1, 12])
model.fit(y)
forecasts = model.predict(h=12)
```

## ADAMX — ADAM with Explanatory Variables

This also works with the exponential smoothing (ETSX) via the `ES()` class.

```python
import numpy as np
from smooth import ADAM

# Simulate data where y depends on two external regressors
rng = np.random.default_rng(42)
n = 120
X = rng.standard_normal((n, 2))
y = 10 + 2 * X[:, 0] - 1.5 * X[:, 1] + rng.standard_normal(n)

# Fit ETSX(AAN) — use all regressors with fixed coefficients
model = ADAM(model="AAN", regressors="use")
model.fit(y, X)
print(model)           # shows fitted coefficients including xreg

# Forecast 12 steps ahead with future regressor values
X_future = rng.standard_normal((12, 2))
fc = model.predict(h=12, X=X_future)
print(fc.mean)

# Automatic variable selection (drops insignificant regressors)
model_sel = ADAM(model="AAN", regressors="select")
model_sel.fit(y, X)

# Adaptive (time-varying) regressor coefficients
model_adp = ADAM(model="AAN", regressors="adapt")
model_adp.fit(y, X)
```

`X` accepts a NumPy array or a pandas DataFrame (column names are preserved as regressor names).
`regressors` controls treatment: `"use"` (fixed coefficients), `"select"` (stepwise selection via greybox), or `"adapt"` (ETS-style time-varying coefficients).

## AutoMSARIMA — Automatic ARIMA Order Selection

`AutoMSARIMA` selects the best ARIMA orders automatically using information criteria,
mirroring R's `auto.msarima()`. It fixes `distribution="dnorm"` and uses pure ARIMA
(no ETS components).

```python
import numpy as np
from smooth import AutoMSARIMA

# Monthly time series (e.g. AirPassengers, 144 observations)
y = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    # ... remaining observations
], dtype=float)

# Automatic seasonal ARIMA — searches up to ARIMA(3,2,3)(3,1,3)[12]
model = AutoMSARIMA(lags=[1, 12])
model.fit(y)
print(model)   # AutoMSARIMA: ARIMA([p,P],[d,D],[q,Q])

# Reduce search space for speed
model = AutoMSARIMA(
    lags=[1, 12],
    ar_order=[2, 1],   # max AR: p≤2 at lag 1, P≤1 at lag 12
    i_order=[2, 1],    # max I:  d≤2 at lag 1, D≤1 at lag 12
    ma_order=[2, 1],   # max MA: q≤2 at lag 1, Q≤1 at lag 12
)
model.fit(y)
fc = model.predict(h=24)
```

## Documentation

- [GitHub Wiki](https://github.com/config-i1/smooth/wiki) - Full documentation
- [ADAM](https://github.com/config-i1/smooth/wiki/ADAM) - Main unified ETS/ARIMA framework
- [Installation Guide](https://github.com/config-i1/smooth/wiki/Installation) - Dependencies and troubleshooting

The pages below document the models and their Python classes:

- [ADAM](https://github.com/config-i1/smooth/wiki/ADAM) — Augmented Dynamic Adaptive Model — unified ETS/ARIMA/regression framework
- [AutoADAM](https://github.com/config-i1/smooth/wiki/AutoADAM) — Automatic ADAM with distribution and ARIMA order selection
- [ES](https://github.com/config-i1/smooth/wiki/ES) — Exponential Smoothing (ETS) wrapper for ADAM
- [MSARIMA](https://github.com/config-i1/smooth/wiki/MSARIMA) — Multiple Seasonal ARIMA (fixed orders) and automatic selection (`AutoMSARIMA`)
- [OM](https://github.com/config-i1/smooth/wiki/OM) — Occurrence Model for intermittent demand (`OM`, `OMG`, `AutoOM`)
- [SMA](https://github.com/config-i1/smooth/wiki/SMA) — Simple Moving Average in state-space form with automatic order selection

**Book:** Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic Adaptive Model (ADAM)*. Chapman and Hall/CRC. Online: https://openforecast.org/adam/

## See Also

- [R package on CRAN](https://cran.r-project.org/package=smooth) - Production-ready R implementation
