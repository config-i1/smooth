# smooth (Python)

[![Python CI](https://github.com/config-i1/smooth/actions/workflows/python_ci.yml/badge.svg)](https://github.com/config-i1/smooth/actions/workflows/python_ci.yml)

Python implementation of the **smooth** package for time series forecasting using Single Source of Error (SSOE) state-space models.

![hex-sticker of the smooth package for Python](https://github.com/config-i1/smooth/blob/master/python/img/smooth-python.png?raw=true)

**Status:** Work in progress

## Installation

**From GitHub:**
```bash
pip install "git+https://github.com/config-i1/smooth.git@master#subdirectory=python"
```

**From source (development):**
```bash
git clone https://github.com/config-i1/smooth.git
cd smooth/python
pip install -e ".[dev]"
```

## System Requirements

This package requires compilation of C++ extensions. Before installing, ensure you have:
- **C++ compiler** (g++, clang++, or MSVC)
- **CMake** >= 3.25
- **Armadillo** linear algebra library

See the [Installation Guide](https://github.com/config-i1/smooth/wiki/Installation) for platform-specific instructions.

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

## Documentation

- [GitHub Wiki](https://github.com/config-i1/smooth/wiki) - Full documentation
- [ADAM](https://github.com/config-i1/smooth/wiki/ADAM) - Main unified ETS/ARIMA framework
- [Installation Guide](https://github.com/config-i1/smooth/wiki/Installation) - Dependencies and troubleshooting

**Book:** Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic Adaptive Model (ADAM)*. Chapman and Hall/CRC. Online: https://openforecast.org/adam/

## See Also

- [R package on CRAN](https://cran.r-project.org/package=smooth) - Production-ready R implementation
