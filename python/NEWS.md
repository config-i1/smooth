# smooth (Python) NEWS

Release history of the Python implementation of the **smooth** forecasting package.


## v1.0.4 (unreleased)

Changes:
* Added support for Python 3.14. Wheels are now built for cp311–cp314.
* Dropped support for Python 3.10. Required because scipy 1.16+, the first to ship cp314 wheels, no longer provides cp310 wheels. `requires-python` is now `>=3.11`.
* Relaxed the upper bound on the scipy dependency to `<1.19.0` (was `<1.16.0`).
* Bumped `cibuildwheel` to 3.x in CI to recognise cp314 build targets.
* `lowess` smoother moved to the **greybox** dependency.
* mypy added to the linting workflow.
* `vcov`, `confint` and `summary` methods added to the `OM` and `OMG` classes, matching the R implementation. OMG uses a joint Fisher Information matrix and prefixes coefficient rows with `A:` / `B:` to identify the sub-model. `bootstrap=True` is reserved but not implemented yet.
* `OM.coef_names` now returns the proper parameter names (`alpha`, `level`, …) instead of falling back to `b1, b2, …`. As a result, `OM.confint` correctly clamps lower bounds for persistence parameters (e.g. `alpha >= 0`) — previously the fallback names silently disabled the clamping.


## v1.0.3 (Release date: 2026-05-11)

Changes:
* Occurrence model ported from R: `OM`, `OMG` (general two-component) and `AutoOM` for automatic occurrence-type selection.
* `SMA` (Simple Moving Average) class with automatic order selection and R parity tests.
* `vcov`, `confint` and `summary` methods for fitted Python models.
* `rstandard` and `rstudent` methods for `OM`.
* Removed the redundant `frequency` parameter — seasonal period is inferred from data, `lags`, or the model spec.
* Numerous bugfixes in `OM`/`OMG` to align with the R implementation.


## v1.0.2 (Release date: 2026-05-05)

Changes:
* `matplotlib` is now a hard requirement.
* Python-version badges added to the README.

Bugfixes:
* `predict()` now recalculates error measures when called repeatedly.
* `xreg` initialisation fixed for multiplicative models and for record-array inputs.
* Fix for simulation-based prediction intervals.
* Fixes in degrees-of-freedom calculation.


## v1.0.1 (Release date: 2026-04-22)

Changes:
* `AutoADAM` — automatic ADAM with distribution and ARIMA-order selection (R parity for `auto.adam()`).
* `AutoMSARIMA` — automatic seasonal ARIMA order selection.
* Plotting for ADAM fits and prediction intervals after `predict()`.
* `rstandard`, `rstudent` and `outlierdummy` methods.
* `autoforecast` print/plot support with error-measure summaries, mirroring R.
* Numerical stability: use `linalg.norm` in place of direct sums of squares where appropriate.

Bugfixes:
* `dgnorm` shape parameter was not being optimised.
* Various MSARIMA / ARMA fixes.


## v1.0.0 (Release date: 2026-04-09)

* First PyPI release. Provides `ADAM`, `ES`, `MSARIMA` and supporting classes, translated from the R implementation.
