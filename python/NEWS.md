# smooth (Python) NEWS

Release history of the Python implementation of the **smooth** forecasting package.


## v1.0.5 (unreleased)

Changes:
* Full single-step loss menu (`likelihood`, `MSE`, `MAE`, `HAM`, `LASSO`, `RIDGE`) plus custom-callable loss are now wired up for `OM` and `OMG` on both Python and R. Previously Python `OM` only honoured `likelihood`/`MSE`, Python `OMG` was restricted to `likelihood`, and R `om()`/`omg()` accepted the strings in their signatures but the cost dispatch silently routed everything except `likelihood` and `MSE` to MSE. `LASSO`/`RIDGE` reuse ADAM's `(1 − λ) · √mean(errors²) + λ · penalty(B)` formula (`R/adam.R:894-937`; Python helper `trim_b_for_penalty` factored out of `cost_functions.py` so OM/OMG/ADAM share the exact same B-trimming). The custom callable matches ADAM's `(actual, fitted, B) → scalar` API; for OMG the parameter `B` is the joint vector `concat(B_A, B_B)`. Lambda is exposed as `lambda` on the R signature and `reg_lambda` on the Python signature (consistent with ADAM's existing surface).
* `multicov()` method added to `ADAM`, `ES`, `MSARIMA`, `OM`, and `OMG`, matching R's `multicov.adam` / `multicov.smooth`. Returns the `(h, h)` covariance matrix of multi-step-ahead forecast errors as a `pandas.DataFrame` labelled `h1..hh`. Supports `type="analytical"` (closed-form via the existing `covar_anal` / `var_anal` helpers), `type="empirical"` (rolling-origin cross-product `(errorsᵀ errors) / (nobs - h)` using the existing `rmultistep` method — both R and Python call the same C++ `adamCore::ferrors` backend, so per-cell residuals are bit-equivalent), and `type="simulated"` (averages over `nsim` simulator paths via the existing `adam_simulator` pybind binding).
* `OM.sigma` / `OM.scale` now return `sqrt(mean(residuals²))` — the link-scale residual std-dev — instead of `NaN`. Matches R's `oes_old` (R/oes.R:1253) and `oesg_old` (R/oesg.R:1039, 1049): the OM residuals are on the link-transformed (logit / log-odds) scale, so their second moment is a meaningful scale parameter for the underlying ETS, even though there's no equivalent on the probability axis. Makes `OM.multicov(type='analytical')` produce a finite, PSD covariance matrix that agrees with R to ~5% relative tolerance under matched fits. R-side: a new `sigma.om()` S3 method is registered (R/om.R) so `sigma(om_obj)` and `multicov(om_obj)` work on the R side too — previously they returned `numeric(0)` / crashed because `sigma.adam`'s distribution switch didn't include `"plogis"`.

Bugfixes:
* `ADAM.rmultistep()` (and therefore `multicov(type='empirical')`) no longer crashes with `IndexError: Mat::cols(): indices out of bounds` on models with `i_order >= 1` or seasonal lags greater than 1 — i.e. anything where `lags_model_max >= 2` (most ARIMA models, all MSARIMA defaults, ADAM ETS with seasonal lags >= 2). Root cause: `_compute_multistep_errors` pre-sliced `index_lookup_table[:, lags_model_max:]` before passing it to the C++ `adamCore::ferrors`, which then re-applied a `+lagsModelMax` column offset internally — a double-offset that walked off the end of the table. The fix matches R's `rmultistep.adam` (R/rmultistep.R:101-111), which passes the full lookup unchanged. The case `lags_model_max == 1` (non-seasonal ETS, ARIMA(p,0,q)) used to work by luck — the off-by-one happened to stay in bounds.


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
* `smooth.__version__` is now exposed at the package top level (read from the installed package metadata via `importlib.metadata.version`).
* `AutoADAM` now accepts `lags=12` (scalar) in addition to `lags=[12]` (list). The scalar is normalised to a single-element list, matching `ADAM`'s existing behaviour.
* New ARIMA-orders precedence rule shared by `ADAM` and `AutoADAM`: when `orders` (dict) is supplied it takes precedence and the scalar `ar_order` / `i_order` / `ma_order` arguments are ignored (a warning is emitted). When `orders=None` and one of the scalar arguments has a non-zero value, those are used (as fixed orders by default, or as upper bounds for selection when `arima_select=True`). With no ARIMA spec at all, the model is pure ETS — `AutoADAM`'s `arima_select` default is now `False` so no ARIMA selection runs unless explicitly requested.
* `AutoADAM(verbose=True)` now prints the distribution-selection progress as it runs (mirrors R's `auto.adam` `"Evaluating models with different distributions... dnorm, dlaplace, …, Done!"` output) plus the selected distribution and, when applicable, the selected ARIMA orders.
* `OMG.actuals` (Python) and `actuals.omg` (R) now return the binary occurrence indicator with the **same class** as `actuals(om(y))` would on the same series (`ts` / `zoo` / numeric preserved). The R side additionally stores the input series on the top-level `omg` object as `result$data`.
* `actuals()` on an `OMG` sub-model (R: `omg$modelA` / `omg$modelB`, Python: `OMG.model_a` / `OMG.model_b`) now returns the reconstructed *unobservable* value the sub-model was implicitly fitting before the link function: `fitted + residuals` for `Etype="A"`, `fitted * (1 + residuals)` for `Etype="M"`. R dispatches via a new `omg_submodel` S3 class on the sub-models; Python branches on an `_is_omg_submodel` flag set in `OMG._om_from_side`.
* Log-domain scale computations (`dlnorm`, `dllaplace`, `dls`, `dlgnorm`) now route their `log(1 + errors)` (or `log(1 + errors/yFitted)`) through complex arithmetic — `log(as.complex(...))` in R, `log((...).astype(complex))` in Python — and take the modulus of the resulting complex value. Removes the `RuntimeWarning: invalid value encountered in log` that previously fired when residuals dropped below `-1`, and gives a continuous, finite scale rather than a NaN that propagates downstream.
* `coefbootstrap()` method added to `ADAM`, `ES`, `MSARIMA`, `OM`, and `OMG`, matching R's `coefbootstrap.adam` / `coefbootstrap.om` / `coefbootstrap.omg`. Returns a `BootstrapResult` (the Python analogue of R's `"bootstrap"` S3 class) with the empirical replicate matrix, mean-centred-cross-product covariance, and run metadata. `vcov(bootstrap=True, …)` and `confint(bootstrap=True, …)` now dispatch to it (previously raised `NotImplementedError`). `parallel=True` (or `parallel=<int>`) runs replicates concurrently via `joblib.Parallel` — optional dependency, install with `pip install "smooth[parallel]"` or `pip install joblib`; if not installed, a one-line warning is emitted and execution falls back to serial. The internal sampler now mirrors R's variable-length contiguous-window resampling (R/adam.R:5011-5024) so the empirical bootstrap distribution agrees with R's to within Monte Carlo noise — covered by a new `r_parity`-marked test suite spanning ADAM ETS, ADAM ARIMA, OM, and OMG. Only `method="cr"` (case resampling) is implemented; `method="dsr"` (R's `greybox::dsrboot`) is not yet ported.

Bugfixes:
* `OMG.fitted` (and `model_a.fitted`, `model_b.fitted`) now match R's `omg()` to machine precision on mixed-error scenarios (e.g. `ANA`/`MNM`). The Python OMG post-fit reconstruction was rebuilding the sub-model state-space matrices with the user-requested error/trend/season types (e.g. multiplicative for `MNM`), while R's `omgFinalFit` reuses the forced-additive matrices the joint optimiser saw. The combined link function applied to the rebuilt-multiplicative raw fits drifted from R by up to 5–10% on the probability scale. Python now mirrors R: shares the joint-optimisation matrices with each sub-model post-fit (`OMG._om_from_side`). Coefficients, log-likelihood and loss were already identical between the two implementations.
* Shared C++ numerical Hessian: `src/headers/hessianCore.h` is now the single source of truth for the finite-difference Hessian used by both R (`vcov.adam` / `vcov.om` / `vcov.omg`, plus `vcov.ces` / `vcov.gum` / `vcov.msarima` for symmetry) and Python (`_numDeriv.hessian`). The previous R-side dependency on `pracma::hessian` is removed (`pracma` dropped from `Imports`). Same algorithm, same step size, same operation order on both sides — given the same inputs, the Hessian computation contributes zero divergence between languages. The residual ADAM-with-`initial="optimal"` cross-language gap (~1e-5) is unrelated to the Hessian and traces to NLopt-vs-nloptr converging to non-bit-identical optima on multi-parameter problems with large-magnitude initial states in B; both implementations now share that floor symmetrically.
* `OM.vcov`, `OM.confint`, `OMG.vcov`, `OMG.confint` are now aligned with R's Hessian computation. Three Python-side fixes: (i) the numerical Hessian uses `bounds="none"` (matching R/adam.R:2797 `boundsFI <- "none"`) so finite-difference perturbations near a boundary parameter no longer return the 1e300 penalty and explode the inverse FI; (ii) `invert_fisher_information` takes `abs(diag(...))` to mirror R/adam.R:5226 ("just in case, take absolute values for the diagonal"); (iii) OMG sub-models now receive proper coefficient names (`alpha`, `gamma`, …) from the joint initialiser instead of falling back to `b1, b2, …`, restoring the persistence-parameter clamping in `OMG.confint`. Paired with a fix in R's `vcov.om` (R/om.R fitted-model intake) that preserves the original `initialType`/`nIterations` during the FI refit, both OM and OMG vcov / SE / CI now agree with R to machine precision (~1e-9 to 1e-17).


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
