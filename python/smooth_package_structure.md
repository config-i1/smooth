# SMOOTH Package Structure and Function Flow

## Package Overview

The SMOOTH forecasting package is a Python implementation of advanced time series forecasting methods, with ADAM (Augmented Dynamic Adaptive Model) as its central component. The package provides a flexible framework for forecasting that combines various models including ETS (Error, Trend, Seasonal), ARIMA (Autoregressive Integrated Moving Average), and their hybrid combinations.


## Directory Structure

```
smooth/
├── __init__.py
└── adam_general/
    ├── __init__.py
    ├── _adam_general.py  # Low-level implementation functions (e.g., adam_fitter, adam_forecaster)
    └── core/
        ├── __init__.py
        ├── adam.py       # Main ADAM class interface
        ├── checker.py    # Parameter validation
        ├── creator.py    # Model matrix creation, optimization parameter initialization
        ├── estimator.py  # Parameter estimation & model selection
        ├── forecaster/   # Forecast generation
        │   ├── forecaster.py  # Main forecaster logic
        │   ├── result.py      # ForecastResult container class
        │   └── ...            # helpers, intervals, preparator
        └── utils/        # Utilities and helper functions
            ├── __init__.py
            ├── cost_functions.py # Cost functions for optimization (CF, log_Lik_ADAM)
            ├── dump.py         # (Currently empty)
            ├── ic.py           # Information criteria (AIC, BIC, AICc, BICc, ic_function)
            ├── likelihood.py   # (Currently empty)
            ├── polynomials.py  # ARIMA polynomial utilities (adam_polynomialiser)
            ├── utils.py        # General utilities (msdecompose, calculate_acf, calculate_pacf, calculate_likelihood, scaler, etc.)
            └── var_covar.py    # Variance-covariance utilities (sigma, covar_anal, var_anal, matrix_power_wrap)
```

## Core Components

The package consists of five main components that work together to implement the ADAM forecasting framework:

1. **ADAM Class (`adam.py`)**: The high-level interface for users, providing methods to configure, fit, and forecast with the ADAM model.

2. **Parameter Checker (`checker.py`)**: Validates and processes user inputs via `parameters_checker()`, converting them into the appropriate format for model estimation.

3. **Model Creator (`creator.py`)**: Contains functions to define the model structure.
   * `architector()`: Defines the high-level architecture (number of ETS/ARIMA components, lags, profiles).
   * `creator()`: Constructs the state-space matrices (`mat_wt`, `mat_f`, `vec_g`) and initializes the state vector (`mat_vt`).
   * `initialiser()`: Prepares the initial parameter vector (`B`) and bounds (`Bl`, `Bu`) for optimization (called by `estimator.py`).
   * `filler()`: Populates the model matrices with specific parameter values from vector `B` (called during optimization by cost functions and by `preparator()` in `forecaster.py`).

4. **Parameter Estimator (`estimator.py`)**: Estimates optimal model parameters using `estimator()` based on provided data, or selects the best model using `selector()`. It utilizes cost functions (e.g., `CF` from `utils.cost_functions.py`) which internally use `filler()` from `creator.py`.

5. **Forecaster (`forecaster.py`)**: Generates point forecasts and prediction intervals.
   * `preparator()`: Prepares the fitted model and its matrices for forecasting.
   * `forecaster()`: Produces the actual forecast values and intervals.

## Function Flow

The typical workflow follows this sequence:

```
User Input → ADAM.__init__() → ADAM.fit() → ADAM.predict() / ADAM.predict_intervals()
```

Let's look at each step in detail:

### 1. Initialization Phase: `ADAM.__init__()`

```
ADAM.__init__()
├── Store configuration parameters (model, lags, orders, loss, ic, etc.)
└── Set up default values
```

The initialization phase sets up all model configuration parameters such as model type, seasonality, ARIMA orders, loss function, etc. This follows scikit-learn conventions by storing all model-related parameters during initialization.

### 2. Parameter Validation and Data Processing: `ADAM.fit() → checker.py`

```
ADAM.fit(y, X)
├── _check_parameters(ts)  // Calls parameters_checker from checker.py
│   └── parameters_checker() [checker.py]
│       ├── _check_model_composition() // and many other internal _check_* functions
│       ├── _process_observations()
│       ├── _check_lags()
│       ├── _check_persistence()
│       ├── _check_initial() // For how initial states/params are specified (optimal, provided)
│       ├── _check_arima_orders()
│       ├── _check_constant()
│       └── _check_explanatory_vars()
├── _execute_estimation() / _execute_selection()
└── _prepare_results()
```

During the fitting phase, the `parameters_checker()` function in `checker.py` validates and processes all input parameters and data. It checks the model specification, processes observations, validates lags, persistence parameters, initial state/parameter specifications, ARIMA orders, constants, and explanatory variables. The processed parameters are returned as a collection of dictionaries that will be used in subsequent steps.

### 3. Model Structure Creation: `_execute_estimation() → creator.py (architector, creator)`

When `ADAM.fit()` calls `_execute_estimation()`:
```
_execute_estimation()
├── // ... (handle special cases like LASSO/RIDGE)
├── architector() [creator.py] // Defines model architecture
│   ├── _setup_components()    // Determines number of ETS, ARIMA components
│   ├── _setup_lags()          // Finalizes lags based on components
│   └── _create_profiles()     // Creates profile matrices (uses adam_profile_creator)
└── creator() [creator.py]     // Creates state-space matrices and initializes states
    ├── _extract_model_parameters()
    ├── _setup_matrices()          // Initializes mat_vt, mat_wt, mat_f, vec_g
    ├── _setup_measurement_vector()// Configures mat_wt, mat_f (phi)
    ├── _setup_persistence_vector()// Configures mat_f, vec_g (fixed persistence)
    ├── _handle_polynomial_setup() // For fixed ARIMA params
    └── _initialize_states()       // Sets initial values in mat_vt
```
The `architector()` function first defines the counts of various components (ETS, ARIMA) and finalizes the lag structure. Then, `creator()` builds the actual state-space matrices (`mat_wt` for measurement, `mat_f` for transition, `vec_g` for persistence) and the initial state matrix (`mat_vt`), filling them based on the model specification and data characteristics.


### 4. Parameter Estimation: `estimator() [estimator.py]`

If `estimation=True` (default for `_execute_estimation`, or after model selection):
```
_execute_estimation()
└── estimator() [estimator.py] // Called if estimation=True
    ├── initialiser() [creator.py] // Gets initial parameter vector B and bounds Bl, Bu
    ├── _create_objective_function() // Wraps the cost function (e.g., CF from utils.cost_functions.py)
    │   └── CF() [utils.cost_functions.py]
    │       └── filler() [creator.py] // Fills matrices with current B during optimization
    ├── _run_optimization() [nlopt]   // Finds optimal B
    ├── _calculate_loglik()           // Calculates final log-likelihood, AIC, etc.
    ├── _generate_forecasts()         // Generates in-sample forecasts (fitted values)
    ├── _format_output()              // Prepares results (errors, scale, final parameters)
```
The `estimator.py` module handles parameter estimation.
1. It first calls `initialiser()` (from `creator.py`) for the initial parameter guess, sets up and runs the optimization (using a cost function like `CF` from `utils.cost_functions.py`, which internally uses `filler()`), and then processes results (log-likelihood, fitted values, errors, scale).

### 5. Model Selection (Optional): `_execute_selection() → selector() [estimator.py]`

```
_execute_selection()
├── selector() [estimator.py]
│   ├── _form_model_pool() / _build_models_pool_from_components() // Generates candidate models
│   ├── For each candidate model:
│   │   └── _estimate_model() // Calls estimator() for each model
│   │       ├── architector() [creator.py]
│   │       ├── creator() [creator.py]
│   │       └── estimator() [estimator.py] // (Simplified: actual estimation logic)
│   └── _select_best_model() // Based on IC (e.g., AICc)
└── // After best model is selected:
    // _execute_estimation(estimation=False) is called to set up the chosen model's matrices
    ├── architector() [creator.py]
    └── creator() [creator.py]
```
When using model selection (`model_do="select"`), the `selector()` function in `estimator.py` evaluates multiple candidate models. For each candidate, it typically goes through a simplified estimation process to get its information criterion value. The best model is then chosen. Finally, `_execute_estimation(estimation=False)` is called for the selected model to properly set up its matrices using `architector()` and `creator()`.

### 6. Results Preparation: `ADAM._prepare_results()`

```
_prepare_results() // Called in ADAM.fit() after estimation/selection
├── _format_time_series_data() // Ensures y_in_sample, y_holdout are pandas Series
└── _select_distribution()     // Determines final distribution if 'default' was used
```
After model estimation or selection, the results are prepared for user consumption, including formatting time series data and selecting the appropriate distribution for prediction intervals. Fitted parameters are also set as attributes on the ADAM object (e.g., `model.persistence_level_`).

### 7. Forecast Generation: `ADAM.predict() / ADAM.predict_intervals()`

```
ADAM.predict(h, X, ...) or ADAM.predict_intervals(h, X, ...)
├── _validate_prediction_inputs()
├── _prepare_prediction_data()
│   └── preparator() [forecaster.py] // Prepares model for forecasting
│       ├── _fill_matrices_if_needed() // Calls filler() from creator.py
│       ├── _prepare_profiles_recent_table()
│       ├── _prepare_fitter_inputs() // Uses adam_fitter from _adam_general.py for in-sample if needed
│       └── _initialize_fitted_series()
├── _execute_prediction()
│   └── forecaster() [forecaster.py] // Generates forecasts and intervals
│       ├── _prepare_forecast_index()
│       ├── _check_fitted_values()
│       ├── _initialize_forecast_series()
│       ├── _prepare_lookup_table()      // Uses adam_profile_creator from creator.py
│       ├── _prepare_matrices_for_forecast()
│       ├── _generate_point_forecasts()  // Uses adam_forecaster from _adam_general.py
│       ├── _handle_forecast_safety_checks()
│       ├── _process_occurrence_forecast()
│       ├── _prepare_forecast_intervals() // (for predict_intervals or if calculate_intervals=True)
│       │   └── (uses sigma, covar_anal/var_anal from utils.var_covar.py, or simulation)
│       └── _build_ForecastResult()
└── return ForecastResult(mean, lower, upper, level, side, interval)
```
The prediction phase:
1. Validates inputs.
2. Calls `preparator()` (from `forecaster.py`) which readies the fitted model for forecasting. This might involve filling matrices using `filler()` (from `creator.py`) with the estimated parameters if they weren't already in their final form.
3. Calls `forecaster()` (from `forecaster.py`) which generates point forecasts (potentially using `adam_forecaster` from `_adam_general.py`) and, if requested, prediction intervals. Interval calculation can be parametric (using variance calculations from `utils.var_covar.py`) or simulation-based. Returns a `ForecastResult` object with `.mean` (pd.Series), `.lower`/`.upper` (pd.DataFrame or None).

## Data Flow Diagram

```
┌───────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐     ┌───────────┐
│ User Input│────►│ Checker  │────►│ Creator  │────►│ Estimator │────►│ Forecaster│
└───────────┘     └──────────┘     └──────────┘     └───────────┘     └───────────┘
      │                                                                      │
      │                                                                      │
      │                     ┌───────────────────────┐                        │
      └────────────────────  ADAM (Main Interface)   ────────────────────────┘
                            └───────────────────────┘
```

## Main Functions and Their Responsibilities

### ADAM (Main Interface) - `adam.py`

The ADAM class provides the primary user interface, modeled after scikit-learn's API:

- **`__init__(...)`**: Configure the model parameters.
- **`fit(y, X=None)`**: Fit the model to the data. This orchestrates calls to `parameters_checker`, `architector`, `creator`, and `estimator`/`selector`.
- **`predict(h, X=None, interval="none", level=0.95, ...)`**: Generate forecasts. Returns a `ForecastResult` with `.mean` (pd.Series), `.lower`/`.upper` (pd.DataFrame or None).
- **`predict_intervals(h, X=None, levels=[0.8, 0.95], ...)`**: Convenience wrapper calling `predict()` with `interval="prediction"`. Returns `ForecastResult`.

### Checker (Parameter Validation) - `checker.py`

The `parameters_checker()` function validates all user inputs and converts them into the format needed by the rest of the system:
- Validates model specification (ETS components, ARIMA orders).
- Processes observation data, handles occurrence models for intermittent data.
- Checks lags and seasonal periods.
- Validates persistence parameters (smoothing parameters) specifications.
- Checks initial state/parameter specifications (e.g., "optimal", "provided").
- Validates constant terms and explanatory variables.

### Creator (Model Structure and Optimization Initialization) - `creator.py`

This module builds the state-space model structure and prepares for optimization:

- **`architector()`**: Defines the high-level model architecture: number of ETS/ARIMA components, lag structure (calling `_setup_components`, `_setup_lags`), and forecasting profiles (calling `_create_profiles` which uses `adam_profile_creator`).
- **`creator()`**: Constructs the core state-space matrices (measurement `mat_wt`, transition `mat_f`, persistence `vec_g`) and initializes the actual state vector (`mat_vt`) based on data characteristics or provided initial values.
- **`initialiser()`**: **Called by `estimator.py`**. Prepares the initial parameter vector (`B`) for optimization, along with their lower (`Bl`) and upper (`Bu`) bounds, and parameter names.
- **`filler()`**: **Called by cost functions (e.g., `CF`) during optimization and by `preparator()` in `forecaster.py`**. Updates model matrices (`mat_vt`, `mat_wt`, `mat_f`, `vec_g`) based on a given parameter vector `B`.

### Estimator (Parameter Estimation and Model Selection) - `estimator.py`

This module handles parameter optimization and model selection:

- **`estimator()`**: Manages the estimation process for a single model. It calls `initialiser()` (from `creator.py`) for the initial parameter guess, sets up and runs the optimization (using a cost function like `CF` from `utils.cost_functions.py`, which internally uses `filler()`), and then processes results (log-likelihood, fitted values, errors, scale).
- **`selector()`**: Manages the model selection process. It generates a pool of candidate models, estimates each (typically a simplified run of `estimator()`), and selects the best one based on an information criterion (e.g., AICc calculated via `ic_function` from `utils.ic.py`).
- **`CF()` (in `utils.cost_functions.py`)**: The main cost function used by `estimator()`. It takes a parameter vector `B`, uses `filler()` to update matrices, runs the `adam_fitter` (from `_adam_general.py`), and computes the loss (e.g., likelihood, MSE).
- **`log_Lik_ADAM()` (in `utils.cost_functions.py`)**: Calculates the log-likelihood of the ADAM model.

### Forecaster (Forecast Generation) - `forecaster.py`

This module generates forecasts and prediction intervals:

- **`preparator()`**: Prepares the fitted model for forecasting. This involves setting up the final state vector, matrices (possibly calling `filler()` from `creator.py` with estimated parameters), and profiles.
- **`forecaster()`**: Generates point forecasts (using `adam_forecaster` from `_adam_general.py`) and, if requested, prediction intervals. Returns a `ForecastResult` object. Interval calculation can be parametric (using `sigma`, `covar_anal`/`var_anal` from `utils.var_covar.py`) or simulation-based.
- **`ForecastResult`** (in `result.py`): Structured container with `.mean` (pd.Series), `.lower`/`.upper` (pd.DataFrame or None), `.level`, `.side`, `.interval`. Supports backward-compatible DataFrame-style access.
- **`generate_prediction_interval()`**: A utility function for generating prediction intervals, used internally by `forecaster`.


## Class Attributes and Fitted Parameters

After calling `fit()`, the ADAM class stores fitted parameters as attributes with trailing underscores (scikit-learn convention):

- **`persistence_level_`**: Smoothing parameter for level.
- **`persistence_trend_`**: Smoothing parameter for trend.
- **`persistence_seasonal_`**: Smoothing parameters for seasonal components.
- **`persistence_xreg_`**: Smoothing parameters for exogenous regressors.
- **`phi_`**: Damping parameter.
- **`arma_parameters_`**: ARIMA parameters (coefficients for AR and MA terms).
- **`initial_states_`**: Initial state values used for the model.
- (`Other fitted parameters like scale, specific distribution parameters may also be stored`).

## Examples of Usage

Simple example with ETS model:

```python
from smooth.adam_general.core.adam import ADAM
import numpy as np

# Sample data
y_data = np.array([10, 12, 15, 13, 16, 18, 20, 19, 22, 25, 28, 30,
                   11, 13, 16, 14, 17, 19, 21, 20, 23, 26, 29, 31])


# Initialize the model
model = ADAM(model="ANN", lags=[1,12])  # Additive error, no trend, no seasonality, lags for level and seasonality

# Fit the model to data
model.fit(y_data)

# Generate forecasts (returns ForecastResult)
fc = model.predict(h=10)
fc.mean              # pd.Series of point forecasts

# Generate prediction intervals
fc = model.predict(h=10, interval="prediction", level=[0.8, 0.95])
fc.lower             # pd.DataFrame with quantile columns
fc.upper             # pd.DataFrame with quantile columns
fc.to_dataframe()    # flat pd.DataFrame with prefixed column names
```

Example with ARIMA:

```python
# Initialize an ARIMA(1,1,1) model with seasonality 12 for the AR/MA parts too.
# Assuming non-seasonal ARIMA part applied with lags=[1]
# and seasonal ARIMA part (if any) would need lags=[12] and orders specified per lag.
# For a simple ARIMA(p,d,q) on the deseasonalized series (if ETS part exists)
# or on the original series (if no ETS part), lags=[1] is typical for orders.
model_arima = ADAM(ar_order=[1], i_order=[1], ma_order=[1], lags=[1])


# Fit and forecast
model_arima.fit(y_data)
forecasts_arima = model_arima.predict(h=10)
```

Example with exogenous variables:

```python
# Sample exogenous data
X_data = np.random.rand(len(y_data), 2)
X_future = np.random.rand(10, 2)


# Initialize a model with exogenous variables
# Assuming "AAN" with lags=[1,12] for level and season
model_xreg = ADAM(model="AAN", lags=[1,12], regressors="use")

# Fit with exogenous variables
model_xreg.fit(y=y_data, X=X_data)

# Forecast with future exogenous variables
forecasts_xreg = model_xreg.predict(h=10, X=X_future)
```

## Performance Considerations

- The package relies on numerical optimization (NLopt) for parameter estimation, which can be computationally intensive for complex models or large datasets.
- Models with high-frequency seasonality (e.g., hourly data with multiple seasonal lags) or high-order ARIMA components may require more computation time.
- The `fast=True` option in `ADAM` initialization can speed up estimation but might lead to less accurate results.

## Summary of Refactoring Improvements

The refactored package offers several improvements over the original translated code:

1. **Improved Code Organization**: Functions are broken down into smaller, focused units following the single responsibility principle, organized into logical modules (`checker`, `creator`, `estimator`, `forecaster`, `utils`).
2. **Better Documentation**: Comprehensive docstrings and comments explain how the code works (ongoing effort). This markdown document aims to provide a high-level overview.
3. **Standardized Interface**: Follows scikit-learn conventions for a familiar API (`__init__`, `fit`, `predict`).
4. **Improved Type Hints**: Clear type annotations help prevent errors and improve code readability.
5. **Better Error Handling**: More descriptive error messages and robust validation are incorporated.

These improvements make the package more maintainable, easier to understand, and more user-friendly, while preserving the original functionality and accuracy of the forecasting methods. 