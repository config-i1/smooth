# SMOOTH Package Structure and Function Flow

## Package Overview

The SMOOTH forecasting package is a Python implementation of advanced time series forecasting methods, with ADAM (Augmented Dynamic Adaptive Model) as its central component. The package provides a flexible framework for forecasting that combines various models including ETS (Error, Trend, Seasonal), ARIMA (Autoregressive Integrated Moving Average), and their hybrid combinations.

## Directory Structure

```
smooth/
├── __init__.py
└── adam_general/
    ├── __init__.py
    ├── _adam_general.py  # Low-level implementation functions
    └── core/
        ├── __init__.py
        ├── adam.py       # Main ADAM class interface
        ├── checker.py    # Parameter validation
        ├── creator.py    # Model matrix creation
        ├── estimator.py  # Parameter estimation
        ├── forecaster.py # Forecast generation
        └── utils/        # Utilities and helper functions
            ├── __init__.py
            ├── ic.py     # Information criteria
            └── ...       # Other utility modules
```

## Core Components

The package consists of five main components that work together to implement the ADAM forecasting framework:

1. **ADAM Class (`adam.py`)**: The high-level interface for users, providing methods to configure, fit, and forecast with the ADAM model.

2. **Parameter Checker (`checker.py`)**: Validates and processes user inputs, converting them into the appropriate format for model estimation.

3. **Model Creator (`creator.py`)**: Creates the state-space matrices and other structures needed for the model.

4. **Parameter Estimator (`estimator.py`)**: Estimates optimal model parameters based on provided data.

5. **Forecaster (`forecaster.py`)**: Generates point forecasts and prediction intervals.

## Function Flow

The typical workflow follows this sequence:

```
User Input → ADAM.__init__() → ADAM.fit() → ADAM.predict() / ADAM.predict_intervals()
```

Let's look at each step in detail:

### 1. Initialization Phase: `ADAM.__init__()`

```
ADAM.__init__()
├── Store configuration parameters
└── Set up default values
```

The initialization phase sets up all model configuration parameters such as model type, seasonality, ARIMA orders, etc. This follows scikit-learn conventions by storing all model-related parameters during initialization.

### 2. Parameter Validation and Data Processing: `ADAM.fit() → checker.py`

```
ADAM.fit(y, X)
├── _check_parameters(ts)
│   └── parameters_checker() [checker.py]
│       ├── _check_model_specification()
│       ├── _process_observations()
│       ├── _check_lags()
│       ├── _check_persistence()
│       ├── _check_initials()
│       ├── _check_arima_orders()
│       ├── _check_constants()
│       └── _check_explanatory_vars()
├── _execute_estimation() / _execute_selection()
└── _prepare_results()
```

During the fitting phase, the `checker.py` module validates and processes all input parameters and data. It checks the model specification, processes observations, validates lags, persistence parameters, initial values, ARIMA orders, constants, and explanatory variables. The processed parameters are returned as a collection of dictionaries that will be used in subsequent steps.

### 3. Model Structure Creation: `_execute_estimation() → creator.py`

```
_execute_estimation()
├── _handle_lasso_ridge_special_case()
├── estimator() [estimator.py]
├── architector() [creator.py]
│   ├── _setup_components()
│   ├── _setup_lags()
│   └── _create_profiles()
└── creator() [creator.py]
    ├── _extract_model_parameters()
    ├── _setup_matrices()
    ├── _setup_measurement_vector()
    ├── _setup_persistence_vector()
    ├── _handle_polynomial_setup()
    └── _initialize_states()
```

The `creator.py` module handles the creation of the state-space matrices and other structures needed for the model. This includes:

- **Architector**: Sets up model components, lag structure, and profiles
- **Creator**: Creates measurement vector, transition matrix, and initializes states
- **Initializer**: Handles initial state estimation
- **Filler**: Fills in model matrices with parameters

### 4. Parameter Estimation: `estimator() [estimator.py]`

```
estimator()
├── _extract_estimation_params()
├── _prepare_optimization_inputs()
├── _run_optimization()
│   ├── _setup_optimization_parameters()
│   └── _execute_optimization()
├── _extract_optimization_results()
├── _generate_forecasts()
├── _update_distribution()
├── _process_initial_values()
├── _process_arma_parameters()
├── _calculate_scale()
└── _format_output()
```

The `estimator.py` module handles parameter estimation through numerical optimization. It sets up the optimization problem, runs the optimizer, and processes the results. The key steps are:

- **Optimization Setup**: Prepares inputs for optimization
- **Optimization Execution**: Runs the optimizer to find optimal parameters
- **Results Processing**: Processes the optimized parameters
- **Statistics Calculation**: Calculates model statistics like AIC, BIC
- **Output Formatting**: Organizes the results for further use

### 5. Model Selection (Optional): `_execute_selection() → selector() [estimator.py]`

```
_execute_selection()
├── selector()
│   ├── _setup_model_selection()
│   ├── _generate_candidate_models()
│   ├── _evaluate_models()
│   └── _select_best_model()
└── For each selected model:
    ├── _update_model_from_selection()
    ├── _create_matrices_for_selected_model()
    └── _update_parameters_for_selected_model()
```

When using model selection, the `selector()` function in `estimator.py` evaluates multiple candidate models and selects the best one based on information criteria.

### 6. Results Preparation: `_prepare_results()`

```
_prepare_results()
├── _format_time_series_data()
└── _select_distribution()
```

After model estimation or selection, the results are prepared for user consumption, including formatting time series data and selecting the appropriate distribution for prediction intervals.

### 7. Forecast Generation: `ADAM.predict() / ADAM.predict_intervals()`

```
ADAM.predict(h, X)
├── _validate_prediction_inputs()
├── _prepare_prediction_data()
│   └── preparator() [forecaster.py]
│       ├── _fill_matrices_if_needed()
│       ├── _prepare_profiles_recent_table()
│       ├── _prepare_fitter_inputs()
│       ├── _correct_multiplicative_components()
│       └── _initialize_fitted_series()
├── _execute_prediction()
│   └── forecaster() [forecaster.py]
│       ├── _prepare_forecast_index()
│       ├── _check_fitted_values()
│       ├── _initialize_forecast_series()
│       ├── _prepare_lookup_table()
│       ├── _prepare_matrices_for_forecast()
│       ├── _generate_point_forecasts()
│       ├── _handle_forecast_safety_checks()
│       ├── _process_occurrence_forecast()
│       ├── _prepare_forecast_intervals() (for predict_intervals)
│       └── _format_forecast_output()
└── return forecasts
```

The prediction phase validates inputs, prepares data for prediction, and executes the forecasting process:

1. **Preparation**: The `preparator()` function prepares the fitted model for forecasting
2. **Forecasting**: The `forecaster()` function generates point forecasts
3. **Intervals**: For `predict_intervals()`, additional prediction intervals are generated

## Data Flow Diagram

```
┌───────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐     ┌───────────┐
│ User Input │────►│ Checker  │────►│ Creator  │────►│ Estimator │────►│ Forecaster│
└───────────┘     └──────────┘     └──────────┘     └───────────┘     └───────────┘
      │                                                                      │
      │                                                                      │
      │                     ┌───────────────────────┐                        │
      └────────────────────► ADAM (Main Interface) ◄────────────────────────┘
                           └───────────────────────┘
```

## Main Functions and Their Responsibilities

### ADAM (Main Interface)

The ADAM class provides the primary user interface, modeled after scikit-learn's API:

- **`__init__()`**: Configure the model parameters
- **`fit(y, X=None)`**: Fit the model to the data
- **`predict(h, X=None)`**: Generate point forecasts
- **`predict_intervals(h, X=None, levels=[0.8, 0.95], side="both")`**: Generate prediction intervals

### Checker (Parameter Validation)

The `parameters_checker()` function validates all user inputs and converts them into the format needed by the rest of the system:

- Validate model specification
- Process observation data
- Check lags and seasonal periods
- Validate persistence parameters (smoothing parameters)
- Check initial state values
- Validate ARIMA orders
- Check constant terms
- Validate explanatory variables

### Creator (Model Structure)

The creator module builds the state-space model structure:

- **`architector()`**: Set up model components and structure
- **`creator()`**: Create state-space matrices (measurement, transition)
- **`initialiser()`**: Initialize state values
- **`filler()`**: Fill matrices with parameters

### Estimator (Parameter Estimation)

The estimator module handles parameter optimization:

- **`estimator()`**: Estimate model parameters
- **`selector()`**: Select the best model from candidates
- Optimization functions for finding optimal parameters
- Calculation of information criteria
- Generation of fitted values and residuals

### Forecaster (Forecast Generation)

The forecaster module generates forecasts:

- **`preparator()`**: Prepare model for forecasting
- **`forecaster()`**: Generate point forecasts and intervals
- Handle occurrence models for intermittent data
- Generate various types of prediction intervals
- Format and return forecast results

## Class Attributes and Fitted Parameters

After calling `fit()`, the ADAM class stores fitted parameters as attributes with trailing underscores (scikit-learn convention):

- **`persistence_level_`**: Smoothing parameter for level
- **`persistence_trend_`**: Smoothing parameter for trend
- **`persistence_seasonal_`**: Smoothing parameters for seasonal components
- **`persistence_xreg_`**: Smoothing parameters for exogenous regressors
- **`phi_`**: Damping parameter
- **`arma_parameters_`**: ARIMA parameters
- **`initial_states_`**: Initial state values

## Examples of Usage

Simple example with ETS model:

```python
from smooth.adam_general.core.adam import ADAM

# Initialize the model
model = ADAM(model="ANN")  # Additive error, no trend, no seasonality

# Fit the model to data
model.fit(y_data)

# Generate forecasts
forecasts = model.predict(h=10)

# Generate prediction intervals
intervals = model.predict_intervals(h=10, levels=[0.8, 0.95])
```

Example with ARIMA:

```python
# Initialize an ARIMA(2,1,1) model
model = ADAM(ar_order=2, i_order=1, ma_order=1)

# Fit and forecast
model.fit(y_data)
forecasts = model.predict(h=10)
```

Example with exogenous variables:

```python
# Initialize a model with exogenous variables
model = ADAM(model="AAN", regressors="use")

# Fit with exogenous variables
model.fit(y=y_data, X=X_data)

# Forecast with future exogenous variables
forecasts = model.predict(h=10, X=X_future)
```

## Performance Considerations

- The package relies on numerical optimization for parameter estimation, which can be computationally intensive for complex models
- Models with high-frequency seasonality (e.g., hourly data) may require more computation time
- Large datasets with many parameters (ARIMA with high orders) may take longer to estimate

## Summary of Refactoring Improvements

The refactored package offers several improvements over the original translated code:

1. **Improved Code Organization**: Functions are broken down into smaller, focused units following the single responsibility principle
2. **Better Documentation**: Comprehensive docstrings and comments explain how the code works
3. **Standardized Interface**: Follows scikit-learn conventions for a familiar API
4. **Improved Type Hints**: Clear type annotations help prevent errors
5. **Better Error Handling**: More descriptive error messages and better validation

These improvements make the package more maintainable, easier to understand, and more user-friendly, while preserving the original functionality and accuracy of the forecasting methods. 