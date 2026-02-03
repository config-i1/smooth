import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general.core.checker import parameters_checker
from smooth.adam_general.core.creator import architector, creator
from smooth.adam_general.core.estimator import (
    estimator,
    selector,
)
from smooth.adam_general.core.forecaster import forecaster, preparator
from smooth.adam_general.core.utils.ic import ic_function

# Note: adam_cpp instance is stored in self and passed to forecasting functions


# Type hint groups
DISTRIBUTION_OPTIONS = Literal[
    "default", "dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dinvgauss", "dgamma"
]

LOSS_OPTIONS = Literal[
    "likelihood",
    "GPL",
    "MSE",
    "MAE",
    "HAM",
    "MSEh",
    "MAEh",
    "HAMh",
    "MSCE",
    "MACE",
    "CHAM",
    "TMSE",
    "TMAE",
    "THAM",
    "GTMSE",
    "GTAME",
    "GTHAM",
    "LASSO",
    "RIDGE",
]

OCCURRENCE_OPTIONS = Literal[
    "none", "auto", "fixed", "general", "odds-ratio", "inverse-odds-ratio", "direct"
]

INITIAL_OPTIONS = Optional[
    Union[
        Dict[str, Any],
        Literal["backcasting", "optimal", "complete", "two-stage", "provided"],
        Tuple[str, ...],
    ]
]


class ADAM:
    """
    ADAM: Augmented Dynamic Adaptive Model for Time Series Forecasting.

    ADAM is an advanced state-space modeling framework that combines
    **ETS** (Error, Trend, Seasonal) and **ARIMA** components into a unified
    Single Source of Error (SSOE) model. It provides a flexible, data-driven
    approach to time series forecasting with automatic model selection,
    parameter estimation, and prediction intervals.

    **Mathematical Form**:

    The ADAM model is specified in state-space form as:

    .. math::

        y_t &= o_t(w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l})\\epsilon_t)

        v_t &= f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_t)\\epsilon_t

    where:

    - :math:`y_t`: Observed value at time t
    - :math:`o_t`: Occurrence indicator (Bernoulli variable for intermittent
      data, 1 otherwise)
    - :math:`v_t`: State vector (level, trend, seasonal, ARIMA components)
    - :math:`l`: Vector of lags
    - :math:`x_t`: Vector of exogenous variables
    - :math:`a_t`: Parameters for exogenous variables
    - :math:`w(\\cdot)`: Measurement function
    - :math:`r(\\cdot)`: Error function (additive or multiplicative)
    - :math:`f(\\cdot)`: Transition function (state evolution)
    - :math:`g(\\cdot)`: Persistence function (smoothing parameters)
    - :math:`\\epsilon_t`: Error term (various distributions supported)

    **Key Features**:

    1. **Unified Framework**: Seamlessly combines ETS and ARIMA in a single model
    2. **Multiple Seasonality**: Supports multiple seasonal periods
       (e.g., daily + weekly)
    3. **Automatic Selection**: Branch & Bound algorithm for efficient model selection
    4. **Flexible Distributions**: Normal, Laplace, Gamma, Log-Normal, and more
    5. **Intermittent Demand**: Built-in occurrence models for sparse data
    6. **External Regressors**: Include covariates with adaptive or fixed coefficients
    7. **Scikit-learn Compatible**: Familiar `.fit()` and `.predict()` API

    **Model Specification**:

    Models are specified using a string notation:

    - **ETS Models**: "ETS" where E=Error, T=Trend, S=Seasonal

      * E (Error): "A" (Additive), "M" (Multiplicative)
      * T (Trend): "N" (None), "A" (Additive), "Ad" (Additive Damped),
        "M" (Multiplicative), "Md" (Multiplicative Damped)
      * S (Seasonal): "N" (None), "A" (Additive), "M" (Multiplicative)

      Examples: "ANN" (Simple Exponential Smoothing), "AAN" (Holt's Linear),
      "AAA" (Holt-Winters Additive)

    - **Automatic Selection**:

      * "ZZZ": Select best model using Branch & Bound
      * "XXX": Select only additive components
      * "YYY": Select only multiplicative components
      * "ZXZ": Auto-select error and seasonal, additive trend only (**default**, safer)
      * "FFF": Full search across all 30 ETS model types

    - **ARIMA Models**: Specified via `ar_order`, `i_order`, `ma_order` parameters

      * Supports seasonal ARIMA: SARIMA(p,d,q)(P,D,Q)m
      * Multiple seasonality: e.g., hourly data with daily (24) and
        weekly (168) patterns

    **Supported Error Distributions**:

    - **Normal** (``distribution="dnorm"``): Default for additive errors
    - **Gamma** (``distribution="dgamma"``): Default for multiplicative errors
    - **Laplace** (``distribution="dlaplace"``): For heavy-tailed errors (MAE loss)
    - **Log-Normal** (``distribution="dlnorm"``): For positive-only data
    - **Inverse Gaussian** (``distribution="dinvgauss"``): For skewed positive data
    - **S distribution** (``distribution="ds"``): For extremely heavy-tailed data
    - **Generalized Normal** (``distribution="dgnorm"``): Flexible shape parameter

    Distribution is auto-selected based on loss function if ``distribution=None``.

    **Loss Functions**:

    - ``loss="likelihood"``: Maximum likelihood estimation (**default**)
    - ``loss="MSE"``: Mean Squared Error
    - ``loss="MAE"``: Mean Absolute Error
    - ``loss="HAM"``: Half-Absolute Moment
    - ``loss="MSEh"``: Multi-step MSE (h-step ahead)
    - ``loss="LASSO"``: L1 regularization for variable selection
    - ``loss="RIDGE"``: L2 regularization for shrinkage

    **Multistep loss functions**:

    - ``loss="MSEh"``: Mean Squared Error for specific h-steps ahead
    - ``loss="TMSE"``: Trace Mean Squared Error (sum of MSEh from 1 to h)
    - ``loss="GTMSE"``: Geometric Trace Mean Squared Error (sum of logs of MSEh
      from 1 to h)
    - ``loss="MSCE"``: Mean Squared Cumulative Error (sum of MSEh from 1 to h and
      covariances between them)
    - ``loss="GPL"``: Generalised Predictive Likelihood (minimum of the determinant of
      the covariance matrix of multistep errors)

    **Initialization Methods**:

    - ``initial="optimal"``: Optimize all initial states (default)
    - ``initial="backcasting"``: Use backcasting to initialize states
    - ``initial="two-stage"``: Backcast then optimize
    - ``initial="complete"``: Pure backcasting without optimization
    - ``initial={"level": 100, ...}``: Provide custom initial states

    **Workflow Example**:

    .. code-block:: python

        from smooth import ADAM
        import numpy as np

        # Generate sample data
        y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 3)

        # Automatic model selection
        model = ADAM(model="ZZZ", lags=[12], ic="AICc")
        ADAM_fit = model.fit(y)
        print(ADAM_fit)

        # Generate 12-step ahead forecasts with intervals
        forecasts = model.predict(h=12, calculate_intervals=True, level=0.95)
        print(forecasts)

        # Access fitted parameters
        print(f"Model: {model.model_type_dict['model']}")
        print(f"Alpha: {model.persistence_level_:.3f}")
        print(f"AICc: {model.ic_selection}")

    **Attributes (After Fitting)**:

    The model stores fitted results as attributes with trailing underscores
    (scikit-learn convention):

    - ``persistence_level_``: Level smoothing parameter (α)
    - ``persistence_trend_``: Trend smoothing parameter (β)
    - ``persistence_seasonal_``: Seasonal smoothing parameter(s) (γ)
    - ``phi_``: Damping parameter (φ)
    - ``initial_states_``: Estimated initial states
    - ``arma_parameters_``: AR/MA coefficients (if ARIMA)

    Additional fitted attributes:

    - ``model_type_dict``: Complete model specification
    - ``adam_estimated``: Full estimation results
    - ``adam_created``: State-space matrices
    - ``ic_selection``: Information criterion value
    - ``prepared_model``: Model prepared for forecasting

    **Performance Considerations**:

    - **Small Data** (T < 100): Use "backcasting" initialization,
      it's faster
    - **Large Data** (T > 1000): "optimal" initialization works well
    - **Multiple Seasonality**: Can be slow; consider simpler models
      first
    - **Model Selection**: "ZZZ" with Branch & Bound is much faster than
      "FFF" exhaustive search

    **Common Use Cases**:

    1. **Automatic Forecasting**: ``ADAM(model="ZXZ", lags=[12])`` -
       Let the model choose
    2. **Intermittent Demand**: ``ADAM(model="ANN", occurrence="auto")`` -
       For sparse data
    3. **External Regressors**: ``ADAM(model="AAN").fit(y, X=regressors)`` -
       Include covariates
    4. **Multiple Seasonality**: ``ADAM(model="AAA", lags=[24, 168])`` -
       Hourly data with daily/weekly patterns
    5. **ARIMA**: ``ADAM(model="NNN", ar_order=1, i_order=1, ma_order=1)`` -
       Pure ARIMA(1,1,1)
    6. **Custom Model**: ``ADAM(model="MAM", persistence={"alpha": 0.3})`` -
       Fix some parameters

    **Comparison to R's smooth::adam**:

    This Python implementation is a direct translation of the R smooth
    package's ``adam()`` function, maintaining mathematical equivalence
    while adapting to scikit-learn conventions:

    - R: ``adam(data, model="ZZZ", h=10)`` →
      Python: ``ADAM(model="ZZZ").fit(y).predict(h=10)``
    - R: ``persistence=list(alpha=0.3)`` → Python: ``persistence={"alpha": 0.3}``
    - R: ``orders=list(ar=c(1,1))`` → Python: ``ar_order=[1, 1]``

    **References**:

    - Svetunkov, I. (2023). Forecasting and Analytics with the Augmented Dynamic
      Adaptive Model. https://openforecast.org/adam/
    - Hyndman, R.J., et al. (2008). "Forecasting with Exponential Smoothing"
    - Svetunkov, I. & Boylan, J.E. (2017). "State-space ARIMA for
      supply-chain forecasting"
    - Svetunkov, I. & Kourentzes, N. & Killick, R. (2023). "Multi-step
      estimators and shrinkage effect in time series models".
      DOI: 10.1007/s00180-023-01377-x

    See Also
    --------
    adam.fit : Fit the ADAM model to data
    adam.predict : Generate point forecasts
    adam.predict_intervals : Generate prediction intervals
    selector : Automatic model selection function
    estimator : Parameter estimation function
    forecaster : Forecasting function
    print : Print the outputs of the ADAM class

    Examples
    --------
    Simple exponential smoothing::

        >>> from smooth import ADAM
        >>> import numpy as np
        >>> y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118])
        >>> model = ADAM(model="ANN", lags=[1])
        >>> model.fit(y)
        >>> forecasts = model.predict(h=6)
        >>> print(forecasts)

    Automatic model selection with multiple seasonality::

        >>> model = ADAM(model="ZZZ", lags=[12], ic="AICc")
        >>> model.fit(y)
        >>> print(f"Selected model: {model.model_type_dict['model']}")

    SARIMA(1,1,1)(1,1,1)₁₂::

        >>> model = ADAM(
        ...     model="NNN",  # Pure ARIMA
        ...     ar_order=[1, 1],
        ...     i_order=[1, 1],
        ...     ma_order=[1, 1],
        ...     lags=[1, 12]
        ... )
        >>> model.fit(y)
        >>> forecasts = model.predict(h=12)

    With external regressors::

        >>> X = np.random.randn(len(y), 2)  # Two regressors
        >>> X_future = np.random.randn(6, 2)  # Regressors for forecast period
        >>> model = ADAM(model="AAN", regressors="use")
        >>> model.fit(y, X=X)
        >>> forecasts = model.predict(h=6, X=X_future)

    Fix some parameters, estimate others::

        >>> model = ADAM(
        ...     model="AAA",
        ...     lags=[12],
        ...     persistence={"alpha": 0.3},  # Fix alpha, estimate beta and gamma
        ...     initial="backcasting"
        ... )
        >>> model.fit(y)

    Intermittent demand forecasting::

        >>> sparse_data = np.array([0, 0, 15, 0, 0, 23, 0, 0, 0, 18, 0, 0])
        >>> model = ADAM(model="ANN", occurrence="auto")
        >>> model.fit(sparse_data)
        >>> forecasts = model.predict(h=6)  # Accounts for zero-demand probability
    """

    def __init__(
        self,
        model: Union[str, List[str]] = "ZXZ",
        lags: Optional[NDArray] = None,
        # ARIMA specific parameters
        ar_order: Union[int, List[int]] = 0,
        i_order: Union[int, List[int]] = 0,
        ma_order: Union[int, List[int]] = 0,
        arima_select: bool = False,
        # end of ARIMA specific parameters
        constant: bool = False,
        regressors: Literal["use", "select", "adapt"] = "use",
        distribution: Optional[DISTRIBUTION_OPTIONS] = None,
        loss: LOSS_OPTIONS = "likelihood",
        loss_horizon: Optional[int] = None,
        # outlier detection
        outliers: Literal["ignore", "detect", "use"] = "ignore",
        outliers_level: float = 0.99,
        # end of outlier detection
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        occurrence: OCCURRENCE_OPTIONS = "none",
        # ---- These are the estimated parameters that we can choose to fix ----
        # Dictionary of terms e.g. {"alpha": 0.5, "beta": 0.5}
        persistence: Optional[Dict[str, float]] = None,
        phi: Optional[float] = None,
        initial: INITIAL_OPTIONS = "backcasting",
        # Number of iterations for backcasting (default 2 for backcasting, 1 otherwise)
        n_iterations: Optional[int] = None,
        # TODO: enforce the structure of this
        arma: Optional[Dict[str, Any]] = None,
        # ----- End of parameters----
        verbose: int = 0,
        # Parameters moved from fit
        h: Optional[int] = None,
        holdout: bool = False,
        model_do: Literal["estimate", "select", "combine"] = "estimate",
        fast: bool = False,
        models_pool: Optional[List[Dict[str, Any]]] = None,
        lambda_param: Optional[float] = None,
        frequency: Optional[str] = None,
        # Profile parameters
        profiles_recent_provided: bool = False,
        profiles_recent_table: Optional[Any] = None,
        # Fisher information matrix: We're skipping for now and we'll use composition
        # for it like Grid Search in scikit-learn.
        # initial values for optimization parameters:
        nlopt_initial: Optional[Dict[str, Any]] = None,
        nlopt_upper: Optional[Dict[str, Any]] = None,
        nlopt_lower: Optional[Dict[str, Any]] = None,
        nlopt_kargs: Optional[Dict[str, Any]] = None,
        # specific to losses or distributions
        reg_lambda: Optional[float] = None,
        gnorm_shape: Optional[float] = None,
        smoother: Literal["lowess", "ma", "global"] = "lowess",
        **kwargs,
    ) -> None:
        """
        Initialize the ADAM model with specified parameters.

        Parameters
        ----------
        model : Union[str, List[str]], default="ZXZ"
            Model specification string (e.g., "ANN" for ETS) or
            list of model strings.
        lags : Optional[Union[int, List[int]]], default=None
            Seasonal period(s). Can be a single integer or a list of integers.
            E.g., ``lags=12`` is equivalent to ``lags=[12]``.
        ar_order : Union[int, List[int]], default=0
            Autoregressive order(s) for ARIMA components.
        i_order : Union[int, List[int]], default=0
            Integration order(s) for ARIMA components.
        ma_order : Union[int, List[int]], default=0
            Moving average order(s) for ARIMA components.
        arima_select : bool, default=False
            Whether to perform automatic ARIMA order selection.
        constant : bool, default=False
            Whether to include a constant term.
        regressors : Literal["use", "select", "adapt"], default="use"
            How to handle external regressors.
        distribution : Optional[DISTRIBUTION_OPTIONS], default=None
            Error distribution. If None, it is selected automatically based
            on the loss function.
        loss : LOSS_OPTIONS, default="likelihood"
            Loss function for parameter estimation.
        loss_horizon : Optional[int], default=None
            Number of steps for multi-step loss functions (e.g., MSEh).
        outliers : Literal["ignore", "detect", "use"], default="ignore"
            Outlier handling method.
        outliers_level : float, default=0.99
            Confidence level for outlier detection.
        ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
            Information criterion for model selection.
        bounds : Literal["usual", "admissible", "none"], default="usual"
            Parameter bounds specification during optimization.
        occurrence : OCCURRENCE_OPTIONS, default="none"
            Occurrence model type for intermittent data.
        persistence : Optional[Dict[str, float]], default=None
            Fixed persistence parameters (e.g., {"alpha": 0.5, "beta": 0.5}).
            If None, parameters are estimated.
        phi : Optional[float], default="backcasting"
            Fixed damping parameter for damped trend models. If None,
            estimated if applicable.
        initial : INITIAL_OPTIONS, default=None
            Method for initializing states or fixed initial states. Can be a string
            (e.g., 'optimal', 'backcasting'), a dictionary of initial state values,
            or a tuple of state names to initialize.
        arma : Optional[Dict[str, Any]], default=None
            Fixed ARMA parameters specification. If None, estimated if applicable.
        verbose : int, default=0
            Verbosity level (0=silent, higher values indicate more output).
        h : Optional[int], default=None
            Forecast horizon. If None during initialization, can be set in `predict`.
        holdout : bool, default=False
            Whether to use a holdout sample for validation during the fit process.
        model_do : Literal["estimate", "select", "combine"], default="estimate"
            Action to perform:
            - "estimate": Estimate a single specified model.
            - "select": Select the best model from a pool or based on components.
            - "combine": Combine forecasts from multiple models (Not Implemented).
        fast : bool, default=False
            Whether to use faster, possibly less accurate, estimation methods.
        models_pool : Optional[List[Dict[str, Any]]], default=None
            A pool of model configurations for selection or combination.
        lambda_param : Optional[float], default=None
            Lambda parameter for Box-Cox transformation or regularization.
        frequency : Optional[str], default=None
            Time series frequency (e.g., "D", "M", "Y").
            Inferred if data is pandas Series with DatetimeIndex.
        profiles_recent_provided : bool, default=False
            Whether recent profiles (e.g., for exogenous variables) are provided.
        profiles_recent_table : Optional[Any], default=None
            Table containing recent profiles data.
        nlopt_initial : Optional[Dict[str, Any]], default=None
            Initial values for optimization parameters for NLopt solver.
        nlopt_upper : Optional[Dict[str, Any]], default=None
            Upper bounds for optimization parameters for NLopt solver.
        nlopt_lower : Optional[Dict[str, Any]], default=None
            Lower bounds for optimization parameters for NLopt solver.
        nlopt_kargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments for optimization. Supported keys:

            - ``print_level`` (int): Verbosity level for optimization progress
              (default=0). When >0, prints parameter vector B and cost function
              value on every iteration.
              Output format: ``Iter N: B=[val1, val2, ...] -> CF=value``
            - ``xtol_rel`` (float): Relative tolerance on optimization parameters
              (default=1e-6). Optimization stops when parameter changes are
              smaller than xtol_rel * |params|.
            - ``xtol_abs`` (float): Absolute tolerance on optimization parameters
              (default=1e-8). Optimization stops when parameter changes are
              smaller than xtol_abs.
            - ``ftol_rel`` (float): Relative tolerance on function value
              (default=1e-8). Optimization stops when CF changes are smaller
              than ftol_rel * |CF|.
            - ``ftol_abs`` (float): Absolute tolerance on function value (default=0).
              Optimization stops when CF changes are smaller than ftol_abs.
            - ``algorithm`` (str): NLopt algorithm name
              (default="NLOPT_LN_NELDERMEAD"). Common alternatives:
              "NLOPT_LN_SBPLX" (Subplex), "NLOPT_LN_COBYLA" (COBYLA),
              "NLOPT_LN_BOBYQA" (BOBYQA). Use "LN_" prefix for derivative-free
              algorithms.

            Example::

                model = ADAM(model="AAN", nlopt_kargs={
                    "print_level": 1,
                    "xtol_rel": 1e-8,
                    "algorithm": "NLOPT_LN_SBPLX"
                })
        reg_lambda : Optional[float], default=None
            Regularization parameter specifically for LASSO/RIDGE losses.
        gnorm_shape : Optional[float], default=None
            Shape parameter 's' for the generalized normal distribution.
        smoother : Literal["lowess", "ma", "global"], default="lowess"
            Smoother type for time series decomposition in initial state estimation.

            - "lowess": Uses LOWESS (Locally Weighted Scatterplot Smoothing) for both
              trend and seasonal pattern extraction. This is the default.
            - "ma": Uses simple moving average for both trend and seasonality.
            - "global": Uses lowess for trend and "ma" (moving average) for seasonality.
              Provides robust trend estimation while avoiding lowess instability for
              small seasonal samples.
        """
        # Start measuring the time of calculations
        self.start_time = time.time()

        # Store initialization parameters
        self.model = model
        self.lags = lags
        self.ar_order = ar_order
        self.i_order = i_order
        self.ma_order = ma_order
        self.arima_select = arima_select
        self.constant = constant
        self.regressors = regressors
        self.distribution = distribution
        self.loss = loss
        self.loss_horizon = loss_horizon
        self.outliers = outliers
        self.outliers_level = outliers_level
        self.ic = ic
        self.bounds = bounds
        self.occurrence = occurrence
        self.persistence = persistence
        self.phi = phi
        self.initial = initial
        self.n_iterations = n_iterations
        self.arma = arma
        self.verbose = verbose
        self.nlopt_initial = nlopt_initial
        self.nlopt_upper = nlopt_upper
        self.nlopt_lower = nlopt_lower
        self.nlopt_kargs = nlopt_kargs
        self.reg_lambda = reg_lambda
        self.gnorm_shape = gnorm_shape
        self.smoother = smoother

        # Store parameters that were moved from fit
        self.h = h
        self.holdout = holdout
        self.model_do = model_do
        self.fast = fast
        self.models_pool = models_pool
        # Handle 'lambda' from kwargs (since 'lambda' is a reserved word in Python)
        # Users can pass either lambda_param=0.5 or **{'lambda': 0.5}
        if "lambda" in kwargs:
            self.lambda_param = kwargs["lambda"]
        else:
            self.lambda_param = lambda_param

        # Handle 'print_level' from kwargs for convenience
        # Users can pass print_level=1 directly or via nlopt_kargs={'print_level': 1}
        if "print_level" in kwargs:
            if self.nlopt_kargs is None:
                self.nlopt_kargs = {}
            self.nlopt_kargs["print_level"] = kwargs["print_level"]

        self.frequency = frequency

        # Store profile parameters
        self.profiles_recent_provided = profiles_recent_provided
        self.profiles_recent_table = profiles_recent_table

    def fit(
        self,
        y: NDArray,
        X: Optional[NDArray] = None,
    ):
        """
        Fit the ADAM model to time series data.

        This method estimates the model parameters, selects the best model
        (if automatic selection is enabled), and prepares the model for
        forecasting. It implements the complete ADAM estimation pipeline:
        parameter checking, model architecture creation, state-space matrix
        construction, parameter optimization, and model preparation.

        **Estimation Process**:

        1. **Parameter Validation**: Check all inputs via ``parameters_checker()``
        2. **Model Selection** (if ``model_do="select"``): Use Branch & Bound algorithm
        3. **Model Architecture**: Define components and lags via ``architector()``
        4. **Matrix Creation**: Build state-space matrices via ``creator()``
        5. **Parameter Estimation**: Optimize using NLopt via ``estimator()``
        6. **Model Preparation**: Compute fitted values and final states
           via ``preparator()``

        After fitting, the model stores all results as attributes:

        - Estimated parameters (``persistence_level_``, ``phi_``, etc.)
        - State-space matrices (``adam_created``)
        - Fitted values and residuals (``prepared_model``)
        - Information criteria (``ic_selection``)
        - Model specification (``model_type_dict``)

        Parameters
        ----------
        y : array-like, shape (T,)
            Time series data to fit. Can be:

            - ``numpy.ndarray``: Shape (T,) for univariate time series
            - ``pandas.Series``: Will use index for time information if DatetimeIndex

            Data requirements:

            - **Minimum length**: Depends on model complexity. Rule of thumb:
              T ≥ 3 × (number of parameters)
            - **Multiplicative models**: Require strictly positive data (y > 0)
            - **Missing values**: Currently not supported in Python version
            - **Frequency**: Auto-detected from pandas Series with DatetimeIndex

        X : array-like, shape (T, n_features), optional
            External regressors (explanatory variables). If provided:

            - Must have same length as ``y`` (T observations)
            - Each column is a separate regressor
            - Used only if ``regressors`` parameter was set in ``__init__``
            - Can be adaptive (with persistence) or fixed coefficients

            Example::

                >>> X = np.column_stack([trend, holidays, temperature])
                >>> model.fit(y, X=X)

        Returns
        -------
        self : ADAM
            The fitted model instance with populated attributes:

            **Fitted Parameters** (scikit-learn style with trailing underscores):

            - ``persistence_level_``: α (level smoothing), range [0, 1]
            - ``persistence_trend_``: β (trend smoothing), range [0, α]
            - ``persistence_seasonal_``: γ (seasonal smoothing), list if multiple
            - ``phi_``: Damping parameter, range [0, 1]
            - ``arma_parameters_``: AR/MA coefficients (if ARIMA)
            - ``initial_states_``: Initial state values

            **Model Components**:

            - ``model_type_dict``: Complete model specification
            - ``adam_estimated``: Optimization results including parameter vector B
            - ``adam_created``: State-space matrices (mat_vt, mat_wt, mat_f, vec_g)
            - ``prepared_model``: Fitted values, residuals, final states
            - ``ic_selection``: AIC, AICc, BIC, or BICc value

            **Data and Configuration**:

            - ``general``: General configuration dictionary
            - ``observations_dict``: Observation information
            - ``lags_dict``: Lag structure
            - ``components_dict``: Component counts

        Raises
        ------
        ValueError
            If data validation fails:

            - y contains NaN values
            - y has insufficient length for model complexity
            - Multiplicative model specified for non-positive data
            - X and y have mismatched lengths

        RuntimeError
            If optimization fails to converge. Check:

            - Model specification is appropriate for data
            - Initial values are reasonable (try different ``initial`` method)
            - Bounds are not too restrictive

        Notes
        -----
        **Optimization Algorithm**:

        Uses NLopt's Nelder-Mead simplex algorithm by default (derivative-free, robust).
        For large models or difficult optimization, consider:

        - Changing ``initial`` method
        - Adjusting bounds (``bounds="usual"`` vs ``bounds="admissible"``)
        - Providing custom starting values via ``nlopt_initial`` parameter

        **Computational Complexity**:

        Fitting time depends on:

        - Sample size T: O(T) per function evaluation
        - Number of parameters k: ~40k function evaluations
        - Model selection: Estimates ~10-15 models for "ZZZ"

        Typical fitting times:

        - T=100, simple ETS: ~0.1 seconds
        - T=1000, ETS with 2 seasonalities: ~1-2 seconds
        - T=1000, automatic selection: ~10-20 seconds

        **Model Selection Details**:

        When ``model_do="select"``:

        1. Branch & Bound explores model space efficiently
        2. Each candidate model is fully estimated
        3. Best model selected based on ``ic`` criterion
        4. Selected model is re-estimated with full optimization

        To see which models were tested::

            >>> model.fit(y)
            >>> print(model.ic_selection)  # Dict of model names -> IC values

        **Holdout Validation**:

        If ``holdout=True`` in ``__init__``:

        - Last ``h`` observations are withheld for validation
        - Model estimated on first T-h observations
        - Can use holdout for out-of-sample accuracy assessment

        **Two-Stage Initialization**:

        When ``initial="two-stage"``:

        1. **Stage 1**: Quick backcasting estimation for initial states
        2. **Stage 2**: Refined optimization starting from stage 1 results

        Often provides better results than pure ``initial="optimal"`` for
        complex models.

        **Memory Usage**:

        - State matrix: O(n_components × (T + max_lag)) floats
        - Modest for typical models (~1-10 MB)
        - Multiple seasonality can increase memory usage

        See Also
        --------
        predict : Generate forecasts from fitted model
        predict_intervals : Generate prediction intervals
        estimator : Underlying estimation function
        parameters_checker : Input validation function

        Examples
        --------
        Basic fitting::

            >>> from smooth import ADAM
            >>> import numpy as np
            >>> y = np.array([
            ...     112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118
            ... ])
            >>> model = ADAM(model="ANN", lags=[1])
            >>> model.fit(y)
            >>> print(f"Alpha: {model.persistence_level_:.3f}")

        With external regressors::

            >>> X = np.random.randn(len(y), 2)
            >>> model = ADAM(model="AAN", regressors="use")
            >>> model.fit(y, X=X)
            >>> print(
            ...     "Regressor coefficients:"
            ...     f" {model.explanatory_dict['xreg_parameters']}"
            ... )

        Automatic model selection::

            >>> model = ADAM(model="ZZZ", lags=[12], ic="AICc")
            >>> model.fit(y)
            >>> print(f"Selected: {model.model_type_dict['model']}")
            >>> print(f"AICc: {model.ic_selection}")

        Access fitted values and residuals::

            >>> model.fit(y)
            >>> fitted = model.prepared_model['y_fitted']
            >>> residuals = model.prepared_model['residuals']
            >>> print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.3f}")

        Using pandas Series with datetime index::

            >>> import pandas as pd
            >>> dates = pd.date_range(
            ...     '2020-01-01', periods=len(y), freq='M'
            ... )
            >>> y_series = pd.Series(y, index=dates)
            >>> model.fit(y_series)
            >>> # Frequency auto-detected from index
        """
        # Store fit parameters - these are now set in __init__
        # No need to call _setup_parameters as those parameters are now
        # instance attributes

        # Store raw data for two-stage initialization
        # (needed to create fresh ADAM instance)
        self._y_data = y
        self._X_data = X

        # Use X if provided
        if X is not None:
            # Exogenous variables X are passed to _check_parameters
            # and handled downstream.
            pass

        # Check parameters and prepare data
        self._check_parameters(y)
        # Execute model estimation or selection based on model_do
        if self.model_type_dict["model_do"] == "estimate":
            self._execute_estimation()
        elif self.model_type_dict["model_do"] == "select":
            # get the best model
            self._execute_selection()
            # Execute estimation for the selected model
            # Note: estimator() handles two-stage initialization internally,
            # so all models in the pool use consistent initialization
            self._execute_estimation(estimation=True)

        elif self.model_type_dict["model_do"] == "combine":
            ...  # I need to implement this
            raise NotImplementedError("Combine is not implemented yet")
        else:
            model_do = self.model_type_dict["model_do"]
            warnings.warn(
                f"Unknown model_do value: {model_do}. "
                "Expected one of: 'estimate', 'select', 'combine'"
            )

        # Prepare final results and format output data
        self._prepare_results()

        # Store fitted parameters with trailing underscores
        self._set_fitted_attributes()

        return self

    def _set_fitted_attributes(self):
        """
        Set fitted parameters as attributes with trailing underscores.

        This follows scikit-learn conventions for fitted attributes.
        """
        # Set persistence parameters
        if hasattr(self, "persistence_results") and self.persistence_results:
            if "persistence_level" in self.persistence_results:
                self.persistence_level_ = self.persistence_results["persistence_level"]
            if "persistence_trend" in self.persistence_results:
                self.persistence_trend_ = self.persistence_results["persistence_trend"]
            if "persistence_seasonal" in self.persistence_results:
                self.persistence_seasonal_ = self.persistence_results[
                    "persistence_seasonal"
                ]
            if "persistence_xreg" in self.persistence_results:
                self.persistence_xreg_ = self.persistence_results["persistence_xreg"]

        # Set phi parameter - only if model is damped and phi was estimated or provided
        if hasattr(self, "phi_dict") and self.phi_dict:
            if self.phi_dict.get("phi_estimate", False) or self.model_type_dict.get(
                "damped", False
            ):
                self.phi_ = self.phi_dict.get("phi")
            else:
                self.phi_ = None
        else:
            self.phi_ = None

        # Set ARIMA parameters
        if hasattr(self, "arima_results") and self.arima_results:
            if "arma_parameters" in self.arima_results:
                self.arma_parameters_ = self.arima_results["arma_parameters"]

        # Set initial states
        if hasattr(self, "initials_results") and self.initials_results:
            if "initial_states" in self.initials_results:
                self.initial_states_ = self.initials_results["initial_states"]

        # Update self.model with the selected/estimated model name
        if hasattr(self, "model_type_dict") and self.model_type_dict:
            # Use best_model if available (from model selection), otherwise
            # construct from components
            if hasattr(self, "best_model") and self.best_model:
                ets_str = self.best_model
            else:
                # Construct from components (for fixed model specification)
                e = self.model_type_dict.get("error_type", "")
                t = self.model_type_dict.get("trend_type", "")
                s = self.model_type_dict.get("season_type", "")
                damped = self.model_type_dict.get("damped", False)
                if damped and t not in ["N", ""]:
                    t = t + "d"
                ets_str = e + t + s if (e or t or s) else self.model

            # Build the model name
            model_parts = []

            # Add ETS part if present
            is_ets = self.model_type_dict.get("ets_model", False)
            if is_ets:
                model_parts.append(f"ETS({ets_str})")

            # Add ARIMA part if present
            is_arima = self.model_type_dict.get("arima_model", False)
            if is_arima and hasattr(self, "arima_results") and self.arima_results:
                ar_orders = self.arima_results.get("ar_orders", [0])
                i_orders = self.arima_results.get("i_orders", [0])
                ma_orders = self.arima_results.get("ma_orders", [0])

                # Format ARIMA orders - sum across lags for simple display
                ar = sum(ar_orders) if ar_orders else 0
                i = sum(i_orders) if i_orders else 0
                ma = sum(ma_orders) if ma_orders else 0
                model_parts.append(f"ARIMA({ar},{i},{ma})")

            # Combine parts
            if model_parts:
                self.model = "+".join(model_parts)
            else:
                self.model = ets_str

    # =========================================================================
    # Extraction Properties - R-style accessors implemented as Python properties
    # =========================================================================

    def _check_is_fitted(self):
        """Check if model has been fitted."""
        if not hasattr(self, "prepared_model") or self.prepared_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

    @property
    def fitted(self) -> NDArray:
        """
        Return in-sample fitted values.

        Returns
        -------
        NDArray
            Array of fitted values for the in-sample period.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> fitted_values = model.fitted
        """
        self._check_is_fitted()
        return self.prepared_model["y_fitted"]

    @property
    def actuals(self) -> NDArray:
        """
        Return original in-sample data.

        Returns
        -------
        NDArray
            Array of original observations used for fitting.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> original_data = model.actuals
        """
        self._check_is_fitted()
        return np.array(self.observations_dict["y_in_sample"])

    @property
    def coef(self) -> NDArray:
        """
        Return estimated coefficients (parameter vector B).

        The parameter vector B contains all optimized parameters in order:
        1. ETS persistence parameters (α, β, γ)
        2. Damping parameter (φ)
        3. Initial states
        4. ARIMA parameters (AR, MA coefficients)
        5. Regression coefficients
        6. Constant term
        7. Distribution parameters

        Returns
        -------
        NDArray
            Parameter vector B with all estimated coefficients.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> coefficients = model.coef
        """
        self._check_is_fitted()
        return self.adam_estimated["B"]

    @property
    def residuals(self) -> NDArray:
        """
        Return model residuals (errors from fitting).

        For additive error models, residuals are y_t - fitted_t.
        For multiplicative error models, residuals are y_t / fitted_t - 1.

        Returns
        -------
        NDArray
            Array of residuals from the fitted model.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> errors = model.residuals
        >>> rmse = np.sqrt(np.mean(errors**2))
        """
        self._check_is_fitted()
        return self.prepared_model["residuals"]

    @property
    def nobs(self) -> int:
        """
        Return number of observations used for fitting.

        Returns
        -------
        int
            Number of in-sample observations.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> n = model.nobs
        """
        self._check_is_fitted()
        return len(self.observations_dict["y_in_sample"])

    @property
    def nparam(self) -> int:
        """
        Return number of estimated parameters.

        Returns
        -------
        int
            Number of parameters estimated during optimization.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> k = model.nparam
        """
        self._check_is_fitted()
        return self.adam_estimated["n_param_estimated"]

    @property
    def sigma(self) -> float:
        """
        Return scale/standard error estimate.

        This is the estimated scale parameter of the error distribution,
        which equals the standard deviation for normal errors.

        Returns
        -------
        float
            Scale parameter estimate.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> std_error = model.sigma
        """
        self._check_is_fitted()
        return self.prepared_model["scale"]

    @property
    def loglik(self) -> float:
        """
        Return log-likelihood of the fitted model.

        Returns
        -------
        float
            Log-likelihood value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> ll = model.loglik
        """
        self._check_is_fitted()
        return self.adam_estimated["log_lik_adam_value"]["value"]

    @property
    def aic(self) -> float:
        """
        Return Akaike Information Criterion.

        Returns
        -------
        float
            AIC value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> aic_val = model.aic
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import AIC

        log_lik = self.adam_estimated["log_lik_adam_value"]
        return AIC(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def aicc(self) -> float:
        """
        Return corrected Akaike Information Criterion.

        Returns
        -------
        float
            AICc value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> aicc_val = model.aicc
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import AICc

        log_lik = self.adam_estimated["log_lik_adam_value"]
        return AICc(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def bic(self) -> float:
        """
        Return Bayesian Information Criterion.

        Returns
        -------
        float
            BIC value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> bic_val = model.bic
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import BIC

        log_lik = self.adam_estimated["log_lik_adam_value"]
        return BIC(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def bicc(self) -> float:
        """
        Return corrected Bayesian Information Criterion.

        Returns
        -------
        float
            BICc value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> bicc_val = model.bicc
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import BICc

        log_lik = self.adam_estimated["log_lik_adam_value"]
        return BICc(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def error_type(self) -> str:
        """
        Return error type: 'A' (additive) or 'M' (multiplicative).

        Returns
        -------
        str
            'A' for additive errors, 'M' for multiplicative errors.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> err_type = model.error_type  # Returns 'A'
        """
        self._check_is_fitted()
        return self.model_type_dict["error_type"]

    @property
    def model_type(self) -> str:
        """
        Return ETS model type (e.g., 'AAN', 'AAA', 'MAdM').

        Returns
        -------
        str
            Three-letter ETS model code where:
            - First letter: Error type (A/M)
            - Second letter: Trend type (N/A/Ad/M/Md)
            - Third letter: Seasonal type (N/A/M)

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="ZZZ")  # Auto-selection
        >>> model.fit(y)
        >>> selected_type = model.model_type  # e.g., 'AAN'
        """
        self._check_is_fitted()
        model = self.model_type_dict.get("model", "")
        if "(" in model and ")" in model:
            return model[model.index("(") + 1 : model.index(")")]
        return model

    @property
    def orders(self) -> Dict[str, List[int]]:
        """
        Return ARIMA orders as dict with 'ar', 'i', 'ma' keys.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary with keys 'ar', 'i', 'ma' containing lists of orders
            for each lag. For pure ETS models, returns [0] for each.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="NNN", ar_order=1, i_order=1, ma_order=1)
        >>> model.fit(y)
        >>> arima_orders = model.orders
        >>> print(arima_orders)  # {'ar': [1], 'i': [1], 'ma': [1]}
        """
        self._check_is_fitted()
        ar = self.arima_results.get("ar_orders")
        i = self.arima_results.get("i_orders")
        ma = self.arima_results.get("ma_orders")
        return {
            "ar": ar if ar is not None else [0],
            "i": i if i is not None else [0],
            "ma": ma if ma is not None else [0],
        }

    @property
    def model_name(self) -> str:
        """
        Return full model name string.

        Returns the complete model specification string, e.g.,
        'ETS(AAN)' or 'ETS(AAA)+ARIMA(1,1,1)'.

        Returns
        -------
        str
            Full model name.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> name = model.model_name  # 'ETS(AAN)'
        """
        self._check_is_fitted()
        return self.model_type_dict.get("model", "")

    @property
    def lags_used(self) -> List[int]:
        """
        Return the vector of lags used in the model.

        Returns
        -------
        List[int]
            List of lag values (seasonal periods) used in the model.
            For example, [1, 12] for monthly data with annual seasonality.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=[1, 12])
        >>> model.fit(y)
        >>> model.lags_used  # [1, 12]
        """
        self._check_is_fitted()
        return list(self.lags_dict.get("lags", [1]))

    def predict(
        self,
        h: int,
        X: Optional[NDArray] = None,
        calculate_intervals: bool = True,
        interval_method: Optional[
            Literal["parametric", "simulation", "bootstrap"]
        ] = "parametric",
        level: Optional[Union[float, List[float]]] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        nsim: int = 10000,
    ) -> NDArray:
        """
        Generate point forecasts using the fitted ADAM model.

        If `calculate_intervals` is True, prediction intervals are also
        computed and stored in `self.forecast_results` but only point
        forecasts are returned by this method. Use `predict_intervals`
        to get the intervals directly.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps to forecast).
        X : Optional[NDArray], default=None
            Exogenous variables for the forecast period.
            Ensure that X covers the entire forecast horizon `h`.
        calculate_intervals : bool, default=True
            Whether to calculate prediction intervals along with point forecasts.
            The intervals are stored in `self.forecast_results`.
        interval_method : Optional[Literal['parametric', 'simulation', 'bootstrap']],
                default='parametric'
            Method to calculate prediction intervals:
            - 'parametric': Assumes a known distribution for errors.
            - 'simulation': Simulates future paths to derive intervals.
            - 'bootstrap': Uses bootstrapping techniques.
            This parameter is used if `calculate_intervals` is True.
        level : Optional[Union[float, List[float]]], default=0.95
            Confidence level(s) for prediction intervals (e.g., 0.95 for 95% interval,
            or [0.8, 0.95] for 80% and 95% intervals).
            Used if `calculate_intervals` is True.
        side : Literal['both', 'upper', 'lower'], default='both'
            Which side(s) of the intervals to compute:
            - 'both': Both lower and upper bounds.
            - 'lower': Only the lower bound.
            - 'upper': Only the upper bound.
            Used if `calculate_intervals` is True.
        nsim : int, default=10000
            Number of simulations to run for simulation-based intervals.
            Only used when `interval_method='simulation'`.

        Returns
        -------
        NDArray
            Point forecasts for the next `h` periods.

        Raises
        ------
        ValueError
            If the model has not been fitted yet or `h` is not set.
        """
        # Set forecast horizon
        if h is not None:
            self.h = h
            self.general["h"] = self.h
        else:
            if self.general["h"] is None:
                raise ValueError("Forecast horizon is not set.")

        # add interval methods
        self.calculate_intervals = calculate_intervals
        self.interval_method = interval_method
        self.level = level
        self.side = side
        self.general["nsim"] = nsim

        # Handle exogenous variables if provided
        if X is not None:
            # Exogenous variables X are handled by _prepare_prediction_data
            # and forecaster.
            pass

        # Validate prediction inputs and prepare data for forecasting
        self._validate_prediction_inputs()

        # Prepare data for prediction
        self._prepare_prediction_data()
        # Execute the prediction
        predictions = self._execute_prediction()

        # Return the point forecasts
        return predictions

    def predict_intervals(
        self,
        h: int,
        X: Optional[NDArray] = None,
        levels: List[float] = [0.8, 0.95],
        side: Literal["both", "upper", "lower"] = "both",
    ) -> Dict[str, NDArray]:
        """
        Generate prediction intervals using the fitted ADAM model.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps to forecast).
        X : Optional[NDArray], default=None
            Exogenous variables for the forecast period.
        levels : List[float], default=[0.8, 0.95]
            Confidence levels for prediction intervals.
        side : Literal["both", "upper", "lower"], default="both"
            Which side(s) of the intervals to return.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary containing point forecasts and prediction intervals.
            Keys include 'forecast', and 'lower'/'upper' depending on `side`.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        # Set forecast horizon
        self.h = h

        # Set confidence levels and side
        self.levels = levels
        self.interval_side = side

        # Handle exogenous variables if provided
        if X is not None:
            # Exogenous variables X are handled by _prepare_prediction_data
            # and forecaster.
            pass

        # Validate prediction inputs and prepare data for forecasting
        self._validate_prediction_inputs()

        # Prepare data for prediction
        self._prepare_prediction_data()

        # Execute the prediction
        self._execute_prediction()

        # Return the forecasts and intervals
        result = {"forecast": self.forecast_results["forecast"]}

        # Add intervals based on the requested side
        if side in ["both", "lower"]:
            result["lower"] = self.forecast_results["lower"]
        if side in ["both", "upper"]:
            result["upper"] = self.forecast_results["upper"]

        return result

    def _check_parameters(self, ts):
        """
        Check parameters using parameters_checker and store results.

        Parameters
        ----------
        ts : NDArray or pd.Series
            Time series data (numpy array or pandas Series).
        """
        # Convert ar_order, i_order, ma_order to orders format expected by
        # parameters_checker
        orders = None
        if any(param != 0 for param in [self.ar_order, self.i_order, self.ma_order]):
            orders = {
                "ar": self.ar_order,
                "i": self.i_order,
                "ma": self.ma_order,
                "select": self.arima_select,
            }

        (
            self.general,
            self.observations_dict,
            self.persistence_results,
            self.initials_results,
            self.arima_results,
            self.constant_dict,
            self.model_type_dict,
            self.components_dict,
            self.lags_dict,
            self.occurrence_dict,
            self.phi_dict,
            self.explanatory_dict,
            self.params_info,
        ) = parameters_checker(
            ts,
            model=self.model,
            lags=self.lags,
            orders=orders,
            constant=self.constant,
            outliers=self.outliers,
            level=self.outliers_level,
            persistence=self.persistence,
            phi=self.phi,
            initial=self.initial,
            n_iterations=self.n_iterations,
            distribution=self.distribution,
            loss=self.loss,
            h=self.h,
            holdout=self.holdout,
            occurrence=self.occurrence,
            ic=self.ic,
            bounds=self.bounds,
            silent=(self.verbose == 0),
            model_do=self.model_do,
            fast=self.fast,
            models_pool=self.models_pool,
            lambda_param=self.lambda_param,
            frequency=self.frequency,
        )

    def _handle_lasso_ridge_special_case(self):
        """
        Handle special case for LASSO/RIDGE with lambda=1.

        Sets appropriate parameter values. This is a special case where we use
        MSE to estimate initials only and disable other parameter estimation.
        """
        if self.general["loss"] in ["LASSO", "RIDGE"] and self.general["lambda"] == 1:
            if self.model_type_dict["ets_model"]:
                # Pre-set ETS parameters
                self.persistence_results["persistence_estimate"] = False
                self.persistence_results["persistence_level_estimate"] = False
                self.persistence_results["persistence_trend_estimate"] = False
                self.persistence_results["persistence_seasonal_estimate"] = [False]
                self.persistence_results["persistence_level"] = 0
                self.persistence_results["persistence_trend"] = 0
                self.persistence_results["persistence_seasonal"] = [0]
                # Phi
                self.phi_dict["phi_estimate"] = False
                self.phi_dict["phi"] = 1

            if self.model_type_dict["xreg_model"]:
                # ETSX parameters
                self.persistence_results["persistence_xreg_estimate"] = False
                self.persistence_results["persistence_xreg"] = 0

            if self.model_type_dict["arima_model"]:
                # Pre-set ARMA parameters
                self.arima_results["ar_estimate"] = [False]
                self.arima_results["ma_estimate"] = [False]
                self._preset_arima_parameters()

            self.general["lambda"] = 0

    def _preset_arima_parameters(self):
        """Set up ARIMA parameters for special cases where estimation is disabled."""
        arma_parameters = []
        for i, lag in enumerate(self.lags_dict["lags"]):
            if self.arima_results["ar_orders"][i] > 0:
                arma_parameters.extend([1] * self.arima_results["ar_orders"][i])
            if self.arima_results["ma_orders"][i] > 0:
                arma_parameters.extend([0] * self.arima_results["ma_orders"][i])
        self.arima_results["arma_parameters"] = arma_parameters

    def _execute_estimation(self, estimation=True):
        """
        Execute model estimation when model_do is 'estimate'.

        This handles parameter estimation and model creation.
        """
        # Handle special case for LASSO/RIDGE with lambda=1
        self._handle_lasso_ridge_special_case()

        # Estimate the model
        # Note: estimator() handles two-stage initialization internally
        if estimation:
            nlopt_params = self.nlopt_kargs if self.nlopt_kargs else {}
            self.adam_estimated = estimator(
                general_dict=self.general,
                model_type_dict=self.model_type_dict,
                lags_dict=self.lags_dict,
                observations_dict=self.observations_dict,
                arima_dict=self.arima_results,
                constant_dict=self.constant_dict,
                explanatory_dict=self.explanatory_dict,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided,
                persistence_dict=self.persistence_results,
                initials_dict=self.initials_results,
                occurrence_dict=self.occurrence_dict,
                phi_dict=self.phi_dict,
                components_dict=self.components_dict,
                multisteps=self.general.get("multisteps", False),
                smoother=self.smoother,
                **nlopt_params,
            )
            # Extract adam_cpp from estimation results
            self.adam_cpp = self.adam_estimated["adam_cpp"]

        # Build the model structure - architector() returns 6 values including
        # adam_cpp, but we already have adam_cpp from estimation
        (
            self.model_type_dict,
            self.components_dict,
            self.lags_dict,
            self.observations_dict,
            self.profile_dict,
            _,  # adam_cpp - already stored from estimation result
        ) = architector(
            model_type_dict=self.model_type_dict,
            lags_dict=self.lags_dict,
            observations_dict=self.observations_dict,
            arima_checked=self.arima_results,
            constants_checked=self.constant_dict,
            explanatory_checked=self.explanatory_dict,
            profiles_recent_table=self.profiles_recent_table,
            profiles_recent_provided=self.profiles_recent_provided,
        )
        # print(self.components_dict)
        # Create the model matrices
        self.adam_created = creator(
            model_type_dict=self.model_type_dict,
            lags_dict=self.lags_dict,
            profiles_dict=self.profile_dict,
            observations_dict=self.observations_dict,
            persistence_checked=self.persistence_results,
            initials_checked=self.initials_results,
            arima_checked=self.arima_results,
            constants_checked=self.constant_dict,
            phi_dict=self.phi_dict,
            components_dict=self.components_dict,
            explanatory_checked=self.explanatory_dict,
            smoother=self.smoother,
        )

        # Calculate information criterion
        if estimation:
            self.ic_selection = ic_function(
                self.general["ic"], self.adam_estimated["log_lik_adam_value"]
            )

        # Update parameters number
        self._update_parameters_number(self.adam_estimated["n_param_estimated"])

    def _update_parameters_number(self, n_param_estimated):
        """
        Update the parameters number in the general dictionary.

        Parameters
        ----------
        n_param_estimated : int
            Number of estimated parameters from optimization
        """
        # Store number of estimated parameters
        self.n_param_estimated = n_param_estimated

        # Update the n_param table
        if "n_param" in self.general:
            n_param = self.general["n_param"]
            # The n_param_estimated from optimizer is the total internal params
            # We need to update it based on what was actually estimated
            n_param.estimated["internal"] = n_param_estimated

            # Handle likelihood loss case - scale parameter is estimated
            if self.general["loss"] == "likelihood":
                n_param.estimated["scale"] = 1
            else:
                n_param.estimated["scale"] = 0

            # Update totals
            n_param.update_totals()

            # Store reference for easy access
            self.n_param = n_param

        # Legacy format for backward compatibility
        if "parameters_number" not in self.general:
            self.general["parameters_number"] = self.params_info.get(
                "parameters_number", [[0], [0]]
            )
        self.general["parameters_number"][0][0] = n_param_estimated

        # Handle likelihood loss case in legacy format
        if self.general["loss"] == "likelihood":
            if len(self.general["parameters_number"][0]) <= 3:
                self.general["parameters_number"][0].append(1)
            else:
                self.general["parameters_number"][0][3] = 1

        # Calculate row sums in legacy format
        if len(self.general["parameters_number"][0]) <= 4:
            self.general["parameters_number"][0].append(
                sum(self.general["parameters_number"][0][0:4])
            )
            self.general["parameters_number"][1].append(
                sum(self.general["parameters_number"][1][0:4])
            )
        else:
            self.general["parameters_number"][0][4] = sum(
                self.general["parameters_number"][0][0:4]
            )
            self.general["parameters_number"][1][4] = sum(
                self.general["parameters_number"][1][0:4]
            )

    def _execute_selection(self):
        """
        Execute model selection when model_do is 'select'.

        This handles model selection and creation of the selected model.
        """
        # Get nlopt parameters from nlopt_kargs if provided
        nlopt_params = self.nlopt_kargs if self.nlopt_kargs else {}
        # Run model selection
        self.adam_selected = selector(
            model_type_dict=self.model_type_dict,
            phi_dict=self.phi_dict,
            general_dict=self.general,
            lags_dict=self.lags_dict,
            observations_dict=self.observations_dict,
            arima_dict=self.arima_results,
            constant_dict=self.constant_dict,
            explanatory_dict=self.explanatory_dict,
            occurrence_dict=self.occurrence_dict,
            components_dict=self.components_dict,
            profiles_recent_table=self.profiles_recent_table,
            profiles_recent_provided=self.profiles_recent_provided,
            persistence_results=self.persistence_results,
            initials_results=self.initials_results,
            criterion=self.general["ic"],
            silent=self.verbose == 0,
            smoother=self.smoother,
            **nlopt_params,
        )
        # print(self.adam_selected)
        # print(self.adam_selected["ic_selection"])

        # Updates parametes with the selected model and updates adam_estimated
        self.select_best_model()

        # print(self.adam_selected["ic_selection"])
        # Process each selected model
        # The following commented-out loop and its associated helper method
        # calls (_update_model_from_selection, _create_matrices_for_selected_model,
        # _update_parameters_for_selected_model) appear to be placeholders
        # or remnants of a "combine" functionality that is not fully implemented
        # yet, as indicated by the NotImplementedError in the fit method for
        # model_do="combine". These will be kept for now as they might be
        # relevant for future development.
        # for i, result in enumerate(self.adam_selected["results"]):
        #     # Update model parameters with the selected model
        #     self._update_model_from_selection(i, result)
        #
        #     # Create matrices for this model
        #     self._create_matrices_for_selected_model(i)
        #
        #     # Update parameters number for this model
        #     self._update_parameters_for_selected_model(i, result)

    def select_best_model(self):
        """
        Select the best model based on information criteria and update model parameters.
        """
        self.ic_selection = self.adam_selected["ic_selection"]
        self.results = self.adam_selected["results"]
        # Find best model
        self.best_model = min(self.ic_selection.items(), key=lambda x: x[1])[0]
        self.best_id = next(
            i
            for i, result in enumerate(self.results)
            if result["model"] == self.best_model
        )
        # Update dictionaries with best model results
        self.model_type_dict = self.results[self.best_id]["model_type_dict"]
        self.phi_dict = self.results[self.best_id]["phi_dict"]
        self.adam_estimated = self.results[self.best_id]["adam_estimated"]
        self.adam_cpp = self.adam_estimated["adam_cpp"]

    def _update_model_from_selection(self, index, result):
        """
        Update model parameters with the selected model parameters.

        Parameters
        ----------
        index : int
            Index of the selected model in the results list from model selection.
        result : dict
            The dictionary containing all parameters and estimation results
            for the selected model.
        """
        # Update global dictionaries with the selected model info
        self.general.update(result["general"])
        self.model_type_dict.update(result["model_type_dict"])
        self.arima_results.update(result["arima_dict"])
        self.constant_dict.update(result["constant_dict"])
        self.persistence_results.update(result["persistence_dict"])
        self.initials_results.update(result["initials_dict"])
        self.phi_dict.update(result["phi_dict"])
        self.components_dict.update(result["components_dict"])
        self.lags_dict.update(result["lags_dict"])
        self.observations_dict.update(result["observations_dict"])
        self.profile_dict = result.get(
            "profile_dict",
            {
                "profiles_recent_provided": self.profiles_recent_provided,
                "profiles_recent_table": self.profiles_recent_table,
            },
        )

        # Store the estimated model and adam_cpp
        self.adam_estimated = result["adam_estimated"]
        self.adam_cpp = self.adam_estimated["adam_cpp"]

    def _create_matrices_for_selected_model(self, index):
        """
        Create matrices for a selected model. This is typically used when iterating
        through models in a selection process, particularly for a "combine" feature.

        Parameters
        ----------
        index : int
            Index of the selected model.
        """
        # Create matrices for this model
        self.adam_created = creator(
            model_type_dict=self.model_type_dict,
            lags_dict=self.lags_dict,
            profiles_dict=self.profile_dict,
            observations_dict=self.observations_dict,
            persistence_checked=self.persistence_results,
            initials_checked=self.initials_results,
            arima_checked=self.arima_results,
            constants_checked=self.constant_dict,
            phi_dict=self.phi_dict,
            components_dict=self.components_dict,
            explanatory_checked=self.explanatory_dict,
            smoother=self.smoother,
        )

        # Store created matrices
        self.adam_selected["results"][index]["adam_created"] = self.adam_created

    def _update_parameters_for_selected_model(self, index, result):
        """
        Update parameters number for a selected model.

        This is typically used when iterating through models in a selection
        process, particularly for a "combine" feature.

        Parameters
        ----------
        index : int
            Index of the selected model.
        result : dict
            Selected model result containing `adam_estimated` which has
            `n_param_estimated`.
        """
        # Update parameters number
        n_param_estimated = result["adam_estimated"]["n_param_estimated"]

        # Update n_param table if available
        if "n_param" in self.general and self.general["n_param"] is not None:
            n_param = self.general["n_param"]
            n_param.estimated["internal"] = n_param_estimated

            if self.general["loss"] == "likelihood":
                n_param.estimated["scale"] = 1
            else:
                n_param.estimated["scale"] = 0

            n_param.update_totals()
            self.n_param = n_param

        # Legacy format
        self.general["parameters_number"] = self.params_info.get(
            "parameters_number", [[0], [0]]
        )
        self.general["parameters_number"][0][0] = n_param_estimated

        # Handle likelihood loss case
        if self.general["loss"] == "likelihood":
            if len(self.general["parameters_number"][0]) <= 3:
                self.general["parameters_number"][0].append(1)
            else:
                self.general["parameters_number"][0][3] = 1

        # Calculate row sums
        if len(self.general["parameters_number"][0]) <= 4:
            self.general["parameters_number"][0].append(
                sum(self.general["parameters_number"][0][0:4])
            )
            self.general["parameters_number"][1].append(
                sum(self.general["parameters_number"][1][0:4])
            )
        else:
            self.general["parameters_number"][0][4] = sum(
                self.general["parameters_number"][0][0:4]
            )
            self.general["parameters_number"][1][4] = sum(
                self.general["parameters_number"][1][0:4]
            )

        # Store parameters number
        self.adam_selected["results"][index]["parameters_number"] = self.general[
            "parameters_number"
        ]

    def _prepare_results(self):
        """
        Prepare final results and format output data.

        This transforms data into appropriate formats and handles distribution
        selection.
        """
        # Transform data into appropriate classes
        self._format_time_series_data()

        # Handle distribution selection
        self._select_distribution()

        # Compute fitted values and residuals by calling preparator
        self._compute_fitted_values()

    def _format_time_series_data(self):
        """
        Format time series data into pandas Series with appropriate indexes.
        """
        if isinstance(self.observations_dict["y_in_sample"], np.ndarray):
            # Check if frequency is a valid pandas frequency (not just "1" string)
            freq = self.observations_dict.get("frequency", "1")
            try:
                # Try to use date_range if frequency looks valid
                if freq != "1" and isinstance(
                    self.observations_dict.get("y_start"), (pd.Timestamp, str)
                ):
                    self.y_in_sample = pd.Series(
                        self.observations_dict["y_in_sample"],
                        index=pd.date_range(
                            start=self.observations_dict["y_start"],
                            periods=len(self.observations_dict["y_in_sample"]),
                            freq=freq,
                        ),
                    )
                else:
                    # Use simple range index for non-datetime data
                    self.y_in_sample = pd.Series(self.observations_dict["y_in_sample"])
            except (ValueError, TypeError):
                # Fallback to simple range index if date_range fails
                self.y_in_sample = pd.Series(self.observations_dict["y_in_sample"])

            if self.general["holdout"]:
                try:
                    if freq != "1" and isinstance(
                        self.observations_dict.get("y_forecast_start"),
                        (pd.Timestamp, str),
                    ):
                        self.y_holdout = pd.Series(
                            self.observations_dict["y_holdout"],
                            index=pd.date_range(
                                start=self.observations_dict["y_forecast_start"],
                                periods=len(self.observations_dict["y_holdout"]),
                                freq=freq,
                            ),
                        )
                    else:
                        self.y_holdout = pd.Series(self.observations_dict["y_holdout"])
                except (ValueError, TypeError):
                    self.y_holdout = pd.Series(self.observations_dict["y_holdout"])
        else:
            self.y_in_sample = self.observations_dict["y_in_sample"].copy()
            if self.general["holdout"]:
                self.y_holdout = pd.Series(
                    self.observations_dict["y_holdout"],
                    index=self.observations_dict["y_forecast_index"],
                )

    def _select_distribution(self):
        """
        Select appropriate distribution based on model and loss function.

        Distribution mapping:
        - likelihood: dnorm (additive) or dgamma (multiplicative)
        - MAE-based losses: dlaplace
        - HAM-based losses: ds
        - MSE-based and other losses: dnorm
        """
        if self.general["distribution"] == "default":
            loss = self.general["loss"]
            if loss == "likelihood":
                if self.model_type_dict["error_type"] == "A":
                    self.general["distribution_new"] = "dnorm"
                elif self.model_type_dict["error_type"] == "M":
                    self.general["distribution_new"] = "dgamma"
            elif loss in [
                "MAE",
                "MAEh",
                "TMAE",
                "GTMAE",
                "MACE",
                "aTMAE",
                "aGTMAE",
                "aMACE",
            ]:
                self.general["distribution_new"] = "dlaplace"
            elif loss in [
                "HAM",
                "HAMh",
                "THAM",
                "GTHAM",
                "CHAM",
                "aTHAM",
                "aGTHAM",
                "aCHAM",
            ]:
                self.general["distribution_new"] = "ds"
            elif loss in [
                "MSE",
                "MSEh",
                "TMSE",
                "GTMSE",
                "MSCE",
                "GPL",
                "aMSEh",
                "aTMSE",
                "aGTMSE",
                "aMSCE",
                "aGPL",
                "LASSO",
                "RIDGE",
                "custom",
            ]:
                self.general["distribution_new"] = "dnorm"
            else:
                # Fallback to dnorm for any unrecognized loss
                self.general["distribution_new"] = "dnorm"
        else:
            self.general["distribution_new"] = self.general["distribution"]

    def _compute_fitted_values(self):
        """
        Compute fitted values and residuals after model estimation.

        This calls preparator() to run the fitter with final parameters
        and extract fitted values, residuals, and scale.
        """
        # Set a default h if not provided (needed for preparator)
        if self.h is None:
            if self.lags_dict and len(self.lags_dict["lags"]) > 0:
                self.general["h"] = max(self.lags_dict["lags"])
            else:
                self.general["h"] = 10

        # Call preparator to compute fitted values and residuals
        self.prepared_model = preparator(
            model_type_dict=self.model_type_dict,
            components_dict=self.components_dict,
            lags_dict=self.lags_dict,
            matrices_dict=self.adam_created,
            persistence_checked=self.persistence_results,
            initials_checked=self.initials_results,
            arima_checked=self.arima_results,
            explanatory_checked=self.explanatory_dict,
            phi_dict=self.phi_dict,
            constants_checked=self.constant_dict,
            observations_dict=self.observations_dict,
            occurrence_dict=self.occurrence_dict,
            general_dict=self.general,
            profiles_dict=self.profile_dict,
            adam_estimated=self.adam_estimated,
            adam_cpp=self.adam_cpp,
            bounds="usual",
            other=None,
        )

    def _validate_prediction_inputs(self):
        """
        Validate that the model is properly fitted before prediction.

        Raises
        ------
        ValueError
            If the model has not been fitted yet or is missing required components.
        """
        # Verify that the model has been fitted
        if not hasattr(self, "model_type_dict"):
            raise ValueError("Model must be fitted before prediction.")

        # Check if we have the necessary components based on the model type
        if self.model_type_dict["model_do"] == "estimate" and not hasattr(
            self, "adam_estimated"
        ):
            raise ValueError("Model estimation results not found.")
        elif self.model_type_dict["model_do"] == "select" and not hasattr(
            self, "adam_selected"
        ):
            raise ValueError("Model selection results not found.")

    def _prepare_prediction_data(self):
        """
        Prepare data for prediction by setting up necessary matrices and parameters.
        """
        # If h wasn't provided, use default h
        if self.h is None:
            if self.lags_dict and len(self.lags_dict["lags"]) > 0:
                self.h = max(self.lags_dict["lags"])

            else:
                self.h = 10
                self.general["h"] = self.h

        # Prepare necessary data for forecasting
        self.prepared_model = preparator(
            # Model info
            model_type_dict=self.model_type_dict,
            # Components info
            components_dict=self.components_dict,
            # Lags info
            lags_dict=self.lags_dict,
            # Matrices from creator
            matrices_dict=self.adam_created,
            # Parameter dictionaries
            persistence_checked=self.persistence_results,
            initials_checked=self.initials_results,
            arima_checked=self.arima_results,
            explanatory_checked=self.explanatory_dict,
            phi_dict=self.phi_dict,
            constants_checked=self.constant_dict,
            # Other parameters
            observations_dict=self.observations_dict,
            occurrence_dict=self.occurrence_dict,
            general_dict=self.general,
            profiles_dict=self.profile_dict,
            # The parameter vector
            adam_estimated=self.adam_estimated,
            # adamCore C++ object
            adam_cpp=self.adam_cpp,
            # Optional parameters
            bounds="usual",
            other=None,
        )

    def _execute_prediction(self):
        """
        Execute the forecasting process based on the prepared data.

        This method calls the core `forecaster` function with all necessary
        prepared data structures and parameters to generate point forecasts
        and, if requested, prediction intervals.

        Returns
        -------
        dict
            A dictionary containing the forecast results, including point forecasts
            and potentially prediction intervals (e.g., 'forecast', 'lower', 'upper').
            This dictionary is also stored in `self.forecast_results`.
        """
        # Generate forecasts using the forecaster function
        self.forecast_results = forecaster(
            model_prepared=self.prepared_model,
            observations_dict=self.observations_dict,
            general_dict=self.general,
            occurrence_dict=self.occurrence_dict,
            lags_dict=self.lags_dict,
            model_type_dict=self.model_type_dict,
            explanatory_checked=self.explanatory_dict,
            components_dict=self.components_dict,
            constants_checked=self.constant_dict,
            params_info=self.params_info,
            adam_cpp=self.adam_cpp,
            calculate_intervals=self.calculate_intervals,
            interval_method=self.interval_method,
            level=self.level,
            side=self.side,
        )
        return self.forecast_results

    def _format_prediction_results(self):
        """
        Format the prediction results into a more user-friendly structure.
        Currently, this method primarily adds the elapsed time to the forecast results.

        Note: This method is defined but not explicitly called within the ADAM class's
        current public interface (fit, predict, predict_intervals).
        It might be intended for internal use or future extensions.

        Returns
        -------
        dict
            Formatted prediction results including point forecasts and intervals.
        """
        # Calculate and include elapsed time before returning
        self.forecast_results["elapsed_time"] = time.time() - self.start_time

        return self.forecast_results

    def __str__(self) -> str:
        """
        Return a formatted string representation of the fitted model.

        Returns a comprehensive summary including model type, parameters,
        information criteria, and forecast errors (if holdout was used).

        Returns
        -------
        str
            Formatted model summary
        """
        from smooth.adam_general.core.utils.printing import model_summary

        # Check if model has been fitted
        if not hasattr(self, "model_type_dict"):
            return f"ADAM(model={self.model}) - not fitted"

        return model_summary(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the ADAM model.

        Returns
        -------
        str
            Brief model representation
        """
        if hasattr(self, "model_type_dict") and self.model_type_dict:
            model_str = self.model_type_dict.get("model", self.model)
            if self.model_type_dict.get("ets_model", False):
                return f"ADAM(ETS({model_str}), fitted=True)"
            return f"ADAM({model_str}, fitted=True)"
        return f"ADAM(model={self.model}, fitted=False)"

    def summary(self, digits: int = 4) -> str:
        """
        Generate a formatted summary of the fitted model.

        Parameters
        ----------
        digits : int, default=4
            Number of decimal places for numeric output

        Returns
        -------
        str
            Formatted model summary

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> print(model.summary())
        """
        from smooth.adam_general.core.utils.printing import model_summary

        if not hasattr(self, "model_type_dict"):
            return "Model has not been fitted yet. Call fit() first."

        return model_summary(self, digits=digits)
