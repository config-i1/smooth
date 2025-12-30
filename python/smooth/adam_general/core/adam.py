import time
import warnings
from typing import Union, List, Optional, Literal, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general.core.checker import parameters_checker
from smooth.adam_general.core.estimator import estimator, selector, _process_initial_values
from smooth.adam_general.core.creator import creator, initialiser, architector, filler
from smooth.adam_general.core.utils.ic import ic_function
from smooth.adam_general.core.forecaster import preparator, forecaster

from smooth.adam_general._adam_general import adam_fitter, adam_forecaster


# Type hint groups
DISTRIBUTION_OPTIONS = Literal[
    "default", "dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dinvgauss", "dgamma"
]

LOSS_OPTIONS = Literal[
    "likelihood",
    "MSE",
    "MAE",
    "HAM",
    "LASSO",
    "RIDGE",
    "MSEh",
    "TMSE",
    "GTMSE",
    "MSCE",
]

OCCURRENCE_OPTIONS = Literal[
    "none", "auto", "fixed", "general", "odds-ratio", "inverse-odds-ratio", "direct"
]

INITIAL_OPTIONS = Optional[
    Union[
        Dict[str, Any],
        Literal["optimal", "backcasting", "complete", "provided"],
        Tuple[str, ...],
    ]
]


class ADAM:
    """
    ADAM: Augmented Dynamic Adaptive Model for Time Series Forecasting.

    ADAM is an advanced state-space modeling framework that combines **ETS** (Error, Trend,
    Seasonal) and **ARIMA** components into a unified Single Source of Error (SSOE) model.
    It provides a flexible, data-driven approach to time series forecasting with automatic
    model selection, parameter estimation, and prediction intervals.

    **Mathematical Form**:

    The ADAM model is specified in state-space form as:

    .. math::

        y_t &= o_t(w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l})\\epsilon_t)

        v_t &= f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_t)\\epsilon_t

    where:

    - :math:`y_t`: Observed value at time t
    - :math:`o_t`: Occurrence indicator (Bernoulli variable for intermittent data, 1 otherwise)
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
    2. **Multiple Seasonality**: Supports multiple seasonal periods (e.g., daily + weekly)
    3. **Automatic Selection**: Branch & Bound algorithm for efficient model selection
    4. **Flexible Distributions**: Normal, Laplace, Gamma, Log-Normal, and more
    5. **Intermittent Demand**: Built-in occurrence models for sparse data
    6. **External Regressors**: Include covariates with adaptive or fixed coefficients
    7. **Scikit-learn Compatible**: Familiar `.fit()` and `.predict()` API

    **Model Specification**:

    Models are specified using a string notation:

    - **ETS Models**: "ETS" where E=Error, T=Trend, S=Seasonal

      * E (Error): "A" (Additive), "M" (Multiplicative)
      * T (Trend): "N" (None), "A" (Additive), "Ad" (Additive Damped), "M" (Multiplicative), "Md" (Multiplicative Damped)
      * S (Seasonal): "N" (None), "A" (Additive), "M" (Multiplicative)

      Examples: "ANN" (Simple Exponential Smoothing), "AAN" (Holt's Linear), "AAA" (Holt-Winters Additive)

    - **Automatic Selection**:

      * "ZZZ": Select best model using Branch & Bound
      * "XXX": Select only additive components
      * "YYY": Select only multiplicative components
      * "ZXZ": Auto-select error and seasonal, additive trend only (**default**, safer)
      * "FFF": Full search across all 30 ETS model types

    - **ARIMA Models**: Specified via `ar_order`, `i_order`, `ma_order` parameters

      * Supports seasonal ARIMA: SARIMA(p,d,q)(P,D,Q)m
      * Multiple seasonality: e.g., hourly data with daily (24) and weekly (168) patterns

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

    **Initialization Methods**:

    - ``initial="optimal"``: Optimize all initial states (default)
    - ``initial="backcasting"``: Use backcasting to initialize states
    - ``initial="two-stage"``: Backcast then optimize
    - ``initial="complete"``: Pure backcasting without optimization
    - ``initial={"level": 100, ...}``: Provide custom initial states

    **Workflow Example**:

    .. code-block:: python

        from smooth.adam_general.core.adam import ADAM
        import numpy as np

        # Generate sample data
        y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 3)

        # Automatic model selection
        model = ADAM(model="ZZZ", lags=[12], ic="AICc")
        model.fit(y)

        # Generate 12-step ahead forecasts with intervals
        forecasts = model.predict(h=12, calculate_intervals=True, level=0.95)
        print(forecasts)

        # Access fitted parameters
        print(f"Model: {model.model_type_dict['model']}")
        print(f"Alpha: {model.persistence_level_:.3f}")
        print(f"AICc: {model.ic_selection}")

    **Attributes (After Fitting)**:

    The model stores fitted results as attributes with trailing underscores (scikit-learn convention):

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

    - **Small Data** (T < 100): Use "backcasting" initialization, it's faster
    - **Large Data** (T > 1000): "optimal" initialization works well
    - **Multiple Seasonality**: Can be slow; consider simpler models first
    - **Model Selection**: "ZZZ" with Branch & Bound is much faster than "FFF" exhaustive search

    **Common Use Cases**:

    1. **Automatic Forecasting**: ``ADAM(model="ZXZ", lags=[12])`` - Let the model choose
    2. **Intermittent Demand**: ``ADAM(model="ANN", occurrence="auto")`` - For sparse data
    3. **External Regressors**: ``ADAM(model="AAN").fit(y, X=regressors)`` - Include covariates
    4. **Multiple Seasonality**: ``ADAM(model="AAA", lags=[24, 168])`` - Hourly data with daily/weekly patterns
    5. **ARIMA**: ``ADAM(model="NNN", ar_order=1, i_order=1, ma_order=1)`` - Pure ARIMA(1,1,1)
    6. **Custom Model**: ``ADAM(model="MAM", persistence={"alpha": 0.3})`` - Fix some parameters

    **Comparison to R's smooth::adam**:

    This Python implementation is a direct translation of the R smooth package's ``adam()`` function,
    maintaining mathematical equivalence while adapting to scikit-learn conventions:

    - R: ``adam(data, model="ZZZ", h=10)`` → Python: ``ADAM(model="ZZZ").fit(y).predict(h=10)``
    - R: ``persistence=list(alpha=0.3)`` → Python: ``persistence={"alpha": 0.3}``
    - R: ``orders=list(ar=c(1,1))`` → Python: ``ar_order=[1, 1]``

    **References**:

    - Svetunkov, I. (2023). "Smooth forecasting in R". https://openforecast.org/adam/
    - Hyndman, R.J., et al. (2008). "Forecasting with Exponential Smoothing"
    - Svetunkov, I. & Boylan, J.E. (2017). "State-space ARIMA for supply-chain forecasting"

    See Also
    --------
    adam.fit : Fit the ADAM model to data
    adam.predict : Generate point forecasts
    adam.predict_intervals : Generate prediction intervals
    selector : Automatic model selection function
    estimator : Parameter estimation function
    forecaster : Forecasting function

    Examples
    --------
    Simple exponential smoothing::

        >>> from smooth.adam_general.core.adam import ADAM
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
        initial: INITIAL_OPTIONS = None,
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
    ) -> None:
        """
        Initialize the ADAM model with specified parameters.

        Parameters
        ----------
        model : Union[str, List[str]], default="ZXZ"
            Model specification string (e.g., "ANN" for ETS) or list of model strings.
        lags : Optional[NDArray], default=None
            List of seasonal periods.
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
            Error distribution. If None, it is selected automatically based on the loss function.
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
        phi : Optional[float], default=None
            Fixed damping parameter for damped trend models. If None, estimated if applicable.
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
            Additional keyword arguments for the NLopt optimizer.
        reg_lambda : Optional[float], default=None
            Regularization parameter specifically for LASSO/RIDGE losses.
        gnorm_shape : Optional[float], default=None
            Shape parameter 's' for the generalized normal distribution.
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

        # Store parameters that were moved from fit
        self.h = h
        self.holdout = holdout
        self.model_do = model_do
        self.fast = fast
        self.models_pool = models_pool
        self.lambda_param = lambda_param
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

        This method estimates the model parameters, selects the best model (if automatic selection
        is enabled), and prepares the model for forecasting. It implements the complete ADAM
        estimation pipeline: parameter checking, model architecture creation, state-space matrix
        construction, parameter optimization, and model preparation.

        **Estimation Process**:

        1. **Parameter Validation**: Check all inputs via ``parameters_checker()``
        2. **Model Selection** (if ``model_do="select"``): Use Branch & Bound algorithm
        3. **Model Architecture**: Define components and lags via ``architector()``
        4. **Matrix Creation**: Build state-space matrices via ``creator()``
        5. **Parameter Estimation**: Optimize using NLopt via ``estimator()``
        6. **Model Preparation**: Compute fitted values and final states via ``preparator()``

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

            - **Minimum length**: Depends on model complexity. Rule of thumb: T ≥ 3 × (number of parameters)
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

        Often provides better results than pure ``initial="optimal"`` for complex models.

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

            >>> from smooth.adam_general.core.adam import ADAM
            >>> import numpy as np
            >>> y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118])
            >>> model = ADAM(model="ANN", lags=[1])
            >>> model.fit(y)
            >>> print(f"Alpha: {model.persistence_level_:.3f}")

        With external regressors::

            >>> X = np.random.randn(len(y), 2)
            >>> model = ADAM(model="AAN", regressors="use")
            >>> model.fit(y, X=X)
            >>> print(f"Regressor coefficients: {model.explanatory_dict['xreg_parameters']}")

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
            >>> dates = pd.date_range('2020-01-01', periods=len(y), freq='M')
            >>> y_series = pd.Series(y, index=dates)
            >>> model.fit(y_series)
            >>> # Frequency auto-detected from index
        """
        # Store fit parameters - these are now set in __init__
        # No need to call _setup_parameters as those parameters are now instance attributes

        # Store raw data for two-stage initialization (needed to create fresh ADAM instance)
        self._y_data = y
        self._X_data = X

        # Use X if provided
        if X is not None:
            # Exogenous variables X are passed to _check_parameters and handled downstream.
            pass

        # Check parameters and prepare data
        self._check_parameters(y)
        # Execute model estimation or selection based on model_do
        if self.model_type_dict["model_do"] == "estimate":
            self._execute_estimation()
        elif self.model_type_dict["model_do"] == "select":
            # get the best model
            self._execute_selection()
            # Execute estimation for the selected model with calling the estimator
            self._execute_estimation(estimation=False)

        elif self.model_type_dict["model_do"] == "combine":
            ... # I need to implement this
            raise NotImplementedError("Combine is not implemented yet")
        else:
            warnings.warn(
                f"Unknown model_do value: {self.model_type_dict['model_do']}. Expected one of: 'estimate', 'select', 'combine'"
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

        # Set phi parameter
        if hasattr(self, "phi_dict") and self.phi_dict and "phi" in self.phi_dict:
            self.phi_ = self.phi_dict["phi"]

        # Set ARIMA parameters
        if hasattr(self, "arima_results") and self.arima_results:
            if "arma_parameters" in self.arima_results:
                self.arma_parameters_ = self.arima_results["arma_parameters"]

        # Set initial states
        if hasattr(self, "initials_results") and self.initials_results:
            if "initial_states" in self.initials_results:
                self.initial_states_ = self.initials_results["initial_states"]

    def predict(
        self,
        h: int,
        X: Optional[NDArray] = None,
        calculate_intervals: bool = True,
        interval_method: Optional[Literal['parametric', 'simulation', 'bootstrap']] = 'parametric',
        level: Optional[Union[float, List[float]]] = 0.95,
        side: Literal['both', 'upper', 'lower'] = 'both',
        nsim: int = 10000,
    ) -> NDArray:
        """
        Generate point forecasts using the fitted ADAM model.

        If `calculate_intervals` is True, prediction intervals are also computed
        and stored in `self.forecast_results` but only point forecasts are returned by this method.
        Use `predict_intervals` to get the intervals directly.

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
        interval_method : Optional[Literal['parametric', 'simulation', 'bootstrap']], default='parametric'
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
            # Exogenous variables X are handled by _prepare_prediction_data and forecaster.
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
            # Exogenous variables X are handled by _prepare_prediction_data and forecaster.
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
        # Convert ar_order, i_order, ma_order to orders format expected by parameters_checker
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
        Handle special case for LASSO/RIDGE with lambda=1 by setting appropriate parameter values.

        This is a special case where we use MSE to estimate initials only and disable other parameter estimation.
        """
        lambda_original = self.general["lambda"]
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

    def _run_two_stage_initialization(self):
        """
        Run two-stage initialization matching R's recursive approach (adam.R:2625-2767).

        Stage 1: Create a fresh ADAM instance with initial="complete" and fast=True,
                 which runs the ENTIRE pipeline (parameters_checker -> architector ->
                 creator -> estimator) with fresh data structures.
        Stage 2: Use Stage 1 results (persistence + backcasted initial states) as
                 starting values for optimization with initial="optimal".

        This matches R's behavior where adam() is called recursively with
        clNew$initial <- "complete" (line 2636) and clNew$fast <- TRUE (line 2648).
        """
        import os
        DEBUG_TWOSTAGE = os.environ.get('DEBUG_TWOSTAGE', 'False').lower() == 'true'

        if DEBUG_TWOSTAGE:
            print("\n[PYTHON TWOSTAGE DEBUG] Starting two-stage initialization")
            print(f"  Creating fresh ADAM instance with initial='complete', fast=True")

        # =================================================================
        # Stage 1: Create fresh ADAM instance with initial="complete"
        # This matches R's recursive call: clNew$initial <- "complete" (adam.R:2636)
        # =================================================================

        # Build orders dict if ARIMA is used
        orders = None
        if any(param != 0 for param in [self.ar_order, self.i_order, self.ma_order]):
            orders = {
                "ar": self.ar_order,
                "i": self.i_order,
                "ma": self.ma_order,
                "select": self.arima_select,
            }

        stage1_model = ADAM(
            model=self.model,
            lags=self.lags,
            ar_order=self.ar_order,
            i_order=self.i_order,
            ma_order=self.ma_order,
            arima_select=self.arima_select,
            constant=self.constant,
            regressors=self.regressors,
            distribution=self.distribution,
            loss=self.loss,
            loss_horizon=self.loss_horizon,
            outliers=self.outliers,
            outliers_level=self.outliers_level,
            ic=self.ic,
            bounds=self.bounds,
            occurrence=self.occurrence,
            persistence=self.persistence,
            phi=self.phi,
            initial="complete",  # KEY: Use "complete" for backcasting
            n_iterations=self.n_iterations,
            arma=self.arma,
            verbose=0,  # Silent for Stage 1
            h=self.h,
            holdout=self.holdout,
            model_do="estimate",
            fast=True,  # KEY: Match R's clNew$fast <- TRUE (adam.R:2648)
            models_pool=self.models_pool,
            lambda_param=self.lambda_param,
            frequency=self.frequency,
            profiles_recent_provided=self.profiles_recent_provided,
            profiles_recent_table=self.profiles_recent_table,
            nlopt_initial=self.nlopt_initial,
            nlopt_upper=self.nlopt_upper,
            nlopt_lower=self.nlopt_lower,
            nlopt_kargs=self.nlopt_kargs,
            reg_lambda=self.reg_lambda,
            gnorm_shape=self.gnorm_shape,
        )

        # Fit Stage 1 model to the same data
        stage1_model.fit(self._y_data, X=self._X_data)

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] Stage 1 (complete) results:")
            print(f"  B_stage1: {stage1_model.adam_estimated['B']}")
            print(f"  Stage 1 model type: {stage1_model.model_type_dict.get('model', 'N/A')}")

        # =================================================================
        # Extract results from Stage 1
        # Matches R's adam.R:2673-2733:
        # 1. Calculate nParametersBack (persistence + phi + ARIMA params only)
        # 2. Copy ONLY B[1:nParametersBack] from Stage 1
        # 3. Place new initial states at positions nParametersBack+1:end
        # =================================================================

        B_stage1 = stage1_model.adam_estimated["B"]

        # Calculate nParametersBack - matches R's adam.R:2673-2677
        # This is the number of persistence/xreg_persistence/phi/ARIMA params (NOT initial states)
        persistence_estimate_vector = [
            self.persistence_results.get('persistence_level_estimate', False),
            self.model_type_dict.get("model_is_trendy", False) and self.persistence_results.get('persistence_trend_estimate', False),
            self.model_type_dict.get("model_is_seasonal", False) and any(self.persistence_results.get('persistence_seasonal_estimate', [False]))
        ]
        n_persistence = sum(persistence_estimate_vector)
        n_xreg_persistence = (
            self.explanatory_dict.get('xreg_model', False) *
            self.persistence_results.get('persistence_xreg_estimate', False) *
            max(self.explanatory_dict.get('xreg_parameters_persistence', [0]) or [0])
        )
        n_phi = 1 if self.phi_dict.get('phi_estimate', False) else 0
        n_ar = sum(self.arima_results.get('ar_orders', []) or []) if self.arima_results.get('ar_estimate', False) else 0
        n_ma = sum(self.arima_results.get('ma_orders', []) or []) if self.arima_results.get('ma_estimate', False) else 0
        n_params_back = n_persistence + n_xreg_persistence + n_phi + n_ar + n_ma

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] nParametersBack calculation:")
            print(f"  n_persistence: {n_persistence}")
            print(f"  n_xreg_persistence: {n_xreg_persistence}")
            print(f"  n_phi: {n_phi}")
            print(f"  n_ar: {n_ar}, n_ma: {n_ma}")
            print(f"  n_params_back: {n_params_back}")
            print(f"  B_stage1 length: {len(B_stage1)}")

        # Extract initial states from Stage 1's fitted model
        # R uses adamBack$initial which contains the backcasted states
        initial_value = self._extract_stage1_initials(stage1_model)

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] Extracted initial_value dict:")
            for key, val in initial_value.items():
                if isinstance(val, np.ndarray):
                    print(f"  {key}: shape={val.shape}, values={val}")
                elif isinstance(val, list):
                    print(f"  {key}: list with {len(val)} elements")
                    for i, v in enumerate(val):
                        if isinstance(v, np.ndarray):
                            print(f"    [{i}]: shape={v.shape}, values={v}")
                        else:
                            print(f"    [{i}]: {v}")
                else:
                    print(f"  {key}: {val}")

        # Convert initial_value dict to list (matching R's unlist() behavior)
        initial_states = self._build_initial_states_list(initial_value)

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] initial_states list:")
            print(f"  Length: {len(initial_states)}")
            print(f"  Values: {initial_states}")

        # Build B_initial matching R's structure (adam.R:2729):
        # Take ONLY the first n_params_back elements from Stage 1's B
        # Then concatenate with new initial states
        # This matches: B[1:nParametersBack] <- adamBack$B[1:nParametersBack]
        #               B[nParametersBack + 1:length(initials)] <- initialsUnlisted
        params_from_stage1 = B_stage1[:n_params_back]
        B_initial = np.concatenate([params_from_stage1, np.array(initial_states)])

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] Final B_initial (before stage 2):")
            print(f"  Length: {len(B_initial)}")
            print(f"  params_from_stage1 length: {len(params_from_stage1)}")
            print(f"  initial_states length: {len(initial_states)}")
            print(f"  B_initial: {B_initial}")
            print(f"  B_initial breakdown:")
            print(f"    B[0:{n_params_back}] (persistence/phi/ARIMA from Stage1): {B_initial[:n_params_back]}")
            print(f"    B[{n_params_back}:] (new initial states): {B_initial[n_params_back:]}")

        # =================================================================
        # Stage 2: Run optimization with initial="optimal" using B_initial
        # =================================================================

        # Save and set initial type to "optimal" for Stage 2
        original_initial_type = self.initials_results["initial_type"]
        self.initials_results["initial_type"] = "optimal"

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] Starting Stage 2 with initial='optimal'")
            print(f"  Using B_initial as starting values")

        # Run estimation for Stage 2, passing B_initial as starting values
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
            B_initial=B_initial
        )

        # Restore initial type
        self.initials_results["initial_type"] = original_initial_type

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] Stage 2 complete")
            print(f"  Final B: {self.adam_estimated['B']}")

    def _extract_stage1_initials(self, stage1_model):
        """
        Extract initial values from Stage 1 model matching R's adamBack$initial.

        This calls predict() on Stage 1 to trigger preparator(), which computes
        the 'initial' field using _process_initial_values(). This field contains
        the backcasted initial states extracted from profiles_recent_table.

        R renormalizes seasonal components in the two-stage-specific code (adam.R:2687-2715):
        - Renormalizes (subtract mean for 'A', divide by geometric mean for 'M')
        - Truncates last element (it's redundant since sum=0 or product=1)

        Parameters
        ----------
        stage1_model : ADAM
            The fitted Stage 1 ADAM model with initial="complete"

        Returns
        -------
        dict
            Dictionary with keys like 'level', 'trend', 'seasonal', etc.
            containing the extracted and renormalized initial states.
        """
        import os
        import copy
        DEBUG_TWOSTAGE = os.environ.get('DEBUG_TWOSTAGE', 'False').lower() == 'true'
        DEBUG_MAM = os.environ.get('DEBUG_MAM', 'False').lower() == 'true'

        # Call predict() to trigger preparator() which computes the 'initial' field
        # preparator() calls _process_initial_values() which extracts from profiles_recent_table
        stage1_model.predict(h=1)

        # Now prepared_model is available with the 'initial' field
        if not hasattr(stage1_model, 'prepared_model') or stage1_model.prepared_model is None:
            raise ValueError("Stage 1 model prepared_model not available after predict()")

        # Get initial values from prepared_model
        # This is the equivalent of R's adamBack$initial (adam.R:2722)
        initial_value = copy.deepcopy(stage1_model.prepared_model['initial'])

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] Stage 1 extraction (before renormalization):")
            print(f"  Using stage1_model.prepared_model['initial'] (matches R's adamBack$initial)")
            for key, val in initial_value.items():
                if hasattr(val, 'shape'):
                    print(f"  {key}: shape={val.shape}, values={val}")
                elif isinstance(val, list):
                    print(f"  {key}: list with {len(val)} elements")
                else:
                    print(f"  {key}: {val}")

        # Renormalize seasonal initials and truncate to lag-1 elements
        # This matches R's adam.R:2687-2715
        if 'seasonal' in initial_value and stage1_model.model_type_dict.get("model_is_seasonal"):
            season_type = stage1_model.model_type_dict.get("season_type", "A")
            seasonal = initial_value['seasonal']

            if DEBUG_TWOSTAGE:
                print(f"\n[PYTHON TWOSTAGE DEBUG] Seasonal renormalization:")
                print(f"  season_type: {season_type}")
                print(f"  seasonal before: {seasonal}")

            # Handle both single and multiple seasonalities
            if isinstance(seasonal, list):
                # Multiple seasonalities
                for i, s in enumerate(seasonal):
                    if isinstance(s, np.ndarray) and len(s) > 0:
                        # Renormalize
                        if season_type == "A":
                            s = s - np.mean(s)
                        elif season_type == "M":
                            # Safety: Replace non-positive values with 1e-6
                            s = np.where(s <= 0, 1e-6, s)
                            # Use prod(x)^(1/n) to match R's geometric mean calculation exactly
                            # R: prod(adamBack$initial$seasonal[[i]])^{1/length(...)}
                            geo_mean = np.prod(s) ** (1.0 / len(s))
                            if DEBUG_MAM:
                                print(f"\n[DEBUG_MAM] Two-stage seasonal normalization (multi, {i}):")
                                print(f"  Raw seasonal (after safety check): {s}")
                                print(f"  Geometric mean (prod^(1/n)): {geo_mean}")
                            if geo_mean > 0 and np.isfinite(geo_mean):
                                s = s / geo_mean
                            if DEBUG_MAM:
                                print(f"  After normalization: {s}")
                                print(f"  After truncation (:-1): {s[:-1]}")
                        # Truncate to lag-1 elements
                        seasonal[i] = s[:-1]
            elif isinstance(seasonal, np.ndarray) and len(seasonal) > 0:
                # Single seasonality
                if season_type == "A":
                    seasonal = seasonal - np.mean(seasonal)
                elif season_type == "M":
                    # Safety: Replace non-positive values with 1e-6
                    seasonal = np.where(seasonal <= 0, 1e-6, seasonal)
                    # Use prod(x)^(1/n) to match R's geometric mean calculation exactly
                    # R: prod(adamBack$initial$seasonal)^{1/length(adamBack$initial$seasonal)}
                    geo_mean = np.prod(seasonal) ** (1.0 / len(seasonal))
                    if DEBUG_MAM:
                        print(f"\n[DEBUG_MAM] Two-stage seasonal normalization (single):")
                        print(f"  Raw seasonal (after safety check): {seasonal}")
                        print(f"  Geometric mean (prod^(1/n)): {geo_mean}")
                    if geo_mean > 0 and np.isfinite(geo_mean):
                        seasonal = seasonal / geo_mean
                    if DEBUG_MAM:
                        print(f"  After normalization: {seasonal}")
                        print(f"  After truncation (:-1): {seasonal[:-1]}")
                # Truncate to lag-1 elements
                initial_value['seasonal'] = seasonal[:-1]

            if DEBUG_TWOSTAGE:
                print(f"  seasonal after: {initial_value['seasonal']}")

        if DEBUG_TWOSTAGE:
            print(f"\n[PYTHON TWOSTAGE DEBUG] Final initial_value dict:")
            for key, val in initial_value.items():
                if hasattr(val, 'shape'):
                    print(f"  {key}: shape={val.shape}, values={val}")
                elif isinstance(val, list):
                    print(f"  {key}: list with {len(val)} elements")
                else:
                    print(f"  {key}: {val}")

        return initial_value

    def _build_initial_states_list(self, initial_value):
        """
        Convert initial_value dict to list matching R's unlist() order.

        R unlist() returns values in order: level, trend, seasonal(s), arima, xreg
        (adam.R:2732-2733)

        Parameters
        ----------
        initial_value : dict
            Dictionary from _extract_stage1_initials() with keys like
            'level', 'trend', 'seasonal', 'arima', 'xreg'

        Returns
        -------
        list
            Flat list of initial state values in the correct order
        """
        initial_states = []

        # Level
        if "level" in initial_value:
            initial_states.append(initial_value["level"])

        # Trend
        if "trend" in initial_value:
            initial_states.append(initial_value["trend"])

        # Seasonal - already normalized by _process_initial_values
        # R renormalizes seasonals before unlisting (adam.R:2657-2677)
        if "seasonal" in initial_value:
            seasonal_vals = initial_value["seasonal"]
            # Handle both single seasonality (array) and multiple seasonalities (list of arrays)
            if isinstance(seasonal_vals, list):
                # Multiple seasonalities: flatten the list
                for seasonal_array in seasonal_vals:
                    if isinstance(seasonal_array, (list, np.ndarray)):
                        initial_states.extend(seasonal_array)
                    else:
                        initial_states.append(seasonal_array)
            elif isinstance(seasonal_vals, np.ndarray):
                # Single seasonality: extend with array values
                initial_states.extend(seasonal_vals)
            else:
                # Fallback: append as single value
                initial_states.append(seasonal_vals)

        # ARIMA initial states
        if "arima" in initial_value:
            arima_vals = initial_value["arima"]
            if isinstance(arima_vals, (list, np.ndarray)):
                initial_states.extend(arima_vals)
            else:
                initial_states.append(arima_vals)

        # Xreg initial states (if present)
        if "xreg" in initial_value:
            xreg_vals = initial_value["xreg"]
            if isinstance(xreg_vals, (list, np.ndarray)):
                initial_states.extend(xreg_vals)
            else:
                initial_states.append(xreg_vals)

        return initial_states

    def _execute_estimation(self, estimation = True):
        """
        Execute model estimation when model_do is 'estimate'.

        This handles parameter estimation and model creation.
        """
        # Handle special case for LASSO/RIDGE with lambda=1
        self._handle_lasso_ridge_special_case()

        # Estimate the model
        if estimation:
            if self.initials_results["initial_type"] == "two-stage":
                self._run_two_stage_initialization()
            else:
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
                )

        # Build the model structure
        (
            self.model_type_dict,
            self.components_dict,
            self.lags_dict,
            self.observations_dict,
            self.profile_dict,
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
        #print(self.components_dict)
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
            Number of estimated parameters
        """
        # Store number of estimated parameters
        self.n_param_estimated = n_param_estimated

        # Initialize parameters_number if not present
        if "parameters_number" not in self.general:
            self.general["parameters_number"] = self.params_info["parameters_number"]
        self.general["parameters_number"][0][0] = self.n_param_estimated

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

    def _execute_selection(self):
        """
        Execute model selection when model_do is 'select'.

        This handles model selection and creation of the selected model.
        """
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
        )
        #print(self.adam_selected)
        #print(self.adam_selected["ic_selection"])
        
        # Updates parametes wit hthe selected model and updates adam_estimated
        self.select_best_model()

        
        #print(self.adam_selected["ic_selection"])
        # Process each selected model
        # The following commented-out loop and its associated helper method calls
        # (_update_model_from_selection, _create_matrices_for_selected_model, 
        # _update_parameters_for_selected_model) appear to be placeholders
        # or remnants of a "combine" functionality that is not fully implemented yet,
        # as indicated by the NotImplementedError in the fit method for model_do="combine".
        # These will be kept for now as they might be relevant for future development.
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
        self.ic_selection = self.adam_selected['ic_selection']
        self.results = self.adam_selected['results']
        # Find best model
        self.best_model = min(self.ic_selection.items(), key=lambda x: x[1])[0]
        self.best_id = next(i for i, result in enumerate(self.results)
                            if result['model'] == self.best_model)
        # Update dictionaries with best model results
        self.model_type_dict = self.results[self.best_id]['model_type_dict']
        self.phi_dict = self.results[self.best_id]['phi_dict']
        self.adam_estimated = self.results[self.best_id]['adam_estimated']


    def _update_model_from_selection(self, index, result):
        """
        Update model parameters with the selected model parameters.

        Parameters
        ----------
        index : int
            Index of the selected model in the results list from model selection.
        result : dict
            The dictionary containing all parameters and estimation results for the selected model.
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

        # Store the estimated model
        self.adam_estimated = result["adam_estimated"]

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
        )

        # Store created matrices
        self.adam_selected["results"][index]["adam_created"] = self.adam_created

    def _update_parameters_for_selected_model(self, index, result):
        """
        Update parameters number for a selected model. This is typically used when
        iterating through models in a selection process, particularly for a "combine" feature.

        Parameters
        ----------
        index : int
            Index of the selected model.
        result : dict
            Selected model result containing `adam_estimated` which has `n_param_estimated`.
        """
        # Update parameters number
        n_param_estimated = result["adam_estimated"]["n_param_estimated"]
        self.general["parameters_number"] = self.params_info["parameters_number"]
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

        This transforms data into appropriate formats and handles distribution selection.
        """
        # Transform data into appropriate classes
        self._format_time_series_data()

        # Handle distribution selection
        self._select_distribution()

    def _format_time_series_data(self):
        """
        Format time series data into pandas Series with appropriate indexes.
        """
        if isinstance(self.observations_dict["y_in_sample"], np.ndarray):
            # Check if frequency is a valid pandas frequency (not just "1" string)
            freq = self.observations_dict.get("frequency", "1")
            try:
                # Try to use date_range if frequency looks valid
                if freq != "1" and isinstance(self.observations_dict.get("y_start"), (pd.Timestamp, str)):
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
                    if freq != "1" and isinstance(self.observations_dict.get("y_forecast_start"), (pd.Timestamp, str)):
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
        """
        if self.general["distribution"] == "default":
            if self.general["loss"] == "likelihood":
                if self.model_type_dict["error_type"] == "A":
                    self.general["distribution_new"] = "dnorm"
                elif self.model_type_dict["error_type"] == "M":
                    self.general["distribution_new"] = "dgamma"
            elif self.general["loss"] in ["MAEh", "MACE", "MAE"]:
                self.general["distribution_new"] = "dlaplace"
            elif self.general["loss"] in ["HAMh", "CHAM", "HAM"]:
                self.general["distribution_new"] = "ds"
            elif self.general["loss"] in ["MSEh", "MSCE", "MSE", "GPL"]:
                self.general["distribution_new"] = "dnorm"
        else:
            self.general["distribution_new"] = self.general["distribution"]

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
