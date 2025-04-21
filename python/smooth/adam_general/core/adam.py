import time
import warnings
from typing import Union, List, Optional, Literal, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general.core.checker import parameters_checker
from smooth.adam_general.core.estimator import estimator, selector
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
    ADAM (Augmented Dynamic Adaptive Model) class for time series forecasting.

    This class implements various time series models including ETS, ARIMA, and
    their combinations for forecasting purposes. It handles parameter estimation,
    model selection, and prediction.
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
            Error distribution.
        loss : LOSS_OPTIONS, default="likelihood"
            Loss function for parameter estimation.
        loss_horizon : Optional[int], default=None
            Number of steps for multi-step loss.
        outliers : Literal["ignore", "detect", "use"], default="ignore"
            Outlier handling method.
        outliers_level : float, default=0.99
            Confidence level for outlier detection.
        ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
            Information criterion for model selection.
        bounds : Literal["usual", "admissible", "none"], default="usual"
            Parameter bounds specification.
        occurrence : OCCURRENCE_OPTIONS, default="none"
            Occurrence model type for intermittent data.
        persistence : Optional[Dict[str, float]], default=None
            Persistence parameters (e.g., {"alpha": 0.5, "beta": 0.5}).
        phi : Optional[float], default=None
            Damping parameter for damped trend models.
        initial : INITIAL_OPTIONS, default=None
            Initial values for states.
        arma : Optional[Dict[str, Any]], default=None
            ARMA parameters specification.
        verbose : int, default=0
            Verbosity level (0=silent, higher values=more output).
        h : Optional[int], default=None
            Forecast horizon.
        holdout : bool, default=False
            Whether to use holdout sample for validation.
        model_do : Literal["estimate", "select", "combine"], default="estimate"
            Model action ('estimate', 'select', or 'combine').
        fast : bool, default=False
            Whether to use fast estimation.
        models_pool : Optional[List[Dict[str, Any]]], default=None
            Pool of models for selection.
        lambda_param : Optional[float], default=None
            Lambda parameter for regularization.
        frequency : Optional[str], default=None
            Time series frequency.
        profiles_recent_provided : bool, default=False
            Whether recent profiles are provided.
        profiles_recent_table : Optional[Any], default=None
            Table of recent profiles.
        nlopt_initial : Optional[Dict[str, Any]], default=None
            Initial values for optimization parameters.
        nlopt_upper : Optional[Dict[str, Any]], default=None
            Upper bounds for optimization parameters.
        nlopt_lower : Optional[Dict[str, Any]], default=None
            Lower bounds for optimization parameters.
        nlopt_kargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments for the optimizer.
        reg_lambda : Optional[float], default=None
            Regularization parameter for LASSO/RIDGE losses.
        gnorm_shape : Optional[float], default=None
            Shape parameter for generalized normal distribution.
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
        Fit the ADAM model to the provided time series data.

        Parameters
        ----------
        y : NDArray
            Time series data (numpy array or pandas Series).
        X : Optional[NDArray], default=None
            Exogenous variables (regressors).

        Returns
        -------
        self : ADAM
            The fitted model object.
        """
        # Store fit parameters - these are now set in __init__
        # No need to call _setup_parameters as those parameters are now instance attributes

        # Use X if provided
        ts = y
        if X is not None:
            # Here we would handle X, but implementation details depend on
            # how exogenous variables are handled in the original code
            pass

        # Check parameters and prepare data
        self._check_parameters(ts)

        # Execute model estimation or selection based on model_do
        if self.model_type_dict["model_do"] == "estimate":
            self._execute_estimation()
        elif self.model_type_dict["model_do"] == "select":
            self._execute_selection()
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
    ) -> NDArray:
        """
        Generate point forecasts using the fitted ADAM model.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps to forecast).
        X : Optional[NDArray], default=None
            Exogenous variables for the forecast period.

        Returns
        -------
        NDArray
            Point forecasts for the next h periods.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        # Set forecast horizon
        self.h = h

        # Handle exogenous variables if provided
        if X is not None:
            # Implementation would depend on how X is used in the original code
            pass

        # Validate prediction inputs and prepare data for forecasting
        self._validate_prediction_inputs()

        # Prepare data for prediction
        self._prepare_prediction_data()

        # Execute the prediction
        self._execute_prediction()

        # Return the point forecasts
        return self.forecast_results["forecast"]

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
            Dictionary containing forecast and prediction intervals.

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
            # Implementation would depend on how X is used in the original code
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

    def _execute_estimation(self):
        """
        Execute model estimation when model_do is 'estimate'.

        This handles parameter estimation and model creation.
        """
        # Handle special case for LASSO/RIDGE with lambda=1
        self._handle_lasso_ridge_special_case()

        # Estimate the model
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
            lags_dict=self.lags_dict,
            observations_dict=self.observations_dict,
            general_dict=self.general,
            models_pool=self.models_pool,
        )

        # Process each selected model
        for i, result in enumerate(self.adam_selected["results"]):
            # Update model parameters with the selected model
            self._update_model_from_selection(i, result)

            # Create matrices for this model
            self._create_matrices_for_selected_model(i)

            # Update parameters number for this model
            self._update_parameters_for_selected_model(i, result)

    def _update_model_from_selection(self, index, result):
        """
        Update model parameters with the selected model parameters.

        Parameters
        ----------
        index : int
            Index of the selected model
        result : dict
            Selected model result
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
        Create matrices for a selected model.

        Parameters
        ----------
        index : int
            Index of the selected model
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
        Update parameters number for a selected model.

        Parameters
        ----------
        index : int
            Index of the selected model
        result : dict
            Selected model result
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
            self.y_in_sample = pd.Series(
                self.observations_dict["y_in_sample"],
                index=pd.date_range(
                    start=self.observations_dict["y_start"],
                    periods=len(self.observations_dict["y_in_sample"]),
                    freq=self.observations_dict["frequency"],
                ),
            )
            if self.general["holdout"]:
                self.y_holdout = pd.Series(
                    self.observations_dict["y_holdout"],
                    index=pd.date_range(
                        start=self.observations_dict["y_forecast_start"],
                        periods=len(self.observations_dict["y_holdout"]),
                        freq=self.observations_dict["frequency"],
                    ),
                )
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
        )

    def _format_prediction_results(self):
        """
        Format the prediction results into a more user-friendly structure.

        Returns
        -------
        dict
            Formatted prediction results including point forecasts and intervals.
        """
        # Calculate and include elapsed time before returning
        self.forecast_results["elapsed_time"] = time.time() - self.start_time

        return self.forecast_results
