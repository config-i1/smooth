"""Python wrapper for the ADAM forecasting model.

 * Equ
valent code in R is 
"""
from itertools import product
from typing import Union, List, Literal, Optional

import numpy as np

from numpy.typing import NDArray

DISTRIBUTON_OPTIONS = Literal[
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
    "TMSE" "GTMSE",
    "MSCE",
]


class ADAM:
    def __init__(
        self,
        model: Union[str, List[str]] = "ZXZ",
        lags: Optional[NDArray] = None,
        ar_order: Union[int, List[int]] = 0,
        i_order: Union[int, List[int]] = 0,
        ma_order: Union[int, List[int]] = 0,
        # SELECT: skipping this for now (auto.arima thingy)
        constant: bool = False,
        regressors: Literal["use", "select", "adapt"] = "use",
        distribution: Optional[DISTRIBUTON_OPTIONS] = None,
        loss: LOSS_OPTIONS = "likelihood",
        # outliers: we're skipping this for now
        loss_horizon: Optional[int] = None,
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        # occurrence: skipping this for now,
        # ---- These are the estimated parameters that we can choose to fix ----
        # Dictionary of terms e.g. {"alpha": 0.5, "beta": 0.5}
        persistence: Optional[dict] = None,
        phi: Optional[float] = None,
        initial: Optional[dict] = None,
        # TODO: enforce the structure of this
        arma: Optional[dict] = None,
        # ----- End of parameters----
        verbose: int = 0,
        # Fisher information matrix: We're skipping for now and we'll use composition
        # for it like Grid Search in scikit-learn.
        # initial values for optimization parameters:
        nlopt_initial: Optional[dict] = None,
        nlopt_upper: Optional[dict] = None,
        nlopt_lower: Optional[dict] = None,
        nlopt_kargs: Optional[dict] = None,
        # specific to losses or distributions
        reg_lambda: Optional[float] = None,
        gnorm_shape: Optional[float] = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        model : str, optional
            _description_, by default "ZXZ"
        lags : NDArray, optional
            _description_, by default None
        ar_order : Union[int, List[int]], optional
            _description_, by default 0
        i_order : Union[int, List[int]], optional
            _description_, by default 0
        ma_order : Union[int, List[int]], optional
            _description_, by default 0
        constant : bool, optional
            _description_, by default False
        regressors : Literal[&quot;use&quot;, &quot;select&quot;, &quot;adapt&quot;], optional
            _description_, by default "use"
        distribution : DISTRIBUTON_OPTIONS, optional
            _description_, by default None
        loss : LOSS_OPTIONS, optional
            _description_, by default "likelihood"
        loss_horizon : Optional[int], optional
            _description_, by default None
        ic : Literal[&quot;AIC&quot;, &quot;AICc&quot;, &quot;BIC&quot;, &quot;BICc&quot;], optional
            _description_, by default "AICc"
        bounds : Literal[&quot;usual&quot;, &quot;admissible&quot;, &quot;none&quot;], optional
            _description_, by default "usual"
        persistence : Optional[dict], optional
            _description_, by default None
        phi : Optional[float], optional
            _description_, by default None
        initial : Optional[dict], optional
            _description_, by default None
        arma : Optional[dict], optional
            _description_, by default None
        verbose : int, optional
            _description_, by default 0
        nlopt_initial : Optional[dict], optional
            _description_, by default None
        nlopt_upper : Optional[dict], optional
            _description_, by default None
        nlopt_lower : Optional[dict], optional
            _description_, by default None
        nlopt_kargs : Optional[dict], optional
            _description_, by default None
        reg_lambda : Optional[float], optional
            _description_, by default None
        gnorm_shape : Optional[float], optional
            _description_, by default None
        """
        self.model = model
        self.lags = lags
        self.ar_order = ar_order
        self.i_order = i_order
        self.ma_order = ma_order
        self.constant = constant
        self.regressors = regressors
        self.distribution = distribution
        self.loss = loss
        self.loss_horizon = loss_horizon
        self.ic = ic
        self.bounds = bounds
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

    def _parameters_checker(self):
        """Checks the parameters for the model.

        Note: this is in line R/adamGeneral.R
        """
        # initialise a matrix storing the number of parameters to estimate
        # top row corresponds to parameters estimated internally, bottom row to provided
        # parameters by the user.
        # The columns are:
        # 1. ETS & ARIMA parameters
        # 2. Explanatory variable parameters
        # 3. Occurence parameters
        # 4. Scale model parameters
        # 5. All the parameters (sum of the above)
        parameters_number = np.zeros((2, 5))

        if isinstance(self.model, list):
            pool_error_msg = f"You have defined strange models in the pool:\n{self.model}"
            if not all([isinstance(i, str) for i in self.model]):
                raise ValueError(
                    "The model parameter should be a string or a list of strings"
                )
            if any([(len(m) > 4 or len(m) < 3) for m in self.model]):
                raise ValueError(pool_error_msg)
            if any([m[0] not in ["A", "M"] for m in self.model]):
                raise ValueError(pool_error_msg)
            if any([m[1] not in ["A", "M", "N"] for m in self.model]):
                raise ValueError(pool_error_msg)
            if any([m[2] not in ["A", "M", "N", "d"] for m in self.model]):
                raise ValueError(pool_error_msg)
            if any([m[3] not in ["A", "M", "N"] for m in self.model if len(m) > 3]):
                raise ValueError(pool_error_msg)
            models_pool = self.model
        elif isinstance(self.model, str):
            model_error_msg = f"You have defined a strange model:\n{self.model}"
            if len(self.model) > 4 or len(self.model) < 3:
                raise ValueError(model_error_msg)
            if self.model[0] not in ["A", "M", "Z", "X", "Y", "P", "F"]:
                raise ValueError(model_error_msg)
            if self.model[1] not in ["N", "A", "M", "Z", "X", "Y", "P", "F"]:
                raise ValueError(model_error_msg)
            if self.model[2] not in ["N", "A", "M", "Z", "X", "Y", "P", "F", "d"]:
                raise ValueError(model_error_msg)
            if len(self.model) > 3 and self.model[3] not in ["N", "A", "M", "Z", "X", "Y", "P", "F"]:
                raise ValueError(model_error_msg)

            if any([m not in ["A", "M", "N", "d"] for m in self.model]):
                # pool creating logic
                if "P" in self.model:
                    mul_models = list(product(("M"), ("M", "Md", "N"), ("M", "N")))
                    add_models = list(product(("A"), ("A", "Ad", "N"), ("A", "N")))
                    models_pool = mul_models + add_models
                elif "F" in self.model:
                    models_pool = list(
                        product(
                            ("A", "M", "N"),
                            ("A", "M", "Ad", "Md", "N"),
                            ("A", "M", "N"),
                        )
                    )
                else:
                    # create the possible error types for the pool.
                    if self.model[0] in ("A", "M"):
                        error_type = (self.model[0],)
                    elif self.model[0] == "Z":
                        error_type = ("A", "M")
                    elif self.model[0] == "X":
                        error_type = ("A",)
                    elif self.model[0] == "Y":
                        error_type = ("M",)

                    # create the possible trend types for the pool.
                    if len(self.model) == 3:
                        if self.model[1] in ("A", "M", "N"):
                            trend_type = (self.model[1],)
                        elif self.model[1] == "Z":
                            trend_type = ("A", "M", "Ad", "Md", "N")
                        elif self.model[1] == "X":
                            trend_type = ("A", "Ad", "N")
                        elif self.model[1] == "Y":
                            trend_type = ("M", "Md", "N")
                    elif len(self.model) == 4:
                        trend_type = (self.model[1:3],)

                    if self.model[-1] in ("A", "M", "N"):
                        season_type = (self.model[-1],)
                    elif self.model[-1] == "Z":
                        season_type = ("A", "M", "N")
                    elif self.model[-1] == "X":
                        season_type = ("A", "N")
                    elif self.model[-1] == "Y":
                        season_type = ("M", "N")
                    
                    models_pool = list(product(error_type, trend_type, season_type))

            else:
                models_pool = [self.model]

        return models_pool


    def _architector(self):
        """Creates the technical variables (lags etc) based on the type of the model.

        Note: this is in line 679 in R/adam.R
        """
        pass

    def fit(self, y: NDArray, X: Optional[NDArray] = None):
        """Fit"""
        return self

    def predict(
        self,
        h: int,
        X: Optional[NDArray] = None,
    ) -> NDArray:
        """Point forecasts only."""
        pass


# TODO: Add methods for intervals and simulated future paths
