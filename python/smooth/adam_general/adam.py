"""Python wrapper for the ADAM forecasting model.

 * Equ
valent code in R is 
"""

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
        model: str = "ZXZ",
        lags: NDArray = None,
        ar_order: Union[int, List[int]] = 0,
        i_order: Union[int, List[int]] = 0,
        ma_order: Union[int, List[int]] = 0,
        # SELECT: skipping this for now (auto.arima thingy)
        constant: bool = False,
        regressors: Literal["use", "select", "adapt"] = "use",
        distribution: DISTRIBUTON_OPTIONS = None,
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
