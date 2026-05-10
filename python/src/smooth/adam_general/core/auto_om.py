"""Automatic Occurrence Model Selection (AutoOM): port of R's ``auto.om()``.

Tries each occurrence type from ``occurrence`` and returns the model with the
lowest information criterion.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

from numpy.typing import NDArray

from smooth.adam_general.core.om import OM
from smooth.adam_general.core.omg import OMG

_VALID_OCCURRENCE = (
    "fixed",
    "odds-ratio",
    "inverse-odds-ratio",
    "direct",
    "general",
)

_IC_ATTR = {"AIC": "aic", "AICc": "aicc", "BIC": "bic", "BICc": "bicc"}


def _get_ic(model: Union[OM, OMG], ic_name: str) -> float:
    return float(getattr(model, _IC_ATTR[ic_name]))


class AutoOM:
    """Automatic occurrence model selection — port of R's ``auto.om()``.

    Fits an :class:`OM` (or :class:`OMG`) for each entry in ``occurrence``
    and returns the configuration with the lowest information criterion.

    Parameters
    ----------
    model :
        ETS spec passed to every non-general OM candidate.
    model_a, model_b :
        ETS specs for the OMG (``"general"``) candidate.
        ``model_b`` defaults to ``model_a``.
    lags :
        Seasonal lags shared across all candidates.
    orders :
        ARIMA orders for non-general OM candidates.
    orders_a, orders_b :
        ARIMA orders for the OMG candidate's two sub-models.
    occurrence :
        Sequence of occurrence types to try.
    h, holdout :
        Forecast horizon and holdout flag forwarded to every candidate.
    ic :
        Information criterion used for selection.
    """

    def __init__(
        self,
        model: str = "ZXZ",
        model_a: str = "MNN",
        model_b: Optional[str] = None,
        lags: Optional[List[int]] = None,
        orders: Optional[Dict[str, Any]] = None,
        orders_a: Optional[Dict[str, Any]] = None,
        orders_b: Optional[Dict[str, Any]] = None,
        occurrence: Union[List[str], str] = list(_VALID_OCCURRENCE),
        h: int = 0,
        holdout: bool = False,
        persistence: Optional[Dict[str, float]] = None,
        phi: Optional[float] = None,
        initial: Union[str, Dict[str, Any]] = "backcasting",
        constant: bool = False,
        arma: Optional[Dict[str, Any]] = None,
        regressors: Literal["use", "select", "adapt"] = "use",
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        verbose: int = 0,
        nlopt_kargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(occurrence, str):
            occurrence = [occurrence]
        unknown = set(occurrence) - set(_VALID_OCCURRENCE)
        if unknown:
            raise ValueError(f"Unknown occurrence type(s): {unknown!r}")
        if ic not in _IC_ATTR:
            raise ValueError(f"ic must be one of {list(_IC_ATTR)}; got {ic!r}")
        self.model = model
        self.model_a = model_a
        self.model_b = model_b if model_b is not None else model_a
        self.lags = lags
        self.orders = orders
        self.orders_a = orders_a
        self.orders_b = orders_b
        self.occurrence = list(occurrence)
        self.h = h
        self.holdout = holdout
        self.persistence = persistence
        self.phi = phi
        self.initial = initial
        self.constant = constant
        self.arma = arma
        self.regressors = regressors
        self.ic = ic
        self.bounds = bounds
        self.verbose = verbose
        self.nlopt_kargs = nlopt_kargs
        self.best_model: Optional[Union[OM, OMG]] = None
        self.occurrence_: Optional[str] = None
        self.ic_values: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _common_kwargs(self) -> Dict[str, Any]:
        return dict(
            lags=self.lags,
            h=self.h,
            holdout=self.holdout,
            initial=self.initial,
            ic=self.ic,
            bounds=self.bounds,
            verbose=self.verbose,
            nlopt_kargs=self.nlopt_kargs,
        )

    def _build_candidate(self, occ: str) -> Union[OM, OMG]:
        common = self._common_kwargs()
        if occ == "general":
            return OMG(
                model_a=self.model_a,
                model_b=self.model_b,
                orders_a=self.orders_a if self.orders_a is not None else self.orders,
                orders_b=self.orders_b if self.orders_b is not None else self.orders,
                constant_a=self.constant,
                persistence_a=self.persistence,
                phi_a=self.phi,
                arma_a=self.arma,
                regressors_a=self.regressors,
                **common,
            )
        return OM(
            model=self.model,
            occurrence=occ,
            orders=self.orders,
            constant=self.constant,
            persistence=self.persistence,
            phi=self.phi,
            arma=self.arma,
            regressors=self.regressors,
            **common,
        )

    def _check_is_fitted(self) -> None:
        if self.best_model is None:
            raise RuntimeError("AutoOM has not been fitted yet. Call fit() first.")

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> "AutoOM":
        """Fit all candidates and select the best by IC.

        Parameters
        ----------
        y :
            Observation vector (binarised internally: any non-zero → 1).
        X :
            Optional regressor matrix forwarded to every candidate.
        """
        start = time.time()
        best_ic = float("inf")
        best_model: Optional[Union[OM, OMG]] = None
        best_occ: Optional[str] = None
        ic_values: Dict[str, float] = {}

        for occ in self.occurrence:
            try:
                m = self._build_candidate(occ)
                m.fit(y, X)
                ic_val = _get_ic(m, self.ic)
                ic_values[occ] = ic_val
                if ic_val < best_ic:
                    best_ic = ic_val
                    best_model = m
                    best_occ = occ
            except Exception as exc:
                warnings.warn(
                    f"AutoOM: occurrence='{occ}' failed with {type(exc).__name__}: "
                    f"{exc} — skipping.",
                    stacklevel=2,
                )

        if best_model is None:
            raise RuntimeError("AutoOM: all occurrence types failed to fit.")

        self.best_model = best_model
        self.occurrence_ = best_occ
        self.ic_values = ic_values
        self.time_elapsed_ = time.time() - start
        return self

    def predict(self, h: int, X: Optional[NDArray] = None, **kwargs):
        """Forecast from the best occurrence model."""
        self._check_is_fitted()
        return self.best_model.predict(h=h, X=X, **kwargs)

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> NDArray:
        self._check_is_fitted()
        return self.best_model.fitted

    @property
    def residuals(self) -> NDArray:
        self._check_is_fitted()
        return self.best_model.residuals

    @property
    def actuals(self) -> NDArray:
        self._check_is_fitted()
        return self.best_model.actuals

    @property
    def coef(self) -> NDArray:
        self._check_is_fitted()
        return self.best_model.coef

    @property
    def b_value(self) -> NDArray:
        self._check_is_fitted()
        return self.best_model.b_value

    @property
    def loss_value(self) -> float:
        self._check_is_fitted()
        return self.best_model.loss_value

    @property
    def loglik(self) -> float:
        self._check_is_fitted()
        return self.best_model.loglik

    @property
    def aic(self) -> float:
        self._check_is_fitted()
        return self.best_model.aic

    @property
    def aicc(self) -> float:
        self._check_is_fitted()
        return self.best_model.aicc

    @property
    def bic(self) -> float:
        self._check_is_fitted()
        return self.best_model.bic

    @property
    def bicc(self) -> float:
        self._check_is_fitted()
        return self.best_model.bicc

    @property
    def nobs(self) -> int:
        self._check_is_fitted()
        return self.best_model.nobs

    @property
    def nparam(self) -> int:
        self._check_is_fitted()
        return self.best_model.nparam

    @property
    def model_name(self) -> str:
        self._check_is_fitted()
        return self.best_model.model_name

    @property
    def lags_used(self) -> List[int]:
        self._check_is_fitted()
        return self.best_model.lags_used

    @property
    def holdout_data(self) -> Optional[NDArray]:
        self._check_is_fitted()
        return self.best_model.holdout_data

    @property
    def scale(self) -> float:
        return float("nan")

    @property
    def sigma(self) -> float:
        return float("nan")

    @property
    def distribution_(self) -> str:
        return "plogis"

    @property
    def loss_(self) -> str:
        return "likelihood"

    @property
    def time_elapsed(self) -> float:
        self._check_is_fitted()
        return self.time_elapsed_
