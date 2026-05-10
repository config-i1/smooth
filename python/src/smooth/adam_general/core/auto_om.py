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
    and returns the best-fitting model (lowest IC) directly.  The returned
    object is a plain :class:`OM` or :class:`OMG` with one extra attribute
    ``time_elapsed_`` (total selection time in seconds), matching R's
    ``$timeElapsed`` on the returned ``om`` object.

    Parameters
    ----------
    model :
        ETS spec passed to every candidate, including both sub-models of the
        general (OMG) candidate.
    lags :
        Seasonal lags shared across all candidates.
    orders :
        ARIMA orders forwarded to every candidate.
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
        lags: Optional[List[int]] = None,
        orders: Optional[Dict[str, Any]] = None,
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
        ets: Literal["conventional", "adam"] = "conventional",
    ) -> None:
        if isinstance(occurrence, str):
            occurrence = [occurrence]
        unknown = set(occurrence) - set(_VALID_OCCURRENCE)
        if unknown:
            raise ValueError(f"Unknown occurrence type(s): {unknown!r}")
        if ic not in _IC_ATTR:
            raise ValueError(f"ic must be one of {list(_IC_ATTR)}; got {ic!r}")
        self.model = model
        self.lags = lags
        self.orders = orders
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
        if ets not in ("conventional", "adam"):
            raise ValueError(f"Invalid ets: {ets!r}. Must be 'conventional' or 'adam'.")
        self.ets = ets

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
                model_a=self.model,
                model_b=self.model,
                orders_a=self.orders,
                constant_a=self.constant,
                persistence_a=self.persistence,
                phi_a=self.phi,
                arma_a=self.arma,
                regressors_a=self.regressors,
                ets=self.ets,
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
            ets=self.ets,
            **common,
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> Union[OM, OMG]:
        """Fit all candidates and return the best model by IC.

        Parameters
        ----------
        y :
            Observation vector (binarised internally: any non-zero → 1).
        X :
            Optional regressor matrix forwarded to every candidate.

        Returns
        -------
        OM or OMG
            The best-fitting occurrence model, with ``time_elapsed_`` set to
            the total selection time in seconds.
        """
        start = time.time()
        best_ic = float("inf")
        best_model: Optional[Union[OM, OMG]] = None

        for occ in self.occurrence:
            try:
                m = self._build_candidate(occ)
                m.fit(y, X)
                ic_val = _get_ic(m, self.ic)
                if ic_val < best_ic:
                    best_ic = ic_val
                    best_model = m
            except Exception as exc:
                warnings.warn(
                    f"AutoOM: occurrence='{occ}' failed with {type(exc).__name__}: "
                    f"{exc} — skipping.",
                    stacklevel=2,
                )

        if best_model is None:
            raise RuntimeError("AutoOM: all occurrence types failed to fit.")

        best_model.time_elapsed_ = time.time() - start
        return best_model
