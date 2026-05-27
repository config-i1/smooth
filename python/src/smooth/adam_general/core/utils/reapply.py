"""Container for the result of ``ADAM.reapply``.

Mirrors R's ``"reapply"`` S3 class produced by ``reapply.adam``
(R/reapply.R:772-777). The field set and array dimensions match R
exactly so downstream code (notably ``ADAM.reforecast``) can be a
straight port.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReapplyResult:
    """Container for an ``ADAM.reapply`` run.

    Attributes
    ----------
    time_elapsed : float
        Wall-clock seconds taken by the reapply call.
    y : pandas.Series
        The in-sample actuals (R: ``$y``). Index matches ``ADAM.fitted``.
    states : numpy.ndarray
        State cube of shape ``(n_components, obs_in_sample + lags_model_max,
        nsim)`` — components × time × replicate, mirroring R's
        ``$states`` array.
    refitted : pandas.DataFrame
        ``(obs_in_sample, nsim)`` matrix of fitted paths (R: ``$refitted``).
        Index matches ``y.index``; columns are ``nsim1 … nsimN``.
    fitted : pandas.Series
        Conditional-mean fitted values from the underlying fit
        (R: ``$fitted``) — identical to ``ADAM.fitted`` and provided for
        plotting against ``refitted``.
    model : str
        Model spec string (e.g. ``"ETS(MAM)"``).
    transition : numpy.ndarray
        Transition-matrix cube ``(n_components, n_components, nsim)``.
    measurement : numpy.ndarray
        Measurement-matrix cube ``(obs_in_sample, n_components, nsim)``.
    persistence : pandas.DataFrame
        Persistence matrix ``(n_components, nsim)``. Row index gives the
        component names where available.
    profile : numpy.ndarray
        Final profile cube ``(n_components, lags_model_max, nsim)`` —
        the profile after each replicate's in-sample reapply run.
    random_parameters : pandas.DataFrame
        ``(nsim, n_parameters)`` sampled-from-MVN parameter matrix.
        Columns are :attr:`ADAM.coef_names`.
    nsim : int
        Number of replicates.
    """

    time_elapsed: float
    y: pd.Series
    states: np.ndarray
    refitted: pd.DataFrame
    fitted: pd.Series
    model: str
    transition: np.ndarray
    measurement: np.ndarray
    persistence: pd.DataFrame
    profile: np.ndarray
    random_parameters: pd.DataFrame
    nsim: int

    def __repr__(self) -> str:
        return (
            f"ReapplyResult(model={self.model!r}, nsim={self.nsim}, "
            f"refitted={self.refitted.shape}, "
            f"time_elapsed={self.time_elapsed:.2f}s)"
        )

    def __str__(self) -> str:
        return (
            f"Reapply for {self.model!r}\n"
            f"  replicates : {self.nsim}\n"
            f"  refitted   : {self.refitted.shape[0]} x {self.refitted.shape[1]}\n"
            f"  parameters : {self.random_parameters.shape[1]}\n"
            f"  elapsed    : {self.time_elapsed:.2f}s"
        )
