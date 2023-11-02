import numpy as np
from smooth.adam_general._adam_general import adam_fitter


def sma(y, order=1, h=10):
    """SMA"""
    y = y.astype(np.float64)

    ic = lambda e: np.sum(e**2)
    obs_all = len(y) + h
    obs_in_sample = len(y)
    y_in_sample = y

    E_type = "A"
    T_type = "N"
    S_type = "N"

    components_num_ETS = 0
    components_num_ETS_seasonal = 0
    xreg_number = 0
    constant_required = False
    ot = np.ones_like(y_in_sample)

    def creator_sma(order):
        lags_model_all = np.ones(shape=(order, 1))
        lags_model_max = 1
        obs_states = obs_in_sample + 1

        # profiles_recent_table = np.zeros(
        #     shape=(order, lags_model_max), dtype=np.float64
        # )

        profiles_recent_table = np.mean(y_in_sample[0 : (order - 1)]) * np.ones(
            shape=(order, lags_model_max), dtype=np.float64
        )

        profiles_observed_table = np.tile(
            np.arange(order), (obs_all + lags_model_max, 1)
        ).T

        matF = np.ones((order, order)) / order
        matWt = np.ones((obs_in_sample, order))

        vecG = np.ones(order) / order
        # matVt = np.zeros((order, obs_states))
        matVt = np.empty((order, obs_states))
        # matVt.fill(np.nan)

        adam_fitted = adam_fitter(
            matrixVt=matVt,
            matrixWt=matWt,
            matrixF=matF,
            vectorG=vecG,
            lags=lags_model_all,
            profilesObserved=profiles_observed_table,
            profilesRecent=profiles_recent_table,
            E=E_type,
            T=T_type,
            S=S_type,
            nNonSeasonal=components_num_ETS,
            nSeasonal=components_num_ETS_seasonal,
            nArima=order,
            nXreg=xreg_number,
            constant=constant_required,
            vectorYt=y_in_sample,
            vectorOt=ot,
            backcast=True,
        )

        return adam_fitted

    return creator_sma(order=order)
