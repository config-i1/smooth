"""Tests for ADAM with explanatory variables (ETSX models).

Data generated in R:
    set.seed(42)
    n <- 120
    x1 <- rnorm(n); x2 <- rnorm(n)
    y <- 10 + 2*x1 - 1.5*x2 + rnorm(n)
    write.csv(data.frame(y=y, x1=x1, x2=x2), "tests/data/etsx_data.csv", row.names=FALSE)

R reference (adam(df, model="AAN", regressors="use", formula=y~x1+x2)):
    alpha=0.07559, beta=0, x1=1.908, x2=-1.464
    AIC=345.51
    Fitted[1:5]=[15.21289, 11.39521, 10.86503, 13.07484, 11.23583]
    Forecast[1:3]=[11.00165, 8.05857, 9.99434] (using last 12 rows as X_new)
"""

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM

DATA_PATH = "tests/data/etsx_data.csv"

# R reference values (from adam(df, model="AAN", regressors="use", formula=y~x1+x2))
R_AIC = 345.5101
R_FITTED_5 = [15.21289, 11.39521, 10.86503, 13.07484, 11.23583]
R_FORECAST_3 = [11.00165, 8.05857, 9.99434]
R_COEF_ALPHA = 0.07559
R_COEF_X1 = 1.908
R_COEF_X2 = -1.464


@pytest.fixture
def etsx_data():
    df = pd.read_csv(DATA_PATH)
    return df["y"].values, df[["x1", "x2"]].values


# --- regressors="use" ---

def test_etsx_use_fits(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True
    assert model._explanatory["xreg_number"] == 2


def test_etsx_use_aic_matches_r(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    assert abs(model.aic - R_AIC) < 1.0


def test_etsx_use_coefs_match_r(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    # B = [alpha, beta, x1, x2]
    B = model.coef
    assert abs(B[0] - R_COEF_ALPHA) < 0.01
    assert abs(B[2] - R_COEF_X1) < 0.1
    assert abs(B[3] - R_COEF_X2) < 0.1


def test_etsx_use_fitted_matches_r(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    assert np.allclose(model.fitted[:5].values, R_FITTED_5, atol=0.1)


def test_etsx_use_forecast_matches_r(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    X_new = X[-12:]
    fc = model.predict(h=12, X=X_new)
    assert np.allclose(fc.mean.values[:3], R_FORECAST_3, atol=0.1)


def test_etsx_use_forecast_shape(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    rng = np.random.default_rng(99)
    X_new = rng.standard_normal((12, 2))
    fc = model.predict(h=12, X=X_new)
    assert len(fc.mean) == 12
    assert not np.any(np.isnan(fc.mean.values))


def test_etsx_use_states_shape(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    # AAN + 2 xreg = 4 state rows
    assert model.states.shape[0] == 4
    assert model.states.shape[1] == len(y) + 1


# --- regressors="select" ---

def test_etsx_select_fits(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="select")
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True


def test_etsx_select_keeps_signal(etsx_data):
    """Both signal variables should be selected from the data."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="select")
    model.fit(y, X)
    assert model._explanatory["xreg_number"] == 2


def test_etsx_select_drops_noise(etsx_data):
    """Noise columns should be dropped by stepwise selection."""
    y, X = etsx_data
    rng = np.random.default_rng(7)
    X_noise = np.column_stack([X, rng.standard_normal((len(y), 3))])
    model = ADAM(model="AAN", regressors="select")
    model.fit(y, X_noise)
    assert model._explanatory["xreg_number"] <= 2


# --- regressors="adapt" ---

def test_etsx_adapt_fits(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="adapt")
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True


def test_etsx_adapt_states_shape(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="adapt")
    model.fit(y, X)
    # Level + trend + 2 adaptive xreg = 4 rows
    assert model.states.shape[0] == 4


def test_etsx_adapt_aic_better_than_null(etsx_data):
    """Adapt model should fit reasonably (AIC finite)."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="adapt")
    model.fit(y, X)
    assert np.isfinite(model.aic)


# --- edge cases ---

def test_etsx_single_regressor(etsx_data):
    y, X = etsx_data
    model = ADAM(model="ANN", regressors="use")
    model.fit(y, X[:, :1])
    assert model._explanatory["xreg_number"] == 1
    assert model.states.shape[0] == 2  # level + 1 xreg


def test_etsx_holdout(etsx_data):
    y, X = etsx_data
    h = 12
    model = ADAM(model="AAN", regressors="use", holdout=True, h=h)
    model.fit(y, X)
    fc = model.predict(h=h, X=X[-h:])
    assert len(fc.mean) == h


def test_etsx_no_x_fits_plain_ets(etsx_data):
    """regressors='use' with no X just fits a plain ETS model."""
    y, _ = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y)
    assert model._explanatory.get("xreg_model") is False


def test_etsx_short_x_pads_with_warning(etsx_data):
    """X shorter than y: last row is repeated and a warning is issued."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    with pytest.warns(UserWarning, match="repeated"):
        model.fit(y, X[:-5])
    assert model._explanatory["xreg_number"] == 2


def test_etsx_dataframe_x(etsx_data):
    y, X = etsx_data
    X_df = pd.DataFrame(X, columns=["x1", "x2"])
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X_df)
    assert model._explanatory["xreg_number"] == 2
    assert model._explanatory["xreg_names"] == ["x1", "x2"]


# --- initial type R/Python equivalence ---
# R reference: adam(df, model="AAN", regressors="use", formula=y~x1+x2, initial=X)

def test_etsx_initial_backcasting(etsx_data):
    """initial='backcasting': xreg coefs in B, AIC and fitted match R."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial="backcasting")
    model.fit(y, X)
    B = model.coef
    # R: alpha=0.07559, x1=1.90847, x2=-1.46394, AIC=345.51
    assert abs(B[0] - 0.07559) < 0.02
    assert abs(B[2] - 1.90847) < 0.1
    assert abs(B[3] - (-1.46394)) < 0.1
    assert abs(model.aic - 345.5101) < 1.0
    assert abs(model.fitted.values[0] - 15.213) < 0.5


def test_etsx_initial_complete_no_xreg_in_b(etsx_data):
    """initial='complete': xreg coefs NOT in B (backcasting handles them)."""
    y, X = etsx_data
    model_bc = ADAM(model="AAN", regressors="use", initial="backcasting")
    model_bc.fit(y, X)
    model_co = ADAM(model="AAN", regressors="use", initial="complete")
    model_co.fit(y, X)
    # "complete" B must have 2 fewer params (the 2 xreg coefs are excluded)
    assert len(model_co.coef) == len(model_bc.coef) - 2
    # AIC and fitted[0] should be close to R reference
    assert abs(model_co.aic - 341.5459) < 1.5
    assert abs(model_co.fitted.values[0] - 15.238) < 0.5


def test_etsx_initial_optimal(etsx_data):
    """initial='optimal': xreg coefs in B (after ETS initials), match R reference."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial="optimal")
    model.fit(y, X)
    B = model.coef
    # For "optimal", B = [alpha, beta, level_init, trend_init, x1, x2]
    # xreg coefs are always the last 2 (no constant in this model)
    # R: x1=1.91701, x2=-1.47879, AIC=338.37, fitted[0]=14.958
    assert abs(B[-2] - 1.91701) < 0.15
    assert abs(B[-1] - (-1.47879)) < 0.15
    assert abs(model.aic - 338.3654) < 2.0
    assert abs(model.fitted.values[0] - 14.958) < 0.5


def test_etsx_initial_two_stage(etsx_data):
    """initial='two-stage': xreg coefs in B (after ETS initials), match R reference."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial="two-stage")
    model.fit(y, X)
    B = model.coef
    # For "two-stage", B = [alpha, beta, level_init, trend_init, x1, x2]
    # R: x1=1.91715, x2=-1.47850, AIC=338.37, fitted[0]=14.959
    assert abs(B[-2] - 1.91715) < 0.15
    assert abs(B[-1] - (-1.47850)) < 0.15
    assert abs(model.aic - 338.3654) < 2.0
    assert abs(model.fitted.values[0] - 14.959) < 0.5
