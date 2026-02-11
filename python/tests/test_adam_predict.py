"""Tests for ADAM.predict() method options: intervals, levels, cumulative, scenarios."""

import warnings

import numpy as np
import pytest

from smooth.adam_general.core.adam import ADAM


# ---------------------------------------------------------------------------
# Module-level data and pre-fitted models (created once, shared across tests)
# ---------------------------------------------------------------------------

def _make_simple():
    np.random.seed(42)
    return 10 + 0.5 * np.arange(50) + np.random.randn(50) * 2

def _make_seasonal():
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    return 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 3

def _make_multiplicative():
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
    noise = 1 + np.random.randn(n) * 0.05
    return trend * seasonal * noise


# Pre-fit models at module load time
_simple_data = _make_simple()
_seasonal_data = _make_seasonal()
_mult_data = _make_multiplicative()

_simple_model = ADAM(model="ANN", lags=[1])
_simple_model.fit(_simple_data)

_additive_model = ADAM(model="ANA", lags=[1, 12])
_additive_model.fit(_seasonal_data)

_mult_model = ADAM(model="MAM", lags=[1, 12])
_mult_model.fit(_mult_data)


# ---------------------------------------------------------------------------
# 1. Interval types
# ---------------------------------------------------------------------------

class TestPredictIntervalTypes:
    """Tests for each interval type with additive and multiplicative models."""

    def test_interval_none(self):
        r = _simple_model.predict(h=5, interval="none")
        assert r.shape == (5, 1)
        assert list(r.columns) == ["mean"]

    def test_interval_approximate(self):
        r = _additive_model.predict(h=5, interval="approximate")
        assert r.shape == (5, 3)
        assert "mean" in r.columns
        lower_cols = [c for c in r.columns if c.startswith("lower")]
        upper_cols = [c for c in r.columns if c.startswith("upper")]
        assert len(lower_cols) == 1
        assert len(upper_cols) == 1
        assert (r[lower_cols[0]].values <= r["mean"].values).all()
        assert (r["mean"].values <= r[upper_cols[0]].values).all()

    def test_interval_simulated(self):
        r = _additive_model.predict(h=5, interval="simulated", nsim=1000)
        assert r.shape == (5, 3)
        lower_col = [c for c in r.columns if c.startswith("lower")][0]
        upper_col = [c for c in r.columns if c.startswith("upper")][0]
        assert (r[lower_col].values <= r["mean"].values).all()
        assert (r["mean"].values <= r[upper_col].values).all()

    def test_interval_prediction_additive_resolves_approximate(self):
        """For additive models, interval='prediction' auto-resolves to 'approximate'."""
        r_pred = _additive_model.predict(h=5, interval="prediction")
        r_approx = _additive_model.predict(h=5, interval="approximate")
        np.testing.assert_array_almost_equal(
            r_pred.values, r_approx.values, decimal=10
        )

    def test_interval_prediction_multiplicative_has_intervals(self):
        """For multiplicative models, interval='prediction' resolves to 'simulated'."""
        r = _mult_model.predict(h=5, interval="prediction", nsim=1000)
        assert r.shape == (5, 3)
        lower_col = [c for c in r.columns if c.startswith("lower")][0]
        upper_col = [c for c in r.columns if c.startswith("upper")][0]
        assert (r[lower_col].values <= r["mean"].values).all()
        assert (r["mean"].values <= r[upper_col].values).all()

    def test_interval_simulated_multiplicative(self):
        r = _mult_model.predict(h=5, interval="simulated", nsim=1000)
        assert r.shape == (5, 3)
        assert "mean" in r.columns


# ---------------------------------------------------------------------------
# 2. Confidence levels
# ---------------------------------------------------------------------------

class TestPredictLevels:
    """Tests for different confidence level values."""

    def test_level_80_column_names(self):
        r = _simple_model.predict(h=5, interval="approximate", level=0.80)
        assert "lower_0.1" in r.columns
        assert "upper_0.9" in r.columns

    def test_level_99_column_names(self):
        r = _simple_model.predict(h=5, interval="approximate", level=0.99)
        assert "lower_0.005" in r.columns
        assert "upper_0.995" in r.columns

    def test_wider_level_has_wider_interval(self):
        r80 = _simple_model.predict(h=5, interval="approximate", level=0.80)
        r99 = _simple_model.predict(h=5, interval="approximate", level=0.99)
        lower80 = r80[[c for c in r80.columns if c.startswith("lower")][0]]
        upper80 = r80[[c for c in r80.columns if c.startswith("upper")][0]]
        lower99 = r99[[c for c in r99.columns if c.startswith("lower")][0]]
        upper99 = r99[[c for c in r99.columns if c.startswith("upper")][0]]
        width80 = (upper80 - lower80).values
        width99 = (upper99 - lower99).values
        assert (width99 > width80).all()

    def test_level_as_percentage_auto_converts(self):
        """level=95 should be auto-converted to 0.95."""
        r = _simple_model.predict(h=5, interval="approximate", level=95)
        assert "lower_0.025" in r.columns
        assert "upper_0.975" in r.columns

    def test_default_level_95(self):
        r = _simple_model.predict(h=5, interval="approximate")
        assert "lower_0.025" in r.columns
        assert "upper_0.975" in r.columns


# ---------------------------------------------------------------------------
# 3. Side parameter
# ---------------------------------------------------------------------------

class TestPredictSides:
    """Tests for the side parameter (both, upper, lower)."""

    def test_side_both(self):
        r = _simple_model.predict(h=5, interval="approximate", side="both")
        lower_cols = [c for c in r.columns if c.startswith("lower")]
        upper_cols = [c for c in r.columns if c.startswith("upper")]
        assert len(lower_cols) == 1
        assert len(upper_cols) == 1

    def test_side_upper_raises(self):
        """side='upper' currently raises UnboundLocalError (not yet implemented)."""
        with pytest.raises(UnboundLocalError):
            _simple_model.predict(h=5, interval="approximate", side="upper")

    def test_side_lower_raises(self):
        """side='lower' currently raises UnboundLocalError (not yet implemented)."""
        with pytest.raises(UnboundLocalError):
            _simple_model.predict(h=5, interval="approximate", side="lower")


# ---------------------------------------------------------------------------
# 4. Cumulative forecasts
# ---------------------------------------------------------------------------

class TestPredictCumulative:
    """Tests for cumulative forecast output."""

    def test_cumulative_mean_equals_sum(self):
        """Cumulative mean should equal sum of individual step means."""
        r_cum = _simple_model.predict(h=5, interval="none", cumulative=True)
        r_ind = _simple_model.predict(h=5, interval="none", cumulative=False)
        np.testing.assert_almost_equal(
            r_cum["mean"].values[0], r_ind["mean"].sum(), decimal=8
        )

    def test_cumulative_replicates_value(self):
        """Cumulative mode replicates the cumulative value across all h rows."""
        r = _simple_model.predict(h=5, interval="none", cumulative=True)
        assert r.shape == (5, 1)
        assert (r["mean"].values == r["mean"].values[0]).all()

    def test_cumulative_with_simulated_intervals(self):
        """Cumulative with simulated intervals produces interval columns."""
        r = _simple_model.predict(
            h=5, interval="simulated", cumulative=True, nsim=1000
        )
        assert r.shape == (5, 3)
        lower_col = [c for c in r.columns if c.startswith("lower")][0]
        upper_col = [c for c in r.columns if c.startswith("upper")][0]
        assert (r[lower_col].values <= r["mean"].values).all()
        assert (r["mean"].values <= r[upper_col].values).all()


# ---------------------------------------------------------------------------
# 5. Scenarios
# ---------------------------------------------------------------------------

class TestPredictScenarios:
    """Tests for scenario output storage."""

    def test_scenarios_stored(self):
        _simple_model.predict(
            h=5, interval="simulated", nsim=1000, scenarios=True
        )
        sc = _simple_model._general.get("_scenarios_matrix")
        assert sc is not None

    def test_scenarios_shape(self):
        _simple_model.predict(
            h=5, interval="simulated", nsim=1000, scenarios=True
        )
        sc = _simple_model._general["_scenarios_matrix"]
        assert sc.shape == (5, 1000)

    def test_scenarios_requires_simulated_warns(self):
        """scenarios=True with interval='approximate' issues a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _simple_model.predict(
                h=5, interval="approximate", scenarios=True
            )
            scenario_warnings = [
                x for x in w if "scenarios=True" in str(x.message)
            ]
            assert len(scenario_warnings) >= 1


# ---------------------------------------------------------------------------
# 6. Multiplicative model specifics
# ---------------------------------------------------------------------------

class TestPredictMultiplicative:
    """Tests specific to multiplicative models and simulation-based means."""

    def test_additive_same_across_intervals(self):
        """Additive model point forecasts are identical for none/approximate."""
        r_none = _additive_model.predict(h=5, interval="none")
        r_approx = _additive_model.predict(h=5, interval="approximate")
        np.testing.assert_array_almost_equal(
            r_none["mean"].values, r_approx["mean"].values, decimal=10
        )

    def test_multiplicative_approximate_produces_forecasts(self):
        """MAM with interval='approximate' produces valid finite forecasts."""
        r = _mult_model.predict(h=5, interval="approximate")
        assert np.all(np.isfinite(r["mean"].values))
        assert (r["mean"].values > 0).all()

    def test_multiplicative_simulated_produces_forecasts(self):
        """MAM with interval='simulated' produces valid finite forecasts."""
        r = _mult_model.predict(h=5, interval="simulated", nsim=1000)
        assert np.all(np.isfinite(r["mean"].values))
        assert (r["mean"].values > 0).all()


# ---------------------------------------------------------------------------
# 7. Combined model prediction
# ---------------------------------------------------------------------------

class TestPredictCombined:
    """Tests for predict on combined (CXC) models."""

    @pytest.fixture(scope="class")
    def combined_model(self):
        m = ADAM(model="CXC", lags=[1, 12])
        m.fit(_seasonal_data)
        return m

    def test_combined_predict_with_intervals(self, combined_model):
        r = combined_model.predict(h=10, interval="prediction")
        assert r.shape == (10, 3)
        assert "mean" in r.columns
        lower_col = [c for c in r.columns if c.startswith("lower")][0]
        upper_col = [c for c in r.columns if c.startswith("upper")][0]
        assert (r[lower_col].values <= r["mean"].values).all()
        assert (r["mean"].values <= r[upper_col].values).all()

    def test_combined_predict_none(self, combined_model):
        r = combined_model.predict(h=10, interval="none")
        assert r.shape == (10, 1)
        assert list(r.columns) == ["mean"]
        assert np.all(np.isfinite(r["mean"].values))
