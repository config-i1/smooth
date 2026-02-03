"""
Unit tests for the ADAM class.

Tests cover:
- Initialization and configuration
- Model fitting
- Prediction
- Different model types (ETS combinations)
"""

import numpy as np
import pytest

from smooth import ADAM


class TestADAMInitialization:
    """Tests for ADAM initialization."""

    def test_import(self):
        """Test that ADAM can be imported from smooth."""
        from smooth import ADAM
        assert ADAM is not None

    def test_basic_init(self):
        """Test basic initialization."""
        model = ADAM(model="ANN")
        assert model is not None

    def test_init_with_lags(self):
        """Test initialization with lags."""
        model = ADAM(model="ANA", lags=[12])
        assert model is not None
        assert 12 in model.lags


class TestADAMFit:
    """Tests for ADAM fitting."""

    def test_fit_basic(self, simple_series):
        """Test basic model fitting."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        # Model should have been fitted (adam_estimated contains results dict)
        assert model.adam_estimated is not None
        assert isinstance(model.adam_estimated, dict)
        assert 'B' in model.adam_estimated

    def test_fit_seasonal(self, seasonal_series):
        """Test fitting seasonal model."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        assert isinstance(model.adam_estimated, dict)

    def test_fit_returns_self(self, simple_series):
        """Test that fit returns self for chaining."""
        model = ADAM(model="ANN")
        result = model.fit(simple_series)

        assert result is model

    def test_fit_stores_data(self, simple_series):
        """Test that fit stores the training data."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        # y_in_sample should contain the training data
        assert hasattr(model, 'y_in_sample')
        assert len(model.y_in_sample) == len(simple_series)


class TestADAMPredict:
    """Tests for ADAM prediction."""

    def test_predict_basic(self, simple_series):
        """Test basic prediction."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        forecast = model.predict(h=10)

        # Should return DataFrame with 'mean' column
        assert hasattr(forecast, 'shape')
        assert forecast.shape[0] == 10
        assert 'mean' in forecast.columns

    def test_predict_seasonal(self, seasonal_series):
        """Test seasonal prediction."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(seasonal_series)

        forecast = model.predict(h=24)

        assert forecast.shape[0] == 24
        assert not forecast['mean'].isna().any()

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        model = ADAM(model="ANN")

        with pytest.raises((AttributeError, ValueError, RuntimeError, KeyError)):
            model.predict(h=10)

    def test_predict_includes_intervals(self, simple_series):
        """Test that predict includes prediction intervals."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        forecast = model.predict(h=10)

        # Should have lower and upper bounds
        cols = forecast.columns.tolist()
        assert any('lower' in c for c in cols)
        assert any('upper' in c for c in cols)


class TestADAMModelTypes:
    """Tests for different model types."""

    @pytest.mark.parametrize("model_code", [
        "ANN",  # Simple exponential smoothing
        "AAN",  # Holt's linear trend
        "AAdN", # Damped trend
    ])
    def test_ets_models_nonseasonal(self, simple_series, model_code):
        """Test non-seasonal ETS models."""
        model = ADAM(model=model_code)
        model.fit(simple_series)
        forecast = model.predict(h=5)

        assert forecast.shape[0] == 5
        assert not forecast['mean'].isna().any()

    @pytest.mark.parametrize("model_code", [
        "ANA",  # Additive seasonality
        "AAA",  # Trend + seasonality
    ])
    def test_ets_models_seasonal(self, seasonal_series, model_code):
        """Test seasonal ETS models."""
        model = ADAM(model=model_code, lags=[12])
        model.fit(seasonal_series)
        forecast = model.predict(h=12)

        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_multiplicative_error(self, multiplicative_series):
        """Test multiplicative error model."""
        model = ADAM(model="MNN")
        model.fit(multiplicative_series)
        forecast = model.predict(h=5)

        assert forecast.shape[0] == 5
        assert not forecast['mean'].isna().any()
        # Multiplicative error model forecasts should stay positive for positive data
        assert (forecast['mean'] > 0).all()


class TestADAMModelSelection:
    """Tests for automatic model selection."""

    def test_model_zzz(self, seasonal_series):
        """Test automatic model selection with ZZZ."""
        model = ADAM(model="ZZZ", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_zxz(self, seasonal_series):
        """Test automatic selection for error and seasonality (no trend) with ZXZ."""
        model = ADAM(model="ZXZ", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_fff(self, seasonal_series):
        """Test full model with FFF."""
        model = ADAM(model="FFF", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_ppp(self, seasonal_series):
        """Test partial automatic selection with PPP."""
        model = ADAM(model="PPP", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_zzz_nonseasonal(self, simple_series):
        """Test ZZZ model selection without seasonality."""
        model = ADAM(model="ZZZ", lags=[1])
        model.fit(simple_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=5)
        assert forecast.shape[0] == 5


class TestADAMEdgeCases:
    """Edge case tests for ADAM."""

    def test_short_series(self, short_series):
        """Test with short series."""
        model = ADAM(model="ANN")
        model.fit(short_series)
        forecast = model.predict(h=3)

        assert forecast.shape[0] == 3

    def test_series_with_zeros(self):
        """Test series containing zeros."""
        np.random.seed(42)
        y = np.abs(np.random.randn(50)) + 0.1
        y[10] = 0.001  # Near zero but not exactly zero
        y[20] = 0.001

        model = ADAM(model="ANN")
        model.fit(y)
        forecast = model.predict(h=5)

        assert not forecast['mean'].isna().any()

    def test_large_horizon(self, simple_series):
        """Test prediction with large horizon."""
        model = ADAM(model="ANN")
        model.fit(simple_series)
        forecast = model.predict(h=100)

        assert forecast.shape[0] == 100
        assert not forecast['mean'].isna().any()


class TestADAMAttributes:
    """Tests for ADAM attributes after fitting."""

    def test_persistence_level_attribute(self, simple_series):
        """Test that persistence level (alpha) is accessible."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        # Should have persistence_level_ (alpha parameter)
        assert hasattr(model, 'persistence_level_')
        # persistence_level_ may be None for some models, check if numeric
        if model.persistence_level_ is not None:
            assert 0 <= model.persistence_level_ <= 1

    def test_persistence_trend_attribute(self, simple_series):
        """Test that persistence trend (beta) is accessible for trend models."""
        model = ADAM(model="AAN")
        model.fit(simple_series)

        assert hasattr(model, 'persistence_trend_')

    def test_phi_attribute(self, simple_series):
        """Test that phi (damping) is accessible for damped models."""
        model = ADAM(model="AAdN")
        model.fit(simple_series)

        assert hasattr(model, 'phi_')


class TestADAMBounds:
    """Tests for parameter bounds."""

    def test_admissible_bounds_linear_series(self):
        """Test admissible bounds with linear series.

        For a linear series (1 to 20), ETS(ANN) with admissible bounds
        should produce a smoothing parameter (alpha) greater than 1,
        which is outside the usual [0,1] bounds but still admissible.
        """
        y = np.arange(1, 21, dtype=float)
        model = ADAM(model="ANN", bounds="admissible")
        model.fit(y)

        assert model.persistence_level_ > 1, (
            f"Expected alpha > 1 for linear series with admissible bounds, "
            f"got {model.persistence_level_}"
        )


class TestADAMReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_same_result(self, simple_series):
        """Test that same random seed gives same result."""
        np.random.seed(42)
        model1 = ADAM(model="ANN")
        model1.fit(simple_series)
        forecast1 = model1.predict(h=5)

        np.random.seed(42)
        model2 = ADAM(model="ANN")
        model2.fit(simple_series)
        forecast2 = model2.predict(h=5)

        np.testing.assert_array_almost_equal(
            forecast1['mean'].values,
            forecast2['mean'].values
        )
