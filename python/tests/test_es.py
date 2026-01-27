"""
Unit tests for the ES (Exponential Smoothing) class.

Tests cover:
- Initialization
- Model fitting
- Prediction
"""

import numpy as np
import pytest

from smooth import ES


class TestESInitialization:
    """Tests for ES initialization."""

    def test_import(self):
        """Test that ES can be imported from smooth."""
        from smooth import ES
        assert ES is not None

    def test_basic_init(self):
        """Test basic initialization."""
        model = ES(model="ANN")
        assert model is not None

    def test_init_with_lags(self):
        """Test initialization with seasonal lags."""
        model = ES(model="ANA", lags=[12])
        assert model is not None


class TestESFit:
    """Tests for ES fitting."""

    def test_fit_basic(self, simple_series):
        """Test basic model fitting."""
        model = ES(model="ANN")
        model.fit(simple_series)

        # Model should have been fitted (adam_estimated contains results dict)
        assert model.adam_estimated is not None
        assert isinstance(model.adam_estimated, dict)

    def test_fit_returns_self(self, simple_series):
        """Test that fit returns self."""
        model = ES(model="ANN")
        result = model.fit(simple_series)

        assert result is model

    def test_fit_seasonal(self, seasonal_series):
        """Test fitting seasonal ES model."""
        model = ES(model="ANA", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        assert isinstance(model.adam_estimated, dict)


class TestESPredict:
    """Tests for ES prediction."""

    def test_predict_basic(self, simple_series):
        """Test basic prediction."""
        model = ES(model="ANN")
        model.fit(simple_series)

        forecast = model.predict(h=10)

        assert forecast.shape[0] == 10
        assert 'mean' in forecast.columns
        assert not forecast['mean'].isna().any()

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        model = ES(model="ANN")

        with pytest.raises((AttributeError, ValueError, RuntimeError, KeyError)):
            model.predict(h=10)


class TestESModelTypes:
    """Tests for different ES model types."""

    @pytest.mark.parametrize("model_code", [
        "ANN",  # Simple
        "AAN",  # Holt
        "AAdN", # Damped
    ])
    def test_nonseasonal_models(self, simple_series, model_code):
        """Test non-seasonal ES models."""
        model = ES(model=model_code)
        model.fit(simple_series)
        forecast = model.predict(h=5)

        assert forecast.shape[0] == 5

    @pytest.mark.parametrize("model_code", [
        "ANA",  # Additive seasonal
        "AAA",  # Holt-Winters additive
    ])
    def test_seasonal_models(self, seasonal_series, model_code):
        """Test seasonal ES models."""
        model = ES(model=model_code, lags=[12])
        model.fit(seasonal_series)
        forecast = model.predict(h=12)

        assert forecast.shape[0] == 12


class TestESModelSelection:
    """Tests for automatic model selection."""

    def test_model_zzz(self, seasonal_series):
        """Test automatic model selection with ZZZ."""
        model = ES(model="ZZZ", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_zxz(self, seasonal_series):
        """Test automatic selection for error and seasonality (no trend) with ZXZ."""
        model = ES(model="ZXZ", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_fff(self, seasonal_series):
        """Test full model with FFF."""
        model = ES(model="FFF", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_ppp(self, seasonal_series):
        """Test partial automatic selection with PPP."""
        model = ES(model="PPP", lags=[12])
        model.fit(seasonal_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_zzz_nonseasonal(self, simple_series):
        """Test ZZZ model selection without seasonality."""
        model = ES(model="ZZZ", lags=[1])
        model.fit(simple_series)

        assert model.adam_estimated is not None
        forecast = model.predict(h=5)
        assert forecast.shape[0] == 5


class TestESEdgeCases:
    """Edge case tests for ES."""

    def test_short_series(self, short_series):
        """Test with short series."""
        model = ES(model="ANN")
        model.fit(short_series)
        forecast = model.predict(h=3)

        assert forecast.shape[0] == 3

    def test_constant_series(self):
        """Test with constant series."""
        y = np.ones(30) * 50

        model = ES(model="ANN")
        model.fit(y)
        forecast = model.predict(h=5)

        # Forecast mean should be close to 50
        assert np.all(np.abs(forecast['mean'].values - 50) < 5)
