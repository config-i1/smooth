"""
Pytest configuration and shared fixtures for smooth package tests.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_series():
    """Simple linear time series with noise."""
    np.random.seed(42)
    n = 50
    t = np.arange(n)
    y = 10 + 0.5 * t + np.random.randn(n) * 2
    return y


@pytest.fixture
def seasonal_series():
    """Time series with trend and seasonality."""
    np.random.seed(42)
    n = 120  # 10 years of monthly data
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(n) * 3
    y = trend + seasonal + noise
    return y


@pytest.fixture
def multiplicative_series():
    """Time series with multiplicative seasonality."""
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
    noise = 1 + np.random.randn(n) * 0.05
    y = trend * seasonal * noise
    return y


@pytest.fixture
def short_series():
    """Short time series for edge case testing."""
    np.random.seed(42)
    return np.array([10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 16.0])


@pytest.fixture
def series_with_outliers():
    """Time series with outliers."""
    np.random.seed(42)
    n = 50
    t = np.arange(n)
    y = 10 + 0.5 * t + np.random.randn(n) * 2
    # Add outliers
    y[10] = 100
    y[30] = -50
    return y


@pytest.fixture
def airpassengers():
    """Classic AirPassengers dataset (monthly airline passengers 1949-1960)."""
    # First 48 values of the classic dataset
    return np.array([
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194
    ], dtype=float)
