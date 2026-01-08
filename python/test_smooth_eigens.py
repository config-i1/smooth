#!/usr/bin/env python3
"""
Test smooth_eigens function and ADAM with bounds='admissible'
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from smooth.adam_general.core.utils.utils import smooth_eigens
from smooth.adam_general.core.adam import ADAM

print("="*60)
print("Testing smooth_eigens function")
print("="*60)

# Test 1: Simple case without xreg
print("\n=== Test 1: Simple ETS model ===")
persistence = np.array([0.3])
transition = np.array([[1.0]])
measurement = np.tile([1.0], (100, 1))
lags_model_all = np.array([1])
obs_in_sample = 100

eigenvalues = smooth_eigens(
    persistence, transition, measurement,
    lags_model_all, xreg_model=False,
    obs_in_sample=obs_in_sample,
    has_delta_persistence=False
)

print(f"Persistence: {persistence}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Max abs eigenvalue: {np.max(np.abs(eigenvalues)):.4f}")
assert np.max(np.abs(eigenvalues)) < 1, "Eigenvalues should be < 1 for stable model"
print("✓ Test 1 passed!")

# Test 2: Model with multiple components
print("\n=== Test 2: Model with multiple lags ===")
persistence = np.array([0.3, 0.2, 0.1])
transition = np.eye(3)
measurement = np.ones((100, 3))
lags_model_all = np.array([1, 1, 12])
obs_in_sample = 100

eigenvalues = smooth_eigens(
    persistence, transition, measurement,
    lags_model_all, xreg_model=False,
    obs_in_sample=obs_in_sample,
    has_delta_persistence=False
)

print(f"Persistence: {persistence}")
print(f"Lags: {lags_model_all}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Max abs eigenvalue: {np.max(np.abs(eigenvalues)):.4f}")
assert len(eigenvalues) == len(lags_model_all), "Should return eigenvalue for each component"
print("✓ Test 2 passed!")

# Test 3: Model with xreg and adaptive persistence
print("\n=== Test 3: Model with adaptive xreg ===")
n_components = 4
persistence = np.array([0.3, 0.2, 0.1, 0.05])
transition = np.eye(n_components)
measurement = np.random.randn(100, n_components)
lags_model_all = np.array([1, 1, 12, 1])
obs_in_sample = 100

eigenvalues = smooth_eigens(
    persistence, transition, measurement,
    lags_model_all, xreg_model=True,
    obs_in_sample=obs_in_sample,
    has_delta_persistence=True
)

print(f"Number of eigenvalues: {len(eigenvalues)}")
print(f"Max abs eigenvalue: {np.max(np.abs(eigenvalues)):.4f}")
print("✓ Test 3 passed!")

print("\n" + "="*60)
print("Testing ADAM with bounds='admissible'")
print("="*60)

# Generate sample data
np.random.seed(42)
y_data = np.array([10, 12, 15, 13, 16, 18, 20, 19, 22, 25, 28, 30,
                   11, 13, 16, 14, 17, 19, 21, 20, 23, 26, 29, 31,
                   12, 14, 17, 15, 18, 20, 22, 21, 24, 27, 30, 32], dtype=float)

# Test 4: ADAM with ANN model and admissible bounds
print("\n=== Test 4: ADAM ANN with bounds='admissible' ===")
try:
    model = ADAM(model="ANN", lags=1)
    model.fit(y_data, bounds="admissible")
    print(f"Model fitted successfully")
    print(f"Alpha (persistence): {model.persistence_level_:.4f}")
    forecasts = model.predict(h=5)
    print(f"Forecasts: {forecasts}")
    print("✓ Test 4 passed!")
except Exception as e:
    print(f"✗ Test 4 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: ADAM with seasonal model and admissible bounds
print("\n=== Test 5: ADAM ANA with bounds='admissible' ===")
try:
    model = ADAM(model="ANA", lags=[1, 12])
    model.fit(y_data, bounds="admissible")
    print(f"Model fitted successfully")
    print(f"Alpha (level): {model.persistence_level_:.4f}")
    if hasattr(model, 'persistence_seasonal_'):
        print(f"Gamma (seasonal): {model.persistence_seasonal_:.4f}")
    forecasts = model.predict(h=5)
    print(f"Forecasts: {forecasts[:5]}")
    print("✓ Test 5 passed!")
except Exception as e:
    print(f"✗ Test 5 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: ADAM with trend model and admissible bounds
print("\n=== Test 6: ADAM AAN with bounds='admissible' ===")
try:
    model = ADAM(model="AAN", lags=1)
    model.fit(y_data, bounds="admissible")
    print(f"Model fitted successfully")
    print(f"Alpha (level): {model.persistence_level_:.4f}")
    if hasattr(model, 'persistence_trend_'):
        print(f"Beta (trend): {model.persistence_trend_:.4f}")
    forecasts = model.predict(h=5)
    print(f"Forecasts: {forecasts}")
    print("✓ Test 6 passed!")
except Exception as e:
    print(f"✗ Test 6 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: ADAM with ARIMA and admissible bounds
print("\n=== Test 7: ADAM with ARIMA(1,0,1) and bounds='admissible' ===")
try:
    model = ADAM(model="NNN", lags=1, orders={"ar": 1, "i": 0, "ma": 1})
    model.fit(y_data, bounds="admissible")
    print(f"Model fitted successfully")
    if hasattr(model, 'ar_'):
        print(f"AR coefficient: {model.ar_}")
    if hasattr(model, 'ma_'):
        print(f"MA coefficient: {model.ma_}")
    forecasts = model.predict(h=5)
    print(f"Forecasts: {forecasts}")
    print("✓ Test 7 passed!")
except Exception as e:
    print(f"✗ Test 7 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Compare admissible vs usual bounds
print("\n=== Test 8: Compare admissible vs usual bounds ===")
try:
    model_usual = ADAM(model="AAN", lags=1)
    model_usual.fit(y_data, bounds="usual")

    model_admissible = ADAM(model="AAN", lags=1)
    model_admissible.fit(y_data, bounds="admissible")

    print(f"Usual bounds - Alpha: {model_usual.persistence_level_:.4f}, Beta: {model_usual.persistence_trend_:.4f}")
    print(f"Admissible bounds - Alpha: {model_admissible.persistence_level_:.4f}, Beta: {model_admissible.persistence_trend_:.4f}")
    print("✓ Test 8 passed!")
except Exception as e:
    print(f"✗ Test 8 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
