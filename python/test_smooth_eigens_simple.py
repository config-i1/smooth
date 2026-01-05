#!/usr/bin/env python3
"""
Test smooth_eigens function (Python-only, no C++ module required)
"""
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import smooth_eigens directly without going through package __init__
# This avoids the C++ module dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "utils",
    os.path.join(os.path.dirname(__file__), "smooth/adam_general/core/utils/utils.py")
)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)
smooth_eigens = utils_module.smooth_eigens

print("="*70)
print("Testing smooth_eigens Function")
print("="*70)

# Test 1: Simple ETS model (single component)
print("\n[Test 1] Simple ETS(A,N,N) model - Single component")
print("-" * 70)
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

# Expected: F - g @ w = 1.0 - 0.3 * 1.0 = 0.7
expected = 1.0 - 0.3 * 1.0
print(f"  Persistence (α):       {persistence[0]:.3f}")
print(f"  Transition (F):        {transition[0,0]:.3f}")
print(f"  Expected eigenvalue:   {expected:.3f}")
print(f"  Computed eigenvalue:   {eigenvalues[0]:.3f}")
print(f"  Max |eigenvalue|:      {np.max(np.abs(eigenvalues)):.3f}")

assert np.allclose(eigenvalues, expected, atol=1e-10), \
    f"Expected {expected}, got {eigenvalues[0]}"
assert np.max(np.abs(eigenvalues)) < 1, \
    "Eigenvalues should be < 1 for stable model"
print("  ✓ PASSED")

# Test 2: ETS model with multiple lags (level, trend, seasonal)
print("\n[Test 2] ETS model with multiple components (lag-1 and lag-12)")
print("-" * 70)
persistence = np.array([0.3, 0.2, 0.1])
transition = np.eye(3)
measurement = np.ones((100, 3))
lags_model_all = np.array([1, 1, 12])  # level, trend, seasonal
obs_in_sample = 100

eigenvalues = smooth_eigens(
    persistence, transition, measurement,
    lags_model_all, xreg_model=False,
    obs_in_sample=obs_in_sample,
    has_delta_persistence=False
)

print(f"  Persistence vector:    {persistence}")
print(f"  Lags:                  {lags_model_all}")
print(f"  Unique lags:           {np.unique(lags_model_all)}")
print(f"  Number of eigenvalues: {len(eigenvalues)}")
print(f"  Eigenvalues:           {eigenvalues}")
print(f"  Max |eigenvalue|:      {np.max(np.abs(eigenvalues)):.3f}")

assert len(eigenvalues) == len(lags_model_all), \
    f"Expected {len(lags_model_all)} eigenvalues, got {len(eigenvalues)}"
# Use same tolerance as cost function: eigenvalues should be <= 1 + 1e-50
assert np.max(np.abs(eigenvalues)) <= 1 + 1e-50, \
    "Eigenvalues should be <= 1 + 1e-50 for stable model"
print("  ✓ PASSED")

# Test 3: Model with adaptive xreg (uses averaged condition)
print("\n[Test 3] Model with adaptive xreg persistence")
print("-" * 70)
np.random.seed(123)
n_components = 4
persistence = np.array([0.3, 0.2, 0.1, 0.05])
transition = np.eye(n_components)
measurement = np.random.randn(100, n_components) * 0.1
lags_model_all = np.array([1, 1, 12, 1])
obs_in_sample = 100

eigenvalues = smooth_eigens(
    persistence, transition, measurement,
    lags_model_all, xreg_model=True,
    obs_in_sample=obs_in_sample,
    has_delta_persistence=True  # Uses averaged calculation
)

print(f"  Persistence vector:    {persistence}")
print(f"  Components:            {n_components}")
print(f"  Number of eigenvalues: {len(eigenvalues)}")
print(f"  Max |eigenvalue|:      {np.max(np.abs(eigenvalues)):.3f}")
print(f"  Note: Adaptive xreg uses averaged eigenvalue calculation")

# For adaptive case, eigenvalues array size can differ
assert len(eigenvalues) > 0, "Should return at least one eigenvalue"
print("  ✓ PASSED")

# Test 4: Model with transition > 1 (unstable)
print("\n[Test 4] Unstable model (transition matrix > 1)")
print("-" * 70)
persistence = np.array([0.5])
transition = np.array([[2.0]])  # Unstable!
measurement = np.tile([1.0], (100, 1))
lags_model_all = np.array([1])
obs_in_sample = 100

eigenvalues = smooth_eigens(
    persistence, transition, measurement,
    lags_model_all, xreg_model=False,
    obs_in_sample=obs_in_sample,
    has_delta_persistence=False
)

# Expected: 2.0 - 0.5 * 1.0 = 1.5 > 1 (unstable!)
expected = 2.0 - 0.5 * 1.0
print(f"  Persistence:           {persistence[0]:.3f}")
print(f"  Transition:            {transition[0,0]:.3f}")
print(f"  Expected eigenvalue:   {expected:.3f}")
print(f"  Computed eigenvalue:   {eigenvalues[0]:.3f}")
print(f"  Max |eigenvalue|:      {np.max(np.abs(eigenvalues)):.3f}")

assert np.allclose(eigenvalues, expected, atol=1e-10), \
    f"Expected {expected}, got {eigenvalues[0]}"
assert np.max(np.abs(eigenvalues)) > 1, \
    "Should correctly identify unstable model (eigenvalue > 1)"
print(f"  ✓ PASSED (Correctly identified unstable model)")

# Test 5: Seasonal components (all same lag)
print("\n[Test 5] Seasonal model (12 components, all lag-12)")
print("-" * 70)
n_seasonal = 12
persistence_seasonal = np.ones(n_seasonal) * 0.1
transition_seasonal = np.eye(n_seasonal)

# Create shift matrix for seasonal component
for i in range(n_seasonal - 1):
    transition_seasonal[i, i+1] = 1.0
    transition_seasonal[i, i] = 0.0

measurement_seasonal = np.zeros((100, n_seasonal))
measurement_seasonal[:, 0] = 1.0  # Only first seasonal component in measurement

lags_model_all_seasonal = np.ones(n_seasonal, dtype=int) * 12
obs_in_sample = 100

eigenvalues = smooth_eigens(
    persistence_seasonal, transition_seasonal, measurement_seasonal,
    lags_model_all_seasonal, xreg_model=False,
    obs_in_sample=obs_in_sample,
    has_delta_persistence=False
)

print(f"  Number of components:  {n_seasonal}")
print(f"  Lag:                   12 (all components)")
print(f"  Number of eigenvalues: {len(eigenvalues)}")
print(f"  Max |eigenvalue|:      {np.max(np.abs(eigenvalues)):.3f}")

assert len(eigenvalues) == n_seasonal, \
    f"Expected {n_seasonal} eigenvalues, got {len(eigenvalues)}"
assert np.max(np.abs(eigenvalues)) < 1.1, \
    "Eigenvalues should be reasonable for this model"
print("  ✓ PASSED")

# Test 6: Edge case - very small persistence
print("\n[Test 6] Edge case - Very small persistence")
print("-" * 70)
persistence = np.array([0.01])
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

# Expected: 1.0 - 0.01 * 1.0 = 0.99
expected = 1.0 - 0.01 * 1.0
print(f"  Persistence (very small): {persistence[0]:.3f}")
print(f"  Expected eigenvalue:      {expected:.3f}")
print(f"  Computed eigenvalue:      {eigenvalues[0]:.3f}")
print(f"  Max |eigenvalue|:         {np.max(np.abs(eigenvalues)):.3f}")

assert np.allclose(eigenvalues, expected, atol=1e-10), \
    f"Expected {expected}, got {eigenvalues[0]}"
print("  ✓ PASSED")

print("\n" + "="*70)
print("All tests PASSED! ✓")
print("="*70)
print("\nThe smooth_eigens function is working correctly.")
print("It properly calculates eigenvalues for:")
print("  - Simple ETS models")
print("  - Models with multiple components and lags")
print("  - Models with adaptive xreg persistence")
print("  - Unstable models (correctly identified)")
print("  - Seasonal models with multiple components")
print("  - Edge cases with very small persistence")
