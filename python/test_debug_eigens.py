import numpy as np
import sys
import os
import importlib.util

# Import smooth_eigens directly
spec = importlib.util.spec_from_file_location(
    "utils",
    os.path.join(os.path.dirname(__file__), "smooth/adam_general/core/utils/utils.py")
)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)
smooth_eigens = utils_module.smooth_eigens

print("Debugging Test 2:")
persistence = np.array([0.3, 0.2, 0.1])
transition = np.eye(3)
measurement = np.ones((100, 3))
lags_model_all = np.array([1, 1, 12])
obs_in_sample = 100

print(f"Persistence: {persistence}")
print(f"Transition:\n{transition}")
print(f"Lags: {lags_model_all}")
print(f"Unique lags: {np.unique(lags_model_all)}")
print(f"Measurement[99,:]: {measurement[99,:]}")

# Manual calculation for lag-1 components (indices 0, 1)
mask_lag1 = lags_model_all == 1
print(f"\nLag-1 mask: {mask_lag1}")
print(f"Lag-1 indices: {np.where(mask_lag1)[0]}")

transition_sub = transition[np.ix_(mask_lag1, mask_lag1)]
persistence_sub = persistence[mask_lag1].reshape(-1, 1)
measurement_sub = measurement[obs_in_sample - 1, mask_lag1].reshape(1, -1)

print(f"Transition subset (2x2):\n{transition_sub}")
print(f"Persistence subset:\n{persistence_sub}")
print(f"Measurement subset: {measurement_sub}")

matrix_to_decompose = transition_sub - persistence_sub @ measurement_sub
print(f"Matrix to decompose:\n{matrix_to_decompose}")
print(f"Eigenvalues: {np.linalg.eigvals(matrix_to_decompose)}")

# Manual calculation for lag-12 component (index 2)
mask_lag12 = lags_model_all == 12
print(f"\nLag-12 mask: {mask_lag12}")
transition_sub_12 = transition[np.ix_(mask_lag12, mask_lag12)]
persistence_sub_12 = persistence[mask_lag12].reshape(-1, 1)
measurement_sub_12 = measurement[obs_in_sample - 1, mask_lag12].reshape(1, -1)

print(f"Transition subset (1x1):\n{transition_sub_12}")
print(f"Persistence subset: {persistence_sub_12}")
print(f"Measurement subset: {measurement_sub_12}")

matrix_to_decompose_12 = transition_sub_12 - persistence_sub_12 @ measurement_sub_12
print(f"Matrix to decompose:\n{matrix_to_decompose_12}")
print(f"Eigenvalues: {np.linalg.eigvals(matrix_to_decompose_12)}")

# Now call smooth_eigens
print("\n" + "="*60)
eigenvalues = smooth_eigens(
    persistence, transition, measurement,
    lags_model_all, xreg_model=False,
    obs_in_sample=obs_in_sample,
    has_delta_persistence=False
)
print(f"smooth_eigens result: {eigenvalues}")
