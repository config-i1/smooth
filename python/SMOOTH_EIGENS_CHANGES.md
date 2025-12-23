# smooth_eigens Implementation Summary

## Changes Made

### 1. Created `smooth_eigens()` function
**File**: `smooth/adam_general/core/utils/utils.py` (lines 692-716)

Python translation of R's `smoothEigens()` function that calculates eigenvalues for ADAM model stability checking.

**Function signature**:
```python
def smooth_eigens(persistence, transition, measurement,
                  lags_model_all, xreg_model, obs_in_sample,
                  has_delta_persistence=False)
```

**Key features**:
- Calculates eigenvalues per unique lag component (standard case)
- Uses averaged calculation for models with adaptive xreg persistence
- Returns eigenvalue array matching component structure

### 2. Updated cost function to use `smooth_eigens()`
**File**: `smooth/adam_general/core/utils/cost_functions.py`

**Changes**:
- Added `smooth_eigens` to imports (line 4)
- Replaced manual eigenvalue calculation with `smooth_eigens()` call (lines 344-356)
- Commented out old eigenvalue calculation code for reference (lines 358-385)
- Simplified logic - no longer needs separate cases for xreg/non-xreg models

**Benefits**:
- Centralized eigenvalue calculation (matches R package refactoring)
- Cleaner, more maintainable code
- Consistent with R implementation

### 3. Updated Python version constraint
**File**: `pyproject.toml` (line 10)

Changed from `requires-python = ">=3.10,<=3.13.3"` to `requires-python = ">=3.10"`
- Removed unnecessary upper bound restriction
- Allows installation on newer Python versions

## Testing

Created comprehensive test suite (`test_smooth_eigens_simple.py`) covering:
- ✓ Simple ETS models (single component)
- ✓ Multiple components with different lags
- ✓ Adaptive xreg models
- ✓ Unstable models (correctly identified)
- ✓ Seasonal models (12 components)
- ✓ Edge cases (very small persistence)

All tests passed successfully!

## Technical Notes

**R vs Python differences**:
- R uses 1-based indexing: `measurement[obsInSample,]` 
- Python uses 0-based: `measurement[obs_in_sample-1,:]`
- Added `has_delta_persistence` parameter to replace R's name checking: `any(substr(names(persistence),1,5)=="delta")`

**Eigenvalue calculation**:
- For standard models: Calculates per unique lag using `F - g @ w`
- For adaptive xreg: Uses averaged condition `F - diag(g) @ inv(W)^T @ W / T`
- Uses `np.abs()` in cost function to handle potential complex eigenvalues

