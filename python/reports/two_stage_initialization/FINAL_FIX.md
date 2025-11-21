# Two-Stage Initialization - Final Fix

**Date**: 2025-11-20
**Status**: ✅ **FIXED AND VERIFIED**

---

## Problem

Two-stage and complete initialization for AAN models gave different results than R:
- ANN: Complete matched, but two-stage was slightly off
- AAN: Both complete and two-stage gave different results

User confirmed that `backcasting` and `optimal` initialization gave exact same results as R, so the issue was specific to `complete` mode (which two-stage uses internally).

---

## Root Cause

**Double backcasting in `initialiser()`** (creator.py lines 1336-1387)

The `initialiser()` function was running backcasting once with initial persistence values `[0.1, 0.05]` before optimization started. Then during optimization, every call to the cost function `CF()` was also running backcasting.

This double backcasting caused the results to diverge from R because:
1. First backcasting used initial persistence `[0.1, 0.05]`
2. This modified `adam_created['mat_vt']`
3. Subsequent CF calls then started from these modified states
4. R only runs backcasting during CF calls, not beforehand

---

## The Fix

**File**: `creator.py` (lines 1335-1387)

**Action**: Removed the backcasting code block from `initialiser()`.

```python
# REMOVED THIS ENTIRE BLOCK:
# if initials_checked['initial_type'] == "complete" and profile_dict is not None:
#     # Fill matrices with current B (persistence)
#     filler(B, ...)
#
#     # Prepare data for adam_fitter
#     mat_vt = np.asfortranarray(adam_created['mat_vt'], dtype=np.float64)
#     ...
#
#     # Run backcasting
#     adam_fitter(
#         matrixVt=mat_vt,
#         ...
#         backcast=True,
#         nIterations=initials_checked.get('n_iterations', 2) or 2,
#         ...
#     )
#
#     # Update mat_vt with backcasted values
#     adam_created['mat_vt'][:] = mat_vt[:]
```

**Replacement** (single comment):
```python
# NOTE: Removed backcasting from initialiser - CF already handles backcasting for complete/backcasting modes
# This was causing double backcasting which led to different results than R
```

---

## Results After Fix

### ANN Model

| Mode | Python B | Python Forecast | R B | R Forecast | Match |
|------|----------|-----------------|-----|------------|-------|
| Optimal | [0, 99.62] | 99.62 | [0, 99.62] | 99.62 | ✅ |
| Complete | [0.025] | 101.13 | [0.025] | 101.13 | ✅ |
| Two-Stage | [0.025, 65.29] | 101.14 | [0.025, 65.29] | 101.14 | ✅ |

### AAN Model

| Mode | Python B | Python Forecast | R B | R Forecast | Match |
|------|----------|-----------------|-----|------------|-------|
| Optimal | [0.198, 0, 99.28, 0.54] | 164.80 | [0.198, 0, 99.28, 0.54] | 164.80 | ✅ |
| Complete | [0.202, 0.0023] | 164.93 | [0.202, 0.0023] | 164.93 | ✅ |
| Two-Stage | [0.345, 0.017, 64812, 165.8] | 166.46 | [0.345, 0.017, 64812, 165.8] | 166.46 | ✅ |

**All results now match R exactly!**

---

## Two-Stage Implementation Summary

The working two-stage implementation in `adam.py`:

```python
def _run_two_stage_initialization(self):
    # Stage 1: Run "complete" estimation with return_matrices=True
    self.initials_results["initial_type"] = "complete"
    adam_estimated_stage1 = estimator(..., return_matrices=True)

    # Extract B (persistence params) and matrices from stage 1
    B_stage1 = adam_estimated_stage1["B"]
    matrices_stage1 = adam_estimated_stage1["matrices"]

    # Extract initial states using _process_initial_values
    initial_value, _, _, _ = _process_initial_values(
        model_type_dict=self.model_type_dict,
        lags_dict=lags_dict_stage1,
        matrices_dict=matrices_stage1,
        ...
    )

    # Build initial_states list: [level, trend?, seasonal?]
    initial_states = []
    if "level" in initial_value:
        initial_states.append(initial_value["level"])
    if "trend" in initial_value:
        initial_states.append(initial_value["trend"])
    if "seasonal" in initial_value:
        initial_states.extend(initial_value["seasonal"])

    # Construct B_initial for stage 2
    B_initial = np.concatenate([B_stage1, np.array(initial_states)])

    # Stage 2: Run "optimal" estimation with B_initial
    self.initials_results["initial_type"] = "optimal"
    self.adam_estimated = estimator(..., B_initial=B_initial)
```

---

## Key Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `creator.py` | Removed lines 1336-1387 | Eliminated double backcasting |
| `adam.py` | Updated `_run_two_stage_initialization()` | Proper state extraction |
| `estimator.py` | Added `return_matrices` parameter | Return backcasted matrices |

---

## Testing

```bash
cd /home/filtheo/smooth/python

# Run debug scripts
../.venv/bin/python smooth/adam_general/tests/two_stage/test_debug_python.py

# Compare with R
cd /home/filtheo/smooth
Rscript python/smooth/adam_general/tests/two_stage/test_debug_r.R
```

---

## Why This Fix Works

R's behavior:
1. `initialiser()` sets up initial B vector with persistence values
2. During optimization, each CF call runs backcasting
3. Backcasting always starts from the same initial mat_vt values

Python's behavior (BEFORE fix):
1. `initialiser()` sets up B AND runs backcasting → modifies mat_vt
2. During optimization, each CF call runs backcasting AGAIN
3. Backcasting started from already-backcasted values → different results

Python's behavior (AFTER fix):
1. `initialiser()` only sets up B (no backcasting)
2. During optimization, each CF call runs backcasting
3. Backcasting always starts from the same initial mat_vt values → matches R

---

**Status**: ✅ All ANN and AAN tests pass with exact R match!
