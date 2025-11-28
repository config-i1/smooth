# Two-Stage Initialization Implementation Plan

**Overall Progress:** `100%` ✅ **COMPLETE**

---

## Overview

Implement "two-stage" initialization method for ADAM models in Python, matching R implementation. Two-stage works by:
1. First running a model with `initial="complete"` (full backcasting) to get good starting values
2. Using those values as initial guesses for optimization, allowing parameter refinement

---

## Tasks

- [x] 🟩 **Step 1: Add "two-stage" to valid initialization types**
  - [x] 🟩 Update `checker.py::_check_initial()` to accept "two-stage" string (line ~1103)
  - [x] 🟩 Update `adam.py::INITIAL_OPTIONS` type hint to include "two-stage" (line ~39-45)
  - [x] 🟩 Update `adam.py` docstring to document "two-stage" option (line ~154-162)

- [x] 🟩 **Step 2: Implement two-stage logic in ADAM.fit()**
  - [x] 🟩 Add two-stage detection in `adam.py::_execute_estimation()` (before calling estimator)
  - [x] 🟩 Create internal ADAM instance with `initial="complete"`, `silent=True`, `fast=True`
  - [x] 🟩 Fit the backcasting model and extract parameter vector B
  - [x] 🟩 Extract persistence parameters (alpha, beta, gamma, phi, AR/MA)
  - [x] 🟩 Extract and normalize seasonal initial states (if applicable)
  - [x] 🟩 Extract constant term (if used)
  - [x] 🟩 Extract distribution parameters (if applicable)
  - [x] 🟩 Pass extracted B to main estimator call

- [x] 🟩 **Step 3: Update estimator to accept pre-warmed parameters**
  - [x] 🟩 Modify `estimator.py::estimator()` to accept optional `B_initial` parameter
  - [x] 🟩 Use `B_initial` instead of `b_values["B"]` when provided
  - [x] 🟩 Ensure bounds (lb, ub) remain valid for the provided B

- [x] 🟩 **Step 4: Update parameter counting logic**
  - [x] 🟩 Treat "two-stage" like "optimal" in `creator.py::initialiser()` (line ~1612)
  - [x] 🟩 Ensure initials ARE counted as parameters to estimate (unlike backcasting/complete)
  - [x] 🟩 Verify parameter vector structure matches expectations

- [x] 🟩 **Step 5: Add tests for two-stage initialization**
  - [x] 🟩 Create test file `test_two_stage_python.ipynb` in `tests/two_stage/` folder
  - [x] 🟩 Test ETS(A,N,N) with two-stage vs optimal
  - [x] 🟩 Test ETS(A,A,N) with two-stage vs optimal
  - [x] 🟩 Test ETS(A,A,A) with seasonal two-stage
  - [x] 🟩 Test additional models (damped, multiplicative)

- [x] 🟩 **Step 6: Documentation and cleanup**
  - [x] 🟩 Add docstring examples showing two-stage usage
  - [x] 🟩 Update any relevant comments in code
  - [x] 🟩 Create comprehensive documentation (README, summary)

---

## Implementation Notes

### Key Design Decisions

1. **Location**: Implement two-stage logic in `adam.py::_execute_estimation()` before calling `estimator()` (separation of concerns)

2. **Recursion Prevention**: The nested ADAM call uses `initial="complete"`, not "two-stage", preventing infinite recursion

3. **Parameter Extraction Order** (matching R):
   - Persistence parameters (α, β, γ, φ)
   - ARMA parameters (AR/MA coefficients)
   - Initial states (level, trend, seasonal - with normalization)
   - Constant term
   - Distribution parameters

4. **Seasonal Normalization**:
   - Additive: Subtract mean from seasonal components
   - Multiplicative: Divide by geometric mean
   - Keep only first (m-1) seasonal values

5. **Scope**: Focus on ETS models initially, keep implementation simple and minimal

### Files Modified

- `python/smooth/adam_general/core/checker.py`
- `python/smooth/adam_general/core/adam.py`
- `python/smooth/adam_general/core/estimator.py`
- `python/smooth/adam_general/core/creator.py`

### Testing Strategy

Compare Python two-stage results with R two-stage results using same data, expect similar (not identical) parameter estimates and forecasts.

---

## Completion Criteria

- [x] ✅ "two-stage" accepted as valid `initial` parameter
- [x] ✅ Two-stage produces different results from pure backcasting and pure optimal
- [x] ✅ Implementation follows R logic (parameter extraction, normalization)
- [x] ✅ New tests demonstrate two-stage functionality
- [x] ✅ Comprehensive documentation provided

---

## ✅ Implementation Status: COMPLETE

All planned features have been successfully implemented. The two-stage initialization is now available for use in the Python ADAM module.

To use:
```python
model = ADAM(model='AAA', lags=[12], initial='two-stage', n_iterations=2)
model.fit(data)
forecasts = model.predict(h=12)
```
