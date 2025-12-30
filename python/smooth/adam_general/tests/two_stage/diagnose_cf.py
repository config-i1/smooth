#!/usr/bin/env python
"""
Diagnostic script to understand why MAM two-stage gamma converges to 0 instead of ~0.18
"""
import sys
sys.path.insert(0, '/home/filtheo/smooth/python')

import numpy as np
import pandas as pd

# Load test data
test_data = pd.read_csv('/home/filtheo/smooth/python/test_data.csv')
y = test_data['value'].values

from smooth.adam_general.core.adam import ADAM

print("=" * 80)
print("DIAGNOSTIC: MAM model gamma sensitivity analysis")
print("=" * 80)

# First, let's fit the optimal model and examine the fitted model internals
print("\n[1] Fitting optimal model to get internal structures...")
model = ADAM(model="MAM", lags=[12], initial="optimal", verbose=0)
model.fit(y)
B_opt = model.adam_estimated["B"]
print(f"    Optimal B: {B_opt}")
print(f"    Optimal persistence: alpha={B_opt[0]:.6f}, beta={B_opt[1]:.6f}, gamma={B_opt[2]:.10f}")

# Get the internal structures needed for CF evaluation
from smooth.adam_general.core.utils.cost_functions import CF
from smooth.adam_general.core.creator import filler

# We have access to the fitted model's internal structures
print("\n[2] Evaluating CF at different gamma values...")

# Use the optimal B as base and vary gamma
def evaluate_cf(model, B_test):
    """Evaluate CF using model's internal structures"""
    # Make copies of matrices
    matrices_dict = {
        'mat_vt': model.adam_created['mat_vt'].copy(),
        'mat_wt': model.adam_created['mat_wt'].copy(),
        'mat_f': model.adam_created['mat_f'].copy(),
        'vec_g': model.adam_created['vec_g'].copy(),
        'arima_polynomials': model.adam_created.get('arima_polynomials', None),
    }

    cf_val = CF(
        B_test,
        model.model_type_dict,
        model.components_dict,
        model.lags_dict,
        matrices_dict,
        model.persistence_results,
        model.initials_results,
        model.arima_results,
        model.explanatory_dict,
        model.phi_dict,
        model.constant_dict,
        model.observations_dict,
        model.profile_dict,
        model.general,
        bounds=model.bounds,
    )
    return cf_val

# Test different gamma values
gamma_values = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.18, 0.2, 0.25, 0.3]

print(f"\n    Base parameters:")
print(f"    alpha={B_opt[0]:.6f}, beta={B_opt[1]:.6f}")
print(f"    Initial states: level={B_opt[3]:.2f}, trend={B_opt[4]:.4f}")
print(f"    Seasonal: {B_opt[5:]}")
print()
print(f"    {'Gamma':<10} {'CF Value':<20} {'Note':<30}")
print(f"    {'-'*10} {'-'*20} {'-'*30}")

for gamma in gamma_values:
    B_test = B_opt.copy()
    B_test[2] = gamma
    cf_val = evaluate_cf(model, B_test)
    note = ""
    if cf_val >= 1e100:
        note = "PENALTY (constraint violation)"
    elif abs(gamma - B_opt[2]) < 0.001:
        note = "<-- optimal gamma"
    print(f"    {gamma:<10.4f} {cf_val:<20.6f} {note}")

# Now check what R's two-stage gamma would give us
print(f"\n[3] Testing R's two-stage gamma value (0.18046)...")
B_r_gamma = B_opt.copy()
B_r_gamma[2] = 0.18046
cf_r = evaluate_cf(model, B_r_gamma)
print(f"    CF at gamma=0.18046 (R's two-stage result): {cf_r:.6f}")
cf_opt = evaluate_cf(model, B_opt)
print(f"    CF at gamma={B_opt[2]:.6f} (Python's optimal result): {cf_opt:.6f}")

# Check constraint: gamma <= 1 - alpha
max_gamma = 1 - B_opt[0]
print(f"\n    Constraint check: gamma <= 1 - alpha = 1 - {B_opt[0]:.6f} = {max_gamma:.6f}")
print(f"    R's gamma (0.18046) satisfies constraint: {0.18046 <= max_gamma}")

# Let's also compare with Stage 1 results
print("\n[4] Stage 1 (complete) analysis...")
stage1 = ADAM(model="MAM", lags=[12], initial="complete", fast=True, verbose=0)
stage1.fit(y)
B_stage1 = stage1.adam_estimated["B"]
print(f"    Stage 1 B: {B_stage1}")
print(f"    Stage 1 gamma: {B_stage1[2]:.6f}")

# Let's also check two-stage
print("\n[5] Two-stage analysis...")
twostage = ADAM(model="MAM", lags=[12], initial="two-stage", verbose=0)
twostage.fit(y)
B_twostage = twostage.adam_estimated["B"]
print(f"    Two-stage B: {B_twostage}")
print(f"    Two-stage gamma: {B_twostage[2]:.6f}")

# Compare CF values
print("\n[6] Comparing CF values at converged solutions...")
print(f"    Optimal gamma={B_opt[2]:.6f}: CF = {cf_opt:.6f}")
print(f"    Complete gamma={B_stage1[2]:.6f}: CF = {evaluate_cf(model, np.concatenate([B_stage1, B_opt[3:]])):.6f}")
print(f"    Two-stage gamma={B_twostage[2]:.6f}: CF = {evaluate_cf(model, B_twostage):.6f}")
print(f"    R's two-stage gamma=0.18046: CF = {cf_r:.6f}")

print("\n[7] Key insight: Why does Python prefer gamma~0 while R prefers gamma~0.18?")
print("    Let's check if there's a difference in the seasonal initial values...")

# Get the initial seasonal values from both models
print("\n    Seasonal initial values:")
print(f"    Optimal: {B_opt[5:]}")
print(f"    Two-stage: {B_twostage[5:]}")

# Check if the difference in initial values explains the preference for gamma
print("\n[8] Testing: What if we use R's gamma but Python's other parameters?")
# If we force gamma=0.18 with Python's initial states, is CF worse than gamma=0.001?
B_forced_high_gamma = B_opt.copy()
B_forced_high_gamma[2] = 0.18
cf_forced = evaluate_cf(model, B_forced_high_gamma)
print(f"    CF with forced gamma=0.18: {cf_forced:.6f}")
print(f"    CF with optimal gamma={B_opt[2]:.6f}: {cf_opt:.6f}")
print(f"    Difference: {cf_forced - cf_opt:.6f}")
if cf_forced > cf_opt:
    print("    => Python correctly prefers lower gamma with its seasonal values")
else:
    print("    => Higher gamma would be better but optimizer didn't find it")
