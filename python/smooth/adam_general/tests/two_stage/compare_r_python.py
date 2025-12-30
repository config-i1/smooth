#!/usr/bin/env python
"""
Compare R and Python MAM model results for all initialization methods.
"""
import sys
sys.path.insert(0, '/home/filtheo/smooth/python')

import subprocess
import numpy as np
import pandas as pd

# Load test data
test_data = pd.read_csv('/home/filtheo/smooth/python/test_data.csv')
y = test_data['value'].values

from smooth.adam_general.core.adam import ADAM

print("=" * 80)
print("Comparison: Python vs R for MAM model")
print("=" * 80)

# Python results
print("\n[PYTHON RESULTS]")
print("-" * 40)

init_methods = ["optimal", "backcasting", "complete", "two-stage"]
python_results = {}

for method in init_methods:
    print(f"\n  {method}:")
    try:
        model = ADAM(model="MAM", lags=[12], initial=method, verbose=0)
        model.fit(y)
        B = model.adam_estimated["B"]
        python_results[method] = {
            'alpha': B[0],
            'beta': B[1],
            'gamma': B[2],
        }
        print(f"    alpha={B[0]:.6f}, beta={B[1]:.6f}, gamma={B[2]:.10f}")
        # Also print CF value if available
        print(f"    CF value: {model.adam_estimated.get('CF_value', 'N/A')}")
    except Exception as e:
        print(f"    Error: {e}")

# Now run R and get its results
print("\n\n[R RESULTS]")
print("-" * 40)

r_script = """
library(smooth)

# Read the data
data <- read.csv("/home/filtheo/smooth/python/test_data.csv")
y <- data$value

# Test different initialization methods
methods <- c("optimal", "backcasting", "complete", "two-stage")

for (method in methods) {
    cat(sprintf("\\n  %s:\\n", method))
    tryCatch({
        model <- adam(y, model="MAM", lags=12, initial=method, silent=TRUE)
        B <- model$B[1:3]  # Get persistence parameters
        cat(sprintf("    alpha=%.6f, beta=%.6f, gamma=%.10f\\n", B[1], B[2], B[3]))
        cat(sprintf("    log-likelihood: %.4f\\n", model$logLik))
    }, error = function(e) {
        cat(sprintf("    Error: %s\\n", e$message))
    })
}
"""

# Run R script
result = subprocess.run(
    ["Rscript", "-e", r_script],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("R stderr:", result.stderr[:500] if len(result.stderr) > 500 else result.stderr)

print("\n" + "=" * 80)
print("KEY COMPARISON")
print("=" * 80)

print("\n  Method       | Python gamma   | R gamma (from output above)")
print("  " + "-" * 55)
for method in init_methods:
    if method in python_results:
        print(f"  {method:<12} | {python_results[method]['gamma']:.10f} |")
