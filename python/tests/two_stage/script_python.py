# %%
%load_ext autoreload
%autoreload 2

# %%
from smooth.adam_general.core.adam import ADAM
import numpy as np
import pandas as pd

# %% [markdown]
# # Two-Stage Initialization Tests
# 
# Two-stage initialization works by:
# 1. First running a model with `initial="complete"` (full backcasting) to get good starting values
# 2. Using those values as initial gueand sses for optimization, allowing parameter refinement
# 
# This should produce results that are:
# - Different from pure backcasting (since parameters are refined)
# - Different from pure optimal (since starting values are better)
# - Generally better or similar quality to both methods

# %% [markdown]
# ### Test 1: Global level ETS(A,N,N) - two-stage vs optimal vs backcasting

# %%
np.random.seed(33)
n_points = 100
time_series = np.random.normal(100, 10, n_points)
ts_df = pd.DataFrame({'value': time_series}, index=pd.date_range(start='2023-01-01', periods=n_points, freq='ME'))

# %%
model_optimal = ADAM(model='ANN', lags=[12], initial='optimal')
model_optimal.fit(ts_df)
print('ETS(A,N,N) with optimal initial:')
print('Parameters:', model_optimal.adam_estimated['B'])
print('Forecast:', model_optimal.predict(h=12)['mean'].values[:3])

# %%
model_backcasting = ADAM(model='ANN', lags=[12], initial='backcasting', n_iterations=2)
model_backcasting.fit(ts_df)
print('ETS(A,N,N) with backcasting (n_iterations=2):')
print('Parameters:', model_backcasting.adam_estimated['B'])
print('Forecast:', model_backcasting.predict(h=12)['mean'].values[:3])

# %%
model_two_stage = ADAM(model='ANN', lags=[12], initial='complete', n_iterations=2)
model_two_stage.fit(ts_df)
print('ETS(A,N,N) with complete initialization:')
print('Parameters:', model_two_stage.adam_estimated['B'])
print('Forecast:', model_two_stage.predict(h=12)['mean'].values[:3])

# %%
model_two_stage = ADAM(model='ANN', lags=[12], initial='two-stage', n_iterations=2)
model_two_stage.fit(ts_df)
print('ETS(A,N,N) with two-stage initialization:')
print('Parameters:', model_two_stage.adam_estimated['B'])
print('Forecast:', model_two_stage.predict(h=12)['mean'].values[:3])

# %% [markdown]
# ### Test 2: Local trend ETS(A,A,N)

# %%
np.random.seed(42)
n_points = 120
errors = np.random.normal(0, 10, n_points)
trend = np.random.normal(0.5, 2, n_points)
time_series = np.zeros(n_points)
time_series[0] = 100
for i in range(n_points-1):
    time_series[i+1] = time_series[i] + (0.1-1) * errors[i] + trend[i] + errors[i+1]
ts_df = pd.DataFrame({'value': time_series}, index=pd.date_range(start='2023-01-01', periods=n_points, freq='ME'))

# %%
model_optimal = ADAM(model='AAN', lags=[12], initial='optimal')
model_optimal.fit(ts_df)
print('ETS(A,A,N) with optimal initial:')
print('Parameters:', model_optimal.adam_estimated['B'])
print('Forecast:', model_optimal.predict(h=12)['mean'].values[:3])

# %%
model_optimal = ADAM(model='AAN', lags=[12], initial='complete')
model_optimal.fit(ts_df)
print('ETS(A,A,N) with complete initial:')
print('Parameters:', model_optimal.adam_estimated['B'])
print('Forecast:', model_optimal.predict(h=12)['mean'].values[:3])

# %%
model_optimal = ADAM(model='AAN', lags=[12], initial='complete')
model_optimal.fit(ts_df)
print('ETS(A,A,N) with complete initial:')
print('Parameters:', model_optimal.adam_estimated['B'])
print('Forecast:', model_optimal.predict(h=12)['mean'].values[:3])

# %%
model_two_stage = ADAM(model='AAN', lags=[12], initial='two-stage', n_iterations=2)
model_two_stage.fit(ts_df)
print('ETS(A,A,N) with two-stage initialization:')
print('Parameters:', model_two_stage.adam_estimated['B'])
print('Forecast:', model_two_stage.predict(h=12)['mean'].values[:3])

# %% [markdown]
# ### Test 3: Seasonal data ETS(A,A,A)

# %%
np.random.seed(42)
n_points = 120
errors = (1+np.random.normal(0, 0.1, n_points))
trend = np.random.normal(0.5, 2, n_points)
seasonal_sd = 0.2
seasonal_pattern = np.exp(np.random.normal(0, seasonal_sd, 12))
seasonal_pattern = seasonal_pattern / np.mean(seasonal_pattern)
time_series = np.zeros(n_points)
time_series[0] = 200 * seasonal_pattern[0] * errors[0]
for i in range(n_points-1):
    time_series[i+1] = ((time_series[i] / seasonal_pattern[(i) % 12]-trend[i]) * errors[i] ** (0.1-1) + trend[i+1]) * seasonal_pattern[(i+1) % 12] * errors[i+1]
ts_df = pd.DataFrame({'value': time_series}, index=pd.date_range(start='2023-01-01', periods=n_points, freq='ME'))

# %%
model_optimal = ADAM(model='AAA', lags=[12], distribution='dnorm', initial='optimal')
model_optimal.fit(ts_df)
print('ETS(A,A,A) with optimal initial:')
print('Alpha, Beta, Gamma:', model_optimal.adam_estimated['B'][:3])
print('Forecast:', model_optimal.   predict(h=12)['mean'].values[:3])

# %%
model_backcasting = ADAM(model='AAA', lags=[12], distribution='dnorm', initial='backcasting')
model_backcasting.fit(ts_df)
print('ETS(A,A,A) with backcasting initial:')
print('Alpha, Beta, Gamma:', model_backcasting.adam_estimated['B'][:3])
print('Forecast:', model_backcasting.   predict(h=12)['mean'].values[:3])

# %%
model_complete = ADAM(model='AAA', lags=[12], distribution='dnorm', initial='complete')
model_complete.fit(ts_df)
print('ETS(A,A,A) with complete initial:')
print('Alpha, Beta, Gamma:', model_complete.adam_estimated['B'][:3])
print('Forecast:', model_complete.   predict(h=12)['mean'].values[:3])

# %%
model_two_stage = ADAM(model='AAA', lags=[12], distribution='dnorm', initial='two-stage', n_iterations=2)
model_two_stage.fit(ts_df)
print('ETS(A,A,A) with two-stage initialization:')
print('Alpha, Beta, Gamma:', model_two_stage.adam_estimated['B'][:3])
print('Forecast:', model_two_stage.predict(h=12)['mean'].values[:3])

# %% [markdown]
# ### Test 4: Damped trend ETS(A,Ad,N)

# %%
model_optimal = ADAM(model='AAdN', lags=[12], initial='optimal')
model_optimal.fit(ts_df)
print('ETS(A,Ad,N) with optimal initial:')
print('Parameters:', model_optimal.adam_estimated['B'])
print('Forecast:', model_optimal.predict(h=12)['mean'].values[:3])

# %%
model_backcasting = ADAM(model='AAdN', lags=[12], initial='backcasting')
model_backcasting.fit(ts_df)
print('ETS(A,Ad,N) with backcasting initial:')
print('Parameters:', model_backcasting.adam_estimated['B'])
print('Forecast:', model_backcasting.predict(h=12)['mean'].values[:3])

# %%
model_two_stage = ADAM(model='AAdN', lags=[12], initial='two-stage', n_iterations=2)
model_two_stage.fit(ts_df)
print('ETS(A,Ad,N) with two-stage initialization:')
print('Parameters:', model_two_stage.adam_estimated['B'])
print('Forecast:', model_two_stage.predict(h=12)['mean'].values[:3])

# %% [markdown]
# ### Test 5: Multiplicative error ETS(M,N,N)

# %%
model_optimal = ADAM(model='MNN', lags=[12], distribution='dnorm', initial='optimal')
model_optimal.fit(ts_df)
print('ETS(M,N,N) with optimal initial:')
print('Parameters:', model_optimal.adam_estimated['B'])
print('Forecast:', model_optimal.predict(h=12)['mean'].values[:3])

# %%
model_backcasting = ADAM(model='MNN', lags=[12], distribution='dnorm', initial='backcasting')
model_backcasting.fit(ts_df)
print('ETS(M,N,N) with backcasting initial:')
print('Parameters:', model_backcasting.adam_estimated['B'])
print('Forecast:', model_backcasting.predict(h=12)['mean'].values[:3])

# %%
model_two_stage = ADAM(model='MNN', lags=[12], distribution='dnorm', initial='two-stage', n_iterations=2)
model_two_stage.fit(ts_df)
print('ETS(M,N,N) with two-stage initialization:')
print('Parameters:', model_two_stage.adam_estimated['B'])
print('Forecast:', model_two_stage.predict(h=12)['mean'].values[:3])

# %% [markdown]
# ### Test 6: Multiplicative seasonal ETS(M,A,M)

# %%
model_optimal = ADAM(model='MAM', lags=[12], distribution='dnorm', initial='optimal')
model_optimal.fit(ts_df)
print('ETS(M,A,M) with optimal initial:')
print('Alpha, Beta, Gamma:', model_optimal.adam_estimated['B'][:3])
print('Forecast:', model_optimal.predict(h=12)['mean'].values[:3])

# %%
model_backcasting = ADAM(model='MAM', lags=[12], distribution='dnorm', initial='backcasting')
model_backcasting.fit(ts_df)
print('ETS(M,A,M) with backcasting initial:')
print('Alpha, Beta, Gamma:', model_backcasting.adam_estimated['B'][:3])
print('Forecast:', model_backcasting.predict(h=12)['mean'].values[:3])

# %%
model_complete = ADAM(model='MAM', lags=[12], distribution='dnorm', initial='complete')
model_complete.fit(ts_df)
print('ETS(M,A,M) with complete initial:')
print('Alpha, Beta, Gamma:', model_complete.adam_estimated['B'][:3])
print('Forecast:', model_complete.predict(h=12)['mean'].values[:3])

# %%
model_two_stage = ADAM(model='MAM', lags=[12], distribution='dnorm', initial='two-stage', n_iterations=2)
model_two_stage.fit(ts_df)
print('ETS(M,A,M) with two-stage initialization:')
print('Alpha, Beta, Gamma:', model_two_stage.adam_estimated['B'][:3])
print('Forecast:', model_two_stage.predict(h=12)['mean'].values[:3])

# %% [markdown]
# ## Summary
# 
# Two-stage initialization successfully:
# - Runs a complete backcasting model first
# - Extracts parameters (persistence, damping, ARMA)
# - Extracts and normalizes initial states
# - Uses these as starting values for optimization
# - Produces results that combine benefits of both approaches


