# %%
import numpy as np
import pandas as pd

%load_ext rpy2.ipython

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import os

def load_smooth_dev():
    """Load the smooth package in development mode"""
    smooth_path = "/home/filtheo/smooth/"
    
    ro.r(f'''
    if (!requireNamespace("devtools", quietly=TRUE)) {{
        install.packages("devtools", repos="https://cran.rstudio.com/")
    }}
    devtools::load_all("{smooth_path}")
    ''')
    
    print("Smooth package loaded in development mode")

load_smooth_dev()


# %% [markdown]
# # Two-Stage Initialization Tests - R Implementation
# 
# Two-stage initialization works by:
# 1. First running a model with `initial="two-stage"` (full backcasting) to get good starting values
# 2. Using those values as initial guesses for optimization, allowing parameter refinement
# 
# This should produce results that are:
# - Different from pure backcasting (since parameters are refined)
# - Different from pure optimal (since starting values are better)
# - Generally better or similar quality to both methods
# 

# %% [markdown]
# ### Test 1: Global level ETS(A,N,N) - two-stage vs optimal vs backcasting
# 

# %%
np.random.seed(33)
n_points = 100
time_series = np.random.normal(100, 10, n_points)
ts_df = pd.DataFrame({'value': time_series}, index=pd.date_range(start='2023-01-01', periods=n_points, freq='ME'))

# %%
%%R -i ts_df

model <- adam(ts_df, model = "ANN", lags = c(12), initial = 'optimal')
cat('ETS(A,N,N) with optimal initial:\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)


# %%
%%R -i ts_df

model <- adam(ts_df, model = "ANN", lags = c(12), initial = 'backcasting', nIterations = 2)
cat('ETS(A,N,N) with backcasting (nIterations=2):\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "ANN", lags = c(12), initial = 'complete')
cat('ETS(A,N,N) with complete (nIterations=2):\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "ANN", lags = c(12), initial = 'two-stage', nIterations = 2)
cat('ETS(A,N,N) with two-stage initialization (two-stage):\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)


# %% [markdown]
# ### Test 2: Local trend ETS(A,A,N)
# 

# %%
np.random.seed(42)
n_points = 120
errors = np.random.normal(0, 10, n_points)
trend = np.random.normal(0.5, 2, n_points)
time_series = np.zeros(n_points)
time_series[0] = 100
for i in range(n_points-1):
    time_series[i+1] = time_series[i] + (0.1-1) * errors[i] + trend[i] + errors[i+1]
ts_df = pd.DataFrame({'value': time_series})


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAN", lags = c(12), initial = 'optimal')
cat('ETS(A,A,N) with optimal initial:\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAN", lags = c(12), initial = 'complete')
cat('ETS(A,A,N) with complete initial:\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] 


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAN", lags = c(12), initial = 'two-stage', nIterations = 2)
cat('ETS(A,A,N) with two-stage initialization (two-stage):\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %% [markdown]
# ### Test 3: Seasonal data ETS(A,A,A)
# 

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
ts_df = pd.DataFrame({'value': time_series})


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAA", lags = c(12), distribution = 'dnorm', initial = 'optimal')
cat('ETS(A,A,A) with optimal initial:\n')
cat('Alpha, Beta, Gamma:', model$B[1:3], '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAA", lags = c(12), distribution = 'dnorm', initial = 'backcasting')
cat('ETS(A,A,A) with backcasting initial:\n')
cat('Alpha, Beta, Gamma:', model$B[1:3], '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAA", lags = c(12), distribution = 'dnorm', initial = 'complete')
cat('ETS(A,A,A) with complete initial:\n')
cat('Alpha, Beta, Gamma:', model$B[1:3], '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAA", lags = c(12), distribution = 'dnorm', initial = 'two-stage', nIterations = 2)
cat('ETS(A,A,A) with two-stage initialization (two-stage):\n')
cat('Alpha, Beta, Gamma:', model$B[1:3], '\n')
forecast(model, h = 12)[1:3] |> print()


# %% [markdown]
# ### Test 4: Damped trend ETS(A,Ad,N)
# 

# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAdN", lags = c(12), initial = 'optimal')
cat('ETS(A,Ad,N) with optimal initial:\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAdN", lags = c(12), initial = 'backcasting')
cat('ETS(A,Ad,N) with backcasting initial:\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "AAdN", lags = c(12), initial = 'two-stage', nIterations = 2)
cat('ETS(A,Ad,N) with two-stage initialization (two-stage):\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %% [markdown]
# ### Test 5: Multiplicative error ETS(M,N,N)
# 

# %%
%%R -i ts_df

model <- adam(ts_df, model = "MNN", lags = c(12), distribution = 'dnorm', initial = 'optimal')
cat('ETS(M,N,N) with optimal initial:\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "MNN", lags = c(12), distribution = 'dnorm', initial = 'backcasting')
cat('ETS(M,N,N) with backcasting initial:\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "MNN", lags = c(12), distribution = 'dnorm', initial = 'two-stage', nIterations = 2)
cat('ETS(M,N,N) with two-stage initialization (two-stage):\n')
cat('Parameters:', model$B, '\n')
forecast(model, h = 12)[1:3] |> print()


# %% [markdown]
# ### Test 6: Multiplicative seasonal ETS(M,A,M)
# 

# %%
%%R -i ts_df

model <- adam(ts_df, model = "MAM", lags = c(12), distribution = 'dnorm', initial = 'optimal')
cat('ETS(M,A,M) with optimal initial:\n')
cat('Alpha, Beta, Gamma:', model$B[1:3], '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "MAM", lags = c(12), distribution = 'dnorm', initial = 'backcasting')
cat('ETS(M,A,M) with backcasting initial:\n')
cat('Alpha, Beta, Gamma:', model$B[1:3], '\n')
forecast(model, h = 12)[1:3] |> print()


# %%
%%R -i ts_df

model <- adam(ts_df, model = "MAM", lags = c(12), distribution = 'dnorm', initial = 'complete')
cat('ETS(M,A,M) with complete initial:\n')
cat('Alpha, Beta, Gamma:', model$B[1:3], '\n')
forecast(model, h = 12)[1:3] |> print()


# %%


# %%
%%R -i ts_df

model <- adam(ts_df, model = "MAM", lags = c(12), distribution = 'dnorm', initial = 'two-stage', nIterations = 2)
cat('ETS(M,A,M) with two-stage initialization (two-stage):\n')
cat('Alpha, Beta, Gamma:', model$B[1:3], '\n')
forecast(model, h = 12)[1:3] |> print()


# %% [markdown]
# ## Summary
# 
# Two-stage initialization successfully:
# - Runs a two-stage backcasting model first
# - Extracts parameters (persistence, damping, ARMA)
# - Extracts and normalizes initial states
# - Uses these as starting values for optimization
# - Produces results that combine benefits of both approaches
# 


