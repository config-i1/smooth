from core.adam import Adam

import pandas as pd
import numpy as np
from core.checker import parameters_checker
from typing import List, Union, Dict, Any
from smooth.adam_general._adam_general import adam_fitter, adam_forecaster
from core.utils.utils import measurement_inverter, scaler, calculate_likelihood, calculate_entropy, calculate_multistep_loss
from numpy.linalg import eigvals
import nlopt

from core.estimator import estimator, selector
from core.creator import creator, initialiser, architector, filler
from core.utils.ic import ic_function

from smooth.adam_general._adam_general import adam_fitter, adam_forecaster

import warnings

# Generate random monthly time series data
np.random.seed(41)  # For reproducibility
n_points = 24  # 2 years of monthly data
time_series = np.random.randint(1, 100, size=n_points).cumsum()  # Random walk with strictly positive integers
dates = pd.date_range(start='2023-01-01', periods=n_points, freq='M')  # Monthly frequency
ts_df = pd.DataFrame({'value': time_series}, index=dates)

model = "ANN"
lags = [12]
multisteps = False,
lb = None,
ub = None,
maxtime = None,
print_level = 1, # 1 or 0
maxeval = None,
h = 12



# Assume that the model is not provided
# these will be default arguments
profiles_recent_provided = False
profiles_recent_table = None

adam = Adam(model, lags)
adam.fit(ts_df, h = h)