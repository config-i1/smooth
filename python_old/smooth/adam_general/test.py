
import pandas as pd
import numpy as np
from core.checker import parameters_checker
from typing import List, Union, Dict, Any
from smooth.adam_general._adam_general import adam_fitter, adam_forecaster
from core.utils.utils import measurement_inverter, scaler, calculate_likelihood, calculate_entropy, calculate_multistep_loss
from numpy.linalg import eigvals
import nlopt
from core.adam import Adam


import pandas as pd
import numpy as np
# Create the AirPassengers dataset manually
data = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
]

# Create a proper datetime index
dates = pd.date_range(start='1949-01-01', periods=len(data), freq='MS')

# Create a pandas Series with the data
air_passengers_series = pd.Series(data, index=dates, name='AirPassengers')

# Create a DataFrame with the time series
ts_df = pd.DataFrame({'value': air_passengers_series})

print("AirPassengers dataset loaded:")
print(ts_df.head())

import time

start_time = time.time()

model = "AAA"
lags = [12]
h = 12

adam = Adam(model, lags)
adam.fit(ts_df, h = h)
fc = adam.predict()
execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.4f} seconds")
fc['forecast']
