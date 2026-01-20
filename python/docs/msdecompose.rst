msdecompose
===========

Multiple seasonal decomposition for time series with multiple frequencies.

.. currentmodule:: smooth

.. autofunction:: smooth.msdecompose

Example Usage
-------------

.. code-block:: python

   from smooth import msdecompose
   import numpy as np

   # Create sample data with trend and seasonality
   t = np.arange(100)
   y = 10 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(100)

   # Decompose with monthly seasonality
   result = msdecompose(y, lags=[12], type="additive")

   # Access components
   trend = result["states"][:, 0]      # Trend component
   seasonal = result["seasonal"][0]     # Seasonal pattern
   initial = result["initial"]          # Initial values for ADAM

   # Use with ADAM model
   from smooth import ADAM
   model = ADAM(model="AAN", lags=[12])
   model.fit(y)

Returns Dictionary Keys
-----------------------

+----------------+------------------------------------------------------------------+
| Key            | Description                                                      |
+================+==================================================================+
| y              | Original time series                                             |
+----------------+------------------------------------------------------------------+
| states         | Matrix of states: [Level, Trend, Seasonal_1, ..., Seasonal_n]    |
+----------------+------------------------------------------------------------------+
| initial        | Dictionary with initial values for ADAM model initialization     |
+----------------+------------------------------------------------------------------+
| seasonal       | List of seasonal patterns, one array per lag                     |
+----------------+------------------------------------------------------------------+
| fitted         | Fitted values from decomposition                                 |
+----------------+------------------------------------------------------------------+
| lags           | Sorted lag periods used                                          |
+----------------+------------------------------------------------------------------+
| type           | Decomposition type ('additive' or 'multiplicative')              |
+----------------+------------------------------------------------------------------+
| smoother       | Smoothing method used                                            |
+----------------+------------------------------------------------------------------+
