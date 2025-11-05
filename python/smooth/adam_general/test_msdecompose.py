import numpy as np
from smooth import msdecompose

y = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 3.5, 4.0, 5.0, 6.0, 5.5, 5.0, 6.0])
res = msdecompose(y, lags=[12], type="multiplicative", smoother="ma", y_name="MySeries")
print(res.initial, res.lags, res.type)