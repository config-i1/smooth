import numpy as np

from smooth.adam_general.sma import sma

if __name__ == "__main__":
    y = np.arange(0, 100)
    results = sma(y, order=5)
    print(results)
