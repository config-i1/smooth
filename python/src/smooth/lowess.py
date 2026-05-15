"""
LOWESS (Locally Weighted Scatterplot Smoothing) implementation.

This module provides a LOWESS smoother based on Cleveland's (1979) algorithm:
locally weighted polynomial regression with a tricube weight function and
iterative reweighting for robustness to outliers.
"""

import numpy as np

# Import C++ implementation
from smooth.adam_general import lowess_cpp as _lowess_cpp


def lowess(x, y=None, f=2 / 3, iter=3, delta=None):
    """
    LOWESS smoother (Locally Weighted Scatterplot Smoothing).

    Performs locally weighted polynomial regression using Cleveland's LOWESS
    algorithm. Useful for smoothing noisy data while preserving local
    patterns and for robust trend estimation in the presence of outliers.

    Parameters
    ----------
    x : array-like
        The x values for the data. Can also be a 2D array or list with two elements,
        in which case the first column/element is used as x and second as y.
    y : array-like, optional
        The y values for the data. If None, x must contain both x and y values
        as a 2D array or a list/tuple of two arrays.
    f : float, default=2/3
        The smoother span. This gives the proportion of points in the plot which
        influence the smooth at each value. Larger values give more smoothness.
    iter : int, default=3
        The number of robustifying iterations which should be performed.
        Using smaller values of iter will make lowess run faster.
    delta : float, optional
        Values within delta of each other are treated as being at the same point.
        If None (default), uses 0.01 * (max(x) - min(x)).

    Returns
    -------
    dict
        A dictionary with two keys:
        - 'x': The sorted x values
        - 'y': The smoothed y values corresponding to the sorted x values

    Notes
    -----
    Implements Cleveland's (1979) LOWESS algorithm: locally weighted
    polynomial regression with a tricube weight function and iterative
    reweighting for robustness to outliers.

    The returned dictionary holds ``x`` sorted in ascending order and ``y``
    containing the corresponding smoothed values.

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing
    Scatterplots". Journal of the American Statistical Association 74(368): 829-836.

    See Also
    --------
    smooth.adam_general.lowess_cpp : Direct C++ lowess function (unsorted output)
    smooth.adam_general.core.utils.utils.lowess_r : Pure Python implementation

    Examples
    --------
    Basic usage with separate x and y arrays:

    >>> import numpy as np
    >>> from smooth import lowess
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = np.array([1.1, 2.0, 2.9, 4.2, 4.8, 6.1, 7.0, 7.9, 9.2, 9.8])
    >>> result = lowess(x, y)
    >>> result['x']  # Sorted x values
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    >>> result['y']  # Smoothed y values
    array([...])

    With custom smoother span:

    >>> result = lowess(x, y, f=0.5)

    Using a 2D array input:

    >>> xy = np.column_stack([x, y])
    >>> result = lowess(xy)
    """
    # Handle the case where x contains both x and y
    x = np.asarray(x, dtype=np.float64)

    if y is None:
        # x should be a 2D array or similar structure
        if x.ndim == 2 and x.shape[1] >= 2:
            y = x[:, 1].copy()
            x = x[:, 0].copy()
        elif x.ndim == 1:
            raise ValueError(
                "If y is not provided, x must be a 2D array with at least 2 columns"
            )
        else:
            raise ValueError(
                "Invalid input: x must be a 2D array with 2 columns when y is None"
            )
    else:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )

    n = len(x)
    if n < 2:
        return {"x": x.copy(), "y": y.copy()}

    # Compute delta if not provided (default: 0.01 * diff(range(x)))
    if delta is None:
        delta = 0.01 * (x.max() - x.min())

    # Sort by x so the returned (x, y) pair is ascending in x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Call C++ implementation
    # Note: the C++ routine handles sorting internally; we pre-sort so the
    # smoothed output stays aligned with the returned sorted x.
    smoothed = _lowess_cpp(x_sorted, y_sorted, f=f, nsteps=iter, delta=delta)

    return {"x": x_sorted, "y": smoothed}
