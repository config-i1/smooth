#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <armadillo>

#ifndef PYTHON_BUILD
#define PYTHON_BUILD 1
#endif
#include <carma>

#include "../headers/lowess.h"

namespace py = pybind11;

/**
 * Python wrapper for lowess function.
 *
 * @param x X values as numpy array
 * @param y Y values as numpy array
 * @param f Smoother span (fraction of points), default 2/3
 * @param nsteps Number of robustifying iterations, default 3
 * @param delta Distance threshold for interpolation. If < 0, uses 0.01 * range(x)
 * @return Smoothed y values as numpy array
 */
py::array_t<double> lowess_wrapper(
    const arma::vec& x,
    const arma::vec& y,
    double f = 2.0/3.0,
    int nsteps = 3,
    double delta = -1.0
) {
    arma::vec result = lowess(x, y, f, nsteps, delta);
    // Convert to 1D numpy array
    size_t n = result.n_elem;
    py::array_t<double> arr({static_cast<py::ssize_t>(n)});
    auto buf = arr.mutable_unchecked<1>();
    for (size_t i = 0; i < n; i++) {
        buf(i) = result(i);
    }
    return arr;
}

PYBIND11_MODULE(_lowess, m) {
    m.doc() = "LOWESS smoother matching R's stats::lowess exactly";

    m.def(
        "lowess",
        &lowess_wrapper,
        R"pbdoc(
            LOWESS smoother that exactly matches R's stats::lowess function.

            This is a C++ implementation of Cleveland's LOWESS algorithm
            as implemented in R's stats package (clowess C function).

            Parameters
            ----------
            x : array-like
                X values (will be converted to float)
            y : array-like
                Y values (will be converted to float)
            f : float, optional
                Smoother span (fraction of points), default 2/3
            nsteps : int, optional
                Number of robustifying iterations, default 3
            delta : float, optional
                Distance threshold for interpolation.
                If < 0 (default), uses 0.01 * range(x)

            Returns
            -------
            ndarray
                Smoothed y values in original x order

            References
            ----------
            Cleveland, W.S. (1979) "Robust Locally Weighted Regression and
            Smoothing Scatterplots". JASA 74(368): 829-836.
        )pbdoc",
        py::arg("x"),
        py::arg("y"),
        py::arg("f") = 2.0/3.0,
        py::arg("nsteps") = 3,
        py::arg("delta") = -1.0
    );
}
