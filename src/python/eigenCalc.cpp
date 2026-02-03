#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <armadillo>

#ifndef PYTHON_BUILD
#define PYTHON_BUILD 1
#endif
#include <carma>

#include "../headers/eigenCalc.h"

namespace py = pybind11;

py::array_t<double> smooth_eigens_wrapper(
    const arma::mat& persistence,
    const arma::mat& transition,
    const arma::mat& measurement,
    const arma::ivec& lags_model_all,
    bool xreg_model,
    int obs_in_sample,
    bool has_delta
) {
    arma::vec result = smoothEigensCpp(persistence, transition, measurement,
                                        lags_model_all, xreg_model,
                                        obs_in_sample, has_delta);
    size_t n = result.n_elem;
    py::array_t<double> arr({static_cast<py::ssize_t>(n)});
    auto buf = arr.mutable_unchecked<1>();
    for (size_t i = 0; i < n; i++) {
        buf(i) = result(i);
    }
    return arr;
}

PYBIND11_MODULE(_eigenCalc, m) {
    m.def(
        "smooth_eigens",
        &smooth_eigens_wrapper,
        py::arg("persistence"),
        py::arg("transition"),
        py::arg("measurement"),
        py::arg("lags_model_all"),
        py::arg("xreg_model"),
        py::arg("obs_in_sample"),
        py::arg("has_delta")
    );
}
