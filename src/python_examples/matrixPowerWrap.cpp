#include <iostream>
#include <cmath>

#include <carma>
#include <armadillo>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <ssGeneral.h>

namespace py = pybind11;

// Wrapper for matrixPower function
py::array_t<double> matrixPowerWrapper(arma::mat const &matrixA, int power) {
    arma::mat result = matrixPower(matrixA, power);
    return carma::to_numpy(result);
}

PYBIND11_MODULE(_matrix_power, m) {
    m.doc() = "Matrix power functions"; // module docstring
    m.def(
        "matrix_power",
        &matrixPowerWrapper,
        "Computes the power of a matrix",
        py::arg("matrixA"),
        py::arg("power")
    );
}