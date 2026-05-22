#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <vector>

namespace py = pybind11;

// Numerical Hessian matching pracma::hessian() exactly: a plain finite-difference
// scheme with a single absolute step h and no Richardson extrapolation.
//   diagonal:  H[i,i] = (f(x-h e_i) - 2 f(x) + f(x+h e_i)) / h^2
//   off-diag:  H[i,j] = (f(x+h e_i+h e_j) - f(x+h e_i-h e_j)
//                        - f(x-h e_i+h e_j) + f(x-h e_i-h e_j)) / (4 h^2)
// f is a Python callable taking a 1-D numpy array and returning a scalar.
py::array_t<double> hessian(std::function<double(py::array_t<double>)> f,
                            py::array_t<double> x0, double h) {
    auto x = x0.unchecked<1>();
    const py::ssize_t n = x.shape(0);

    auto eval = [&](const std::vector<double>& v) {
        py::array_t<double> a(n);
        auto ab = a.mutable_unchecked<1>();
        for (py::ssize_t k = 0; k < n; ++k) {
            ab(k) = v[k];
        }
        return f(a);
    };

    std::vector<double> base(n);
    for (py::ssize_t k = 0; k < n; ++k) {
        base[k] = x(k);
    }

    py::array_t<double> H({n, n});
    auto Hb = H.mutable_unchecked<2>();
    const double f0 = eval(base);
    const double h2 = h * h;

    for (py::ssize_t i = 0; i < n; ++i) {
        std::vector<double> p = base, m = base;
        p[i] += h;
        m[i] -= h;
        Hb(i, i) = (eval(m) - 2.0 * f0 + eval(p)) / h2;

        for (py::ssize_t j = i + 1; j < n; ++j) {
            std::vector<double> pp = base, pm = base, mp = base, mm = base;
            pp[i] += h; pp[j] += h;
            pm[i] += h; pm[j] -= h;
            mp[i] -= h; mp[j] += h;
            mm[i] -= h; mm[j] -= h;
            double v = (eval(pp) - eval(pm) - eval(mp) + eval(mm)) / (4.0 * h2);
            Hb(i, j) = v;
            Hb(j, i) = v;
        }
    }

    return H;
}

PYBIND11_MODULE(_numDeriv, m) {
    m.def("hessian", &hessian, py::arg("f"), py::arg("x0"),
          py::arg("h") = 1.220703125e-04);  // ~ .Machine$double.eps^(1/4)
}
