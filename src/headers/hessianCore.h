// Shared numerical Hessian implementation used by both the R (Rcpp)
// wrapper in src/hessianCpp.cpp and the Python (pybind11) wrapper in
// src/python/numDeriv.cpp. Single source of truth so both languages get
// bit-identical results given the same inputs.
//
// Algorithm: plain central finite difference, matching pracma::hessian()
// exactly (which is what the smooth package historically used in R):
//
//   diagonal:  H[i,i] = (f(x - h e_i) - 2 f(x) + f(x + h e_i)) / h^2
//   off-diag:  H[i,j] = (f(x + h e_i + h e_j) - f(x + h e_i - h e_j)
//                       - f(x - h e_i + h e_j) + f(x - h e_i - h e_j)) / (4 h^2)
//
// No Richardson extrapolation. Single absolute step size h applied to every
// parameter — matches pracma's design. The default h = .Machine$double.eps^(1/4)
// is set by the wrappers, not here.
//
// f is supplied as a templated callable so the same body works with
// std::function<double(std::vector<double>)> (pybind11) and a small
// callable wrapping an Rcpp::Function (Rcpp).

#ifndef HESSIAN_CORE_H
#define HESSIAN_CORE_H

#include <vector>
#include <cstddef>

namespace smooth_hessian {

// Eval f on a vector-typed callable. Returns the n×n symmetric Hessian as a
// row-major std::vector<double> of size n*n so the wrappers can copy it into
// their native matrix types without further allocation.
template <typename F>
std::vector<double> hessian(F f, const std::vector<double>& x0, double h)
{
    const std::size_t n = x0.size();
    std::vector<double> H(n * n, 0.0);

    const double f0  = f(x0);
    const double h2  = h * h;
    const double h2x = 4.0 * h2;

    std::vector<double> xp(x0), xm(x0);
    std::vector<double> xpp(x0), xpm(x0), xmp(x0), xmm(x0);

    for (std::size_t i = 0; i < n; ++i) {
        // Diagonal entry: 2-eval central FD
        xp = x0; xm = x0;
        xp[i] += h;
        xm[i] -= h;
        H[i * n + i] = (f(xm) - 2.0 * f0 + f(xp)) / h2;

        // Off-diagonal entries: 4-eval central FD, symmetric copy.
        for (std::size_t j = i + 1; j < n; ++j) {
            xpp = x0; xpm = x0; xmp = x0; xmm = x0;
            xpp[i] += h; xpp[j] += h;
            xpm[i] += h; xpm[j] -= h;
            xmp[i] -= h; xmp[j] += h;
            xmm[i] -= h; xmm[j] -= h;
            const double v = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / h2x;
            H[i * n + j] = v;
            H[j * n + i] = v;
        }
    }

    return H;
}

} // namespace smooth_hessian

#endif // HESSIAN_CORE_H
