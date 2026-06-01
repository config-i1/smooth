#ifndef LOWESS_H
#define LOWESS_H

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

/**
 * LOWESS smoother that exactly matches R's stats::lowess function.
 *
 * This is a C++ implementation of Cleveland's LOWESS algorithm as implemented
 * in R's stats package (clowess C function). The inner loops intentionally use
 * the same 1-based indexing and scalar accumulation order as R's lowess.c;
 * Armadillo reductions and vectorized arithmetic can change the last bits.
 *
 * @param x X values (must be sorted or will be sorted internally)
 * @param y Y values
 * @param f Smoother span (fraction of points), default 2/3
 * @param nsteps Number of robustifying iterations, default 3
 * @param delta Distance threshold for interpolation. If < 0, uses 0.01 * range(x)
 * @return Smoothed y values in original x order
 *
 * References:
 * Cleveland, W.S. (1979) "Robust Locally Weighted Regression and
 * Smoothing Scatterplots". JASA 74(368): 829-836.
 * R source: src/library/stats/src/lowess.c
 */
namespace smooth_lowess_detail {

inline double square(double x) {
    return x * x;
}

inline double cube(double x) {
    return x * x * x;
}

inline int min2(int a, int b) {
    return a < b ? a : b;
}

inline int max2(int a, int b) {
    return a > b ? a : b;
}

inline double max2(double a, double b) {
    return a > b ? a : b;
}

inline int compare_double(const void* lhs, const void* rhs) {
    double a = *static_cast<const double*>(lhs);
    double b = *static_cast<const double*>(rhs);
    return (a > b) - (a < b);
}

inline void lowest(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int n,
    double* xs,
    double* ys,
    int nleft,
    int nright,
    std::vector<double>& w,
    bool userw,
    const std::vector<double>& rw,
    bool* ok
) {
    int nrt;
    int j;
    double a;
    double b;
    double c;
    double h;
    double h1;
    double h9;
    double r;
    double range;

    range = x[n] - x[1];
    h = max2(*xs - x[nleft], x[nright] - *xs);
    h9 = 0.999 * h;
    h1 = 0.001 * h;

    a = 0.0;
    j = nleft;
    while (j <= n) {
        w[j] = 0.0;
        r = std::abs(x[j] - *xs);
        if (r <= h9) {
            if (r <= h1) {
                w[j] = 1.0;
            } else {
                w[j] = cube(1.0 - cube(r / h));
            }
            if (userw) {
                w[j] *= rw[j];
            }
            a += w[j];
        } else if (x[j] > *xs) {
            break;
        }
        j = j + 1;
    }

    nrt = j - 1;
    if (a <= 0.0) {
        *ok = false;
    } else {
        *ok = true;
        for (j = nleft; j <= nrt; j++) {
            w[j] /= a;
        }
        if (h > 0.0) {
            a = 0.0;
            for (j = nleft; j <= nrt; j++) {
                a += w[j] * x[j];
            }
            b = *xs - a;
            c = 0.0;
            for (j = nleft; j <= nrt; j++) {
                c += w[j] * square(x[j] - a);
            }
            if (std::sqrt(c) > 0.001 * range) {
                b /= c;
                for (j = nleft; j <= nrt; j++) {
                    w[j] *= (b * (x[j] - a) + 1.0);
                }
            }
        }
        *ys = 0.0;
        for (j = nleft; j <= nrt; j++) {
            *ys += w[j] * y[j];
        }
    }
}

}  // namespace smooth_lowess_detail

inline arma::vec lowess(const arma::vec& x, const arma::vec& y,
                        double f = 2.0 / 3.0, int nsteps = 3, double delta = -1.0) {
    const int n = static_cast<int>(x.n_elem);

    if (n < 2) {
        return y;
    }

    std::vector<size_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (x(lhs) == x(rhs)) {
            return lhs < rhs;
        }
        return x(lhs) < x(rhs);
    });

    std::vector<double> x_sorted(static_cast<size_t>(n) + 1);
    std::vector<double> y_sorted(static_cast<size_t>(n) + 1);
    for (int i = 1; i <= n; i++) {
        x_sorted[static_cast<size_t>(i)] = x(order[static_cast<size_t>(i - 1)]);
        y_sorted[static_cast<size_t>(i)] = y(order[static_cast<size_t>(i - 1)]);
    }

    if (delta < 0.0) {
        delta = 0.01 * (x_sorted[static_cast<size_t>(n)] - x_sorted[1]);
    }

    const int ns = smooth_lowess_detail::max2(
        2,
        smooth_lowess_detail::min2(n, static_cast<int>(f * n + 1e-7))
    );

    int i;
    int iter;
    int j;
    int last;
    int m1;
    int m2;
    int nleft;
    int nright;
    bool ok;
    double alpha;
    double c1;
    double c9;
    double cmad;
    double cut;
    double d1;
    double d2;
    double denom;
    double r;
    double sc;
    std::vector<double> ys(static_cast<size_t>(n) + 1, 0.0);
    std::vector<double> rw(static_cast<size_t>(n) + 1, 0.0);
    std::vector<double> res(static_cast<size_t>(n) + 1, 0.0);

    iter = 1;
    while (iter <= nsteps + 1) {
        nleft = 1;
        nright = ns;
        last = 0;
        i = 1;

        for (;;) {
            if (nright < n) {
                d1 = x_sorted[static_cast<size_t>(i)] - x_sorted[static_cast<size_t>(nleft)];
                d2 = x_sorted[static_cast<size_t>(nright + 1)] -
                     x_sorted[static_cast<size_t>(i)];
                if (d1 > d2) {
                    nleft++;
                    nright++;
                    continue;
                }
            }

            smooth_lowess_detail::lowest(
                x_sorted,
                y_sorted,
                n,
                &x_sorted[static_cast<size_t>(i)],
                &ys[static_cast<size_t>(i)],
                nleft,
                nright,
                res,
                iter > 1,
                rw,
                &ok
            );
            if (!ok) {
                ys[static_cast<size_t>(i)] = y_sorted[static_cast<size_t>(i)];
            }

            if (last < i - 1) {
                denom = x_sorted[static_cast<size_t>(i)] -
                        x_sorted[static_cast<size_t>(last)];
                for (j = last + 1; j < i; j++) {
                    alpha = (x_sorted[static_cast<size_t>(j)] -
                             x_sorted[static_cast<size_t>(last)]) /
                            denom;
                    ys[static_cast<size_t>(j)] =
                        alpha * ys[static_cast<size_t>(i)] +
                        (1.0 - alpha) * ys[static_cast<size_t>(last)];
                }
            }

            last = i;
            cut = x_sorted[static_cast<size_t>(last)] + delta;
            for (i = last + 1; i <= n; i++) {
                if (x_sorted[static_cast<size_t>(i)] > cut) {
                    break;
                }
                if (x_sorted[static_cast<size_t>(i)] ==
                    x_sorted[static_cast<size_t>(last)]) {
                    ys[static_cast<size_t>(i)] = ys[static_cast<size_t>(last)];
                    last = i;
                }
            }
            i = smooth_lowess_detail::max2(last + 1, i - 1);
            if (last >= n) {
                break;
            }
        }

        for (i = 1; i <= n; i++) {
            res[static_cast<size_t>(i)] =
                y_sorted[static_cast<size_t>(i)] - ys[static_cast<size_t>(i)];
        }

        sc = 0.0;
        for (i = 1; i <= n; i++) {
            sc += std::abs(res[static_cast<size_t>(i)]);
        }
        sc /= n;

        if (iter > nsteps) {
            break;
        }

        for (i = 1; i <= n; i++) {
            rw[static_cast<size_t>(i)] = std::abs(res[static_cast<size_t>(i)]);
        }

        m1 = n / 2;
        std::qsort(rw.data() + 1, static_cast<size_t>(n), sizeof(double),
                   smooth_lowess_detail::compare_double);
        if (n % 2 == 0) {
            m2 = n - m1 - 1;
            cmad = 3.0 * (
                rw[static_cast<size_t>(m1 + 1)] + rw[static_cast<size_t>(m2 + 1)]
            );
        } else {
            cmad = 6.0 * rw[static_cast<size_t>(m1 + 1)];
        }

        if (cmad < 1e-7 * sc) {
            break;
        }
        c9 = 0.999 * cmad;
        c1 = 0.001 * cmad;
        for (i = 1; i <= n; i++) {
            r = std::abs(res[static_cast<size_t>(i)]);
            if (r <= c1) {
                rw[static_cast<size_t>(i)] = 1.0;
            } else if (r <= c9) {
                rw[static_cast<size_t>(i)] =
                    smooth_lowess_detail::square(
                        1.0 - smooth_lowess_detail::square(r / cmad)
                    );
            } else {
                rw[static_cast<size_t>(i)] = 0.0;
            }
        }
        iter++;
    }

    arma::vec result(static_cast<arma::uword>(n));
    for (int i = 1; i <= n; i++) {
        result(order[static_cast<size_t>(i - 1)]) = ys[static_cast<size_t>(i)];
    }

    return result;
}

#endif  // LOWESS_H
