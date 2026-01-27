#ifndef LOWESS_H
#define LOWESS_H

#include <armadillo>
#include <cmath>
#include <algorithm>

/**
 * LOWESS smoother that exactly matches R's stats::lowess function.
 *
 * This is a C++ implementation of Cleveland's LOWESS algorithm
 * as implemented in R's stats package (clowess C function).
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
 * R source: https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/lowess.c
 */
inline arma::vec lowess(const arma::vec& x, const arma::vec& y,
                        double f = 2.0/3.0, int nsteps = 3, double delta = -1.0) {
    size_t n = x.n_elem;

    if (n < 2) {
        return y;
    }

    // Compute delta if not provided
    if (delta < 0.0) {
        delta = 0.01 * (x.max() - x.min());
    }

    // Number of points in local window - at least 2, at most n
    size_t ns = std::max(size_t(2), std::min(n, size_t(f * n + 1e-7)));

    // Sort by x
    arma::uvec order = arma::sort_index(x);
    arma::vec x_sorted(n);
    arma::vec y_sorted(n);
    for (size_t i = 0; i < n; i++) {
        x_sorted(i) = x(order(i));
        y_sorted(i) = y(order(i));
    }

    arma::vec ys(n, arma::fill::zeros);   // smoothed values
    arma::vec rw(n, arma::fill::ones);    // robustness weights
    arma::vec res(n, arma::fill::zeros);  // residuals

    // Compute range for stability checks
    double x_range = x_sorted(n - 1) - x_sorted(0);

    // Lambda function for weighted local regression at a point
    auto lowest = [&](double xs, size_t nleft, size_t nright, const arma::vec& rw_iter,
                      double& ys_out, bool& ok) {
        // Compute bandwidth h
        double h = std::max(xs - x_sorted(nleft), x_sorted(nright) - xs);

        // Thresholds for weight calculation (R's h9 and h1)
        double h9 = 0.999 * h;
        double h1 = 0.001 * h;

        // Sum of weights
        double a = 0.0;
        // Store weights
        arma::vec w(n, arma::fill::zeros);

        // Loop through points - continue past nright to pick up ties
        size_t j = nleft;
        size_t nrt = nright;  // will track rightmost point with non-zero weight

        while (j < n) {
            // Compute absolute distance
            double r = std::abs(x_sorted(j) - xs);

            // Check if within bandwidth (using h9 threshold)
            if (r <= h9) {
                if (r <= h1) {
                    // Very close - weight = 1
                    w(j) = 1.0;
                } else {
                    // Tricube weight: (1 - (r/h)^3)^3
                    double ratio = r / h;
                    double cube = ratio * ratio * ratio;
                    double omc = 1.0 - cube;
                    w(j) = omc * omc * omc;
                }

                // Apply robustness weight from previous iteration
                w(j) *= rw_iter(j);
                a += w(j);
                nrt = j;
            } else if (x_sorted(j) > xs) {
                // Past the point, no more ties possible
                break;
            }

            j++;
        }

        // Check if we have any non-zero weights
        if (a <= 0.0) {
            ys_out = 0.0;
            ok = false;
            return;
        }

        // Normalize weights to sum to 1
        for (j = nleft; j <= nrt; j++) {
            w(j) /= a;
        }

        // Check if we can fit a line (h > 0)
        if (h > 0.0) {
            // Compute weighted center of x values
            a = 0.0;
            for (j = nleft; j <= nrt; j++) {
                a += w(j) * x_sorted(j);
            }

            // Compute slope if points are spread out enough
            double b = xs - a;
            double c = 0.0;
            for (j = nleft; j <= nrt; j++) {
                double diff = x_sorted(j) - a;
                c += w(j) * diff * diff;
            }

            // Stability check - only use slope if points are spread out
            // (R checks: sqrt(c) > 0.001 * range)
            if (std::sqrt(c) > 0.001 * x_range) {
                b /= c;
                // Adjust weights for linear fit
                for (j = nleft; j <= nrt; j++) {
                    w(j) *= (b * (x_sorted(j) - a) + 1.0);
                }
            }
        }

        // Compute fitted value as weighted sum
        ys_out = 0.0;
        for (j = nleft; j <= nrt; j++) {
            ys_out += w(j) * y_sorted(j);
        }

        ok = true;
    };

    // Main robustness iterations
    int iteration = 0;
    while (iteration <= nsteps) {
        size_t nleft = 0;
        size_t nright = ns - 1;
        int last = -1;  // index of previous estimated point
        size_t i = 0;   // index of current point

        while (true) {
            // Move window right if it decreases radius
            if (nright < n - 1) {
                double d1 = x_sorted(i) - x_sorted(nleft);
                double d2 = x_sorted(nright + 1) - x_sorted(i);

                if (d1 > d2) {
                    // Radius decreases by moving right
                    nleft++;
                    nright++;
                    continue;
                }
            }

            // Compute fitted value at x[i]
            double ys_i;
            bool ok;
            lowest(x_sorted(i), nleft, nright, rw, ys_i, ok);
            ys(i) = ys_i;

            if (!ok) {
                // All weights zero - copy over value
                ys(i) = y_sorted(i);
            }

            // Interpolate skipped points
            if (last < (int)i - 1) {
                double denom = x_sorted(i) - x_sorted(last);
                // Should be non-zero by construction
                if (denom > 0.0) {
                    for (size_t j = last + 1; j < i; j++) {
                        double alpha = (x_sorted(j) - x_sorted(last)) / denom;
                        ys(j) = alpha * ys(i) + (1.0 - alpha) * ys(last);
                    }
                }
            }

            // Update last estimated point
            last = (int)i;

            // Skip ahead using delta - find next point beyond delta
            double cut = x_sorted(last) + delta;
            i++;
            while (i < n) {
                if (x_sorted(i) > cut) {
                    break;
                }
                // Special case: exact ties get same value
                if (x_sorted(i) == x_sorted(last)) {
                    ys(i) = ys(last);
                    last = (int)i;
                }
                i++;
            }

            // Adjust i (R's: i = max(last+1, i-1))
            i = std::max((size_t)(last + 1), i > 0 ? i - 1 : 0);

            if (last >= (int)n - 1) {
                break;
            }
        }

        // Compute residuals
        for (size_t i = 0; i < n; i++) {
            res(i) = y_sorted(i) - ys(i);
        }

        // Overall scale estimate (mean absolute residual)
        double sc = arma::sum(arma::abs(res)) / n;

        // Compute robustness weights (except on last iteration)
        if (iteration >= nsteps) {
            break;
        }

        // Compute median absolute deviation (cmad = 6 * median)
        arma::vec abs_res = arma::abs(res);
        size_t m1 = n / 2;

        // Partial sort to find median
        arma::vec abs_res_sorted = arma::sort(abs_res);
        double cmad;
        if (n % 2 == 0) {
            size_t m2 = n - m1 - 1;
            cmad = 3.0 * (abs_res_sorted(m1) + abs_res_sorted(m2));
        } else {
            cmad = 6.0 * abs_res_sorted(m1);
        }

        // Check if effectively zero (R's threshold: 1e-7 * sc)
        if (cmad < 1e-7 * sc) {
            break;
        }

        // Compute biweight robustness weights
        double c9 = 0.999 * cmad;
        double c1 = 0.001 * cmad;
        for (size_t i = 0; i < n; i++) {
            double r = std::abs(res(i));
            if (r <= c1) {
                rw(i) = 1.0;
            } else if (r <= c9) {
                double ratio = r / cmad;
                double omc = 1.0 - ratio * ratio;
                rw(i) = omc * omc;
            } else {
                rw(i) = 0.0;
            }
        }

        iteration++;
    }

    // Restore original order
    arma::vec result(n);
    for (size_t i = 0; i < n; i++) {
        result(order(i)) = ys(i);
    }

    return result;
}

#endif // LOWESS_H
