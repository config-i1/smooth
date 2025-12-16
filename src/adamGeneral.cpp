#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "adamCore.cpp"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;


// ============================================================================
// WRAPPER FUNCTIONS TO CONVERT STRUCTURES TO R LISTS
// ============================================================================

namespace Rcpp {
    // Wrapper for PolyResult
    template <> SEXP wrap(const PolyResult& result) {
        return List::create(
            Named("arPolynomial") = result.arPolynomial,
            Named("iPolynomial") = result.iPolynomial,
            Named("ariPolynomial") = result.ariPolynomial,
            Named("maPolynomial") = result.maPolynomial
        );
    }

    // Wrapper for FitResult
    template <> SEXP wrap(const FitResult& result) {
        return List::create(
            Named("matVt") = result.matVt,
            Named("yFitted") = result.yFitted,
            Named("errors") = result.errors,
            Named("profile") = result.profile
        );
    }

    // Wrapper for ForecastResult
    template <> SEXP wrap(const ForecastResult& result) {
        return List::create(
            Named("yForecast") = result.yForecast
        );
    }

    // Wrapper for ErrorResult
    template <> SEXP wrap(const ErrorResult& result) {
        return List::create(
            Named("matErrors") = result.matErrors
        );
    }

    // Wrapper for SimulateResult
    template <> SEXP wrap(const SimulateResult& result) {
        return List::create(
            Named("arrayVt") = result.arrayVt,
            Named("matrixYt") = result.matrixYt
        );
    }
}

// ============================================================================
// RCPP MODULES FOR EXPOSING THE CLASS TO R
// ============================================================================

RCPP_MODULE(adamCore_module) {
    class_<adamCore>("adamCore")
    .constructor()
    .method("polynomialise", &adamCore::polynomialise)
    .method("fit", &adamCore::fit)
    .method("forecast", &adamCore::forecast)
    .method("error", &adamCore::error)
    .method("simulate", &adamCore::simulate);
}
