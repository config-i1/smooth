#include <RcppArmadillo.h>
#include "ssGeneral.h"

// [[Rcpp::export]]
RcppExport SEXP matrixPowerWrap(SEXP matA, SEXP power){
    NumericMatrix matA_n(matA);
    arma::mat matrixA(matA_n.begin(), matA_n.nrow(), matA_n.ncol(), false);

    int pow = as<int>(power);

    return wrap(matrixPower(matrixA, pow));
}
