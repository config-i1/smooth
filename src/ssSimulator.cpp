#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "ssGeneral.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;


// ##### Script for simulate functions
List simulator(arma::cube &arrayVt, arma::mat const &matrixerrors, arma::mat const &matrixot,
                 arma::cube const &arrayF, arma::rowvec const &rowvecW, arma::mat const &matrixG,
                 unsigned int const &obs, unsigned int const &nseries,
                 char const &E, char const &T, char const &S, arma::uvec &lags) {

    arma::mat matY(obs, nseries);
    arma::rowvec rowvecXt(1, arma::fill::zeros);
    arma::vec vecAt(1, arma::fill::zeros);

    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);
    int obsall = obs + maxlag;

    lags = maxlag - lags;

    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + obsall * i;
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);
    arma::mat matrixVt(obsall, lagslength, arma::fill::zeros);
    arma::mat matrixF(arrayF.n_rows, arrayF.n_cols, arma::fill::zeros);

    for(unsigned int i=0; i<nseries; i=i+1){
        matrixVt = arrayVt.slice(i);
        matrixF = arrayF.slice(i);
        for (int j=maxlag; j<obsall; j=j+1) {

            lagrows = lags - maxlag + j;
/* # Measurement equation and the error term */
            matY(j-maxlag,i) = matrixot(j-maxlag,i) * (wvalue(matrixVt(lagrows), rowvecW, E, T, S, rowvecXt, vecAt) +
                                 rvalue(matrixVt(lagrows), rowvecW, E, T, S, rowvecXt, vecAt) * matrixerrors(j-maxlag,i));
/* # Transition equation */
            matrixVt.row(j) = arma::trans(fvalue(matrixVt(lagrows), matrixF, T, S) +
                                          gvalue(matrixVt(lagrows), matrixF, rowvecW, E, T, S) % matrixG.col(i) * matrixerrors(j-maxlag,i));
/* Failsafe for cases when unreasonable value for state vector was produced */
            if(!matrixVt.row(j).is_finite()){
                matrixVt.row(j) = trans(matrixVt(lagrows));
            }
            if((S=='M') & (matrixVt(j,matrixVt.n_cols-1) <= 0)){
                matrixVt(j,matrixVt.n_cols-1) = arma::as_scalar(trans(matrixVt(lagrows.row(matrixVt.n_cols-1))));
            }
            if(T=='M'){
                if((matrixVt(j,0) <= 0) | (matrixVt(j,1) <= 0)){
                    matrixVt(j,0) = arma::as_scalar(trans(matrixVt(lagrows.row(0))));
                    matrixVt(j,1) = arma::as_scalar(trans(matrixVt(lagrows.row(1))));
                }
            }
        }
        arrayVt.slice(i) = matrixVt;
    }

    return List::create(Named("arrvt") = arrayVt, Named("matyt") = matY);
}

/* # Wrapper for simulator */
// [[Rcpp::export]]
RcppExport SEXP simulatorwrap(SEXP arrvt, SEXP matErrors, SEXP matot, SEXP matF, SEXP matw, SEXP matg,
                                SEXP Etype, SEXP Ttype, SEXP Stype, SEXP modellags) {

// ### arrvt should contain array of obs x ncomponents x nseries elements.
    NumericVector arrvt_n(arrvt);
    IntegerVector arrvt_dim = arrvt_n.attr("dim");
    arma::cube arrayVt(arrvt_n.begin(),arrvt_dim[0], arrvt_dim[1], arrvt_dim[2], false);

    NumericMatrix matErrors_n(matErrors);
    arma::mat matrixerrors(matErrors_n.begin(), matErrors_n.nrow(), matErrors_n.ncol(), false);

    NumericMatrix matot_n(matot);
    arma::mat matrixot(matot_n.begin(), matot_n.nrow(), matot_n.ncol(), false);

    NumericVector arrF_n(matF);
    IntegerVector arrF_dim = arrF_n.attr("dim");
    arma::cube arrayF(arrF_n.begin(),arrF_dim[0], arrF_dim[1], arrF_dim[2], false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

// ### matg should contain persistence vectors in each column
    NumericMatrix matg_n(matg);
    arma::mat matrixG(matg_n.begin(), matg_n.nrow(), matg_n.ncol(), false);

    unsigned int obs = matErrors_n.nrow();
    unsigned int nseries = matErrors_n.ncol();
    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    return wrap(simulator(arrayVt, matrixerrors, matrixot, arrayF, rowvecW, matrixG,
                            obs, nseries, E, T, S, lags));
}
