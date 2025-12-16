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
    .method("fit", &adamCore::fit)
    .method("forecast", &adamCore::forecast)
    .method("error", &adamCore::error)
    .method("simulate", &adamCore::simulate);
}




// [[Rcpp::export]]
RcppExport SEXP adamPolynomialiser(arma::vec const &B,
                                   arma::uvec const &arOrders, arma::uvec const &iOrders, arma::uvec const &maOrders,
                                   bool const &arEstimate, bool const &maEstimate,
                                   SEXP armaParameters, arma::uvec const &lags){

    // Sometimes armaParameters is NULL. Treat this correctly
    arma::vec armaParametersValue;
    if(!Rf_isNull(armaParameters)){
        armaParametersValue = as<arma::vec>(armaParameters);
    }

// Form matrices with parameters, that are then used for polynomial multiplication
    arma::mat arParameters(max(arOrders % lags)+1, arOrders.n_elem, arma::fill::zeros);
    arma::mat iParameters(max(iOrders % lags)+1, iOrders.n_elem, arma::fill::zeros);
    arma::mat maParameters(max(maOrders % lags)+1, maOrders.n_elem, arma::fill::zeros);

    arParameters.row(0).fill(1);
    iParameters.row(0).fill(1);
    maParameters.row(0).fill(1);

    int nParam = 0;
    int armanParam = 0;
    for(unsigned int i=0; i<lags.n_rows; ++i){
        if(arOrders(i) * lags(i) != 0){
            for(unsigned int j=0; j<arOrders(i); ++j){
                if(arEstimate){
                    arParameters((j+1)*lags(i),i) = -B(nParam);
                    nParam += 1;
                }
                else{
                    arParameters((j+1)*lags(i),i) = -armaParametersValue(armanParam);
                    armanParam += 1;
                }
            }
        }

        if(iOrders(i) * lags(i) != 0){
            iParameters(lags(i),i) = -1;
        }

        if(maOrders(i) * lags(i) != 0){
            for(unsigned int j=0; j<maOrders(i); ++j){
                if(maEstimate){
                    maParameters((j+1)*lags(i),i) = B(nParam);
                    nParam += 1;
                }
                else{
                    maParameters((j+1)*lags(i),i) = armaParametersValue(armanParam);
                    armanParam += 1;
                }
            }
        }
    }

// Prepare vectors with coefficients for polynomials
    arma::vec arPolynomial(sum(arOrders % lags)+1, arma::fill::zeros);
    arma::vec iPolynomial(sum(iOrders % lags)+1, arma::fill::zeros);
    arma::vec maPolynomial(sum(maOrders % lags)+1, arma::fill::zeros);
    arma::vec ariPolynomial(sum(arOrders % lags)+sum(iOrders % lags)+1, arma::fill::zeros);
    arma::vec bufferPolynomial;

    arPolynomial.rows(0,arOrders(0)*lags(0)) = arParameters.submat(0,0,arOrders(0)*lags(0),0);
    iPolynomial.rows(0,iOrders(0)*lags(0)) = iParameters.submat(0,0,iOrders(0)*lags(0),0);
    maPolynomial.rows(0,maOrders(0)*lags(0)) = maParameters.submat(0,0,maOrders(0)*lags(0),0);

    for(unsigned int i=0; i<lags.n_rows; ++i){
// Form polynomials
        if(i!=0){
            bufferPolynomial = polyMult(arPolynomial, arParameters.col(i));
            arPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;

            bufferPolynomial = polyMult(maPolynomial, maParameters.col(i));
            maPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;

            bufferPolynomial = polyMult(iPolynomial, iParameters.col(i));
            iPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;
        }
        if(iOrders(i)>1){
            for(unsigned int j=1; j<iOrders(i); ++j){
                bufferPolynomial = polyMult(iPolynomial, iParameters.col(i));
                iPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;
            }
        }

    }
    // ariPolynomial contains 1 in the first place
    ariPolynomial = polyMult(arPolynomial, iPolynomial);

    // Check if the length of polynomials is correct. Fix if needed
    // This might happen if one of parameters became equal to zero
    if(maPolynomial.n_rows!=sum(maOrders % lags)+1){
        maPolynomial.resize(sum(maOrders % lags)+1);
    }
    if(ariPolynomial.n_rows!=sum(arOrders % lags)+sum(iOrders % lags)+1){
        ariPolynomial.resize(sum(arOrders % lags)+sum(iOrders % lags)+1);
    }
    if(arPolynomial.n_rows!=sum(arOrders % lags)+1){
        arPolynomial.resize(sum(arOrders % lags)+1);
    }

    return wrap(List::create(Named("arPolynomial") = arPolynomial, Named("iPolynomial") = iPolynomial,
                             Named("ariPolynomial") = ariPolynomial, Named("maPolynomial") = maPolynomial));
}
