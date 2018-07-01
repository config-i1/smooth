#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;


// ##### Simulator for vector models
List vSimulator(arma::cube &cubeStates, arma::cube const &cubeErrors, arma::cube const &cubeF,
                arma::cube const &cubeW, arma::cube const &cubeG, arma::uvec &lags,
                unsigned int const &obs, unsigned int const &nSeries, unsigned int const &nSim) {

    arma::cube cubeY(nSeries, obs, nSim);

    int lagsLength = lags.n_rows;
    unsigned int lagsMax = max(lags);
    int obsall = obs + lagsMax;

    lags = lags * lagsLength;

    for(int i=0; i<lagsLength; i=i+1){
        lags(i) = lags(i) + (lagsLength - i - 1);
    }

    arma::uvec lagrows(lagsLength, arma::fill::zeros);
    arma::mat matrixStates(lagsLength, obsall, arma::fill::zeros);
    arma::mat matrixF(cubeF.n_rows, cubeF.n_cols, arma::fill::zeros);
    arma::mat matrixW(cubeW.n_rows, cubeW.n_cols, arma::fill::zeros);
    arma::mat matrixG(cubeG.n_rows, cubeG.n_cols, arma::fill::zeros);

    // Matrices for the interim operations
    arma::mat matrixY(nSeries, obs, arma::fill::zeros);
    arma::mat matrixErrors(nSeries, obs, arma::fill::zeros);

    for(unsigned int i=0; i<nSim; i=i+1){
        matrixStates = cubeStates.slice(i);
        matrixF = cubeF.slice(i);
        matrixW = cubeW.slice(i);
        matrixG = cubeG.slice(i);
        matrixErrors = cubeErrors.slice(i);

        for (int j=lagsMax; j<obsall; j=j+1) {
            lagrows = (j+1) * lagsLength - lags - 1;

            /* # Measurement equation and the error term */
            matrixY.col(j-lagsMax) = (matrixW * matrixStates(lagrows) + matrixErrors.col(j-lagsMax));

            /* # Transition equation */
            matrixStates.col(j) = matrixF * matrixStates(lagrows) + matrixG * matrixErrors.col(j-lagsMax);
        }
        cubeStates.slice(i) = matrixStates;
        cubeY.slice(i) = matrixY;
    }

    return List::create(Named("arrayStates") = cubeStates, Named("arrayActuals") = cubeY);
}

/* # Wrapper for simulator */
// [[Rcpp::export]]
RcppExport SEXP vSimulatorWrap(SEXP arrayStates, SEXP arrayErrors, SEXP arrayF,
                               SEXP arrayW, SEXP arrayG, SEXP modelLags) {

    // ### arrayStates should contain array of obs x ncomponents*nSeries x nsim elements.
    NumericVector arrayStates_n(arrayStates);
    IntegerVector arrayStates_dim = arrayStates_n.attr("dim");
    arma::cube cubeStates(arrayStates_n.begin(), arrayStates_dim[0], arrayStates_dim[1], arrayStates_dim[2], false);

    NumericVector arrayErrors_n(arrayErrors);
    IntegerVector arrayErrors_dim = arrayErrors_n.attr("dim");
    unsigned int nSeries = arrayErrors_dim[0];
    unsigned int obs = arrayErrors_dim[1];
    unsigned int nSim = arrayErrors_dim[2];
    arma::cube cubeErrors(arrayErrors_n.begin(), arrayErrors_dim[0], arrayErrors_dim[1], arrayErrors_dim[2], false);

    NumericVector arrayF_n(arrayF);
    IntegerVector arrayF_dim = arrayF_n.attr("dim");
    arma::cube cubeF(arrayF_n.begin(), arrayF_dim[0], arrayF_dim[1], arrayF_dim[2], false);

    NumericVector arrayW_n(arrayW);
    IntegerVector arrayW_dim = arrayW_n.attr("dim");
    arma::cube cubeW(arrayW_n.begin(), arrayW_dim[0], arrayW_dim[1], arrayW_dim[2], false);

    NumericVector arrayG_n(arrayG);
    IntegerVector arrayG_dim = arrayG_n.attr("dim");
    arma::cube cubeG(arrayG_n.begin(), arrayG_dim[0], arrayG_dim[1], arrayG_dim[2], false);

    IntegerVector modellags_n(modelLags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    return wrap(vSimulator(cubeStates, cubeErrors, cubeF, cubeW, cubeG, lags, obs, nSeries, nSim));

}
