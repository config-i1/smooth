#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// Helper: Invert measurement matrix (1/x, with inf->0)
arma::mat measurementInverterCpp(const arma::mat& measurement) {
    arma::mat result = 1.0 / measurement;
    result.replace(arma::datum::inf, 0.0);
    result.replace(-arma::datum::inf, 0.0);
    return result;
}

// [[Rcpp::export]]
arma::vec smoothEigensCpp(const arma::mat& persistence,
                          const arma::mat& transition,
                          const arma::mat& measurement,
                          const arma::ivec& lagsModelAll,
                          bool xregModel,
                          int obsInSample,
                          bool hasDelta) {

    arma::ivec lagsUnique = arma::unique(lagsModelAll);
    int lagsUniqueLength = lagsUnique.n_elem;
    int nComponents = lagsModelAll.n_elem;
    arma::vec eigenValues(nComponents, arma::fill::zeros);

    if (xregModel && hasDelta) {
        // xreg case: compute eigenvalues on average condition
        arma::mat measSub = measurement.rows(0, obsInSample - 1);
        arma::mat measInv = measurementInverterCpp(measSub);
        arma::mat matToDecomp = transition -
            arma::diagmat(persistence) * measInv.t() * measSub / obsInSample;

        arma::cx_vec eigVals = arma::eig_gen(matToDecomp);
        return arma::abs(eigVals);
    }
    else {
        // Normal case: loop through unique lags
        for (int i = 0; i < lagsUniqueLength; i++) {
            // Find indices where lagsModelAll == lagsUnique[i]
            arma::uvec idx = arma::find(lagsModelAll == lagsUnique[i]);

            // Extract submatrices
            arma::mat transSub = transition.submat(idx, idx);
            arma::mat persSub = persistence.rows(idx);
            arma::rowvec measRow = measurement.row(obsInSample - 1);
            arma::mat measSub = measRow.cols(idx);

            // Compute: transition_sub - persistence_sub * measurement_sub
            arma::mat matToDecomp = transSub - persSub * measSub;

            // Get eigenvalues
            arma::cx_vec eigVals = arma::eig_gen(matToDecomp);
            arma::vec absEigVals = arma::abs(eigVals);

            // Assign to result
            for (arma::uword j = 0; j < idx.n_elem; j++) {
                eigenValues(idx(j)) = absEigVals(j);
            }
        }
    }

    return eigenValues;
}
