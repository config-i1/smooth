// Helper: Invert measurement matrix (1/x, with inf->0)
arma::mat measurementInverterCpp(const arma::mat& measurement) {
    arma::mat result = 1.0 / measurement;
    result.replace(arma::datum::inf, 0.0);
    result.replace(-arma::datum::inf, 0.0);
    return result;
}

// Function to calculate eigenvalues
arma::vec smoothEigensCpp(const arma::mat& persistence,
                          const arma::mat& transition,
                          const arma::mat& measurement,
                          const arma::ivec& lagsModelAll,
                          bool& xregModel,
                          int& obsInSample,
                          bool& hasDelta,
                          int xregNumber = 0,
                          bool constantRequired = false) {

    int nComponents = lagsModelAll.n_elem;

    if (xregModel && hasDelta) {
        // Handle xreg with adaptive regressors
        // Non-xreg components use loop approach, xreg components use averaging

        int effectiveComponents = nComponents;
        // Drop constant if present (last element)
        if (constantRequired) {
            effectiveComponents -= 1;
        }

        arma::vec eigenValues(effectiveComponents, arma::fill::zeros);
        int nonXregEnd = effectiveComponents - xregNumber;

        // Part 1: Non-xreg components (loop approach)
        if (nonXregEnd > 0) {
            arma::ivec lagsNonXreg = lagsModelAll.head(nonXregEnd);
            arma::ivec lagsUniqueNonXreg = arma::unique(lagsNonXreg);
            arma::uvec nonXregIdx = arma::regspace<arma::uvec>(0, nonXregEnd - 1);

            for (arma::uword i = 0; i < lagsUniqueNonXreg.n_elem; i++) {
                // Find indices where lagsNonXreg == lagsUniqueNonXreg[i]
                arma::uvec idx = arma::find(lagsNonXreg == lagsUniqueNonXreg[i]);

                // Extract submatrices for non-xreg portion
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

        // Part 2: Xreg components (averaging approach)
        if (xregNumber > 0) {
            arma::uvec xregIdx = arma::regspace<arma::uvec>(nonXregEnd, effectiveComponents - 1);
            arma::mat transSub = transition.submat(xregIdx, xregIdx);
            arma::mat persSub = persistence.rows(xregIdx);
            arma::mat measSub = measurement.submat(
                arma::regspace<arma::uvec>(0, obsInSample - 1), xregIdx);
            arma::mat measInv = measurementInverterCpp(measSub);
            arma::mat matToDecomp = transSub -
                arma::diagmat(persSub) * measInv.t() * measSub / obsInSample;

            arma::cx_vec eigVals = arma::eig_gen(matToDecomp);
            arma::vec absEigVals = arma::abs(eigVals);

            for (int j = 0; j < xregNumber; j++) {
                eigenValues(nonXregEnd + j) = absEigVals(j);
            }
        }

        return eigenValues;
    }
    else {
        // Normal case: loop through unique lags
        arma::ivec lagsUnique = arma::unique(lagsModelAll);
        int lagsUniqueLength = lagsUnique.n_elem;
        arma::vec eigenValues(nComponents, arma::fill::zeros);

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

        return eigenValues;
    }
}
