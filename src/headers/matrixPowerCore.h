#pragma once

// Assumes armadillo is included by the wrapper
// Function to compute matrix power
arma::mat matrixPowerCore(arma::mat const &matrixA, int power) {
    if(power == 0) {
        return arma::eye(matrixA.n_rows, matrixA.n_cols);
    }
    else if(power == 1) {
        return matrixA;
    }
    else if(power < 0) {
        // Negative power: compute inverse and raise to positive power
        arma::mat matrixInv = arma::inv(matrixA);
        return matrixPowerCore(matrixInv, -power);
    }
    else {
        // Positive power: use repeated multiplication
        arma::mat result = matrixA;
        for(int i = 1; i < power; i++) {
            result = result * matrixA;
        }
        return result;
    }
}
