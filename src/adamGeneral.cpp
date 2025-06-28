#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "ssGeneral.h"
#include "adamGeneral.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// # Fitter for univariate models
List adamFitter(arma::mat &matrixVt, arma::mat const &matrixWt, arma::mat &matrixF, arma::vec const &vectorG,
                arma::uvec &lags, arma::umat const &indexLookupTable, arma::mat profilesRecent,
                char const &E, char const &T, char const &S,
                unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                arma::vec const &vectorYt, arma::vec const &vectorOt, bool const &backcast,
                unsigned int const &nIterations, bool const &refineHead){
    /* # matrixVt should have a length of obs + lagsModelMax.
     * # matrixWt is a matrix with nrows = obs
     * # vecG should be a vector
     * # lags is a vector of lags
     */

    int obs = vectorYt.n_rows;
    unsigned int nETS = nNonSeasonal + nSeasonal;
    int nComponents = matrixVt.n_rows;
    int lagsModelMax = max(lags);

    // Fitted values and the residuals
    arma::vec vecYfit(obs, arma::fill::zeros);
    arma::vec vecErrors(obs, arma::fill::zeros);

    // These are objects used in backcasting.
    // Needed for some experiments.
    arma::mat &matrixFInv = matrixF;
    arma::vec const &vectorGInv = vectorG;

    // if(!inv(matrixFInv, matrixF)){
    //     matrixFInv = matrixF;
    //     vectorGInv = vectorG;
    // }
    // else{
    //     vectorGInv = matrixFInv * vectorG;
    // }

    // Loop for the backcast
    for (unsigned int j=1; j<=nIterations; j=j+1) {

        // Refine the head (in order for it to make sense)
        // This is only needed for ETS(*,Z,*) models, with trend.
        if(refineHead){
            for (int i=0; i<lagsModelMax; i=i+1) {
                matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                               matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
            }
        }
        ////// Run forward
        // Loop for the model construction
        for (int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
            matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));

            /* # Measurement equation and the error term */
            vecYfit(i-lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                    matrixWt.row(i-lagsModelMax), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

            // If this is zero (intermittent), then set error to zero
            if(vectorOt(i-lagsModelMax)==0){
                vecErrors(i-lagsModelMax) = 0;
            }
            else{
                // We need this multiplication for cases, when occurrence is fractional
                vecYfit(i-lagsModelMax) = vectorOt(i-lagsModelMax)*vecYfit(i-lagsModelMax);
                vecErrors(i-lagsModelMax) = errorf(vectorYt(i-lagsModelMax), vecYfit(i-lagsModelMax), E);
            }

            /* # Transition equation */
            profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(i-lagsModelMax), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant, vectorG, vecErrors(i-lagsModelMax));

        }

        ////// Backwards run
        if(backcast && j<(nIterations)){
            // Change the specific element in the state vector to negative
            if(T=='A'){
                profilesRecent(1) = -profilesRecent(1);
            }
            else if(T=='M'){
                profilesRecent(1) = 1/profilesRecent(1);
            }

            for (int i=obs+lagsModelMax-1; i>=lagsModelMax; i=i-1) {
                /* # Measurement equation and the error term */
                vecYfit(i-lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                        matrixWt.row(i-lagsModelMax), E, T, S,
                        nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                // If this is zero (intermittent), then set error to zero
                if(vectorOt(i-lagsModelMax)==0){
                    vecErrors(i-lagsModelMax) = 0;
                }
                else{
                    // We need this multiplication for cases, when occurrence is fractional
                    vecYfit(i-lagsModelMax) = vectorOt(i-lagsModelMax)*vecYfit(i-lagsModelMax);
                    vecErrors(i-lagsModelMax) = errorf(vectorYt(i-lagsModelMax), vecYfit(i-lagsModelMax), E);
                }

                /* # Transition equation */
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                               matrixFInv, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                   adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixFInv,
                                              matrixWt.row(i-lagsModelMax), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                              vectorGInv, vecErrors(i-lagsModelMax));
            }

            // Fill in the head of the series.
            if(refineHead){
                for (int i=lagsModelMax-1; i>=0; i=i-1) {
                    profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                   matrixFInv, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);

                    // matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
                }
            }

            // Change back the specific element in the state vector
            if(T=='A'){
                profilesRecent(1) = -profilesRecent(1);
                // Write down correct initials
                // This is needed in case the profileRecent has changed in previous lines
                // matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
            }
            else if(T=='M'){
                profilesRecent(1) = 1/profilesRecent(1);
                // Write down correct initials
                // This is needed in case the profileRecent has changed in previous lines
                // matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
            }
        }
    }

    return List::create(Named("matVt") = matrixVt, Named("yFitted") = vecYfit,
                        Named("errors") = vecErrors, Named("profile") = profilesRecent);
}

/* # Wrapper for fitter */
// [[Rcpp::export]]
RcppExport SEXP adamFitterWrap(arma::mat matrixVt, arma::mat &matrixWt, arma::mat &matrixF, arma::vec &vectorG,
                               arma::uvec &lags, arma::umat &indexLookupTable, arma::mat &profilesRecent,
                               char const &Etype, char const &Ttype, char const &Stype,
                               unsigned int const &componentsNumberETS, unsigned int const &nSeasonal,
                               unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                               arma::vec &vectorYt, arma::vec &vectorOt, bool const &backcast,
                               unsigned int const &nIterations, bool const &refineHead){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamFitter(matrixVt, matrixWt, matrixF, vectorG,
                           lags, indexLookupTable, profilesRecent, Etype, Ttype, Stype,
                           nNonSeasonal, nSeasonal, nArima, nXreg, constant,
                           vectorYt, vectorOt, backcast, nIterations, refineHead));
}

/* # Function produces the point forecasts for the specified model */
arma::vec adamForecaster(arma::mat const &matrixWt, arma::mat const &matrixF,
                         arma::uvec lags, arma::umat const &indexLookupTable, arma::mat profilesRecent,
                         char const &E, char const &T, char const &S,
                         unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                         unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                         unsigned int const &horizon){
    // unsigned int lagslength = lags.n_rows;
    unsigned int nETS = nNonSeasonal + nSeasonal;
    unsigned int nComponents = indexLookupTable.n_rows;

    arma::vec vecYfor(horizon, arma::fill::zeros);

    /* # Fill in the new xt matrix using F. Do the forecasts. */
    for (unsigned int i=0; i<horizon; i=i+1) {
        vecYfor.row(i) = adamWvalue(profilesRecent(indexLookupTable.col(i)), matrixWt.row(i), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

        profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                       matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
    }

    // return List::create(Named("matVt") = matrixVtnew, Named("yForecast") = vecYfor);
    return vecYfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP adamForecasterWrap(arma::mat &matrixWt, arma::mat &matrixF,
                                   arma::uvec &lags, arma::umat &indexLookupTable, arma::mat &profilesRecent,
                                   char const &E, char const &T, char const &S,
                                   unsigned int const &componentsNumberETS, unsigned int const &nSeasonal,
                                   unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                                   unsigned int const &horizon){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamForecaster(matrixWt, matrixF,
                               lags, indexLookupTable, profilesRecent,
                               E, T, S,
                               nNonSeasonal, nSeasonal,
                               nArima, nXreg, constant,
                               horizon));
}

/* # Function produces matrix of errors based on multisteps forecast */
arma::mat adamErrorer(arma::mat const &matrixVt, arma::mat const &matrixWt, arma::mat const &matrixF,
                      arma::uvec &lags, arma::umat const &indexLookupTable, arma::mat profilesRecent,
                      char const &E, char const &T, char const &S,
                      unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                      unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                      unsigned int const &horizon,
                      arma::vec const &vectorYt, arma::vec const &vectorOt){
    unsigned int obs = vectorYt.n_rows;
    unsigned int lagsModelMax = max(lags);
    // This is needed for cases, when hor>obs
    unsigned int hh = 0;
    arma::mat matErrors(horizon, obs, arma::fill::zeros);

    // Fill in the head, similar to how it's done in the fitter
    for (unsigned int i=0; i<lagsModelMax; i=i+1) {
        profilesRecent(indexLookupTable.col(i)) = matrixVt.col(i);
    }

    for(unsigned int i = 0; i < (obs-horizon); i=i+1){
        hh = std::min(horizon, obs-i);
        // Update the profile to get the recent value from the state matrix
        profilesRecent(indexLookupTable.col(i+lagsModelMax-1)) = matrixVt.col(i+lagsModelMax-1);
        // profilesRecent(indexLookupTable.col(i)) = matrixVt.col(i);
        // This also needs to take probability into account in order to deal with intermittent models
        matErrors.submat(0, i, hh-1, i) = (errorvf(vectorYt.rows(i, i+hh-1),
                                           adamForecaster(matrixWt.rows(i,i+hh-1), matrixF,
                                                          lags, indexLookupTable.cols(i+lagsModelMax,i+lagsModelMax+hh-1), profilesRecent,
                                                          E, T, S, nNonSeasonal, nSeasonal, nArima, nXreg, constant, hh), E));
    }

    // Cut-off the redundant last part
    if(obs>horizon){
        matErrors = matErrors.cols(0,obs-horizon-1);
    }

    // Fix for GV in order to perform better in the sides of the series
    // for(int i=0; i<(hor-1); i=i+1){
    //     matErrors.submat((hor-2)-(i),i+1,(hor-2)-(i),hor-1) = matErrors.submat(hor-1,0,hor-1,hor-i-2) * sqrt(1.0+i);
    // }

    return matErrors.t();
}

/* # Wrapper for error function */
// [[Rcpp::export]]
RcppExport SEXP adamErrorerWrap(arma::mat matrixVt, arma::mat matrixWt, arma::mat matrixF,
                                arma::uvec lags, arma::umat indexLookupTable, arma::mat profilesRecent,
                                char Etype, char Ttype, char Stype,
                                unsigned int &componentsNumberETS, unsigned int &nSeasonal,
                                unsigned int nArima, unsigned int nXreg, bool constant,
                                unsigned int horizon, arma::vec vectorYt, arma::vec vectorOt){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamErrorer(matrixVt, matrixWt, matrixF,
                            lags, indexLookupTable, profilesRecent,
                            Etype, Ttype, Stype,
                            nNonSeasonal, nSeasonal, nArima, nXreg, constant,
                            horizon, vectorYt, vectorOt));
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
