#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "ssGeneral.h"
#include "adamGeneral.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// # Fitter for univariate models
List adamFitter(arma::mat &matrixVt, arma::mat const &matrixWt, arma::mat &matrixF, arma::vec const &vectorG,
                arma::uvec &lags, arma::umat const &profilesObserved, arma::mat profilesRecent,
                char const &E, char const &T, char const &S,
                unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                arma::vec const &vectorYt, arma::vec const &vectorOt, bool const &backcast){
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

    // Loop for the backcasting
    unsigned int nIterations = 1;
    if(backcast){
        nIterations = 2;
    }

    // Loop for the backcast
    for (unsigned int j=1; j<=nIterations; j=j+1) {

        // Refine the head (in order for it to make sense)
        // This is only needed for ETS(*,Z,*) models, with trend.
        // if(!backcast){
        for (int i=0; i<lagsModelMax; i=i+1) {
            matrixVt.col(i) = profilesRecent(profilesObserved.col(i));
            profilesRecent(profilesObserved.col(i)) = adamFvalue(profilesRecent(profilesObserved.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
        }
        // }
        ////// Run forward
        // Loop for the model construction
        for (int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {

            /* # Measurement equation and the error term */
            vecYfit(i-lagsModelMax) = adamWvalue(profilesRecent(profilesObserved.col(i)),
                    matrixWt.row(i-lagsModelMax), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

            // If this is zero (intermittent), then set error to zero
            if(vectorOt(i-lagsModelMax)==0){
                vecErrors(i-lagsModelMax) = 0;
            }
            else{
                vecErrors(i-lagsModelMax) = errorf(vectorYt(i-lagsModelMax), vecYfit(i-lagsModelMax), E);
            }

            /* # Transition equation */
            profilesRecent(profilesObserved.col(i)) = adamFvalue(profilesRecent(profilesObserved.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                adamGvalue(profilesRecent(profilesObserved.col(i)), matrixF, matrixWt.row(i-lagsModelMax), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant, vectorG, vecErrors(i-lagsModelMax));

            matrixVt.col(i) = profilesRecent(profilesObserved.col(i));
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
                vecYfit(i-lagsModelMax) = adamWvalue(profilesRecent(profilesObserved.col(i)),
                        matrixWt.row(i-lagsModelMax), E, T, S,
                        nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                // If this is zero (intermittent), then set error to zero
                if(vectorOt(i-lagsModelMax)==0){
                    vecErrors(i-lagsModelMax) = 0;
                }
                else{
                    vecErrors(i-lagsModelMax) = errorf(vectorYt(i-lagsModelMax), vecYfit(i-lagsModelMax), E);
                }

                /* # Transition equation */
                profilesRecent(profilesObserved.col(i)) = adamFvalue(profilesRecent(profilesObserved.col(i)),
                               matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                   adamGvalue(profilesRecent(profilesObserved.col(i)), matrixF,
                                              matrixWt.row(i-lagsModelMax), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                              vectorG, vecErrors(i-lagsModelMax));
            }

            // Fill in the head of the series.
            // if(nArima==0){
                for (int i=lagsModelMax-1; i>=0; i=i-1) {
                    profilesRecent(profilesObserved.col(i)) = adamFvalue(profilesRecent(profilesObserved.col(i)),
                                   matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);

                    matrixVt.col(i) = profilesRecent(profilesObserved.col(i));
                }
            // }

            // Change back the specific element in the state vector
            if(T=='A'){
                profilesRecent(1) = -profilesRecent(1);
            }
            else if(T=='M'){
                profilesRecent(1) = 1/profilesRecent(1);
            }
        }
    }

    return List::create(Named("matVt") = matrixVt, Named("yFitted") = vecYfit,
                        Named("errors") = vecErrors, Named("profile") = profilesRecent);
}

/* # Wrapper for fitter */
// [[Rcpp::export]]
RcppExport SEXP adamFitterWrap(arma::mat matrixVt, arma::mat &matrixWt, arma::mat &matrixF, arma::vec &vectorG,
                               arma::uvec &lags, arma::umat &profilesObserved, arma::mat &profilesRecent,
                               char const &Etype, char const &Ttype, char const &Stype,
                               unsigned int const &componentsNumberETS, unsigned int const &nSeasonal,
                               unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                               arma::vec &vectorYt, arma::vec &vectorOt, bool const &backcast){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamFitter(matrixVt, matrixWt, matrixF, vectorG,
                           lags, profilesObserved, profilesRecent, Etype, Ttype, Stype,
                           nNonSeasonal, nSeasonal, nArima, nXreg, constant,
                           vectorYt, vectorOt, backcast));
}

/* # Function produces the point forecasts for the specified model */
arma::vec adamForecaster(arma::mat const &matrixWt, arma::mat const &matrixF,
                         arma::uvec lags, arma::umat const &profilesObserved, arma::mat profilesRecent,
                         char const &E, char const &T, char const &S,
                         unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                         unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                         unsigned int const &horizon){
    // unsigned int lagslength = lags.n_rows;
    unsigned int nETS = nNonSeasonal + nSeasonal;
    unsigned int nComponents = profilesObserved.n_rows;

    arma::vec vecYfor(horizon, arma::fill::zeros);

    /* # Fill in the new xt matrix using F. Do the forecasts. */
    for (unsigned int i=0; i<horizon; i=i+1) {
        vecYfor.row(i) = adamWvalue(profilesRecent(profilesObserved.col(i)), matrixWt.row(i), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

        profilesRecent(profilesObserved.col(i)) = adamFvalue(profilesRecent(profilesObserved.col(i)),
                       matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
    }

    // return List::create(Named("matVt") = matrixVtnew, Named("yForecast") = vecYfor);
    return vecYfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP adamForecasterWrap(arma::mat &matrixWt, arma::mat &matrixF,
                                   arma::uvec &lags, arma::umat &profilesObserved, arma::mat &profilesRecent,
                                   char const &E, char const &T, char const &S,
                                   unsigned int const &componentsNumberETS, unsigned int const &nSeasonal,
                                   unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                                   unsigned int const &horizon){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamForecaster(matrixWt, matrixF,
                               lags, profilesObserved, profilesRecent,
                               E, T, S,
                               nNonSeasonal, nSeasonal,
                               nArima, nXreg, constant,
                               horizon));
}

/* # Function produces matrix of errors based on multisteps forecast */
arma::mat adamErrorer(arma::mat const &matrixVt, arma::mat const &matrixWt, arma::mat const &matrixF,
                      arma::uvec &lags, arma::umat const &profilesObserved, arma::mat profilesRecent,
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
        profilesRecent(profilesObserved.col(i)) = matrixVt.col(i);
    }

    for(unsigned int i = 0; i < (obs-horizon); i=i+1){
        hh = std::min(horizon, obs-i);
        // Update the profile to get the recent value from the state matrix
        profilesRecent(profilesObserved.col(i+lagsModelMax-1)) = matrixVt.col(i+lagsModelMax-1);
        // profilesRecent(profilesObserved.col(i)) = matrixVt.col(i);
        // This also needs to take probability into account in order to deal with intermittent models
        matErrors.submat(0, i, hh-1, i) = (errorvf(vectorYt.rows(i, i+hh-1),
                                           adamForecaster(matrixWt.rows(i,i+hh-1), matrixF,
                                                          lags, profilesObserved.cols(i+lagsModelMax,i+lagsModelMax+hh-1), profilesRecent,
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
                                arma::uvec lags, arma::umat profilesObserved, arma::mat profilesRecent,
                                char Etype, char Ttype, char Stype,
                                unsigned int &componentsNumberETS, unsigned int &nSeasonal,
                                unsigned int nArima, unsigned int nXreg, bool constant,
                                unsigned int horizon, arma::vec vectorYt, arma::vec vectorOt){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamErrorer(matrixVt, matrixWt, matrixF,
                            lags, profilesObserved, profilesRecent,
                            Etype, Ttype, Stype,
                            nNonSeasonal, nSeasonal, nArima, nXreg, constant,
                            horizon, vectorYt, vectorOt));
}
