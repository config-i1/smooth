#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "ssGeneral.h"
#include "adamGeneral.h"

using namespace Rcpp;

// ============================================================================
// STRUCTURE DEFINITIONS
// ============================================================================

// Result structure for fitter
struct FitResult {
    arma::mat matVt;
    arma::vec yFitted;
    arma::vec errors;
    arma::mat profile;
};

// Result structure for forecaster
struct ForecastResult {
    arma::vec yForecast;
};

// Result structure for errorer
struct ErrorResult {
    arma::mat matErrors;
};

// Result structure for simulator
struct SimulateResult {
    arma::cube arrayVt;
    arma::mat matrixYt;
};

// ============================================================================
// ADAMCORE CLASS
// ============================================================================

class adamCore {
public:
    // Method 1: Fitter
    FitResult fit(arma::mat &matrixVt, arma::mat const &matrixWt,
                  arma::mat &matrixF, arma::vec const &vectorG,
                  arma::uvec &lags, arma::umat const &indexLookupTable,
                  arma::mat profilesRecent,
                  char const &E, char const &T, char const &S,
                  unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                  unsigned int const &nETS, unsigned int const &nArima, unsigned int const &nXreg,
                  bool const &constant, arma::vec const &vectorYt, arma::vec const &vectorOt,
                  bool const &backcast, unsigned int const &nIterations, bool const &refineHead,
                  bool const &adamETS) {
        /* # matrixVt should have a length of obs + lagsModelMax.
         * # matrixWt is a matrix with nrows = obs
         * # vecG should be a vector
         * # lags is a vector of lags
         */

        int obs = vectorYt.n_rows;
        int nComponents = matrixVt.n_rows;
        int lagsModelMax = max(lags);

        // Fitted values and the residuals
        arma::vec vecYfit(obs, arma::fill::zeros);
        arma::vec vecErrors(obs, arma::fill::zeros);

        // These are objects used in backcasting.
        // Needed for some experiments.
        arma::mat &matrixFInv = matrixF;
        arma::vec const &vectorGInv = vectorG;

        // Loop for the backcast
        for (unsigned int j=1; j<=nIterations; j=j+1) {

            // Refine the head (in order for it to make sense)
            // This is only needed for ETS(*,Z,Z) models, with trend.
            // This is not needed for lagsMax=1, because there is nothing to fill in
            if(refineHead && (T!='N')){
                // Record the initial profile to the first column
                matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
                if(lagsModelMax>1){
                    // Update the head, but only for the trend component
                    for (int i=1; i<lagsModelMax; i=i+1) {
                        profilesRecent(indexLookupTable.col(i).rows(0,1)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                       matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant).rows(0,1);
                        matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
                    }
                }
            }
            ////// Run forward
            // Loop for the model construction
            for (int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
                /* # Measurement equation and the error term */
                vecYfit(i-lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
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
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                               matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                   adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(i-lagsModelMax), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                              vectorG, vecErrors(i-lagsModelMax), vecYfit(i-lagsModelMax), adamETS);

                matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));

                // If ot is fractional, amend the fitted value
                if(vectorOt(i-lagsModelMax)!=0 && vectorOt(i-lagsModelMax)!=1){
                    // We need this multiplication for cases, when occurrence is fractional
                    vecYfit(i-lagsModelMax) = vectorOt(i-lagsModelMax)*vecYfit(i-lagsModelMax);
                }
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
                                                  vectorGInv, vecErrors(i-lagsModelMax), vecYfit(i-lagsModelMax), adamETS);
                }

                // Fill in the head of the series.
                if(refineHead){
                    for (int i=lagsModelMax-1; i>=0; i=i-1) {
                        profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                       matrixFInv, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
                    }
                }

                // Change back the specific element in the state vector
                if(T=='A'){
                    profilesRecent(1) = -profilesRecent(1);
                }
                else if(T=='M'){
                    profilesRecent(1) = 1/profilesRecent(1);
                }
            }
        }

        FitResult result;
        result.matVt = matrixVt;
        result.yFitted = vecYfit;
        result.errors = vecErrors;
        result.profile = profilesRecent;
        return result;
    }

    // Method 2: Forecaster
    ForecastResult forecast(arma::mat const &matrixWt, arma::mat const &matrixF,
                            arma::uvec lags, arma::umat const &indexLookupTable,
                            arma::mat profilesRecent,
                            char const &E, char const &T, char const &S,
                            unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                            unsigned int const &nETS, unsigned int const &nArima, unsigned int const &nXreg,
                            bool const &constant,
                            unsigned int const &horizon) {

        unsigned int nComponents = indexLookupTable.n_rows;

        arma::vec vecYfor(horizon, arma::fill::zeros);

        /* # Fill in the new xt matrix using F. Do the forecasts. */
        for (unsigned int i=0; i<horizon; i=i+1) {
            vecYfor.row(i) = adamWvalue(profilesRecent(indexLookupTable.col(i)), matrixWt.row(i), E, T, S,
                        nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

            profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
        }

        ForecastResult result;
        result.yForecast = vecYfor;
        return result;
    }

    // Method 3: Errorer
    ErrorResult error(arma::mat matrixVt, arma::mat matrixWt, arma::mat matrixF,
                          arma::uvec lags, arma::umat const &indexLookupTable,
                          arma::mat profilesRecent,
                          char const &E, char const &T, char const &S,
                          unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                          unsigned int const &nETS, unsigned int const &nArima, unsigned int const &nXreg,
                          bool const &constant, unsigned int const &horizon,
                          arma::vec vectorYt) {
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
            // lagsModelMax moves the thing to the next obs. This way, we have the structure
            // similar to the fitter
            profilesRecent(indexLookupTable.col(i+lagsModelMax)) = matrixVt.col(i+lagsModelMax);
            // This needs to take probability of occurrence into account in order to deal with intermittent models
            // The problem is that the probability needs to be a matrix, i.e. to reflect multistep from each point
            matErrors.submat(0, i, hh-1, i) =
                errorvf(vectorYt.rows(i, i+hh-1),
                        forecast(matrixWt.rows(i,i+hh-1), matrixF,
                                   lags, indexLookupTable.cols(i+lagsModelMax,i+lagsModelMax+hh-1), profilesRecent,
                                   E, T, S, nNonSeasonal, nSeasonal, nETS, nArima, nXreg, constant, hh).yForecast,
                                       // vectorPt.rows(i, i+hh-1),
                                   E);
        }

        // Cut-off the redundant last part
        if(obs>horizon){
            matErrors = matErrors.cols(0,obs-horizon-1);
        }

        ErrorResult result;
        result.matErrors = matErrors.t();
        return result;
    }

    // Method 4: Simulator
    SimulateResult simulate(arma::cube &arrayVt, arma::mat const &matrixErrors, arma::mat const &matrixOt,
                            arma::cube const &arrayF, arma::mat const &matrixWt, arma::mat const &matrixG,
                            arma::uvec &lags, arma::umat const &indexLookupTable,
                            arma::mat profilesRecent,
                            char const &E, char const &T, char const &S,
                            unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                            unsigned int const &nETS, unsigned int const &nArima, unsigned int const &nXreg,
                            bool const &constant,
                            bool const &adamETS){

        unsigned int obs = matrixErrors.n_rows;
        unsigned int nSeries = matrixErrors.n_cols;

        int lagsModelMax = max(lags);
        int nComponents = lags.n_rows;
        int obsAll = obs + lagsModelMax;
        arma::mat profilesRecentOriginal = profilesRecent;

        double yFitted;

        arma::mat matrixVt(nComponents, obsAll, arma::fill::zeros);
        arma::mat matrixF(arrayF.n_rows, arrayF.n_cols, arma::fill::zeros);

        arma::mat matY(obs, nSeries);

        for(unsigned int i=0; i<nSeries; i=i+1){
            matrixVt = arrayVt.slice(i);
            matrixF = arrayF.slice(i);
            profilesRecent = profilesRecentOriginal;
            for(int j=lagsModelMax; j<obsAll; j=j+1) {
                /* # Measurement equation and the error term */
                yFitted = adamWvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                                     matrixWt.row(j-lagsModelMax), E, T, S,
                                     nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                     nComponents, constant);
                matY(j-lagsModelMax,i) = matrixOt(j-lagsModelMax,i) *
                    (yFitted +
                    adamRvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                               matrixWt.row(j-lagsModelMax), E, T, S,
                               nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) *
                                   matrixErrors(j-lagsModelMax,i));

                /* # Transition equation */
                profilesRecent(indexLookupTable.col(j-lagsModelMax)) =
                (adamFvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                            matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                            nComponents, constant) +
                                adamGvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                                           matrixF, matrixWt.row(j-lagsModelMax),
                                           E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                           nComponents, constant, matrixG.col(i),
                                           matrixErrors(j-lagsModelMax,i), yFitted, adamETS));

                matrixVt.col(j) = profilesRecent(indexLookupTable.col(j-lagsModelMax));
            }
            arrayVt.slice(i) = matrixVt;
        }

        SimulateResult result;
        result.arrayVt = arrayVt;
        result.matrixYt = matY;
        return result;
    }
};
