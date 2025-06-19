#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "ssGeneral.h"
#include "adamGeneral.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// ##### Script for refitting functions
List adamRefitter(arma::mat const &matrixYt, arma::mat const &matrixOt, arma::cube &arrayVt, arma::cube const &arrayF,
                  arma::cube const &arrayWt, arma::mat const &matrixG,
                  char const &E, char const &T, char const &S, arma::uvec &lags,
                  arma::umat const &indexLookupTable, arma::cube arrayProfilesRecent,
                  unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                  unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                  bool const &backcast, bool const &refineHead) {

    int obs = matrixYt.n_rows;
    unsigned int nSeries = matrixG.n_cols;

    // nIterations=1 means that we don't do backcasting
    // It doesn't seem to matter anyway...
    unsigned int nIterations = 1;
    if(backcast){
        nIterations = 2;
    }

    int lagsModelMax = max(lags);
    unsigned int nETS = nNonSeasonal + nSeasonal;
    int nComponents = lags.n_rows;

    arma::mat matYfit(obs, nSeries, arma::fill::zeros);
    arma::vec vecErrors(obs, arma::fill::zeros);

    for(unsigned int k=0; k<nSeries; k=k+1){
        // Loop for the backcasting
        for (unsigned int j=1; j<=nIterations; j=j+1) {
            // Refine the head (in order for it to make sense)
            if(refineHead){
                for(int i=0; i<lagsModelMax; i=i+1) {
                    arrayVt.slice(k).col(i) = arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i));
                    arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                        adamFvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                   arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
                }
            }
            // Loop for the model construction
            for(int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
                /* # Measurement equation and the error term */
                matYfit(i-lagsModelMax,k) = adamWvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                        arrayWt.slice(k).row(i-lagsModelMax), E, T, S,
                        nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                // Fix potential issue with negatives in mixed models
                if((E=='M' || T=='M' || S=='M') && (matYfit(i-lagsModelMax,k)<=0)){
                    matYfit(i-lagsModelMax,k) = 1;
                }

                // If this is zero (intermittent), then set error to zero
                if(matrixOt(i-lagsModelMax)==0){
                    vecErrors(i-lagsModelMax) = 0;
                }
                else{
                    vecErrors(i-lagsModelMax) = errorf(matrixYt(i-lagsModelMax), matYfit(i-lagsModelMax,k), E);
                }

                /* # Transition equation */
                arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                adamFvalue(arrayProfilesRecent.slice(k)(indexLookupTable.col(i)),
                           arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                               adamGvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                          arrayF.slice(k), arrayWt.slice(k).row(i-lagsModelMax), E, T, S,
                                          nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                          matrixG.col(k), vecErrors(i-lagsModelMax));

                arrayVt.slice(k).col(i) = arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i));
            }

            ////// Backwards run
            if(backcast && j<(nIterations)){
                // Change the specific element in the state vector to negative
                if(T=='A'){
                    arrayProfilesRecent.slice(k)(1) = -arrayProfilesRecent.slice(k)(1);
                }
                else if(T=='M'){
                    arrayProfilesRecent.slice(k)(1) = 1/arrayProfilesRecent.slice(k)(1);
                }

                for(int i=obs+lagsModelMax-1; i>=lagsModelMax; i=i-1) {
                    /* # Measurement equation and the error term */
                    matYfit(i-lagsModelMax,k) = adamWvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                            arrayWt.slice(k).row(i-lagsModelMax), E, T, S,
                            nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                    // Fix potential issue with negatives in mixed models
                    if((E=='M' || T=='M' || S=='M') && (matYfit(i-lagsModelMax,k)<=0)){
                        matYfit(i-lagsModelMax,k) = 1;
                    }

                    // If this is zero (intermittent), then set error to zero
                    if(matrixOt(i-lagsModelMax)==0){
                        vecErrors(i-lagsModelMax) = 0;
                    }
                    else{
                        vecErrors(i-lagsModelMax) = errorf(matrixYt(i-lagsModelMax), matYfit(i-lagsModelMax,k), E);
                    }

                    /* # Transition equation */
                    arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                    adamFvalue(arrayProfilesRecent.slice(k)(indexLookupTable.col(i)),
                               arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                   adamGvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                              arrayF.slice(k), arrayWt.slice(k).row(i-lagsModelMax), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                              matrixG.col(k), vecErrors(i-lagsModelMax));
                }

                if(refineHead){
                    // Fill in the head of the series.
                    for(int i=lagsModelMax-1; i>=0; i=i-1) {
                        arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                            adamFvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                       arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
                    }
                }

                // Change the specific element in the state vector to negative
                if(T=='A'){
                    arrayProfilesRecent.slice(k)(1) = -arrayProfilesRecent.slice(k)(1);
                }
                else if(T=='M'){
                    arrayProfilesRecent.slice(k)(1) = 1/arrayProfilesRecent.slice(k)(1);
                }
            }
        }
    }

    return List::create(Named("states") = arrayVt, Named("fitted") = matYfit,
                        Named("profilesRecent") = arrayProfilesRecent);
}

/* # Wrapper for simulator */
// [[Rcpp::export]]
RcppExport SEXP adamRefitterWrap(arma::mat matrixYt, arma::mat matrixOt, arma::cube arrayVt,
                                 arma::cube arrayF, arma::cube arrayWt, arma::mat matrixG,
                                 char const &E, char const &T, char const &S,
                                 arma::uvec lags, arma::umat indexLookupTable, arma::cube arrayProfilesRecent,
                                 unsigned int const &nSeasonal, unsigned int const &componentsNumberETS,
                                 unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                                 bool const &backcast, bool const &refineHead){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamRefitter(matrixYt, matrixOt, arrayVt, arrayF, arrayWt, matrixG,
                             E, T, S, lags, indexLookupTable, arrayProfilesRecent,
                             nNonSeasonal, nSeasonal, nArima, nXreg, constant,
                             backcast, refineHead));
}


// ##### Script for the simulator for refitted model
List adamReforecaster(arma::cube const &arrayErrors, arma::cube const &arrayOt,
                      arma::cube const &arrayF, arma::cube const &arrayWt, arma::mat const &matrixG,
                      char const &E, char const &T, char const &S, arma::uvec &lags,
                      arma::umat const &indexLookupTable, arma::cube arrayProfileRecent,
                      unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                      unsigned int const &nArima, unsigned int const &nXreg, bool const &constant) {

    unsigned int obs = arrayErrors.n_rows;
    unsigned int nSeries = arrayErrors.n_cols;
    unsigned int nsim = arrayErrors.n_slices;

    unsigned int lagsModelMax = max(lags);
    unsigned int nETS = nNonSeasonal + nSeasonal;
    int nComponents = lags.n_rows;
    arma::cube profilesRecentOriginal = arrayProfileRecent;

    arma::cube arrY(obs, nSeries, nsim);

    for(unsigned int j=0; j<nsim; j=j+1){
        for(unsigned int k=0; k<nSeries; k=k+1){
            arrayProfileRecent.slice(j) = profilesRecentOriginal.slice(j);
            for(unsigned int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
                /* # Measurement equation and the error term */
                arrY(i-lagsModelMax,k,j) = arrayOt(i-lagsModelMax,k,j) *
                        (adamWvalue(arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)),
                                    arrayWt.slice(j).row(i-lagsModelMax), E, T, S,
                                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) +
                                        adamRvalue(arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)),
                                                   arrayWt.slice(j).row(i-lagsModelMax), E, T, S,
                                                   nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) *
                                                       arrayErrors.slice(j)(i-lagsModelMax,k));

                // Fix potential issue with negatives in mixed models
                if((E=='M' || T=='M' || S=='M') && (arrY(i-lagsModelMax,k,j)<0)){
                    arrY(i-lagsModelMax,k,j) = 0;
                }

                /* # Transition equation */
                arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)) =
                        (adamFvalue(arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)),
                                    arrayF.slice(j), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                        adamGvalue(arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)),
                                                   arrayF.slice(j), arrayWt.slice(j).row(i-lagsModelMax),
                                                   E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                                   nComponents, constant, matrixG.col(k),
                                                   arrayErrors.slice(j)(i-lagsModelMax,k)));
            }
        }
    }

    return List::create(Named("matrixYt") = arrY);
}

/* # Wrapper for reforecaster */
// [[Rcpp::export]]
RcppExport SEXP adamReforecasterWrap(arma::cube arrayErrors, arma::cube arrayOt,
                                     arma::cube arrayF, arma::cube arrayWt, arma::mat matrixG,
                                     char const &E, char const &T, char const &S, arma::uvec &lags,
                                     arma::umat const &indexLookupTable, arma::cube arrayProfileRecent,
                                     unsigned int const &nSeasonal, unsigned int const &componentsNumberETS,
                                     unsigned int const &nArima, unsigned int const &nXreg, bool const &constant){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamReforecaster(arrayErrors, arrayOt, arrayF, arrayWt, matrixG,
                                 E, T, S, lags, indexLookupTable, arrayProfileRecent,
                                 nNonSeasonal, nSeasonal, nArima, nXreg, constant));
}
