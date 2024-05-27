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
                  unsigned int const &nArima, unsigned int const &nXreg, bool const &constant) {

    unsigned int obs = matrixYt.n_rows;
    unsigned int nSeries = matrixG.n_cols;

    unsigned int lagsModelMax = max(lags);
    unsigned int nETS = nNonSeasonal + nSeasonal;
    int nComponents = lags.n_rows;

    arma::mat matYfit(obs, nSeries, arma::fill::zeros);
    arma::vec vecErrors(obs, arma::fill::zeros);

    for(unsigned int i=0; i<nSeries; i=i+1){
        // Refine the head (in order for it to make sense)
        for(unsigned int j=0; j<lagsModelMax; j=j+1) {
            arrayVt.slice(i).col(j) = arrayProfilesRecent.slice(i).elem(indexLookupTable.col(j));
            arrayProfilesRecent.slice(i).elem(indexLookupTable.col(j)) = adamFvalue(arrayProfilesRecent.slice(i).elem(indexLookupTable.col(j)),
                                      arrayF.slice(i), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
        }
        // Loop for the model construction
        for(unsigned int j=lagsModelMax; j<obs+lagsModelMax; j=j+1) {

            /* # Measurement equation and the error term */
            matYfit(j-lagsModelMax,i) = adamWvalue(arrayProfilesRecent.slice(i).elem(indexLookupTable.col(j)),
                    arrayWt.slice(i).row(j-lagsModelMax), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

            // Fix potential issue with negatives in mixed models
            if((E=='M' || T=='M' || S=='M') && (matYfit(j-lagsModelMax,i)<=0)){
                matYfit(j-lagsModelMax,i) = 1;
            }

            // If this is zero (intermittent), then set error to zero
            if(matrixOt(j-lagsModelMax)==0){
                vecErrors(j-lagsModelMax) = 0;
            }
            else{
                vecErrors(j-lagsModelMax) = errorf(matrixYt(j-lagsModelMax), matYfit(j-lagsModelMax,i), E);
            }

            /* # Transition equation */
            arrayProfilesRecent.slice(i).elem(indexLookupTable.col(j)) =
            adamFvalue(arrayProfilesRecent.slice(i)(indexLookupTable.col(j)),
                       arrayF.slice(i), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                           adamGvalue(arrayProfilesRecent.slice(i).elem(indexLookupTable.col(j)),
                                      arrayF.slice(i), arrayWt.slice(i).row(j-lagsModelMax), E, T, S,
                                      nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                      matrixG.col(i), vecErrors(j-lagsModelMax));

            arrayVt.slice(i).col(j) = arrayProfilesRecent.slice(i).elem(indexLookupTable.col(j));
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
                                 unsigned int const &nArima, unsigned int const &nXreg, bool const &constant){

    unsigned int nNonSeasonal = componentsNumberETS - nSeasonal;

    return wrap(adamRefitter(matrixYt, matrixOt, arrayVt, arrayF, arrayWt, matrixG,
                             E, T, S, lags, indexLookupTable, arrayProfilesRecent,
                             nNonSeasonal, nSeasonal, nArima, nXreg, constant));
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

    for(unsigned int k=0; k<nsim; k=k+1){
        for(unsigned int i=0; i<nSeries; i=i+1){
            arrayProfileRecent.slice(k) = profilesRecentOriginal.slice(k);
            for(unsigned int j=lagsModelMax; j<obs+lagsModelMax; j=j+1) {
                /* # Measurement equation and the error term */
                arrY(j-lagsModelMax,i,k) = arrayOt(j-lagsModelMax,i,k) *
                        (adamWvalue(arrayProfileRecent.slice(k).elem(indexLookupTable.col(j-lagsModelMax)),
                                    arrayWt.slice(k).row(j-lagsModelMax), E, T, S,
                                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) +
                                        adamRvalue(arrayProfileRecent.slice(k).elem(indexLookupTable.col(j-lagsModelMax)),
                                                   arrayWt.slice(k).row(j-lagsModelMax), E, T, S,
                                                   nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) *
                                                       arrayErrors.slice(k)(j-lagsModelMax,i));

                // Fix potential issue with negatives in mixed models
                if((E=='M' || T=='M' || S=='M') && (arrY(j-lagsModelMax,i,k)<0)){
                    arrY(j-lagsModelMax,i,k) = 0;
                }

                /* # Transition equation */
                arrayProfileRecent.slice(k).elem(indexLookupTable.col(j-lagsModelMax)) =
                        (adamFvalue(arrayProfileRecent.slice(k).elem(indexLookupTable.col(j-lagsModelMax)),
                                    arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                        adamGvalue(arrayProfileRecent.slice(k).elem(indexLookupTable.col(j-lagsModelMax)),
                                                   arrayF.slice(k), arrayWt.slice(k).row(j-lagsModelMax),
                                                   E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                                   nComponents, constant, matrixG.col(i),
                                                   arrayErrors.slice(k)(j-lagsModelMax,i)));
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
