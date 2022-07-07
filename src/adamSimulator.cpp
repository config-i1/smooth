#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "adamGeneral.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// ##### Script for simulate functions
List adamSimulator(arma::cube &arrayVt, arma::mat const &matrixErrors, arma::mat const &matrixOt,
                   arma::cube const &arrayF, arma::mat const &matrixWt, arma::mat const &matrixG,
                   char const &E, char const &T, char const &S, arma::uvec &lags,
                   arma::umat const &profilesObserved, arma::mat profilesRecent,
                   unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                   unsigned int const &nArima, unsigned int const &nXreg, bool const &constant) {

    unsigned int obs = matrixErrors.n_rows;
    unsigned int nSeries = matrixErrors.n_cols;

    int lagsModelMax = max(lags);
    unsigned int nETS = nNonSeasonal + nSeasonal;
    int nComponents = lags.n_rows;
    int obsAll = obs + lagsModelMax;
    arma::mat profilesRecentOriginal = profilesRecent;

    arma::mat matrixVt(nComponents, obsAll, arma::fill::zeros);
    arma::mat matrixF(arrayF.n_rows, arrayF.n_cols, arma::fill::zeros);

    arma::mat matY(obs, nSeries);

    for(unsigned int i=0; i<nSeries; i=i+1){
        matrixVt = arrayVt.slice(i);
        matrixF = arrayF.slice(i);
        profilesRecent = profilesRecentOriginal;
        for(int j=lagsModelMax; j<obsAll; j=j+1) {
            /* # Measurement equation and the error term */
            matY(j-lagsModelMax,i) = matrixOt(j-lagsModelMax,i) *
                                             (adamWvalue(profilesRecent(profilesObserved.col(j-lagsModelMax)),
                                                         matrixWt.row(j-lagsModelMax), E, T, S,
                                                         nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                                         nComponents, constant) +
                                              adamRvalue(profilesRecent(profilesObserved.col(j-lagsModelMax)),
                                                         matrixWt.row(j-lagsModelMax), E, T, S,
                                                         nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) *
                                              matrixErrors(j-lagsModelMax,i));

            /* # Transition equation */
            profilesRecent(profilesObserved.col(j-lagsModelMax)) =
                                                (adamFvalue(profilesRecent(profilesObserved.col(j-lagsModelMax)),
                                                            matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                                                            nComponents, constant) +
                                                 adamGvalue(profilesRecent(profilesObserved.col(j-lagsModelMax)),
                                                            matrixF, matrixWt.row(j-lagsModelMax),
                                                            E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                                            nComponents, constant, matrixG.col(i),
                                                            matrixErrors(j-lagsModelMax,i)));

            /* Failsafe for cases when unreasonable value for state vector was produced */
            // if(!matrixVt.col(j).is_finite()){
            //     matrixVt.col(j) = matrixVt(lagrows);
            // }
            // if((S=='M') && (matrixVt(nNonSeasonal,j) <= 0)){
            //     matrixVt(nNonSeasonal,j) = arma::as_scalar(matrixVt(lagrows.row(nNonSeasonal)));
            // }
            // if(T=='M'){
            //     if((matrixVt(0,j) <= 0) || (matrixVt(1,j) <= 0)){
            //         matrixVt(0,j) = arma::as_scalar(matrixVt(lagrows.row(0)));
            //         matrixVt(1,j) = arma::as_scalar(matrixVt(lagrows.row(1)));
            //     }
            // }
            // if(any(matrixVt.col(j)>1e+100)){
            //     matrixVt.col(j) = matrixVt(lagrows);
            // }
            matrixVt.col(j) = profilesRecent(profilesObserved.col(j-lagsModelMax));
        }
        arrayVt.slice(i) = matrixVt;
    }

    return List::create(Named("arrayVt") = arrayVt, Named("matrixYt") = matY);
}

/* # Wrapper for simulator */
// [[Rcpp::export]]
RcppExport SEXP adamSimulatorWrap(arma::cube arrayVt, arma::mat matrixErrors, arma::mat matrixOt,
                                  arma::cube arrayF, arma::mat matrixWt, arma::mat matrixG,
                                  char const &E, char const &T, char const &S, arma::uvec lags,
                                  arma::umat profilesObserved, arma::mat profilesRecent,
                                  unsigned int const &nSeasonal, unsigned int const &componentsNumber,
                                  unsigned int const &nArima, unsigned int const &nXreg, bool const &constant){

    unsigned int nNonSeasonal = componentsNumber - nSeasonal;

    return wrap(adamSimulator(arrayVt, matrixErrors, matrixOt, arrayF, matrixWt, matrixG,
                              E, T, S, lags, profilesObserved, profilesRecent,
                              nNonSeasonal, nSeasonal, nArima, nXreg, constant));
}
