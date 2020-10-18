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
                   unsigned int const &nArima, unsigned int const &nXreg) {

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
            matY(j-lagsModelMax,i) = matrixOt(j-lagsModelMax,i) * (adamWvalue(profilesRecent(profilesObserved.col(j-lagsModelMax)),
                                              matrixWt.row(j-lagsModelMax), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents) +
                                                  adamRvalue(profilesRecent(profilesObserved.col(j-lagsModelMax)),
                                                             matrixWt.row(j-lagsModelMax), E, T, S,
                                                             nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents) *
                                                                 matrixErrors(j-lagsModelMax,i));

            /* # Transition equation */
            profilesRecent(profilesObserved.col(j-lagsModelMax)) = (adamFvalue(profilesRecent(profilesObserved.col(j-lagsModelMax)),
                                                matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents) +
                                                    adamGvalue(profilesRecent(profilesObserved.col(j-lagsModelMax)),
                                                               matrixF, matrixWt.row(j-lagsModelMax),
                                                               E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                                               nComponents, matrixG.col(i),
                                                               matrixErrors(j-lagsModelMax,i)));

            /* Failsafe for cases when unreasonable value for state vector was produced */
            // if(!matrixVt.col(j).is_finite()){
            //     matrixVt.col(j) = matrixVt(lagrows);
            // }
            // if((S=='M') && (matrixVt(nNonSeasonal,j) <= 0)){
            //     matrixVt(nNonSeasonal,j) = arma::as_scalar(matrixVt(lagrows.row(nNonSeasonal)));
            // }
            // if(T=='M'){
            //     if((matrixVt(0,j) <= 0) | (matrixVt(1,j) <= 0)){
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
RcppExport SEXP adamSimulatorWrap(SEXP arrVt, SEXP matErrors, SEXP matOt, SEXP matF, SEXP matWt, SEXP matG,
                                  SEXP Etype, SEXP Ttype, SEXP Stype, SEXP lagsModelAll,
                                  SEXP profilesObservedTable, SEXP profilesRecentTable,
                                  SEXP componentsNumberSeasonal, SEXP componentsNumber,
                                  SEXP componentsNumberArima, SEXP xregNumber){

    // ### arrvt should contain array of obs x ncomponents x nSeries elements.
    NumericVector arrVt_n(arrVt);
    IntegerVector arrVt_dim = arrVt_n.attr("dim");
    arma::cube arrayVt(arrVt_n.begin(),arrVt_dim[0], arrVt_dim[1], arrVt_dim[2], false);

    NumericMatrix matErrors_n(matErrors);
    arma::mat matrixErrors(matErrors_n.begin(), matErrors_n.nrow(), matErrors_n.ncol(), false);

    NumericMatrix matOt_n(matOt);
    arma::mat matrixOt(matOt_n.begin(), matOt_n.nrow(), matOt_n.ncol(), false);

    NumericVector arrF_n(matF);
    IntegerVector arrF_dim = arrF_n.attr("dim");
    arma::cube arrayF(arrF_n.begin(),arrF_dim[0], arrF_dim[1], arrF_dim[2], false);

    NumericMatrix matWt_n(matWt);
    arma::mat matrixWt(matWt_n.begin(), matWt_n.nrow(), matWt_n.ncol(), false);

    // ### matG should contain persistence vectors in each column
    NumericMatrix matG_n(matG);
    arma::mat matrixG(matG_n.begin(), matG_n.nrow(), matG_n.ncol(), false);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector lagsModelAll_n(lagsModelAll);
    arma::uvec lags = as<arma::uvec>(lagsModelAll_n);

    // Get the observed profiles
    IntegerMatrix profilesObservedTable_n(profilesObservedTable);
    arma::umat profilesObserved = as<arma::umat>(profilesObservedTable_n);

    // Create a numeric matrix. The states will be saved here as in a buffer
    NumericMatrix profilesRecentTable_n(profilesRecentTable);
    arma::mat profilesRecent(profilesRecentTable_n.begin(), profilesRecentTable_n.nrow(), profilesRecentTable_n.ncol());

    unsigned int nSeasonal = as<int>(componentsNumberSeasonal);
    unsigned int nNonSeasonal = as<int>(componentsNumber) - nSeasonal;
    unsigned int nArima = as<int>(componentsNumberArima);
    unsigned int nXreg = as<int>(xregNumber);

    return wrap(adamSimulator(arrayVt, matrixErrors, matrixOt, arrayF, matrixWt, matrixG,
                              E, T, S, lags, profilesObserved, profilesRecent,
                              nNonSeasonal, nSeasonal, nArima, nXreg));
}
