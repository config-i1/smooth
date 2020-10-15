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
                  arma::umat const &profilesObserved, arma::cube arrayProfilesRecent,
                  unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                  unsigned int const &nArima, unsigned int const &nXreg) {

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
            arrayVt.slice(i).col(j) = arrayProfilesRecent.slice(i).elem(profilesObserved.col(j));
            arrayProfilesRecent.slice(i).elem(profilesObserved.col(j)) = adamFvalue(arrayProfilesRecent.slice(i).elem(profilesObserved.col(j)),
                                      arrayF.slice(i), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents);
        }
        ////// Run forward
        // Loop for the model construction
        for(unsigned int j=lagsModelMax; j<obs+lagsModelMax; j=j+1) {

            /* # Measurement equation and the error term */
            matYfit(j-lagsModelMax,i) = adamWvalue(arrayProfilesRecent.slice(i).elem(profilesObserved.col(j-lagsModelMax)),
                    arrayWt.slice(i).row(j-lagsModelMax), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents);

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
            arrayProfilesRecent.slice(i).elem(profilesObserved.col(j-lagsModelMax)) =
            adamFvalue(arrayProfilesRecent.slice(i)(profilesObserved.col(j-lagsModelMax)),
                       arrayF.slice(i), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents) +
                           adamGvalue(arrayProfilesRecent.slice(i).elem(profilesObserved.col(j-lagsModelMax)),
                                      arrayF.slice(i), arrayWt.slice(i).row(j-lagsModelMax), E, T, S,
                                      nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents,
                                      matrixG.col(i), vecErrors(j-lagsModelMax));

            arrayVt.slice(i).col(j) = arrayProfilesRecent.slice(i).elem(profilesObserved.col(j-lagsModelMax));
        }
    }

    return List::create(Named("states") = arrayVt, Named("fitted") = matYfit,
                        Named("profilesRecent") = arrayProfilesRecent);
}

/* # Wrapper for simulator */
// [[Rcpp::export]]
RcppExport SEXP adamRefitterWrap(SEXP yt, SEXP ot, SEXP arrVt, SEXP arrF, SEXP arrWt, SEXP matG,
                                 SEXP Etype, SEXP Ttype, SEXP Stype,
                                 SEXP lagsModelAll, SEXP profilesObservedTable, SEXP profilesRecentArray,
                                 SEXP componentsNumberETSSeasonal, SEXP componentsNumberETS,
                                 SEXP componentsNumberARIMA, SEXP xregNumber){

    NumericMatrix yt_n(yt);
    arma::mat matrixYt(yt_n.begin(), yt_n.nrow(), yt_n.ncol(), false);

    NumericMatrix ot_n(ot);
    arma::mat matrixOt(ot_n.begin(), ot_n.nrow(), ot_n.ncol(), false);

    // ### arrvt should contain array of obs x ncomponents x nSeries elements.
    NumericVector arrVt_n(arrVt);
    IntegerVector arrVt_dim = arrVt_n.attr("dim");
    arma::cube arrayVt(arrVt_n.begin(),arrVt_dim[0], arrVt_dim[1], arrVt_dim[2]);

    NumericVector arrF_n(arrF);
    IntegerVector arrF_dim = arrF_n.attr("dim");
    arma::cube arrayF(arrF_n.begin(),arrF_dim[0], arrF_dim[1], arrF_dim[2], false);

    NumericVector arrWt_n(arrWt);
    IntegerVector arrWt_dim = arrWt_n.attr("dim");
    arma::cube arrayWt(arrWt_n.begin(),arrWt_dim[0], arrWt_dim[1], arrWt_dim[2], false);

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

    NumericVector profilesRecentArray_n(profilesRecentArray);
    IntegerVector profilesRecentArray_dim = profilesRecentArray_n.attr("dim");
    arma::cube arrayProfilesRecent(profilesRecentArray_n.begin(),profilesRecentArray_dim[0],
                                   profilesRecentArray_dim[1], profilesRecentArray_dim[2]);

    unsigned int nSeasonal = as<int>(componentsNumberETSSeasonal);
    unsigned int nNonSeasonal = as<int>(componentsNumberETS) - nSeasonal;
    unsigned int nArima = as<int>(componentsNumberARIMA);
    unsigned int nXreg = as<int>(xregNumber);

    return wrap(adamRefitter(matrixYt, matrixOt, arrayVt, arrayF, arrayWt, matrixG,
                             E, T, S, lags, profilesObserved, arrayProfilesRecent,
                             nNonSeasonal, nSeasonal, nArima, nXreg));
}


// ##### Script for the simulator for refitted model
List adamReforecaster(arma::cube const &arrayErrors, arma::cube const &arrayOt,
                      arma::cube const &arrayF, arma::cube const &arrayWt, arma::mat const &matrixG,
                      char const &E, char const &T, char const &S, arma::uvec &lags,
                      arma::umat const &profilesObserved, arma::cube arrayProfileRecent,
                      unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                      unsigned int const &nArima, unsigned int const &nXreg) {

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
                        (adamWvalue(arrayProfileRecent.slice(k).elem(profilesObserved.col(j-lagsModelMax)),
                                    arrayWt.slice(k).row(j-lagsModelMax), E, T, S,
                                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents) +
                                        adamRvalue(arrayProfileRecent.slice(k).elem(profilesObserved.col(j-lagsModelMax)),
                                                   arrayWt.slice(k).row(j-lagsModelMax), E, T, S,
                                                   nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents) *
                                                       arrayErrors.slice(k)(j-lagsModelMax,i));

                // Fix potential issue with negatives in mixed models
                if((E=='M' || T=='M' || S=='M') && (arrY(j-lagsModelMax,i,k)<0)){
                    arrY(j-lagsModelMax,i,k) = 0;
                }

                /* # Transition equation */
                arrayProfileRecent.slice(k).elem(profilesObserved.col(j-lagsModelMax)) =
                        (adamFvalue(arrayProfileRecent.slice(k).elem(profilesObserved.col(j-lagsModelMax)),
                                    arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents) +
                                        adamGvalue(arrayProfileRecent.slice(k).elem(profilesObserved.col(j-lagsModelMax)),
                                                   arrayF.slice(k), arrayWt.slice(k).row(j-lagsModelMax),
                                                   E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                                   nComponents, matrixG.col(i),
                                                   arrayErrors.slice(k)(j-lagsModelMax,i)));
            }
        }
    }

    return List::create(Named("matrixYt") = arrY);
}

/* # Wrapper for reforecaster */
// [[Rcpp::export]]
RcppExport SEXP adamReforecasterWrap(SEXP arrErrors, SEXP arrOt, SEXP arrF, SEXP arrWt, SEXP matG,
                                     SEXP Etype, SEXP Ttype, SEXP Stype, SEXP lagsModelAll,
                                     SEXP profilesObservedTable, SEXP profilesRecentArray,
                                     SEXP componentsNumberSeasonal, SEXP componentsNumber,
                                     SEXP componentsNumberArima, SEXP xregNumber){

    NumericVector arrErrors_n(arrErrors);
    IntegerVector arrErrors_dim = arrErrors_n.attr("dim");
    arma::cube arrayErrors(arrErrors_n.begin(),arrErrors_dim[0], arrErrors_dim[1], arrErrors_dim[2], false);

    NumericVector arrOt_n(arrOt);
    IntegerVector arrOt_dim = arrOt_n.attr("dim");
    arma::cube arrayOt(arrOt_n.begin(),arrOt_dim[0], arrOt_dim[1], arrOt_dim[2], false);

    NumericVector arrF_n(arrF);
    IntegerVector arrF_dim = arrF_n.attr("dim");
    arma::cube arrayF(arrF_n.begin(),arrF_dim[0], arrF_dim[1], arrF_dim[2], false);

    NumericVector arrWt_n(arrWt);
    IntegerVector arrWt_dim = arrWt_n.attr("dim");
    arma::cube arrayWt(arrWt_n.begin(),arrWt_dim[0], arrWt_dim[1], arrWt_dim[2], false);

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

    // Create a numeric array. The states will be saved here as in a buffer
    NumericVector profilesRecentArray_n(profilesRecentArray);
    IntegerVector profilesRecentArray_dim = profilesRecentArray_n.attr("dim");
    arma::cube arrayProfileRecent(profilesRecentArray_n.begin(),profilesRecentArray_dim[0],
                                  profilesRecentArray_dim[1], profilesRecentArray_dim[2], false);

    unsigned int nSeasonal = as<int>(componentsNumberSeasonal);
    unsigned int nNonSeasonal = as<int>(componentsNumber) - nSeasonal;
    unsigned int nArima = as<int>(componentsNumberArima);
    unsigned int nXreg = as<int>(xregNumber);

    return wrap(adamReforecaster(arrayErrors, arrayOt, arrayF, arrayWt, matrixG,
                                 E, T, S, lags, profilesObserved, arrayProfileRecent,
                                 nNonSeasonal, nSeasonal, nArima, nXreg));
}
