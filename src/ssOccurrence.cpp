#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "ssGeneral.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;


/* # Function returns a and b errors as a vector, depending on the types of E, O and the others */
// In case of O=="p", the error is provided in the first element of the vector
// In case of O=="i", the error is moved in the first element.
std::vector<double> occurrenceError(double const &yAct, double aFit, double bFit, char const &E, char const &O){
// aFit is the fitted values of a, bFit is the same for the b. O is the type of occurrence.
// In cases of O!="g", the aFit is used as a variable.

// The value for the probability and the error
    double pfit = 0;
// error is the error in the probability scale
    double error = 0;
// kappa is the stuff for TSB
    double kappa = 1E-10;
// The returned value. The first one is the aError, the second one is the bError.
    std::vector<double> output(2);

// Correct the fitted depending on the type of the error
    switch(E){
        case 'A':
            aFit = exp(aFit);
            bFit = exp(bFit);
        break;
    }

// Produce fitted values
    switch(O){
        case 'g':
            pfit = aFit / (aFit + bFit);
        break;
        case 'o':
            pfit = aFit / (aFit + 1);
        break;
        case 'i':
            pfit = 1 / (1 + aFit);
        break;
        case 'p':
            pfit = aFit;
        break;
    }

// Calculate the error and the respective a and b errors
    error = (1 + yAct - pfit) / 2;
    switch(O){
        case 'g':
            output[0] = bFit * error / (1 - error);
        break;
        case 'o':
            output[0] = error / (1 - error);
        break;
        case 'i':
            output[0] = (1 - error) / error;
        break;
        case 'p':
            // If this is "probability" model, calculate the error differently
            output[0] = (yAct * (1 - 2 * kappa) + kappa) / pfit;
        break;
    }
    output[1] = aFit * (1 - error) / error;

    switch(E){
        case 'M':
            output[0] = output[0] - 1;
            output[1] = output[1] - 1;
        break;
        case 'A':
            output[0] = log(output[0]);
            output[1] = log(output[1]);
        break;
    }

    return output;
}

// # Fitter for the occurrence part of the univariate models. This does not include iETS_G model
List occurenceFitter(arma::mat &matrixVt, arma::mat const &matrixF, arma::rowvec const &rowvecW, arma::vec const &vecG,
                     arma::vec const &vecYt, arma::uvec &lags, char const &E, char const &T, char const &S, char const &O,
                     arma::mat const &matrixXt, arma::mat &matrixAt, arma::mat const &matrixFX, arma::vec const &vecGX){
    /* # matrixVt should have a length of obs + maxlag.
    * # rowvecW should have 1 row.
    * # vecG should be a vector
    * # lags is a vector of lags
    * # matrixXt is the matrix with the exogenous variables
    * # matrixAt is the matrix with the parameters for the exogenous
    */

    arma::mat matrixXtTrans = matrixXt.t();

    int obs = vecYt.n_rows;
    int obsall = matrixVt.n_cols;
    unsigned int nComponents = matrixVt.n_rows;
    arma::uvec lagsInternal = lags;
    unsigned int maxlag = max(lagsInternal);
    int lagslength = lagsInternal.n_rows;

    lagsInternal = lagsInternal * nComponents;

    for(int i=0; i<lagslength; i=i+1){
        lagsInternal(i) = lagsInternal(i) + (lagslength - i - 1);
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);

    // The vector for the fitted values
    arma::vec vecYfit(obs, arma::fill::zeros);
    // The vector for the fitted probabilities
    arma::vec vecPfit(obs, arma::fill::zeros);

    arma::vec vecErrors(obs, arma::fill::zeros);
    arma::vec bufferforat(vecGX.n_rows);

    for(unsigned int i=maxlag; i<obs+maxlag; i=i+1) {
        lagrows = i * nComponents - lagsInternal + nComponents - 1;

/* # Measurement equation and the error term */
        vecYfit(i-maxlag) = wvalue(matrixVt(lagrows), rowvecW, E, T, S,
                                   matrixXt.row(i-maxlag), matrixAt.col(i-1));

        // This is a failsafe for cases of ridiculously high and ridiculously low values
        if(vecYfit(i-maxlag) > 1e+100){
            vecYfit(i-maxlag) = vecYfit(i-maxlag-1);
        }

        // For O==c("o","i","p") the bFit is set to 1 and aFit is the variable under consideration
        vecErrors(i-maxlag) = occurrenceError(vecYt(i-maxlag), vecYfit(i-maxlag), 1.0, E, O)[0];

/* # Transition equation */
        matrixVt.col(i) = fvalue(matrixVt(lagrows), matrixF, T, S) +
                          gvalue(matrixVt(lagrows), matrixF, rowvecW, E, T, S) % vecG * vecErrors(i-maxlag);

/* Failsafe for cases when unreasonable value for state vector was produced */
        if(!matrixVt.col(i).is_finite()){
            matrixVt.col(i) = matrixVt(lagrows);
        }
        if((S=='M') & (matrixVt(matrixVt.n_rows-1,i) <= 0)){
            matrixVt(matrixVt.n_rows-1,i) = arma::as_scalar(matrixVt(lagrows.row(matrixVt.n_rows-1)));
        }
        if(T=='M'){
            if((matrixVt(0,i) <= 0) | (matrixVt(1,i) <= 0)){
                matrixVt(0,i) = arma::as_scalar(matrixVt(lagrows.row(0)));
                matrixVt(1,i) = arma::as_scalar(matrixVt(lagrows.row(1)));
            }
        }
        if(any(matrixVt.col(i)>1e+100)){
            matrixVt.col(i) = matrixVt(lagrows);
        }

/* Renormalise components if the seasonal model is chosen */
        if(S!='N'){
            if(double(i+1) / double(maxlag) == double((i+1) / maxlag)){
                matrixVt.cols(i-maxlag+1,i) = normaliser(matrixVt.cols(i-maxlag+1,i), obsall, maxlag, S, T);
            }
        }

/* # Transition equation for xreg */
        bufferforat = gXvalue(matrixXtTrans.col(i-maxlag), vecGX, vecErrors.row(i-maxlag), E);
        matrixAt.col(i) = matrixFX * matrixAt.col(i-1) + bufferforat;
    }

    for (int i=obs+maxlag; i<obsall; i=i+1) {
        lagrows = i * nComponents - lagsInternal + nComponents - 1;
        matrixVt.col(i) = fvalue(matrixVt(lagrows), matrixF, T, S);
        matrixAt.col(i) = matrixFX * matrixAt.col(i-1);

/* Failsafe for cases when unreasonable value for state vector was produced */
        if(!matrixVt.col(i).is_finite()){
            matrixVt.col(i) = matrixVt(lagrows);
        }
        if((S=='M') & (matrixVt(matrixVt.n_rows-1,i) <= 0)){
            matrixVt(matrixVt.n_rows-1,i) = arma::as_scalar(matrixVt(lagrows.row(matrixVt.n_rows-1)));
        }
        if(T=='M'){
            if((matrixVt(0,i) <= 0) | (matrixVt(1,i) <= 0)){
                matrixVt(0,i) = arma::as_scalar(matrixVt(lagrows.row(0)));
                matrixVt(1,i) = arma::as_scalar(matrixVt(lagrows.row(1)));
            }
        }
    }

    switch(O){
        case 'o':
            switch(E){
                case 'A':
                    vecPfit = exp(vecYfit) / (1+exp(vecYfit));
                break;
                case 'M':
                    vecPfit = vecYfit / (1+vecYfit);
                break;
            }
        break;
        case 'i':
            switch(E){
                case 'A':
                    vecPfit = 1 / (1+exp(vecYfit));
                break;
                case 'M':
                    vecPfit = 1 / (1+vecYfit);
                break;
            }
        break;
        case 'p':
            vecPfit = vecYfit;
            // This is not correct statistically. See the (50) - (52) in order to see how this needs to be done properly.
            // But this works and I don't have time to do that 100% correctly.
            vecPfit.elem(find(vecPfit>1)).fill(1.0-1E-10);
            vecPfit.elem(find(vecPfit<0)).fill(1E-10);
        break;
    }

    return List::create(Named("matvt") = matrixVt, Named("yfit") = vecYfit, Named("pfit") = vecPfit,
                        Named("errors") = vecErrors, Named("matat") = matrixAt);
}


/* # Wrapper for fitter */
// [[Rcpp::export]]
RcppExport SEXP occurenceFitterWrap(SEXP matvt, SEXP matF, SEXP matw, SEXP vecg, SEXP yt,
                                    SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP Otype,
                                    SEXP matxt, SEXP matat, SEXP matFX, SEXP vecgX) {

    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

    NumericMatrix yt_n(yt);
    arma::vec vecYt(yt_n.begin(), yt_n.nrow(), false);

    NumericMatrix vecg_n(vecg);
    arma::vec vecG(vecg_n.begin(), vecg_n.nrow(), false);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    char O = as<char>(Otype);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow(), false);

    return wrap(occurenceFitter(matrixVt, matrixF, rowvecW, vecG, vecYt, lags, E, T, S, O,
                                matrixXt, matrixAt, matrixFX, vecGX));
}

// # Fitter for the occurrence part of the univariate models. General Beta model
// List occurenceFitterGeneral(arma::mat &matrixVt, arma::mat const &matrixF, arma::rowvec const &rowvecW, arma::vec const &vecG,
//                      arma::vec const &vecYt, arma::uvec &lags, char const &E, char const &T, char const &S, char const &O,
//                      arma::mat const &matrixXt, arma::mat &matrixAt, arma::mat const &matrixFX, arma::vec const &vecGX){
//
//     return List::create(Named("matvt") = matrixVt.t(), Named("yfit") = vecYfit,
//                         Named("errors") = vecErrors, Named("matat") = matrixAt.t());
// }


/* # Function returns the chosen Cost Function based on the chosen model and produced errors */
double occurrenceOptimizer(arma::mat &matrixVt, arma::mat const &matrixF, arma::rowvec const &rowvecW,
                           arma::vec const &vecG, arma::vec const &vecYt,
                           arma::uvec &lags, char const &E, char const &T, char const &S, char const &O,
                           arma::mat const &matrixXt, arma::mat &matrixAt, arma::mat const &matrixFX, arma::vec const &vecGX){

    List fitting = occurenceFitter(matrixVt, matrixF, rowvecW, vecG, vecYt, lags, E, T, S, O,
                                   matrixXt, matrixAt, matrixFX, vecGX);

    NumericMatrix pfitfromfit = as<NumericMatrix>(fitting["pfit"]);
    arma::vec vecPfit = as<arma::vec>(pfitfromfit);

    // 0.5 is needed for cases, when the variable is continuous in (0, 1)
    double CFres = -sum(log(vecPfit.elem(find(vecYt>=0.5)))) - sum(log(1-vecPfit.elem(find(vecYt<0.5))));

    return CFres;
}

/* # Function is used in cases when the persistence vector needs to be estimated.
# If bounds are violated, it returns variance of yt. */
// [[Rcpp::export]]
RcppExport SEXP occurrenceOptimizerWrap(SEXP matvt, SEXP matF, SEXP matw, SEXP vecg, SEXP yt,
                                        SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP Otype,
                                        SEXP matxt, SEXP matat, SEXP matFX, SEXP vecgX,
                                        SEXP bounds) {
/* Function is needed to implement admissible constrains on smoothing parameters */
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

    NumericMatrix yt_n(yt);
    arma::vec vecYt(yt_n.begin(), yt_n.nrow(), false);

    NumericMatrix vecg_n(vecg);
    arma::vec vecG(vecg_n.begin(), vecg_n.nrow(), false);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    char O = as<char>(Otype);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow(), false);

    char boundtype = as<char>(bounds);

// Values needed for eigenvalues calculation
    arma::cx_vec eigval;

    if(boundtype=='u'){
// alpha in (0,1)
        if((vecG(0)>1) || (vecG(0)<0)){
            // vecG.zeros();
            // matrixVt.zeros();
            return wrap(1E+300);
        }
        if(T!='N'){
// beta in (0,alpha)
            if((vecG(1)>vecG(0)) || (vecG(1)<0)){
                // vecG.zeros();
                // matrixVt.zeros();
                return wrap(1E+300);
            }
            if(S!='N'){
// gamma in (0,1-alpha)
                if((vecG(2)>(1-vecG(0))) || (vecG(2)<0)){
                    // vecG.zeros();
                    // matrixVt.zeros();
                    return wrap(1E+300);
                }
            }
        }
        if(S!='N'){
// gamma in (0,1-alpha)
            if((vecG(1)>(1-vecG(0))) || (vecG(1)<0)){
                // vecG.zeros();
                // matrixVt.zeros();
                return wrap(1E+300);
            }
        }
    }
    else if((boundtype=='a') | (boundtype=='r')){
        if(arma::eig_gen(eigval, matrixF - vecG * rowvecW)){
            if(max(abs(eigval))> (1 + 1E-50)){
                return wrap(max(abs(eigval))*1E+100);
            }
        }
        else{
            return wrap(1E+300);
        }
    }

    if(matrixAt(0,0)!=0){
        arma::rowvec rowvecWX(matFX_n.nrow(), arma::fill::ones);
        if(arma::eig_gen(eigval, matrixFX - vecGX * rowvecWX)){
            if(max(abs(eigval))> (1 + 1E-50)){
                return wrap(max(abs(eigval))*1E+100);
            }
        }
        else{
            return wrap(1E+300);
        }
    }

    return wrap(occurrenceOptimizer(matrixVt, matrixF, rowvecW, vecG, vecYt,
                                    lags, E, T, S, O,
                                    matrixXt, matrixAt, matrixFX, vecGX));
}
