#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// Fitter for vector models
List vFitter(arma::mat &matrixV, arma::mat const &matrixF, arma::mat const &matrixW, arma::mat const &matrixY, arma::mat const &matrixG,
             arma::uvec &lags, char const &E, char const &T, char const &S,
             arma::mat const &matrixX, arma::mat &matrixA, arma::mat const &matrixFX, arma::mat const &matrixGX, arma::mat const &matrixO) {
    /* matrixY has nrow = nSeries, ncol = obs
     * matrixV has nrow = nSeries * nComponents, ncol = obs + maxlag
     * matrixW, matrixF, matrixG are nSeries * nComponents x nSeries * nComponents.
     * lags is a vector of lags of length nSeries * nComponents
     * matrixX and matrixA are not defined yet.
     */

    int obs = matrixY.n_cols;
    int nSeries = matrixY.n_rows;
    int obsall = matrixV.n_cols;
    // unsigned int nComponents = matrixV.n_rows / nSeries;
    unsigned int maxlag = max(lags);
    int lagslength = lags.n_rows;

    lags = lags * lagslength;

    for(int i=0; i<lagslength; i=i+1){
        lags(i) = lags(i) + (lagslength - i - 1);
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);

    arma::mat matrixYfit(nSeries, obs, arma::fill::zeros);
    arma::mat matrixE(nSeries, obs, arma::fill::zeros);
    // arma::mat bufferforat(matrixGX.n_rows);

    for (unsigned int i=maxlag; i<obs+maxlag; i=i+1) {
        lagrows = (i+1) * lagslength - lags - 1;

        /* # Measurement equation and the error term */
        matrixYfit.row(i-maxlag) = matrixO(i-maxlag) * (matrixW * matrixV(lagrows));
        matrixE(i-maxlag) = (matrixY(i-maxlag) - matrixYfit(i-maxlag));

        /* # Transition equation */
        matrixV.col(i) = matrixF * matrixV(lagrows) + matrixG * matrixE(i-maxlag);

        /* Failsafe for cases when unreasonable value for state vector was produced */
        //         if(!matrixV.col(i).is_finite()){
        //             matrixV.col(i) = matrixV(lagrows);
        //         }
        //         if((S=='M') & (matrixV(matrixV.n_rows-1,i) <= 0)){
        //             matrixV(matrixV.n_rows-1,i) = arma::as_scalar(matrixV(lagrows.row(matrixV.n_rows-1)));
        //         }
        //         if(T=='M'){
        //             if((matrixV(0,i) <= 0) | (matrixV(1,i) <= 0)){
        //                 matrixV(0,i) = arma::as_scalar(matrixV(lagrows.row(0)));
        //                 matrixV(1,i) = arma::as_scalar(matrixV(lagrows.row(1)));
        //             }
        //         }
        //
        // /* Renormalise components if the seasonal model is chosen */
        //         if(S!='N'){
        //             if(double(i+1) / double(maxlag) == double((i+1) / maxlag)){
        //                 matrixV.cols(i-maxlag+1,i) = normaliser(matrixV.cols(i-maxlag+1,i), obsall, maxlag, S, T);
        //             }
        //         }

        /* # Transition equation for xreg */
        // bufferforat = gXvalue(matrixX.col(i-maxlag), matrixGX, matrixE.row(i-maxlag), E);
        // matrixA.col(i) = matrixFX * matrixA.col(i-1) + bufferforat;
    }

    for (int i=obs+maxlag; i<obsall; i=i+1) {
        lagrows = (i+1) * lagslength - lags - 1;
        matrixV.col(i) = matrixF * matrixV(lagrows);
        // matrixA.col(i) = matrixFX * matrixA.col(i-1);
    }

    return List::create(Named("matvt") = matrixV, Named("yfit") = matrixYfit,
                        Named("errors") = matrixE, Named("matat") = matrixA);
}

/* # Wrapper for fitter */
// [[Rcpp::export]]
RcppExport SEXP vFitterWrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP matG,
                            SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype,
                            SEXP matxt, SEXP matat, SEXP matFX, SEXP matGX, SEXP ot) {

    NumericMatrix matvt_n(matvt);
    arma::mat matrixV(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::mat matrixW(matw_n.begin(), matw_n.nrow(), matw_n.ncol(), false);

    NumericMatrix yt_n(yt);
    arma::mat matrixY(yt_n.begin(), yt_n.nrow(), yt_n.ncol(), false);

    NumericMatrix matG_n(matG);
    arma::mat matrixG(matG_n.begin(), matG_n.nrow(), matG_n.ncol(), false);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixX(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixA(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix matGX_n(matGX);
    arma::mat matrixGX(matGX_n.begin(), matGX_n.nrow(), matGX_n.ncol(), false);

    NumericMatrix ot_n(ot);
    arma::mat matrixO(ot_n.begin(), ot_n.nrow(), ot_n.ncol(), false);

    return wrap(vFitter(matrixV, matrixF, matrixW, matrixY, matrixG, lags, E, T, S,
                        matrixX, matrixA, matrixFX, matrixGX, matrixO));
}


/* # Function produces the point forecasts for the specified model */
arma::mat vForecaster(arma::mat const & matrixV, arma::mat const &matrixF, arma::mat const &matrixW,
                      unsigned int const &nSeries, unsigned int const &hor, char const &E, char const &T, char const &S, arma::uvec lags,
                      arma::mat const &matrixX, arma::mat const &matrixA, arma::mat const &matrixFX){
    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);
    unsigned int hh = hor + maxlag;

    arma::uvec lagrows(lagslength, arma::fill::zeros);
    arma::mat matYfor(nSeries, hor, arma::fill::zeros);
    arma::mat matrixVnew(hh, matrixV.n_cols, arma::fill::zeros);
    // arma::mat matrixAnew(hh, matrixA.n_cols, arma::fill::zeros);

    lags = lags * lagslength;

    for(int i=0; i<lagslength; i=i+1){
        lags(i) = lags(i) + (lagslength - i - 1);
    }

    matrixVnew.submat(0,0,matrixVnew.n_cols-1,maxlag-1) = matrixV.submat(0,0,matrixVnew.n_cols-1,maxlag-1);
    // matrixAnew.submat(0,0,maxlag-1,matrixAnew.n_cols-1) = matrixAnew.submat(0,0,maxlag-1,matrixAnew.n_cols-1);

    /* # Fill in the new xt matrix using F. Do the forecasts. */
    for (unsigned int i=maxlag; i<(hor+maxlag); i=i+1) {
        lagrows = (i+1) * lagslength - lags - 1;

        /* # Transition equation */
        matrixVnew.col(i) = matrixF * matrixVnew(lagrows);
        // matrixAnew.row(i) = matrixAnew.row(i-1) * matrixFX;

        matYfor.col(i-maxlag) = matrixW * matrixVnew(lagrows);
    }

    return matYfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP vForecasterWrap(SEXP matvt, SEXP matF, SEXP matw,
                                SEXP series, SEXP h, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP modellags,
                                SEXP matxt, SEXP matat, SEXP matFX){

    NumericMatrix matvt_n(matvt);
    arma::mat matrixV(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol(), false);

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::mat matrixW(matw_n.begin(), matw_n.nrow(), matw_n.ncol(), false);

    unsigned int nSeries = as<int>(series);
    unsigned int hor = as<int>(h);
    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixX(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixA(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    return wrap(vForecaster(matrixV, matrixF, matrixW, nSeries, hor, E, T, S, lags, matrixX, matrixA, matrixFX));
}


/* # Function returns the chosen Cost Function based on the chosen model and produced errors */
double vOptimiser(arma::mat &matrixV, arma::mat const &matrixF, arma::mat const &matrixW, arma::mat const &matrixY, arma::mat const &matrixG,
                  unsigned int const &hor, arma::uvec &lags, char const &E, char const &T, char const &S,
                  double const &normalize,
                  arma::mat const &matrixX, arma::mat &matrixA, arma::mat const &matrixFX, arma::mat const &matrixGX, arma::mat const &matrixO){
    // bool const &multi, std::string const &CFtype, char const &fitterType,
    // # Make decomposition functions shut up!
    std::ostream nullstream(0);
    arma::set_stream_err2(nullstream);

    arma::uvec nonzeroes = find(matrixO>0);
    int obs = nonzeroes.n_rows;
    double CFres = 0;
    int matobs = obs + hor - 1;

    // yactsum is needed for multiplicative error models
    // double yactsum = arma::as_scalar(sum(log(matrixY.elem(nonzeroes))));

    List fitting = vFitter(matrixV, matrixF, matrixW, matrixY, matrixG, lags, E, T, S,
                      matrixX, matrixA, matrixFX, matrixGX, matrixO);

    NumericMatrix mvtfromfit = as<NumericMatrix>(fitting["matvt"]);
    matrixV = as<arma::mat>(mvtfromfit);
    NumericMatrix errorsfromfit = as<NumericMatrix>(fitting["errors"]);
    // NumericMatrix matrixAfromfit = as<NumericMatrix>(fitting["matat"]);
    // matrixA = as<arma::mat>(matrixAfromfit);

    arma::mat matErrors(errorsfromfit.begin(), errorsfromfit.nrow(), errorsfromfit.ncol(), false);;
    // arma::mat matErrorsfromfit(errorsfromfit.begin(), errorsfromfit.nrow(), errorsfromfit.ncol(), false);
    // matErrors = matErrorsfromfit;
    matErrors = matErrors.elem(nonzeroes);
    // if(E=='M'){
    //     matErrors = log(1 + matErrors);
    // }

    // arma::vec veccij(hor, arma::fill::ones);
    // arma::mat matrixSigma(hor, hor, arma::fill::eye);

    try{
        CFres = double(log(arma::prod(eig_sym(trans(matErrors / normalize) * (matErrors / normalize) / matobs))) +
            hor * log(pow(normalize,2)));
    }
    catch(const std::runtime_error){
        CFres = double(log(arma::det(arma::trans(matErrors / normalize) * (matErrors / normalize) / matobs)) +
            hor * log(pow(normalize,2)));
    }

    return CFres;
}


/* # This is a wrapper for optimizer, which currently uses admissible bounds */
// [[Rcpp::export]]
RcppExport SEXP vOptimiserWrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP matG,
                               SEXP h, SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype,
                               SEXP normalizer,
                               SEXP matxt, SEXP matat, SEXP matFX, SEXP matGX, SEXP ot) {
    // SEXP multisteps, SEXP CFt, SEXP fittertype, SEXP bounds
    /* Function is needed to implement admissible constrains on smoothing parameters */
    NumericMatrix matvt_n(matvt);
    arma::mat matrixV(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::mat matrixW(matw_n.begin(), matw_n.nrow(), matw_n.ncol(), false);

    NumericMatrix yt_n(yt);
    arma::mat matrixY(yt_n.begin(), yt_n.nrow(), yt_n.ncol(), false);

    NumericMatrix matG_n(matG);
    arma::mat matrixG(matG_n.begin(), matG_n.nrow(), matG_n.ncol(), false);

    int hor = as<int>(h);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    // bool multi = as<bool>(multisteps);

    // std::string CFtype = as<std::string>(CFt);

    // char fitterType = as<char>(fittertype);

    // char boundtype = as<char>(bounds);

    double normalize = as<double>(normalizer);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixX(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixA(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix matGX_n(matGX);
    arma::mat matrixGX(matGX_n.begin(), matGX_n.nrow(), matGX_n.ncol(), false);

    NumericMatrix ot_n(ot);
    arma::mat matrixO(ot_n.begin(), ot_n.nrow(), ot_n.ncol(), false);

    // Values needed for eigenvalues calculation
    arma::cx_vec eigval;

    //     if(boundtype=='u'){
    // // alpha in (0,1)
    //         if((vecG(0)>1) || (vecG(0)<0)){
    //             vecG.zeros();
    //             matrixVt.zeros();
    //         }
    //         if(T!='N'){
    // // beta in (0,alpha)
    //             if((vecG(1)>vecG(0)) || (vecG(1)<0)){
    //                 vecG.zeros();
    //                 matrixVt.zeros();
    //             }
    //             if(S!='N'){
    // // gamma in (0,1-alpha)
    //                 if((vecG(2)>(1-vecG(0))) || (vecG(2)<0)){
    //                     vecG.zeros();
    //                     matrixVt.zeros();
    //                 }
    //             }
    //         }
    //         if(S!='N'){
    // // gamma in (0,1-alpha)
    //             if((vecG(1)>(1-vecG(0))) || (vecG(1)<0)){
    //                 vecG.zeros();
    //                 matrixVt.zeros();
    //             }
    //         }
    //     }
    // else if(boundtype=='a'){
    if(arma::eig_gen(eigval, matrixF - matrixG * matrixW)){
        if(max(abs(eigval)) > (1 + 1E-50)){
            return wrap(max(abs(eigval))*1E+100);
        }
    }
    else{
        return wrap(1E+300);
    }
    // }

    // multi, CFtype, fitterType,
    return wrap(vOptimiser(matrixV, matrixF, matrixW, matrixY, matrixG,
                           hor, lags, E, T, S,
                           normalize,
                           matrixX, matrixA, matrixFX, matrixGX, matrixO));
}
