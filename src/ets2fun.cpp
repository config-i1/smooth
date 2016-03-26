#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/* # Function returns multiplicative or additive error for scalar */
double errorf(double yact, double yfit, char Etype){
    if(Etype=='A'){
        return yact - yfit;
    }
    else{
        if(yfit==0){
            return R_PosInf;
        }
        else{
            return (yact - yfit) / yfit;
        }
    }
}

/* # Function is needed to estimate the correct error for ETS when multisteps model selection with r(matxt) is sorted out. */
arma::mat errorvf(arma::mat yact, arma::mat yfit, char Etype){
    if(Etype=='A'){
        return yact - yfit;
    }
    else{
        yfit.elem(find(yfit==0)).fill(1e-100);
        return (yact - yfit) / yfit;
    }
}

/* # Function returns value of w() -- y-fitted -- used in the measurement equation */
double wvalue(arma::vec matrixxt, arma::rowvec rowvecW, char T, char S){
// matrixxt is a vector here!
    double yfit;

    switch(S){
// ZZN
    case 'N':
        switch(T){
        case 'N':
        case 'A':
            yfit = as_scalar(rowvecW * matrixxt);
        break;
        case 'M':
            yfit = as_scalar(exp(rowvecW * log(matrixxt)));
        break;
        }
    break;
// ZZA
    case 'A':
        switch(T){
        case 'N':
        case 'A':
            yfit = as_scalar(rowvecW * matrixxt);
        break;
        case 'M':
            yfit = as_scalar(exp(rowvecW.cols(0,1) * log(matrixxt.rows(0,1)))) + matrixxt(2);
        break;
        }
    break;
// ZZM
    case 'M':
        switch(T){
        case 'N':
        case 'M':
            yfit = as_scalar(exp(rowvecW * log(matrixxt)));
        break;
        case 'A':
            yfit = as_scalar(rowvecW.cols(0,1) * matrixxt.rows(0,1)) * matrixxt(2);
        break;
        }
    break;
    }

    return yfit;
}

/* # Function returns value of r() -- additive or multiplicative error -- used in the error term of measurement equation.
     This is mainly needed by sim.ets */
double rvalue(arma::vec matrixxt, arma::rowvec rowvecW, char E, char T, char S){
    double yfit = 1;

    switch(E){
// MZZ
    case 'M':
        return wvalue(matrixxt, rowvecW, T, S);
    break;
// AZZ
    case 'A':
    default:
        return yfit;
    }
}

/* # Function returns value of f() -- new states without the update -- used in the transition equation */
arma::vec fvalue(arma::vec matrixxt, arma::mat matrixF, char T, char S){
    arma::vec matrixxtnew = matrixxt;

    switch(S){
// ZZN
    case 'N':
        switch(T){
        case 'N':
        case 'A':
            matrixxtnew = matrixF * matrixxt;
        break;
        case 'M':
            matrixxtnew = exp(matrixF * log(matrixxt));
        break;
        }
    break;
// ZZA
    case 'A':
        switch(T){
        case 'N':
        case 'A':
            matrixxtnew = matrixF * matrixxt;
        break;
        case 'M':
            matrixxtnew.rows(0,1) = exp(matrixF.submat(0,0,1,1) * log(matrixxt.rows(0,1)));
            matrixxtnew(2) = matrixxt(2);
        break;
        }
    break;
// ZZM
    case 'M':
        switch(T){
        case 'N':
        case 'M':
            matrixxtnew = exp(matrixF * log(matrixxt));
        break;
        case 'A':
            matrixxtnew = matrixF * matrixxt;
        break;
        }
    break;
    }

    return matrixxtnew;
}

/* # Function returns value of g() -- the update of states -- used in components estimation for the persistence */
arma::vec gvalue(arma::vec matrixxt, arma::mat matrixF, arma::mat rowvecW, char E, char T, char S){
    arma::vec g(matrixxt.n_rows, arma::fill::ones);

// AZZ
    switch(E){
    case 'A':
// ANZ
        switch(T){
        case 'N':
            switch(S){
            case 'M':
                g(0) = 1 / matrixxt(1);
                g(1) = 1 / matrixxt(0);
            break;
            }
        break;
// AAZ
        case 'A':
            switch(S){
            case 'M':
                g.rows(0,1) = g.rows(0,1) / matrixxt(2);
                g(2) = 1 / as_scalar(rowvecW.cols(0,1) * matrixxt.rows(0,1));
            break;
            }
        break;
// AMZ
        case 'M':
            switch(S){
            case 'N':
            case 'A':
                g(1) = g(1) / matrixxt(0);
            break;
            case 'M':
                g(0) = g(0) / matrixxt(2);
                g(1) = g(1) / (matrixxt(0) * matrixxt(2));
                g(2) = g(2) / as_scalar(exp(rowvecW.cols(0,1) * log(matrixxt.rows(0,1))));
            break;
            }
        break;
        }
    break;
// MZZ
    case 'M':
// MNZ
        switch(T){
        case 'N':
            switch(S){
            case 'N':
                g(0) = matrixxt(0);
            break;
            case 'A':
                g.rows(0,1).fill(matrixxt(0) + matrixxt(1));
            break;
            case 'M':
                g = matrixxt;
            break;
            }
        break;
// MAZ
        case 'A':
            switch(S){
            case 'N':
            case 'A':
                g.fill(as_scalar(rowvecW * matrixxt));
            break;
            case 'M':
                g.rows(0,1).fill(as_scalar(rowvecW.cols(0,1) * matrixxt.rows(0,1)));
                g(2) = matrixxt(2);
            break;
            }
        break;
// MMZ
        case 'M':
            switch(S){
            case 'N':
                g(0) = as_scalar(exp(rowvecW * log(matrixxt)));
                g(1) = pow(matrixxt(1),rowvecW(1));
            break;
            case 'A':
                g.rows(0,2).fill(as_scalar(exp(rowvecW.cols(0,1) * log(matrixxt.rows(0,1))) + matrixxt(2)));
                g(1) = g(0) / matrixxt(0);
            break;
            case 'M':
                g = exp(matrixF * log(matrixxt));
            break;
            }
        break;
        }
    break;
    }

    return g;
}

/* # Function is needed for the renormalisation of seasonal components. NOT IMPLEMENTED YET! */
arma::mat avalue(int maxlag, double(error), double gamma, double yfit, char E, char S, char T){
    arma::mat a(1,maxlag);
    if(S=='A'){
        if(E=='M'){
            a.fill(gamma / maxlag * yfit * error);
        }
        else{
            a.fill(gamma / maxlag * error);
        }
    }
    else if(S=='M'){
        if(E=='M'){
            a.fill(1 + gamma / maxlag * yfit * error);
        }
        else{
            a.fill(1 + gamma / maxlag * error);
        }
    }
    else{
        a.fill(0);
    }
    return(a);
}

/* # initparams - function that initialises the basic parameters of ETS */
// [[Rcpp::export]]
RcppExport SEXP initparams(SEXP Ttype, SEXP Stype, SEXP datafreq, SEXP obsR, SEXP yt,
                           SEXP damped, SEXP phi, SEXP smoothingparameters, SEXP initialstates, SEXP seasonalcoefs){

    char T = as<char>(Ttype);
    char S = as<char>(Stype);
    int freq = as<int>(datafreq);
    int obs = as<int>(obsR);
    NumericMatrix vyt(yt);
    arma::mat vecY(vyt.begin(), vyt.nrow(), vyt.ncol(), false);
    bool damping = as<bool>(damped);
    double phivalue;
    if(!Rf_isNull(phi)){
        phivalue = as<double>(phi);
    }

    NumericMatrix smoothingparam(smoothingparameters);
    arma::mat persistence(smoothingparam.begin(), smoothingparam.nrow(), smoothingparam.ncol(), false);
    NumericMatrix initials(initialstates);
    arma::mat initial(initials.begin(), initials.nrow(), initials.ncol(), false);
    NumericMatrix seasonalc(seasonalcoefs);
    arma::mat seascoef(seasonalc.begin(), seasonalc.nrow(), seasonalc.ncol(), false);

    unsigned int ncomponents = 1;
    int maxlag = 1;
    arma::vec modellags(3, arma::fill::ones);

/* # Define the number of components */
    if(T!='N'){
        ncomponents += 1;
        if(S!='N'){
            ncomponents += 1;
            maxlag = freq;
            modellags(2) = freq;
        }
        else{
            modellags.resize(2);
        }
    }
    else{
/* # Define the number of components and model frequency */
        if(S!='N'){
            ncomponents += 1;
            maxlag = freq;
            modellags(1) = freq;
            modellags.resize(2);
        }
        else{
            modellags.resize(1);
        }
    }

    arma::mat matrixxt(obs+maxlag, ncomponents, arma::fill::ones);
    arma::vec vecG(ncomponents, arma::fill::zeros);
    bool estimphi = TRUE;

// # Define the initial states for level and trend components
    switch(T){
    case 'N':
        matrixxt.submat(0,0,maxlag-1,0).each_row() = initial.submat(0,2,0,2);
    break;
    case 'A':
        matrixxt.submat(0,0,maxlag-1,1).each_row() = initial.submat(0,0,0,1);
    break;
// # The initial matrix is filled with ones, that is why we don't need to fill in initial trend
    case 'M':
        matrixxt.submat(0,0,maxlag-1,1).each_row() = initial.submat(0,2,0,3);
    break;
    }

/* # Define the initial states for seasonal component */
    switch(S){
    case 'A':
        matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = seascoef.col(0);
    break;
    case 'M':
        matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = seascoef.col(1);
    break;
    }

//    matrixxt.resize(obs+maxlag, ncomponents);

    if(persistence.n_rows < ncomponents){
        if((T=='M') | (S=='M')){
            vecG = persistence.submat(0,1,persistence.n_rows-1,1);
        }
        else{
            vecG = persistence.submat(0,0,persistence.n_rows-1,0);
        }
    }
    else{
        if((T=='M') | (S=='M')){
            vecG = persistence.submat(0,1,ncomponents-1,1);
        }
        else{
            vecG = persistence.submat(0,0,ncomponents-1,0);
        }
    }

    if(Rf_isNull(phi)){
        if(damping==TRUE){
            phivalue = 0.95;
        }
        else{
            phivalue = 1.0;
            estimphi = FALSE;
        }
    }
    else{
        if(damping==FALSE){
            phivalue = 1.0;
        }
        estimphi = FALSE;
    }

    return wrap(List::create(Named("n.components") = ncomponents, Named("maxlag") = maxlag, Named("modellags") = modellags,
                             Named("matxt") = matrixxt, Named("vecg") = vecG, Named("estimate.phi") = estimphi,
                             Named("phi") = phivalue));
}

/*
# etsmatrices - function that returns matF and matw.
# Needs to be stand alone to change the damping parameter during the estimation.
# Cvalues includes persistence, phi, initials, intials for seasons, matrixX coeffs.
*/
// [[Rcpp::export]]
RcppExport SEXP etsmatrices(SEXP matxt, SEXP vecg1, SEXP phi, SEXP Cvalues, SEXP ncomponentsR,
                            SEXP modellags, SEXP Ttype, SEXP Stype, SEXP nexovars, SEXP matxtreg,
                            SEXP estimpersistence, SEXP estimphi, SEXP estiminit, SEXP estiminitseason, SEXP estimxreg){

    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());

    NumericMatrix vg(vecg1);
    arma::vec vecG(vg.begin(), vg.nrow(), false);

    double phivalue = as<double>(phi);

    NumericMatrix Cv(Cvalues);
    arma::rowvec C(Cv.begin(), Cv.ncol(), false);

    int ncomponents = as<int>(ncomponentsR);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);
    int maxlag = max(lags);

    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    int nexo = as<int>(nexovars);

    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    bool estimatepersistence = as<bool>(estimpersistence);
    bool estimatephi = as<bool>(estimphi);
    bool estimateinitial = as<bool>(estiminit);
    bool estimateinitialseason = as<bool>(estiminitseason);
    bool estimatexreg = as<bool>(estimxreg);

    arma::mat matrixF(1,1,arma::fill::ones);
    arma::mat rowvecW(1,1,arma::fill::ones);

    if(estimatepersistence==TRUE){
        vecG = C.cols(0,ncomponents-1).t();
    }

    if(estimatephi==TRUE){
        phivalue = as_scalar(C.cols(ncomponents*estimatepersistence,ncomponents*estimatepersistence));
    }

    if(estimateinitial==TRUE){
        matrixxt.col(0).fill(as_scalar(C.cols(ncomponents*estimatepersistence + estimatephi,ncomponents*estimatepersistence + estimatephi).t()));
        if(T!='N'){
            matrixxt.col(1).fill(as_scalar(C.cols(ncomponents*estimatepersistence + estimatephi + 1,ncomponents*estimatepersistence + estimatephi + 1).t()));
        }
    }

    if(S!='N'){
        if(estimateinitialseason==TRUE){
            matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = C.cols(ncomponents*estimatepersistence + estimatephi + (ncomponents - 1)*estimateinitial,ncomponents*estimatepersistence + estimatephi + (ncomponents - 1)*estimateinitial + maxlag - 1).t();
/* # Normalise the initial seasons */
            if(S=='A'){
                matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) - as_scalar(mean(matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1)));
            }
            else{
                matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = exp(log(matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1)) - as_scalar(mean(log(matrixxt.submat(0,ncomponents-1,maxlag-1,ncomponents-1)))));
            }
        }
    }

    if(estimatexreg==TRUE){
        matrixX.each_row() = C.cols(C.n_cols - nexo,C.n_cols - 1);
    }

/* # The default values of matrices are set for ZNN  models */
    switch(S){
    case 'N':
        switch(T){
        case 'A':
        case 'M':
            matrixF.set_size(2,2);
            matrixF(0,0) = 1.0;
            matrixF(1,0) = 0.0;
            matrixF(0,1) = phivalue;
            matrixF(1,1) = phivalue;

            rowvecW.set_size(1,2);
            rowvecW(0,0) = 1.0;
            rowvecW(0,1) = phivalue;
        break;
        }
    break;
    case 'A':
    case 'M':
        switch(T){
        case 'A':
        case 'M':
            matrixF.set_size(3,3);
            matrixF.fill(0.0);
            matrixF(0,0) = 1.0;
            matrixF(1,0) = 0.0;
            matrixF(0,1) = phivalue;
            matrixF(1,1) = phivalue;
            matrixF(2,2) = 1.0;

            rowvecW.set_size(1,3);
            rowvecW(0,0) = 1.0;
            rowvecW(0,1) = phivalue;
            rowvecW(0,2) = 1.0;
        break;
        case 'N':
            matrixF.set_size(2,2);
            matrixF.fill(0.0);
            matrixF.diag().fill(1.0);

            rowvecW.set_size(1,2);
            rowvecW.fill(1.0);
        break;
        }
    }

    return wrap(List::create(Named("matF") = matrixF, Named("matw") = rowvecW, Named("vecg") = vecG,
                              Named("phi") = phivalue, Named("matxt") = matrixxt, Named("matxtreg") = matrixX));
}

List fitter(arma::mat matrixxt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecY, arma::vec vecG,
             arma::uvec lags, char E, char T, char S,
             arma::mat matrixWX, arma::mat matrixX, arma::mat matrixweightsX, arma::mat matrixFX, arma::vec vecGX) {
    /* # matrixxt should have a length of obs + maxlag.
    * # rowvecW should have 1 row.
    * # matgt should be a vector
    * # lags is a vector of lags
    * # matrixWX is the matrix with the exogenous variables
    * # matrixX is the matrix with the parameters for the exogenous
    */

    int obs = vecY.n_rows;
    int obsall = matrixxt.n_rows;
    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);

    lags = maxlag - lags;

    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + obsall * i;
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);

    arma::vec matyfit(obs, arma::fill::zeros);
    arma::vec materrors(obs, arma::fill::zeros);
    arma::rowvec bufferforxtreg(vecGX.n_rows);

    for (int i=maxlag; i<obsall; i=i+1) {

        lagrows = lags - maxlag + i;

/* # Measurement equation and the error term */
        matyfit.row(i-maxlag) = wvalue(matrixxt(lagrows), rowvecW, T, S) +
                                       matrixWX.row(i-maxlag) * arma::trans(matrixX.row(i-maxlag));
        materrors(i-maxlag) = errorf(vecY(i-maxlag), matyfit(i-maxlag), E);

/* # Transition equation */
        matrixxt.row(i) = arma::trans(fvalue(matrixxt(lagrows), matrixF, T, S) +
                                      gvalue(matrixxt(lagrows), matrixF, rowvecW, E, T, S) % vecG * materrors(i-maxlag));

/* # Transition equation for xreg */
        bufferforxtreg = arma::trans(vecGX / arma::trans(matrixweightsX.row(i-maxlag)) * materrors(i-maxlag));
        bufferforxtreg.elem(find_nonfinite(bufferforxtreg)).fill(0);
        matrixX.row(i) = matrixX.row(i-1) * matrixFX + bufferforxtreg;
    }

    return List::create(Named("matxt") = matrixxt, Named("yfit") = matyfit,
                        Named("errors") = materrors, Named("xtreg") = matrixX);
}

/* # Wrapper for fitter2 */
// [[Rcpp::export]]
RcppExport SEXP fitterwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg1,
                            SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype,
                            SEXP matwex, SEXP matxtreg, SEXP matv, SEXP matF2, SEXP vecg2) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());

    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);

    NumericMatrix vw(matw);
    arma::rowvec rowvecW(vw.begin(), vw.ncol(), false);

    NumericMatrix vyt(yt);
    arma::vec vecY(vyt.begin(), vyt.nrow(), false);

    NumericMatrix vg(vecg1);
    arma::vec vecG(vg.begin(), vg.nrow(), false);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    NumericMatrix mwex(matwex);
    arma::mat matrixWX(mwex.begin(), mwex.nrow(), mwex.ncol(), false);

    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    NumericMatrix vv(matv);
    arma::mat matrixweightsX(vv.begin(), vv.nrow(), vv.ncol(), false);

    NumericMatrix mF2(matF2);
    arma::mat matrixFX(mF2.begin(), mF2.nrow(), mF2.ncol(), false);

    NumericMatrix vg2(vecg2);
    arma::vec vecGX(vg2.begin(), vg2.nrow(), false);

    return wrap(fitter(matrixxt, matrixF, rowvecW, vecY, vecG, lags, E, T, S,
                matrixWX, matrixX, matrixweightsX, matrixFX, vecGX));
}

/* # Function fills in the values of the provided matrixX using the transition matrix. Needed for forecast of coefficients of xreg. */
List statetail(arma::mat matrixxt, arma::mat matrixF, arma::mat matrixX, arma::mat matrixFX,
               arma::uvec lags, char T, char S){

    int obsall = matrixxt.n_rows;
    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);

    lags = maxlag - lags;

    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + obsall * i;
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);

    for (int i=maxlag; i<obsall; i=i+1) {
        lagrows = lags - maxlag + i;
        matrixxt.row(i) = arma::trans(fvalue(matrixxt(lagrows), matrixF, T, S));
      }

    for(int i=0; i<(matrixX.n_rows-1); i=i+1){
        matrixX.row(i+1) = matrixX.row(i) * matrixFX;
    }

    return(List::create(Named("matxt") = matrixxt, Named("xtreg") = matrixX));
}

/* # Wrapper for ssstatetail */
// [[Rcpp::export]]
RcppExport SEXP statetailwrap(SEXP matxt, SEXP matF, SEXP matxtreg, SEXP matF2,
                              SEXP modellags, SEXP Ttype, SEXP Stype){
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());
    NumericMatrix mF2(matF2);
    arma::mat matrixFX(mF2.begin(), mF2.nrow(), mF2.ncol(), false);
    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    return(wrap(statetail(matrixxt, matrixF, matrixX, matrixFX, lags, T, S)));
}

/* # Function produces the point forecasts for the specified model */
arma::mat forecaster(arma::mat matrixxt, arma::mat matrixF, arma::rowvec rowvecW,
                     unsigned int hor, char T, char S, arma::uvec lags,
                     arma::mat matrixWX, arma::mat matrixX) {
    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);
    unsigned int hh = hor + maxlag;

    arma::uvec lagrows(lagslength, arma::fill::zeros);
    arma::vec matyfor(hor, arma::fill::zeros);
    arma::mat matrixxtnew(hh, matrixxt.n_cols, arma::fill::zeros);
    arma::mat matrixxtregnew(hh, matrixX.n_cols, arma::fill::zeros);

    lags = maxlag - lags;
    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + hh * i;
    }

    matrixxtnew.submat(0,0,maxlag-1,matrixxtnew.n_cols-1) = matrixxt.submat(0,0,maxlag-1,matrixxtnew.n_cols-1);

/* # Fill in the new xt matrix using F. Do the forecasts. */
    for (int i=maxlag; i<(hor+maxlag); i=i+1) {
        lagrows = lags - maxlag + i;
        matrixxtnew.row(i) = arma::trans(fvalue(matrixxtnew(lagrows), matrixF, T, S));
        matyfor.row(i-maxlag) = wvalue(matrixxtnew(lagrows), rowvecW, T, S) + matrixWX.row(i-maxlag) * arma::trans(matrixX.row(i-maxlag));
    }

    return matyfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP forecasterwrap(SEXP matxt, SEXP matF, SEXP matw,
                               SEXP h, SEXP Ttype, SEXP Stype, SEXP modellags,
                               SEXP matwex, SEXP matxtreg){
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol(), false);

    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);

    NumericMatrix vw(matw);
    arma::mat rowvecW(vw.begin(), vw.nrow(), vw.ncol(), false);

    unsigned int hor = as<int>(h);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    NumericMatrix mwex(matwex);
    arma::mat matrixWX(mwex.begin(), mwex.nrow(), mwex.ncol(), false);

    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return wrap(forecaster(matrixxt, matrixF, rowvecW, hor, T, S, lags, matrixWX, matrixX));
}

/* # Function produces matrix of errors based on multisteps forecast */
arma::mat errorer(arma::mat matrixxt, arma::mat matrixF, arma::mat rowvecW, arma::mat vecY,
                  int hor, char E, char T, char S, arma::uvec lags,
                  arma::mat matrixWX, arma::mat matrixX){
    int obs = vecY.n_rows;
    int hh = 0;
    arma::mat materrors(obs, hor);
    unsigned int maxlag = max(lags);

    materrors.fill(NA_REAL);

    for(int i = 0; i < obs; i=i+1){
        hh = std::min(hor, obs-i);
        materrors.submat(i, 0, i, hh-1) = trans(errorvf(vecY.rows(i, i+hh-1),
            forecaster(matrixxt.rows(i,i+maxlag-1), matrixF, rowvecW, hh, T, S, lags, matrixWX.rows(i, i+hh-1),
                matrixX.rows(i, i+hh-1)), E));
    }
    return materrors;
}

/* # Wrapper for errorer */
// [[Rcpp::export]]
RcppExport SEXP errorerwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt,
                            SEXP h, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP modellags,
                            SEXP matwex, SEXP matxtreg, SEXP matv, SEXP matF2, SEXP vecg2){
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol(), false);

    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);

    NumericMatrix vw(matw);
    arma::rowvec rowvecW(vw.begin(), vw.ncol(), false);

    NumericMatrix vyt(yt);
    arma::vec vecY(vyt.begin(), vyt.nrow(), false);

    int hor = as<int>(h);
    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    NumericMatrix mwex(matwex);
    arma::mat matrixWX(mwex.begin(), mwex.nrow(), mwex.ncol(), false);

    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

/*    NumericMatrix vv(matv);
    arma::mat matrixweightsX(vv.begin(), vv.nrow(), vv.ncol(), false);

    NumericMatrix mF2(matF2);
    arma::mat matrixFX(mF2.begin(), mF2.nrow(), mF2.ncol(), false);

    NumericMatrix vg2(vecg2);
    arma::vec vecGX(vg2.begin(), vg2.nrow(), false); */

    return wrap(errorer(matrixxt, matrixF, rowvecW, vecY, hor, E, T, S, lags,
                        matrixWX, matrixX));
}

int CFtypeswitch (std::string const& CFtype) {
    if (CFtype == "GV") return 1;
    if (CFtype == "trace") return 2;
    if (CFtype == "TV") return 3;
    if (CFtype == "MSEh") return 4;
    if (CFtype == "MAE") return 5;
    if (CFtype == "HAM") return 6;
    if (CFtype == "MSE") return 7;
    else return 7;
}

/* # Function returns the chosen Cost Function based on the chosen model and produced errors */
double optimizer(arma::mat matrixxt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecY, arma::vec vecG,
                 unsigned int hor, arma::uvec lags, char E, char T, char S, bool multi, std::string CFtype, double normalize,
                 arma::mat  matrixWX, arma::mat matrixX, arma::mat matrixweightsX, arma::mat matrixFX, arma::vec vecGX){
// # Make decomposition functions shut up!
    std::ostream nullstream(0);
    arma::set_stream_err2(nullstream);

    int obs = vecY.n_rows;
    double CFres = 0;
    int matobs = obs - hor + 1;
    double yactsum = arma::as_scalar(arma::sum(log(vecY)));
    unsigned int maxlag = max(lags);

    List fitting = fitter(matrixxt, matrixF, rowvecW, vecY, vecG, lags, E, T, S,
                          matrixWX, matrixX, matrixweightsX, matrixFX, vecGX);
    NumericMatrix mxtfromfit = as<NumericMatrix>(fitting["matxt"]);
    matrixxt = as<arma::mat>(mxtfromfit);
    NumericMatrix errorsfromfit = as<NumericMatrix>(fitting["errors"]);
    NumericMatrix mxtregfromfit = as<NumericMatrix>(fitting["xtreg"]);
    matrixX = as<arma::mat>(mxtregfromfit);

    arma::mat materrors;
    arma::rowvec horvec(hor);

    if(multi==true){
        for(int i=0; i<hor; i=i+1){
            horvec(i) = hor - i;
        }
        materrors = errorer(matrixxt, matrixF, rowvecW, vecY, hor, E, T, S, lags, matrixWX, matrixX);
        if(E=='M'){
            materrors = log(1 + materrors);
            materrors.elem(arma::find_nonfinite(materrors)).fill(1e10);
        }
        materrors.row(0) = materrors.row(0) % horvec;
    }
    else{
        arma::mat materrorsfromfit(errorsfromfit.begin(), errorsfromfit.nrow(), errorsfromfit.ncol(), false);
        materrors = materrorsfromfit;
        if(E=='M'){
            materrors = log(1 + materrors);
        }
    }

    switch(E){
    case 'M':
        switch(CFtypeswitch(CFtype)){
        case 1:
            materrors.resize(matobs,hor);
            try{
                CFres = double(log(arma::prod(eig_sym(trans(materrors) * (materrors) / matobs))));
            }
            catch(const std::runtime_error){
                CFres = double(log(arma::det(arma::trans(materrors) * materrors / double(matobs))));
            }
            CFres = CFres + (2 / double(matobs)) * double(hor) * yactsum;
        break;
        case 2:
            for(int i=0; i<hor; i=i+1){
                CFres = CFres + arma::as_scalar(log(mean(pow(materrors.submat(0,i,obs-i-1,i),2))));
            }
            CFres = CFres + (2 / double(obs)) * double(hor) * yactsum;
        break;
        case 3:
            for(int i=0; i<hor; i=i+1){
                CFres = CFres + arma::as_scalar(mean(pow(materrors.submat(0,i,obs-i-1,i),2)));
            }
            CFres = exp(log(CFres) + (2 / double(obs)) * double(hor) * yactsum);
        break;
        case 4:
            CFres = arma::as_scalar(exp(log(mean(pow(materrors.submat(0,hor-1,obs-hor,hor-1),2))) + (2 / double(obs)) * yactsum));
        break;
        case 5:
            CFres = arma::as_scalar(exp(log(mean(abs(materrors))) + (2 / double(obs)) * yactsum));
        break;
        case 6:
            CFres = arma::as_scalar(exp(log(mean(sqrt(abs(materrors)))) + (2 / double(obs)) * yactsum));
        break;
        case 7:
            CFres = arma::as_scalar(exp(log(mean(pow(materrors,2))) + (2 / double(obs)) * yactsum));
        }
    break;
    case 'A':
        switch(CFtypeswitch(CFtype)){
        case 1:
            materrors.resize(matobs,hor);
            try{
                CFres = double(log(arma::prod(eig_sym(trans(materrors / normalize) * (materrors / normalize) / matobs))) + hor * log(pow(normalize,2)));
            }
            catch(const std::runtime_error){
                CFres = double(log(arma::det(arma::trans(materrors / normalize) * (materrors / normalize) / matobs)) + hor * log(pow(normalize,2)));
            }
        break;
        case 2:
            for(int i=0; i<hor; i=i+1){
                CFres = CFres + arma::as_scalar(log(mean(pow(materrors.submat(0,i,obs-i-1,i),2))));
            }
        break;
        case 3:
            for(int i=0; i<hor; i=i+1){
                CFres = CFres + arma::as_scalar(mean(pow(materrors.submat(0,i,obs-i-1,i),2)));
            }
        break;
        case 4:
            CFres = arma::as_scalar(mean(pow(materrors.submat(0,hor-1,obs-hor,hor-1),2)));
        break;
        case 5:
            CFres = arma::as_scalar(mean(abs(materrors)));
        break;
        case 6:
            CFres = arma::as_scalar(mean(sqrt(abs(materrors))));
        break;
        case 7:
            CFres = arma::as_scalar(mean(pow(materrors,2)));
        }
    }
    return CFres;
}

/* # Wrapper for optimiser */
// [[Rcpp::export]]
RcppExport SEXP optimizerwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg1,
                              SEXP h, SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype,
                              SEXP multisteps, SEXP CFt, SEXP normalizer,
                              SEXP matwex, SEXP matxtreg, SEXP matv, SEXP matF2, SEXP vecg2) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());

    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);

    NumericMatrix vw(matw);
    arma::rowvec rowvecW(vw.begin(), vw.ncol(), false);

    NumericMatrix vyt(yt);
    arma::vec vecY(vyt.begin(), vyt.nrow(), false);

    NumericMatrix vg(vecg1);
    arma::vec vecG(vg.begin(), vg.nrow(), false);

    unsigned int hor = as<unsigned int>(h);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    bool multi = as<bool>(multisteps);

    std::string CFtype = as<std::string>(CFt);

    double normalize = as<double>(normalizer);

    NumericMatrix mwex(matwex);
    arma::mat matrixWX(mwex.begin(), mwex.nrow(), mwex.ncol(), false);

    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    NumericMatrix vv(matv);
    arma::mat matrixweightsX(vv.begin(), vv.nrow(), vv.ncol(), false);

    NumericMatrix mF2(matF2);
    arma::mat matrixFX(mF2.begin(), mF2.nrow(), mF2.ncol(), false);

    NumericMatrix vg2(vecg2);
    arma::vec vecGX(vg2.begin(), vg2.nrow(), false);

    return wrap(optimizer(matrixxt,matrixF,rowvecW,vecY,vecG,
                          hor,lags,E,T,S,multi,CFtype,normalize,
                          matrixWX, matrixX, matrixweightsX, matrixFX, vecGX));
}

/* # Function is used in cases when the persistence vector needs to be estimated.
# If bounds are violated, it returns a state vector with zeroes. */
// [[Rcpp::export]]
RcppExport SEXP costfunc(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg1,
                              SEXP h, SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype,
                              SEXP multisteps, SEXP CFt, SEXP normalizer,
                              SEXP matwex, SEXP matxtreg, SEXP matv, SEXP matF2, SEXP vecg2,
                              SEXP bounds, SEXP phi, SEXP Theta) {
/* Function is needed to implement admissible constrains on smoothing parameters */
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());

    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);

    NumericMatrix vw(matw);
    arma::rowvec rowvecW(vw.begin(), vw.ncol(), false);

    NumericMatrix vyt(yt);
    arma::vec vecY(vyt.begin(), vyt.nrow(), false);

    NumericMatrix vg(vecg1);
    arma::vec vecG(vg.begin(), vg.nrow(), false);

    int hor = as<int>(h);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    bool multi = as<bool>(multisteps);

    std::string CFtype = as<std::string>(CFt);

    double normalize = as<double>(normalizer);

    NumericMatrix mwex(matwex);
    arma::mat matrixWX(mwex.begin(), mwex.nrow(), mwex.ncol(), false);

    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    NumericMatrix vv(matv);
    arma::mat matrixweightsX(vv.begin(), vv.nrow(), vv.ncol(), false);

    NumericMatrix mF2(matF2);
    arma::mat matrixFX(mF2.begin(), mF2.nrow(), mF2.ncol(), false);

    NumericMatrix vg2(vecg2);
    arma::vec vecGX(vg2.begin(), vg2.nrow(), false);

    char boundtype = as<char>(bounds);
    double phivalue = as<double>(phi);
    double theta = as<double>(Theta);

    unsigned int maxlag = max(lags);

    if(boundtype=='u'){
// alpha in (0,1)
        if((vecG(0)>1) || (vecG(0)<0)){
            vecG.zeros();
            matrixxt.zeros();
        }
        if(T!='N'){
// beta in (0,alpha)
            if((vecG(1)>vecG(0)) || (vecG(1)<0)){
                vecG.zeros();
                matrixxt.zeros();
            }
            if(S!='N'){
// gamma in (0,1-alpha)
                if((vecG(2)>(1-vecG(0))) || (vecG(2)<0)){
                    vecG.zeros();
                    matrixxt.zeros();
                }
            }
        }
        if(S!='N'){
// gamma in (0,1-alpha)
            if((vecG(1)>(1-vecG(0))) || (vecG(1)<0)){
                vecG.zeros();
                matrixxt.zeros();
            }
        }
    }
    else{
        if(S=='N'){
// alpha restrictions with no seasonality
            if((vecG(0)>1+1/phivalue) || (vecG(0)<1-1/phivalue)){
                vecG.zeros();
                matrixxt.zeros();
            }
            if(T!='N'){
// beta restrictions with no seasonality
                if((vecG(1)>(1+phivalue)*(2-vecG(0))) || (vecG(1)<vecG(0)*(phivalue-1))){
                    vecG.zeros();
                    matrixxt.zeros();
                }
            }
        }
        else{
            if(T=='N'){
// alpha restrictions with no trend
                if((vecG(0)>2-vecG(1)) ||  (vecG(0)<(-2/(maxlag-1)))){
                    vecG.zeros();
                    matrixxt.zeros();
                }
// gamma restrictions with no trend
                if((vecG(1)>2-vecG(0)) || (vecG(1)<std::max(-maxlag*vecG(0),0.0))){
                    vecG.zeros();
                    matrixxt.zeros();
                }
            }
            else{
                double Bvalue = phivalue*(4-3*vecG(2))+vecG(2)*(1-phivalue) / maxlag;
                double Cvalue = sqrt(pow(Bvalue,2)-8*(pow(phivalue,2)*pow((1-vecG(2)),2)+2*(phivalue-1)*(1-vecG(2))-1)+8*pow(vecG(2),2)*(1-phivalue) / maxlag);
                double Dvalue = (phivalue*(1-vecG(0))+1)*(1-cos(theta))-vecG(2)*((1+phivalue)*(1-cos(theta)-cos(maxlag*theta))+cos((maxlag-1)*theta)+phivalue*cos((maxlag+1)*theta))/(2*(1+cos(theta))*(1-cos(maxlag*theta)));
// alpha restriction
                if((vecG(0)>((Bvalue + Cvalue)/(4*phivalue))) || (vecG(0)<(1-1/phivalue-vecG(2)*(1-maxlag+phivalue*(1+maxlag))/(2*phivalue*maxlag)))){
                    vecG.zeros();
                    matrixxt.zeros();
                }
// beta restriction
                if((vecG(1)>(Dvalue+vecG(0)*(phivalue-1))) || (vecG(1)<(phivalue-1)*(vecG(2)/maxlag+vecG(0)))){
                    vecG.zeros();
                    matrixxt.zeros();
                }
// gamma restriction
                if((vecG(2)>(1+1/phivalue-vecG(0))) || (vecG(2)<(std::max(1-1/phivalue-vecG(0),0.0)))){
                    vecG.zeros();
                    matrixxt.zeros();
                }
            }
        }
    }

    return wrap(optimizer(matrixxt,matrixF,rowvecW,vecY,vecG,
                          hor,lags,E,T,S,multi,CFtype,normalize,
                          matrixWX, matrixX, matrixweightsX, matrixFX, vecGX));
}

/*
# autoets - function estimates all the necessary ETS models and returns the one with the smallest chosen IC.
*/

// ##### Script for sim.ets function
List simulateETS(arma::mat matrixxt, arma::mat matrixerrors, arma::mat matrixot,
                 arma::mat matrixF, arma::mat rowvecW, arma::mat vecG,
                 unsigned int obs, unsigned int nseries,
                 char E, char T, char S, arma::uvec lags) {
    arma::mat matY(obs, nseries);


    return List::create(Named("matxt") = matrixxt, Named("y") = matY);
}

/* # Wrapper for simulateets */
// [[Rcpp::export]]
RcppExport SEXP simulateETSwrap(SEXP matxt, SEXP errors, SEXP ot, SEXP matF, SEXP matw, SEXP vecg1,
                                SEXP Etype, SEXP Ttype, SEXP Stype, SEXP modellags) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());
    NumericMatrix merrors(errors);
    arma::mat matrixerrors(merrors.begin(), merrors.nrow(), merrors.ncol(), false);
    NumericMatrix mot(ot);
    arma::mat matrixot(mot.begin(), mot.nrow(), mot.ncol(), false);
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::mat rowvecW(vw.begin(), vw.nrow(), vw.ncol(), false);
    NumericMatrix vg(vecg1);
    arma::vec vecG(vg.begin(), vg.nrow(), vg.ncol(), false);
    unsigned int obs = merrors.nrow();
    unsigned int nseries = merrors.ncol();
    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);
    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    return wrap(simulateETS(matrixxt, matrixerrors, matrixot, matrixF, rowvecW, vecG,
                            obs, nseries, E, T, S, lags));
}
