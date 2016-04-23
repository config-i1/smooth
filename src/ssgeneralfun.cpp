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
        if((yact==0) & (yfit==0)){
            return 0;
        }
        else if((yact!=0) & (yfit==0)){
            return R_PosInf;
        }
        else{
            return (yact - yfit) / yfit;
        }
    }
}

/* # Function is needed to estimate the correct error for ETS when multisteps model selection with r(matvt) is sorted out. */
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
double wvalue(arma::vec matrixVt, arma::rowvec rowvecW, char T, char S){
// matrixVt is a vector here!
    double yfit = 0;

    switch(S){
// ZZN
    case 'N':
        switch(T){
        case 'N':
        case 'A':
            yfit = as_scalar(rowvecW * matrixVt);
        break;
        case 'M':
            yfit = as_scalar(exp(rowvecW * log(matrixVt)));
        break;
        }
    break;
// ZZA
    case 'A':
        switch(T){
        case 'N':
        case 'A':
            yfit = as_scalar(rowvecW * matrixVt);
        break;
        case 'M':
            yfit = as_scalar(exp(rowvecW.cols(0,1) * log(matrixVt.rows(0,1)))) + matrixVt(2);
        break;
        }
    break;
// ZZM
    case 'M':
        switch(T){
        case 'N':
        case 'M':
            yfit = as_scalar(exp(rowvecW * log(matrixVt)));
        break;
        case 'A':
            yfit = as_scalar(rowvecW.cols(0,1) * matrixVt.rows(0,1)) * matrixVt(2);
        break;
        }
    break;
    }

    return yfit;
}

/* # Function returns value of r() -- additive or multiplicative error -- used in the error term of measurement equation.
     This is mainly needed by sim.ets */
double rvalue(arma::vec matrixVt, arma::rowvec rowvecW, char E, char T, char S){

    switch(E){
// MZZ
    case 'M':
        return wvalue(matrixVt, rowvecW, T, S);
    break;
// AZZ
    case 'A':
    default:
        return 1.0;
    }
}

/* # Function returns value of f() -- new states without the update -- used in the transition equation */
arma::vec fvalue(arma::vec matrixVt, arma::mat matrixF, char T, char S){
    arma::vec matrixVtnew = matrixVt;

    switch(S){
// ZZN
    case 'N':
        switch(T){
        case 'N':
        case 'A':
            matrixVtnew = matrixF * matrixVt;
        break;
        case 'M':
            matrixVtnew = exp(matrixF * log(matrixVt));
        break;
        }
    break;
// ZZA
    case 'A':
        switch(T){
        case 'N':
        case 'A':
            matrixVtnew = matrixF * matrixVt;
        break;
        case 'M':
            matrixVtnew.rows(0,1) = exp(matrixF.submat(0,0,1,1) * log(matrixVt.rows(0,1)));
            matrixVtnew(2) = matrixVt(2);
        break;
        }
    break;
// ZZM
    case 'M':
        switch(T){
        case 'N':
        case 'M':
            matrixVtnew = exp(matrixF * log(matrixVt));
        break;
        case 'A':
            matrixVtnew = matrixF * matrixVt;
        break;
        }
    break;
    }

    return matrixVtnew;
}

/* # Function returns value of g() -- the update of states -- used in components estimation for the persistence */
arma::vec gvalue(arma::vec matrixVt, arma::mat matrixF, arma::mat rowvecW, char E, char T, char S){
    arma::vec g(matrixVt.n_rows, arma::fill::ones);

// AZZ
    switch(E){
    case 'A':
// ANZ
        switch(T){
        case 'N':
            switch(S){
            case 'M':
                g(0) = 1 / matrixVt(1);
                g(1) = 1 / matrixVt(0);
            break;
            }
        break;
// AAZ
        case 'A':
            switch(S){
            case 'M':
                g.rows(0,1) = g.rows(0,1) / matrixVt(2);
                g(2) = 1 / as_scalar(rowvecW.cols(0,1) * matrixVt.rows(0,1));
            break;
            }
        break;
// AMZ
        case 'M':
            switch(S){
            case 'N':
            case 'A':
                g(1) = g(1) / matrixVt(0);
            break;
            case 'M':
                g(0) = g(0) / matrixVt(2);
                g(1) = g(1) / (matrixVt(0) * matrixVt(2));
                g(2) = g(2) / as_scalar(exp(rowvecW.cols(0,1) * log(matrixVt.rows(0,1))));
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
                g(0) = matrixVt(0);
            break;
            case 'A':
                g.rows(0,1).fill(matrixVt(0) + matrixVt(1));
            break;
            case 'M':
                g = matrixVt;
            break;
            }
        break;
// MAZ
        case 'A':
            switch(S){
            case 'N':
            case 'A':
                g.fill(as_scalar(rowvecW * matrixVt));
            break;
            case 'M':
                g.rows(0,1).fill(as_scalar(rowvecW.cols(0,1) * matrixVt.rows(0,1)));
                g(2) = matrixVt(2);
            break;
            }
        break;
// MMZ
        case 'M':
            switch(S){
            case 'N':
                g(0) = as_scalar(exp(rowvecW * log(matrixVt)));
                g(1) = pow(matrixVt(1),rowvecW(1));
            break;
            case 'A':
                g.rows(0,2).fill(as_scalar(exp(rowvecW.cols(0,1) * log(matrixVt.rows(0,1))) + matrixVt(2)));
                g(1) = g(0) / matrixVt(0);
            break;
            case 'M':
                g = exp(matrixF * log(matrixVt));
            break;
            }
        break;
        }
    break;
    }

    return g;
}

/* # Function is needed for the renormalisation of seasonal components. It should be done seasonal-wise.*/
arma::mat normaliser(arma::mat Vt, int obsall, unsigned int maxlag, char S, char T){

    unsigned int ncomponents = Vt.n_cols;
    arma::vec meanseason(maxlag, arma::fill::zeros);

    switch(S){
    case 'A':
        meanseason.fill(arma::as_scalar(mean(Vt.col(ncomponents-1))));
        Vt.col(ncomponents-1) = Vt.col(ncomponents-1) - meanseason;
        switch(T){
        case 'N':
        case 'A':
            Vt.col(0) = Vt.col(0) + meanseason;
        break;
        case 'M':
            Vt.col(0) = Vt.col(0) + meanseason / Vt.col(1);
        break;
        }
    break;
    case 'M':
        meanseason.fill(arma::as_scalar(exp(mean(log(Vt.col(ncomponents-1))))));
        Vt.col(ncomponents-1) = Vt.col(ncomponents-1) / meanseason;
        switch(T){
        case 'N':
        case 'M':
            Vt.col(0) = Vt.col(0) / meanseason;
        break;
        case 'A':
            Vt.col(0) = Vt.col(0) % meanseason;
            Vt.col(1) = Vt.col(1) % meanseason;
        break;
        }
    break;
    }

    return(Vt);
}

/* # initparams - function that initialises the basic parameters of ETS */
// [[Rcpp::export]]
RcppExport SEXP initparams(SEXP Ttype, SEXP Stype, SEXP datafreq, SEXP obsR, SEXP yt,
                           SEXP damped, SEXP phi, SEXP smoothingparameters, SEXP initialstates, SEXP seasonalcoefs){

    char T = as<char>(Ttype);
    char S = as<char>(Stype);
    int freq = as<int>(datafreq);
    int obs = as<int>(obsR);
    NumericMatrix yt_n(yt);
    arma::mat vecYt(yt_n.begin(), yt_n.nrow(), yt_n.ncol(), false);
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

    arma::mat matrixVt(obs+maxlag, ncomponents, arma::fill::ones);
    arma::vec vecG(ncomponents, arma::fill::zeros);
    bool estimphi = TRUE;

// # Define the initial states for level and trend components
    switch(T){
    case 'N':
        matrixVt.submat(0,0,maxlag-1,0).each_row() = initial.submat(0,2,0,2);
    break;
    case 'A':
        matrixVt.submat(0,0,maxlag-1,1).each_row() = initial.submat(0,0,0,1);
    break;
// # The initial matrix is filled with ones, that is why we don't need to fill in initial trend
    case 'M':
        matrixVt.submat(0,0,maxlag-1,1).each_row() = initial.submat(0,2,0,3);
    break;
    }

/* # Define the initial states for seasonal component */
    switch(S){
    case 'A':
        matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = seascoef.col(0);
    break;
    case 'M':
        matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = seascoef.col(1);
    break;
    }

//    matrixVt.resize(obs+maxlag, ncomponents);

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
                             Named("matvt") = matrixVt, Named("vecg") = vecG, Named("estimate.phi") = estimphi,
                             Named("phi") = phivalue));
}

/*
# etsmatrices - function that returns matF and matw.
# Needs to be stand alone to change the damping parameter during the estimation.
# Cvalues includes persistence, phi, initials, intials for seasons, matrixAt coeffs.
*/
// [[Rcpp::export]]
RcppExport SEXP etsmatrices(SEXP matvt, SEXP vecg, SEXP phi, SEXP Cvalues, SEXP ncomponentsR,
                            SEXP modellags, SEXP Ttype, SEXP Stype, SEXP nexovars, SEXP matat,
                            SEXP estimpersistence, SEXP estimphi, SEXP estiminit, SEXP estiminitseason, SEXP estimxreg,
                            SEXP matFX, SEXP vecgX, SEXP gowild, SEXP estimFX, SEXP estimgX){

    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix vecg_n(vecg);
    arma::vec vecG(vecg_n.begin(), vecg_n.nrow(), false);

    double phivalue = as<double>(phi);

    NumericMatrix Cv(Cvalues);
    arma::rowvec C(Cv.begin(), Cv.ncol(), false);

    int ncomponents = as<int>(ncomponentsR);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);
    int maxlag = max(lags);

    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    int nexo = as<int>(nexovars);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    bool estimatepersistence = as<bool>(estimpersistence);
    bool estimatephi = as<bool>(estimphi);
    bool estimateinitial = as<bool>(estiminit);
    bool estimateinitialseason = as<bool>(estiminitseason);
    bool estimatexreg = as<bool>(estimxreg);

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol());

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow());

    bool wild = as<bool>(gowild);
    bool estimateFX = as<bool>(estimFX);
    bool estimategX = as<bool>(estimgX);

    arma::mat matrixF(1,1,arma::fill::ones);
    arma::mat rowvecW(1,1,arma::fill::ones);

    int currentelement = 0;

    if(estimatepersistence==TRUE){
        vecG = C.cols(currentelement,currentelement + ncomponents-1).t();
        currentelement = currentelement + ncomponents;
    }

    if(estimatephi==TRUE){
        phivalue = as_scalar(C.col(currentelement));
        currentelement = currentelement + 1;
    }

    if(estimateinitial==TRUE){
        matrixVt.col(0).fill(as_scalar(C.col(currentelement).t()));
        currentelement = currentelement + 1;
        if(T!='N'){
            matrixVt.col(1).fill(as_scalar(C.col(currentelement).t()));
            currentelement = currentelement + 1;
        }
    }

    if(S!='N'){
        if(estimateinitialseason==TRUE){
            matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = C.cols(currentelement, currentelement + maxlag - 1).t();
            currentelement = currentelement + maxlag;
/* # Normalise the initial seasons */
            if(S=='A'){
                matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) -
                            as_scalar(mean(matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1)));
            }
            else{
                matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1) = exp(log(matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1)) -
                            as_scalar(mean(log(matrixVt.submat(0,ncomponents-1,maxlag-1,ncomponents-1)))));
            }
        }
    }

    if(estimatexreg==TRUE){
        matrixAt.each_row() = C.cols(currentelement,currentelement + nexo - 1);
        currentelement = currentelement + nexo;
        if(wild==TRUE){
            if(estimateFX==TRUE){
                for(int i=0; i < nexo; i = i+1){
                    matrixFX.row(i) = C.cols(currentelement, currentelement + nexo - 1);
                    currentelement = currentelement + nexo;
                }
            }

            if(estimategX==TRUE){
                vecGX = C.cols(currentelement, currentelement + nexo - 1).t();
                currentelement = currentelement + nexo;
            }
        }
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
                             Named("phi") = phivalue, Named("matvt") = matrixVt, Named("matat") = matrixAt,
                             Named("matFX") = matrixFX, Named("vecgX") = vecGX));
}

List fitter(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecYt, arma::vec vecG,
             arma::uvec lags, char E, char T, char S,
             arma::mat matrixXt, arma::mat matrixAt, arma::mat matrixFX, arma::vec vecGX, arma::vec vecOt) {
    /* # matrixVt should have a length of obs + maxlag.
    * # rowvecW should have 1 row.
    * # matgt should be a vector
    * # lags is a vector of lags
    * # matrixXt is the matrix with the exogenous variables
    * # matrixAt is the matrix with the parameters for the exogenous
    */

    int obs = vecYt.n_rows;
    int obsall = matrixVt.n_rows;
    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);

    lags = maxlag - lags;

    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + obsall * i;
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);

    arma::vec matyfit(obs, arma::fill::zeros);
    arma::vec materrors(obs, arma::fill::zeros);
    arma::rowvec bufferforat(vecGX.n_rows);

    for (int i=maxlag; i<obsall; i=i+1) {

        lagrows = lags - maxlag + i;

/* # Measurement equation and the error term */
        matyfit.row(i-maxlag) = vecOt(i-maxlag) * (wvalue(matrixVt(lagrows), rowvecW, T, S) +
                                       matrixXt.row(i-maxlag) * arma::trans(matrixAt.row(i-maxlag)));
        materrors(i-maxlag) = errorf(vecYt(i-maxlag), matyfit(i-maxlag), E);

/* # Transition equation */
        matrixVt.row(i) = arma::trans(fvalue(matrixVt(lagrows), matrixF, T, S) +
                                      gvalue(matrixVt(lagrows), matrixF, rowvecW, E, T, S) % vecG * materrors(i-maxlag));

/* Failsafe for cases when unreasonable value for state vector was produced */
        if(!matrixVt.row(i).is_finite()){
            matrixVt.row(i) = trans(matrixVt(lagrows));
        }
        if(((T=='M') | (S=='M')) & (any(matrixVt.row(i) < 0))){
            matrixVt.row(i) = trans(matrixVt(lagrows));
        }

/* Renormalise components if the seasonal model is chosen */
        if(S!='N'){
            if(double(i+1) / double(maxlag) == double((i+1) / maxlag)){
                matrixVt.rows(i-maxlag+1,i) = normaliser(matrixVt.rows(i-maxlag+1,i), obsall, maxlag, S, T);
            }
        }

/* # Transition equation for xreg */
        bufferforat = arma::trans(vecGX / arma::trans(matrixXt.row(i-maxlag)) * materrors(i-maxlag));
        bufferforat.elem(find_nonfinite(bufferforat)).fill(0);
        matrixAt.row(i) = matrixAt.row(i-1) * matrixFX + bufferforat;
    }

    return List::create(Named("matvt") = matrixVt, Named("yfit") = matyfit,
                        Named("errors") = materrors, Named("matat") = matrixAt);
}

/* # Wrapper for fitter2 */
// [[Rcpp::export]]
RcppExport SEXP fitterwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg,
                            SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype,
                            SEXP matxt, SEXP matat, SEXP matFX, SEXP vecgX, SEXP ot) {
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

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow(), false);

    NumericVector ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.size(), false);

    return wrap(fitter(matrixVt, matrixF, rowvecW, vecYt, vecG, lags, E, T, S,
                matrixXt, matrixAt, matrixFX, vecGX, vecOt));
}

/* # Function fills in the values of the provided matrixAt using the transition matrix. Needed for forecast of coefficients of xreg. */
List statetail(arma::mat matrixVt, arma::mat matrixF, arma::mat matrixAt, arma::mat matrixFX,
               arma::uvec lags, char T, char S){

    int obsall = matrixVt.n_rows;
    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);

    lags = maxlag - lags;

    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + obsall * i;
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);

    for (int i=maxlag; i<obsall; i=i+1) {
        lagrows = lags - maxlag + i;
        matrixVt.row(i) = arma::trans(fvalue(matrixVt(lagrows), matrixF, T, S));
      }

    for(int i=0; i<(matrixAt.n_rows-1); i=i+1){
        matrixAt.row(i+1) = matrixAt.row(i) * matrixFX;
    }

    return(List::create(Named("matvt") = matrixVt, Named("matat") = matrixAt));
}

/* # Wrapper for ssstatetail */
// [[Rcpp::export]]
RcppExport SEXP statetailwrap(SEXP matvt, SEXP matF, SEXP matat, SEXP matFX,
                              SEXP modellags, SEXP Ttype, SEXP Stype){
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    return(wrap(statetail(matrixVt, matrixF, matrixAt, matrixFX, lags, T, S)));
}

/* # Function produces the point forecasts for the specified model */
arma::mat forecaster(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW,
                     unsigned int hor, char T, char S, arma::uvec lags,
                     arma::mat matrixXt, arma::mat matrixAt, arma::mat matrixFX) {
    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);
    unsigned int hh = hor + maxlag;

    arma::uvec lagrows(lagslength, arma::fill::zeros);
    arma::vec matyfor(hor, arma::fill::zeros);
    arma::mat matrixVtnew(hh, matrixVt.n_cols, arma::fill::zeros);
    arma::mat matrixAtnew(hh, matrixAt.n_cols, arma::fill::zeros);

    lags = maxlag - lags;
    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + hh * i;
    }

    matrixVtnew.submat(0,0,maxlag-1,matrixVtnew.n_cols-1) = matrixVt.submat(0,0,maxlag-1,matrixVtnew.n_cols-1);
    matrixAtnew.submat(0,0,maxlag-1,matrixAtnew.n_cols-1) = matrixAtnew.submat(0,0,maxlag-1,matrixAtnew.n_cols-1);

/* # Fill in the new xt matrix using F. Do the forecasts. */
    for (int i=maxlag; i<(hor+maxlag); i=i+1) {
        lagrows = lags - maxlag + i;
        matrixVtnew.row(i) = arma::trans(fvalue(matrixVtnew(lagrows), matrixF, T, S));
        matrixAtnew.row(i) = matrixAtnew.row(i-1) * matrixFX;

        matyfor.row(i-maxlag) = (wvalue(matrixVtnew(lagrows), rowvecW, T, S) + matrixXt.row(i-maxlag) * arma::trans(matrixAt.row(i-maxlag)));
    }

    return matyfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP forecasterwrap(SEXP matvt, SEXP matF, SEXP matw,
                               SEXP h, SEXP Ttype, SEXP Stype, SEXP modellags,
                               SEXP matxt, SEXP matat, SEXP matFX){
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol(), false);

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::mat rowvecW(matw_n.begin(), matw_n.nrow(), matw_n.ncol(), false);

    unsigned int hor = as<int>(h);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    return wrap(forecaster(matrixVt, matrixF, rowvecW, hor, T, S, lags, matrixXt, matrixAt, matrixFX));
}

/* # Function produces matrix of errors based on multisteps forecast */
arma::mat errorer(arma::mat matrixVt, arma::mat matrixF, arma::mat rowvecW, arma::mat vecYt,
                  int hor, char E, char T, char S, arma::uvec lags,
                  arma::mat matrixXt, arma::mat matrixAt, arma::mat matrixFX, arma::vec vecOt){
    int obs = vecYt.n_rows;
    int hh = 0;
    arma::mat materrors(obs+hor-1, hor, arma::fill::zeros);
    unsigned int maxlag = max(lags);

//    materrors.fill(NA_REAL);

    for(int i = 0; i < obs; i=i+1){
        hh = std::min(hor, obs-i);
        materrors.submat(hor-1+i, 0, hor-1+i, hh-1) = trans(vecOt.rows(i, i+hh-1) % errorvf(vecYt.rows(i, i+hh-1),
            forecaster(matrixVt.rows(i,i+maxlag-1), matrixF, rowvecW, hh, T, S, lags, matrixXt.rows(i, i+hh-1),
                matrixAt.rows(i, i+hh-1), matrixFX), E));
    }

// Fix for GV in order to perform better in the sides of the series
    for(unsigned int i=0; i<(hor-1); i=i+1){
        materrors.submat((hor-2)-(i),i+1,(hor-2)-(i),hor-1) = materrors.submat(hor-1,0,hor-1,hor-i-2);
    }

    return materrors;
}

/* # Wrapper for errorer */
// [[Rcpp::export]]
RcppExport SEXP errorerwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt,
                            SEXP h, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP modellags,
                            SEXP matxt, SEXP matat, SEXP matFX, SEXP ot){
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol(), false);

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

    NumericMatrix yt_n(yt);
    arma::vec vecYt(yt_n.begin(), yt_n.nrow(), false);

    int hor = as<int>(h);
    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericVector ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.size(), false);

    return wrap(errorer(matrixVt, matrixF, rowvecW, vecYt,
                        hor, E, T, S, lags,
                        matrixXt, matrixAt, matrixFX, vecOt));
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
double optimizer(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecYt, arma::vec vecG,
                 unsigned int hor, arma::uvec lags, char E, char T, char S, bool multi, std::string CFtype, double normalize,
                 arma::mat  matrixXt, arma::mat matrixAt, arma::mat matrixFX, arma::vec vecGX, arma::vec vecOt){
// # Make decomposition functions shut up!
    std::ostream nullstream(0);
    arma::set_stream_err2(nullstream);

    arma::uvec nonzeroes = find(vecOt>0);
    int obs = nonzeroes.n_rows;
    double CFres = 0;
    int matobs = obs + hor - 1;
// yactsum is needed for multiplicative error models
    double yactsum = arma::as_scalar(sum(log(vecYt.elem(nonzeroes))));

    List fitting = fitter(matrixVt, matrixF, rowvecW, vecYt, vecG, lags, E, T, S,
                          matrixXt, matrixAt, matrixFX, vecGX, vecOt);
    NumericMatrix mxtfromfit = as<NumericMatrix>(fitting["matvt"]);
    matrixVt = as<arma::mat>(mxtfromfit);
    NumericMatrix errorsfromfit = as<NumericMatrix>(fitting["errors"]);
    NumericMatrix matrixAtfromfit = as<NumericMatrix>(fitting["matat"]);
    matrixAt = as<arma::mat>(matrixAtfromfit);

    arma::mat materrors;
    arma::rowvec horvec(hor);

    if(multi==true){
        for(unsigned int i=0; i<hor; i=i+1){
            horvec(i) = hor - i;
        }
        materrors = errorer(matrixVt, matrixF, rowvecW, vecYt, hor, E, T, S, lags, matrixXt, matrixAt, matrixFX, vecOt);
        if(E=='M'){
            materrors = log(1 + materrors);
            materrors.elem(arma::find_nonfinite(materrors)).fill(1e10);
// This correction is needed in order to take the correct number of observations in the error matrix
            yactsum = yactsum / obs * matobs;
        }
    }
    else{
        arma::mat materrorsfromfit(errorsfromfit.begin(), errorsfromfit.nrow(), errorsfromfit.ncol(), false);
        materrors = materrorsfromfit;
        materrors = materrors.elem(nonzeroes);
        if(E=='M'){
            materrors = log(1 + materrors);
        }
    }

    switch(E){
    case 'M':
        switch(CFtypeswitch(CFtype)){
        case 1:
            try{
                CFres = double(log(arma::prod(eig_sym(trans(materrors) * (materrors) / matobs))));
            }
            catch(const std::runtime_error){
                CFres = double(log(arma::det(arma::trans(materrors) * materrors / double(matobs))));
            }
            CFres = CFres + (2 / double(matobs)) * double(hor) * yactsum;
        break;
        case 2:
            CFres = arma::as_scalar(sum(log(sum(pow(materrors,2)) / double(matobs)), 1))
                    + (2 / double(obs)) * double(hor) * yactsum;
        break;
        case 3:
            CFres = arma::as_scalar(exp(log(sum(sum(pow(materrors,2)) / double(matobs), 1))
                        + (2 / double(obs)) * double(hor) * yactsum));
        break;
        case 4:
            CFres = arma::as_scalar(exp(log(sum(pow(materrors.col(hor-1),2)) / double(matobs))
                                        + (2 / double(obs)) * yactsum));
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
            CFres = arma::as_scalar(sum(log(sum(pow(materrors,2)) / double(matobs)), 1));
        break;
        case 3:
            CFres = arma::as_scalar(sum(sum(pow(materrors,2)) / double(matobs), 1));
        break;
        case 4:
            CFres = arma::as_scalar(sum(pow(materrors.col(hor-1),2)) / double(matobs));
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
RcppExport SEXP optimizerwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg,
                              SEXP h, SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype,
                              SEXP multisteps, SEXP CFt, SEXP normalizer,
                              SEXP matxt, SEXP matat, SEXP matFX, SEXP vecgX, SEXP ot) {
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

    unsigned int hor = as<unsigned int>(h);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    bool multi = as<bool>(multisteps);

    std::string CFtype = as<std::string>(CFt);

    double normalize = as<double>(normalizer);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow(), false);

    NumericVector ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.size(), false);

    return wrap(optimizer(matrixVt,matrixF,rowvecW,vecYt,vecG,
                          hor,lags,E,T,S,multi,CFtype,normalize,
                          matrixXt, matrixAt, matrixFX, vecGX, vecOt));
}

/* # Function is used in cases when the persistence vector needs to be estimated.
# If bounds are violated, it returns variance of yt. */
// [[Rcpp::export]]
RcppExport SEXP costfunc(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg,
                              SEXP h, SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype,
                              SEXP multisteps, SEXP CFt, SEXP normalizer,
                              SEXP matxt, SEXP matat, SEXP matFX, SEXP vecgX, SEXP ot,
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

    int hor = as<int>(h);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    bool multi = as<bool>(multisteps);

    std::string CFtype = as<std::string>(CFt);

    double normalize = as<double>(normalizer);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow(), false);

    NumericVector ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.size(), false);

    char boundtype = as<char>(bounds);

// Values needed for eigenvalues calculation
    arma::cx_vec eigval;
    arma::mat matrixD = matrixF;

    if(boundtype=='u'){
// alpha in (0,1)
        if((vecG(0)>1) || (vecG(0)<0)){
            vecG.zeros();
            matrixVt.zeros();
        }
        if(T!='N'){
// beta in (0,alpha)
            if((vecG(1)>vecG(0)) || (vecG(1)<0)){
                vecG.zeros();
                matrixVt.zeros();
            }
            if(S!='N'){
// gamma in (0,1-alpha)
                if((vecG(2)>(1-vecG(0))) || (vecG(2)<0)){
                    vecG.zeros();
                    matrixVt.zeros();
                }
            }
        }
        if(S!='N'){
// gamma in (0,1-alpha)
            if((vecG(1)>(1-vecG(0))) || (vecG(1)<0)){
                vecG.zeros();
                matrixVt.zeros();
            }
        }
    }
    else if(boundtype=='a'){
        if(arma::eig_gen(eigval, matrixF - vecG * rowvecW)){
            if(max(abs(eigval))>1){
                return wrap(max(abs(eigval))*1E+100);
            }
        }
        else{
            return wrap(1E+300);
        }
    }

    return wrap(optimizer(matrixVt, matrixF, rowvecW, vecYt, vecG,
                          hor, lags, E, T, S, multi, CFtype, normalize,
                          matrixXt, matrixAt, matrixFX, vecGX, vecOt));
}

/*
# autoets - function estimates all the necessary ETS models and returns the one with the smallest chosen IC.
*/

// ##### Script for sim.ets function
List simulateETS(arma::cube arrayVt, arma::mat matrixerrors, arma::mat matrixot,
                 arma::mat matrixF, arma::rowvec rowvecW, arma::mat matrixG,
                 unsigned int obs, unsigned int nseries,
                 char E, char T, char S, arma::uvec lags) {

    arma::mat matY(obs, nseries);

    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);
    int obsall = obs + maxlag;

    lags = maxlag - lags;

    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + obsall * i;
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);
    arma::mat matrixVt(obsall, lagslength, arma::fill::zeros);

    for(unsigned int i=0; i<nseries; i=i+1){
        matrixVt = arrayVt.slice(i);
        for (int j=maxlag; j<obsall; j=j+1) {

            lagrows = lags - maxlag + j;
/* # Measurement equation and the error term */
            matY(j-maxlag,i) = matrixot(j-maxlag,i) * (wvalue(matrixVt(lagrows), rowvecW, T, S) +
                                 rvalue(matrixVt(lagrows), rowvecW, E, T, S) * matrixerrors(j-maxlag,i));
/* # Transition equation */
            matrixVt.row(j) = arma::trans(fvalue(matrixVt(lagrows), matrixF, T, S) +
                                          gvalue(matrixVt(lagrows), matrixF, rowvecW, E, T, S) % matrixG.col(i) * matrixerrors(j-maxlag,i));
/* Failsafe for cases when unreasonable value for state vector was produced */
            if(!matrixVt.row(j).is_finite()){
                matrixVt.row(j) = trans(matrixVt(lagrows));
            }
            if(((E=='M') | (T=='M') | (S=='M')) & (any(matrixVt.row(j) < 0))){
                matrixVt.row(j) = trans(matrixVt(lagrows));
            }
        }
        arrayVt.slice(i) = matrixVt;
    }

    return List::create(Named("arrvt") = arrayVt, Named("matyt") = matY);
}

/* # Wrapper for simulateets */
// [[Rcpp::export]]
RcppExport SEXP simulateETSwrap(SEXP arrvt, SEXP materrors, SEXP matot, SEXP matF, SEXP matw, SEXP matg,
                                SEXP Etype, SEXP Ttype, SEXP Stype, SEXP modellags) {

// ### arrvt should contain array of obs x ncomponents x nseries elements.
    NumericVector arrvt_n(arrvt);
    IntegerVector arrvt_dim = arrvt_n.attr("dim");
    arma::cube arrayVt(arrvt_n.begin(),arrvt_dim[0], arrvt_dim[1], arrvt_dim[2], false);

    NumericMatrix materrors_n(materrors);
    arma::mat matrixerrors(materrors_n.begin(), materrors_n.nrow(), materrors_n.ncol(), false);

    NumericMatrix matot_n(matot);
    arma::mat matrixot(matot_n.begin(), matot_n.nrow(), matot_n.ncol(), false);

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

// ### matg should contain persistence vectors in each column
    NumericMatrix matg_n(matg);
    arma::mat matrixG(matg_n.begin(), matg_n.nrow(), matg_n.ncol(), false);

    unsigned int obs = materrors_n.nrow();
    unsigned int nseries = materrors_n.ncol();
    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    return wrap(simulateETS(arrayVt, matrixerrors, matrixot, matrixF, rowvecW, matrixG,
                            obs, nseries, E, T, S, lags));
}
