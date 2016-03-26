#include <RcppArmadillo.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

List ssfitter(arma::mat matrixxt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecY, arma::vec vecG, arma::uvec lags,
              arma::mat matrixWX, arma::mat matrixX, arma::mat matrixweightsX, arma::mat matrixFX, arma::mat vecGX) {
/* # matrixxt should have a length of obs + maxlag.
 * # rowvecW should have obs rows (can be all similar).
 * # matgt should be a vector
 * # lags is a vector of lags
 * # matrixWX is the matrix with the exogenous variables
 * # matrixX is the matrix with the parameters for the exogenous (repeated)
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
        matyfit.row(i-maxlag) = rowvecW * matrixxt(lagrows) + matrixWX.row(i-maxlag) * arma::trans(matrixX.row(i-maxlag));
        materrors(i-maxlag) = vecY(i-maxlag) - matyfit(i-maxlag);

/* # Transition equation */
        matrixxt.row(i) = arma::trans(matrixF * matrixxt(lagrows) + vecG * materrors(i-maxlag));

/* # Transition equation for xreg */
        bufferforxtreg = arma::trans(vecGX / arma::trans(matrixweightsX.row(i-maxlag)) * materrors(i-maxlag));
        bufferforxtreg.elem(find_nonfinite(bufferforxtreg)).fill(0);
        matrixX.row(i) = matrixX.row(i-1) * matrixFX + bufferforxtreg;
      }

    return List::create(Named("matxt") = matrixxt, Named("yfit") = matyfit,
                        Named("errors") = materrors, Named("xtreg") = matrixX);
}

/* # Wrapper for ssfitter */
// [[Rcpp::export]]
RcppExport SEXP ssfitterwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg1, SEXP modellags,
                             SEXP matwex, SEXP matxtreg, SEXP matv, SEXP matF2, SEXP vecg2) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());

    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);

    NumericMatrix vw(matw);
    arma::rowvec rowvecW(vw.begin(), vw.ncol(), false);

    NumericMatrix vyt(yt);
    arma::vec vecY(vyt.begin(), vyt.nrow(), vyt.ncol(), false);

    NumericMatrix vg(vecg1);
    arma::vec vecG(vg.begin(), vg.nrow(), false);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

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

    return wrap(ssfitter(matrixxt, matrixF, rowvecW, vecY, vecG, lags, matrixWX, matrixX, matrixweightsX, matrixFX, vecGX));
}

/* # Function fills in the values of the provided matrixX using the transition matrix. Needed for forecast of coefficients of xreg. */
List ssstatetail(arma::mat matrixxt, arma::mat matrixF, arma::mat matrixX, arma::mat matrixFX, arma::uvec lags){

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
        matrixxt.row(i) = arma::trans(matrixF * matrixxt(lagrows));
      }

    for(int i=0; i<(matrixX.n_rows-1); i=i+1){
        matrixX.row(i+1) = matrixX.row(i) * matrixFX;
    }

    return(List::create(Named("matxt") = matrixxt, Named("xtreg") = matrixX));
}

/* # Wrapper for ssstatetail */
// [[Rcpp::export]]
RcppExport SEXP ssstatetailwrap(SEXP matxt, SEXP matF, SEXP matxtreg, SEXP matF2, SEXP modellags){
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

    return(wrap(ssstatetail(matrixxt, matrixF, matrixX, matrixFX, lags)));
}

/* # Function produces the point forecasts for the specified model */
arma::mat ssforecaster(arma::mat matrixxt, arma::mat matrixF, arma::rowvec rowvecW,
                       unsigned int hor, arma::uvec lags,
                       arma::mat matrixWX, arma::mat matrixX) {
/* # Provide only the sufficient matrixxt (with the length = maxlag).
 * # nrows of rowvecW, matrixWX and matrixX should be >= hor
 */

    int lagslength = lags.n_rows;
    unsigned int maxlag = max(lags);
    unsigned int hh = hor + maxlag;

    arma::uvec lagrows(lagslength, arma::fill::zeros);
    arma::vec matyfor(hor, arma::fill::zeros);
    arma::mat matrixxtnew(hh, matrixxt.n_cols, arma::fill::zeros);
// This needs to be fixed! The matrixXnew should change with matrixFX!!!
    arma::mat matrixXnew(hh, matrixX.n_cols, arma::fill::zeros);

    lags = maxlag - lags;
    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + hh * i;
    }

    matrixxtnew.submat(0,0,maxlag-1,matrixxtnew.n_cols-1) = matrixxt.submat(0,0,maxlag-1,matrixxtnew.n_cols-1);

/* # Fill in the new xt matrix using F. Do the forecasts. */
    for (int i=maxlag; i<(hor+maxlag); i=i+1) {
        lagrows = lags - maxlag + i;
        matrixxtnew.row(i) = arma::trans(matrixF * matrixxtnew(lagrows));
        matyfor.row(i-maxlag) = rowvecW * matrixxtnew(lagrows) + matrixWX.row(i-maxlag) * arma::trans(matrixX.row(i-maxlag));
    }

    return matyfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP ssforecasterwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP h,
                                 SEXP modellags, SEXP matwex, SEXP matxtreg){
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::rowvec rowvecW(vw.begin(), vw.ncol(), false);
    unsigned int hor = as<int>(h);
    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);
    NumericMatrix mwex(matwex);
    arma::mat matrixWX(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return wrap(ssforecaster(matrixxt, matrixF, rowvecW, hor, lags, matrixWX, matrixX));
}

arma::mat sserrorer(arma::mat matrixxt, arma::mat matrixF, arma::rowvec rowvecW,
                    arma::vec vecY, unsigned int hor, arma::uvec lags,
                    arma::mat matrixWX, arma::mat matrixX){
    unsigned int obs = vecY.n_rows;
    unsigned int maxlag = max(lags);
    unsigned int hh;
    arma::mat materrors(obs, hor);

    materrors.fill(NA_REAL);

    for(unsigned int i=maxlag; i<obs+maxlag; i=i+1){
        hh = std::min(hor, obs+maxlag-i);
        materrors.submat(i-maxlag, 0, i-maxlag, hh-1) = arma::trans(vecY.rows(i-maxlag, i-maxlag+hh-1) -
            ssforecaster(matrixxt.rows(i-maxlag,i-1), matrixF, rowvecW, hh, lags,
                         matrixWX.rows(i-maxlag,i-maxlag+hh-1), matrixX.rows(i-maxlag,i-maxlag+hh-1)));
    }

    return materrors;
}

/* # Wrapper for errorer */
// [[Rcpp::export]]
RcppExport SEXP sserrorerwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP h, SEXP modellags,
                              SEXP matwex, SEXP matxtreg, SEXP matv, SEXP matF2, SEXP vecg2) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol(), false);
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::rowvec rowvecW(vw.begin(), vw.ncol(), false);
    NumericMatrix vyt(yt);
    arma::vec vecY(vyt.begin(), vyt.nrow(), false);
    unsigned int hor = as<int>(h);
    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);
    NumericMatrix mwex(matwex);
    arma::mat matrixWX(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat matrixX(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol(), false);

  return wrap(sserrorer(matrixxt, matrixF, rowvecW, vecY, hor, lags,
                        matrixWX, matrixX));
}

/* # Cost function calculation */
double ssoptimizer(arma::mat matrixxt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecY, arma::vec vecG,
                   unsigned int hor, arma::uvec lags, bool multi, std::string CFtype, double normalize,
                   arma::mat matrixWX, arma::mat matrixX, arma::mat matrixweightsX, arma::mat matrixFX, arma::mat vecGX) {
/* # Silent the output of try catch */
    std::ostream nullstream(0);
    arma::set_stream_err2(nullstream);

    double CFres = 0;
    int obs = vecY.n_rows;
    int matobs = obs - hor + 1;

    List fitting;

    fitting = ssfitter(matrixxt, matrixF, rowvecW, vecY, vecG, lags, matrixWX, matrixX, matrixweightsX, matrixFX, vecGX);

    NumericMatrix mxtfromfit = as<NumericMatrix>(fitting["matxt"]);
    matrixxt = as<arma::mat>(mxtfromfit);
    NumericMatrix errorsfromfit = as<NumericMatrix>(fitting["errors"]);
    NumericMatrix mxtregfromfit = as<NumericMatrix>(fitting["xtreg"]);
    matrixX = as<arma::mat>(mxtregfromfit);

    arma::mat materrors;
    arma::rowvec horvec(hor);

    if(multi==true){
        for(unsigned int i=0; i<hor; i=i+1){
            horvec(i) = hor - i;
        }
        materrors = sserrorer(matrixxt, matrixF, rowvecW, vecY, hor, lags, matrixWX, matrixX);
        materrors.row(0) = materrors.row(0) % horvec;
    }

/* # The matrix is cut of to be square. If the backcast is done to the additional points, this can be fixed. */
    if(CFtype=="GV"){
        materrors.resize(matobs,hor);
        try{
            CFres = double(log(arma::prod(eig_sym(trans(materrors / normalize) * (materrors / normalize) / matobs))) + hor * log(pow(normalize,2)));
        }
        catch(const std::runtime_error){
            CFres = double(log(arma::det(arma::trans(materrors / normalize) * (materrors / normalize) / matobs)) + hor * log(pow(normalize,2)));
        }
    }
    else if(CFtype=="trace"){
        for(unsigned int i=0; i<hor; i=i+1){
            CFres = CFres + arma::as_scalar(log(mean(pow(materrors.submat(0,i,obs-i-1,i),2))));
        }
    }
    else if(CFtype=="TV"){
        for(unsigned int i=0; i<hor; i=i+1){
            CFres = CFres + arma::as_scalar(mean(pow(materrors.submat(0,i,obs-i-1,i),2)));
        }
    }
    else if(CFtype=="MSEh"){
        CFres = arma::as_scalar(mean(pow(materrors.submat(0,hor-1,obs-hor,hor-1),2)));
    }
    else if(CFtype=="MSE"){
        arma::mat materrors(errorsfromfit.begin(), errorsfromfit.nrow(), errorsfromfit.ncol(), false);
        CFres = arma::as_scalar(mean(pow(materrors,2)));
    }
    else if(CFtype=="MAE"){
        arma::mat materrors(errorsfromfit.begin(), errorsfromfit.nrow(), errorsfromfit.ncol(), false);
        CFres = arma::as_scalar(mean(abs(materrors)));
    }
    else{
        arma::mat materrors(errorsfromfit.begin(), errorsfromfit.nrow(), errorsfromfit.ncol(), false);
        CFres = arma::as_scalar(mean(pow(abs(materrors),0.5)));
    }

    return CFres;
}


/* # Wrapper for optimiser */
// [[Rcpp::export]]
RcppExport SEXP ssoptimizerwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg1, SEXP h,
                                SEXP modellags, SEXP multisteps, SEXP CFt, SEXP normalizer,
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

    unsigned int hor = as<int>(h);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

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

    return wrap(ssoptimizer(matrixxt,matrixF,rowvecW,vecY,vecG,hor,lags,multi,
                            CFtype,normalize,matrixWX,matrixX,matrixweightsX,matrixFX,vecGX));
}
