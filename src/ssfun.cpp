#include <RcppArmadillo.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

List ssfitter(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecYt, arma::vec vecG, arma::uvec lags,
              arma::mat matrixXt, arma::mat matrixAt, arma::mat matrixFX, arma::mat vecGX, arma::vec vecOt) {
/* # matrixVt should have a length of obs + maxlag.
 * # rowvecW should have obs rows (can be all similar).
 * # matgt should be a vector
 * # lags is a vector of lags
 * # matrixXt is the matrix with the exogenous variables
 * # matrixAt is the matrix with the parameters for the exogenous (repeated)
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
        matyfit.row(i-maxlag) = vecOt(i-maxlag) * (rowvecW * matrixVt(lagrows) +
                                matrixXt.row(i-maxlag) * arma::trans(matrixAt.row(i-maxlag)));
        materrors(i-maxlag) = vecYt(i-maxlag) - matyfit(i-maxlag);

/* # Transition equation */
        matrixVt.row(i) = arma::trans(matrixF * matrixVt(lagrows) + vecG * materrors(i-maxlag));

/* # Transition equation for xreg */
        bufferforat = arma::trans(vecGX / arma::trans(matrixXt.row(i-maxlag)) * materrors(i-maxlag));
        bufferforat.elem(find_nonfinite(bufferforat)).fill(0);
        matrixAt.row(i) = matrixAt.row(i-1) * matrixFX + bufferforat;
      }

    return List::create(Named("matvt") = matrixVt, Named("yfit") = matyfit,
                        Named("errors") = materrors, Named("matat") = matrixAt);
}

/* # Wrapper for ssfitter */
// [[Rcpp::export]]
RcppExport SEXP ssfitterwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg, SEXP modellags,
                             SEXP matxt, SEXP matat, SEXP matFX, SEXP vecgX, SEXP ot) {
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

    NumericMatrix yt_n(yt);
    arma::vec vecYt(yt_n.begin(), yt_n.nrow(), yt_n.ncol(), false);

    NumericMatrix vecg_n(vecg);
    arma::vec vecG(vecg_n.begin(), vecg_n.nrow(), false);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

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

    return wrap(ssfitter(matrixVt, matrixF, rowvecW, vecYt, vecG, lags, matrixXt, matrixAt, matrixFX, vecGX, vecOt));
}

/* # Function fills in the values of the provided matrixAt using the transition matrix. Needed for forecast of coefficients of xreg. */
List ssstatetail(arma::mat matrixVt, arma::mat matrixF, arma::mat matrixAt, arma::mat matrixFX, arma::uvec lags){

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
        matrixVt.row(i) = arma::trans(matrixF * matrixVt(lagrows));
      }

    for(int i=0; i<(matrixAt.n_rows-1); i=i+1){
        matrixAt.row(i+1) = matrixAt.row(i) * matrixFX;
    }

    return(List::create(Named("matvt") = matrixVt, Named("matat") = matrixAt));
}

/* # Wrapper for ssstatetail */
// [[Rcpp::export]]
RcppExport SEXP ssstatetailwrap(SEXP matvt, SEXP matF, SEXP matat, SEXP matFX, SEXP modellags){
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());
    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);
    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());
    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);
    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    return(wrap(ssstatetail(matrixVt, matrixF, matrixAt, matrixFX, lags)));
}

/* # Function produces the point forecasts for the specified model */
arma::mat ssforecaster(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW,
                       unsigned int hor, arma::uvec lags,
                       arma::mat matrixXt, arma::mat matrixAt, arma::mat matrixFX) {
/* # Provide only the sufficient matrixVt (with the length = maxlag).
 * # nrows of rowvecW, matrixXt and matrixAt should be >= hor
 */

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
        matrixVtnew.row(i) = arma::trans(matrixF * matrixVtnew(lagrows));
        matrixAtnew.row(i) = matrixAtnew.row(i-1) * matrixFX;

        matyfor.row(i-maxlag) = (rowvecW * matrixVtnew(lagrows) + matrixXt.row(i-maxlag) * arma::trans(matrixAt.row(i-maxlag)));
    }

    return matyfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP ssforecasterwrap(SEXP matvt, SEXP matF, SEXP matw,
                                 SEXP h, SEXP modellags,
                                 SEXP matxt, SEXP matat, SEXP matFX){
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

    unsigned int hor = as<int>(h);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    return wrap(ssforecaster(matrixVt, matrixF, rowvecW, hor, lags, matrixXt, matrixAt, matrixFX));
}

arma::mat sserrorer(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW,
                    arma::vec vecYt, unsigned int hor, arma::uvec lags,
                    arma::mat matrixXt, arma::mat matrixAt, arma::mat matrixFX, arma::vec vecOt){
    unsigned int obs = vecYt.n_rows;
    unsigned int maxlag = max(lags);
    unsigned int hh;
    arma::mat materrors(obs, hor);

    materrors.fill(NA_REAL);

    for(unsigned int i=maxlag; i<obs+maxlag; i=i+1){
        hh = std::min(hor, obs+maxlag-i);
        materrors.submat(i-maxlag, 0, i-maxlag, hh-1) = arma::trans(vecOt.rows(i-maxlag, i-maxlag+hh-1) % (vecYt.rows(i-maxlag, i-maxlag+hh-1) -
            ssforecaster(matrixVt.rows(i-maxlag,i-1), matrixF, rowvecW, hh, lags,
                         matrixXt.rows(i-maxlag,i-maxlag+hh-1), matrixAt.rows(i-maxlag,i-maxlag+hh-1), matrixFX)));
    }

    return materrors;
}

/* # Wrapper for errorer */
// [[Rcpp::export]]
RcppExport SEXP sserrorerwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP h, SEXP modellags,
                              SEXP matxt, SEXP matat, SEXP matFX, SEXP ot) {
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol(), false);

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

    NumericMatrix yt_n(yt);
    arma::vec vecYt(yt_n.begin(), yt_n.nrow(), false);

    unsigned int hor = as<int>(h);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol(), false);

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericVector ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.size(), false);

    return wrap(sserrorer(matrixVt, matrixF, rowvecW, vecYt,
                          hor, lags,
                          matrixXt, matrixAt, matrixFX, vecOt));
}

/* # Cost function calculation */
double ssoptimizer(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecYt, arma::vec vecG,
                   unsigned int hor, arma::uvec lags, bool multi, std::string CFtype, double normalize,
                   arma::mat matrixXt, arma::mat matrixAt, arma::mat matrixFX, arma::mat vecGX, arma::vec vecOt) {
/* # Silent the output of try catch */
    std::ostream nullstream(0);
    arma::set_stream_err2(nullstream);

    double CFres = 0;
    int obs = vecYt.n_rows;
    int matobs = obs - hor + 1;

    List fitting;

    fitting = ssfitter(matrixVt, matrixF, rowvecW, vecYt, vecG, lags, matrixXt, matrixAt, matrixFX, vecGX, vecOt);

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
        materrors = sserrorer(matrixVt, matrixF, rowvecW, vecYt, hor, lags, matrixXt, matrixAt, matrixFX, vecOt);
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
RcppExport SEXP ssoptimizerwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg, SEXP h,
                                SEXP modellags, SEXP multisteps, SEXP CFt, SEXP normalizer,
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

    unsigned int hor = as<int>(h);

    IntegerVector mlags(modellags);
    arma::uvec lags = as<arma::uvec>(mlags);

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

    return wrap(ssoptimizer(matrixVt,matrixF,rowvecW,vecYt,vecG,hor,lags,multi,
                            CFtype,normalize,matrixXt,matrixAt,matrixFX,vecGX,vecOt));
}
