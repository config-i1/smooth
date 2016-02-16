#include <RcppArmadillo.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

List ssfitter(arma::mat matrixxt, arma::mat matrixF, arma::mat matrixw, arma::mat matrixv, arma::vec matyt,
            arma::vec matg, arma::uvec freqs, arma::mat wex, arma::mat xtreg) {
/* # matrixxt should have a length of obs + freq.
 * # matrixw should have obs rows (can be all similar).
 * # matgt should be a vector
 * # freqs is a vector of lags
 * # wex is the matrix with the exogenous variables
 * # xtreg is the matrix with the parameters for the exogenous (repeated)
 */

    int obs = matyt.n_rows;
    int obsall = matrixxt.n_rows;
    int freqslength = freqs.n_rows;
    unsigned int freq = max(freqs);

    freqs = freq - freqs;

    for(int i=1; i<freqslength; i=i+1){
        freqs(i) = freqs(i) + obsall * i;
    }

    arma::uvec freqrows(freqslength, arma::fill::zeros);

    arma::vec matyfit(obs, arma::fill::zeros);
    arma::vec materrors(obs, arma::fill::zeros);

    for (int i=freq; i<obsall; i=i+1) {

        freqrows = freqs - freq + i;

        matyfit.row(i-freq) = matrixw.row(i-freq) * matrixxt(freqrows) + wex.row(i-freq) * arma::trans(xtreg.row(i-freq));
        materrors(i-freq) = matyt(i-freq) - matyfit(i-freq);
        matrixxt.row(i) = arma::trans(matrixF * matrixxt(freqrows) + matg / arma::trans(matrixv.row(i-freq)) * materrors(i-freq));
        matrixxt.elem(find_nonfinite(matrixxt)) = matrixxt.elem(find_nonfinite(matrixxt) - 1);
      }

    return List::create(Named("matxt") = matrixxt, Named("yfit") = matyfit, Named("errors") = materrors, Named("xtreg") = xtreg);
}

/* # Wrapper for ssfitter */
// [[Rcpp::export]]
RcppExport SEXP ssfitterwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP matv, SEXP yt, SEXP vecg,
                             SEXP seasfreqs, SEXP matwex, SEXP matxtreg) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::mat matrixw(vw.begin(), vw.nrow(), vw.ncol(), false);
    NumericMatrix vv(matv);
    arma::mat matrixv(vv.begin(), vv.nrow(), vv.ncol(), false);
    NumericMatrix vyt(yt);
    arma::vec matyt(vyt.begin(), vyt.nrow(), vyt.ncol(), false);
    NumericMatrix vg(vecg);
    arma::vec matg(vg.begin(), vg.nrow(), false);
    IntegerVector sfreqs(seasfreqs);
    arma::uvec freqs = as<arma::uvec>(sfreqs);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return wrap(ssfitter(matrixxt, matrixF, matrixw, matrixv, matyt, matg, freqs, wex, xtreg));
}

/* # Function produces the point forecasts for the specified model */
arma::mat ssforecaster(arma::mat matrixxt, arma::mat matrixF, arma::mat matrixw,
                       unsigned int hor, arma::uvec freqs, arma::mat wex, arma::mat xtreg) {
/* # Provide only the sufficient matrixxt (with the length = freq).
 * # nrows of matrixw, wex and xtreg should be >= hor
 */

    int freqslength = freqs.n_rows;
    unsigned int freq = max(freqs);
    unsigned int hh = hor + freq;

    arma::uvec freqrows(freqslength, arma::fill::zeros);
    arma::vec matyfor(hor, arma::fill::zeros);
    arma::mat matrixxtnew(hh, matrixxt.n_cols, arma::fill::zeros);

    freqs = freq - freqs;
    for(int i=1; i<freqslength; i=i+1){
        freqs(i) = freqs(i) + hh * i;
    }

    matrixxtnew.submat(0,0,freq-1,matrixxtnew.n_cols-1) = matrixxt.submat(0,0,freq-1,matrixxtnew.n_cols-1);

/* # Fill in the new xt matrix using F. Do the forecasts. */
    for (int i=freq; i<(hor+freq); i=i+1) {
        freqrows = freqs - freq + i;
        matrixxtnew.row(i) = arma::trans(matrixF * matrixxtnew(freqrows));
        matyfor.row(i-freq) = matrixw.row(i-freq) * matrixxtnew(freqrows) + wex.row(i-freq) * arma::trans(xtreg.row(i-freq));
    }

    return matyfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP ssforecasterwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP h,
                                 SEXP seasfreqs, SEXP matwex, SEXP matxtreg){
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::mat matrixw(vw.begin(), vw.nrow(), vw.ncol(), false);
    unsigned int hor = as<int>(h);
    IntegerVector sfreqs(seasfreqs);
    arma::uvec freqs = as<arma::uvec>(sfreqs);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return wrap(ssforecaster(matrixxt, matrixF, matrixw, hor, freqs, wex, xtreg));
}

arma::mat sserrorer(arma::mat matrixxt, arma::mat matrixF, arma::mat matrixw,
                    arma::vec matyt, unsigned int hor, arma::uvec freqs, arma::mat wex, arma::mat xtreg){
    unsigned int obs = matyt.n_rows;
    unsigned int freq = max(freqs);
    unsigned int hh;
    arma::mat materrors(obs, hor);

    materrors.fill(NA_REAL);

    for(int i=freq; i<obs+freq; i=i+1){
        hh = std::min(hor, obs+freq-i);
        materrors.submat(i-freq, 0, i-freq, hh-1) = arma::trans(matyt.rows(i-freq, i-freq+hh-1) -
            ssforecaster(matrixxt.rows(i-freq,i-1), matrixF, matrixw.rows(i-freq,i-freq+hh-1), hh, freqs,
                         wex.rows(i-freq,i-freq+hh-1), xtreg.rows(i-freq,i-freq+hh-1)));
    }

    return materrors;
}

/* # Wrapper for errorer */
// [[Rcpp::export]]
RcppExport SEXP sserrorerwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP h,
                                    SEXP seasfreqs, SEXP matwex, SEXP matxtreg) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol(), false);
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::mat matrixw(vw.begin(), vw.nrow(), vw.ncol(), false);
    NumericMatrix vyt(yt);
    arma::vec matyt(vyt.begin(), vyt.nrow(), false);
    unsigned int hor = as<int>(h);
    IntegerVector sfreqs(seasfreqs);
    arma::uvec freqs = as<arma::uvec>(sfreqs);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol(), false);

  return wrap(sserrorer(matrixxt, matrixF, matrixw, matyt, hor, freqs, wex, xtreg));
}

/* # Cost function calculation */
double ssoptimizer(arma::mat matrixxt, arma::mat matrixF, arma::mat matrixw, arma::mat matrixv, arma::vec matyt, arma::vec matg,
                 unsigned int hor, arma::uvec freqs, std::string CFtype, double normalize, arma::mat wex, arma::mat xtreg) {
/* # Silent the output of try catch */
    std::ostream nullstream(0);
    arma::set_stream_err2(nullstream);

    double CFres = 0;
    int obs = matyt.n_rows;
    int matobs = obs - hor + 1;

    List fitting = ssfitter(matrixxt, matrixF, matrixw, matrixv, matyt, matg, freqs, wex, xtreg);
    NumericMatrix mxtfromfit = as<NumericMatrix>(fitting["matxt"]);
    matrixxt = as<arma::mat>(mxtfromfit);
    NumericMatrix errorsfromfit = as<NumericMatrix>(fitting["errors"]);
    NumericMatrix mxtregfromfit = as<NumericMatrix>(fitting["xtreg"]);
    xtreg = as<arma::mat>(mxtregfromfit);

    arma::mat materrors;
/* # The matrix is cut of to be square. If the backcast is done to the additional points, this can be fixed. */
    if(CFtype=="GV"){
        materrors = sserrorer(matrixxt, matrixF, matrixw, matyt, hor, freqs, wex, xtreg);
        materrors.resize(matobs,hor);
        try{
            CFres = double(log(arma::prod(eig_sym(trans(materrors / normalize) * (materrors / normalize) / matobs))) + hor * log(pow(normalize,2)));
        }
        catch(const std::runtime_error){
            CFres = double(log(arma::det(arma::trans(materrors / normalize) * (materrors / normalize) / matobs)) + hor * log(pow(normalize,2)));
        }
    }
    else if(CFtype=="TLV"){
        materrors = sserrorer(matrixxt, matrixF, matrixw, matyt, hor, freqs, wex, xtreg);
        for(int i=0; i<hor; i=i+1){
            CFres = CFres + arma::as_scalar(log(mean(pow(materrors.submat(0,i,obs-i-1,i),2))));
        }
    }
    else if(CFtype=="TV"){
        materrors = sserrorer(matrixxt, matrixF, matrixw, matyt, hor, freqs, wex, xtreg);
        for(int i=0; i<hor; i=i+1){
            CFres = CFres + arma::as_scalar(mean(pow(materrors.submat(0,i,obs-i-1,i),2)));
        }
    }
    else if(CFtype=="hsteps"){
        materrors = sserrorer(matrixxt, matrixF, matrixw, matyt, hor, freqs, wex, xtreg);
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
RcppExport SEXP ssoptimizerwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP matv, SEXP yt, SEXP vecg, SEXP h,
                                SEXP seasfreqs, SEXP CFt, SEXP normalizer, SEXP matwex, SEXP matxtreg) {

    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::mat matrixw(vw.begin(), vw.nrow(), vw.ncol(), false);
    NumericMatrix vv(matv);
    arma::mat matrixv(vv.begin(), vv.nrow(), vv.ncol(), false);
    NumericMatrix vyt(yt);
    arma::vec matyt(vyt.begin(), vyt.nrow(), false);
    NumericMatrix vg(vecg);
    arma::vec matg(vg.begin(), vg.nrow(), false);
    unsigned int hor = as<int>(h);
    IntegerVector sfreqs(seasfreqs);
    arma::uvec freqs = as<arma::uvec>(sfreqs);
    std::string CFtype = as<std::string>(CFt);
    double normalize = as<double>(normalizer);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return wrap(ssoptimizer(matrixxt,matrixF,matrixw,matrixv,matyt,matg,hor,freqs,CFtype,normalize,wex,xtreg));
}
