#include <RcppArmadillo.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/* # The function allows to calculate the power of a matrix. */
arma::mat cesmatrixpower(arma::mat A, int power){
    arma::mat B = A;
    if(power>1){
        for(int i = 1; i < power; i=i+1){
            B = B * A;
        }
    }
    else if(power==0){
        B.eye();
    }
    return B;
}

List cesfitter(arma::mat matrixxt, arma::mat matrixF, arma::mat matrixw, arma::mat matyt,
            arma::mat matg, char S, int freq, arma::mat wex, arma::mat xtreg) {
/* # xt contains obs + h + freq observations or obs + 2 freq. freq observations are in the preparation part,
#  that's why we start from t=1-freq here, while yt and et start from t=1 */

    int obs = matyt.n_rows;
    int obsall = matrixxt.n_rows;
    int ncomponents = matg.n_rows;

    arma::mat matyfit(obs, 1, arma::fill::zeros);
    arma::mat materrors(obs, 1, arma::fill::zeros);

    arma::rowvec xtnew(ncomponents);

/* # The first 3 runs of backcast */
    for(int j=0; j<2; j=j+1){
      for (int i=freq; i<obs+freq; i=i+1) {
/* # Fill in the series with actuals (obs) */
        if((S=='N') || (S=='S')){
          xtnew = matrixxt.row(i-freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixxt.submat(i-1,0,i-1,1);
          xtnew.col(2) = matrixxt.submat(i-freq,2,i-freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixxt.submat(i-1,0,i-1,1);
          xtnew.cols(2,3) = matrixxt.submat(i-freq,2,i-freq,3);
        }
        matyfit.row(i-freq) = matrixw * trans(xtnew) + wex.row(i-freq) * trans(xtreg.row(i-freq));
        materrors(i-freq,0) = matyt(i-freq,0) - matyfit(i-freq,0);
        matrixxt.row(i) = xtnew * trans(matrixF) + trans(matg * materrors.row(i-freq));
        xtreg.row(i) = xtreg.row(i-freq);
      }
      for(int i=obs+freq; i<obsall; i=i+1){
/* # Fill in the small additional bit at the end of series, that doesn't contain actuals anymore */
        if((S=='N') || (S=='S')){
          xtnew = matrixxt.row(i-freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixxt.submat(i-1,0,i-1,1);
          xtnew.col(2) = matrixxt.submat(i-freq,2,i-freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixxt.submat(i-1,0,i-1,1);
          xtnew.cols(2,3) = matrixxt.submat(i-freq,2,i-freq,3);
        }
        matrixxt.row(i) = xtnew * trans(matrixF);
      }
      for (int i=obs+freq-1; i>=freq; i=i-1) {
        if((S=='N') || (S=='S')){
          xtnew = matrixxt.row(i+freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixxt.submat(i+1,0,i+1,1);
          xtnew.col(2) = matrixxt.submat(i+freq,2,i+freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixxt.submat(i+1,0,i+1,1);
          xtnew.cols(2,3) = matrixxt.submat(i+freq,2,i+freq,3);
        }
        matyfit.row(i-freq) = matrixw * trans(xtnew) + wex.row(i-freq) * trans(xtreg.row(i-freq));
        materrors(i-freq,0) = matyt(i-freq,0) - matyfit(i-freq,0);
        matrixxt.row(i) = xtnew * trans(matrixF) + trans(matg * materrors.row(i-freq));
      }
      for (int i=freq-1; i>=0; i=i-1) {
        if((S=='N') || (S=='S')){
          xtnew = matrixxt.row(i+freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixxt.submat(i+1,0,i+1,1);
          xtnew.col(2) = matrixxt.submat(i+freq,2,i+freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixxt.submat(i+1,0,i+1,1);
          xtnew.cols(2,3) = matrixxt.submat(i+freq,2,i+freq,3);
        }
        matrixxt.row(i) = xtnew * trans(matrixF);
      }
    }

/* # The final run */
      for (int i=freq; i<obs+freq; i=i+1) {
        if((S=='N') || (S=='S')){
          xtnew = matrixxt.row(i-freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixxt.submat(i-1,0,i-1,1);
          xtnew.col(2) = matrixxt.submat(i-freq,2,i-freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixxt.submat(i-1,0,i-1,1);
          xtnew.cols(2,3) = matrixxt.submat(i-freq,2,i-freq,3);
        }
        matyfit.row(i-freq) = matrixw * trans(xtnew) + wex.row(i-freq) * trans(xtreg.row(i-freq));
        materrors(i-freq,0) = matyt(i-freq,0) - matyfit(i-freq,0);
        matrixxt.row(i) = xtnew * trans(matrixF) + trans(matg * materrors.row(i-freq));
      }
      for(int i=obs+freq; i<obsall; i=i+1){
        if((S=='N') || (S=='S')){
          xtnew = matrixxt.row(i-freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixxt.submat(i-1,0,i-1,1);
          xtnew.col(2) = matrixxt.submat(i-freq,2,i-freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixxt.submat(i-1,0,i-1,1);
          xtnew.cols(2,3) = matrixxt.submat(i-freq,2,i-freq,3);
        }
        matrixxt.row(i) = xtnew * trans(matrixF);
        xtreg.row(i) = xtreg.row(i-freq);
      }

    return List::create(Named("matxt") = matrixxt, Named("yfit") = matyfit, Named("errors") = materrors, Named("xtreg") = xtreg);
}

/* # Wrapper for cesfitter */
// [[Rcpp::export]]
RcppExport SEXP cesfitterwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg,
                           SEXP Stype, SEXP seasfreq, SEXP matwex, SEXP matxtreg) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::mat matrixw(vw.begin(), vw.nrow(), vw.ncol(), false);
    NumericMatrix vyt(yt);
    arma::mat matyt(vyt.begin(), vyt.nrow(), vyt.ncol(), false);
    NumericMatrix vg(vecg);
    arma::mat matg(vg.begin(), vg.nrow(), vg.ncol(), false);
    char S = as<char>(Stype);
    int freq = as<int>(seasfreq);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return wrap(cesfitter(matrixxt, matrixF, matrixw, matyt, matg, S, freq, wex, xtreg));
}

arma::vec cesforecaster(arma::mat matrixxt,arma::mat matrixF,arma::rowvec matrixw,
                     int hor,char S,int freq,arma::mat wex,arma::mat xtreg){
    int hh;
    int ncomponents = matrixxt.n_cols;

    arma::mat matxtnew(hor, ncomponents);
    arma::rowvec xtnew(ncomponents);

    arma::vec matyfor(hor, arma::fill::zeros);
/* # Preparation of all the matrices for the mean */
    if(S=='N'){
      matxtnew.submat(0,0,0,1) = matrixxt.submat(matrixxt.n_rows-1,0,matrixxt.n_rows-1,1);
        if(hor > freq){
            for(int i = 1; i < hor; i=i+1){
                matxtnew.row(i) = matxtnew.row(i-1) * arma::trans(matrixF);
            }
        }
    }
    else if(S=='S'){
      hh = std::min(hor,freq);
      matxtnew.submat(0,0,hh-1,1) = matrixxt.submat(0,0,hh-1,1);
        if(hor > freq){
            for(int i = freq; i < hor; i=i+1){
                matxtnew.row(i) = matxtnew.row(i-freq) * arma::trans(matrixF);
            }
        }
    }
    else if(S=='P'){
      matxtnew.submat(0,0,0,1) = matrixxt.submat(matrixxt.n_rows-1,0,matrixxt.n_rows-1,1);
      matxtnew(0,2) = matrixxt(0,2);
      hh = std::min(hor,freq);
      for(int i = 1; i<hh; i=i+1){
        xtnew.cols(0,1) = matxtnew.submat(i-1,0,i-1,1);
        xtnew.col(2) = matrixxt(i,2);
        matxtnew.row(i) = xtnew * arma::trans(matrixF);
      }
      if(hor > freq){
          for(int i = freq; i < hor; i=i+1){
              xtnew.cols(0,1) = matxtnew.submat(i-1,0,i-1,1);
              xtnew.col(2) = matxtnew.submat(i-freq,2,i-freq,2);
              matxtnew.row(i) = xtnew * arma::trans(matrixF);
          }
      }
    }
    else if(S=='F'){
      matxtnew.submat(0,0,0,1) = matrixxt.submat(matrixxt.n_rows-1,0,matrixxt.n_rows-1,1);
      matxtnew.submat(0,2,0,3) = matrixxt.submat(0,2,0,3);
      hh = std::min(hor,freq);
      for(int i=1; i<hh; i=i+1){
        xtnew.cols(0,1) = matxtnew.submat(i-1,0,i-1,1);
        xtnew.cols(2,3) = matrixxt.submat(i,2,i,3);
        matxtnew.row(i) = xtnew * arma::trans(matrixF);
      }
      if(hor>freq){
          for(int i=freq; i<hor; i=i+1){
              xtnew.cols(0,1) = matxtnew.submat(i-1,0,i-1,1);
              xtnew.cols(2,3) = matxtnew.submat(i-freq,2,i-freq,3);
              matxtnew.row(i) = xtnew * arma::trans(matrixF);
          }
      }
    }
/* # Forecast */
    for(int i=0; i<hor; i=i+1){
        matyfor.row(i) = matrixw * trans(matxtnew.row(i)) + wex.row(i) * trans(xtreg.row(i));
    }
    return matyfor;
}

// [[Rcpp::export]]
RcppExport SEXP cesforecasterwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP h,
                                    SEXP Stype, SEXP seasfreq, SEXP matwex, SEXP matxtreg) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol(), false);
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::rowvec matrixw(vw.begin(), vw.ncol(), false);
    unsigned int hor = as<int>(h);
    char S = as<char>(Stype);
    unsigned int freq = as<int>(seasfreq);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol(), false);

  return wrap(cesforecaster(matrixxt,matrixF,matrixw,hor,S,freq,wex,xtreg));
}

arma::mat ceserrorer(arma::mat matrixxt,arma::mat matrixF,arma::rowvec matrixw,arma::vec matyt,int hor,char S,int freq,arma::mat wex,arma::mat xtreg){
    int obs = matyt.n_rows;
    int hh;
    arma::mat materrors(obs, hor);

    materrors.fill(NA_REAL);
/* # This part was needed to backcast the errors...
     for(int i=obs+freq; i>obs-hor; i=i-1){
        hh = std::min(hor, i-freq);
        materrors.submat(i-freq-1, 0, i-freq-1, hh-1) = arma::trans(arma::flipud(matyt.rows(i-freq-hh, i-freq-1)) - cesforecaster(arma::flipud(matrixxt.rows(i,i+freq-1)),matrixF,matrixw,hh,S,freq,arma::flipud(wex.rows(i-freq-hh, i-freq-1)),xtreg.rows(i-freq-hh, i-freq-1)));
      } */
    for(int i=freq; i<obs+freq; i=i+1){
        hh = std::min(hor, obs+freq-i);
        materrors.submat(i-freq, 0, i-freq, hh-1) = arma::trans(matyt.rows(i-freq, i-freq+hh-1) - cesforecaster(matrixxt.rows(i-freq,i-1),matrixF,matrixw,hh,S,freq,wex.rows(i-freq,i-freq+hh-1),xtreg.rows(i-freq,i-freq+hh-1)));
    }
    return materrors;
}

// [[Rcpp::export]]
RcppExport SEXP ceserrorerwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP h, SEXP Stype,
                                 SEXP seasfreq, SEXP matwex, SEXP matxtreg) {
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol(), false);
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::rowvec matrixw(vw.begin(), vw.ncol(), false);
    NumericMatrix vyt(yt);
    arma::vec matyt(vyt.begin(), vyt.nrow(), false);
    unsigned int hor = as<int>(h);
    char S = as<char>(Stype);
    unsigned int freq = as<int>(seasfreq);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol(), false);

  return wrap(ceserrorer(matrixxt,matrixF,matrixw,matyt,hor,S,freq,wex,xtreg));
}

/* # Cost function calculation */
double cesoptimizer(arma::mat matrixxt, arma::mat matrixF, arma::mat matrixw, arma::mat matyt, arma::mat matg,
                 int hor, char S, int freq, bool multi, std::string CFtype, double normalize, arma::mat wex, arma::mat xtreg) {
    double CFres = 0;

    List fitting = cesfitter(matrixxt, matrixF, matrixw, matyt, matg, S, freq, wex, xtreg);
    NumericMatrix mxtfromfit = as<NumericMatrix>(fitting["matxt"]);
    matrixxt = as<arma::mat>(mxtfromfit);
    NumericMatrix errorsfromfit = as<NumericMatrix>(fitting["errors"]);
    NumericMatrix mxtregfromfit = as<NumericMatrix>(fitting["xtreg"]);
    xtreg = as<arma::mat>(mxtregfromfit);

    arma::mat materrors;
    arma::rowvec horvec(hor);

    if(multi==true){
        for(int i=0; i<hor; i=i+1){
            horvec(i) = hor - i;
        }
        materrors = ceserrorer(matrixxt,matrixF,matrixw,matyt,hor,S,freq,wex,xtreg);
        materrors.row(0) = materrors.row(0) % horvec;
    }

/* # The matrix is cut of to be square. If the bcakcast is done to the additional points, this can be fixed. */
    if(multi==true){
/* #  Matrix may be cut off if needed... */
        materrors = materrors.rows(0,(materrors.n_rows-hor));
        if(CFtype=="GV"){
            materrors = materrors / normalize;
            CFres = double(log(det(trans(materrors) * (materrors) / materrors.n_rows)) + hor * log(materrors.n_rows * pow(normalize,2)));
        }
        else if(CFtype=="trace"){
            for(int i=0; i<hor; i=i+1){
                CFres = CFres + arma::as_scalar(log(mean(pow(materrors.col(i),2))));
            }
        }
        else if(CFtype=="TV"){
            CFres = arma::as_scalar(sum(mean(pow(materrors,2),0),1));
        }
        else if(CFtype=="MSEh"){
            CFres = arma::as_scalar(mean(pow(materrors.col(hor-1),2)));
        }
    }
    else{
        if(CFtype=="MSE"){
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
    }

    return CFres;
}

/* # Wrapper for optimiser */
// [[Rcpp::export]]
RcppExport double cesoptimizerwrap(SEXP matxt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg, SEXP h,
                                SEXP Stype, SEXP seasfreq, SEXP multisteps, SEXP CFt, SEXP normalizer, SEXP matwex, SEXP matxtreg) {
/*  std::cout << "Function started" << std::endl; */
    NumericMatrix mxt(matxt);
    arma::mat matrixxt(mxt.begin(), mxt.nrow(), mxt.ncol());
    NumericMatrix mF(matF);
    arma::mat matrixF(mF.begin(), mF.nrow(), mF.ncol(), false);
    NumericMatrix vw(matw);
    arma::mat matrixw(vw.begin(), vw.nrow(), vw.ncol(), false);
    NumericMatrix vyt(yt);
    arma::mat matyt(vyt.begin(), vyt.nrow(), vyt.ncol(), false);
    NumericMatrix vg(vecg);
    arma::mat matg(vg.begin(), vg.nrow(), vg.ncol(), false);
    int hor = as<int>(h);
    char S = as<char>(Stype);
    int freq = as<int>(seasfreq);
    bool multi = as<bool>(multisteps);
    std::string CFtype = as<std::string>(CFt);
    double normalize = as<double>(normalizer);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return cesoptimizer(matrixxt,matrixF,matrixw,matyt,matg,hor,S,freq,multi,CFtype,normalize,wex,xtreg);
}
