#include <RcppArmadillo.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

List cesfitter(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW, arma::vec vecYt,
            arma::vec vecG, char S, int freq, arma::mat wex, arma::mat xtreg) {
/* # xt contains obs + h + freq observations or obs + 2 freq. freq observations are in the preparation part,
#  that's why we start from t=1-freq here, while yt and et start from t=1 */

    int obs = vecYt.n_rows;
    int obsall = matrixVt.n_rows;
    int ncomponents = vecG.n_rows;

    arma::vec matyfit(obs, arma::fill::zeros);
    arma::vec materrors(obs, arma::fill::zeros);

    arma::rowvec xtnew(ncomponents);

/* # The first 3 runs of backcast */
    for(int j=0; j<2; j=j+1){
      for (int i=freq; i<obs+freq; i=i+1) {
/* # Fill in the series with actuals (obs) */
        if((S=='N') || (S=='S')){
          xtnew = matrixVt.row(i-freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixVt.submat(i-1,0,i-1,1);
          xtnew.col(2) = matrixVt.submat(i-freq,2,i-freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixVt.submat(i-1,0,i-1,1);
          xtnew.cols(2,3) = matrixVt.submat(i-freq,2,i-freq,3);
        }
        matyfit.row(i-freq) = rowvecW * trans(xtnew) + wex.row(i-freq) * trans(xtreg.row(i-freq));
        materrors(i-freq) = vecYt(i-freq) - matyfit(i-freq);
        matrixVt.row(i) = xtnew * trans(matrixF) + trans(vecG * materrors.row(i-freq));
        xtreg.row(i) = xtreg.row(i-freq);
      }
      for(int i=obs+freq; i<obsall; i=i+1){
/* # Fill in the small additional bit at the end of series, that doesn't contain actuals anymore */
        if((S=='N') || (S=='S')){
          xtnew = matrixVt.row(i-freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixVt.submat(i-1,0,i-1,1);
          xtnew.col(2) = matrixVt.submat(i-freq,2,i-freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixVt.submat(i-1,0,i-1,1);
          xtnew.cols(2,3) = matrixVt.submat(i-freq,2,i-freq,3);
        }
        matrixVt.row(i) = xtnew * trans(matrixF);
      }
      for (int i=obs+freq-1; i>=freq; i=i-1) {
        if((S=='N') || (S=='S')){
          xtnew = matrixVt.row(i+freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixVt.submat(i+1,0,i+1,1);
          xtnew.col(2) = matrixVt.submat(i+freq,2,i+freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixVt.submat(i+1,0,i+1,1);
          xtnew.cols(2,3) = matrixVt.submat(i+freq,2,i+freq,3);
        }
        matyfit.row(i-freq) = rowvecW * trans(xtnew) + wex.row(i-freq) * trans(xtreg.row(i-freq));
        materrors(i-freq) = vecYt(i-freq) - matyfit(i-freq);
        matrixVt.row(i) = xtnew * trans(matrixF) + trans(vecG * materrors.row(i-freq));
      }
      for (int i=freq-1; i>=0; i=i-1) {
        if((S=='N') || (S=='S')){
          xtnew = matrixVt.row(i+freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixVt.submat(i+1,0,i+1,1);
          xtnew.col(2) = matrixVt.submat(i+freq,2,i+freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixVt.submat(i+1,0,i+1,1);
          xtnew.cols(2,3) = matrixVt.submat(i+freq,2,i+freq,3);
        }
        matrixVt.row(i) = xtnew * trans(matrixF);
      }
    }

/* # The final run */
      for (int i=freq; i<obs+freq; i=i+1) {
        if((S=='N') || (S=='S')){
          xtnew = matrixVt.row(i-freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixVt.submat(i-1,0,i-1,1);
          xtnew.col(2) = matrixVt.submat(i-freq,2,i-freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixVt.submat(i-1,0,i-1,1);
          xtnew.cols(2,3) = matrixVt.submat(i-freq,2,i-freq,3);
        }
        matyfit.row(i-freq) = rowvecW * trans(xtnew) + wex.row(i-freq) * trans(xtreg.row(i-freq));
        materrors(i-freq) = vecYt(i-freq) - matyfit(i-freq);
        matrixVt.row(i) = xtnew * trans(matrixF) + trans(vecG * materrors.row(i-freq));
      }
      for(int i=obs+freq; i<obsall; i=i+1){
        if((S=='N') || (S=='S')){
          xtnew = matrixVt.row(i-freq);
        }
        else if(S=='P'){
          xtnew.cols(0,1) = matrixVt.submat(i-1,0,i-1,1);
          xtnew.col(2) = matrixVt.submat(i-freq,2,i-freq,2);
        }
        else if(S=='F'){
          xtnew.cols(0,1) = matrixVt.submat(i-1,0,i-1,1);
          xtnew.cols(2,3) = matrixVt.submat(i-freq,2,i-freq,3);
        }
        matrixVt.row(i) = xtnew * trans(matrixF);
        xtreg.row(i) = xtreg.row(i-freq);
      }

    return List::create(Named("matvt") = matrixVt, Named("yfit") = matyfit,
                        Named("errors") = materrors, Named("xtreg") = xtreg);
}

/* # Wrapper for cesfitter */
// [[Rcpp::export]]
RcppExport SEXP cesfitterwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg,
                           SEXP Stype, SEXP seasfreq, SEXP matwex, SEXP matxtreg) {
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

    char S = as<char>(Stype);
    int freq = as<int>(seasfreq);

    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);

    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return wrap(cesfitter(matrixVt, matrixF, rowvecW, vecYt, vecG, S, freq, wex, xtreg));
}

arma::vec cesforecaster(arma::mat matrixVt,arma::mat matrixF,arma::rowvec rowvecW,
                     int hor,char S,int freq,arma::mat wex,arma::mat xtreg){
    int hh;
    int ncomponents = matrixVt.n_cols;

    arma::mat matxtnew(hor, ncomponents);
    arma::rowvec xtnew(ncomponents);

    arma::vec matyfor(hor, arma::fill::zeros);
/* # Preparation of all the matrices for the mean */
    if(S=='N'){
      matxtnew.submat(0,0,0,1) = matrixVt.submat(matrixVt.n_rows-1,0,matrixVt.n_rows-1,1);
        if(hor > freq){
            for(int i = 1; i < hor; i=i+1){
                matxtnew.row(i) = matxtnew.row(i-1) * arma::trans(matrixF);
            }
        }
    }
    else if(S=='S'){
      hh = std::min(hor,freq);
      matxtnew.submat(0,0,hh-1,1) = matrixVt.submat(0,0,hh-1,1);
        if(hor > freq){
            for(int i = freq; i < hor; i=i+1){
                matxtnew.row(i) = matxtnew.row(i-freq) * arma::trans(matrixF);
            }
        }
    }
    else if(S=='P'){
      matxtnew.submat(0,0,0,1) = matrixVt.submat(matrixVt.n_rows-1,0,matrixVt.n_rows-1,1);
      matxtnew(0,2) = matrixVt(0,2);
      hh = std::min(hor,freq);
      for(int i = 1; i<hh; i=i+1){
        xtnew.cols(0,1) = matxtnew.submat(i-1,0,i-1,1);
        xtnew.col(2) = matrixVt(i,2);
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
      matxtnew.submat(0,0,0,1) = matrixVt.submat(matrixVt.n_rows-1,0,matrixVt.n_rows-1,1);
      matxtnew.submat(0,2,0,3) = matrixVt.submat(0,2,0,3);
      hh = std::min(hor,freq);
      for(int i=1; i<hh; i=i+1){
        xtnew.cols(0,1) = matxtnew.submat(i-1,0,i-1,1);
        xtnew.cols(2,3) = matrixVt.submat(i,2,i,3);
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
        matyfor.row(i) = rowvecW * trans(matxtnew.row(i)) + wex.row(i) * trans(xtreg.row(i));
    }
    return matyfor;
}

// [[Rcpp::export]]
RcppExport SEXP cesforecasterwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP h,
                                    SEXP Stype, SEXP seasfreq, SEXP matwex, SEXP matxtreg) {
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol(), false);
    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);
    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);
    unsigned int hor = as<int>(h);
    char S = as<char>(Stype);
    unsigned int freq = as<int>(seasfreq);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol(), false);

  return wrap(cesforecaster(matrixVt,matrixF,rowvecW,hor,S,freq,wex,xtreg));
}

arma::mat ceserrorer(arma::mat matrixVt,arma::mat matrixF,arma::rowvec rowvecW,arma::vec vecYt,int hor,char S,int freq,arma::mat wex,arma::mat xtreg){
    int obs = vecYt.n_rows;
    int hh;
    arma::mat materrors(obs, hor);

    materrors.fill(NA_REAL);
/* # This part was needed to backcast the errors...
     for(int i=obs+freq; i>obs-hor; i=i-1){
        hh = std::min(hor, i-freq);
        materrors.submat(i-freq-1, 0, i-freq-1, hh-1) = arma::trans(arma::flipud(vecYt.rows(i-freq-hh, i-freq-1)) - cesforecaster(arma::flipud(matrixVt.rows(i,i+freq-1)),matrixF,rowvecW,hh,S,freq,arma::flipud(wex.rows(i-freq-hh, i-freq-1)),xtreg.rows(i-freq-hh, i-freq-1)));
      } */
    for(int i=freq; i<obs+freq; i=i+1){
        hh = std::min(hor, obs+freq-i);
        materrors.submat(i-freq, 0, i-freq, hh-1) = arma::trans(vecYt.rows(i-freq, i-freq+hh-1) - cesforecaster(matrixVt.rows(i-freq,i-1),matrixF,rowvecW,hh,S,freq,wex.rows(i-freq,i-freq+hh-1),xtreg.rows(i-freq,i-freq+hh-1)));
    }
    return materrors;
}

// [[Rcpp::export]]
RcppExport SEXP ceserrorerwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP h, SEXP Stype,
                                 SEXP seasfreq, SEXP matwex, SEXP matxtreg) {
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol(), false);
    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);
    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);
    NumericMatrix yt_n(yt);
    arma::vec vecYt(yt_n.begin(), yt_n.nrow(), false);
    unsigned int hor = as<int>(h);
    char S = as<char>(Stype);
    unsigned int freq = as<int>(seasfreq);
    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);
    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol(), false);

  return wrap(ceserrorer(matrixVt,matrixF,rowvecW,vecYt,hor,S,freq,wex,xtreg));
}

/* # Cost function calculation */
double cesoptimizer(arma::mat matrixVt, arma::mat matrixF, arma::rowvec rowvecW, arma::mat vecYt, arma::vec vecG,
                 int hor, char S, int freq, bool multi, std::string CFtype, double normalize, arma::mat wex, arma::mat xtreg) {
    double CFres = 0;

    List fitting = cesfitter(matrixVt, matrixF, rowvecW, vecYt, vecG, S, freq, wex, xtreg);
    NumericMatrix mxtfromfit = as<NumericMatrix>(fitting["matvt"]);
    matrixVt = as<arma::mat>(mxtfromfit);
    NumericMatrix errorsfromfit = as<NumericMatrix>(fitting["errors"]);
    NumericMatrix mxtregfromfit = as<NumericMatrix>(fitting["xtreg"]);
    xtreg = as<arma::mat>(mxtregfromfit);

    arma::mat materrors;
    arma::rowvec horvec(hor);

    if(multi==true){
        for(int i=0; i<hor; i=i+1){
            horvec(i) = hor - i;
        }
        materrors = ceserrorer(matrixVt,matrixF,rowvecW,vecYt,hor,S,freq,wex,xtreg);
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
RcppExport double cesoptimizerwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt, SEXP vecg, SEXP h,
                                SEXP Stype, SEXP seasfreq, SEXP multisteps, SEXP CFt, SEXP normalizer, SEXP matwex, SEXP matxtreg) {
/*  std::cout << "Function started" << std::endl; */
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
    char S = as<char>(Stype);
    int freq = as<int>(seasfreq);
    bool multi = as<bool>(multisteps);
    std::string CFtype = as<std::string>(CFt);
    double normalize = as<double>(normalizer);

    NumericMatrix mwex(matwex);
    arma::mat wex(mwex.begin(), mwex.nrow(), mwex.ncol(), false);

    NumericMatrix mxtreg(matxtreg);
    arma::mat xtreg(mxtreg.begin(), mxtreg.nrow(), mxtreg.ncol());

    return cesoptimizer(matrixVt,matrixF,rowvecW,vecYt,vecG,hor,S,freq,multi,CFtype,normalize,wex,xtreg);
}
