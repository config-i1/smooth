#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "headers/ssGeneral.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/*
# polysos - function that transforms AR and MA parameters into polynomials
# and then in matF and other things.
# Cvalues includes AR, MA, initials, constant, matrixAt, transitionX and persistenceX.
# C and constValue can be NULL, so pointer is not suitable here.
*/
List polysos(arma::uvec const &arOrders, arma::uvec const &maOrders, arma::uvec const &iOrders, arma::uvec const &lags,
             unsigned int const &nComponents,
             arma::vec const &arValues, arma::vec const &maValues, double const constValue, arma::vec const C,
             arma::mat &matrixVt, arma::vec &vecG, arma::mat &matrixF,
             char const &fitterType, int const &nexo, arma::mat &matrixAt, arma::mat &matrixFX, arma::vec &vecGX,
             bool const &arEstimate, bool const &maEstimate, bool const &constRequired, bool const &constEstimate,
             bool const &xregEstimate, bool const &wild, bool const &fXEstimate, bool const &gXEstimate, bool const &initialXEstimate,
             bool const &arimaOld, arma::uvec const &lagsModel, arma::umat const &ARILags, arma::umat const &MALags){

// Form matrices with parameters, that are then used for polynomial multiplication
    arma::mat arParameters(max(arOrders % lags)+1, arOrders.n_elem, arma::fill::zeros);
    arma::mat iParameters(max(iOrders % lags)+1, iOrders.n_elem, arma::fill::zeros);
    arma::mat maParameters(max(maOrders % lags)+1, maOrders.n_elem, arma::fill::zeros);

    arParameters.row(0).fill(1);
    iParameters.row(0).fill(1);
    maParameters.row(0).fill(1);

    int lagsModelMax = max(lagsModel);

    int nParam = 0;
    int arnParam = 0;
    int manParam = 0;
    for(unsigned int i=0; i<lags.n_rows; ++i){
        if(arOrders(i) * lags(i) != 0){
            for(unsigned int j=0; j<arOrders(i); ++j){
                if(arEstimate){
                    arParameters((j+1)*lags(i),i) = -C(nParam);
                    nParam += 1;
                }
                else{
                    arParameters((j+1)*lags(i),i) = -arValues(arnParam);
                    arnParam += 1;
                }
            }
        }

        if(iOrders(i) * lags(i) != 0){
            iParameters(lags(i),i) = -1;
        }

        if(maOrders(i) * lags(i) != 0){
            for(unsigned int j=0; j<maOrders(i); ++j){
                if(maEstimate){
                    maParameters((j+1)*lags(i),i) = C(nParam);
                    nParam += 1;
                }
                else{
                    maParameters((j+1)*lags(i),i) = maValues(manParam);
                    manParam += 1;
                }
            }
        }
    }

// Prepare vectors with coefficients for polynomials
    arma::vec arPolynomial(sum(arOrders % lags)+1, arma::fill::zeros);
    arma::vec iPolynomial(sum(iOrders % lags)+1, arma::fill::zeros);
    arma::vec maPolynomial(sum(maOrders % lags)+1, arma::fill::zeros);
    arma::vec ariPolynomial(sum(arOrders % lags)+sum(iOrders % lags)+1, arma::fill::zeros);
    arma::vec buferPolynomial;

    arPolynomial.rows(0,arOrders(0)*lags(0)) = arParameters.submat(0,0,arOrders(0)*lags(0),0);
    iPolynomial.rows(0,iOrders(0)*lags(0)) = iParameters.submat(0,0,iOrders(0)*lags(0),0);
    maPolynomial.rows(0,maOrders(0)*lags(0)) = maParameters.submat(0,0,maOrders(0)*lags(0),0);

    for(unsigned int i=0; i<lags.n_rows; ++i){
// Form polynomials
        if(i!=0){
            buferPolynomial = polyMult(arPolynomial, arParameters.col(i));
            arPolynomial.rows(0,buferPolynomial.n_rows-1) = buferPolynomial;

            buferPolynomial = polyMult(maPolynomial, maParameters.col(i));
            maPolynomial.rows(0,buferPolynomial.n_rows-1) = buferPolynomial;

            buferPolynomial = polyMult(iPolynomial, iParameters.col(i));
            iPolynomial.rows(0,buferPolynomial.n_rows-1) = buferPolynomial;
        }
        if(iOrders(i)>1){
            for(unsigned int j=1; j<iOrders(i); ++j){
                buferPolynomial = polyMult(iPolynomial, iParameters.col(i));
                iPolynomial.rows(0,buferPolynomial.n_rows-1) = buferPolynomial;
            }
        }

    }
    // ariPolynomial contains 1 in the first place
    ariPolynomial = polyMult(arPolynomial, iPolynomial);

    if(arimaOld){
        maPolynomial.resize(nComponents+1);
        ariPolynomial.resize(nComponents+1);
        arPolynomial.resize(nComponents+1);
    }
    else{
        // R dies without this resize... Weird!
        maPolynomial.resize(sum(maOrders % lags)+1);
        ariPolynomial.resize(sum(arOrders % lags)+sum(iOrders % lags)+1);
        arPolynomial.resize(sum(arOrders % lags)+1);
    }
    int ariPolynomialSize = ariPolynomial.n_elem;

// Fill in transition matrix
    if(ariPolynomialSize>1){
        if(arimaOld){
            matrixF.submat(0,0,ariPolynomialSize-2,0) = -ariPolynomial.rows(1,ariPolynomialSize-1);
        }
        else{
            // This thing does not take into account the possibility of q > p
            arma::rowvec unitVector(matrixF.n_cols, arma::fill::ones);
            if(constRequired){
                unitVector(unitVector.n_elem-1) = 0;
            }
            matrixF.rows(ARILags.col(1)) = -ariPolynomial(ARILags.col(0)) * unitVector;
            if(constRequired){
                matrixF.col(matrixF.n_cols-1).fill(1);
            }
        }
    }

// Fill in persistence vector
    if(nComponents>0){
        if(arimaOld){
            vecG.rows(0,ariPolynomialSize-2) = -ariPolynomial.rows(1,ariPolynomialSize-1) + maPolynomial.rows(1,maPolynomial.n_elem-1);
        }
        else{
            vecG.fill(0);
            if(ARILags.n_rows>0){
                vecG(ARILags.col(1)) = -ariPolynomial(ARILags.col(0));
            }
            if(MALags.n_rows>0){
                vecG(MALags.col(1)) += maPolynomial(MALags.col(0));
            }
        }

// Fill in initials of state vector
        if(fitterType=='o'){
            if(arimaOld){
                matrixVt.submat(0,0,0,nComponents-1) = C.rows(nParam,nParam+nComponents-1).t();
                nParam += nComponents;
            }
            else{
                matrixVt.submat(0,0,lagsModelMax-1,nComponents-1).fill(0);
                if(nComponents>0){
                    // If the length of ARI polynomials larger than MA, use them
                    if(ARILags.n_rows>0 && (ARILags.n_rows>=MALags.n_rows)){
                        // std::cout << C.rows(nParam,nParam+lagsModelMax-1);
                        // std::cout << " ; ";
                        // Fill in the values based on the ARI values. MA stays zero
                        matrixVt.submat(0,0,lagsModelMax-1,nComponents-1) = C.rows(nParam,nParam+lagsModelMax-1) *
                            ariPolynomial(ARILags.col(0)).t() / arma::as_scalar(ariPolynomial(ARILags(nComponents-1,0)));
                        nParam += lagsModelMax;
                    }
                    // Otherwise use MA polynomials for the initialisation
                    else if(MALags.n_rows>0 && (ARILags.n_rows<MALags.n_rows)){
                        matrixVt.submat(0,0,lagsModelMax-1,nComponents-1) = C.rows(nParam,nParam+lagsModelMax-1) *
                            maPolynomial(MALags.col(0)).t() / arma::as_scalar(maPolynomial(MALags(nComponents-1,0)));
                        nParam += lagsModelMax;
                    }
                }
            }
        }
        else if(fitterType=='b'){
            if(arimaOld){
                for(unsigned int i=1; i < nComponents; i=i+1){
                    matrixVt.submat(0,i,nComponents-i-1,i) = matrixVt.submat(1,i-1,nComponents-i,i-1) -
                        matrixVt.submat(0,0,nComponents-i-1,0) * matrixF.submat(i-1,0,i-1,0);
                }
            }
            // else{
            //     for(unsigned int i=0; i < nComponents; i=i+1){
            //         matrixVt.submat(0,i,nComponents-i,i) = matrixVt.submat(0,i,nComponents-i,i) * matrixF.submat(i,0,i,0);
            //     }
            // }
        }
    }
    // std::cout << "test";

// Deal with constant if needed
    if(constRequired){
        if(arimaOld){
            if(constEstimate){
                matrixVt(0,matrixVt.n_cols-1) = C(nParam);
                nParam += 1;
            }
            else{
                matrixVt(0,matrixVt.n_cols-1) = constValue;
            }
        }
        else{
            if(constEstimate){
                matrixVt.col(matrixVt.n_cols-1).fill(C(nParam));
                nParam += 1;
            }
            else{
                matrixVt.col(matrixVt.n_cols-1).fill(constValue);
            }
        }
    }

    if(xregEstimate){
        if(initialXEstimate){
            matrixAt.each_row() = C.rows(nParam,nParam + nexo - 1).t();
            nParam += nexo;
        }

        if(wild){
            if(fXEstimate){
                for(int i=0; i < nexo; i = i+1){
                    matrixFX.row(i) = C.rows(nParam, nParam + nexo - 1).t();
                    nParam += nexo;
                }
            }

            if(gXEstimate){
                vecGX = C.rows(nParam, nParam + nexo - 1);
                nParam += nexo;
            }
        }
    }

    return List::create(Named("matF") = matrixF, Named("vecg") = vecG,
                        Named("arPolynomial") = arPolynomial, Named("maPolynomial") = maPolynomial,
                        Named("matvt") = matrixVt, Named("matat") = matrixAt,
                        Named("matFX") = matrixFX, Named("vecgX") = vecGX);
}

// [[Rcpp::export]]
RcppExport SEXP polysoswrap(SEXP ARorders, SEXP MAorders, SEXP Iorders, SEXP ARIMAlags, SEXP nComp,
                            SEXP AR, SEXP MA, SEXP constant, SEXP Cvalues,
                            SEXP matvt, SEXP vecg, SEXP matF,
                            SEXP fittertype, SEXP nexovars, SEXP matat, SEXP matFX, SEXP vecgX,
                            SEXP estimAR, SEXP estimMA, SEXP requireConst, SEXP estimConst,
                            SEXP estimxreg, SEXP gowild, SEXP estimFX, SEXP estimgX, SEXP estiminitX,
                            SEXP ssarimaOld, SEXP lagsModelR, SEXP nonZeroARI, SEXP nonZeroMA){

    IntegerVector ARorders_n(ARorders);
    arma::uvec arOrders = as<arma::uvec>(ARorders_n);

    IntegerVector MAorders_n(MAorders);
    arma::uvec maOrders = as<arma::uvec>(MAorders_n);

    IntegerVector Iorders_n(Iorders);
    arma::uvec iOrders = as<arma::uvec>(Iorders_n);

    IntegerVector ARIMAlags_n(ARIMAlags);
    arma::uvec lags = as<arma::uvec>(ARIMAlags_n);

    unsigned int nComponents = as<int>(nComp);

    NumericVector AR_n;
    if(!Rf_isNull(AR)){
        AR_n = as<NumericVector>(AR);
    }
    arma::vec arValues(AR_n.begin(), AR_n.size(), false);

    NumericVector MA_n;
    if(!Rf_isNull(MA)){
        MA_n = as<NumericVector>(MA);
    }
    arma::vec maValues(MA_n.begin(), MA_n.size(), false);

    double constValue = 0;
    if(!Rf_isNull(constant)){
        constValue = as<double>(constant);
    }

    NumericVector Cvalues_n;
    if(!Rf_isNull(Cvalues)){
        Cvalues_n = as<NumericVector>(Cvalues);
    }
    arma::vec C(Cvalues_n.begin(), Cvalues_n.size(), false);

    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix vecg_n(vecg);
    arma::vec vecG(vecg_n.begin(), vecg_n.nrow());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol());

    char fitterType = as<char>(fittertype);

    int nexo = as<int>(nexovars);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol());

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow());

    bool arEstimate = as<bool>(estimAR);
    bool maEstimate = as<bool>(estimMA);
    bool constRequired = as<bool>(requireConst);
    bool constEstimate = as<bool>(estimConst);
    bool xregEstimate = as<bool>(estimxreg);
    bool wild = as<bool>(gowild);
    bool fXEstimate = as<bool>(estimFX);
    bool gXEstimate = as<bool>(estimgX);
    bool initialXEstimate = as<bool>(estiminitX);

    bool arimaOld = as<bool>(ssarimaOld);

    IntegerVector lagsModel_n(lagsModelR);
    arma::uvec lagsModel = as<arma::uvec>(lagsModel_n);

    // Create two uvec objects instead of umat?
    IntegerMatrix nonZeroARI_n(nonZeroARI);
    arma::umat ARILags = as<arma::umat>(nonZeroARI_n);

    IntegerMatrix nonZeroMA_n(nonZeroMA);
    arma::umat MALags = as<arma::umat>(nonZeroMA_n);

    return wrap(polysos(arOrders, maOrders, iOrders, lags, nComponents,
                        arValues, maValues, constValue, C,
                        matrixVt, vecG, matrixF,
                        fitterType, nexo, matrixAt, matrixFX, vecGX,
                        arEstimate, maEstimate, constRequired, constEstimate,
                        xregEstimate, wild, fXEstimate, gXEstimate, initialXEstimate,
                        arimaOld, lagsModel, ARILags, MALags));
}

/* # Function produces the point forecasts for the specified model */
arma::mat forecaster(arma::mat matrixVt, arma::mat const &matrixF, arma::rowvec const &rowvecW,
                     unsigned int const &hor, char const &E, char const &T, char const &S, arma::uvec lags,
                     arma::mat matrixXt, arma::mat matrixAt, arma::mat const &matrixFX){
    int lagslength = lags.n_rows;
    unsigned int lagsModelMax = max(lags);
    unsigned int hh = hor + lagsModelMax;

    arma::uvec lagrows(lagslength, arma::fill::zeros);
    arma::vec matyfor(hor, arma::fill::zeros);
    arma::mat matrixVtnew(hh, matrixVt.n_cols, arma::fill::zeros);
    arma::mat matrixAtnew(hh, matrixAt.n_cols, arma::fill::zeros);

    lags = lagsModelMax - lags;
    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + hh * i;
    }

    matrixVtnew.submat(0,0,lagsModelMax-1,matrixVtnew.n_cols-1) = matrixVt.submat(0,0,lagsModelMax-1,matrixVtnew.n_cols-1);
    matrixAtnew.submat(0,0,lagsModelMax-1,matrixAtnew.n_cols-1) = matrixAtnew.submat(0,0,lagsModelMax-1,matrixAtnew.n_cols-1);

/* # Fill in the new xt matrix using F. Do the forecasts. */
    for (unsigned int i=lagsModelMax; i<(hor+lagsModelMax); i=i+1) {
        lagrows = lags - lagsModelMax + i;
        matrixVtnew.row(i) = arma::trans(fvalue(matrixVtnew(lagrows), matrixF, T, S));
        matrixAtnew.row(i) = matrixAtnew.row(i-1) * matrixFX;

        matyfor.row(i-lagsModelMax) = (wvalue(matrixVtnew(lagrows), rowvecW, E, T, S, matrixXt.row(i-lagsModelMax), trans(matrixAt.row(i-lagsModelMax))));
    }

    return matyfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP forecasterwrap(SEXP matvt, SEXP matF, SEXP matw,
                               SEXP h, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP lagsModel,
                               SEXP matxt, SEXP matat, SEXP matFX){
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol(), false);

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::mat rowvecW(matw_n.begin(), matw_n.nrow(), matw_n.ncol(), false);

    unsigned int hor = as<int>(h);
    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector lagsModel_n(lagsModel);
    arma::uvec lags = as<arma::uvec>(lagsModel_n);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    return wrap(forecaster(matrixVt, matrixF, rowvecW, hor, E, T, S, lags, matrixXt, matrixAt, matrixFX));
}

/* # Function produces matrix of errors based on multisteps forecast */
arma::mat errorer(arma::mat const &matrixVt, arma::mat const &matrixF, arma::mat const &rowvecW, arma::mat const &vecYt,
                  int const &hor, char const &E, char const &T, char const &S, arma::uvec const &lags,
                  arma::mat const &matrixXt, arma::mat const &matrixAt, arma::mat const &matrixFX, arma::vec const &vecOt){
    int obs = vecYt.n_rows;
    // This is needed for cases, when hor>obs
    int hh = 0;
    arma::mat matErrors(obs, hor, arma::fill::zeros);
    unsigned int lagsModelMax = max(lags);

    for(int i = 0; i < (obs-hor); i=i+1){
        hh = std::min(hor, obs-i);
        matErrors.submat(i, 0, i, hh-1) = arma::trans(vecOt.rows(i, i+hh-1) % errorvf(vecYt.rows(i, i+hh-1),
            forecaster(matrixVt.rows(i,i+lagsModelMax-1), matrixF, rowvecW, hh, E, T, S, lags, matrixXt.rows(i, i+hh-1),
                matrixAt.rows(i, i+hh-1), matrixFX), E));
    }

    // Cut-off the redundant last part
    if(obs>hor){
        matErrors = matErrors.rows(0,obs-hor-1);
    }

// Fix for GV in order to perform better in the sides of the series
    // for(int i=0; i<(hor-1); i=i+1){
    //     matErrors.submat((hor-2)-(i),i+1,(hor-2)-(i),hor-1) = matErrors.submat(hor-1,0,hor-1,hor-i-2) * sqrt(1.0+i);
    // }

    return matErrors;
}

/* # Wrapper for errorer */
// [[Rcpp::export]]
RcppExport SEXP errorerwrap(SEXP matvt, SEXP matF, SEXP matw, SEXP yt,
                            SEXP h, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP lagsModel,
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

    IntegerVector lagsModel_n(lagsModel);
    arma::uvec lags = as<arma::uvec>(lagsModel_n);

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
