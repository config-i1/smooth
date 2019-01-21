#' Intermittent State Space
#'
#' Function calculates the probability for intermittent state space model. This
#' is needed in order to forecast intermittent demand using other functions.
#'
#' The function estimates probability of demand occurrence, using one of the ETS
#' state space models.
#'
#' @template ssIntermittentRef
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param data Either numeric vector or time series vector.
#' @param occurrence Type of method used in probability estimation. Can be
#' \code{"none"} - none, \code{"fixed"} - constant probability,
#' \code{"croston"} - estimated using Croston, 1972 method and \code{"TSB"} -
#' Teunter et al., 2011 method., \code{"sba"} - Syntetos-Boylan Approximation
#' for Croston's method (bias correction) discussed in Syntetos and Boylan,
#' 2005, \code{"logistic"} - probability based on logit model.
#' @param ic Information criteria to use in case of model selection.
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, holdout sample of size \code{h} is taken from
#' the end of the data.
#' @param model Type of ETS model used for the estimation. Normally this should
#' be either \code{"ANN"} or \code{"MNN"}.
#' @param persistence Persistence vector. If \code{NULL}, then it is estimated.
#' @param initial Initial vector. If \code{NULL}, then it is estimated.
#' @param initialSeason Initial vector of seasonal components. If \code{NULL},
#' then it is estimated.
#' @param xreg Vector of matrix of exogenous variables, explaining some parts
#' of occurrence variable (probability).
#' @return The object of class "iss" is returned. It contains following list of
#' values:
#'
#' \itemize{
#' \item \code{model} - the type of the estimated ETS model;
#' \item \code{fitted} - fitted values of the constructed model;
#' \item \code{forecast} - forecast for \code{h} observations ahead;
#' \item \code{states} - values of states (currently level only);
#' \item \code{variance} - conditional variance of the forecast;
#' \item \code{logLik} - likelihood value for the model
#' \item \code{nParam} - number of parameters used in the model;
#' \item \code{residuals} - residuals of the model;
#' \item \code{actuals} - actual values of probabilities (zeros and ones).
#' \item \code{persistence} - the vector of smoothing parameters;
#' \item \code{initial} - initial values of the state vector;
#' \item \code{initialSeason} - the matrix of initials seasonal states;
#' }
#' @seealso \code{\link[forecast]{ets}, \link[forecast]{forecast},
#' \link[smooth]{es}}
#' @keywords iss intermittent demand intermittent demand state space model
#' exponential smoothing forecasting
#' @examples
#'
#' y <- rpois(100,0.1)
#' iss2(y, occurrence="g")
#'
#' iss2(y, occurrence="f")
#'
iss2 <- function(data,
                 occurrence=c("general","fixed","odds-ratio","inverse-odds-ratio","probability","auto"),
                 ic=c("AICc","AIC","BIC","BICc"), h=10, holdout=FALSE,
                 model=NULL, persistence=NULL, initial=NULL, initialSeason=NULL, xreg=NULL){
    # Function returns the occurrence part of the intermittent state space model

    occurrence <- occurrence[1];

    # Create a separate C++ function for iss, focused on ETS?

    # Options for the fitter and forecaster:
    # O / P - M / A odds-ratio - "odds-ratio"
    # I / J - M / A inverse-odds-ratio - "inverse-odds-ratio"
    # G / H - M / A general model - "general" <- This should rely on a vector-based model
    # T - Probability based (TSB like) - "probability"
}
