#' @param loss The type of Loss Function used in optimization. \code{loss} can
#' be: \code{likelihood} (assuming Normal distribution of error term),
#' \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error),
#' \code{HAM} (Half Absolute Moment), \code{TMSE} - Trace Mean Squared Error,
#' \code{GTMSE} - Geometric Trace Mean Squared Error, \code{MSEh} - optimisation
#' using only h-steps ahead error, \code{MSCE} - Mean Squared Cumulative Error.
#' If \code{loss!="MSE"}, then likelihood and model selection is done based
#' on equivalent \code{MSE}. Model selection in this cases becomes not optimal.
#'
#' There are also available analytical approximations for multistep functions:
#' \code{aMSEh}, \code{aTMSE} and \code{aGTMSE}. These can be useful in cases
#' of small samples.
#'
#' Finally, just for fun the absolute and half analogues of multistep estimators
#' are available: \code{MAEh}, \code{TMAE}, \code{GTMAE}, \code{MACE}, \code{TMAE},
#' \code{HAMh}, \code{THAM}, \code{GTHAM}, \code{CHAM}.
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word.
#' @param xreg The vector (either numeric or time series) or the matrix (or
#' data.frame) of exogenous variables that should be included in the model. If
#' matrix included than columns should contain variables and rows - observations.
#' Note that \code{xreg} should have number of observations equal either to
#' in-sample or to the whole series. If the number of observations in
#' \code{xreg} is equal to in-sample, then values for the holdout sample are
#' produced using \link[smooth]{es} function.
#' @param xregDo The variable defines what to do with the provided xreg:
#' \code{"use"} means that all of the data should be used, while
#' \code{"select"} means that a selection using \code{ic} should be done.
#' \code{"combine"} will be available at some point in future...
#' @param initialX The vector of initial parameters for exogenous variables.
#' Ignored if \code{xreg} is NULL.
