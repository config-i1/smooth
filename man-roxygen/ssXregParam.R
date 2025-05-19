#' @param xreg The vector (either numeric or time series) or the matrix (or
#' data.frame) of exogenous variables that should be included in the model. If
#' matrix included than columns should contain variables and rows - observations.
#' Note that \code{xreg} should have number of observations equal either to
#' in-sample or to the whole series. If the number of observations in
#' \code{xreg} is equal to in-sample, then values for the holdout sample are
#' produced using \link[smooth]{es} function.
#' @param regressors The variable defines what to do with the provided xreg:
#' \code{"use"} means that all of the data should be used, while
#' \code{"select"} means that a selection using \code{ic} should be done.
#' @param initialX The vector of initial parameters for exogenous variables.
#' Ignored if \code{xreg} is NULL.
