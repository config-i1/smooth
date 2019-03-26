#' @param cfType Type of Cost Function used in optimization. \code{cfType} can
#' be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error),
#' \code{HAM} (Half Absolute Moment), \code{TMSE} - Trace Mean Squared Error,
#' \code{GTMSE} - Geometric Trace Mean Squared Error, \code{MSEh} - optimisation
#' using only h-steps ahead error, \code{MSCE} - Mean Squared Cumulative Error.
#' If \code{cfType!="MSE"}, then likelihood and model selection is done based
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
#' @param occurrence Type of model used in probability estimation. Can be
#' \code{"none"} - none,
#' \code{"fixed"} - constant probability,
#' \code{"general"} - the general Beta model with two parameters,
#' \code{"odds-ratio"} - the Odds-ratio model with b=1 in Beta distribution,
#' \code{"inverse-odds-ratio"} - the model with a=1 in Beta distribution,
#' \code{"probability"} - the TSB-like (Teunter et al., 2011) probability update
#' mechanism a+b=1,
#' \code{"auto"} - the automatically selected type of occurrence model.
#' @param imodel Type of ETS model used for the modelling of the time varying
#' probability. Object of the class "oes" can be provided here, and its parameters
#' would be used in iETS model.
#' @param xreg Vector (either numeric or time series) or matrix (or data.frame)
#' of exogenous variables that should be included in the model. If matrix
#' included than columns should contain variables and rows - observations. Note
#' that \code{xreg} should have number of observations equal either to
#' in-sample or to the whole series. If the number of observations in
#' \code{xreg} is equal to in-sample, then values for the holdout sample are
#' produced using \link[smooth]{es} function.
#' @param xregDo Variable defines what to do with the provided xreg:
#' \code{"use"} means that all of the data should be used, while
#' \code{"select"} means that a selection using \code{ic} should be done.
#' \code{"combine"} will be available at some point in future...
#' @param initialX Vector of initial parameters for exogenous variables.
#' Ignored if \code{xreg} is NULL.
#' @param updateX If \code{TRUE}, transition matrix for exogenous variables is
#' estimated, introducing non-linear interactions between parameters.
#' Prerequisite - non-NULL \code{xreg}.
#' @param persistenceX Persistence vector \eqn{g_X}, containing smoothing
#' parameters for exogenous variables. If \code{NULL}, then estimated.
#' Prerequisite - non-NULL \code{xreg}.
#' @param transitionX Transition matrix \eqn{F_x} for exogenous variables. Can
#' be provided as a vector. Matrix will be formed using the default
#' \code{matrix(transition,nc,nc)}, where \code{nc} is number of components in
#' state vector. If \code{NULL}, then estimated. Prerequisite - non-NULL
#' \code{xreg}.
