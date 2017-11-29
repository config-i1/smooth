#' @param cfType Type of Cost Function used in optimization. \code{cfType} can
#' be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error),
#' \code{HAM} (Half Absolute Moment), \code{GMSTFE} - Mean Log Squared Trace
#' Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and
#' \code{MSEh} - optimisation using only h-steps ahead error, \code{TFL} -
#' trace forecast likelihood. If \code{cfType!="MSE"}, then likelihood and
#' model selection is done based on equivalent \code{MSE}. Model selection in
#' this cases becomes not optimal.
#'
#' There are also available analytical approximations for multistep functions:
#' \code{aMSEh}, \code{aMSTFE} and \code{aGMSTFE}. These can be useful in cases
#' of small samples.
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word.
#' @param intermittent Defines type of intermittent model used. Can be: 1.
#' \code{none}, meaning that the data should be considered as non-intermittent;
#' 2. \code{fixed}, taking into account constant Bernoulli distribution of
#' demand occurrences; 3. \code{interval}, Interval-based model, underlying
#' Croston, 1972 method; 4. \code{probability}, Probability-based model,
#' underlying Teunter et al., 2011 method. 5. \code{auto} - automatic selection
#' of intermittency type based on information criteria. The first letter can be
#' used instead. 6. \code{"sba"} - Syntetos-Boylan Approximation for Croston's
#' method (bias correction) discussed in Syntetos and Boylan, 2005. 7.
#' \code{"logistic"} - the probability is estimated based on logistic regression
#' model principles.
#' @param imodel Type of ETS model used for the modelling of the time varying
#' probability. Object of the class "iss" can be provided here, and its parameters
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
