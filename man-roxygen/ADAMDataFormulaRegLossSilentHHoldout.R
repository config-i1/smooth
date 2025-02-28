#' @param data Vector, containing data needed to be forecasted. If a matrix (or
#' data.frame / data.table) is provided, then the first column is used as a
#' response variable, while the rest of the matrix is used as a set of explanatory
#' variables. \code{formula} can be used in the latter case in order to define what
#' relation to have.
#' @param formula Formula to use in case of explanatory variables. If \code{NULL},
#' then all the variables are used as is. Can also include \code{trend}, which would add
#' the global trend. Only needed if \code{data} is a matrix or if \code{trend} is provided.
#' @param regressors The variable defines what to do with the provided explanatory
#' variables:
#' \code{"use"} means that all of the data should be used, while
#' \code{"select"} means that a selection using \code{ic} should be done,
#' \code{"adapt"} will trigger the mechanism of time varying parameters for the
#' explanatory variables.
#' @param loss The type of Loss Function used in optimization. \code{loss} can
#' be:
#' \itemize{
#' \item \code{likelihood} - the model is estimated via the maximisation of the
#' likelihood of the function specified in \code{distribution};
#' \item \code{MSE} (Mean Squared Error),
#' \item \code{MAE} (Mean Absolute Error),
#' \item \code{HAM} (Half Absolute Moment),
#' \item \code{LASSO} - use LASSO to shrink the parameters of the model;
#' \item \code{RIDGE} - use RIDGE to shrink the parameters of the model;
#' \item \code{TMSE} - Trace Mean Squared Error,
#' \item \code{GTMSE} - Geometric Trace Mean Squared Error,
#' \item \code{MSEh} - optimisation using only h-steps ahead error,
#' \item \code{MSCE} - Mean Squared Cumulative Error.
#' }
#' In case of LASSO / RIDGE, the variables are not normalised prior to the estimation,
#' but the parameters are divided by the mean values of explanatory variables.
#'
#' Note that model selection and combination works properly only for the default
#' \code{loss="likelihood"}.
#'
#' Furthermore, just for fun the absolute and half analogues of multistep estimators
#' are available: \code{MAEh}, \code{TMAE}, \code{GTMAE}, \code{MACE},
#' \code{HAMh}, \code{THAM}, \code{GTHAM}, \code{CHAM}.
#'
#' Last but not least, user can provide their own function here as well, making sure
#' that it accepts parameters \code{actual}, \code{fitted} and \code{B}. Here is an
#' example:
#'
#' \code{lossFunction <- function(actual, fitted, B) return(mean(abs(actual-fitted)))}
#'
#' \code{loss=lossFunction}
#' @param silent Specifies, whether to provide the progress of the function or not.
#' If \code{TRUE}, then the function will print what it does and how much it has
#' already done.
#' @param h The forecast horizon. Mainly needed for the multistep loss functions.
#' @param holdout Logical. If \code{TRUE}, then the holdout of the size \code{h}
#' is taken from the data (can be used for the model testing purposes).
