#' @param y Vector or ts object, containing data needed to be forecasted.
#' @param h Length of forecasting horizon.
#' @param holdout If \code{TRUE}, holdout sample of size \code{h} is taken from
#' the end of the data.
#' @param ic The information criterion used in the model selection procedure.
#' @param silent accepts \code{TRUE} and \code{FALSE}. If FALSE, the function
#' will print its progress and produce a plot at the end.
