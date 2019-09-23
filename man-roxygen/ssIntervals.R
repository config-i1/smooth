#' @param interval Type of interval to construct. This can be:
#'
#' \itemize{
#' \item \code{"none"}, aka \code{"n"} - do not produce prediction
#' interval.
#' \item \code{"parametric"}, \code{"p"} - use state-space structure of ETS. In
#' case of mixed models this is done using simulations, which may take longer
#' time than for the pure additive and pure multiplicative models. This type
#' of interval relies on unbiased estimate of in-sample error variance, which
#' divides the sume of squared errors by T-k rather than just T.
#' \item \code{"likelihood"}, \code{"l"} - these are the same as \code{"p"}, but
#' relies on the biased estimate of variance from the likelihood (division by
#' T, not by T-k).
#' \item \code{"semiparametric"}, \code{"sp"} - interval based on covariance
#' matrix of 1 to h steps ahead errors and assumption of normal / log-normal
#' distribution (depending on error type).
#' \item \code{"nonparametric"}, \code{"np"} - interval based on values from a
#' quantile regression on error matrix (see Taylor and Bunn, 1999). The model
#' used in this process is e[j] = a j^b, where j=1,..,h.
#' }
#' The parameter also accepts \code{TRUE} and \code{FALSE}. The former means that
#' parametric interval are constructed, while the latter is equivalent to
#' \code{none}.
#' If the forecasts of the models were combined, then the interval are combined
#' quantile-wise (Lichtendahl et al., 2013).
#' @param level Confidence level. Defines width of prediction interval.
