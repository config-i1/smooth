#' @param interval Type of interval to construct.
#'
#' This can be:
#'
#' \itemize{
#' \item \code{"none"}, aka \code{"n"} - do not produce prediction
#' interval.
#' \item \code{"conditional"}, \code{"c"} - produces multidimensional elliptic
#' interval for each step ahead forecast. NOT AVAILABLE YET!
#' \item \code{"unconditional"}, \code{"u"} - produces separate bounds for each series
#' based on ellipses for each step ahead. These bounds correspond to min and max
#' values of the ellipse assuming that all the other series but one take values in
#' the centre of the ellipse. This leads to less accurate estimates of bounds
#' (wider interval than needed), but these could still be useful. NOT AVAILABLE YET!
#' \item \code{"independent"}, \code{"i"} - produces interval based on variances of
#' each separate series. This does not take vector structure into account. In the
#' calculation of covariance matrix, the division is done by T-k rather than T.
#' \item \code{"likelihood"}, \code{"l"} - produces \code{"individual"} interval with
#' the variance matrix estimated from the likelihood, which is a biased estimate of
#' the true matrix. This means that the division of sum of squares is done by T
#' rather than T-k.
#' }
#' The parameter also accepts \code{TRUE} and \code{FALSE}. The former means that
#' the independent interval are constructed, while the latter is equivalent to
#' \code{none}.
#' You can also use the first letter instead of writing the full word.
#' @param level Confidence level. Defines width of prediction interval.
