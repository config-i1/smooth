#' @param cfType Type of Cost Function used in optimization. \code{cfType} can
#' be:
#' \itemize{
#' \item \code{likelihood} - which assumes the minimisation of the determinant
#' of the covariance matrix of errors between the series. This implies that the
#' series could be correlated;
#' \item \code{diagonal} - the covariance matrix is assumed to be diagonal with
#' zeroes off the diagonal. The determinant of this matrix is just a product of
#' variances. This thing is minimised in this siduation in logs.
#' \item \code{trace} - the trace of the covariance matrix. The sum of variances
#' is minimised in this case.
#' }
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word. Currently only \code{"admissible"}
#' bounds are available.
#' @param intermittent Defines type of intermittent model used. Can be:
#' \itemize{
#' \item \code{none}, meaning that the data should be considered as non-intermittent;
#' \item \code{fixed}, taking into account constant Bernoulli distribution of
#' demand occurancies;
#' \item \code{tsb}, based on Teunter et al., 2011 method.
#' \item \code{auto} - automatic selection of intermittency type based on information
#' criteria. The first letter can be used instead.
#' }
