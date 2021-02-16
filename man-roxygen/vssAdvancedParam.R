#' @param loss Type of Loss Function used in optimization. \code{loss} can
#' be:
#' \itemize{
#' \item \code{likelihood} - which assumes the minimisation of the determinant
#' of the covariance matrix of errors between the series. This implies that the
#' series could be correlated;
#' \item \code{diagonal} - the covariance matrix is assumed to be diagonal with
#' zeros off the diagonal. The determinant of this matrix is just a product of
#' variances. This thing is minimised in this situation in logs.
#' \item \code{trace} - the trace of the covariance matrix. The sum of variances
#' is minimised in this case.
#' }
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word. \code{"admissible"} means that the
#' model stability is ensured, while \code{"usual"} means that the all the parameters
#' are restricted by the (0, 1) region.
#' @param occurrence Defines type of occurrence model used. Can be:
#' \itemize{
#' \item \code{none}, meaning that the data should be considered as non-intermittent;
#' \item \code{fixed}, taking into account constant Bernoulli distribution of
#' demand occurrences;
#' \item \code{logistic}, based on logistic regression.
#' }
#' In this case, the ETS model inside the occurrence part will correspond to
#' \code{model} and \code{probability="dependent"}.
#' Alternatively, model estimated using \link[smooth]{viss} function can be provided
#' here.
