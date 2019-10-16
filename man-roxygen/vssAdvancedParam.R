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
#' @param intermittent Defines type of intermittent model used. Can be:
#' \itemize{
#' \item \code{none}, meaning that the data should be considered as non-intermittent;
#' \item \code{fixed}, taking into account constant Bernoulli distribution of
#' demand occurrences;
#' \item \code{tsb}, based on Teunter et al., 2011 method.
#' \item \code{auto} - automatic selection of intermittency type based on information
#' criteria. The first letter can be used instead.
#' }
#' @param imodel Either character specifying what type of VES / ETS model should be
#' used for probability modelling, or a model estimated using \link[smooth]{viss}
#' function.
#' @param iprobability Type of multivariate probability used in the model. Can be
#' either \code{"independent"} or \code{"dependent"}. In the former case it is
#' assumed that non-zeroes occur in each series independently. In the latter case
#' each possible outcome is treated separately.
