\name{ces.auto}
\alias{ces.auto}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Complex Exponential Smoothing Auto
}
\description{
  Function estimates CES in state-space form with information potential equal to errors with different seasonality types and chooses the one with the lowest IC value.
}
\usage{
ces.auto(data, h=1, holdout=FALSE, C=c(1.1, 1), bounds=FALSE,
    models=c("N","F"), xreg=NULL, intervals=FALSE, int.w=0.95,
    IC=c("CIC","AIC","AICc","BIC"), trace=FALSE, CF.type=c("TLV","TV","GV","hsteps"),
    use.test=FALSE, silent=FALSE, legend=TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
    Either numeric vector or time series vector.
  }
  \item{h}{
    The forecasting horizon.
  }
  \item{holdout}{
    If TRUE, the holdout sample of size h will be taken from the data. If FALSE, no holdout is defined.
  }
  \item{C}{
    The initial value of complex smoothing parameter, where C[1] = a0, C[2] = a1.
  }
  \item{bounds}{
    If TRUE, the smoothing parameter is chosen from the stability region, which makes estimation much slower. If FALSE, no bounds are used, which is fast but may result in non-stable model.
  }
  \item{models}{
    The vector containing several types of seasonality that should be used in CES selection. See \link[smooth]{ces} for more details about the possible types of seasonal models.
  }
  \item{xreg}{
    The vector (either numeric or time series) or matrix (or data.frame) of exogenous variables that should be included in CES. If matrix included than columns should contain variables and rows - observations. xreg should have the number of observations equal either to in-sample or to the whole series. If the number of observations in xreg is equal to in-sample, xreg in the holdout sample is taken to be equal to zero. If data is ts object then xreg also must be ts.
  }
  \item{intervals}{
    If TRUE, the prediction intervals are constructed.
  }
  \item{int.w}{
    The width of prediction intervals. Default is 95\%.
  }
  \item{IC}{
    The information criterion used in the model selection procedure. CIC - "Complex IC" is prefered due to the features of complex components etimation.
  }
  \item{trace}{
    If TRUE, the likelihood with h-steps ahead error is used.
  }
  \item{CF.type}{
    Type of Cost Function used in trace optimization. Can be "GV" - Generalized Variance, "TLV" - Total Logarythmic Variation and "TV" - Total Variation, "hsteps" - optimisation using only h-steps ahead error.
  }
  \item{use.test}{
    If TRUE the statistical test is used on the imaginary part of the complex smoothing parameter to define if it is significantly different from 1.
  }
  \item{silent}{
    If TRUE, no output is given and no plot is drawn.
  }
  \item{legend}{
    If FALSE, the legend in the graph is skiped.
  }
}
\details{
  The function estimates several Complex Exponential Smoothing in the state-space 2 described in Svetunkov, Kourentzes (2015) with the information potential equal to the approximation error using different types of seasonality and chooses the one with the lowest value of information criterion.
}
\value{
  The list of the returned values is the same as in \link[smooth]{ces} function:

  \itemize{
    \item \code{A} - complex smoothing parameter in the form a0 + ia1
    \item \code{B} - smoothing parameter for the seasonal component. Can either be real (if \code{seasonality="P"}) or complex (if \code{seasonality="F"}) in a form b0 + ib1.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - The matrix of 1 to h steps ahead errors.
    \item \code{actuals} - The provided data in the call of the function.
    \item \code{fitted} - the fitted values of CES.
    \item \code{forecast} - the point forecast of CES.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{upper} - the upper bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{states} - the matrix of the components of CES. The included minimum is "level" and "potential". In the case of seasonal model the seasonal component is also included. In the case of exogenous variables the estimated coefficients for the exogenous variables are also included.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc, BIC and CIC (Complex IC).
    \item \code{FI} - Fisher Information.
    \item \code{accuracy} - the vector or accuracy measures. Includes MAPE, MASE and MASALE. Available only when \code{holdout=T}.
  }
}
\references{
    \itemize{
        \item{Svetunkov, I., Kourentzes, N. (February 2015). Complex exponential smoothing. Working Paper of Department of Management Science, Lancaster University 2015:1, 1-31.}
        \item{Svetunkov I., Kourentzes N. (2015) Complex Exponential Smoothing for Time Series Forecasting. Not yet published.}
        \item{Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.}
        \item{Svetunkov S. (2012) Complex-Valued Modeling in Economics and Finance. SpringerLink: Bucher. Springer.}
    }
}
\author{
  Ivan Svetunkov, \email{ivan@svetunkov.ru}
}
\seealso{
\code{\link[smooth]{ces}, \link[forecast]{ets}, \link[forecast]{forecast}, \link[stats]{ts}}
}
\examples{
y <- rnorm(100,10,3)
y <- ts(y,frequency=12)
y_ces_holdout <- ces.auto(y,h=20,holdout=TRUE)
y_ces_no_holdout <- ces.auto(y,h=20,holdout=FALSE)

library("Mcomp")
y <- ts(c(M3$N0740$x,M3$N0740$xx),start=start(M3$N0740$x),frequency=frequency(M3$N0740$x))
N0740_ces <- ces.auto(y,h=8,holdout=TRUE,models=c("N","P"),intervals=TRUE,int.w=0.8,IC="AIC")

y <- ts(c(M3$N1683$x,M3$N1683$xx),start=start(M3$N1683$x),frequency=frequency(M3$N1683$x))
N1683_ces_S <- ces.auto(y,h=18,holdout=TRUE,intervals=TRUE)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ces }
\keyword{ complex exponential smoothing }
\keyword{ exponential smoothing }
\keyword{ forecasting }
\keyword{ complex variables }
