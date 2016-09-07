\name{forecast.smooth}
\alias{forecast}
\alias{forecast.smooth}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Forecasting time series using smooth functions
}
\description{
    This function is created in order for the package to be compatible with Rob Hyndman's "forecast" package
}
\usage{
    forecast(object, ...)
    \method{forecast}{smooth}(object, h=10, intervals=TRUE,
      intervalsType=c("parametric","semiparametric","nonparametric","asymmetric"),
      level=0.95, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

    \item{object}{
        Time series model for which forecasts are required.
    }
    \item{h}{
        Forecast horizon
    }
    \item{intervals}{
        If \code{TRUE}, prediction intervals are constructed.
    }
    \item{intervalsType}{
        Type of intervals to construct. See \link[smooth]{es} for details.
    }
    \item{level}{
        Confidence level. Defines width of prediction interval.
    }
    \item{...}{
        Other arguments accepted by either \link[smooth]{es}, \link[smooth]{ces}, \link[smooth]{ges} or \link[smooth]{ssarima}.
    }
}
\details{
    This is not a compulsary function. You can simply use \link[smooth]{es}, \link[smooth]{ces}, \link[smooth]{ges} or \link[smooth]{ssarima} without \code{forecast.smooth}. But if you are really used to \code{forecast} function, then go ahead!
}
\value{
    Returns object of class "forecastSmooth", which contains:

    \itemize{
        \item \code{model} - estimated model (ES / CES / GES / SSARIMA).
        \item \code{fitted} - fitted values of the model.
        \item \code{actuals} - actuals provided in the call of the model.
        \item \code{forecast} aka \code{mean} - point forecasts of the model (conditional mean).
        \item \code{lower} - lower bound of prediction intervals.
        \item \code{upper} - upper bound of prediction intervals.
        \item \code{level} - confidence level.
        \item \code{intervals} - binary variable (whether intervals were produced or not).
    }
}
\references{
        Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.
}
\author{
    Ivan Svetunkov, \email{ivan@svetunkov.ru}
}
\seealso{
    \code{\link[forecast]{ets}, \link[forecast]{forecast}}
}
\examples{
test <- ces(rnorm(100,0,1),h=10)

forecast.smooth(test,h=10)
forecast(test,h=10,intervals=TRUE)
plot(forecast(test,h=10,intervals=TRUE))

}
\keyword{ forecast }
