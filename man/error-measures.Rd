\name{Error measures}
\alias{Error measures}
\alias{MPE}
\alias{MAPE}
\alias{SMAPE}
\alias{MASE}
\alias{RelMAE}
\alias{sMSE}

\title{
    Error measures
}
\description{
    Functions allow to calculate different types of errors:
    \enumerate{
        \item{MPE - Mean Percentage Error,}
        \item{MAPE - Mean Absolute Percentage Error,}
        \item{SMAPE - Symmetric Mean Absolute Percentage Error,}
        \item{MASE - Mean Absolute Scaled Error,}
        \item{RelMAE - Average Rellative Mean Absolute Error,}
        \item{sMSE - Scaled Mean Squared Error.}
    }

    In case of \code{sMSE}, \code{scale} needs to be a squared value. Typical one -- squared mean value of in-sample actuals.
}
\usage{
    MPE(actual,forecast,digits=3)
    MAPE(actual,forecast,digits=3)
    SMAPE(actual,forecast,digits=3)
    MASE(actual,forecast,scale,digits=3)
    RelMAE(actual,forecast,benchmark,digits=3)
    sMSE(actual,forecast,scale,digits=3)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{actual}{
        The vector or matrix of actual values.
  }
  \item{forecast}{
        The vector or matrix of forecasts values.
  }
  \item{scale}{
        The value that should be used in the denominator of MASE. Can be anything but advised values are: mean absolute deviation of in-sample one step ahead Naive error or mean absolute value of the in-sample actuals.
  }
  \item{benchmark}{
        The vector or matrix of the forecasts of the benchmark model.
  }
  \item{digits}{
        Number of digits of the output.
  }
}
\value{
    All the functions return the scalar value.
}
\references{
    \itemize{
        \item{
            Fildes, R. (1992). The evaluation of extrapolative forecasting methods. International Journal of Forecasting, 8, pp.81-98.
        }
        \item{
            Makridakis, S. (1993). Accuracy measures: Theoretical and practical concerns. International Journal of Forecasting, 9, pp.527-529.
        }
        \item{
            Hyndman R.J., Koehler A.B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22, pp.679-688.
        }
    }
}
\author{
    Ivan Svetunkov
}
\examples{

y <- rnorm(100,10,2)
esmodel <- es(y[1:90],model="ANN",h=10)

MPE(y[91:100],esmodel$forecast,digits=5)
MAPE(y[91:100],esmodel$forecast,digits=5)
SMAPE(y[91:100],esmodel$forecast,digits=5)
MASE(y[91:100],esmodel$forecast,mean(abs(y[1:90])),digits=5)
MASE(y[91:100],esmodel$forecast,mean(abs(diff(y[1:90]))),digits=5)

esmodel2 <- es(y[1:90],model="AAN",h=10)
RelMAE(y[91:100],esmodel2$forecast,esmodel$forecast,digits=5)

MASE(y[91:100],esmodel$forecast,mean(abs(y[1:90]))^2,digits=5)
}
\keyword{ error measures }
\keyword{ forecasting }
