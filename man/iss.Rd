\name{iss}
\alias{iss}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Intermittent State Space
}
\description{
  Functin calculates the probability for intermittent state space model. This is needed in order to forecast intermittent demand using other functions.
}
\usage{
  iss(data, intermittent=c("none","fixed","croston","tsb","sba"),
      h=10, holdout=FALSE, model=NULL, persistence=NULL)
}
\arguments{
  \item{data}{
    Either numeric vector or time series vector.
  }
  \item{intermittent}{
    Type of method used in probability estimation. Can be \code{"none"} - none, \code{"fixed"} - constant probability, \code{"croston"} - estimated using Croston, 1972 method and \code{"TSB"} - Teunter et al., 2011 method., \code{"sba"} - Syntetos-Boylan Approximation for Croston's method (bias correction) discussed in Syntetos and Boylan, 2005.
  }
  \item{h}{
    Forecast horizon.
  }
  \item{holdout}{
    If \code{TRUE}, holdout sample of size \code{h} is taken from the end of the data.
  }
  \item{model}{
    Type of ETS model used for the estimation. Normally this should be either \code{"ANN"} or \code{"MNN"}.
  }
  \item{persistence}{
    Persistence vector. If \code{NULL}, then it is estimated.
  }
}
\details{
  The function estimates probability of demand occurance, using one of the ETS state-space models.
}
\value{
    The object of class "iss" is returned. It contains following list of values:

  \itemize{
    \item \code{fitted} - fitted values of the constructed model;
    \item \code{states} - values of states (currently level only);
    \item \code{forecast} - forecast for \code{h} observations ahead;
    \item \code{variance} - conditional variance of the forecast;
    \item \code{logLik} - likelihood value for the model;
    \item \code{nParam} - number of parameters used in the model;
    \item \code{residuals} - residuals of the model;
    \item \code{C} - vector of all the parameters.
    \item \code{actuals} - actual values of probabilities (zeroes and ones).
  }
}
\references{
    \itemize{
        \item{Teunter R., Syntetos A., Babai Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. European Journal of Operational Research, 214, 606-615.}
        \item{Croston, J. (1972) Forecasting and stock control for intermittent demands. Operational Research Quarterly, 23(3), 289-303.}
        \item{Syntetos, A., Boylan J. (2005) The accuracy of intermittent demand estimates. International Journal of Forecasting, 21(2), 303-314.}
    }
}
\author{
    Ivan Svetunkov, \email{ivan@svetunkov.ru}
}
\seealso{
    \code{\link[forecast]{ets}, \link[forecast]{forecast}, \link[smooth]{es}}
}
\examples{
    y <- rpois(100,0.1)
    iss(y, intermittent="t")

    iss(y, intermittent="c", persistence=0.1)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ iss }
\keyword{ intermittent demand }
\keyword{ intermittent demand state space model }
\keyword{ exponential smoothing }
\keyword{ forecasting }
