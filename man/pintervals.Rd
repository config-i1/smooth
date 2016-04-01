\name{pintervals}
\alias{pintervals}
\title{
    Prediction intervals for state-space models
}
\description{
    Function returns quantiles of distribution using different methods based on the provided vector or matrix of random variables.
}
\usage{
pintervals(errors,ev=median(errors),int.w=0.95,int.type=c("a","p","s","n"), df=NULL,
           measurement=NULL, transition=NULL, persistence=NULL, s2=NULL, modellags=NULL,
           y.for=rep(0,ncol(errors)), iprob=1)
}
\arguments{
    \item{errors}{ The vector or matrix of the random variable. Can be, for example, a matrix of 1 to h steps ahead errors.
    }
    \item{ev}{ The Expected Value of \code{errors} used in the construction. Can be a vector of values if \code{errors} is a matrix.
    }
    \item{int.w}{ Defines the width of the prediction interval.
    }
    \item{int.type}{ The type of intervals to construct. The first letter can be used instead of the whole word. \code{"a"} stands for \code{"asymmetric"}, \code{"p"} - for \code{"parametric"}, \code{"s"} - for \code{"semiparametric"} and \code{"n"} - for \code{"nonparametric"}. Only asymmetric intervals are currently implemented.
    }
    \item{df}{ The number of degrees of freedom. If provided, then Student's distribution is used. Otherwise normal distribution is used.
    }
    \item{measurement}{ Measurement vector of State-space model. Only needed for parameteric intervals.
    }
    \item{transition}{ Transition matrix of State-space model. Only needed for parameteric intervals.
    }
    \item{persistence}{ Persistence vector of State-space model. Only needed for parameteric intervals.
    }
    \item{s2}{ Variance of errors. Only needed for parameteric intervals.
    }
    \item{modellags}{ Lags used in State-space model. Only needed for parameteric intervals.
    }
    \item{y.for}{ The point forecast produced by the model. Used for intermittent data only.
    }
    \item{iprob}{ The probability of intermittent demand occurance.
    }
}
\details{
    If \code{errors} is matrix, then several steps ahead prediction intervals are constructed. Otherwise only one step ahead is done.
    \enumerate{
        \item \code{parametric} intervals are available only for additive state-space models (additive trend and or seasonality).
        \item \code{semiparametric} intervals are based on the variances of errors provided in the input.
        \item \code{nonparametric} intervals are based on quantile regressions (Taylor and Bunn, 1999).
        \item \code{asymmetric} intervals are constructed using half moment of distribution. See Svetunkov ISIR, 2016 presentation for the details.
    }
}
\value{
  The function returns:

  \itemize{
    \item \code{lower} - the lower bound of prediction interval.
    \item \code{upper} - the upper bound of prediction interval.
  }
}
\references{
    \enumerate{
        \item{Taylor, J.W. and Bunn, D.W. (1999) A Quantile Regression Approach to Generating Prediction Intervals. Management Science, Vol 45, No 2, pp 225-237.}
    }
}
\author{
    Ivan Svetunkov
}

\seealso{
    Nothing to see here yet.
}
\examples{
x <- rpois(100,1.5)
pintervals(x,int.type="a")

x <- matrix(exp(rnorm(1000,0,1)),100,10)
pintervals(x,ev=apply(x,2,median),int.type="a")
}
\keyword{ intervals }
\keyword{ prediction }
\keyword{ prediction intervals }
