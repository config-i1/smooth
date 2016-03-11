\name{intervals}
\alias{intervals}
\title{
Prediction intervals construction
}
\description{
Function produces different types of intervals depending on the parameters using the provided vector or matrix of random variables.
}
\usage{
intervals(x,ev=median(x),int.w=0.95,int.type=c("a","p","s","n"))
}
\arguments{
  \item{x}{ The vector or matrix of the random variable. Can be, for example, a matrix of 1 to h steps ahead errors.
}
  \item{ev}{ The Expected Value of \code{x} used in the construction. Can be a vector of values if \code{x} is a matrix.
}
  \item{int.w}{ Defines the width of the prediction interval.
  }
  \item{int.type}{ The type of intervals to construct. The first letter can be used instead of the whole word. \code{a} stands for \code{asymmetric}, \code{p} - for \code{parametric}, \code{s} - for \code{semiparametric} and \code{n} - for \code{nonparametric}. Only asymmetric intervals are currently implemented.
  }
}
\details{
    The asymmetric intervals are constructed using half moment of distribution. See Svetunkov ISIR, 2016 presentation for the details.
}
\value{
  The function returns:

  \itemize{
    \item \code{lower} - the lower bound of prediction interval.
    \item \code{upper} - the higher bound of prediction interval.
  }
}
\references{
    Nothing really here...
}
\author{
    Ivan Svetunkov
}

\seealso{
    Nothing to see here yet.
}
\examples{
x <- rpois(100,1.5)
intervals(x,int.type="a")

x <- matrix(exp(rnorm(1000,0,1)),100,10)
intervals(x,ev=apply(x,2,median),int.type="a")
}
\keyword{ intervals }
\keyword{ prediction }
\keyword{ prediction intervals }
