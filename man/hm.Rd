\name{hm}
\alias{hm}
\alias{cbias}
\title{
    Half moment of a distribution and its derivatives.
}
\description{
    \code{hm} function estimates half moment from some predefined constant \code{C}.
    \code{cbias} function calculates bias based on \code{hm}.
}
\usage{
    hm(x,C=mean(x),digits=5,...)
    cbias(x,C=mean(x),digits=5,...)
}

\arguments{
  \item{x}{ A variable based on which HM is estimated.
}
  \item{C}{ Centering parameter.
}
  \item{digits}{ Number of digits for rounding.
}
  \item{...}{ Other parameters passed to mean function.
}
}
\details{
    \code{NA} values of \code{x} are excluded on the first step of calculation.
}
\value{
    A complex variable is returned for \code{hm} function and real value is returned for \code{cbias}.
}
\references{
    TBA
}
\author{
    Ivan Svetunkov
}
\examples{
x <- rnorm(100,0,1)
hm(x)
cbias(x)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ half moment }
\keyword{ moments of distribution }% __ONLY ONE__ keyword per line
