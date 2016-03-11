\name{hm}
\alias{hm}
\title{
    Half moment of a distribution.
}
\description{
    Function estimates half moment from some predefined constant \code{C}.
}
\usage{
    hm(x,C=mean(x),digits=5,...)
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
    A complex variable is returned.
}
\references{
    Nothing yet...
}
\author{
    Ivan Svetunkov
}
\examples{
x <- rnorm(100,0,1)
hm(x)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ half moment }
\keyword{ moments of distribution }% __ONLY ONE__ keyword per line