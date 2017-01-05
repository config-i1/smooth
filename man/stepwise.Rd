\name{stepwise}
\alias{stepwise}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Stepwise selection of regressors
}
\description{
    Function selects variables that give linear regression with the lowest information criteria. The selection is done stepwise (forward) based on partial correlations. This should be a simpler and faster implementation than step() function from `stats' package.
}
\usage{
    stepwise(data, ic=c("AIC","AICc","BIC"), silent=TRUE, df=NULL)
}
\arguments{
  \item{data}{
    Data frame containing dependant variable in the first column and the others in the rest.
  }
  \item{ic}{
    Information criterion to use.
  }
  \item{silent}{
    If \code{silent=FALSE}, then nothing is silent, everything is printed out. \code{silent=TRUE} means that nothing is produced.
  }
  \item{df}{
    Number of degrees of freedom to add (should be used if stepwise is used on residuals).
  }
}
\details{
    The algorithm uses lm() to fit different models and cor() to select the next regressor in the sequence.
}
\value{
    Function returns \code{model} - the final model of the class "lm".
}
%\references{
%% ~put references to the literature/web site here ~
%}
\author{
    Ivan Svetunkov
}

\examples{
xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg)

stepwise(xreg)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ stepwise }% use one of  RShowDoc("KEYWORDS")
\keyword{ linear regression }% __ONLY ONE__ keyword per line
