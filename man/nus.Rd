\name{nus}
\alias{nus}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Non-uniform Smoothing
}
\description{
Function estimates regression using defined formula and updates its parameters.
}
\usage{
nus(formula, data, h=10, holdout=FALSE,
    initial=c("whole","min","quarter","10 obs"),
    weights=c("dynamic","static","manual"),
    persistence=c("adaptive","optimal","optimal.all","manual"),
    filter=c("none","MAE","RMSE","optimal","dynamic"),
    weights.v=NULL, silent=FALSE, legend=TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{ The regression formula in the format provided to \code{lm} function
  }
  \item{data}{ the data that needs to be forecasted.
  }
  \item{h}{ the forecasting horizon.
  }
  \item{holdout}{ if \code{TRUE}, the holdout of the size \code{h} is taken from the end of the data. CURRENTLY if \code{FALSE}, no forecast is produced.
  }
  \item{initial}{ What part of sample take for the initial estimation. WILL BE CHANGED TO EITHER USER DEFINED OR OPTIMISATION BASED.
  }
  \item{weights}{ What type of weights to use in the smoothing parameters. \code{dynamic} means that weights change in time depending on differences of the data. \code{static} - the fixed equal weights are used. \code{manual} - weights will be provided by user.
  }
  \item{persistence}{ What types of smothing parameters to use. \code{adaptive} means that smoothing parameter changes in time. \code{optimal} - one parameter estimated for the whole model. \code{optimal.all} - each parameter is estimated separately. \code{manual} - parameters are provided by user. WILL BE CHANGED TO EITHER USER DEFINED OR OPTIMISATION BASED (SIMILAR TO \code{es}).
  }
  \item{filter}{ What type of filter to use. \code{none} - no filter, update is done on each observation. \code{MAE} - the filter is constant and uses MAE. \code{RMSE} - constant filter based on RMSE. \code{optimal} - the filter is estimated on sample. \code{dynamic} - filter changes depending on the size of the error.
  }
  \item{weights.v}{ is the vector of weights. Needed if \code{weights="manual"}.
  }
  \item{silent}{ if \code{TRUE}, no output or graph is given.
  }
  \item{legend}{ if \code{TRUE}, the legend is drawn on the output plot.
  }
}
\details{
The function creates the model of the following type:
\eqn{y_[t] = w' x_[t-1] + \epsilon_[t]}
\eqn{x_[t] = F x_[t-1] + g \epsilon_[t]}

}
\value{
  The list of the following values is returned:

  \itemize{
    \item \code{fitted} - the fitted values and the forecast.
    \item \code{states} - the matrix of all the coefficients.
    \item \code{weights} - weight matrix of NUS.
    \item \code{persistence} - persistence matrix of NUS.
    \item \code{filter} - filter values.
    \item \code{residuals} - the residuals of the model.
  }
}
\references{
No references yet.
}
\author{
Ivan Svetunkov
}

\seealso{
\code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ges}}
}

\examples{

ourdata <- matrix(c(rnorm(50,50,3)),50,1)
ourdata <- cbind(ourdata,c(rep(0,10),1,1,rep(0,25),1,rep(0,10),1,1))
ourcoefficients <- rep(NA,50)
ourcoefficients[1] <- 0.5
for(i in 2:50){ourcoefficients[i] <- ourcoefficients[i-1] + rnorm(1,0,1)}
ourdata <- cbind(ourdata[,1]*ourcoefficients+50*ourdata[,2]+rnorm(50,0,3)+500,
                 ourdata)
colnames(ourdata) <- c("y","x","dummy")
ourdata <- data.frame(ourdata)

nus(y~x+dummy,ourdata,h=10,holdout=TRUE)->test

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ NUS }
\keyword{ Non-uniform smoothing }
\keyword{ exogenous variables }
\keyword{ regression }
\keyword{ Exponential Smoothing }% __ONLY ONE__ keyword per line
