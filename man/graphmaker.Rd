\name{graphmaker}
\alias{graphmaker}
\title{
  Linear graph construction function
}
\description{
  The function makes a standard linear graph using at least actuals, forecasts.
}
\usage{
  graphmaker(actuals,forecast,fitted=NULL,lower=NULL,upper=NULL,int.w=NULL,legend=TRUE)
}
\arguments{
  \item{actuals}{The vector of actual series.
}
  \item{forecast}{The vector of forecasts. Should be ts object that start at the end of \code{fitted} values.
}
  \item{fitted}{The vector of fitted values.
}
  \item{lower}{The vector of lower bound values of a prediction interval. Should be ts object that start at the end of \code{fitted} values.
}
  \item{upper}{The vector of upper bound values of a prediction interval. Should be ts object that start at the end of \code{fitted} values.
}
  \item{int.w}{The width of the prediction interval.
}
  \item{legend}{If \code{TRUE}, the legend is drawn.
}
}
\details{
  Function uses the provided data to construct a linear graph. It is strongly adviced to use \code{ts} function to define the start of each of the vectors. Otherwise the data may be plotted in a wrong way.
}
\value{Function does not return anything.
}
\author{
  Ivan Svetunkov
}

\seealso{
  \code{\link[stats]{ts}}
}
\examples{
x <- rnorm(100,0,1)
values <- es(x,model="ANN",silent=TRUE,intervals=TRUE,int.w=0.95)

graphmaker(x,values$forecast,values$fitted)
graphmaker(x,values$forecast,values$fitted,legend=FALSE)
graphmaker(x,values$forecast,values$fitted,values$lower,values$upper,int.w=0.95)
graphmaker(x,values$forecast,values$fitted,values$lower,values$upper,int.w=0.95,legend=FALSE)

actuals <- c(1:10)
forecast <- ts(c(11:15),start=end(actuals)[1]+end(actuals)[2]*deltat(actuals),
               frequency=frequency(actuals))
graphmaker(actuals,forecast)
}
\keyword{ plots }
\keyword{ linear graph }