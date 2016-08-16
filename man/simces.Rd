\name{sim.ces}
\alias{sim.ces}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Simulate Complex Exponential Smoothing
}
\description{
Function generates data using state-space CES with Single Source of Error as a data generating process.
}
\usage{
sim.ces(seasonality=c("N","S","P","F"), seas.freq=1,
    persistence=NULL, initial=NULL, initial.season=NULL,
    obs=10, nseries=1,silent=FALSE,
    randomizer=c("rnorm","runif","rbeta","rt"),
    ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seasonality}{
  type of CES model: "N" - non-seasonal, "S" - simple lagged seasonal model (no level), "P" - partial seasonality (real part only), "F" - full (complex) seasonality.
  }
  \item{seas.freq}{
  the frequency of the data. In the case of seasonal models must be greater than 1.
  }
  \item{persistence}{
  the persistence vector, that includes all the smoothing parameters. Must correspond to the chosen model. The maximum length is 4: level, information potential and seasonal smoothing parameters. If \code{NULL}, all the values will be generated.
  ATTENTION! The parameters are generated from the admissible bounds, which may lead to explosive processes. It is adviced to specify the persistence value manually.
  }
  \item{initial}{
  the vector of initial states of level and information potential component. The maximum length is 2. If \code{NULL}, all the values will be generated. Not needed in case of \code{seasonality="S"}.
  }
  \item{initial.season}{
  the vector of initial states for seasonal components. Should correspond to the \code{seas.freq} parameter. If \code{NULL}, all the values will be taken at random. For the \code{seasonality="S"} and \code{seasonality="F"} it should contain \code{seas.freq * 2} components, where first \code{seas.freq} is real components and the other \code{seas.freq} are imaginary. Not needed in case of \code{seasonality="N"}.
  }
  \item{obs}{
  the number of observations in each generated time series.
  }
  \item{nseries}{
  the number of series to generate.
  }
  \item{silent}{
  if TRUE then no command line output is printed.
  }
  \item{randomizer}{
  the type of the random number generator function. The defaults are: \code{rnorm}, \code{rt}, \code{runif}, \code{rbeta}. But any function from \link[stats]{Distributions} will do the trick if the appropriate parameters are passed. For example \code{rpois} with \code{lambda=2} can be used.
  }
  \item{...}{
  additional parameters passed to the chosen randomizer. All the parameters should be passed in the order they are used in the chosen function. For example, passing \code{sd=0.5} to \code{rnorm} function will lead to the call \code{rnorm(obs, mean=0.5, sd=1)}.
  }
}
\value{
The list of the following values is returned:
  \item{data}{The ts vector of the generated series.}
  \item{ip}{The ts vector of the generated imaginary part of series (information potential).}
  \item{states}{The states matrix. States are in columns, time is in rows.}
  \item{residuals}{The error terms used in the simulation.}
  \item{persistence}{The vector of smoothing parameters used in the simulation.}
  \item{model}{The name of the used ces model.}
  \item{likelihood}{Log-likelihood of the constructed model.}
}
\references{
    \itemize{
        \item{Svetunkov, I., Kourentzes, N. (February 2015). Complex exponential smoothing. Working Paper of Department of Management Science, Lancaster University 2015:1, 1-31.}
    }
}
\author{
Ivan Svetunkov, \email{ivan@svetunkov.ru}
}

\seealso{
\code{\link[smooth]{ces}, \link[forecast]{forecast}, \link[stats]{ts}, \link[stats]{Distributions}}
}
\examples{
# Create 60 observations of monthly data using non-seasonal CES
sim.ces(seasonality="N",seas.freq=12,obs=60,persistence=c(0.8,1.01))->x

# Create 60 observations of monthly data using simple seasonal CES
sim.ces(seasonality="S",seas.freq=12,obs=60,persistence=c(1.3,1.5))->x

# Create 60 observations of monthly data using partial seasonal CES
# with specified non-seasonal initials
sim.ces(seasonality="P",seas.freq=12,obs=60,persistence=c(1.8,0.9,0.4),initial=c(5000,2500))->x

# Create 60 observations of monthly data using full seasonal CES
sim.ces(seasonality="F",seas.freq=12,obs=60,persistence=c(1.2,1.01,1.3,0.8))->x

}

\keyword{ exponential smoothing }
\keyword{ CES }
\keyword{ forecasting }
\keyword{ simulation }
