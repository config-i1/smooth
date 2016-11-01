\name{sim.ces}
\alias{sim.ces}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Simulate Complex Exponential Smoothing
}
\description{
    Function generates data using CES with Single Source of Error as a data generating process.
}
\usage{
sim.ces(seasonality=c("none","simple","partial","full"),
        frequency=1, A=NULL, B=NULL,
        initial=NULL,
        obs=10, nsim=1, silent=FALSE,
        randomizer=c("rnorm","runif","rbeta","rt"),
        iprob=1, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seasonality}{
    The type of seasonality used in CES. Can be: \code{none} - No seasonality; \code{simple} - Simple seasonality, using lagged CES (based on \code{t-m} observation, where \code{m} is the seasonality lag); \code{partial} - Partial seasonality with real seasonal components (equivalent to additive seasonality); \code{full} - Full seasonality with complex seasonal components (can do both multiplicative and additive seasonality, depending on the data). First letter can be used instead of full words.
    Any seasonal CES can only be constructed for time series vectors.
  }
  \item{frequency}{
    Frequency of generated data. In cases of seasonal models must be greater than 1.
  }
  \item{A}{
    First complex smoothing parameter. Should be a complex number.

    NOTE! CES is very sensitive to A and B values so it is advised to use values from previously estimated model.
  }
  \item{B}{
    Second complex smoothing parameter. Can be real if \code{seasonality="partial"}. In case of \code{seasonality="full"} must be complex number.
  }
  \item{initial}{
    A matrix with initial values for CES. In case with \code{seasonality="partial"} and \code{seasonality="full"} first two columns should contain initial values for non-seasonal components, repeated \code{frequency} times.
  }
  \item{obs}{
    Number of observations in each generated time series.
  }
  \item{nsim}{
    Number of series to generate (numeber of simulations to do).
  }
  \item{silent}{
    If TRUE then no output is printed.
  }
  \item{randomizer}{
    Type of random number generator function used for error term. Defaults are: \code{rnorm}, \code{rt}, \code{runif}, \code{rbeta}. But any function from \link[stats]{Distributions} will do the trick if the appropriate parameters are passed. For example \code{rpois} with \code{lambda=2} can be used as well.
  }
  \item{iprob}{
    Probability of occurrence, used for intermittent data generation.
  }
  \item{...}{
    Additional parameters passed to the chosen randomizer. All the parameters should be passed in the order they are used in chosen randomizer. For example, passing just \code{sd=0.5} to \code{rnorm} function will lead to the call \code{rnorm(obs, mean=0.5, sd=1)}.
  }
}
\value{
    List of the following values is returned:
        \item{model}{Name of CES model.}
        \item{A}{Value of complex smoothing parameter A. If \code{nsim>1}, then this is a vector.}
        \item{B}{Value of complex smoothing parameter B. If \code{seasonality="n"} or \code{seasonality="s"}, then this is equal to NULL. If \code{nsim>1}, then this is a vector.}
        \item{initial}{Initial values of CES in a form of matrix. If \code{nsim>1}, then this is an array.}
        \item{data}{Time series vector (or matrix if \code{nsim>1}) of the generated series.}
        \item{states}{Matrix (or array if \code{nsim>1}) of states. States are in columns, time is in rows.}
        \item{residuals}{Error terms used in the simulation. Either vector or matrix, depending on \code{nsim}.}
        \item{occurrences}{Values of occurrence variable. Once again, can be either a vector or a matrix...}
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
    \code{\link[smooth]{sim.es}, \link[smooth]{sim.ssarima}, \link[smooth]{ces}, \link[stats]{Distributions}}
}
\examples{
# Create 120 observations from CES(n). Generate 100 time series of this kind.
x <- sim.ces("n",obs=120,nsim=100)

# Generate similar thing for seasonal series of CES(s)_4
x <- sim.ces("s",frequency=4,obs=80,nsim=100)

# Estimate model and then generate 10 time series from it
ourModel <- ces(rnorm(100,100,5))
simulate(ourModel,nsim=10)
}

\keyword{ CES }
\keyword{ forecasting }
\keyword{ simulation }
