\name{sim.es}
\alias{sim.es}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Simulate Exponential Smoothing
}
\description{
    Function generates data using ETS with Single Source of Error as a data generating process.
}
\usage{
sim.es(model="ANN", frequency=1, persistence=NULL, phi=1,
        initial=NULL, initialSeason=NULL,
        bounds=c("usual","admissible","restricted"),
        obs=10, nsim=1, silent=FALSE,
        randomizer=c("rnorm","rlnorm","runif","rbeta","rt"),
        iprob=1, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{model}{
    Type of ETS model according to [Hyndman et. al., 2008] taxonomy. Can consist of 3 or 4 chars: \code{ANN}, \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA}, \code{MAdM} etc.
  }
  \item{frequency}{
    Frequency of generated data. In cases of seasonal models must be greater than 1.
  }
  \item{persistence}{
    Persistence vector, which includes all the smoothing parameters. Must correspond to the chosen model. The maximum length is 3: level, trend and seasonal smoothing parameters. If \code{NULL}, values are generated.
  }
  \item{phi}{
    Value of damping parameter. If trend is not chosen in the model, the parameter is ignored.
  }
  \item{initial}{
    Vector of initial states of level and trend. The maximum length is 2. If \code{NULL}, values are generated.
  }
  \item{initialSeason}{
    Vector of initial states for seasonal coefficients. Should have length equal to \code{frequency} parameter. If \code{NULL}, valuesare generated.
  }
  \item{bounds}{
    Type of bounds to use for persistence vector if values are generated. \code{"usual"} - bounds from p.156 by Hyndman et. al., 2008. \code{"restricted"} - similar to \code{"usual"} but with upper bound equal to 0.3. \code{"admissible"} - bounds from tables 10.1 and 10.2 of Hyndman et. al., 2008. Using first letter of the type of bounds also works. These bounds are also used for multiplicative models, so be careful!
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
    Type of random number generator function used for error term. Defaults are: \code{rnorm}, \code{rlnorm}, \code{rt}, \code{runif}, \code{rbeta}. But any function from \link[stats]{Distributions} will do the trick if the appropriate parameters are passed. For example \code{rpois} with \code{lambda=2} can be used as well.
  }
  \item{iprob}{
    Probability of occurrence, used for intermittent data generation.
  }
  \item{...}{
    Additional parameters passed to the chosen randomizer. All the parameters should be passed in the order they are used in chosen randomizer. For example, passing just \code{sd=0.5} to \code{rnorm} function will lead to the call \code{rnorm(obs, mean=0.5, sd=1)}.
    ATTENTION! When generating the multiplicative errors some tuning might be needed to obtain meaningful data. \code{sd=0.1} is usually already a high value for such models.
  }
}
\value{
    List of the following values is returned:
        \item{model}{Name of ETS model.}
        \item{data}{Time series vector (or matrix if \code{nsim>1}) of the generated series.}
        \item{states}{Matrix (or array if \code{nsim>1}) of states. States are in columns, time is in rows.}
        \item{persistence}{Vector (or matrix if \code{nsim>1}) of smoothing parameters used in the simulation.}
        \item{phi}{Value of damping parameter used in time series generation.}
        \item{residuals}{Error terms used in the simulation. Either vector or matrix, depending on \code{nsim}.}
        \item{occurrences}{Values of occurrence variable. Once again, can be either a vector or a matrix...}
        \item{likelihood}{Log-likelihood of the constructed model.}
}
\references{
    Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.
}
\author{
    Ivan Svetunkov, \email{ivan@svetunkov.ru}
}

\seealso{
    \code{\link[smooth]{es}, \link[forecast]{ets}, \link[forecast]{forecast}, \link[stats]{ts}, \link[stats]{Distributions}}
}
\examples{
# Create 40 observations of quarterly data using AAA model with errors from normal distribution
sim.es(model="AAA",frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100) -> ETS.AAA

# Create 50 series of quarterly data using AAA model
# with 40 observations each with errors from normal distribution
sim.es(model="AAA",frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100,nsim=50) -> ETS.AAA

# Create 50 series of quarterly data using AAdA model
# with 40 observations each with errors from normal distribution
# and smoothing parameters lying in the "admissible" range.
sim.es(model="AAA",phi=0.9,frequency=4,obs=40,bounds="admissible",
        randomizer="rnorm",mean=0,sd=100,nsim=50) -> ETS.AAA

# Create 60 observations of monthly data using ANN model
# with errors from beta distribution
sim.es(model="ANN",persistence=c(1.5),frequency=12,obs=60,
        randomizer="rbeta",sshape1=1.5,sshape2=1.5) -> ETS.ANN
plot(ETS.ANN$states)

# Create 60 observations of monthly data using MAM model
# with errors from uniform distribution
sim.es(model="MAM",persistence=c(0.3,0.2,0.1),initial=c(2000,50),
        phi=0.8,frequency=12,obs=60,randomizer="runif",min=-0.5,max=0.5) -> ETS.MAM

# Create 80 observations of quarterly data using MMM model
# with predefined initial values and errors from the normal distribution
sim.es(model="MMM",persistence=c(0.1,0.1,0.1),initial=c(2000,1),
        initialSeason=c(1.1,1.05,0.9,.95),frequency=4,obs=80,mean=0,sd=0.01) -> ETS.MMM

# Generate intermittent data using AAdN
sim.es("AAdN",frequency=1,obs=30,iprob=0.1,initial=c(3,0),phi=0.8) -> iETS.AAdN
}

\keyword{ exponential smoothing }
\keyword{ ETS }
\keyword{ forecasting }
\keyword{ simulation }
