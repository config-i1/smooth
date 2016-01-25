\name{sim.ets}
\alias{sim.ets}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Simulate Exponential Smoothing
}
\description{
Function generates data using ETS with Single Source of Error as a data generating process.
}
\usage{
sim.ets(model="ANN",frequency=1,
    persistence=NULL, phi=1,
    initial=NULL, initial.season=NULL,
    bounds=c("usual","admissible","restricted"),
    obs=10, nseries=1, silent=FALSE,
    randomizer=c("rnorm","runif","rbeta","rt"),
    ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{model}{
  type of ETS model according to [Hyndman et. al., 2008] taxonomy. Can consist of 3 or 4 chars: \code{ANN}, \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA}, \code{MAdM} etc.
}
  \item{frequency}{
  the frequency of the data. In the case of seasonal models must be greater than 1.
}
  \item{persistence}{
  the persistence vector, that includes all the smoothing parameters. Must correspond to the chosen model. The maximum length is 3: level, trend and seasonal smoothing parameters. If \code{NULL}, all the values will be taken at random.
}
  \item{phi}{
  the value of damping parameter. If trend is not chosen in the model, the parameter is ignored.
}
  \item{initial}{
  the vector of initial states of level and trend. The maximum length is 2. If \code{NULL}, all the values will be taken at random.
}
  \item{initial.season}{
  the vector of initial states for seasonal coefficients. Should correspond to the \code{frequency} parameter If \code{NULL}, all the values will be taken at random.
}
  \item{bounds}{
  the type of bounds to use for the persistence vector. "usual" - the bounds from p.156 by Hyndman et. al., 2008. "restricted" - similar to "usual" but with the high bound equal to 0.3. "admissible" - bounds from tables 10.1 and 10.2 of Hyndman et. al., 2008. Using first letter of the type of bounds also works.
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
  ATTENTION! When generating the multiplicative errors some tuning might be needed to obtain the meaningful data.
}
}
\value{
The list of the following values is returned:
  \item{data}{The ts vector of the generated series.}
  \item{states}{The states matrix. States are in columns, time is in rows.}
  \item{residuals}{The error terms used in the simulation.}
  \item{persistence}{The vector of smoothing parameters used in the simulation.}
  \item{model}{The name of the used ETS model.}
  \item{likelihood}{Log-likelihood of the constructed model.}
}
\references{
Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.
}
\author{
Ivan Svetunkov, \email{ivan@svetunkov.ru}
}

\seealso{
\code{\link[forecast]{ets}, \link[forecast]{forecast}, \link[stats]{ts}, \link[stats]{Distributions}}
}
\examples{
# Create 40 observations of quarterly data using AAA model with errors from normal distribution
sim.ets(model="AAA",frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100)-> ETS.AAA

# Create 50 series of quarterly data using AAA model
# with 40 observations each with errors from normal distribution
sim.ets(model="AAA",frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100,nseries=50)-> ETS.AAA

# Create 50 series of quarterly data using AAdA model
# with 40 observations each with errors from normal distribution
# and smoothing parameters lying in the "admissible" range.
sim.ets(model="AAA",phi=0.9,frequency=4,obs=40,bounds="admissible",
        randomizer="rnorm",mean=0,sd=100,nseries=50)-> ETS.AAA

# Create 60 observations of monthly data using ANN model
# with errors from beta distribution
sim.ets(model="ANN",persistence=c(1.5),frequency=12,obs=60,
        randomizer="rbeta",sshape1=1.5,sshape2=1.5)-> ETS.ANN
plot(ETS.ANN$states)

# Create 60 observations of monthly data using MAM model
# with errors from uniform distribution
sim.ets(model="MAM",persistence=c(0.3,0.2,0.1),initial=c(2000,50),
        phi=0.8,frequency=12,obs=60,randomizer="runif",min=-0.5,max=0.5)-> ETS.MAM

# Create 80 observations of quarterly data using MMM model
# with predefined initial values and errors from the normal distribution
sim.ets(model="MMM",persistence=c(0.1,0.1,0.1),initial=c(2000,1),
        initial.seas=c(1.1,1.05,0.9,.95),frequency=4,obs=80,mean=0,sd=0.01)-> ETS.MMM
}

\keyword{ exponential smoothing }
\keyword{ ETS }
\keyword{ forecasting }
\keyword{ simulation }