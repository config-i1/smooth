\name{orders}
\alias{orders}
\alias{lags}
\alias{modelType}
\alias{orders.default}
\alias{lags.default}
\alias{modelType.default}
\title{
    Functions that extract values from the fitted model
}
\description{
    These functions allow extracting orders for \code{ssarima()}, \code{ges()} and \code{sma()} and type of model from \code{es()} and \code{ces()}.
}
\usage{
    orders(object, ...)
    \method{orders}{default}(object, ...)
    lags(object, ...)
    \method{lags}{default}(object, ...)
    lags(object, ...)
    \method{modelType}{default}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
    \item{object}{
        Model estimated using one of the functions of smooth package.
    }
    \item{...}{
        Currently nothing is accepted via ellipsis.
    }
}
\details{
    \code{orders()} and \code{lags()} are usefull only for SSARIMA, GES and SMA. They return \code{NA} for other functions.

    \code{modelType()} is usefull only for ETS and CES. They return \code{NA} for other functions. This can also be applied to \code{ets()} function from forecast package.

}
\value{
    Either vector or list with values is returned.
    \code{orders()} in case of ssarima returns list of values:
    \itemize{
        \item \code{ar} - AR orders.
        \item \code{i} - I orders.
        \item \code{ma} - MA orders.
    }
}
\author{
    Ivan Svetunkov, \email{ivan@svetunkov.ru}
}
\seealso{
    \link[smooth]{forecast}
}
\examples{
# Just as example
test <- ces(rnorm(100,0,1), h=10)
orders(test)
lags(test)
modelType(test)

test <- ges(rnorm(100,0,1), h=10, orders=c(1,1), lags=c(1,4))
orders(test)
lags(test)
modelType(test)
}
\keyword{ forecasting }
