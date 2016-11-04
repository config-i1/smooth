\name{AICc}
\alias{AICc}
\alias{AICc.default}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Corrected Akaike's Information Criterion
}
\description{
    This function extracts AICc from "smooth" objects.
}
\usage{
    AICc(object, ...)
    \method{AICc}{default}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
    \item{object}{
        Time series model.
    }
    \item{...}{
        Some stuff.
    }
}
\details{
    AICc was proposed by Nariaki Sugiura in 1978 and is used on small samples.
}
\value{
    This function returns numeric value.
}
\references{
   Kenneth P. Burnham, David R. Anderson (1998). Model Selection and Multimodel Inference. Springer Science & Business Media.
}
\author{
    Ivan Svetunkov, \email{ivan@svetunkov.ru}
}
\seealso{
    \code{\link[stats]{AIC}, \link[stats]{BIC}}
}
\examples{
ourModel <- ces(rnorm(100,0,1),h=10)

AICc(ourModel,h=10)
}
\keyword{ information criteria }
\keyword{ information criterion }
