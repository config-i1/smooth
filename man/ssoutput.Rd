\name{ssoutput}
\alias{ssoutput}
\title{
    State-space output function
}
\description{
    The function prints out all the necessary details of the state-space used. This is mainly for the internal use of es(), ces(), ges(), ssarima() and nus().
}
\usage{
ssoutput(timeelapsed, modelname, persistence=NULL, transition=NULL, measurement=NULL,
         phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
         n.components=NULL, s2=NULL, hadxreg=FALSE, wentwild=FALSE,
         CF.type="MSE", CF.objective=NULL, intervals=FALSE,
         int.type=c("p","s","n","a"), int.w=0.95, ICs,
         holdout=FALSE, insideintervals=NULL, errormeasures=NULL)
}

\arguments{
    \item{timeelapsed}{
        Time elapsed on the estimation.
    }
    \item{modelname}{
        The name of the model.
    }
    \item{persistence}{
        Persistence vector.
    }
    \item{transition}{
        Transition matrix.
    }
    \item{measurement}{
        Measurement vector
    }
    \item{phi}{
        Damping parameter
    }
    \item{ARterms}{
        Table with AR parameters.
    }
    \item{MAterms}{
        Table with MA parameters.
    }
    \item{const}{
        The value of a constant term.
    }
    \item{A}{
        Complex smoothing parameter.
    }
    \item{B}{
        Parameter for the seasonality in CES.
    }
    \item{n.components}{
        Number of components estimated.
    }
    \item{s2}{
        The sample variance of error term.
    }
    \item{hadxreg}{
        Were the exogenous variables provided?
    }
    \item{wentwild}{
        Did we go.wild with the exogenous estimation?
    }
    \item{CF.type}{
        Type of cost function used.
    }
    \item{CF.objective}{
        The final value of the cost function.
    }
    \item{intervals}{
        Were the intervals constructed?
    }
    \item{int.type}{
        The type of intervals.
    }
    \item{int.w}{
        The width of intervals.
    }
    \item{ICs}{
        Information criteria for the model. Should be in a form of vector with names.
    }
    \item{holdout}{
        Did we do all of that with the holdout sample?
    }
    \item{insideintervals}{
        How many values we got in the prediction interval (ratio)?
    }
    \item{errormeasures}{
        MPE, MAPE, MASE... Should be in a form of vector with names.
    }
}
\details{
    If you don't know what this function is needed for, don't bother. It's pretty technical...
}
\value{
    It doesn't return anything.
}
\author{
    Ivan Svetunkov
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
