# smooth
The package contains several smoothing (exponential and not) functions that are used in forecasting

Here is the list of functions:

1. es - the ETS function that uses different estimation methods than ets from "forecast" package. It can also handle exogenous variables and has a handy "holdout" parameter. There are several cost function implemented, including trace forecast based ones. Finally, all the possible ETS functions are implemented in the function.
2. ces - Complex Exponential Smoothing. Function estimates CES and makes forecast.
3. ces.auto - selection between seasonal and non-seasonal CES models.
4. ges - Generalised Exponential Smoothing. Next step from CES. The paper on this is in the process.
5. nus - Non-uniform Smoothing. The estimation method used in order to update parameters of a regression model.
6. sim.ets - simulation of data using ETS framework with a predefined (or random) smoothing parameters and initial values.
7. sim.ces - simulation of time series data using CES model with a predefined (or random) smoothing parameters and initials.

Future works:

8. ssarima - SARIMA estimated in state-space framework.
9. sofa.ts - Survival of the fittest algorithm applied to state-space models.

For a quick and easy installation of the package firstly install "devtools" in R:
> if (!require("devtools")){install.packages("devtools")}

And after that run:

> devtools::install_github("config-i1/smooth")

The package now depends on Rcpp and RcppArmadillo, which will be installed automatically.

However Mac OS users may need to install gfortran libraries in order to use Rcpp. Follow the link for the instructions: http://www.thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/
