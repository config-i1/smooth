# smooth
The package contains several smoothing (exponential and not) functions that are used in forecasting

Here is the list of functions:

1. sim.ets - simulation of data using ETS framework with a predefined (or random) smoothing parameters and initial values.
2. es - the ETS function that uses different estimation methods than ets from "forecast" package. It can also handle exogenous variables and has a handy "holdout" parameter. Finally, all the possible ETS functions are implemented.
3. ces - Complex Exponential Smoothing. Function estimates CES and makes a forecast.
4. ces.auto - selection between seasonal and non-seasonal CES models.
5. sim.ces - simulation of time series data using CES model with a predefined (or random) smoothing parameters and initials.
6. ges - Generalised Exponential Smoothing. Next step from CES. Currently only GES(2) is implemented, but a more flexible function will follow.
7. nus - Non-uniform Smoothing. The estimation method used in order to update parameters of a regression model.

Future works:

8. sofa - Survival of the fittest algorythm applied to state-space models.

For a quick and easy installation of the package firstly install "devtools" in R:
> if (!require("devtools")){install.packages("devtools")}

And after that run:

> devtools::install_github("config-i1/smooth")

The package now depends on Rcpp and RcppArmadillo, which will be installed automatically.

However Mac OS users may need to install gfortran libraries in order to use Rcpp. Follow the link for the instructions: http://www.thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/
