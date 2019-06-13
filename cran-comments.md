---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "13 June 2019"
output: html_document
---
## Version
This is ``smooth`` package, v2.5.1.

## Test environments
* local ubuntu 18.04, R 3.5.1
* ubuntu 14.04.5 (on travis-ci), R 3.6.0
* win-builder (devel and release)

## R CMD check results
R CMD check results
checking installed package size ... NOTE
  installed size is  10.1Mb
  sub-directories of 1Mb or more:
    doc    2.4Mb
    libs   9.7Mb
0 errors | 0 warnings | 1 note

# Other checks
There are two issues that I cannot address:

1.

The compilation on clang-UBSAN and gcc-UBSAN produces some errors. I've tried addressing them, looking through my code and amending some potential sources of errors, and now all the errors are related to RcppArmadillo, not to the C++ functions, implemented in this package. If this is caused by the smooth package, then it is not even possible to track those errors, as there is no output about what the function was run: I have 14 forecasting functions and 6 simulation functions, and it's not clear, which of them was run, when that happened. In addition, there is no information on what part of the C++ code could cause this. For example, here's a chunk with the errors from rhub that I get, when I test it:

/home/docker/R/RcppArmadillo/include/armadillo_bits/subview_meat.hpp:1189:54: runtime error: reference binding to null pointer of type 'const unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/subview_meat.hpp:1189:54: runtime error: reference binding to null pointer of type 'const unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/subview_meat.hpp:1189:54: runtime error: reference binding to null pointer of type 'const unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/subview_meat.hpp:1189:54: runtime error: reference binding to null pointer of type 'const unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'unsigned int'

Note that all of these complain about Armadillo functions, not my functions. Also, there are no errors on other platforms, and the code runs well, without issues.


2.
The check on winbuild complains about the url in one of my vignettes:

Found the following (possibly) invalid URLs:
  URL: https://doi.org/10.2307/2533213
    From: inst/doc/ves.html
    Status: 403
    Message: Forbidden

However, the url is accessible from the browser.

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
