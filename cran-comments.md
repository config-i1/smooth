---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "19 August 2019"
output: html_document
---
## Version
This is ``smooth`` package, v2.5.3.

## Test environments
* local ubuntu 19.04, R 3.5.2
* ubuntu 14.04.5 (on travis-ci), R 3.6.1
* win-builder (devel and release)
* rhub with rhub::check_for_cran() command

## R CMD check results
R CMD check results
checking installed package size ... NOTE
    installed size is 18.7Mb
    sub-directories of 1Mb or more:
      doc    2.5Mb
      libs  15.1Mb
0 errors | 0 warnings | 1 note

## win-builder check results
The thing complains about the URLs, which are available from PC

Found the following (possibly) invalid URLs:
  URL: http://doi.org/10.1109/LSP.2006.870353
    From: man/pls.Rd
    Status: Error
    Message: libcurl error code 35:
      	error:1407742E:SSL routines:SSL23_GET_SERVER_HELLO:tlsv1 alert protocol version
  URL: https://doi.org/10.2307/2533213
    From: inst/doc/ves.html
    Status: 403
    Message: Forbidden

These two lead to respectively:
https://ieeexplore.ieee.org/document/1618702
and
https://www.jstor.org/stable/2533213?origin=crossref&seq=1#page_scan_tab_contents

## rhub checks
### Debian Linux, R-devel, GCC ASAN/UBSAN
The compilation on clang-UBSAN and gcc-UBSAN produces some errors. All the errors are related to RcppArmadillo, not to the C++ functions, implemented in this package. If this is caused by the smooth package, then it is not even possible to track those errors, as there is no output about what the function was run: I have 14 forecasting functions and 6 simulation functions, and it's not clear, which of them was run, when that happened. In addition, there is no information on what part of the C++ code could cause this. For example, here's a chunk with the errors from rhub that I get, when I test it:

/home/docker/R/RcppArmadillo/include/armadillo_bits/subview_meat.hpp:1209:54: runtime error: reference binding to null pointer of type 'const unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/subview_meat.hpp:1209:54: runtime error: reference binding to null pointer of type 'const unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/subview_meat.hpp:1209:54: runtime error: reference binding to null pointer of type 'const unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/subview_meat.hpp:1209:54: runtime error: reference binding to null pointer of type 'const unsigned int'
/home/docker/R/RcppArmadillo/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'unsigned int'

Please, note that all of these complain about Armadillo functions. Also, there are no errors on other platforms, and the code runs well, without issues.


## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
