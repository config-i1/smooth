---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "25 April 2018"
output: html_document
---
## Version
This is ``smooth`` package, v2.5.0.

## Test environments
* local ubuntu 18.04, R 3.5.1
* ubuntu 14.04.5 (on travis-ci), R 3.5.2
* win-builder (devel and release)

## R CMD check results
R CMD check results
checking installed package size ... NOTE
  installed size is  10.1Mb
  sub-directories of 1Mb or more:
    doc    1.8Mb
    libs   7.7Mb
0 errors | 0 warnings | 1 note

# Other checks
The automatic compilation on CRAN complains about the url in one of my vignettes:

Found the following (possibly) invalid URLs:
  URL: http://www.exponentialsmoothing.net
    From: man/forecast.smooth.Rd
            man/gsi.Rd
            man/sim.ves.Rd
            man/ves.Rd
      Status: 504
      Message: Gateway Timeout

The url is currently unavailable, but it worked before.

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
