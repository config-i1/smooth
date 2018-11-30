---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "30 November 2018"
output: html_document
---
## Version
This is ``smooth`` package, v2.4.7. It fixes the bug caused by the recent upgrade of greybox package.
* I fixed the warning produced from C++ about the "polymorphic type".

## Test environments
* local ubuntu 18.04, R 3.4.4
* ubuntu 14.04.5 (on travis-ci), R 3.5.1
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
For some reason, the automatic compilation on CRAN complains about the url in one of my vignettes:

Found the following (possibly) invalid URLs:
  URL: https://doi.org/10.2307/2533213
    From: inst/doc/ves.html
    Status: 403
    Message: Forbidden

I checked this url on several PCs and it works on all of them, it should redirect to the paper "Model Selection for Multivariate Regression in Small Samples".

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
