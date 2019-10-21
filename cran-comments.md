---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "22 October 2019"
output: html_document
---
## Version
This is ``smooth`` package, v2.5.4.

## Test environments
* local ubuntu 19.04, R 3.6.1
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
All seems to be okay.

## rhub checks
I've gone through the C++ code of mine and isolated a potentially dangerous part of code, that could have produced previous errors in ASAN/UBSAN. However, the check on rhub with ASAN/UBSAN still produces errors. At least, this time they are definitely not related to my code - they are related to loading of other packages: it looks like, when forecast package is loaded, there is a bunch of warnings and errors on that specific platform.

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
