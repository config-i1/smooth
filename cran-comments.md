---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "20 October 2019"
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

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
