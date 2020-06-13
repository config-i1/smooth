---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "13 June 2020"
output: html_document
---
## Version
This is ``smooth`` package, v2.6.0.

## Test environments
* local ubuntu 19.10, R 3.6.3
* ubuntu 16.04.6 (on travis-ci), R 4.0.0
* win-builder (devel and release)
* rhub with rhub::check_for_cran() command

## R CMD check results
R CMD check results
checking installed package size ... NOTE
    installed size is 20.8Mb
    sub-directories of 1Mb or more:
      doc    2.5Mb
      libs  17.0Mb
0 errors | 0 warnings | 1 note

## win-builder check results
All seems to be okay.

## rhub checks  
Everything was checked without major issues. The only thing is a note about the installed package size for different platforms (similar to R CMD check results).

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
