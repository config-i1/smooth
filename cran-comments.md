---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "12 June 2020"
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
    installed size is 20.1Mb
    sub-directories of 1Mb or more:
      doc    2.5Mb
      libs  16.4Mb
0 errors | 0 warnings | 1 note

## win-builder check results
All seems to be okay.

## rhub checks
Fedora Linux (R-devel, clang, gfortran) and Ubuntu Linux 16.04 LTS (R-release, GCC) produce notes about the examples with CPU or elapsed time > 5s.
* checking examples ... NOTE
Examples with CPU or elapsed time > 5s
               user system elapsed
smoothCombine 4.660  0.068  14.313
auto.gum      3.824  0.024  10.633
es            3.484  0.028   9.256
ces           1.836  0.012   5.348
orders        1.536  0.012   5.543

* checking examples ... NOTE
Examples with CPU (user + system) or elapsed time > 5s
               user system elapsed
smoothCombine 4.628  0.144  12.389
auto.gum      4.068  0.036  11.179
es            3.516  0.052   9.160


## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
