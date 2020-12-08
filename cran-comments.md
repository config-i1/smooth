---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "08 December 2020"
output: html_document
---
## Version
This is ``smooth`` package, v3.0.0. It introduces a new function and C++ code needed for it, together with extensive testthat examples to make sure that it works well.

## Test environments
* local ubuntu 19.10, R 4.0.3
* local Windows 10, R 4.0.3
* ubuntu 16.04.6 (on travis-ci), R 4.0.0
* win-builder (devel and release) - see a comment below
* rhub with rhub::check_for_cran() command

## R CMD check results
R CMD check results
checking installed package size ... NOTE
    installed size is 28.6Mb
    sub-directories of 1Mb or more:
      R      1.3Mb
      doc    3.9Mb
      libs  22.9Mb
0 errors | 0 warnings | 1 note

## win-builder check results
win-bulder quits with the message:
> * checking re-building of vignette outputs ... ERROR
> Check process probably crashed or hung up for 20 minutes ... killed
> Most likely this happened in the example checks (?),
> if not, ignore the following last lines of example output:

I've double checked, runnig the test on a separate MS Windows machine. It passes all the checks, only complaining about the 'doMC' package, which is not available for Windows. Not sure why this happens and what should be done with it.

## rhub checks
### Windows Server 2008 R2 SP1, R-devel, 32/64 bit
> Package suggested but not available: 'doMC'
> The suggested packages are required for a complete check.

This is because doMC is not available for Windows.

### Debian Linux, R-devel, GCC ASAN/UBSAN
Gives PREPERROR because `dependencies ‘greybox’, ‘forecast’ are not available for package ‘smooth’`. This does not have any explanation, because both packages are available on CRAN.

Everything was checked without major issues. The only thing is a note about the installed package size for different platforms (similar to R CMD check results).

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
