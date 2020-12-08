---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "08 December 2020"
output: html_document
---
## Version
This is ``smooth`` package, v3.0.0.

## Update to the previous submission
I've removed links to the functions, which did not pass the test on r-devel-linux-x86_64-debian-gcc.

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
Fixed the error related to the testthat on Windows (mentioned here: https://www.r-project.org/nosvn/R.check/r-release-windows-ix86+x86_64/smooth-00check.html)
Now, all seems to be okay.

## rhub checks
### Windows Server 2008 R2 SP1, R-devel, 32/64 bit
> Package suggested but not available: 'doMC'
> The suggested packages are required for a complete check.

This is because doMC is not available for Windows.

Everything was checked without major issues. The only thing is a note about the installed package size for different platforms (similar to R CMD check results).

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
