---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "16 April 2021"
output: html_document
---
## Version
This is ``smooth`` package, v3.1.1.

## Test environments
* local ubuntu 20.04, R 4.0.5
* ubuntu 16.04.6 (on travis-ci), R 4.0.2
* win-builder (devel and release)
* rhub with rhub::check_for_cran() command

## R CMD check results
R CMD check results
checking installed package size ... NOTE
    installed size is 25.2Mb
    sub-directories of 1Mb or more:
      R      1.2Mb
      doc    3.7Mb
      libs  19.8Mb
0 errors | 0 warnings | 1 note

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.


## rhub checks
### Windows Server 2008 R2 SP1, R-devel, 32/64 bit
> Package suggested but not available: 'doMC'
> The suggested packages are required for a complete check.
    
This is because doMC is not available for Windows.

### Debian Linux, R-devel, GCC ASAN/UBSAN
Gives PREPERROR:
> ERROR: compilation failed for package ‘forecast’

Not clear, why the compilation of `forecast` package failed, but smooth cannot be checked without it.


## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
