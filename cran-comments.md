---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "05 January 2021"
output: html_document
---
## Version
This is ``smooth`` package, v3.0.0. It introduces a new function and C++ code needed for it, together with extensive testthat examples to make sure that it works well.

## Test environments
* local ubuntu 20.04, R 4.0.3
* ubuntu 16.04.6 (on travis-ci), R 4.0.3
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
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

>** running examples for arch 'i386' ... [60s] NOTE
>Examples with CPU (user + system) or elapsed time > 10s
>      user system elapsed
>adam 13.78   0.08   13.85
>** running examples for arch 'x64' ... [57s] NOTE
>Examples with CPU (user + system) or elapsed time > 10s
>      user system elapsed
>adam 13.19   0.08   13.31

Not sure what has happened - the updated version of the function introduces improvements in terms of speed (based on microbenchmark tests), so this is unexpected.


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
