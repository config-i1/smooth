---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "30 March 2022"
output: html_document
---

## Version
This is ``smooth`` package, v3.1.6.

**I've removed some tests to speed up the checks on CRAN. Hopefully, this addresses the issue flagged by Uwe in the previous submission.**

## Test environments
* local ubuntu 20.04.3, R 4.1.2
* github actions
* win-builder (devel and release)
* rhub with rhub::check_for_cran() command

## R CMD check results
R CMD check results
checking installed package size ... NOTE
    installed size is 27.0Mb
    sub-directories of 1Mb or more:
      R      1.2Mb
      doc    3.4Mb
      libs  21.9Mb
0 errors | 0 warnings | 1 note

## Github actions
Successful checks for:

- Windows latest release with R 4.1.2
- MacOS latest macOS Catalina 10.15.7 with R 4.1.2
- Ubuntu 20.04.3 with R 4.1.2

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## rhub checks
### Windows Server 2022, R-devel, 64 bit
> * checking package dependencies ... ERROR
> Package suggested but not available: 'doMC'

This is expected from Windows Server - doMC is not available for that platform.

### Debian Linux, R-devel, GCC ASAN/UBSAN
> ERROR: compilation failed for package ‘Rcpp’

Not clear why the compilation failed


## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
