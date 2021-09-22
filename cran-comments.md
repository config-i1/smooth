---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "22 September 2021"
output: html_document
---
## Version
This is ``smooth`` package, v3.1.3.

## Test environments
* local ubuntu 20.04.2, R 4.1.1
* github actions
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

## Github actions
Successful checks for:

- Windows latest release with R 4.1.1
- MacOS latest macOS Catalina 10.15.7 with R 4.1.1
- Ubuntu 20.04.3 with R 4.1.1

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## rhub checks
### Windows Server 2008 R2 SP1, R-devel, 32/64 bit
> * checking package dependencies ... ERROR
> Package suggested but not available: 'doMC'

This is expected from Windows Server - doMC is not available for that platform.

### Debian Linux, R-devel, GCC ASAN/UBSAN
> ERROR: dependency ‘httr’ is not available for package ‘texreg’

Not clear why httr is not available. It is on CRAN.


## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
