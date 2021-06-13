---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "14 June 2021"
output: html_document
---
## Version
This is ``smooth`` package, v3.1.2.

## Test environments
* local ubuntu 20.04.2, R 4.1.0
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

- Windows latest release with R 4.1.0
- MacOS latest macOS Catalina 10.15.7 with R 4.1.0
- Ubuntu 20.04 with R 4.1.0
- Ubuntu 20.04 with R 4.0.5
- Ubuntu 20.04 with R unstable 2021-06-09 r80471

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## rhub checks
### Windows Server 2008 R2 SP1, R-devel, 32/64 bit
> Error: Bioconductor does not yet build and check packages for R version 4.2;

This has nothing to do with the package

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
