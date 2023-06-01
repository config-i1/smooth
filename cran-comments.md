---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "01 June 2023"
output: html_document
---

## Version
This is ``smooth`` package, v3.2.1.


## Test environments
* local Ubuntu 22.04.2, R 4.3.0
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

- Windows latest release with R 4.3.0
- MacOS latest macOS Big Sur 10.16 with R 4.2.2
- Ubuntu 20.04.6 with R 4.3.0

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## rhub checks
### Windows Server 2022, R-devel, 64 bit
> * checking package dependencies ... ERROR
> Package suggested but not available: 'doMC'

This is expected from Windows Server - doMC is not available for that platform.

All the resources are available online, not clear why the server cannot find them.

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
