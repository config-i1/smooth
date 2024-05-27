---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "1 April 2024"
output: html_document
---

## Version
This is ``smooth`` package, v4.0.1.

## Note
I have changed my email from ivan@svetunkov.ru to ivan@svetunkov.com. Because of that, some checks gave this warning:

>New maintainer:
>  Ivan Svetunkov <ivan@svetunkov.com>
>Old maintainer(s):
>  Ivan Svetunkov <ivan@svetunkov.ru>


## Test environments
* local Ubuntu 22.04.4, R 4.3.3
* github actions
* win-builder (devel and release)
* rhub with rhub::check_for_cran() command

## R CMD check results
R CMD check results
checking installed package size ... NOTE
    installed size is 20.5Mb
    sub-directories of 1Mb or more:
      R      1.3Mb
      doc    3.3Mb
      libs  15.4Mb
0 errors | 0 warnings | 1 note

## Github actions
Successful checks for:

- Windows latest release with R 4.3.3
- MacOS latest macOS Monterey 12.6.8 with R 4.3.3
- Ubuntu latest with R 4.3.3

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## rhub checks
### Windows Server 2022, R-devel, 64 bit
> * checking package dependencies ... ERROR
> Package suggested but not available: 'doMC'

This is expected from Windows Server - doMC is not available for that platform.

All the resources are available online, not clear why the server cannot find some of them.

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
