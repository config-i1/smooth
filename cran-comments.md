---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "02 April 2025"
output: html_document
---

## Version
This is ``smooth`` package, v4.2.0

This is a try to fix the errors on CRAN.

## Test environments
* local Ubuntu 24.04.1, R 4.4.1
* github actions
* win-builder (devel and release)
* rhub v2

## R CMD check results
R CMD check results
checking installed package size ... NOTE
    installed size is 20.8Mb
    sub-directories of 1Mb or more:
      R      1.2Mb
      doc    3.3Mb
      libs  15.7Mb
0 errors | 0 warnings | 1 note

## Github actions
Successful checks for:

- Windows latest release with R 4.4.1
- MacOS latest macOS Sonoma 14.6.1 with R 4.4.1
- Ubuntu 22.04.5 LTS with R 4.4.1

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## R-hub
All is fine

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
