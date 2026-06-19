---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "20 June 2026"
output: html_document
---

## Version
This is ``smooth`` package, v4.5.0

This release retires the legacy occurrence-ETS code path. Four exports
have been removed: ``oes_old()``, ``oesg_old()``, ``is.oes()`` and
``is.oesg()`` (with their `.Rd` pages). Five unreachable S3 methods
(``forecast.oes``, ``plot.oes``, ``print.oes``, ``pointLik.oes``,
``modelType.oesg``) and one C++ translation unit
(``src/ssOccurrence.cpp``, four ``RcppExport`` wrappers) have also been
dropped. The new ``oes()`` / ``oesg()`` wrappers (backed by ``om()`` /
``omg()``) cover every documented feature of the retired path and are
unchanged.


## Test environments
* local Ubuntu 26.04, R 4.5.2
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

- Windows latest release with latest R
- MacOS 15.7.3 with latest R
- Ubuntu 24.04.3 LTS with latest R

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## R-hub
All is fine

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
