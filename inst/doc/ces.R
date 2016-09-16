## ----global_options, include=FALSE---------------------------------------
knitr::opts_chunk$set(fig.width=6, fig.height=4, fig.path='Figs/', fig.show='hold',
                      warning=FALSE, message=FALSE)

## ----load_libraries, message=FALSE, warning=FALSE------------------------
require(smooth)
require(Mcomp)

## ----ces_N2457-----------------------------------------------------------
ces(M3$N2457$x, h=18, holdout=TRUE)

## ----auto_ces_N2457------------------------------------------------------
auto.ces(M3$N2457$x, h=18, holdout=TRUE, intervals=TRUE)

## ----auto_ces_N2457_optimal----------------------------------------------
auto.ces(M3$N2457$x, h=18, holdout=TRUE, initial="o", intervals=TRUE, intervalsType="s")

## ----es_N2457_xreg_create------------------------------------------------
x <- cbind(rnorm(length(M3$N2457$x),50,3),rnorm(length(M3$N2457$x),100,7))

## ----auto_ces_N2457_xreg_simple------------------------------------------
auto.ces(M3$N2457$x, h=18, holdout=TRUE, xreg=x, intervals=TRUE)

## ----auto_ces_N2457_xreg_update------------------------------------------
auto.ces(M3$N2457$x, h=18, holdout=TRUE, xreg=x, updateX=TRUE, intervals=TRUE)

