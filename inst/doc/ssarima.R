## ----global_options, include=FALSE---------------------------------------
knitr::opts_chunk$set(fig.width=6, fig.height=4, fig.path='Figs/', fig.show='hold',
                      warning=FALSE, message=FALSE)

## ----load_libraries, message=FALSE, warning=FALSE------------------------
require(smooth)
require(Mcomp)

## ----ssarima_N2457-------------------------------------------------------
ssarima(M3$N2457$x, h=18)

## ----auto_ssarima_N2457--------------------------------------------------
auto.ssarima(M3$N2457$x, h=18)

## ----auto_ssarima_N1683--------------------------------------------------
auto.ssarima(M3$N1683$x, h=18, initial="backcasting")
auto.ssarima(M3$N1683$x, h=18, initial="optimal")

## ----es_N2457_xreg_create------------------------------------------------
x <- cbind(rnorm(length(M3$N2457$x),50,3),rnorm(length(M3$N2457$x),100,7))

## ----auto_ssarima_N2457_xreg---------------------------------------------
ourModel <- auto.ssarima(M3$N2457$x, h=18, holdout=TRUE, xreg=x, updateX=TRUE)

## ----auto_ssarima_N2457_xreg_update--------------------------------------
ssarima(M3$N2457$x, model=ourModel, h=18, holdout=FALSE, xreg=x, updateX=TRUE, intervals=TRUE)

