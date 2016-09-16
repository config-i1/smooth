## ----global_options, include=FALSE---------------------------------------
knitr::opts_chunk$set(fig.width=6, fig.height=4, fig.path='Figs/', fig.show='hold',
                      warning=FALSE, message=FALSE)

## ----load_libraries, message=FALSE, warning=FALSE------------------------
require(smooth)
require(Mcomp)

## ----es_N2457------------------------------------------------------------
es(M3$N2457$x, h=18, holdout=TRUE)

## ----es_N2457_with_intervals---------------------------------------------
es(M3$N2457$x, h=18, holdout=TRUE, intervals=TRUE)

## ----es_N2457_save_model-------------------------------------------------
ourModel <- es(M3$N2457$x, h=18, holdout=TRUE, silent="all")

## ----es_N2457_reuse_model------------------------------------------------
es(M3$N2457$x, model=ourModel, h=18, holdout=FALSE, intervals=TRUE, intervalsType="n", level=0.93)

## ----es_N2457_reuse_model_parts------------------------------------------
es(M3$N2457$x, model="MNN", h=18, holdout=FALSE, initial=ourModel$initial, silent="graph")
es(M3$N2457$x, model="MNN", h=18, holdout=FALSE, persistence=ourModel$persistence, silent="graph")

## ----es_N2457_set_initial------------------------------------------------
es(M3$N2457$x, model="MNN", h=18, holdout=FALSE, initial=1500, silent="graph")

## ----es_N2457_aMSTFE-----------------------------------------------------
es(M3$N2457$x, h=18, holdout=TRUE, cfType="aMSTFE", bounds="a", ic="BIC", intervals=TRUE)

## ----es_N2457_pool-------------------------------------------------------
es(M3$N2457$x, model=c("ANN","AAN","AAdN","ANA","AAA","AAdA"), h=18, holdout=TRUE, silent="graph")
es(M3$N2457$x, model="CCN", h=18, holdout=TRUE, silent="graph")

## ----es_N2457_xreg_create------------------------------------------------
x <- cbind(rnorm(length(M3$N2457$x),50,3),rnorm(length(M3$N2457$x),100,7))

## ----es_N2457_xreg-------------------------------------------------------
es(M3$N2457$x, model="ZZZ", h=18, holdout=TRUE, xreg=x)

## ----es_N2457_xreg_update------------------------------------------------
es(M3$N2457$x, model="ZZZ", h=18, holdout=TRUE, xreg=x, updateX=TRUE)

