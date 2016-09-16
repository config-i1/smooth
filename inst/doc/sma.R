## ----global_options, include=FALSE---------------------------------------
knitr::opts_chunk$set(fig.width=6, fig.height=4, fig.path='Figs/', fig.show='hold',
                      warning=FALSE, message=FALSE)

## ----load_libraries, message=FALSE, warning=FALSE------------------------
require(smooth)
require(Mcomp)

## ----sma_N2457-----------------------------------------------------------
sma(M3$N2457$x, h=18)

## ----sma_N2568-----------------------------------------------------------
sma(M3$N2568$x, h=18)

