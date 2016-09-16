## ----global_options, include=FALSE---------------------------------------
knitr::opts_chunk$set(fig.width=6, fig.height=4, fig.path='Figs/', fig.show='hold',
                      warning=FALSE, message=FALSE)

## ----load_libraries, message=FALSE, warning=FALSE------------------------
require(smooth)
require(Mcomp)

## ----ges_N2457-----------------------------------------------------------
ges(M3$N2457$x, h=18, holdout=TRUE)

## ----ges_N2457_2[1]_1[12]------------------------------------------------
ges(M3$N2457$x, h=18, holdout=TRUE, orders=c(2,1), lags=c(1,12))

## ----ges_N2457_1[1]------------------------------------------------------
ges(M3$N2457$x, h=18, holdout=TRUE, orders=c(1), lags=c(1), intervals=TRUE)

## ----ges_N2457_predefined------------------------------------------------
	transition <- matrix(c(1,0,0,1,1,0,0,0,1),3,3)
	measurement <- c(1,1,1)
	ges(M3$N2457$x, h=18, holdout=TRUE, orders=c(2,1), lags=c(1,12), transition=transition, measurement=measurement)

