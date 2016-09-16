## ----global_options, include=FALSE---------------------------------------
knitr::opts_chunk$set(fig.width=6, fig.height=4, fig.path='Figs/', fig.show='hold',
                      warning=FALSE, message=FALSE)

## ----load_libraries, message=FALSE, warning=FALSE------------------------
require(smooth)

## ----sim_es_ANN----------------------------------------------------------
ourSimulation <- sim.es("ANN", frequency=12, obs=120)

## ----sim_es_ANN_plot-----------------------------------------------------
plot(ourSimulation$data)

## ----sim_es_MAdM---------------------------------------------------------
ourSimulation <- sim.es("MAdM", frequency=12, obs=120, phi=0.95, persistence=c(0.1,0.05,0.01))
plot(ourSimulation$data)

## ----sim_es_MAdM_lnorm---------------------------------------------------
ourSimulation <- sim.es("MAdM", frequency=12, obs=120, phi=0.95, persistence=c(0.1,0.05,0.01), randomizer="rlnorm", meanlog=0, sdlog=0.015)
plot(ourSimulation$data)

## ----sim_es_iMNN---------------------------------------------------------
ourSimulation <- sim.es("MNN", frequency=12, obs=120, iprob=0.2, initial=10, persistence=0.1)
plot(ourSimulation$data)

## ----sim_es_iMNN_50------------------------------------------------------
ourSimulation <- sim.es("MNN", frequency=12, obs=120, iprob=0.2, initial=10, persistence=0.1, nsim=50)

## ----simulate_smooth_es--------------------------------------------------
x <- ts(rnorm(100,120,15),frequency=12)
ourModel <- es(x, h=18, silent=TRUE)
ourData <- simulate(ourModel,nsim=50,obs=100)

## ----simulate_smooth_es_compare------------------------------------------
par(mfcol=c(1,2))
plot(x)
plot(ourData$data[,1])
par(mfcol=c(1,1))

