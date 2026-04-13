devtools::load_all(".")

xreg <- read.csv("python/tests/data/etsx_data.csv")

model <- adam(xreg, model="AAN", regressors="use", h=5, holdout=T, smoother="global")
model

forecast(model, h=5, newdata=tail(xreg,5))

coef(model)


y <- Mcomp::M3[[2568]]

model <- adam(y, model="NNN", lags=c(12),
              initial="back",
              orders=list(ar=c(1,1), i=c(1,1), ma=c(1,2)),
              print_level=3, maxeval=1)
model
