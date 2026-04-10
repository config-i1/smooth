devtools::load_all(".")

xreg <- read.csv("python/tests/data/etsx_data.csv")

model <- adam(xreg, model="AAN", regressors="use", h=5, holdout=T, smoother="global")
model

forecast(model, h=5, newdata=tail(xreg,5))

coef(model)
