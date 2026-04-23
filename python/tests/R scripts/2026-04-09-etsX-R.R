devtools::load_all(".")

xreg <- read.csv("python/tests/data/etsx_data.csv")

model <- adam(xreg, model="AAN", regressors="use", h=5, holdout=T, smoother="global")
model

forecast(model, h=5, newdata=tail(xreg,5))

coef(model)


y <- Mcomp::M3[[2568]]

model <- adam(y$x, model="ZXZ", lags=c(1,12),
              initial="back", h=18, holdout=T,
              # orders=list(ar=c(1,1), i=c(1,1), ma=c(1,2)),
              print_level=0, maxeval=NULL)
model

auto.adam(y, model="ANN", lags=c(1,12),
              initial="back",
              orders=list(ar=c(3,3), i=c(2,2), ma=c(3,3), select=T)
            )

plot(model,7)


devtools::test(filter="adam_baseline") 

devtools::test()


model <- adam(Mcomp::M3[[2568]]$x, model="CXC", h=18, holdout=T)
model
plot(forecast(model, h=18, interval="pred"))

forecast(model, h=18, interval="pred")
