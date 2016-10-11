context("Tests for ssarima() function");

# Basic SSARIMA selection
testModel <- auto.ssarima(Mcomp::M3$N1234$x, silent=TRUE);
test_that("Test if Auto SSARIMA selected correct model for N1234$x", {
    expect_equal(testModel$model, "ARIMA(0,1,3) with drift");
})

# Reuse previous SSARIMA
test_that("Reuse previous GES on N1234$x", {
    expect_equal(ssarima(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf, testModel$cf);
})

# Test some crazy order of SSARIMA
testModel <- ssarima(Mcomp::M3$N2568$x, ar.orders=c(1,1,0), i.orders=c(1,0,1), ma.orders=c(0,1,1), lags=c(1,6,12), h=18, holdout=TRUE, initial="o", silent=TRUE, intervals=TRUE)
test_that("Test if crazy order SSARIMA was estimated on N1234$x", {
    expect_equal(testModel$model, "SARIMA(1,1,0)[1](1,0,1)[6](0,1,1)[12]");
})

# Automatically select SSARIMA
testModel <- auto.ssarima(Mcomp::M3$N2568$x, silent=TRUE, ic="AIC");
# Define orders of the model
SSARIMAModel <- testModel$model;
arima.orders <- paste0(c("",substring(SSARIMAModel,unlist(gregexpr("\\(",SSARIMAModel))+1,unlist(gregexpr("\\)",SSARIMAModel))-1),"")
                       ,collapse=";");
comas <- unlist(gregexpr("\\,",arima.orders));
semicolons <- unlist(gregexpr("\\;",arima.orders));
ar.orders <- as.numeric(substring(arima.orders,semicolons[-length(semicolons)]+1,comas[2*(1:(length(comas)/2))-1]-1));
i.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))-1]+1,comas[2*(1:(length(comas)/2))-1]+1));
ma.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))]+1,semicolons[-1]-1));
if(any(unlist(gregexpr("\\[",SSARIMAModel))!=-1)){
    lags <- as.numeric(substring(SSARIMAModel,unlist(gregexpr("\\[",SSARIMAModel))+1,unlist(gregexpr("\\]",SSARIMAModel))-1));
}else{
    lags <- 1;
}
# Test how different passed values are accepted by SSARIMA
test_that("Test initials, AR, MA and constant of SSARIMA on N2568$x", {
    expect_equal(ssarima(Mcomp::M3$N2568$x, ar.orders=ar.orders, i.orders=i.orders, ma.orders=ma.orders, lags=lags, constant=TRUE, initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(ssarima(Mcomp::M3$N2568$x, ar.orders=ar.orders, i.orders=i.orders, ma.orders=ma.orders, lags=lags, constant=TRUE, AR=testModel$AR, silent=TRUE)$AR, testModel$AR);
    expect_equal(ssarima(Mcomp::M3$N2568$x, ar.orders=ar.orders, i.orders=i.orders, ma.orders=ma.orders, lags=lags, constant=TRUE, transition=testModel$MA, silent=TRUE)$MA, testModel$MA);
    expect_equal(ssarima(Mcomp::M3$N2568$x, ar.orders=ar.orders, i.orders=i.orders, ma.orders=ma.orders, lags=lags, constant=testModel$constant, silent=TRUE)$constant, testModel$constant);
})

# Test exogenous (normal + updateX) with SSARIMA
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- ssarima(y, h=18, holdout=TRUE, xreg=x, updateX=TRUE, silent=TRUE, cfType="aMSTFE", intervals=TRUE)
test_that("Check exogenous variables for SSARIMA on N1457", {
    expect_equal(suppressWarnings(ssarima(y, h=18, holdout=TRUE, xreg=x, silent=TRUE)$model), testModel$model);
    expect_equal(suppressWarnings(forecast(testModel, h=18, holdout=FALSE)$model), testModel$model);
})
