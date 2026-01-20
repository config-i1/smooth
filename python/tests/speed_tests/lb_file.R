library(Mcomp)
library(Tcomp)
library(forecast)
library(smooth)
# I work on Linux and use doMC. Substitute this with doParallel if you use Windows
library(doMC)
registerDoMC(detectCores())
# Create a small but neat function that will return a vector of error measures
errorMeasuresFunction <- function(object, holdout, insample){
    holdout <- as.vector(holdout);
    insample <- as.vector(insample);
    return(c(measures(holdout, object$mean, insample),
             mean(holdout < object$upper & holdout > object$lower),
             mean(object$upper-object$lower)/mean(insample),
             pinball(holdout, object$upper, 0.975)/mean(insample),
             pinball(holdout, object$lower, 0.025)/mean(insample),
             sMIS(holdout, object$lower, object$upper, mean(insample),0.95),
             object$timeElapsed))
}
# Datasets to use
datasets <- c(M1,M3,tourism)
datasetLength <- length(datasets)
# Types of models to try
methodsNames <- c("ETS", "Auto ARIMA",
                  "ADAM ETS Back", "ADAM ETS Opt", "ADAM ETS Two",
                  "ES Back", "ES Opt", "ES Two",
                  "ADAM ARIMA Back", "ADAM ARIMA Opt", "ADAM ARIMA Two",
                  "MSARIMA Back", "MSARIMA Opt", "MSARIMA Two",
                  "SSARIMA Back", "SSARIMA Opt", "SSARIMA Two",
                  "CES Back", "CES Opt", "CES Two",
                  "GUM Back", "GUM Opt", "GUM Two");
methodsNumber <- length(methodsNames);
test <- adam(datasets[[125]]);
testResults20250603 <- array(NA,c(methodsNumber,datasetLength,length(test$accuracy)+6),
                             dimnames=list(methodsNames, NULL,
                                           c(names(test$accuracy),
                                             "Coverage","Range",
                                             "pinballUpper","pinballLower","sMIS",
                                             "Time")));
#### ETS from forecast package ####
j <- 1;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="forecast") %dopar% {
  startTime <- Sys.time()
  test <- ets(datasets[[i]]$x);
  testForecast <- forecast(test, h=datasets[[i]]$h, level=95);
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### AUTOARIMA ####
j <- 2;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="forecast") %dopar% {
    startTime <- Sys.time()
    test <- auto.arima(datasets[[i]]$x);
    testForecast <- forecast(test, h=datasets[[i]]$h, level=95);
    testForecast$timeElapsed <- Sys.time() - startTime;
    return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ADAM ETS Backcasting ####
j <- 3;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- adam(datasets[[i]],"ZXZ", initial="back");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="pred");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ADAM ETS Optimal ####
j <- 4;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- adam(datasets[[i]],"ZXZ", initial="opt");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="pred");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ADAM ETS Two-stage ####
j <- 5;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- adam(datasets[[i]],"ZXZ", initial="two");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="pred");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ES Backcasting ####
j <- 6;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- es(datasets[[i]],"ZXZ", initial="back");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ES Optimal ####
j <- 7;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- es(datasets[[i]],"ZXZ", initial="opt");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ES Two-stage ####
j <- 8;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- es(datasets[[i]],"ZXZ", initial="two");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ADAM ARIMA Backcasting ####
j <- 9;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.adam(datasets[[i]], "NNN", initial="back", distribution=c("dnorm"));
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="pred");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ADAM ARIMA Optimal ####
j <- 10;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.adam(datasets[[i]], "NNN", initial="opt", distribution=c("dnorm"));
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="pred");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### ADAM ARIMA Two-stage ####
j <- 11;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.adam(datasets[[i]], "NNN", initial="two", distribution=c("dnorm"));
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="pred");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### MSARIMA Backcasting ####
j <- 12;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.msarima(datasets[[i]], initial="back");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### MSARIMA Optimal ####
j <- 13;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.msarima(datasets[[i]], initial="opt");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### MSARIMA Two-stage ####
j <- 14;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.msarima(datasets[[i]], initial="two");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### SSARIMA Backcasting ####
j <- 15;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.ssarima(datasets[[i]], initial="back");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### SSARIMA Optimal ####
j <- 16;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="forecast") %dopar% {
    startTime <- Sys.time()
    test <- auto.ssarima(datasets[[i]], initial="opt");
    testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
    testForecast$timeElapsed <- Sys.time() - startTime;
    return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### SSARIMA Two-stage ####
j <- 17;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="forecast") %dopar% {
    startTime <- Sys.time()
    test <- auto.ssarima(datasets[[i]], initial="two");
    testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
    testForecast$timeElapsed <- Sys.time() - startTime;
    return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### CES Backcasting ####
j <- 18;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.ces(datasets[[i]], initial="back");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### CES Optimal ####
j <- 19;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.ces(datasets[[i]], initial="opt");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### CES Two-stage ####
j <- 20;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.ces(datasets[[i]], initial="two");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### GUM Backcasting ####
j <- 21;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.gum(datasets[[i]], initial="back");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### GUM Optimal ####
j <- 22;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.gum(datasets[[i]], initial="opt");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);
#### GUM Two-stage ####
j <- 23;
result <- foreach(i=1:datasetLength, .combine="cbind", .packages="smooth") %dopar% {
  startTime <- Sys.time()
  test <- auto.gum(datasets[[i]], initial="two");
  testForecast <- forecast(test, h=datasets[[i]]$h, interval="parametric");
  testForecast$timeElapsed <- Sys.time() - startTime;
  return(errorMeasuresFunction(testForecast, datasets[[i]]$xx, datasets[[i]]$x));
}
testResults20250603[j,,] <- t(result);