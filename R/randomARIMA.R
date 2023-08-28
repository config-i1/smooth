# # library(smooth)
# # library(foreach)
#
# randArima <- function(y, lags=c(1,frequency(y)),
#                       orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2)),
#                       ic=c("AICc","AIC","BIC","BICc"),
#                       nsim=100, aggregate=median,
#                       h=10, interval="prediction", level=0.95, holdout=FALSE,
#                       parallel=FALSE, silent=TRUE, ...){
#     # y is the univariate data to forecast
#     # lags - lags of the model
#     # orders - maximum ARIMA orders
#     # ic - IC type to calculate (not used at the moment)
#     # nsim is the number of iterations
#     # aggregate defines function to use for aggregation
#     # h is the forecast horizons
#     # interval - type of the prediction interval
#     # level - confidence level for the interval
#     # holdout - logical, defining whether to use last h obs for the holdout
#     # parallel - either logical to do or not to do, or number of cores to use
#     # silent - defines whether to produce plot and messages or not
#     # ... - parameters passed to msarima
#     #
#     ## Example of application:
#     # test <- randArima(Mcomp::M3[[2568]], silent=F, nsim=40, h=18, parallel=TRUE)
#
#     # Dummy model and forecast to return proper smooth class
#     testForecast <- msarima(y, orders=c(0,1,1), constant=TRUE,
#                             holdout=holdout, h=h, ...) |>
#         forecast(h=h, interval=interval, level=level)
#
#     # Number of in-sample observations
#     obsInSample <- nobs(testForecast$model);
#
#     # Treat M-Competition data
#     if(inherits(y,"Mdata")){
#         h <- y$h;
#         holdout <- TRUE;
#         lags <- unique(c(1,frequency(y$x)));
#         obsInSample[] <- length(y$x);
#     }
#
#     #### Parallel calculations ####
#     # Check the parallel parameter and set the number of cores
#     if(is.numeric(parallel)){
#         nCores <- parallel;
#         parallel <- TRUE
#     }
#     else{
#         nCores <- min(parallel::detectCores() - 1, nsim);
#     }
#
#     # If this is parallel, then load the required packages
#     if(parallel){
#         if(!requireNamespace("foreach", quietly = TRUE)){
#             stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
#         }
#         if(!requireNamespace("parallel", quietly = TRUE)){
#             stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
#         }
#
#         # Check the system and choose the package to use
#         if(Sys.info()['sysname']=="Windows"){
#             if(requireNamespace("doParallel", quietly = TRUE)){
#                 cat("Setting up", nCores, "clusters using 'doParallel'...\n");
#                 cluster <- parallel::makeCluster(nCores);
#                 doParallel::registerDoParallel(cluster);
#             }
#             else{
#                 stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.",
#                      call. = FALSE);
#             }
#         }
#         else{
#             if(requireNamespace("doMC", quietly = TRUE)){
#                 doMC::registerDoMC(nCores);
#                 cluster <- NULL;
#             }
#             else if(requireNamespace("doParallel", quietly = TRUE)){
#                 cat("Setting up", nCores, "clusters using 'doParallel'...\n");
#                 cluster <- parallel::makeCluster(nCores);
#                 doParallel::registerDoParallel(cluster);
#             }
#             else{
#                 stop(paste0("Sorry, but in order to run the function in parallel, you need either ",
#                             "'doMC' (prefered) or 'doParallel' package."),
#                      call. = FALSE);
#             }
#         }
#     }
#     else{
#         cluster <- NULL;
#     }
#
#     # Information criteria
#     ic <- match.arg(ic,c("AICc","AIC","BIC","BICc"));
#     IC <- switch(ic,
#                  "AIC"=AIC,
#                  "AICc"=AICc,
#                  "BIC"=BIC,
#                  "BICc"=BICc);
#
#     # If the orders are provided without seasonal ones: orders=c(3,2,3)
#     if(!is.list(orders) && length(orders)==3){
#         orders <- list(ar=orders[1],i=orders[2],ma=orders[3])
#     }
#
#     # Don't expand to seasonal lags on non-seasonal data
#     if(all(lags==1)){
#         orders$ar <- orders$ar[1];
#         orders$i <- orders$i[1];
#         orders$ma <- orders$ma[1];
#     }
#
#     # Lengths of orders
#     ARLength <- length(orders$ar);
#     ILength <- length(orders$i);
#     MALength <- length(orders$ma);
#
#     #### Matrix of all ARIMA orders from 0 to max with/without constant ####
#     ordersTable <- cbind(as.matrix(expand.grid(sapply(unlist(orders), seq, 0))),
#                          constant=1);
#     ordersTable <- rbind(ordersTable,ordersTable);
#     nOrders <- nrow(ordersTable);
#     nVars <- ncol(ordersTable);
#     # Intercept values
#     ordersTable[(nOrders/2+1):nOrders,nVars] <- 0;
#     # Remove ARIMA(p,0,q) with no intercept - they will converge to zero
#     ordersTable <- ordersTable[!(ordersTable[,2]==0 & ordersTable[,nVars]==0),];
#
#     # The number of estimated parameters for each model + sigma
#     nParam <- apply(ordersTable,1,sum) + 1;
#
#     # The number of ARIMA components
#     ## This is maximum of (AR + I, MA) + Constant
#     nComponents <- pmax(ordersTable[,substr(colnames(ordersTable),1,2)=="ar",drop=FALSE] %*% lags +
#                                    ordersTable[,substr(colnames(ordersTable),1,1)=="i",drop=FALSE] %*% lags,
#                          ordersTable[,substr(colnames(ordersTable),1,2)=="ma",drop=FALSE] %*% lags) + ordersTable[,nVars];
#
#     # Remove orders that are not feasible on the given sample size
#     ordersTable <- ordersTable[(nParam + nComponents) < obsInSample,];
#     nOrders <- nrow(ordersTable);
#
#     # Function generates applies a random ARIMA and generates forecasts from it
#     randomForecaster <- function(...){
#         # floor is used to remove ARIMA(0,0,0)
#         randomRow <- floor(runif(1,1,nOrders));
#         # Prepare orders
#         orders$ar[] <- unlist(ordersTable[randomRow,1:ARLength]);
#         orders$i[] <- unlist(ordersTable[randomRow,ARLength+1:ILength]);
#         orders$ma[] <- unlist(ordersTable[randomRow,ARLength+ILength+1:MALength]);
#
#         # Apply random ARIMA
#         testModel <- msarima(y, orders=orders, holdout=holdout, h=h, lags=lags,
#                              constant=(ordersTable[randomRow,nVars]==1),
#                              ...);
#
#         # Generate forecasts
#         testForecast <- forecast(testModel, h=h, interval=interval, level=level);
#
#         return(list(fitted(testModel), testForecast$mean,
#                     testForecast$lower, testForecast$upper,
#                     ordersTable[randomRow,], IC(testModel)));
#     }
#
#     if(!silent){
#         cat("Starting the calculations... ");
#     }
#
#     if(parallel){
#         randomForecasts <- foreach(i=1:nsim) %dopar% {
#             return(randomForecaster());
#         }
#     }
#     else{
#         randomForecasts <- foreach(i=1:nsim) %do% {
#             return(randomForecaster());
#         }
#     }
#
#     if(!silent){
#         cat(" Done!\n")
#     }
#
#     nLevels <- length(level)
#
#     # Tables for point values
#     point <-
#         matrix(NA, h, nsim, dimnames=list(paste0("h",c(1:h)), NULL));
#
#     # Arrays for lower and upper values
#     lower <- upper <-
#         array(NA, c(h, nLevels, nsim),
#               dimnames=list(paste0("h",c(1:h)),
#                             paste0(sort(level)*100,"%-level"),
#                             NULL));
#
#     # Table for fitted values
#     fitted <-
#         matrix(NA, obsInSample, nsim);
#
#     # Matrix of all orders used
#     ordersUsed <- matrix(NA, nsim, nVars,
#                          dimnames=list(NULL, colnames(ordersTable)));
#
#     # Vector of ICs
#     ICs <- vector("numeric", nsim);
#
#     if(length(levels)>1){
#         dimnames(lower)[[2]] <- colnames(randomForecasts[[1]][[3]]);
#         dimnames(upper)[[2]] <- colnames(randomForecasts[[1]][[4]]);
#     }
#
#     # Record values
#     for(i in 1:nsim){
#         fitted[,i] <- randomForecasts[[i]][[1]]
#         point[,i] <- randomForecasts[[i]][[2]];
#         lower[,,i] <- randomForecasts[[i]][[3]];
#         upper[,,i] <- randomForecasts[[i]][[4]];
#         ordersUsed[i,] <- randomForecasts[[i]][[5]];
#         ICs[i] <- randomForecasts[[i]][[6]];
#     }
#
#     # AIC weights as an alternative aggregation procedure
#     icBest <- min(ICs)
#     ICw <- exp(-0.5*(ICs-icBest)) /
#         sum(exp(-0.5*(ICs-icBest)))
#
#     # Aggregate forecasts
#     if(any(identical(aggregate,mean),
#            identical(aggregate,median),
#            identical(aggregate,quantile))){
#         testForecast$model$fitted[] <- apply(fitted,1,aggregate);
#         testForecast$mean[] <- apply(point,1,aggregate);
#         testForecast$lower[] <- apply(lower,c(1,2),aggregate);
#         testForecast$upper[] <- apply(upper,c(1,2),aggregate);
#     }
#     else{
#         for(i in 1:obsInSample){
#             testForecast$model$fitted[i] <- aggregate(x=fitted[i,], w=ICw, ...);
#         }
#         for(i in 1:h){
#             testForecast$mean[i] <- aggregate(x=point[i,], w=ICw, ...);
#             for(j in 1:nLevels){
#                 testForecast$lower[i,j] <- aggregate(x=lower[i,j,], w=ICw, ...);
#                 testForecast$upper[i,j] <- aggregate(x=upper[i,j,], w=ICw, ...);
#             }
#         }
#     }
#
#     # Record tables with all point, lower and upper values
#     testForecast$forecasts <- list(mean=point, lower=lower, upper=upper,
#                                    orders=ordersUsed, ICs=ICs, ICw=ICw);
#
#     # Amend dummy model to make sense
#     testForecast$model$model <- "Random ARIMA";
#     testForecast$model$orders <- NULL;
#     testForecast$model$lags <- NULL;
#
#     if(!silent){
#         plot(testForecast);
#     }
#
#     # That's it. We are done :)
#     return(testForecast);
# }
