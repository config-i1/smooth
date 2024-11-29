# arimaCompact <- function(y, lags=c(1,frequency(y)), ic=c("AICc","AIC","BIC","BICc"), ...){
#
#     # Start measuring the time of calculations
#     startTime <- Sys.time();
#
#     # If there are no lags for the basic components, correct this.
#     if(sum(lags==1)==0){
#         lags <- c(1,lags);
#     }
#
#     orderLength <- length(lags);
#     ic <- match.arg(ic);
#     IC <- switch(ic,
#                  "AIC"=AIC,
#                  "AICc"=AICc,
#                  "BIC"=BIC,
#                  "BICc"=BICc);
#
#     # We consider the following list of models:
#     # ARIMA(0,1,1), (1,1,2), (0,2,2),
#     # ARIMA(0,0,0)+c, ARIMA(0,1,1)+c,
#     # seasonal orders (0,1,1), (1,1,2), (0,2,2)
#     # And all combinations between seasonal and non-seasonal parts
#     #
#     # Encode all non-seasonal parts
#     nNonSeasonal <- 5
#     arimaNonSeasonal <- matrix(c(0,0,0,1, 0,1,1,0, 0,1,1,1, 1,1,2,0, 0,2,2,0), nNonSeasonal,4,
#                                dimnames=list(NULL, c("ar","i","ma","const")), byrow=TRUE)
#     # Encode all seasonal parts ()
#     nSeasonal <- 4
#     arimaSeasonal <- matrix(c(0,0,0, 0,1,1, 1,1,2, 0,2,2), nSeasonal,3,
#                                dimnames=list(NULL, c("sar","si","sma")), byrow=TRUE)
#
#     # Check all the models in the pool
#     testModels <- vector("list", nSeasonal*nNonSeasonal);
#     stop <- FALSE;
#     m <- 1;
#     for(i in 1:nSeasonal){
#         for(j in 1:nNonSeasonal){
#             testModels[[m]] <- msarima(y, orders=list(ar=c(arimaNonSeasonal[j,1],arimaSeasonal[i,1]),
#                                                       i=c(arimaNonSeasonal[j,2],arimaSeasonal[i,2]),
#                                                       ma=c(arimaNonSeasonal[j,3],arimaSeasonal[i,3])),
#                                        constant=arimaNonSeasonal[j,4]==1, lags=lags, ...);
#             # If SARIMA(0,1,1)(0,1,1) is worse than ARIMA(0,1,1), don't check other seasonal models
#             # If SARIMA(0,1,1)(1,1,2) is worse than SARIMA(0,1,1)(0,1,1), stop
#             # etc
#             if(j==1 && i>1){
#                 if(IC(testModels[[m-nNonSeasonal]])<IC(testModels[[m]])){
#                     stop[] <- TRUE;
#                     break;
#                 }
#             }
#             m[] <- m+1;
#         }
#         if(stop){
#             break;
#         }
#     }
#
#     # Remove not estimated models
#     nullModels <- sapply(testModels, is.null);
#     if(any(nullModels)){
#         testModels <- testModels[!nullModels];
#     }
#     # Find the best one
#     m <- which.min(sapply(testModels, IC));
#     # Amend computational time
#     testModels[[m]]$timeElapsed <- Sys.time()-startTime;
#
#     return(testModels[[m]]);
# }
