auto.ssarima <- function(data,ar.max=c(3), i.max=c(2), ma.max=c(3), lags=c(1),
                         IC=c("AICc","AIC","BIC"),
                         CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                         h=10, holdout=FALSE, silent=FALSE){
# Start measuring the time of calculations
    start.time <- Sys.time();

    IC <- IC[1];
    CF.type <- CF.type[1];

    if(any(is.complex(c(ar.max,i.max,ma.max,lags)))){
        stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
    }

    if(any(c(ar.max,i.max,ma.max)<0)){
        stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
    }

    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

# Order things, so we would deal with highest level of seasonality first
    lags <- sort(lags,decreasing=TRUE);
    ar.max <- ar.max[order(lags,decreasing=TRUE)];
    i.max <- i.max[order(lags,decreasing=TRUE)];
    ma.max <- ma.max[order(lags,decreasing=TRUE)];
# 2 stands for constant + d=0
    models.number <- sum(ar.max,i.max,ma.max) + 2;
    test.models <- list(NA);
    test.ICs <- rep(NA,max(ar.max,i.max,ma.max)+1);
    test.ICs.all <- rep(NA,models.number);
    m <- 0;

    test.lags <- ma.test <- ar.test <- i.test <- rep(0,length(lags));
    ar.best <- ma.best <- i.best <- rep(0,length(lags));

    if(silent==FALSE){
        cat("Estimation progress:     ");
    }

    for(seasSelect in 1:length(lags)){
# Start from highest lag and include one more each iteration
        test.lags[seasSelect] <- lags[seasSelect];
##### Loop for differences
        if(i.max[seasSelect]!=0){
            for(iSelect in (seasSelect-1):i.max[seasSelect]){
                m <- m + 1;
                if(silent==FALSE){
                    cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                    cat(paste0(round((m)/models.number,2)*100,"%"));
                }
# Update the iSelect in i.test preserving the previous values
                i.test[seasSelect] <- iSelect;
                test.models[[m]] <- ssarima(data,ar.orders=rev(ar.best),i.orders=rev(i.test),ma.orders=rev(ma.best),lags=rev(test.lags),
                                            h=h,holdout=holdout,constant=TRUE,silent=TRUE,CF.type=CF.type);
                test.ICs[iSelect+1] <- test.models[[m]]$ICs[IC];
                test.ICs.all[m] <- test.models[[m]]$ICs[IC];
            }
# Save the best differences
            i.best[seasSelect] <- i.test[seasSelect] <- c(0:i.max[seasSelect])[which(test.ICs==min(test.ICs,na.rm=TRUE))];
# Sort in order to put the best one on the first place
            test.ICs <- sort(test.ICs,decreasing=FALSE)
        }

##### Loop for AR
        if(ar.max[seasSelect]!=0){
            for(arSelect in 1:ar.max[seasSelect]){
                m <- m + 1;
                if(silent==FALSE){
                    cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                    cat(paste0(round((m)/models.number,2)*100,"%"));
                }
# Update the iSelect in i.test preserving the previous values
                ar.test[seasSelect] <- arSelect;
                test.models[[m]] <- ssarima(data,ar.orders=rev(ar.test),i.orders=rev(i.best),ma.orders=rev(ma.best),lags=rev(test.lags),
                                            h=h,holdout=holdout,constant=TRUE,silent=TRUE,CF.type=CF.type);
                test.ICs[arSelect+1] <- test.models[[m]]$ICs[IC];
                test.ICs.all[m] <- test.models[[m]]$ICs[IC];
            }
# Save the best AR
            ar.best[seasSelect] <- ar.test[seasSelect] <- c(0:ar.max[seasSelect])[which(test.ICs==min(test.ICs,na.rm=TRUE))];
# Sort in order to put the best one on the first place
            test.ICs <- sort(test.ICs,decreasing=FALSE)
        }

##### Loop for MA
        if(ma.max[seasSelect]!=0){
            for(maSelect in 1:ma.max[seasSelect]){
                m <- m + 1;
                if(silent==FALSE){
                    cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                    cat(paste0(round((m)/models.number,2)*100,"%"));
                }
# Update the iSelect in i.test preserving the previous values
                ma.test[seasSelect] <- maSelect;
                test.models[[m]] <- ssarima(data,ar.orders=rev(ar.best),i.orders=rev(i.best),ma.orders=rev(ma.test),lags=rev(test.lags),
                                            h=h,holdout=holdout,constant=TRUE,silent=TRUE,CF.type=CF.type);
                test.ICs[maSelect+1] <- test.models[[m]]$ICs[IC];
                test.ICs.all[m] <- test.models[[m]]$ICs[IC];
            }
# Save the best MA
            ma.best[seasSelect] <- ma.test[seasSelect] <- c(0:ma.max[seasSelect])[which(test.ICs==min(test.ICs,na.rm=TRUE))];
# Sort in order to put the best one on the first place
            test.ICs <- sort(test.ICs,decreasing=FALSE)
        }
    }
    m <- m + 1;
    if(silent==FALSE){
        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
        cat(paste0(round((m)/models.number,2)*100,"%"));
    }

# Test the constant
    if(any(c(ar.best,i.best,ma.best)!=0)){
        test.models[[m]] <- ssarima(data,ar.orders=rev(ar.best),i.orders=rev(i.best),ma.orders=rev(ma.best),lags=rev(test.lags),
                                    h=h,holdout=holdout,constant=FALSE,silent=TRUE,CF.type=CF.type);
    test.ICs[2] <- test.models[[m]]$ICs[IC];
    test.ICs.all[m] <- test.models[[m]]$ICs[IC];
    }

    constant <- c(TRUE,FALSE)[which(test.ICs[1:2]==min(test.ICs[1:2],na.rm=TRUE))];
    test.ICs <- sort(test.ICs,decreasing=FALSE)

    best.model <- test.models[[which(test.ICs.all==test.ICs[1])[1]]];

    if(silent==FALSE){
        cat("... Done! \n");
    }

    if(silent==FALSE){
        n.components <- max(ar.best %*% lags + i.best %*% lags,ma.best %*% lags);
        s2 <- sum(best.model$residuals^2)/(n.components + constant + 1 + sum(ar.best) + sum(ma.best));
        xreg <- NULL;
        go.wild <- FALSE;
        CF.objective <- best.model$CF;
        intervals <- FALSE;
        int.type <- "p";
        int.w <- 0.95;
        ICs <- best.model$ICs;
        insideintervals <- NULL;
        errormeasures <- best.model$accuracy;

        ssoutput(Sys.time() - start.time, best.model$model, persistence=NULL, transition=NULL, measurement=NULL,
            phi=NULL, ARterms=best.model$AR, MAterms=best.model$MA, const=best.model$constant, A=NULL, B=NULL,
            n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
            CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
            int.type=int.type, int.w=int.w, ICs=ICs,
            holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);

        graphmaker(actuals=data,forecast=best.model$forecast,fitted=best.model$fitted,
                   int.w=0.95,legend=TRUE,main=best.model$model);
    }

    return(best.model);
}
