auto.ssarima <- function(data,ar.max=c(3), i.max=c(2), ma.max=c(3), lags=c(1),
                         initial=c("backcasting","optimal"), IC=c("AICc","AIC","BIC"),
                         CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                         h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                         int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                         silent=FALSE, legend=TRUE){
# Start measuring the time of calculations
    start.time <- Sys.time();

# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

# This is the critical minimum needed in order to at least fit ARIMA(0,0,0) with constant
    if(obs < 4){
        stop("Sorry, but your sample is too small. Come back when you have at least 4 observations...",call.=FALSE);
    }

# Check the provided vector of initials: length and provided values.
    if(is.character(initial)){
        initial <- substring(initial[1],1,1);
        if(initial!="o" & initial!="b"){
            warning("You asked for a strange initial value. We don't do that here. Switching to optimal.",call.=FALSE,immediate.=TRUE);
            initial <- "o";
        }
        fittertype <- initial;
        initial <- NULL;
    }
    else if(is.null(initial)){
        message("Initial value is not selected. Switching to optimal.");
        fittertype <- "o";
    }
    else{
        message("Predefinde initials don't go well with automatic model selection. Switching to optimal.");
        fittertype <- "o";
    }

    IC <- IC[1];
    CF.type <- CF.type[1];
    int.type <- int.type[1];

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
    lags <- sort(lags,decreasing=FALSE);
    ar.max <- ar.max[order(lags,decreasing=FALSE)];
    i.max <- i.max[order(lags,decreasing=FALSE)];
    ma.max <- ma.max[order(lags,decreasing=FALSE)];
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

##### Loop for differences
    if(any(i.max!=0)){
        for(seasSelect in 1:length(lags)){
            test.lags[seasSelect] <- lags[seasSelect];
            if(i.max[seasSelect]!=0){
                for(iSelect in (seasSelect-1):i.max[seasSelect]){
                    m <- m + 1;
                    if(silent==FALSE){
                        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                        cat(paste0(round((m)/models.number,2)*100,"%"));
                    }
# Update the iSelect in i.test preserving the previous values
                    i.test[seasSelect] <- iSelect;
                    n.param <- 1 + max(ar.best %*% lags + i.test %*% lags,ma.best %*% lags) +
                            sum(ar.best) + sum(ma.best) + 1;
                    if(n.param > obs - 2){
                        test.models[[m]] <- NA;
                        test.ICs[iSelect+1] <- Inf;
                        test.ICs.all[m] <- Inf;
                        next
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.test),ma.orders=(ma.best),lags=(test.lags),
                                                constant=TRUE,initial=fittertype,CF.type=CF.type,
                                                h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                                int.type=int.type,silent=TRUE);
                    test.ICs[iSelect+1] <- test.models[[m]]$ICs[IC];
                    test.ICs.all[m] <- test.models[[m]]$ICs[IC];
                }
# Save the best differences
                i.best[seasSelect] <- i.test[seasSelect] <- c(0:i.max[seasSelect])[which(test.ICs==min(test.ICs,na.rm=TRUE))[1]];
# Sort in order to put the best one on the first place
                test.ICs <- sort(test.ICs,decreasing=FALSE)
            }
        }
    }


##### Loop for AR
    if(any(ar.max!=0)){
        for(seasSelect in 1:length(lags)){
            test.lags[seasSelect] <- lags[seasSelect];
            if(ar.max[seasSelect]!=0){
                for(arSelect in 1:ar.max[seasSelect]){
                    m <- m + 1;
                    if(silent==FALSE){
                        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                        cat(paste0(round((m)/models.number,2)*100,"%"));
                    }
# Update the iSelect in ar.test preserving the previous values
                    ar.test[seasSelect] <- arSelect;
                    n.param <- 1 + max(ar.test %*% lags + i.best %*% lags,ma.best %*% lags) +
                            sum(ar.test) + sum(ma.best) + 1;
                    if(n.param > obs - 2){
                        test.models[[m]] <- NA;
                        test.ICs[iSelect+1] <- Inf;
                        test.ICs.all[m] <- Inf;
                        next
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.test),i.orders=(i.best),ma.orders=(ma.best),lags=(test.lags),
                                                constant=TRUE,initial=fittertype,CF.type=CF.type,
                                                h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                                int.type=int.type,silent=TRUE);
                    test.ICs[arSelect+1] <- test.models[[m]]$ICs[IC];
                    test.ICs.all[m] <- test.models[[m]]$ICs[IC];
                }
# Save the best AR
                ar.best[seasSelect] <- ar.test[seasSelect] <- c(0:ar.max[seasSelect])[which(test.ICs==min(test.ICs,na.rm=TRUE))[1]];
# Sort in order to put the best one on the first place
                test.ICs <- sort(test.ICs,decreasing=FALSE)
            }
        }
    }

##### Loop for MA
    if(any(ma.max!=0)){
        for(seasSelect in 1:length(lags)){
            test.lags[seasSelect] <- lags[seasSelect];
            if(ma.max[seasSelect]!=0){
                for(maSelect in 1:ma.max[seasSelect]){
                    m <- m + 1;
                    if(silent==FALSE){
                        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                        cat(paste0(round((m)/models.number,2)*100,"%"));
                    }
# Update the iSelect in i.test preserving the previous values
                    ma.test[seasSelect] <- maSelect;
                    n.param <- 1 + max(ar.best %*% lags + i.best %*% lags,ma.test %*% lags) +
                            sum(ar.best) + sum(ma.test) + 1;
                    if(n.param > obs - 2){
                        test.models[[m]] <- NA;
                        test.ICs[iSelect+1] <- Inf;
                        test.ICs.all[m] <- Inf;
                        next
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.best),ma.orders=(ma.test),lags=(test.lags),
                                                constant=TRUE,initial=fittertype,CF.type=CF.type,
                                                h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                                int.type=int.type,silent=TRUE);
                    test.ICs[maSelect+1] <- test.models[[m]]$ICs[IC];
                    test.ICs.all[m] <- test.models[[m]]$ICs[IC];
                }
# Save the best MA
                ma.best[seasSelect] <- ma.test[seasSelect] <- c(0:ma.max[seasSelect])[which(test.ICs==min(test.ICs,na.rm=TRUE))[1]];
# Sort in order to put the best one on the first place
                test.ICs <- sort(test.ICs,decreasing=FALSE)
            }
        }
    }

    if(any(test.models[[which(test.ICs.all==test.ICs[1])[1]]]$AR>=0.99)){
        ar.parameters <- test.models[[which(test.ICs.all==test.ICs[1])[1]]]$AR;
        if(any(ar.parameters[,1]>=0.99)){
            ar.test <- ar.best;
            ar.test[ar.parameters[,1]>=0.99] <- 0;
            i.test <- i.best;
            i.test[ar.parameters[,1]>=0.99] <- 1;

            test.models[[m+1]] <- ssarima(data,ar.orders=(ar.test),i.orders=(i.test),ma.orders=(ma.best),lags=(test.lags),
                                          constant=TRUE,initial=fittertype,CF.type=CF.type,
                                          h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                          int.type=int.type,silent=TRUE);
            test.ICs[2] <- test.models[[m+1]]$ICs[IC];
            test.ICs.all[m+1] <- test.models[[m+1]]$ICs[IC];

            if(test.ICs[1]>test.ICs[2]){
                ar.best <- ar.test;
                i.best <- i.test;
                test.ICs[1] <- test.ICs[2];
                test.models[[m]] <- test.models[[m+1]];
                test.ICs.all[m] <- test.ICs.all[m+1];
            }
        }
    }

    m <- m + 1;
    if(silent==FALSE){
        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
        cat(paste0(round((m)/models.number,2)*100,"%"));
    }

# Test the constant
    if(any(c(ar.best,i.best,ma.best)!=0)){
        test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.best),ma.orders=(ma.best),lags=(test.lags),
                                    constant=FALSE,initial=fittertype,CF.type=CF.type,
                                    h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                    int.type=int.type,silent=TRUE);
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
        ICs <- best.model$ICs;
        errormeasures <- best.model$accuracy;

# Make plot
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=best.model$forecast,fitted=best.model$fitted,lower=best.model$lower,upper=best.model$upper,
                       int.w=int.w,legend=legend,main=best.model$model);
        }
        else{
            graphmaker(actuals=data,forecast=best.model$forecast,fitted=best.model$fitted,
                       int.w=int.w,legend=legend,main=best.model$model);
        }

# Calculate the number os observations in the interval
        if(all(holdout==TRUE,intervals==TRUE)){
            insideintervals <- sum(as.vector(data)[(obs+1):obs.all]<=best.model$upper &
                                   as.vector(data)[(obs+1):obs.all]>=best.model$lower)/h*100;
        }
        else{
            insideintervals <- NULL;
        }

        ssoutput(Sys.time() - start.time, best.model$model, persistence=NULL, transition=NULL, measurement=NULL,
            phi=NULL, ARterms=best.model$AR, MAterms=best.model$MA, const=best.model$constant, A=NULL, B=NULL,
            n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
            CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
            int.type=int.type, int.w=int.w, ICs=ICs,
            holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);
    }

    return(best.model);
}
