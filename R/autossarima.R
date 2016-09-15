utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType"));

auto.ssarima <- function(data,ar.max=c(3,3), i.max=c(2,1), ma.max=c(3,3), lags=c(1,frequency(data)),
                         initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC"),
                         cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
                         h=10, holdout=FALSE, intervals=FALSE, level=0.95,
                         intervalsType=c("parametric","semiparametric","nonparametric"),
                         intermittent=c("none","auto","fixed","croston","tsb"),
                         bounds=c("admissible","none"),
                         silent=c("none","all","graph","legend","output"),
                         xreg=NULL, updateX=FALSE, ...){
# Function estimates several ssarima models and selects the best one using the selected information criterion.
#
#    Copyright (C) 2015 - 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for ssInput and make all the checks #####
    environment(ssAutoInput) <- environment();
    ssAutoInput(modelType="ssarima",ParentEnvironment=environment());

    if(any(is.complex(c(ar.max,i.max,ma.max,lags)))){
        stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
    }

    if(any(c(ar.max,i.max,ma.max)<0)){
        stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
    }

    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

# If there are zero lags, drop them
    if(any(lags==0)){
        ar.max <- ar.max[lags!=0];
        i.max <- i.max[lags!=0];
        ma.max <- ma.max[lags!=0];
        lags <- lags[lags!=0];
    }

# Define maxorder and make all the values look similar (for the polynomials)
    maxorder <- max(length(ar.max),length(i.max),length(ma.max));
    if(length(ar.max)!=maxorder){
        ar.max <- c(ar.max,rep(0,maxorder-length(ar.max)));
    }
    if(length(i.max)!=maxorder){
        i.max <- c(i.max,rep(0,maxorder-length(i.max)));
    }
    if(length(ma.max)!=maxorder){
        ma.max <- c(ma.max,rep(0,maxorder-length(ma.max)));
    }

# If zeroes are defined as orders for some lags, drop them.
    if(any((ar.max + i.max + ma.max)==0)){
        orders2leave <- (ar.max + i.max + ma.max)!=0;
        if(all(orders2leave==FALSE)){
            orders2leave <- lags==min(lags);
        }
        ar.max <- ar.max[orders2leave];
        i.max <- i.max[orders2leave];
        ma.max <- ma.max[orders2leave];
        lags <- lags[orders2leave];
    }

# Order things, so we would deal with the lowest level of seasonality first
    ar.max <- ar.max[order(lags,decreasing=FALSE)];
    i.max <- i.max[order(lags,decreasing=FALSE)];
    ma.max <- ma.max[order(lags,decreasing=FALSE)];
    lags <- sort(lags,decreasing=FALSE);

# Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        if(frequency(data)!=1){
            warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),"). Getting rid of some of them."),call.=FALSE);
        }
        lags.new <- unique(lags);
        ar.max.new <- i.max.new <- ma.max.new <- lags.new;
        for(i in 1:length(lags.new)){
            ar.max.new[i] <- max(ar.max[which(lags==lags.new[i])]);
            i.max.new[i] <- max(i.max[which(lags==lags.new[i])]);
            ma.max.new[i] <- max(ma.max[which(lags==lags.new[i])]);
        }
        ar.max <- ar.max.new;
        i.max <- i.max.new;
        ma.max <- ma.max.new;
        lags <- lags.new;
    }

# 1 stands for constant, the other one stands for variance
    n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;

# Try to figure out if the number of parameters can be tuned in order to fit something smaller on small samples
# Don't try to fix anything if the number of seasonalities is greater than 2
    if(length(lags)<=2){
        if(obsInsample <= n.param.max){
            arma.length <- length(ar.max)
            while(obsInsample <= n.param.max){
                if(any(c(ar.max[arma.length],ma.max[arma.length])>0)){
                    ar.max[arma.length] <- max(0,ar.max[arma.length] - 1);
                    n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                    if(obsInsample <= n.param.max){
                        ma.max[arma.length] <- max(0,ma.max[arma.length] - 1);
                        n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                    }
                }
                else{
                    if(arma.length==2){
                        ar.max[1] <- ar.max[1] - 1;
                        n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                        if(obsInsample <= n.param.max){
                            ma.max[1] <- ma.max[1] - 1;
                            n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                        }
                    }
                    else{
                        break;
                    }
                }
                if(all(c(ar.max,ma.max)==0)){
                    if(i.max[arma.length]>0){
                        i.max[arma.length] <- max(0,i.max[arma.length] - 1);
                        n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                    }
                    else if(i.max[1]>0){
                        if(obsInsample <= n.param.max){
                            i.max[1] <- max(0,i.max[1] - 1);
                            n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                        }
                    }
                    else{
                        break;
                    }
                }

            }
                n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
        }
    }

    if(obsInsample <= n.param.max){
        message(paste0("Not enough observations for the reasonable fit. Number of possible parameters is ",
                        n.param.max," while the number of observations is ",obsInsample,"!"));
        stop("Redefine maximum orders and try again.",call.=FALSE)
    }

# 1 stands for constant/no constant, another one stands for ARIMA(0,0,0)
    models.number <- sum(ar.max,i.max,ma.max) + 1 + 1;
    test.models <- list(NA);
    ICsTest <- rep(NA,max(ar.max,i.max,ma.max)+1);
    ICsTestAll <- rep(NA,models.number);
    m <- 0;

    test.lags <- ma.test <- ar.test <- i.test <- rep(0,length(lags));
    ar.best <- ma.best <- i.best <- rep(0,length(lags));

    if(silentText==FALSE){
        cat("Estimation progress:     ");
    }

### If for some reason we have model with zeroes for orders, return it.
    if(all(c(ar.max,i.max,ma.max)==0)){
        cat("\b\b\b\bDone!\n");
        test.models <- ssarima(data,ar.orders=(ar.best),i.orders=(i.best),ma.orders=(ma.best),lags=(lags),
                               constant=TRUE,initial=initialType,cfType=cfType,
                               h=h,holdout=holdout,intervals=intervals,level=level,
                               intervalsType=intervalsType,intermittent=intermittent,silent=TRUE,
                               xreg=xreg,updateX=updateX,FI=FI);
        return(test.models);
    }

    if(cfType!="MSE"){
        warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. The results of model selection may be wrong."),call.=FALSE);
    }

##### Loop for differences #####
    if(any(i.max!=0)){
        for(seasSelect in 1:length(lags)){
            test.lags[seasSelect] <- lags[seasSelect];
            if(i.max[seasSelect]!=0){
                for(iSelect in 0:i.max[seasSelect]){
                    if(m!=0 & iSelect==0){
                        next;
                    }
                    m <- m + 1;
                    if(silentText==FALSE){
                        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                        cat(paste0(round((m)/models.number,2)*100,"%"));
                    }
# Update the iSelect in i.test preserving the previous values
                    i.test[seasSelect] <- iSelect;
                    n.param <- 1 + max(ar.best %*% lags + i.test %*% lags,ma.best %*% lags) + sum(ar.best) + sum(ma.best) + 1;
                    if(n.param > obsInsample - 2){
                        test.models[[m]] <- NA;
                        ICsTest[iSelect+1] <- Inf;
                        ICsTestAll[m] <- Inf;
                        next;
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.test),ma.orders=(ma.best),lags=(test.lags),
                                                constant=TRUE,initial=initialType,cfType=cfType,
                                                h=h,holdout=holdout,intervals=intervals,level=level,
                                                intervalsType=intervalsType,intermittent=intermittent,silent=TRUE,
                                                xreg=xreg,updateX=updateX,FI=FI);
                    ICsTest[iSelect+1] <- test.models[[m]]$ICs[ic];
                    ICsTestAll[m] <- test.models[[m]]$ICs[ic];
                }
# Save the best differences
                i.best[seasSelect] <- i.test[seasSelect] <- c(0:i.max[seasSelect])[which(ICsTest==min(ICsTest,na.rm=TRUE))[1]];
# Sort in order to put the best one on the first place
                ICsTest <- sort(ICsTest,decreasing=FALSE)
            }
        }
    }

##### Loop for MA #####
    if(any(ma.max!=0)){
        for(seasSelect in 1:length(lags)){
#            test.lags[seasSelect] <- lags[seasSelect];
            if(ma.max[seasSelect]!=0){
                for(maSelect in 1:ma.max[seasSelect]){
                    m <- m + 1;
                    if(silentText==FALSE){
                        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                        cat(paste0(round((m)/models.number,2)*100,"%"));
                    }
# Update the iSelect in i.test preserving the previous values
                    ma.test[seasSelect] <- ma.max[seasSelect] - maSelect + 1;
                    n.param <- 1 + max(ar.best %*% lags + i.best %*% lags,ma.test %*% lags) + sum(ar.best) + sum(ma.test) + 1;
                    if(n.param > obsInsample - 2){
                        test.models[[m]] <- NA;
                        ICsTest[iSelect+1] <- Inf;
                        ICsTestAll[m] <- Inf;
                        next;
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.best),ma.orders=(ma.test),lags=(test.lags),
                                                constant=TRUE,initial=initialType,cfType=cfType,
                                                h=h,holdout=holdout,intervals=intervals,level=level,
                                                intervalsType=intervalsType,intermittent=intermittent,silent=TRUE,
                                                xreg=xreg,updateX=updateX,FI=FI);
                    ICsTest[maSelect+1] <- test.models[[m]]$ICs[ic];
                    ICsTestAll[m] <- test.models[[m]]$ICs[ic];
                    # If high order MA is not good, break the loop
                    if(ICsTest[maSelect+1] > ICsTest[maSelect]){
                        if(maSelect!=ma.max[seasSelect]){
                            m <- m + ma.max[seasSelect] - maSelect;
                            break;
                        }
                    }
                    else{
                        ma.best[seasSelect] <- ma.test[seasSelect];
                    }
                }
# Save the best MA
#                ma.best[seasSelect] <- ma.test[seasSelect] <- c(ma.max[seasSelect]:0)[which(ICsTest==min(ICsTest,na.rm=TRUE))[1]];
# Sort in order to put the best one on the first place
                ICsTest <- sort(ICsTest,decreasing=FALSE);
                ma.test[seasSelect] <- ma.best[seasSelect];
            }
        }
    }

##### Loop for AR #####
    if(any(ar.max!=0)){
        for(seasSelect in 1:length(lags)){
            test.lags[seasSelect] <- lags[seasSelect];
            if(ar.max[seasSelect]!=0){
                for(arSelect in 1:ar.max[seasSelect]){
                    m <- m + 1;
                    if(silentText==FALSE){
                        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                        cat(paste0(round((m)/models.number,2)*100,"%"));
                    }
# Update the iSelect in ar.test preserving the previous values
                    ar.test[seasSelect] <- ar.max[seasSelect] - arSelect + 1;
                    n.param <- 1 + max(ar.test %*% lags + i.best %*% lags,ma.best %*% lags) + sum(ar.test) + sum(ma.best) + 1;
                    if(n.param > obsInsample - 2){
                        test.models[[m]] <- NA;
                        ICsTest[iSelect+1] <- Inf;
                        ICsTestAll[m] <- Inf;
                        next;
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.test),i.orders=(i.best),ma.orders=(ma.best),lags=(test.lags),
                                                constant=TRUE,initial=initialType,cfType=cfType,
                                                h=h,holdout=holdout,intervals=intervals,level=level,
                                                intervalsType=intervalsType,intermittent=intermittent,silent=TRUE,
                                                xreg=xreg,updateX=updateX,FI=FI);
                    ICsTest[arSelect+1] <- test.models[[m]]$ICs[ic];
                    ICsTestAll[m] <- test.models[[m]]$ICs[ic];
                    # If high order AR is not good, break the loop
                    if(ICsTest[arSelect+1] > ICsTest[arSelect]){
                        if(arSelect!=ar.max[seasSelect]){
                            m <- m + ar.max[seasSelect] - arSelect;
                            break;
                        }
                    }
                    else{
                        ar.best[seasSelect] <- ar.test[seasSelect];
                    }
                }
# Save the best AR
#                ar.best[seasSelect] <- ar.test[seasSelect] <- c(ar.max[seasSelect]:0)[which(ICsTest==min(ICsTest,na.rm=TRUE))[1]];
# Sort in order to put the best one on the first place
                ICsTest <- sort(ICsTest,decreasing=FALSE);
                ar.test[seasSelect] <- ar.best[seasSelect];
            }
        }
    }

    m <- m + 1;
    if(silentText==FALSE){
        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
        cat(paste0(round((m)/models.number,2)*100,"%"));
    }

# Test the constant
    if(any(c(ar.best,i.best,ma.best)!=0)){
        test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.best),ma.orders=(ma.best),lags=(test.lags),
                                    constant=FALSE,initial=initialType,cfType=cfType,
                                    h=h,holdout=holdout,intervals=intervals,level=level,
                                    intervalsType=intervalsType,intermittent=intermittent,silent=TRUE,
                                    xreg=xreg,updateX=updateX,FI=FI);
        ICsTest[2] <- test.models[[m]]$ICs[ic];
        ICsTestAll[m] <- test.models[[m]]$ICs[ic];
    }

    constant <- c(TRUE,FALSE)[which(ICsTest[1:2]==min(ICsTest[1:2],na.rm=TRUE))];
    ICsTest <- sort(ICsTest,decreasing=FALSE)

    bestModel <- test.models[[which(ICsTestAll==ICsTest[1])[1]]];

    if(silentText==FALSE){
        cat("... Done! \n");
    }

    y.fit <- bestModel$fitted;
    y.for <- bestModel$forecast;
    y.high <- bestModel$upper;
    y.low <- bestModel$lower;
    modelname <- bestModel$model;

# Make plot
    if(silentGraph==FALSE){
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       level=level,legend=!silentLegend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    level=level,legend=!silentLegend,main=modelname);
        }
    }

    bestModel$timeElapsed <- Sys.time()-startTime;
    return(bestModel);
}
