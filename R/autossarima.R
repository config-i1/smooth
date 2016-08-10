auto.ssarima <- function(data,ar.max=c(3,3), i.max=c(2,1), ma.max=c(3,3), lags=c(1,frequency(data)),
                         initial=c("backcasting","optimal"), IC=c("AICc","AIC","BIC"),
                         CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                         h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                         int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                         intermittent=c("auto","none","fixed","croston","tsb"),
                         silent=c("none","all","graph","legend","output"),
                         xreg=NULL, go.wild=FALSE, ...){
# Function estimates several ssarima models and selects the best one using the selected information criterion.
#
#    Copyright (C) 2015 - 2016  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

# See if a user asked for Fisher Information
    if(!is.null(list(...)[['FI']])){
        FI <- list(...)[['FI']];
    }
    else{
        FI <- FALSE;
    }

# Make sense out of silent
    silent <- silent[1];
# Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output"))){
            message(paste0("Sorry, I have no idea what 'silent=",silent,"' means. Switching to 'none'."));
            silent <- "none";
        }
        silent <- substring(silent,1,1);
    }

    if(silent==FALSE | silent=="n"){
        silent.text <- FALSE;
        silent.graph <- FALSE;
        legend <- TRUE;
    }
    else if(silent==TRUE | silent=="a"){
        silent.text <- TRUE;
        silent.graph <- TRUE;
        legend <- FALSE;
    }
    else if(silent=="g"){
        silent.text <- FALSE;
        silent.graph <- TRUE;
        legend <- FALSE;
    }
    else if(silent=="l"){
        silent.text <- FALSE;
        silent.graph <- FALSE;
        legend <- FALSE;
    }
    else if(silent=="o"){
        silent.text <- TRUE;
        silent.graph <- FALSE;
        legend <- TRUE;
    }

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
        if(silent.text==FALSE){
            message("Initial value is not selected. Switching to optimal.");
        }
        fittertype <- "o";
    }
    else{
        if(silent.text==FALSE){
            message("Predefinde initials don't go well with automatic model selection. Switching to optimal.");
        }
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
        if(obs <= n.param.max){
            arma.length <- length(ar.max)
            while(obs <= n.param.max){
                if(any(c(ar.max[arma.length],ma.max[arma.length])>0)){
                    ar.max[arma.length] <- max(0,ar.max[arma.length] - 1);
                    n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                    if(obs <= n.param.max){
                        ma.max[arma.length] <- max(0,ma.max[arma.length] - 1);
                        n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                    }
                }
                else{
                    if(arma.length==2){
                        ar.max[1] <- ar.max[1] - 1;
                        n.param.max <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                        if(obs <= n.param.max){
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
                        if(obs <= n.param.max){
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

    if(obs <= n.param.max){
        message(paste0("Not enough observations for the reasonable fit. Number of possible parameters is ",
                        n.param.max," while the number of observations is ",obs,"!"));
        stop("Redefine maximum orders and try again.",call.=FALSE)
    }

# 1 stands for constant/no constant, another one stands for ARIMA(0,0,0)
    models.number <- sum(ar.max,i.max,ma.max) + 1 + 1;
    test.models <- list(NA);
    test.ICs <- rep(NA,max(ar.max,i.max,ma.max)+1);
    test.ICs.all <- rep(NA,models.number);
    m <- 0;

    test.lags <- ma.test <- ar.test <- i.test <- rep(0,length(lags));
    ar.best <- ma.best <- i.best <- rep(0,length(lags));

    if(silent.text==FALSE){
        cat("Estimation progress:     ");
    }

### If for some reason we have model with zeroes for orders, return it.
    if(all(c(ar.max,i.max,ma.max)==0)){
        cat("\b\b\b\bDone!\n");
        test.models <- ssarima(data,ar.orders=(ar.best),i.orders=(i.best),ma.orders=(ma.best),lags=(lags),
                               constant=TRUE,initial=fittertype,CF.type=CF.type,
                               h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                               int.type=int.type,intermittent=intermittent,silent=TRUE,
                               xreg=xreg,go.wild=go.wild,FI=FI);
        return(test.models);
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
                    if(silent.text==FALSE){
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
                        next;
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.test),ma.orders=(ma.best),lags=(test.lags),
                                                constant=TRUE,initial=fittertype,CF.type=CF.type,
                                                h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                                int.type=int.type,intermittent=intermittent,silent=TRUE,
                                                xreg=xreg,go.wild=go.wild,FI=FI);
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

##### Loop for MA #####
    if(any(ma.max!=0)){
        for(seasSelect in 1:length(lags)){
            test.lags[seasSelect] <- lags[seasSelect];
            if(ma.max[seasSelect]!=0){
                for(maSelect in 1:ma.max[seasSelect]){
                    m <- m + 1;
                    if(silent.text==FALSE){
                        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                        cat(paste0(round((m)/models.number,2)*100,"%"));
                    }
# Update the iSelect in i.test preserving the previous values
                    ma.test[seasSelect] <- ma.max[seasSelect] - maSelect + 1;
                    n.param <- 1 + max(ar.best %*% lags + i.best %*% lags,ma.test %*% lags) +
                            sum(ar.best) + sum(ma.test) + 1;
                    if(n.param > obs - 2){
                        test.models[[m]] <- NA;
                        test.ICs[iSelect+1] <- Inf;
                        test.ICs.all[m] <- Inf;
                        next;
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.best),ma.orders=(ma.test),lags=(test.lags),
                                                constant=TRUE,initial=fittertype,CF.type=CF.type,
                                                h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                                int.type=int.type,intermittent=intermittent,silent=TRUE,
                                                xreg=xreg,go.wild=go.wild,FI=FI);
                    test.ICs[maSelect+1] <- test.models[[m]]$ICs[IC];
                    test.ICs.all[m] <- test.models[[m]]$ICs[IC];
                    # If high order MA is not good, break the loop
                    if(test.ICs[maSelect+1] > test.ICs[maSelect]){
                        m <- m + ma.max[seasSelect] - maSelect;
                        break;
                    }
                }
# Save the best MA
                ma.best[seasSelect] <- ma.test[seasSelect] <- c(0:ma.max[seasSelect])[which(test.ICs==min(test.ICs,na.rm=TRUE))[1]];
# Sort in order to put the best one on the first place
                test.ICs <- sort(test.ICs,decreasing=FALSE);
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
                    if(silent.text==FALSE){
                        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
                        cat(paste0(round((m)/models.number,2)*100,"%"));
                    }
# Update the iSelect in ar.test preserving the previous values
                    ar.test[seasSelect] <- ar.max[seasSelect] - arSelect + 1;
                    n.param <- 1 + max(ar.test %*% lags + i.best %*% lags,ma.best %*% lags) +
                            sum(ar.test) + sum(ma.best) + 1;
                    if(n.param > obs - 2){
                        test.models[[m]] <- NA;
                        test.ICs[iSelect+1] <- Inf;
                        test.ICs.all[m] <- Inf;
                        next;
                    }

                    test.models[[m]] <- ssarima(data,ar.orders=(ar.test),i.orders=(i.best),ma.orders=(ma.best),lags=(test.lags),
                                                constant=TRUE,initial=fittertype,CF.type=CF.type,
                                                h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                                int.type=int.type,intermittent=intermittent,silent=TRUE,
                                                xreg=xreg,go.wild=go.wild,FI=FI);
                    test.ICs[arSelect+1] <- test.models[[m]]$ICs[IC];
                    test.ICs.all[m] <- test.models[[m]]$ICs[IC];
                    # If high order AR is not good, break the loop
                    if(test.ICs[arSelect+1] > test.ICs[arSelect]){
                        m <- m + ar.max[seasSelect] - arSelect;
                        break;
                    }
                }
# Save the best AR
                ar.best[seasSelect] <- ar.test[seasSelect] <- c(0:ar.max[seasSelect])[which(test.ICs==min(test.ICs,na.rm=TRUE))[1]];
# Sort in order to put the best one on the first place
                test.ICs <- sort(test.ICs,decreasing=FALSE)
            }
        }
    }

    m <- m + 1;
    if(silent.text==FALSE){
        cat(paste0(rep("\b",nchar(round(m/models.number,2)*100)+1),collapse=""));
        cat(paste0(round((m)/models.number,2)*100,"%"));
    }

# Test the constant
    if(any(c(ar.best,i.best,ma.best)!=0)){
        test.models[[m]] <- ssarima(data,ar.orders=(ar.best),i.orders=(i.best),ma.orders=(ma.best),lags=(test.lags),
                                    constant=FALSE,initial=fittertype,CF.type=CF.type,
                                    h=h,holdout=holdout,intervals=intervals,int.w=int.w,
                                    int.type=int.type,intermittent=intermittent,silent=TRUE,
                                    xreg=xreg,go.wild=go.wild,FI=FI);
    test.ICs[2] <- test.models[[m]]$ICs[IC];
    test.ICs.all[m] <- test.models[[m]]$ICs[IC];
    }

    constant <- c(TRUE,FALSE)[which(test.ICs[1:2]==min(test.ICs[1:2],na.rm=TRUE))];
    test.ICs <- sort(test.ICs,decreasing=FALSE)

    best.model <- test.models[[which(test.ICs.all==test.ICs[1])[1]]];

    if(silent.text==FALSE){
        cat("... Done! \n");
    }

    y.fit <- best.model$fitted;
    y.for <- best.model$forecast;
    y.high <- best.model$upper;
    y.low <- best.model$lower;
    modelname <- best.model$model;

    if(silent.text==FALSE){
        n.components <- max(ar.best %*% lags + i.best %*% lags,ma.best %*% lags);
        s2 <- sum(best.model$residuals^2)/(n.components + constant + 1 + sum(ar.best) + sum(ma.best));
        xreg <- NULL;
        go.wild <- FALSE;
        CF.objective <- best.model$CF;
        ICs <- best.model$ICs;
        errormeasures <- best.model$accuracy;

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
            holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures,intermittent=best.model$intermittent);
    }

# Make plot
    if(silent.graph==FALSE){
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       int.w=int.w,legend=legend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    int.w=int.w,legend=legend,main=modelname);
        }
    }

    return(best.model);
}
