ges <- function(data, orders=c(2), lags=c(1), initial=c("optimal","backcasting"),
                persistence=NULL, transition=NULL, measurement=NULL,
                CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                intermittent=FALSE,
                bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# General Exponential Smoothing function. Crazy thing...
#
#    Copyright (C) 2016  Ivan Svetunkov

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

    bounds <- substring(bounds[1],1,1);
# Check if "bounds" parameter makes any sense
    if(bounds!="n" & bounds!="a"){
        message("The strange bounds are defined. Switching to 'admissible'.");
        bounds <- "a";
    }

    if(length(orders) != length(lags)){
        stop(paste0("The length of 'lags' (",length(lags),") differes from the length of 'orders' (",length(orders),")."), call.=FALSE);
    }

    modellags <- matrix(rep(lags,times=orders),ncol=1);
    maxlag <- max(modellags);
    n.components <- sum(orders);

    CF.type <- CF.type[1];
# Check if the appropriate CF.type is defined
    if(any(CF.type==c("MLSTFE","MSTFE","TFL","MSEh"))){
        multisteps <- TRUE;
    }
    else if(any(CF.type==c("MSE","MAE","HAM"))){
        multisteps <- FALSE;
    }
    else{
        message(paste0("Strange cost function specified: ",CF.type,". Switching to 'MSE'."));
        CF.type <- "MSE";
        multisteps <- FALSE;
    }
    CF.type.original <- CF.type;

    int.type <- substring(int.type[1],1,1);
# Check the provided type of interval
    if(all(int.type!=c("a","p","s","n"))){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'parametric'."));
        int.type <- "p";
    }

# Check if the data is vector
    if(!is.numeric(data) & !is.ts(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }

# Check if the data contains NAs
    if(any(is.na(data))){
        message("Data contains NAs. These observations will be excluded.");
        datanew <- data[!is.na(data)];
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=frequency(data));
        }
        data <- datanew;
    }
    else{
        data <- ts(data,start=start(data),frequency=frequency(data));
    }

# Check the provided vector of initials: length and provided values.
    if(is.character(initial)){
        initial <- substring(initial[1],1,1);
        if(initial!="o" & initial!="b"){
            warning("You asked for a strange initial value. We don't do that here. Switching to optimal.",call.=FALSE,immediate.=TRUE);
            initial <- "o";
        }
        fittertype <- initial;
        estimate.initial <- TRUE;
    }
    else if(is.null(initial)){
        message("Initial value is not selected. Switching to optimal.");
        fittertype <- "o";
        estimate.initial <- TRUE;
    }
    else if(!is.null(initial)){
        if(!is.numeric(initial) | !is.vector(initial)){
            stop("The initial vector is not numeric!",call.=FALSE);
        }
        if(length(initial) != orders %*% lags){
            stop(paste0("Wrong length of initial vector. Should be ",orders %*% lags," instead of ",length(initial),"."),call.=FALSE);
        }
        fittertype <- "o";
        estimate.initial <- FALSE;
    }

# Check the provided vector of initials: length and provided values.
    if(!is.null(persistence)){
        if((!is.numeric(persistence) | !is.vector(persistence)) & !is.matrix(persistence)){
            stop("The persistence vector is not numeric!",call.=FALSE);
        }
        if(length(persistence) != n.components){
            stop(paste0("Wrong length of persistence vector. Should be ",n.components," instead of ",length(persistence),"."),call.=FALSE);
        }
        estimate.persistence <- FALSE;
    }
    else{
        estimate.persistence <- TRUE;
    }

# Check the provided vector of initials: length and provided values.
    if(!is.null(transition)){
        if((!is.numeric(transition) | !is.vector(transition)) & !is.matrix(transition)){
            stop("The transition matrix is not numeric!",call.=FALSE);
        }
        if(length(transition) != n.components^2){
            stop(paste0("Wrong length of transition matrix. Should be ",n.components^2," instead of ",length(transition),"."),call.=FALSE);
        }
        estimate.transition <- FALSE;
    }
    else{
        estimate.transition <- TRUE;
    }

# Check the provided vector of initials: length and provided values.
    if(!is.null(measurement)){
        if((!is.numeric(measurement) | !is.vector(measurement)) & !is.matrix(measurement)){
            stop("The measurement vector is not numeric!",call.=FALSE);
        }
        if(length(measurement) != n.components){
            stop(paste0("Wrong length of measurement vector. Should be ",n.components," instead of ",length(measurement),"."),call.=FALSE);
        }
        estimate.measurement <- FALSE;
    }
    else{
        estimate.measurement <- TRUE;
    }

# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

# If obs is negative, this means that we can't do anything...
    if(obs<=0){
        stop("Not enough observations in sample.",call.=FALSE);
    }

# Define the number of rows that should be in the matvt
    obs.xt <- max(obs.all + maxlag, obs + 2*maxlag);

# Define the actual values
    y <- matrix(data[1:obs],obs,1);
    datafreq <- frequency(data);

    if(intermittent==TRUE){
        ot <- (y!=0)*1;
        iprob <- mean(ot);
        obs.ot <- sum(ot);
        yot <- matrix(y[y!=0],obs.ot,1);
    }
    else{
        ot <- rep(1,obs);
        iprob <- 1;
        obs.ot <- obs;
        yot <- y;
    }

# If the data is not intermittent, let's assume that the parameter was switched unintentionally.
    if(iprob==1){
        intermittent <- FALSE;
    }

# Stop if number of observations is less than horizon and multisteps is chosen.
    if((multisteps==TRUE) & (obs.ot < h+1)){
        message(paste0("Do you seriously think that you can use ",CF.type," with h=",h," on ",obs.ot," non-zero observations?!"));
        stop("Not enough observations for multisteps cost function.",call.=FALSE);
    }
    else if((multisteps==TRUE) & (obs.ot < 2*h)){
        message(paste0("Number of observations is really low for a multisteps cost function! We will try but cannot guarantee anything..."));
    }

##### Prepare exogenous variables #####
    xregdata <- ssxreg(data=data, xreg=xreg, go.wild=go.wild,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obs=obs, obs.all=obs.all, obs.xt=obs.xt, maxlag=maxlag, h=h, silent=silent.text);
    n.exovars <- xregdata$n.exovars;
    matxt <- xregdata$matxt;
    matat <- xregdata$matat;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;
    estimate.xreg <- xregdata$estimate.xreg;
    estimate.FX <- xregdata$estimate.FX;
    estimate.gX <- xregdata$estimate.gX;
    estimate.initialX <- xregdata$estimate.initialX;

# 1 stands for the variance
    n.param <- 1 + n.components + n.components*(fittertype=="o") + n.components^2 + orders %*% lags + intermittent;

    if(estimate.xreg==TRUE){
        n.param <- n.param + estimate.initialX*n.exovars + estimate.FX*(n.exovars^2) + estimate.gX*(n.exovars);
    }

    if(obs.ot <= n.param){
        if(intermittent==TRUE){
            stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                        n.param," while the number of non-zero observations is ",obs.ot,"!"),call.=FALSE);
        }
        else{
            stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                        n.param," while the number of observations is ",obs.ot,"!"),call.=FALSE);
        }
    }

# These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

elements.ges <- function(C){
    n.coef <- 0;
    if(estimate.measurement==TRUE){
        matw <- matrix(C[n.coef+(1:n.components)],1,n.components);
        n.coef <- n.coef + n.components;
    }
    else{
        matw <- matrix(measurement,1,n.components);
    }

    if(estimate.transition==TRUE){
        matF <- matrix(C[n.coef+(1:(n.components^2))],n.components,n.components);
        n.coef <- n.coef + n.components^2;
    }
    else{
        matF <- matrix(transition,n.components,n.components);
    }

    if(estimate.persistence==TRUE){
        vecg <- matrix(C[n.coef+(1:n.components)],n.components,1);
        n.coef <- n.coef + n.components;
    }
    else{
        vecg <- matrix(persistence,n.components,1);
    }

    if(fittertype=="o"){
        if(estimate.initial==TRUE){
            vtvalues <- C[n.coef+(1:(orders %*% lags))];
            n.coef <- n.coef + orders %*% lags;

        }
        else{
            vtvalues <- initial;
        }

        vt <- matrix(NA,maxlag,n.components);
        for(i in 1:n.components){
            vt[(maxlag - modellags + 1)[i]:maxlag,i] <- vtvalues[((cumsum(c(0,modellags))[i]+1):cumsum(c(0,modellags))[i+1])];
            vt[is.na(vt[1:maxlag,i]),i] <- rep(rev(vt[(maxlag - modellags + 1)[i]:maxlag,i]),
                                               ceiling((maxlag - modellags + 1) / modellags)[i])[is.na(vt[1:maxlag,i])];
        }
    }
    else{
        vt <- matvt[1:maxlag,n.components];
    }

# If exogenous are included
    if(estimate.xreg==TRUE){
        at <- matrix(NA,maxlag,n.exovars);
        if(estimate.initialX==TRUE){
            at[,] <- rep(C[n.coef+(1:n.exovars)],each=maxlag);
            n.coef <- n.coef + n.exovars;
        }
        else{
            at <- matat[1:maxlag,];
        }
        if(estimate.FX==TRUE){
            matFX <- matrix(C[n.coef+(1:(n.exovars^2))],n.exovars,n.exovars);
            n.coef <- n.coef + n.exovars^2;
        }

        if(estimate.gX==TRUE){
            vecgX <- matrix(C[n.coef+(1:n.exovars)],n.exovars,1);
            n.coef <- n.coef + n.exovars;
        }
    }
    else{
        at <- matrix(0,maxlag,n.exovars);
    }

    return(list(matw=matw,matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

# Cost function for GES
CF <- function(C){
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    CF.res <- costfunc(matvt, matF, matw, y, vecg,
                       h, modellags, Etype, Ttype, Stype,
                       multisteps, CF.type, normalizer, fittertype,
                       matxt, matat, matFX, vecgX, ot,
                       bounds);

    return(CF.res);
}

# Likelihood function
Likelihood.value <- function(C){
    if(CF.type=="TFL"){
        return(obs.ot*log(iprob)*(h^multisteps)
               -obs.ot/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(obs.ot*log(iprob)
               -obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
    }
}

#####Start the calculations#####

    matvt <- matrix(NA,nrow=obs.xt,ncol=n.components);
    y.fit <- rep(NA,obs);
    errors <- rep(NA,obs);

    if(multisteps==TRUE){
        normalizer <- sum(abs(diff(c(y))))/(obs-1);
    }
    else{
        normalizer <- 0;
    }

# If there is something to optimise, let's do it.
    if((estimate.initial==TRUE) | (estimate.measurement==TRUE) | (estimate.transition==TRUE) | (estimate.persistence==TRUE) |
       (estimate.xreg==TRUE) | (estimate.FX==TRUE) | (estimate.gX==TRUE)){
# Initial values of matvt
        slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
        intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);

        C <- NULL;
# matw, matF, vecg, vt
        if(estimate.measurement==TRUE){
            C <- c(C,rep(1,n.components));
        }
        if(estimate.transition==TRUE){
            C <- c(C,rep(1,n.components^2));
        }
        if(estimate.persistence==TRUE){
            C <- c(C,rep(0.1,n.components));
        }
        if(estimate.initial==TRUE){
            if(fittertype=="o"){
                C <- c(C,intercept);
                if((orders %*% lags)>1){
                    C <- c(C,slope);
                }
                if((orders %*% lags)>2){
                    C <- c(C,yot[1:(orders %*% lags-2),]);
                }
            }
            else{
                vtvalues <- intercept;
                if((orders %*% lags)>1){
                    vtvalues <- c(vtvalues,slope);
                }
                if((orders %*% lags)>2){
                    vtvalues <- c(vtvalues,yot[1:(orders %*% lags-2),]);
                }

                vt <- matrix(NA,maxlag,n.components);
                for(i in 1:n.components){
                    vt[(maxlag - modellags + 1)[i]:maxlag,i] <- vtvalues[((cumsum(c(0,modellags))[i]+1):cumsum(c(0,modellags))[i+1])];
                    vt[is.na(vt[1:maxlag,i]),i] <- rep(rev(vt[(maxlag - modellags + 1)[i]:maxlag,i]),
                                                ceiling((maxlag - modellags + 1) / modellags)[i])[is.na(vt[1:maxlag,i])];
                }
                matvt[1:maxlag,] <- vt;
            }
        }

# initials, transition matrix and persistence vector
        if(estimate.xreg==TRUE){
            if(estimate.initialX==TRUE){
                C <- c(C,matat[maxlag,]);
            }
            if(go.wild==TRUE){
                if(estimate.FX==TRUE){
                    C <- c(C,c(diag(n.exovars)));
                }
                if(estimate.gX==TRUE){
                    C <- c(C,rep(0,n.exovars));
                }
            }
        }

        elements <- elements.ges(C);
        matw <- elements$matw;
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[1:maxlag,] <- elements$vt;
        matat[1:maxlag,] <- elements$at;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;

# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=5000));
        C <- res$solution;

# Optimise model. Second run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
        C <- res$solution;
        CF.objective <- res$objective;
    }
    else{
# matw, matF, vecg, vt
        C <- c(measurement,
               c(transition),
               c(persistence),
               c(initial));

        C <- c(C,matat[maxlag,],
               c(transitionX),
               c(persistenceX));

        CF.objective <- CF(C);
    }

# Change the CF.type in orders to calculate likelihood correctly.
    if(multisteps==TRUE){
        CF.type <- "TFL";
    }
    else{
        CF.type <- "MSE";
    }

    if(FI==TRUE){
        FI <- hessian(Likelihood.value,C);
    }
    else{
        FI <- NA;
    }

# Prepare for fitting
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    fitting <- fitterwrap(matvt, matF, matw, y, vecg,
                          modellags, Etype, Ttype, Stype, fittertype,
                          matxt, matat, matFX, vecgX, ot);
    matvt <- fitting$matvt;
    y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
    matat[1:nrow(fitting$matat),] <- fitting$matat;

    errors.mat <- ts(errorerwrap(matvt, matF, matw, y,
                                 h, Etype, Ttype, Stype, modellags,
                                 matxt, matat, matFX, ot),
                     start=start(data),frequency=frequency(data));
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));

    y.for <- ts(iprob*forecasterwrap(matrix(matvt[(obs+1):(obs+maxlag),],nrow=maxlag),
                               matF, matw, h, Ttype, Stype, modellags,
                               matrix(matxt[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                               matrix(matat[(obs.all-h+1):(obs.all),],ncol=n.exovars), matFX),
                start=time(data)[obs]+deltat(data),frequency=datafreq);

#    s2 <- mean(errors^2);
    s2 <- as.vector(sum((errors*ot)^2)/(obs.ot-n.param));

    if(any(is.na(y.fit),is.na(y.for))){
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

    if(intervals==TRUE){
        if(h==1){
            errors.x <- as.vector(errors);
            ev <- median(errors);
        }
        else{
            errors.x <- errors.mat;
            ev <- apply(errors.mat,2,median,na.rm=TRUE);
        }
        if(int.type!="a"){
            ev <- 0;
        }

        vt <- matrix(matvt[cbind(obs-modellags,c(1:n.components))],n.components,1);

        quantvalues <- pintervals(errors.x, ev=ev, int.w=int.w, int.type=int.type, df=(obs.ot - n.param),
                                 measurement=matw, transition=matF, persistence=vecg, s2=s2, modellags=modellags,
                                 y.for=y.for, iprob=iprob);
        y.low <- ts(c(y.for) + quantvalues$lower,start=start(y.for),frequency=frequency(data));
        y.high <- ts(c(y.for) + quantvalues$upper,start=start(y.for),frequency=frequency(data));
    }
    else{
        y.low <- NA;
        y.high <- NA;
    }

# Information criteria
    llikelihood <- Likelihood.value(C);

    AIC.coef <- 2*n.param*h^multisteps - 2*llikelihood;
    AICc.coef <- AIC.coef + 2 * n.param*h^multisteps * (n.param + 1) / (obs.ot - n.param - 1);
    BIC.coef <- log(obs.ot)*n.param - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

# Revert to the provided cost function
    CF.type <- CF.type.original

# Write down initials of states vector and exogenous
    if(estimate.initial==TRUE){
        initial <- C[2*n.components+n.components^2+(1:(orders %*% lags))];
    }
    if(estimate.initialX==TRUE){
        initialX <- matat[1,];
    }

# Fill in the rest of matvt
    matvt <- ts(matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    if(!is.null(xreg)){
        matvt <- cbind(matvt,matat);
        colnames(matvt) <- c(paste0("Component ",c(1:n.components)),colnames(matat));
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:n.components));
    }

    if(holdout==T){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- c(MAPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(diff(as.vector(data)[1:obs])))),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(as.vector(data)[1:obs]))),
                           MPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           RelMAE(as.vector(y.holdout),as.vector(y.for),rep(y[obs],h),digits=3),
                           SMAPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           cbias(as.vector(y.holdout)-as.vector(y.for),0,digits=5));
        names(errormeasures) <- c("MAPE","MASE","MAE/mean","MPE","RelMAE","SMAPE","cbias");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    modelname <- paste0("GES(",paste(orders,"[",lags,"]",collapse=",",sep=""),")");

    if(silent.text==FALSE){
        if(any(abs(eigen(matF - vecg %*% matw)$values)>(1 + 1E-10))){
            if(bounds!="a"){
                message("Unstable model was estimated! Use bounds='admissible' to address this issue!");
            }
            else{
                message("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.");
            }
        }
# Calculate the number os observations in the interval
        if(all(holdout==TRUE,intervals==TRUE)){
            insideintervals <- sum(as.vector(data)[(obs+1):obs.all]<=y.high &
                                as.vector(data)[(obs+1):obs.all]>=y.low)/h*100;
        }
        else{
            insideintervals <- NULL;
        }
# Print output
        ssoutput(Sys.time() - start.time, modelname, persistence=vecg, transition=matF, measurement=matw,
                phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                n.components=orders %*% lags, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
                CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
                int.type=int.type, int.w=int.w, ICs=ICs,
                holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);
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

return(list(model=modelname,states=matvt,initial=initial,measurement=matw,transition=matF,persistence=vecg,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,
            xreg=xreg,persistenceX=vecgX,transitionX=matFX,initialX=initialX,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
