ges <- function(data, orders=c(2), lags=c(1), initial=NULL,
                persistence=NULL, transition=NULL, measurement=NULL,
                persistenceX=NULL, transitionX=NULL,
                CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
                FI=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                bounds=c("admissible","none"), holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
                xreg=NULL, go.wild=FALSE, intermittent=FALSE, ...){
# General Exponential Smoothing function. Crazy thing...
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

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
    if(any(CF.type==c("trace","TV","GV","MSEh"))){
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
    if(!is.null(initial)){
        if(!is.numeric(initial) | !is.vector(initial)){
            stop("The initial vector is not numeric!",call.=FALSE);
        }
        if(length(initial) != orders %*% lags){
            stop(paste0("Wrong length of initial vector. Should be ",orders %*% lags," instead of ",length(initial),"."),call.=FALSE);
        }
    }

# Check the provided vector of initials: length and provided values.
    if(!is.null(persistence)){
        if((!is.numeric(persistence) | !is.vector(persistence)) & !is.matrix(persistence)){
            stop("The persistence vector is not numeric!",call.=FALSE);
        }
        if(length(persistence) != n.components){
            stop(paste0("Wrong length of persistence vector. Should be ",n.components," instead of ",length(persistence),"."),call.=FALSE);
        }
    }

# Check the provided vector of initials: length and provided values.
    if(!is.null(transition)){
        if((!is.numeric(transition) | !is.vector(transition)) & !is.matrix(transition)){
            stop("The transition matrix is not numeric!",call.=FALSE);
        }
        if(length(transition) != n.components^2){
            stop(paste0("Wrong length of transition matrix. Should be ",n.components^2," instead of ",length(transition),"."),call.=FALSE);
        }
    }

# Check the provided vector of initials: length and provided values.
    if(!is.null(measurement)){
        if((!is.numeric(measurement) | !is.vector(measurement)) & !is.matrix(measurement)){
            stop("The measurement vector is not numeric!",call.=FALSE);
        }
        if(length(measurement) != n.components){
            stop(paste0("Wrong length of measurement vector. Should be ",n.components," instead of ",length(measurement),"."),call.=FALSE);
        }
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
    obs.xt <- obs.all + maxlag;

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


#### Now let's prepare the provided exogenous data for the inclusion in ETS
# Check the exogenous variable if it is present and
# fill in the values of xreg if it is absent in the holdout sample.
    if(!is.null(xreg)){
        if(any(is.na(xreg))){
            message("The exogenous variables contain NAs! This may lead to problems during estimation and forecast.");
        }
## The case with vectors and ts objects, but not matrices #####
        if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
# If xreg is vector or simple ts
        if(length(xreg)!=obs & length(xreg)!=obs.all){
            stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=FALSE);
        }
        if(length(xreg)==obs){
            message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
            xreg <- c(as.vector(xreg),rep(xreg[obs],h));
        }
# Number of exogenous variables
        n.exovars <- 1;
# Define matrix w for exogenous variables
        matxt <- matrix(xreg,ncol=1);
# Define the second matat to fill in the coefs of the exogenous vars
        matat <- matrix(NA,obs.xt,1);
        colnames(matat) <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
        matat[1:maxlag,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
# Redefine the number of components of ETS.
        }
## The case with matrices and data frames #####
        else if(is.matrix(xreg) | is.data.frame(xreg)){
    # If xreg is matrix or data frame
            if(nrow(xreg)!=obs & nrow(xreg)!=obs.all){
                stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=FALSE)
            }
            if(nrow(xreg)==obs){
                message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
                for(j in 1:h){
                    xreg <- rbind(xreg,xreg[obs,]);
                }
            }
# matx is needed for the initial values of coefs estimation using OLS
            n.exovars <- ncol(xreg);
            matx <- as.matrix(cbind(rep(1,obs.all),xreg));
# Define the second matat to fill in the coefs of the exogenous vars
            matat <- matrix(NA,obs.xt,n.exovars);
# Define matrix w for exogenous variables
            matxt <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
            matat[1:maxlag,] <- rep(t(solve(t(matx[1:obs,]) %*% matx[1:obs,],tol=1e-50) %*% t(matx[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],each=maxlag);
            if(is.null(colnames(matat))){
                colnames(matat) <- paste0("x",c(1:n.exovars));
            }
            else{
                colnames(matat) <- colnames(xreg);
            }
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=FALSE);
        }
    }
    else{
        n.exovars <- 1;
        matxt <- matrix(0,max(obs+maxlag,obs.all),1);
        matat <- matrix(0,obs.xt,1);
    }
    matFX <- diag(n.exovars);
    vecgX <- matrix(0,n.exovars,1);

# 1 stands for the variance
    n.param <- 2*n.components+n.components^2 + orders %*% lags + intermittent + 1;

    if(!is.null(xreg)){
# Number of initial states
        n.param <- n.param + n.exovars;
        if(go.wild==TRUE){
# Number of parameters in the transition matrix + persistence vector
            n.param <- n.param + n.exovars^2 + n.exovars;
        }
    }

    if(n.param >= obs.ot-1){
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
    Ttype <- "A";
    Stype <- "N";

elements.ges <- function(C){

    if(is.null(measurement)){
        matw <- matrix(C[1:n.components],1,n.components);
    }
    else{
        matw <- matrix(measurement,1,n.components);
    }

    if(is.null(transition)){
        matF <- matrix(C[n.components+(1:(n.components^2))],n.components,n.components);
    }
    else{
        matF <- matrix(transition,n.components,n.components);
    }

    if(is.null(persistence)){
        vecg <- matrix(C[n.components+n.components^2+(1:n.components)],n.components,1);
    }
    else{
        vecg <- matrix(persistence,n.components,1);
    }

    if(is.null(initial)){
        xtvalues <- C[2*n.components+n.components^2+(1:(orders %*% lags))];
    }
    else{
        xtvalues <- initial;
    }

    xt <- matrix(NA,maxlag,n.components);
    for(i in 1:n.components){
        xt[(maxlag - modellags + 1)[i]:maxlag,i] <- xtvalues[((cumsum(c(0,modellags))[i]+1):cumsum(c(0,modellags))[i+1])];
        xt[is.na(xt[1:maxlag,i]),i] <- rep(rev(xt[(maxlag - modellags + 1)[i]:maxlag,i]),
                                           ceiling((maxlag - modellags + 1) / modellags)[i])[is.na(xt[1:maxlag,i])];
    }

# If exogenous are included
# vecgX - persistence for exogenous variables
# matFX - transition matrix for exogenous variables
    if(!is.null(xreg) & (go.wild==TRUE)){
        matat[1:maxlag,] <- rep(C[(length(C)-2*n.exovars-n.exovars^2+1):(length(C)-n.exovars-n.exovars^2)],each=maxlag);
        if(is.null(transitionX)){
            matFX <- matrix(C[(length(C)-n.exovars^2-n.exovars+1):(length(C)-n.exovars)],n.exovars,n.exovars)
        }
        else{
            matFX <- matrix(transition,n.exovars,n.exovars);
        }

        if(is.null(persistenceX)){
            vecgX <- matrix(C[(length(C)-n.exovars+1):length(C)],n.exovars,1);
        }
        else{
            vecgX <- matrix(persistenceX,n.exovars,1);
        }
    }

    return(list(matw=matw,matF=matF,vecg=vecg,xt=xt,matat=matat,matFX=matFX,vecgX=vecgX));
}

# Cost function for GES
CF <- function(C){
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matat[1:maxlag,] <- elements$matat[1:maxlag,];
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    matvt[1:maxlag,] <- elements$xt;

    CF.res <- costfunc(matvt, matF, matw, y, vecg,
                   h, modellags, Etype, Ttype, Stype, multisteps, CF.type, normalizer,
                   matxt, matat, matFX, vecgX, ot,
                   bounds);

    return(CF.res);
}

# Likelihood function
Likelihood.value <- function(C){
    if(CF.type=="GV"){
        return(obs.ot*log(iprob)*(h^multisteps)
               -obs.ot/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(obs.ot*log(iprob)
               -obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
    }
}

#####Start the calculations#####

    matvt <- matrix(NA,nrow=(obs+maxlag),ncol=n.components);
    y.fit <- rep(NA,obs);
    errors <- rep(NA,obs);

    if(multisteps==TRUE){
        normalizer <- sum(abs(diff(c(y))))/(obs-1);
    }
    else{
        normalizer <- 0;
    }

# If there is something to optimise, let's do it.
    if(is.null(initial) | is.null(measurement) | is.null(transition) | is.null(persistence) | !is.null(xreg)){
# Initial values of matvt
        slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
        intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);

# matw, matF, vecg, xt
        C <- c(rep(1,n.components),
               rep(1,n.components^2),
               rep(0,n.components),
               intercept);

        if((orders %*% lags)>1){
            C <- c(C,slope);
        }
        if((orders %*% lags)>2){
            C <- c(C,yot[1:(orders %*% lags-2),]);
        }
# matat
# initials, transition matrix and persistence vector
        if(!is.null(xreg)){
            C <- c(C,matat[maxlag,]);
            if(go.wild==TRUE){
                C <- c(C,c(diag(n.exovars)));
                C <- c(C,rep(0,n.exovars));
            }
        }

        elements <- elements.ges(C);
        matw <- elements$matw;
        matF <- elements$matF;
        vecg <- elements$vecg;
        matat[1:maxlag,] <- elements$matat[1:maxlag,];
        matvt[1:maxlag,] <- elements$xt;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;

# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=5000));
#                              lb=c(rep(-2,2*n.components+n.components^2),rep(-max(abs(y[1:obs]),intercept),orders %*% lags)),
#                              ub=c(rep(2,2*n.components+n.components^2),rep(max(abs(y[1:obs]),intercept),orders %*% lags)));
        C <- res$solution;

# Optimise model. Second run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
        C <- res$solution;
        CF.objective <- res$objective;
    }
    else{
# matw, matF, vecg, xt
        C <- c(measurement,
               c(transition),
               c(persistence),
               c(initial));

        CF.objective <- CF(C);
    }

    if(any(abs(eigen(matF - vecg %*% matw)$values)>1) & silent==FALSE){
        message("Unstable model estimated! Use a different value of 'bounds' parameter to address this issue!");
    }

# Change the CF.type in orders to calculate likelihood correctly.
    if(multisteps==TRUE){
        CF.type <- "GV";
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
    matvt[1:maxlag,] <- elements$xt;
    matat[1:maxlag,] <- elements$matat[1:maxlag,];
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    if(is.null(initial)){
        initial <- C[2*n.components+n.components^2+(1:(orders %*% lags))];
    }

#    fitting <- ssfitterwrap(matvt, matF, matw, y, vecg, modellags,
#                            matxt, matat, matFX, vecgX, ot);
    fitting <- fitterwrap(matvt, matF, matw, y, vecg,
                          modellags, Etype, Ttype, Stype,
                          matxt, matat, matFX, vecgX, ot);
    matvt <- fitting$matvt;
    y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
    matat[1:nrow(fitting$matat),] <- fitting$matat;

# Calculate the tails of matat and matvt
#    statestails <- ssstatetailwrap(matrix(rbind(matvt[(obs+1):(obs+maxlag),],matrix(NA,h-1,n.components)),h+maxlag-1,n.components), matF,
#                                   matrix(matat[(obs.xt-h):(obs.xt),],h+1,n.exovars), matFX, modellags);
    statestails <- statetailwrap(matrix(rbind(matvt[(obs+1):(obs+maxlag),],matrix(NA,h-1,n.components)),h+maxlag-1,n.components), matF,
                                 matrix(matat[(obs.xt-h):(obs.xt),],h+1,n.exovars), matFX,
                                 modellags, Ttype, Stype);
    if(!is.null(xreg)){
# Write down the matat and produce values for the holdout
        matat[(obs.xt-h):(obs.xt),] <- statestails$matat;
    }

# Produce matrix of errors
#    errors.mat <- ts(sserrorerwrap(matvt, matF, matw, y,
#                                   h, modellags,
#                                   matxt, matat, matFX, ot),
#                     start=start(data), frequency=frequency(data));
    errors.mat <- ts(errorerwrap(matvt, matF, matw, y,
                                 h, Etype, Ttype, Stype, modellags,
                                 matxt, matat, matFX, ot),
                     start=start(data),frequency=frequency(data));
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));

# Produce forecast
#    y.for <- ts(iprob * ssforecasterwrap(matrix(matvt[(obs+1):nrow(matvt),],nrow=maxlag),
#                                         matF, matw, h,
#                                         modellags, matrix(matxt[(obs.all-h+1):(obs.all),],ncol=n.exovars),
#                                         matrix(matat[(obs.all-h+1):(obs.all),],ncol=n.exovars), matFX),
#                start=time(data)[obs]+deltat(data), frequency=frequency(data));
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

# Fill in the rest of matvt
    matvt <- rbind(matvt,as.matrix(statestails$matvt[-c(1:maxlag),]));
    matvt <- ts(matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    if(!is.null(xreg)){
        matvt <- cbind(matvt,matat[1:nrow(matvt),]);
        colnames(matvt) <- c(paste0("Component ",c(1:n.components)),colnames(matat));
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:n.components));
    }

    if(holdout==T){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- c(MAPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           MASE(as.vector(y.holdout),as.vector(y.for),sum(abs(diff(as.vector(data)[1:obs])))/(obs-1)),
                           MASE(as.vector(y.holdout),as.vector(y.for),sum(abs(as.vector(data)[1:obs]))/obs),
                           MPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           SMAPE(as.vector(y.holdout),as.vector(y.for),digits=5));
        names(errormeasures) <- c("MAPE","MASE","MASALE","MPE","SMAPE");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    modelname <- paste0("GES(",paste(orders,"[",lags,"]",collapse=",",sep=""),")");

if(silent==FALSE){
# Make plot
    if(intervals==TRUE){
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                   int.w=int.w,legend=legend,main=modelname);
    }
    else{
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                   int.w=int.w,legend=legend,main=modelname);
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
             n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
             CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
             int.type=int.type, int.w=int.w, ICs=ICs,
             holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);
}

return(list(model=modelname,states=matvt,initial=initial,measurement=matw,transition=matF,persistence=vecg,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,xreg=xreg,persistenceX=vecgX,transitionX=matFX,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
