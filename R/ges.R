ges <- function(data, orders=c(2), lags=c(1), initial=NULL,
                persistence=NULL, transition=NULL, measurement=NULL,
                persistence2=NULL, transition2=NULL,
                CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
                FI=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                bounds=TRUE, holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
                xreg=NULL, go.wild=FALSE, ...){
# General Exponential Smoothing function. Crazy thing...
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

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

# Define the number of rows that should be in the matxt
    obs.xt <- obs.all + maxlag;

# Define the actual values
    y <- matrix(data[1:obs],obs,1);
    datafreq <- frequency(data);

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
        matwex <- matrix(xreg,ncol=1);
# Define the second matxtreg to fill in the coefs of the exogenous vars
        matxtreg <- matrix(NA,obs.xt,1);
# max(obs+maxlag,obs.all)
        colnames(matxtreg) <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
        matxtreg[1:maxlag,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
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
# Define the second matxtreg to fill in the coefs of the exogenous vars
            matxtreg <- matrix(NA,obs.xt,n.exovars);
# max(obs+maxlag,obs.all)
            colnames(matxtreg) <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
            matwex <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
            matxtreg[1:maxlag,] <- rep(t(solve(t(matx[1:obs,]) %*% matx[1:obs,],tol=1e-50) %*% t(matx[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],each=maxlag);
            colnames(matxtreg) <- colnames(xreg);
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=FALSE);
        }
        matv <- matwex;
    }
    else{
        n.exovars <- 1;
        matwex <- matrix(0,max(obs+maxlag,obs.all),1);
        matxtreg <- matrix(0,obs.xt,1);
        matv <- matrix(1,max(obs+maxlag,obs.all),1);
    }

    n.param <- 2*n.components+n.components^2 + orders %*% lags;
    if(!is.null(xreg)){
# Number of initial states
        n.param <- n.param + n.exovars;
        if(go.wild==TRUE){
# Number of parameters in the transition matrix + persistence vector
            n.param <- n.exovars^2 + n.exovars;
        }
    }

    if(n.param >= obs-1){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                    n.param," while the number of observations is ",obs,"!"),call.=FALSE)
    }

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
# vecg2 - persistence for exogenous variables
# matF2 - transition matrix for exogenous variables
    if(!is.null(xreg)){
        if(go.wild==FALSE){
            matxtreg[1:maxlag,] <- rep(C[(length(C)-n.exovars+1):length(C)],each=maxlag);
            matF2 <- diag(n.exovars);
            vecg2 <- matrix(0,n.exovars,1);
        }
        else{
            matxtreg[1:maxlag,] <- rep(C[(length(C)-2*n.exovars-n.exovars^2+1):(length(C)-n.exovars-n.exovars^2)],each=maxlag);
            if(is.null(transition2)){
                matF2 <- matrix(C[(length(C)-n.exovars^2-n.exovars+1):(length(C)-n.exovars)],n.exovars,n.exovars)
            }
            else{
                matF2 <- matrix(transition,n.exovars,n.exovars);
            }

            if(is.null(persistence2)){
                vecg2 <- matrix(C[(length(C)-n.exovars+1):length(C)],n.exovars,1);
            }
            else{
                vecg2 <- matrix(persistence2,n.exovars,1);
            }
        }
    }
    else{
        matF2 <- diag(n.exovars);
        vecg2 <- matrix(0,n.exovars,1);
    }

    return(list(matw=matw,matF=matF,vecg=vecg,xt=xt,matxtreg=matxtreg,matF2=matF2,vecg2=vecg2));
}

# Function creates bounds for the estimates
hin.constrains <- function(C){

    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;

    if(any(is.nan(matF - vecg %*% matw))){
        D <- -0.1;
    }
    else{
        D <- 1 - abs(eigen(matF - vecg %*% matw)$values);
    }
    return(D);
}

# Cost function for GES
CF <- function(C){
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matxtreg[1:maxlag,] <- elements$matxtreg[1:maxlag,];
    matF2 <- elements$matF2;
    vecg2 <- elements$vecg2;
    matxt[1:maxlag,] <- elements$xt;

    if(bounds==TRUE){
        if(any(is.nan(matF - vecg %*% matw))){
            return(1E+300);
        }
        else{
            eigenvalues <- abs(eigen(matF - vecg %*% matw)$values);
            if(any(eigenvalues>1)){
                return(max(eigenvalues)*1E+100);
            }
        }
        if(any(abs(1-vecg2)>1)){
            return(max(abs(1-vecg2))*1E+100);
        }
        if(any(abs(matF2)>1)){
            return(max(abs(matF2))*1E+100);
        }
    }

    CF.res <- ssoptimizerwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE),
                              y, vecg, h, modellags, multisteps, CF.type, normalizer,
                              matwex, matxtreg, matv, matF2, vecg2);
    return(CF.res);
}

Likelihood.value <- function(C){
    if(CF.type=="GV"){
        return(-obs/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(-obs/2 *((h^multisteps)*log(2*pi*exp(1)) + log(CF(C))));
    }
}

#####Start the calculations#####

    matxt <- matrix(NA,nrow=(obs+maxlag),ncol=n.components);
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
# Initial values of matxt
        slope <- cov(y[1:min(12,obs),],c(1:min(12,obs)))/var(c(1:min(12,obs)));
        intercept <- sum(y[1:min(12,obs),])/min(12,obs) - slope * (sum(c(1:min(12,obs)))/min(12,obs) - 1);

# matw, matF, vecg, xt
        C <- c(rep(1,n.components),
               rep(1,n.components^2),
               rep(0,n.components),
               intercept);

        if((orders %*% lags)>1){
            C <- c(C,slope);
        }
        if((orders %*% lags)>2){
            C <- c(C,y[1:(orders %*% lags-2),]);
        }
# xtreg
# initials, transition matrix and persistence vector
        if(!is.null(xreg)){
            C <- c(C,matxtreg[maxlag,]);
            if(go.wild==TRUE){
                C <- c(C,c(diag(n.exovars)));
                C <- c(C,rep(0,n.exovars));
            }
        }

        elements <- elements.ges(C);
        matw <- elements$matw;
        matF <- elements$matF;
        vecg <- elements$vecg;
        matxtreg[1:maxlag,] <- elements$matxtreg[1:maxlag,];
        matxt[1:maxlag,] <- elements$xt;
        matF2 <- elements$matF2;
        vecg2 <- elements$vecg2;

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

    if(any(hin.constrains(C)<0) & silent==FALSE){
        message("Unstable model is estimated! Use 'bounds=TRUE' to address this issue!");
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
    matxt[1:maxlag,] <- elements$xt;
    matxtreg[1:maxlag,] <- elements$matxtreg[1:maxlag,];
    matF2 <- elements$matF2;
    vecg2 <- elements$vecg2;
    if(is.null(initial)){
        initial <- C[2*n.components+n.components^2+(1:(orders %*% lags))];
    }

    fitting <- ssfitterwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE), y,
                            vecg, modellags, matwex, matxtreg, matv, matF2, vecg2);
    matxt <- fitting$matxt;
    y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
    matxtreg[1:nrow(fitting$xtreg),] <- fitting$xtreg;

# Calculate the tails of matxtreg and matxt
    statestails <- ssstatetailwrap(matrix(rbind(matxt[(obs+1):(obs+maxlag),],matrix(NA,h-1,n.components)),h+maxlag-1,n.components), matF,
                                   matrix(matxtreg[(obs.xt-h):(obs.xt),],h+1,n.exovars), matF2, modellags);
    if(!is.null(xreg)){
# Write down the matxtreg and produce values for the holdout
        matxtreg[(obs.xt-h):(obs.xt),] <- statestails$xtreg;
    }

# Produce matrix of errors
    errors.mat <- ts(sserrorerwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE), y, h,
                                  modellags, matwex, matxtreg),
                     start=start(data), frequency=frequency(data));
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));

# Produce forecast
    y.for <- ts(ssforecasterwrap(matrix(matxt[(obs+1):nrow(matxt),],nrow=maxlag),
                                      matF,matrix(matw,obs.all,n.components,byrow=TRUE),h,
                                      modellags,matrix(matwex[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                                      matrix(matxtreg[(obs.all-h+1):(obs.all),],ncol=n.exovars)),
                start=time(data)[obs]+deltat(data), frequency=frequency(data));

#    s2 <- mean(errors^2);
    s2 <- c(sum(errors^2)/(obs-n.param));

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

        quantvalues <- pintervals(errors.x, ev=ev, int.w=int.w, int.type=int.type, df=(obs - n.param),
                                 measurement=matw, transition=matF, persistence=vecg, s2=s2, modellags=modellags);
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
    AICc.coef <- AIC.coef + 2 * n.param*h^multisteps * (n.param + 1) / (obs - n.param - 1);
    BIC.coef <- log(obs)*n.param - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

# Revert to the provided cost function
    CF.type <- CF.type.original

# Fill in the rest of matxt
    matxt <- rbind(matxt,as.matrix(statestails$matxt[-c(1:maxlag),]));
    matxt <- ts(matxt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    if(!is.null(xreg)){
        matxt <- cbind(matxt,matxtreg[1:nrow(matxt),]);
        colnames(matxt) <- c(paste0("Component ",c(1:n.components)),colnames(matxtreg));
    }
    else{
        colnames(matxt) <- paste0("Component ",c(1:n.components));
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

return(list(model=modelname,states=matxt,initial=initial,measurement=matw,transition=matF,persistence=vecg,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,xreg=xreg,persistence2=vecg2,transition2=matF2,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
