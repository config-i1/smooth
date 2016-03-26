es <- function(data, model="ZZZ", persistence=NULL, phi=NULL,
               initial=NULL, initial.season=NULL, IC=c("AICc","AIC","BIC"),
               CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
               FI=FALSE, intervals=FALSE, int.w=0.95,
               int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
               bounds=c("usual","admissible"), holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
               xreg=NULL, ...){
# How could I forget about the Copyright (C) 2015 - 2016  Ivan Svetunkov

    go.wild <- FALSE;
# Start measuring the time of calculations
    start.time <- Sys.time();

    bounds <- substring(bounds[1],1,1);
    IC <- IC[1];

# Check if the data is vector
    if(!is.numeric(data) & !is.ts(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }

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

# Check if "bounds" parameter makes any sense
    if(bounds!="u" & bounds!="a"){
        message("The strange bounds are defined. Switching to 'usual'.");
        bounds <- "u";
    }

    if(!is.character(model)){
        stop(paste0("Something strange is provided instead of character object in model: ",
                    paste0(model,collapse=",")),call.=FALSE);
    }

# Predefine models pool for a model selection
    models.pool <- NULL;
# Deal with the list of models. Check what has been provided. Stop if there is a mistake.
    if(length(model)>1){
        if(any(nchar(model)>4)){
            stop(paste0("You have defined strange model(s) in the pool: ",
                           paste0(model[nchar(model)>4],collapse=",")),call.=FALSE);
        }
        else if(any(substr(model,1,1)!="A" & substr(model,1,1)!="M")){
            stop(paste0("You have defined strange model(s) in the pool: ",
                           paste0(model[substr(model,1,1)!="A" & substr(model,1,1)!="M"],collapse=",")),call.=FALSE);
        }
        else if(any(substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                    substr(model,2,2)!="M")){
            stop(paste0("You have defined strange model(s) in the pool: ",
                           paste0(model[substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                                 substr(model,2,2)!="M"],collapse=",")),call.=FALSE);
        }
        else if(any(substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                    substr(model,3,3)!="M" & substr(model,3,3)!="d")){
            stop(paste0("You have defined strange model(s) in the pool: ",
                           paste0(model[substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                                 substr(model,3,3)!="M" & substr(model,3,3)!="d"],collapse=",")),call.=FALSE);
        }
        else if(any(nchar(model)==4 & substr(model,4,4)!="N" &
                    substr(model,4,4)!="A" & substr(model,4,4)!="M")){
            stop(paste0("You have defined strange model(s) in the pool: ",
                           paste0(model[nchar(model)==4 & substr(model,4,4)!="N" &
                                 substr(model,4,4)!="A" & substr(model,4,4)!="M"],collapse=",")),call.=FALSE);
        }
        else{
            models.pool <- model;
        }
        model <- "ZZZ";
    }

# If chosen model is "AAdN" or anything like that, we are taking the appropriate values
    if(nchar(model)==4){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,4,4);
        damped <- TRUE;
        if(substring(model,3,3)!="d"){
            message(paste0("You have defined a strange model: ",model));
            sowhat(model);
            model <- paste0(Etype,Ttype,"d",Stype);
        }
    }
    else if(nchar(model)==3){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,3,3);
        damped <- FALSE;
    }
    else{
        message(paste0("You have defined a strange model: ",model));
        sowhat(model);
        message("Switching to 'ZZZ'");
        model <- "ZZZ";

        Etype <- "Z";
        Ttype <- "Z";
        Stype <- "Z";
        damped <- TRUE;
    }

    if(any(unlist(strsplit(model,""))=="C")){
        if(Etype=="C"){
            Etype <- "Z";
        }
        if(Ttype=="C"){
            Ttype <- "Z";
        }
        if(Stype=="C"){
            Stype <- "Z";
        }
    }

#### While intervals are not fully supported, use semi-parametric instead of parametric.
    if(any(Ttype==c("M","Z","C"),Stype==c("M","Z","C")) & int.type=="p" & intervals==TRUE){
        message("Sorry, but parametric intervals are not currently available for this ETS model. Switching to semiparametric.");
        int.type <- "s";
    }

    if(any(is.na(data))){
        if(silent==FALSE){
            message("Data contains NAs. These observations will be excluded.");
        }
        datanew <- data[!is.na(data)];
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=frequency(data));
        }
        data <- datanew
    }

# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

# Define the actual values
    y <- matrix(data[1:obs],obs,1);

# Check if the data is ts-object
    if(!is.ts(data) & Stype!="N"){
        message("The provided data is not ts object. Only non-seasonal models are available.");
        Stype <- "N";
    }
    datafreq <- frequency(data);

### Check the length of the provided data. Say bad words if:
# 1. Seasonal model, <=2 seasons of data and no initial seasonals.
# 2. Seasonal model, <=1 season of data, no initial seasonals and no persistence.
    if((Stype!="N" & (obs <= 2*datafreq) & is.null(initial.season)) | (Stype!="N" & (obs <= datafreq) & is.null(initial.season) & is.null(persistence))){
    	if(is.null(initial.season)){
        	message("Are you out of your mind?! We don't have enough observations for the seasonal model! Switching to non-seasonal.");
       		Stype <- "N";
       	}
    }

# If model selection is chosen, forget about the initial values and persistence
    if(any(Etype=="Z",Ttype=="Z",Stype=="Z")){
        if(any(!is.null(initial),!is.null(initial.season),!is.null(persistence),!is.null(phi))){
            message("Model selection doesn't go well with the predefined values.");
            message("Switching to the estimation of all the parameters.");
            initial <- NULL;
            initial.season <- NULL;
            persistence <- NULL;
            phi <- NULL;
        }
    }

### Check all the parameters for the possible errors.
    if(!is.null(persistence)){
        if(any(!is.numeric(persistence),!is.vector(persistence))){
            message("The persistence is not a numeric vector!");
            message("Changing to the estimation of persistence vector values.");
            persistence <- NULL;
        }
        else{
            if(length(persistence)>3){
                message("The length of persistence vector is wrong! It should not be greater than 3.");
                message("Changing to the estimation of persistence vector values.");
                persistence <- NULL;
            }
        }
    }

### Check if the meaningfull initials are passed
    if(!is.null(initial)){
        if(any(!is.numeric(initial),!is.vector(initial))){
            message("The initial vector is not numeric!");
            message("Values of initial vector will be estimated.");
            initial <- NULL;
        }
        else{
            if(length(initial)>2){
                message("The length of initial vector is wrong! It should not be greater than 2.");
                message("Values of initial vector will be estimated.");
                initial <- NULL;
            }
        }
    }

### Check the error type
    if(all(Etype!=c("Z","A","M"))){
        message("Wrong error type! Should be 'Z', 'A' or 'M'.");
        message("Changing to 'Z'");
        Etype <- "Z";
    }

### Check the trend type
    if(all(Ttype!=c("Z","N","A","M"))){
        message("Wrong trend type! Should be 'Z', 'N', 'A' or 'M'.");
        message("Changing to 'Z'");
        Ttype <- "Z";
    }

### Check the seasonality type
    if(all(Stype!=c("Z","N","A","M"))){
        message("Wrong seasonality type! Should be 'Z', 'N', 'A' or 'M'.");
        if(datafreq==1){
            if(silent==FALSE){
                message("Data is non-seasonal. Changing seasonal component to 'N'");
            }
            Stype <- "N";
        }
        else{
            if(silent==FALSE){
                message("Changing to 'Z'");
            }
            Stype <- "Z";
        }
    }
    if(all(Stype!="N",datafreq==1)){
        if(silent==FALSE){
            message("Cannot build the seasonal model on the data with the frequency 1.");
            message(paste0("Switching to non-seasonal model: ETS(",substring(model,1,nchar(model)-1),"N)"));
        }
        Stype <- "N";
    }


    if(any(y<=0)){
        if(Etype=="M"){
            message("Can't apply multiplicative model to non-positive data. Switching error to 'A'");
            Etype <- "A";
        }
        if(Ttype=="M"){
            message("Can't apply multiplicative model to non-positive data. Switching trend to 'A'");
            Ttype <- "A";
        }
        if(Stype=="M"){
            message("Can't apply multiplicative model to non-positive data. Switching seasonal to 'A'");
            Stype <- "A";
        }
    }

# Now let's prepare the provided exogenous data for the inclusion in ETS
# Check the exogenous variable if it is present and
# fill in the values of xreg if it is absent in the holdout sample.
    if(!is.null(xreg)){
        if(any(is.na(xreg))){
            message("The exogenous variables contain NAs! This may lead to problems during estimation and forecast.");
        }
##### The case with vectors and ts objects, but not matrices
        if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
# If xreg is vector or simple ts
        if(length(xreg)!=obs & length(xreg)!=obs.all){
            stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=F);
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
        matxtreg <- matrix(NA,obs.all+datafreq,1);
        exocomponent.names <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
        matxtreg[1:datafreq,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
        }
##### The case with matrices and data frames
        else if(is.matrix(xreg) | is.data.frame(xreg)){
    # If xreg is matrix or data frame
            if(nrow(xreg)!=obs & nrow(xreg)!=obs.all){
                stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=F);
            }
            if(nrow(xreg)==obs){
	            message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
                for(j in 1:h){
                xreg <- rbind(xreg,xreg[obs,]);
                }
            }
# mat.x is needed for the initial values of coefs estimation using OLS
            mat.x <- as.matrix(cbind(rep(1,obs.all),xreg));
            n.exovars <- ncol(xreg);
# Define the second matxtreg to fill in the coefs of the exogenous vars
            matxtreg <- matrix(NA,obs.all+datafreq,n.exovars);
            exocomponent.names <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
            matwex <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
            matxtreg[1:datafreq,] <- rep(t(solve(t(mat.x[1:obs,]) %*% mat.x[1:obs,],tol=1e-50) %*%
                                               t(mat.x[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],
                                         each=datafreq);
            colnames(matxtreg) <- colnames(xreg);
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=F);
        }
        matv <- matwex;
        estimate.xreg <- TRUE;
    }
    else{
# "1" is needed for the final forecast simplification
        n.exovars <- 1;
        matwex <- matrix(0,max(obs+datafreq,obs.all),1);
        matxtreg <- matrix(0,max(obs+datafreq,obs.all),1);
        matv <- matrix(1,max(obs+datafreq,obs.all),1);
        estimate.xreg <- FALSE;
    }

    matF2 <- diag(n.exovars);
    vecg2 <- matrix(0,n.exovars,1);

##### All the function should be transfered into optimizerwrap #####
# Cost function for ETS
CF <- function(C){
    init.ets <- etsmatrices(matxt, vecg, phi, matrix(C,nrow=1), n.components, modellags,
                            Ttype, Stype, n.exovars, matxtreg, estimate.persistence,
                            estimate.phi, estimate.initial, estimate.initial.season,
                            estimate.xreg);

    if(estimate.persistence==TRUE){
        if(bounds=="a" & (Ttype!="N") & (Stype!="N")){
            Theta.func <- function(Theta){
                return(abs((init.ets$phi*C[1]+init.ets$phi+1)/(C[3]) +
                               ((init.ets$phi-1)*(1+cos(Theta)-cos(maxlag*Theta)) +
                                    cos((maxlag-1)*Theta)-init.ets$phi*cos((maxlag+1)*Theta))/
                               (2*(1+cos(Theta))*(1-cos(maxlag*Theta)))));
            }
            Theta <- 0.1;
            Theta <- suppressWarnings(optim(Theta,Theta.func,method="Brent",lower=0,upper=1)$par);
        }
        else{
            Theta <- 0;
        }
        CF.res <- costfunc(init.ets$matxt, init.ets$matF, init.ets$matw, y, init.ets$vecg,
                           h, modellags, Etype, Ttype, Stype, multisteps, CF.type, normalizer,
                           matwex, matxtreg, matv, matF2, vecg2,
                           bounds, init.ets$phi, Theta);
    }
    else{
        CF.res <- optimizerwrap(init.ets$matxt, init.ets$matF, init.ets$matw, y, init.ets$vecg,
                                h, modellags, Etype, Ttype, Stype, multisteps, CF.type, normalizer,
                                matwex, matxtreg, matv, matF2, vecg2);
    }

    if(is.nan(CF.res) | is.na(CF.res) | is.infinite(CF.res)){
        CF.res <- 1e100;
    }

    return(CF.res);
}

# Function constructs default bounds where C values should lie
C.values <- function(bounds,Ttype,Stype,vecg,matxt,phi,maxlag,n.components,matxtreg){
    C <- NA;
    C.lower <- NA;
    C.upper <- NA;

    if(bounds=="u"){
        if(estimate.persistence==TRUE){
            C <- c(C,vecg);
            C.lower <- c(C.lower,rep(0,length(vecg)));
            C.upper <- c(C.upper,rep(1,length(vecg)));
        }
        if(estimate.phi==TRUE){
            C <- c(C,phi);
            C.lower <- c(C.lower,0);
            C.upper <- c(C.upper,1);
        }
        if(estimate.initial==TRUE){
            C <- c(C,matxt[maxlag,1:(n.components - (Stype!="N"))]);
            if(Ttype!="M"){
                C.lower <- c(C.lower,rep(-Inf,(n.components - (Stype!="N"))));
                C.upper <- c(C.upper,rep(Inf,(n.components - (Stype!="N"))));
            }
            else{
                C.lower <- c(C.lower,0.1,0.01);
                C.upper <- c(C.upper,Inf,3);
            }
        }
        if(Stype!="N"){
            if(estimate.initial.season==TRUE){
                C <- c(C,matxt[1:maxlag,n.components]);
                if(Stype=="A"){
                    C.lower <- c(C.lower,rep(-Inf,maxlag));
                    C.upper <- c(C.upper,rep(Inf,maxlag));
                }
                else{
                    C.lower <- c(C.lower,rep(0,maxlag));
                    C.upper <- c(C.upper,rep(10,maxlag));
                }
            }
        }
    }
    else{
        if(estimate.persistence==TRUE){
            C <- c(C,vecg);
            C.lower <- c(C.lower,rep(-5,length(vecg)));
            C.upper <- c(C.upper,rep(5,length(vecg)));
        }
        if(estimate.phi==TRUE){
            C <- c(C,phi);
            C.lower <- c(C.lower,0);
            C.upper <- c(C.upper,1);
        }
        if(estimate.initial==TRUE){
            C <- c(C,matxt[maxlag,1:(n.components - (Stype!="N"))]);
            if(Ttype!="M"){
                C.lower <- c(C.lower,rep(-Inf,(n.components - (Stype!="N"))));
                C.upper <- c(C.upper,rep(Inf,(n.components - (Stype!="N"))));
            }
            else{
                C.lower <- c(C.lower,1,0.01);
                C.upper <- c(C.upper,Inf,3);
            }
        }
        if(Stype!="N"){
            if(estimate.initial.season==TRUE){
                C <- c(C,matxt[1:maxlag,n.components]);
                if(Stype=="A"){
                    C.lower <- c(C.lower,rep(-Inf,maxlag));
                    C.upper <- c(C.upper,rep(Inf,maxlag));
                }
                else{
                    C.lower <- c(C.lower,rep(-0.0001,maxlag));
                    C.upper <- c(C.upper,rep(20,maxlag));
                }
            }
        }
    }

    if(!is.null(xreg)){
        C <- c(C,matxtreg[maxlag,]);
        C.lower <- c(C.lower,rep(-Inf,n.exovars));
        C.upper <- c(C.upper,rep(Inf,n.exovars));
    }

    C <- C[!is.na(C)];
    C.lower <- C.lower[!is.na(C.lower)];
    C.upper <- C.upper[!is.na(C.upper)];

    return(list(C=C,C.lower=C.lower,C.upper=C.upper));
}

Likelihood.value <- function(C){
    if(CF.type=="GV"){
        return(-obs/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(-obs/2 *(log(2*pi*exp(1)) + log(CF(C))));
    }
}

## Function calculates ICs
IC.calc <- function(n.param=n.param,C,Etype=Etype){
# Information criteria are calculated with the constant part "log(2*pi*exp(1)*h+log(obs))*obs".
# And it is based on the mean of the sum squared residuals either than sum.
# Hyndman likelihood is: llikelihood <- obs*log(obs*CF.objective)
    llikelihood <- Likelihood.value(C);

    AIC.coef <- 2*n.param*h^multisteps - 2*llikelihood;
    AICc.coef <- AIC.coef + 2 * n.param*h^multisteps * (n.param + 1) / (obs - n.param - 1);
    BIC.coef <- log(obs)*n.param*h^multisteps - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

    return(list(llikelihood=llikelihood,ICs=ICs));
}

checker <- function(inherits=TRUE){
### Check the length of initials and persistence vectors
# Check the persistence vector length
    if(!is.null(persistence)){
        if(n.components != length(persistence)){
            message("The length of persistence vector does not correspond to the chosen model!");
            message("Values will be estimated");
            assign("persistence",NULL,inherits=inherits);
            assign("smoothingparameters",cbind(c(0.3,0.2,0.1),rep(0.05,3)),inherits=inherits);
            assign("estimate.persistence",TRUE,inherits=inherits);
            assign("basicparams",initparams(Ttype, Stype, datafreq, obs, y,
                                            damped, phi, smoothingparameters, initialstates,
                                            seasonalcoefs),inherits=TRUE);
            assign("vecg",basicparams$vecg,inherits=inherits);
        }
    }

# Check the inital vector length
    if(!is.null(initial)){
        if((n.components - (Stype!="N"))!=length(initial)){
            message("The length of initial state vector does not correspond to the chosen model!");
            message("Values of initial vector will be estimated.");
            initial <- NULL;
            if(Ttype!="N"){
                initialstates <- matrix(NA,1,4);
                initialstates[1,2] <- cov(y[1:min(12,obs),],c(1:min(12,obs)))/var(c(1:min(12,obs)));
                initialstates[1,1] <- mean(y[1:min(12,obs),]) - initialstates[1,2] * (mean(c(1:min(12,obs))) - 1);
                initialstates[1,3] <- mean(y[1:min(12,obs),]);
                initialstates[1,4] <- 1;
            }
            else{
                initialstates <- matrix(rep(mean(y[1:min(12,obs),]),4),nrow=1);
            }
            assign("estimate.initial",TRUE,inherits=inherits);
            assign("initialstates",initialstates,inherits=inherits);
            assign("basicparams",initparams(Ttype, Stype, datafreq, obs, y,
                                            damped, phi, smoothingparameters, initialstates,
                                            seasonalcoefs),inherits=TRUE);
            assign("matxt",basicparams$matxt,inherits=inherits);
        }
    }

# Check the seasonal inital vector length
    if(!is.null(initial.season)){
        if(frequency(data)!=length(initial.season)){
            message("The length of seasonal initial states does not correspond to the frequency of the data!");
            message("Values of initial seasonals will be estimated.");
            seasonalcoefs <- decompose(ts(c(y),frequency=datafreq),type="additive")$seasonal[1:datafreq];
            seasonalcoefs <- cbind(seasonalcoefs,decompose(ts(c(y),frequency=datafreq),
                                                           type="multiplicative")$seasonal[1:datafreq]);
            assign("initial.season",NULL,inherits=inherits);
            assign("seasonalcoefs",seasonalcoefs,inherits=inherits);
        }
    }
}

#################### Basic initialisation of ETS ####################
# If initial values are provided, write them. If not, estimate them.
# First two columns are needed for additive seasonality, the 3rd and 4th - for the multiplicative
    if(Ttype!="N"){
        if(is.null(initial)){
            estimate.initial <- TRUE;
            initialstates <- matrix(NA,1,4);
# "-1" is needed, so the level would correspond to the values before the in-sample
            initialstates[1,2] <- cov(y[1:min(12,obs),],c(1:min(12,obs)))/var(c(1:min(12,obs)));
            initialstates[1,1] <- mean(y[1:min(12,obs),]) - initialstates[1,2] * (mean(c(1:min(12,obs))) - 1);
            initialstates[1,3] <- mean(y[1:min(12,obs),]);
            initialstates[1,4] <- 1;
        }
        else{
            estimate.initial <- FALSE;
            initialstates <- matrix(rep(initial,2),nrow=1);
        }
    }
    else{
        if(is.null(initial)){
            estimate.initial <- TRUE;
            initialstates <- matrix(rep(mean(y[1:min(12,obs),]),4),nrow=1);
        }
        else{
            estimate.initial <- FALSE;
            initialstates <- matrix(rep(initial,4),nrow=1);
        }
    }

# Define matrix of seasonal coefficients. The first column consists of additive, the second - multiplicative elements
# If the seasonal model is chosen and initials are provided, fill in the first "maxlag" values of seasonal component.
    if(Stype!="N"){
        if(is.null(initial.season)){
            estimate.initial.season <- TRUE;
            seasonalcoefs <- decompose(ts(c(y),frequency=datafreq),type="additive")$seasonal[1:datafreq];
            seasonalcoefs <- cbind(seasonalcoefs,decompose(ts(c(y),frequency=datafreq),
                                                           type="multiplicative")$seasonal[1:datafreq]);
        }
        else{
            estimate.initial.season <- FALSE;
            seasonalcoefs <- cbind(initial.season,initial.season);
        }
    }
    else{
        estimate.initial.season <- FALSE;
        seasonalcoefs <- matrix(1,1,1);
    }

# If the persistence vector is provided, use it
    if(!is.null(persistence)){
        smoothingparameters <- cbind(persistence,persistence);
        estimate.persistence <- FALSE;
    }
    else{
        smoothingparameters <- cbind(c(0.3,0.2,0.1),rep(0.05,3));
        estimate.persistence <- TRUE;
    }

    if(!is.null(phi)){
        if(phi<0 | phi>2){
            message("Damping parameter should lie in (0, 2) region.");
            message("Changing to the estimation of phi.");
            phi <- NULL;
        }
    }

# Vectors of fitted data and errors
    y.fit <- rep(NA,obs);
    errors <- rep(NA,obs);

    normalizer <- mean(abs(diff(c(y))));

    basicparams <- initparams(Ttype, Stype, datafreq, obs, y,
                              damped, phi, smoothingparameters, initialstates, seasonalcoefs);
    n.components <- basicparams$n.components;
    maxlag <- basicparams$maxlag;
    matxt <- basicparams$matxt;
    vecg <- basicparams$vecg;
    estimate.phi <- basicparams$estimate.phi;
    phi <- basicparams$phi;
    modellags <- basicparams$modellags;

    checker(inherits=TRUE);

# Define the number of rows that should be in the matxt
    obs.xt <- obs.all;

############ Start the estimation depending on the model ############
# Fill in the vector of initial values and vector of constrains used in estimation
# This also should include in theory  "| estimate.phi==TRUE",
#    but it doesn't make much sense and makes things more complicated
    if(any(estimate.persistence,estimate.initial,estimate.initial.season)){

# Number of observations in the error matrix excluding NAs.
        errors.mat.obs <- obs - h + 1;
##### If auto selection is used (for model="ZZZ" or model="CCC"), then let's start misbehaving...
        if(any(unlist(strsplit(model,""))=="C") | any(Etype=="Z",Ttype=="Z",Stype=="Z")){
# Produce the data for AIC weights

            if(!is.null(models.pool)){
                models.number <- length(models.pool);
            }
            else{
# Define the pool of models in case of "ZZZ" or "CCC" to select from
                if(any(y<=0)){
                    if(silent==FALSE){
                        message("Only additive models are allowed with the negative data.");
                    }
                    errors.pool <- c("A");
                    trends.pool <- c("N","A","Ad");
                    season.pool <- c("N","A");
                }
                else{
                    errors.pool <- c("A","M");
                    trends.pool <- c("N","A","Ad","M","Md");
                    season.pool <- c("N","A","M");
                }

                if(Etype!="Z"){
                    errors.pool <- Etype;
                }

                if(Ttype!="Z"){
                    if(damped==TRUE){
                        trends.pool <- paste0(Ttype,"d");
                    }
                    else{
                        trends.pool <- Ttype;
                    }
                }

                if(Stype!="Z"){
                    season.pool <- Stype;
                }

                models.number <- (length(errors.pool)*length(trends.pool)*length(season.pool));
                models.pool <- paste0(rep(errors.pool,each=length(trends.pool)*length(season.pool)),
                                      trends.pool,
                                      rep(season.pool,each=length(trends.pool)));
            }

# Number of observations in the error matrix excluding NAs.
            errors.mat.obs <- obs - h + 1;

            results <- as.list(c(1:models.number));

            if(silent==FALSE){
                cat("Building model: ");
            }
# Start cycle of models
            for(j in 1:models.number){
                    current.model <- models.pool[j];
                    Etype <- substring(current.model,1,1);
                    Ttype <- substring(current.model,2,2);
                    if(nchar(current.model)==4){
                        damped <- TRUE;
                        phi <- NULL;
                        Stype <- substring(current.model,4,4);
                    }
                    else{
                        damped <- FALSE;
                        phi <- 1;
                        Stype <- substring(current.model,3,3);
                    }

                if(silent==FALSE){
                    cat(paste0(current.model," "));
                }

                basicparams <- initparams(Ttype, Stype, datafreq, obs, y,
                                          damped, phi, smoothingparameters, initialstates, seasonalcoefs);
                n.components <- basicparams$n.components;
                maxlag <- basicparams$maxlag;
                matxt <- basicparams$matxt;
                vecg <- basicparams$vecg;
                estimate.phi <- basicparams$estimate.phi;
                phi <- basicparams$phi;
                modellags <- basicparams$modellags;

                Cs <- C.values(bounds,Ttype,Stype,vecg,matxt,phi,maxlag,n.components,matxtreg);
                C <- Cs$C;
                C.upper <- Cs$C.upper;
                C.lower <- Cs$C.lower;

                res <- nloptr(C, CF, lb=C.lower, ub=C.upper,
                              opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
                C <- res$solution;

                n.param <- n.components + damped + (n.components - (Stype!="N")) + maxlag*(Stype!="N");

# Change CF.type for the more appropriate model selection
                if(multisteps==TRUE){
                    CF.type <- "GV";
                }
                else{
                    CF.type <- "MSE";
                }
                IC.values <- IC.calc(n.param=n.param,C=res$solution,Etype=Etype);
                ICs <- IC.values$ICs;
#Change back
                CF.type <- CF.type.original;

                results[[j]] <- c(ICs,Etype,Ttype,Stype,damped,res$objective,C);
            }
            if(silent==FALSE){
                cat("... Done! \n");
            }
            IC.selection <- rep(NA,length(models.pool));
            for(i in 1:length(models.pool)){
                IC.selection[i] <- as.numeric(eval(parse(text=paste0("results[[",i,"]]['",IC,"']"))));
            }

            IC.selection[is.nan(IC.selection)] <- 1E100;

            if(any(unlist(strsplit(model,""))=="C")){
                IC.selection <- IC.selection/(h^multisteps);
                IC.weights <- exp(-0.5*(IC.selection-min(IC.selection)))/sum(exp(-0.5*(IC.selection-min(IC.selection))));
                ICs <- sum(IC.selection * IC.weights);
            }
            else{
                i <- which(IC.selection==min(IC.selection))[1];
                results <- results[[i]];

                Etype <- results[4];
                Ttype <- results[5];
                Stype <- results[6];
                damped <- as.logical(results[7]);
                if(damped==TRUE){
                    phi <- NULL;
                }
                CF.objective <- as.numeric(results[8]);
                C <- as.numeric(results[-c(1:8)]);

                basicparams <- initparams(Ttype, Stype, datafreq, obs, y,
                                          damped, phi, smoothingparameters, initialstates, seasonalcoefs);
                n.components <- basicparams$n.components;
                maxlag <- basicparams$maxlag;
                matxt <- basicparams$matxt;
                vecg <- basicparams$vecg;
                estimate.phi <- basicparams$estimate.phi;
                phi <- basicparams$phi;
                modellags <- basicparams$modellags;
            }
        }
        else{
            Cs <- C.values(bounds,Ttype,Stype,vecg,matxt,phi,maxlag,n.components,matxtreg);
            C <- Cs$C;
            C.upper <- Cs$C.upper;
            C.lower <- Cs$C.lower;

            res <- nloptr(C, CF, lb=C.lower, ub=C.upper,
                                  opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
            C <- res$solution;
            CF.objective <- res$objective;
        }
    }
    else{
            Cs <- C.values(bounds,Ttype,Stype,vecg,matxt,phi,maxlag,n.components,matxtreg);
            C <- Cs$C;
    }

    if(all(unlist(strsplit(model,""))!="C")){
        init.ets <- etsmatrices(matxt, vecg, phi, matrix(C,nrow=1), n.components, modellags,
                                Ttype, Stype, n.exovars, matxtreg, estimate.persistence,
                                estimate.phi, estimate.initial, estimate.initial.season,
                                estimate.xreg);
        vecg <- init.ets$vecg;
        phi <- init.ets$phi;
        matxt <- init.ets$matxt;
        matxtreg <- init.ets$matxtreg;
        matF <- init.ets$matF;
        matw <- init.ets$matw;

        if(damped==TRUE){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        else{
            model <- paste0(Etype,Ttype,Stype);
        }

        fitting <- fitterwrap(matxt, matF, matw, y, vecg,
                              modellags, Etype, Ttype, Stype,
                              matwex, matxtreg, matv, matF2, vecg2)
        matxt <- ts(fitting$matxt,start=(time(data)[1] - deltat(data)*maxlag),frequency=datafreq);
        y.fit <- ts(fitting$yfit,start=start(data),frequency=datafreq);

        if(!is.null(xreg)){
# Write down the matxtreg and copy values for the holdout
            matxtreg[1:nrow(fitting$xtreg),] <- fitting$xtreg;
        }

# Calculate the tails of matxtreg and matxt
        statestails <- statetailwrap(matrix(rbind(matxt[(obs+1):(obs+maxlag),],matrix(NA,h-1,n.components)),h+maxlag-1,n.components), matF,
                                     matrix(matxtreg[(obs.xt-h):(obs.xt),],h+1,n.exovars), matF2,
                                     modellags, Ttype, Stype);
        if(!is.null(xreg)){
# Write down the matxtreg and produce values for the holdout
            matxtreg[(obs.xt-h):(obs.xt),] <- statestails$xtreg;
        }

        errors.mat <- ts(errorerwrap(matxt, matF, matw, y,
                                     h, Etype, Ttype, Stype, modellags,
                                     matwex, matxtreg, matv, matF2, vecg2),
                         start=start(data),frequency=frequency(data));
        colnames(errors.mat) <- paste0("Error",c(1:h));
        errors <- ts(errors.mat[,1],start=start(data),frequency=datafreq);

        y.for <- ts(forecasterwrap(matrix(matxt[(obs+1):(obs+maxlag),],nrow=maxlag),
                                   matF, matw, h, Ttype, Stype, modellags,
                                   matrix(matwex[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                                   matrix(matxtreg[(obs.all-h+1):(obs.all),],ncol=n.exovars)),
                    start=time(data)[obs]+deltat(data),frequency=datafreq);

        if(estimate.persistence==FALSE & estimate.phi==FALSE & estimate.initial==FALSE & estimate.initial.season==FALSE){
            C <- c(vecg,phi,initial,initial.season);
            errors.mat.obs <- obs - h + 1;
            CF.objective <- CF(C);
            n.param <- 0;
        }
        else{
            n.param <- n.components*estimate.persistence + estimate.phi +
                (n.components - (Stype!="N"))*estimate.initial + maxlag*estimate.initial.season;
        }

        if(!is.null(xreg)){
            n.param <- n.param + n.exovars;
        }

        s2 <- as.vector(sum(errors^2)/(obs-n.param));
# Write down the forecasting intervals
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
            if(Etype=="A"){
                y.low <- ts(c(y.for) + quantvalues$lower,start=start(y.for),frequency=frequency(data));
                y.high <- ts(c(y.for) + quantvalues$upper,start=start(y.for),frequency=frequency(data));
            }
            else{
                y.low <- ts(c(y.for) * (1 + quantvalues$lower),start=start(y.for),frequency=frequency(data));
                y.high <- ts(c(y.for) * (1 + quantvalues$upper),start=start(y.for),frequency=frequency(data));
            }
        }
        else{
            y.low <- NA;
            y.high <- NA;
        }

# Change CF.type for the more appropriate model selection
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
            FI <- NULL;
        }
# Calculate IC values
        IC.values <- IC.calc(n.param=n.param,C=C,Etype=Etype);
        llikelihood <- IC.values$llikelihood;
        ICs <- IC.values$ICs;
# Change back
        if(multisteps==TRUE){
            CF.type <- CF.type.original;
        }
        else{
            CF.type <- CF.type.original;
        }

        component.names <- "level";
        if(Ttype!="N"){
            component.names <- c(component.names,"trend");
        }
        if(Stype!="N"){
            component.names <- c(component.names,"seasonality");
        }

        if(!is.null(xreg)){
            matxt <- cbind(matxt,matxtreg[1:nrow(matxt),]);
            colnames(matxt) <- c(component.names,exocomponent.names);
        }
        else{
            colnames(matxt) <- c(component.names);
        }

# Write down the initials. Done especially for Nikos and issue #10
        initial <- matxt[maxlag,1:(n.components - (Stype!="N"))]

        if(Stype!="N"){
          initial.season <- matxt[1:maxlag,n.components]
        }
    }
    else{
# Produce the forecasts using AIC weights
        models.number <- length(IC.selection);
        model.current <- rep(NA,models.number);
        fitted.list <- matrix(NA,obs,models.number);
        errors.list <- matrix(NA,obs,models.number);
        forecasts.list <- matrix(NA,h,models.number);
        if(intervals==TRUE){
             lower.list <- matrix(NA,h,models.number);
             upper.list <- matrix(NA,h,models.number);
        }

        for(i in 1:length(IC.selection)){
# Get all the parameters from the model
            Etype <- results[[i]][4];
            Ttype <- results[[i]][5];
            Stype <- results[[i]][6];
            damped <- as.logical(results[[i]][7]);
            CF.objective <- as.numeric(results[[i]][8]);
            C <- as.numeric(results[[i]][-c(1:8)]);

            basicparams <- initparams(Ttype, Stype, datafreq, obs, y,
                                      damped, phi, smoothingparameters, initialstates, seasonalcoefs);
            n.components <- basicparams$n.components;
            maxlag <- basicparams$maxlag;
            matxt <- basicparams$matxt;
            vecg <- basicparams$vecg;
            estimate.phi <- basicparams$estimate.phi;
            phi <- basicparams$phi;
            modellags <- basicparams$modellags;

            init.ets <- etsmatrices(matxt, vecg, phi, matrix(C,nrow=1), n.components, modellags,
                                    Ttype, Stype, n.exovars, matxtreg, estimate.persistence,
                                    estimate.phi, estimate.initial, estimate.initial.season,
                                    estimate.xreg);
            vecg <- init.ets$vecg;
            phi <- init.ets$phi;
            matxt <- init.ets$matxt;
            matxtreg <- init.ets$matxtreg;
            matF <- init.ets$matF;
            matw <- init.ets$matw;

            fitting <- fitterwrap(matxt, matF, matw, y, vecg,
                                  modellags, Etype, Ttype, Stype,
                                  matwex, matxtreg, matv, matF2, vecg2);
            matxt <- fitting$matxt;
            y.fit <- fitting$yfit;

            if(!is.null(xreg)){
# Write down the matxtreg and copy values for the holdout
                matxtreg[1:nrow(fitting$xtreg),] <- fitting$xtreg;
            }

# Calculate the tails of matxtreg and matxt
            statestails <- statetailwrap(matrix(rbind(matxt[(obs+1):(obs+maxlag),],matrix(NA,h-1,n.components)),h+maxlag-1,n.components), matF,
                                         matrix(matxtreg[(obs.xt-h):(obs.xt),],h+1,n.exovars), matF2,
                                         modellags, Ttype, Stype);
            if(!is.null(xreg)){
# Write down the matxtreg and produce values for the holdout
                matxtreg[(obs.xt-h):(obs.xt),] <- statestails$xtreg;
            }

            errors.mat <- errorerwrap(matxt, matF, matw, y,
                                      h, Etype, Ttype, Stype, modellags,
                                      matwex, matxtreg, matv, matF2, vecg2);
            colnames(errors.mat) <- paste0("Error",c(1:h));
            errors <- errors.mat[,1];
# Produce point and interval forecasts
            y.for <- forecasterwrap(matrix(matxt[(obs+1):(obs+maxlag),],nrow=maxlag),
                                    matF, matw, h, Ttype, Stype, modellags,
                                    matrix(matwex[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                                    matrix(matxtreg[(obs.all-h+1):(obs.all),],ncol=n.exovars));

        s2 <- as.vector(sum(errors^2)/(obs-n.param));
# Write down the forecasting intervals
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
                if(Etype=="A"){
                    y.low <- ts(c(y.for) + quantvalues$lower,start=start(y.for),frequency=frequency(data));
                    y.high <- ts(c(y.for) + quantvalues$upper,start=start(y.for),frequency=frequency(data));
                }
                else{
                    y.low <- ts(c(y.for) * (1 + quantvalues$lower),start=start(y.for),frequency=frequency(data));
                    y.high <- ts(c(y.for) * (1 + quantvalues$upper),start=start(y.for),frequency=frequency(data));
                }
            }
            else{
                y.low <- NA;
                y.high <- NA;
            }

            fitted.list[,i] <- y.fit;
            forecasts.list[,i] <- y.for;
            if(intervals==TRUE){
                lower.list[,i] <- y.low;
                upper.list[,i] <- y.high;
            }
            phi <- NULL;

            if(damped==TRUE){
                model.current[i] <- paste0(Etype,Ttype,"d",Stype);
            }
            else{
                model.current[i] <- paste0(Etype,Ttype,Stype);
            }
        }
        y.fit <- ts(fitted.list %*% IC.weights,start=start(data),frequency=frequency(data));
        y.for <- ts(forecasts.list %*% IC.weights,start=time(data)[obs]+deltat(data),frequency=frequency(data));
        errors <- ts(c(y) - y.fit,start=start(data),frequency=frequency(data));
        names(IC.weights) <- model.current;
        if(intervals==TRUE){
            y.low <- ts(lower.list %*% IC.weights,start=start(y.for),frequency=frequency(data));
            y.high <- ts(upper.list %*% IC.weights,start=start(y.for),frequency=frequency(data));
        }
        else{
            y.low <- NA;
            y.high <- NA;
        }
        names(ICs) <- paste0("Combined ",IC);
    }

    if(any(is.na(y.fit),is.na(y.for))){
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

    if(holdout==TRUE){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- c(MAPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(diff(as.vector(data)[1:obs])))),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(as.vector(data)[1:obs]))),
                           MPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           SMAPE(as.vector(y.holdout),as.vector(y.for),digits=5));
        names(errormeasures) <- c("MAPE","MASE","MASALE","MPE","SMAPE");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    modelname <- paste0("ETS(",model,")");

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
    if(all(unlist(strsplit(model,""))!="C")){
        if(damped==TRUE){
            phivalue <- phi;
        }
        else{
            phivalue <- NULL;
        }
        ssoutput(Sys.time() - start.time, modelname, persistence=vecg, transition=NULL, measurement=NULL,
                 phi=phivalue, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                 n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
                 CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
                 int.type=int.type, int.w=int.w, ICs=ICs,
                 holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);
    }
    else{
        cat(paste0("AIC weights were used to produce the combination of forecasts\n"));
        ssoutput(Sys.time() - start.time, modelname, persistence=NULL, transition=NULL, measurement=NULL,
                 phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                 n.components=NULL, s2=NULL, hadxreg=!is.null(xreg), wentwild=go.wild,
                 CF.type=CF.type, CF.objective=NULL, intervals=intervals,
                 int.type=int.type, int.w=int.w, ICs=ICs,
                 holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);
    }
}

    if(all(unlist(strsplit(model,""))!="C")){
        return(list(model=model,persistence=as.vector(vecg),phi=phi,states=matxt,
                    initial=initial,initial.season=initial.season,fitted=y.fit,
                    forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                    errors=errors.mat,actuals=data,holdout=y.holdout,ICs=ICs,
                    CF=CF.objective,CF.type=CF.type,FI=FI,xreg=xreg,accuracy=errormeasures));
    }
    else{
        return(list(model=model,fitted=y.fit,forecast=y.for,
                    lower=y.low,upper=y.high,residuals=errors,
                    actuals=data,holdout=y.holdout,ICs=IC.weights,
                    CF.type=CF.type,xreg=xreg,accuracy=errormeasures));
    }
}
