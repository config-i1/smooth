es <- function(data, model="ZZZ", persistence=NULL, phi=NULL,
               bounds=c("usual","admissible"), initial=NULL,
               initial.season=NULL, IC=c("AICc","AIC","BIC"),
               trace=FALSE, CF.type=c("TLV","GV","TV","hsteps","MSE","MAE","HAM"),
               FI=FALSE, intervals=FALSE, int.w=0.95,
               int.type=c("parametric","semiparametric","nonparametric"),
               xreg=NULL, holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
               ...){

# Start measuring the time of calculations
    start.time <- Sys.time();
    
    bounds <- substring(bounds[1],1,1);
    IC <- IC[1];
    CF.type <- CF.type[1];

    int.type <- substring(int.type[1],1,1);
# Check the provided type of interval
    if(int.type!="p" & int.type!="s" & int.type!="n"){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'semiparametric'."));
        int.type <- "s";
    }
#### While intervals are not fully supported, use semi-parametric instead of parametric.
    if(int.type=="p"){
        int.type <- "s";
    }

# Check if the data is vector
    if(!is.numeric(data) & !is.ts(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }

# Check if CF.type is appropriate in the case of trace==TRUE
    if(trace==TRUE){
        if(CF.type!="GV" & CF.type!="TLV" & CF.type!="TV" & CF.type!="hsteps"){
            message(paste0("The strange Cost Function is chosen for trace: ",CF.type));
            sowhat(CF.type);
            message("Switching to 'TLV'");
            CF.type <- "TLV";
        }
    }
    else(
        if(CF.type!="MAE" & CF.type!="HAM" & CF.type!="MSE"){
            if(CF.type!="TLV"){
                message(paste0("Weird Cost Function defined: ",CF.type, ". Did you forget to switch trace on?"));
                sowhat(CF.type);
                message("Switching to 'MSE'");
            }
            CF.type <- "MSE";
        }
    )
    CF.type.original <- CF.type;

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
            stop(paste0("You have defined a strange model(s) in the pool: ",
                           paste0(model[nchar(model)>4],collapse=",")),call.=FALSE);
        }
        else if(any(substr(model,1,1)!="A" & substr(model,1,1)!="M")){
            stop(paste0("You have defined a strange model(s) in the pool: ",
                           paste0(model[substr(model,1,1)!="A" & substr(model,1,1)!="M"],collapse=",")),call.=FALSE);
        }
        else if(any(substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                    substr(model,2,2)!="M")){
            stop(paste0("You have defined a strange model(s) in the pool: ",
                           paste0(model[substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                                 substr(model,2,2)!="M"],collapse=",")),call.=FALSE);
        }
        else if(any(substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                    substr(model,3,3)!="M" & substr(model,3,3)!="d")){
            stop(paste0("You have defined a strange model(s) in the pool: ",
                           paste0(model[substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                                 substr(model,3,3)!="M" & substr(model,3,3)!="d"],collapse=",")),call.=FALSE);
        }
        else if(any(nchar(model)==4 & substr(model,4,4)!="N" &
                    substr(model,4,4)!="A" & substr(model,4,4)!="M")){
            stop(paste0("You have defined a strange model(s) in the pool: ",
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
    y <- as.vector(data);

# Check if the data is ts-object
    if(!is.ts(data) & Stype!="N"){
        message("The provided data is not ts object. Only non-seasonal models are available.");
        Stype <- "N";
    }
    datafreq <- frequency(data);

# Check the length of the provided data
    if(Stype!="N" & (obs/datafreq)<2){
        message("Not enough observations for the seasonal model. Only non-seasonal models are available.");
        Stype <- "N";
    }

# If model selection is chosen, forget about the initial values and persistence
    if(Etype=="Z" | Ttype=="Z" | Stype=="Z"){
        if(!is.null(initial) | !is.null(initial.season) | !is.null(persistence) | !is.null(phi)){
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
        if(!is.numeric(persistence) | !is.vector(persistence)){
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
        if(!is.numeric(initial) | !is.vector(initial)){
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
    if(Etype!="Z" & Etype!="A" & Etype!="M"){
        message("Wrong error type! Should be 'Z', 'A' or 'M'.");
        message("Changing to 'Z'");
        Etype <- "Z";
    }

### Check the trend type
    if(Ttype!="Z" & Ttype!="N" & Ttype!="A" & Ttype!="M"){
        message("Wrong trend type! Should be 'Z', 'N', 'A' or 'M'.");
        message("Changing to 'Z'");
        Ttype <- "Z";
    }

### Check the seasonality type
    if(Stype!="Z" & Stype!="N" & Stype!="A" & Stype!="M"){
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
    if(Stype!="N" & datafreq==1){
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
##### The case with vectors and ts objects, but not matrices #####
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
        matxtreg <- matrix(NA,max(obs+datafreq,obs.all),1);
        exocomponent.names <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
        matxtreg[1:datafreq,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
        }
##### The case with matrices and data frames #####
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
            matxtreg <- matrix(NA,max(obs+datafreq,obs.all),n.exovars);
            exocomponent.names <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
            matwex <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
            matxtreg[1:datafreq,] <- rep(t(solve(t(mat.x[1:obs,]) %*% mat.x[1:obs,],tol=1e-50) %*% t(mat.x[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],each=datafreq);
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=F);
        }
        estimate.xreg <- TRUE;
    }
    else{
# "1" is needed for the final forecast simplification
        n.exovars <- 1;
        matwex <- matrix(0,max(obs+datafreq,obs.all),1);
        matxtreg <- matrix(0,max(obs+datafreq,obs.all),1);
        estimate.xreg <- FALSE;
    }

##### All the function should be transfered into optimizerwrap #####
# Cost function for ETS
CF <- function(C){

    init.ets <- etsmatrices(matxt, vecg, phi, matrix(C,nrow=1), n.components, seasfreq, Ttype, Stype, n.exovars, matxtreg,
                            estimate.persistence, estimate.phi, estimate.initial, estimate.initial.season, estimate.xreg);

    if(estimate.persistence==TRUE){
        if(bounds=="a" & (Ttype!="N") & (Stype!="N")){
            Theta.func <- function(Theta){
                return(abs((init.ets$phi*C[1]+init.ets$phi+1)/(C[3]) + ((init.ets$phi-1)*(1+cos(Theta)-cos(seasfreq*Theta))+cos((seasfreq-1)*Theta)-init.ets$phi*cos((seasfreq+1)*Theta))/(2*(1+cos(Theta))*(1-cos(seasfreq*Theta)))));
            }
            Theta <- 0.1;
            Theta <- suppressWarnings(optim(Theta,Theta.func,method="Brent",lower=0,upper=1)$par);
        }
        else{
            Theta <- 0;
        }
        CF.res <- costfunc(init.ets$matxt,init.ets$matF,init.ets$matw,as.matrix(y[1:obs]),init.ets$vecg,h,Etype,Ttype,Stype,seasfreq,trace,CF.type,normalizer,matwex,init.ets$matxtreg,bounds,init.ets$phi,Theta);
    }
    else{
        CF.res <- optimizerwrap(init.ets$matxt,init.ets$matF,init.ets$matw,as.matrix(y[1:obs]),init.ets$vecg,h,Etype,Ttype,Stype,seasfreq,trace,CF.type,normalizer,matwex,init.ets$matxtreg);
    }

    if(is.nan(CF.res) | is.na(CF.res) | is.infinite(CF.res)){
        CF.res <- 1e100;
    }

    return(CF.res);
}

# Function constructs default bounds where C values should lie
C.values <- function(bounds,Ttype,Stype,vecg,matxt,phi,seasfreq,n.components,matxtreg){
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
            C <- c(C,matxt[seasfreq,1:(n.components - (Stype!="N"))]);
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
                C <- c(C,matxt[1:seasfreq,n.components]);
                if(Stype=="A"){
                    C.lower <- c(C.lower,rep(-Inf,seasfreq));
                    C.upper <- c(C.upper,rep(Inf,seasfreq));
                }
                else{
                    C.lower <- c(C.lower,rep(0,seasfreq));
                    C.upper <- c(C.upper,rep(10,seasfreq));
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
            C <- c(C,matxt[seasfreq,1:(n.components - (Stype!="N"))]);
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
                C <- c(C,matxt[1:seasfreq,n.components]);
                if(Stype=="A"){
                    C.lower <- c(C.lower,rep(-Inf,seasfreq));
                    C.upper <- c(C.upper,rep(Inf,seasfreq));
                }
                else{
                    C.lower <- c(C.lower,rep(-0.0001,seasfreq));
                    C.upper <- c(C.upper,rep(20,seasfreq));
                }
            }
        }
    }

    if(!is.null(xreg)){
        C <- c(C,matxtreg[seasfreq,]);
        C.lower <- c(C.lower,rep(-Inf,n.exovars));
        C.upper <- c(C.upper,rep(Inf,n.exovars));
    }
    
    C <- C[!is.na(C)];
    C.lower <- C.lower[!is.na(C.lower)];
    C.upper <- C.upper[!is.na(C.upper)];
    
    return(list(C=C,C.lower=C.lower,C.upper=C.upper));
}

Likelihood.value <- function(C){
    if((trace==TRUE) & (CF.type=="GV")){
        return(-obs/2 *((h^trace)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(-obs/2 *((h^trace)*log(2*pi*exp(1)) + log(CF(C))));
    }
}

## Function calculates ICs
IC.calc <- function(n.param=n.param,C,Etype=Etype){
# Information criteria are calculated with the constant part "log(2*pi*exp(1)*h+log(obs))*obs".
# And it is based on the mean of the sum squared residuals either than sum.
# Hyndman likelihood is: llikelihood <- obs*log(obs*CF.objective)
    llikelihood <- Likelihood.value(C);

    AIC.coef <- 2*n.param*h^trace - 2*llikelihood;
    AICc.coef <- AIC.coef + 2 * n.param*h^trace * (n.param + 1) / (obs - n.param - 1);
    BIC.coef <- log(obs)*n.param*h^trace - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

    return(list(llikelihood=llikelihood,ICs=ICs));
}

#Function allows to estimate the coefficients of the simple quantile regression. Used in intervals construction.
quantfunc <- function(A){
    ee <- ye - (A[1] + A[2]*xe + A[3]*xe^2);
    return((1-quant)*sum(abs(ee[which(ee<0)]))+quant*sum(abs(ee[which(ee>=0)])));
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
            assign("basicparams",initparams(Ttype, Stype, datafreq, obs, as.matrix(y), damped, phi, smoothingparameters, initialstates, seasonalcoefs),inherits=TRUE);
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
                initialstates[1,2] <- cov(y[1:min(12,obs)],c(1:min(12,obs)))/var(c(1:min(12,obs)));
                initialstates[1,1] <- mean(y[1:min(12,obs)]) - initialstates[1,2] * (mean(c(1:min(12,obs))) - 1);
                initialstates[1,3] <- mean(y[1:min(12,obs)]);
                initialstates[1,4] <- 1;
            }
            else{
                initialstates <- matrix(rep(mean(y[1:min(12,obs)]),4),nrow=1);
            }
            assign("estimate.initial",TRUE,inherits=inherits);
            assign("initialstates",initialstates,inherits=inherits);
            assign("basicparams",initparams(Ttype, Stype, datafreq, obs, as.matrix(y), damped, phi, smoothingparameters, initialstates, seasonalcoefs),inherits=TRUE);
            assign("matxt",basicparams$matxt,inherits=inherits);
        }
    }

# Check the seasonal inital vector length
    if(!is.null(initial.season)){
        if(frequency(data)!=length(initial.season)){
            message("The length of seasonal initial states does not correspond to the frequency of the data!");
            message("Values of initial seasonals will be estimated.");
            seasonalcoefs <- decompose(ts(y[1:obs],frequency=datafreq),type="additive")$seasonal[1:datafreq];
            seasonalcoefs <- cbind(seasonalcoefs,decompose(ts(y[1:obs],frequency=datafreq),type="multiplicative")$seasonal[1:datafreq]);
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
            initialstates[1,2] <- cov(y[1:min(12,obs)],c(1:min(12,obs)))/var(c(1:min(12,obs)));
            initialstates[1,1] <- mean(y[1:min(12,obs)]) - initialstates[1,2] * (mean(c(1:min(12,obs))) - 1);
            initialstates[1,3] <- mean(y[1:min(12,obs)]);
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
            initialstates <- matrix(rep(mean(y[1:min(12,obs)]),4),nrow=1);
        }
        else{
            estimate.initial <- FALSE;
            initialstates <- matrix(rep(initial,4),nrow=1);
        }
    }

# Define matrix of seasonal coefficients. The first column consists of additive, the second - multiplicative elements
# If the seasonal model is chosen and initials are provided, fill in the first "seasfreq" values of seasonal component.
    if(Stype!="N"){
        if(is.null(initial.season)){
            estimate.initial.season <- TRUE;
            seasonalcoefs <- decompose(ts(y[1:obs],frequency=datafreq),type="additive")$seasonal[1:datafreq];
            seasonalcoefs <- cbind(seasonalcoefs,decompose(ts(y[1:obs],frequency=datafreq),type="multiplicative")$seasonal[1:datafreq]);
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

    normalizer <- mean(abs(diff(y[1:obs])));

# If we use trace, define matrix of errors.
    if(trace==TRUE){
        mat.error <- matrix(NA,nrow=obs,ncol=h);
    }
    else{
        mat.error <- matrix(NA,nrow=obs,ncol=1);
    }

    basicparams <- initparams(Ttype, Stype, datafreq, obs, as.matrix(y),
                              damped, phi, smoothingparameters, initialstates, seasonalcoefs);
    n.components <- basicparams$n.components;
    seasfreq <- basicparams$seasfreq;
    matxt <- basicparams$matxt;
    vecg <- basicparams$vecg;
    estimate.phi <- basicparams$estimate.phi;
    phi <- basicparams$phi;

    checker(inherits=TRUE);

############ Start the estimation depending on the model ############
# Fill in the vector of initial values and vector of constrains used in estimation
# This also should include in theory  "| estimate.phi==TRUE",
#    but it doesn't make much sense and makes things more complicated
    if(estimate.persistence==TRUE | estimate.initial==TRUE | estimate.initial.season==TRUE){

# Number of observations in the mat.error matrix excluding NAs.
        errors.mat.obs <- obs - h + 1;
##### If auto selection is used (for model="ZZZ" or model="CCC"), then let's start misbehaving...
        if(any(unlist(strsplit(model,""))=="C") | (Etype=="Z" | Ttype=="Z" | Stype=="Z")){
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

# Number of observations in the mat.error matrix excluding NAs.
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

                basicparams <- initparams(Ttype, Stype, datafreq, obs, as.matrix(y),
                                          damped, phi, smoothingparameters, initialstates, seasonalcoefs);
                n.components <- basicparams$n.components;
                seasfreq <- basicparams$seasfreq;
                matxt <- basicparams$matxt;
                vecg <- basicparams$vecg;
                estimate.phi <- basicparams$estimate.phi;
                phi <- basicparams$phi;

                Cs <- C.values(bounds,Ttype,Stype,vecg,matxt,phi,seasfreq,n.components,matxtreg);
                C <- Cs$C;
                C.upper <- Cs$C.upper;
                C.lower <- Cs$C.lower;

                res <- nloptr::nloptr(C, CF, lb=C.lower, ub=C.upper,
                                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
                C <- res$solution;

                n.param <- n.components + damped + (n.components - (Stype!="N")) + seasfreq*(Stype!="N");

# Change CF.type for the more appropriate model selection
                if(trace==TRUE){
                    CF.type <- "GV";
                }
                else{
                    CF.type <- "MSE";
                }
                IC.values <- IC.calc(n.param=n.param,C=res$solution,Etype=Etype);
                ICs <- IC.values$ICs;
#Change back
                if(trace==TRUE){
                    CF.type <- CF.type.original;
                }
                else{
                    CF.type <- CF.type.original;
                }

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
                IC.selection <- IC.selection/(h^trace);
                IC.weights <- exp(-0.5*(IC.selection-min(IC.selection)))/sum(exp(-0.5*(IC.selection-min(IC.selection))));
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

                basicparams <- initparams(Ttype, Stype, datafreq, obs, as.matrix(y),
                                          damped, phi, smoothingparameters, initialstates, seasonalcoefs);
                n.components <- basicparams$n.components;
                seasfreq <- basicparams$seasfreq;
                matxt <- basicparams$matxt;
                vecg <- basicparams$vecg;
                estimate.phi <- basicparams$estimate.phi;
                phi <- basicparams$phi;
            }
        }
        else{
            Cs <- C.values(bounds,Ttype,Stype,vecg,matxt,phi,seasfreq,n.components,matxtreg);
            C <- Cs$C;
            C.upper <- Cs$C.upper;
            C.lower <- Cs$C.lower;

            res <- nloptr::nloptr(C, CF, lb=C.lower, ub=C.upper,
                                  opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
            C <- res$solution;
            CF.objective <- res$objective;
        }
    }
    else{
            Cs <- C.values(bounds,Ttype,Stype,vecg,matxt,phi,seasfreq,n.components,matxtreg);
            C <- Cs$C;
    }

    if(all(unlist(strsplit(model,""))!="C")){
        init.ets <- etsmatrices(matxt, vecg, phi, matrix(C,nrow=1), n.components, seasfreq, Ttype, Stype, n.exovars, matxtreg,
                                estimate.persistence, estimate.phi, estimate.initial, estimate.initial.season, estimate.xreg);
        vecg <- init.ets$vecg;
        phi <- init.ets$phi;
        matxt <- init.ets$matxt;
        matxtreg <- init.ets$matxtreg;
        matF <- init.ets$matF;
        matw <- init.ets$matw;
    }
    
    if(all(unlist(strsplit(model,""))!="C")){
        if(damped==TRUE){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        else{
            model <- paste0(Etype,Ttype,Stype);
        }

        fitting <- fitterwrap(matxt,matF,matw,as.matrix(y[1:obs]),vecg,Etype,Ttype,Stype,seasfreq,matwex,matxtreg);
        matxt <- ts(fitting$matxt,start=(time(data)[1] - deltat(data)*seasfreq),frequency=datafreq);
        y.fit <- ts(fitting$yfit,start=start(data),frequency=datafreq);

        if(!is.null(xreg)){
# Write down the matxtreg and copy values for the holdout
            matxtreg[1:nrow(fitting$matxtreg),] <- fitting$matxtreg;
            matxtreg[(obs.all-h+1):obs.all,] <- rep(matxtreg[1,],each=h);
        }

        errors.mat <- ts(errorerwrap(matxt,matF,matw,as.matrix(y[1:obs]),h,Etype,Ttype,Stype,seasfreq,TRUE,matwex,matxtreg),start=start(data),frequency=frequency(data));
        colnames(errors.mat) <- paste0("Error",c(1:h));
        errors <- ts(errors.mat[,1],start=start(data),frequency=datafreq);

        y.for <- ts(forecasterwrap(matrix(matxt[(obs+1):(obs+seasfreq),],nrow=seasfreq),matF,matw,h,Ttype,Stype,seasfreq,matrix(matwex[(obs.all-h+1):(obs.all),],ncol=n.exovars),matrix(matxtreg[(obs.all-h+1):(obs.all),],ncol=n.exovars)),start=time(data)[obs]+deltat(data),frequency=datafreq);

# Write down the forecasting intervals
        if(intervals==TRUE){
            if(int.type=="p"){
#            y.var <- forecastervar(matF,matrix(matw[1,],nrow=1),vecg,h,var(errors),Etype,Ttype,Stype,seasfreq)
                y.low <- NA;
                y.high <- NA;
            }
            else if(int.type=="s"){
                y.var <- colMeans(errors.mat^2,na.rm=T);
                if(Etype=="A"){
                    y.low <- ts(y.for + qt((1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var),start=start(y.for),frequency=frequency(data));
                    y.high <- ts(y.for + qt(1-(1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var),start=start(y.for),frequency=frequency(data));
                }
                else{
                    y.low <- ts(y.for*(1 + qt((1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var)),start=start(y.for),frequency=frequency(data));
                    y.high <- ts(y.for*(1 + qt(1-(1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var)),start=start(y.for),frequency=frequency(data));
                }
            }
            else{
                ye <- errors.mat;
                xe <- matrix(c(1:h),byrow=TRUE,ncol=h,nrow=nrow(errors.mat));
                xe <- xe[!is.na(ye)];
                ye <- ye[!is.na(ye)];

                A <- rep(1,3);
                quant <- (1-int.w)/2;
                A1 <- nlminb(A,quantfunc)$par;
                quant <- 1-(1-int.w)/2;
                A2 <- nlminb(A,quantfunc)$par;
                if(Etype=="A"){
                    y.low <- ts(y.for + A1[1] + A1[2]*c(1:h) + A1[3]*c(1:h)^2,start=start(y.for),frequency=frequency(data));
                    y.high <- ts(y.for + A2[1] + A2[2]*c(1:h) + A2[3]*c(1:h)^2,start=start(y.for),frequency=frequency(data));
                }
                else{
                    y.low <- ts(y.for*(1 + A1[1] + A1[2]*c(1:h) + A1[3]*c(1:h)^2),start=start(y.for),frequency=frequency(data));
                    y.high <- ts(y.for*(1 + A2[1] + A2[2]*c(1:h) + A2[3]*c(1:h)^2),start=start(y.for),frequency=frequency(data));
                }
            }
        }
        else{
            y.low <- NA;
            y.high <- NA;
        }

        if(estimate.persistence==FALSE & estimate.phi==FALSE & estimate.initial==FALSE & estimate.initial.season==FALSE){
            C <- c(vecg,phi,initial,initial.season);
            errors.mat.obs <- obs - h + 1;
            CF.objective <- CF(C);
            n.param <- 0;
        }
        else{
            n.param <- n.components*estimate.persistence + estimate.phi + (n.components - (Stype!="N"))*estimate.initial + seasfreq*estimate.initial.season;
        }

        if(!is.null(xreg)){
            n.param <- n.param + n.exovars;
        }

# Change CF.type for the more appropriate model selection
        if(trace==TRUE){
            CF.type <- "GV";
        }
        else{
            CF.type <- "MSE";
        }

        if(FI==TRUE){
            FI <- numDeriv::hessian(Likelihood.value,C);
        }
        else{
            FI <- NULL;
        }
# Calculate IC values
        IC.values <- IC.calc(n.param=n.param,C=C,Etype=Etype);
        llikelihood <- IC.values$llikelihood;
        ICs <- IC.values$ICs;
# Change back
        if(trace==TRUE){
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

            basicparams <- initparams(Ttype, Stype, datafreq, obs, as.matrix(y),
                                      damped, phi, smoothingparameters, initialstates, seasonalcoefs);
            n.components <- basicparams$n.components;
            seasfreq <- basicparams$seasfreq;
            matxt <- basicparams$matxt;
            vecg <- basicparams$vecg;
            estimate.phi <- basicparams$estimate.phi;
            phi <- basicparams$phi;

            init.ets <- etsmatrices(matxt, vecg, phi, matrix(C,nrow=1), n.components, seasfreq, Ttype, Stype, n.exovars, matxtreg,
                                    estimate.persistence, estimate.phi, estimate.initial, estimate.initial.season, estimate.xreg);
            vecg <- init.ets$vecg;
            phi <- init.ets$phi;
            matxt <- init.ets$matxt;
            matxtreg <- init.ets$matxtreg;
            matF <- init.ets$matF;
            matw <- init.ets$matw;

            fitting <- fitterwrap(matxt,matF,matrix(matw,1,length(matw)),as.matrix(y[1:obs]),matrix(vecg,length(vecg),1),Etype,Ttype,Stype,seasfreq,matwex,matxtreg);
            matxt <- fitting$matxt;
            y.fit <- fitting$yfit;

            if(!is.null(xreg)){
# Write down the matxtreg and copy values for the holdout
                matxtreg[1:nrow(fitting$matxtreg),] <- fitting$matxtreg;
                matxtreg[(obs.all-h+1):obs.all,] <- rep(matxtreg[1,],each=h);
            }

            errors.mat <- errorerwrap(matxt,matF,matrix(matw,1,length(matw)),as.matrix(y[1:obs]),h,Etype,Ttype,Stype,seasfreq,TRUE,matwex,matxtreg);
            colnames(errors.mat) <- paste0("Error",c(1:h));
            errors <- errors.mat[,1];
# Produce point and interval forecasts 
            y.for <- forecasterwrap(matrix(matxt[(obs+1):(obs+seasfreq),],nrow=seasfreq),matF,matrix(matw,nrow=1),h,Ttype,Stype,seasfreq,matrix(matwex[(obs.all-h+1):(obs.all),],ncol=n.exovars),matrix(matxtreg[(obs.all-h+1):(obs.all),],ncol=n.exovars));

# Write down the forecasting intervals
            if(intervals==TRUE){
                if(int.type=="p"){
#            y.var <- forecastervar(matF,matrix(matw[1,],nrow=1),vecg,h,var(errors),Etype,Ttype,Stype,seasfreq)
                    y.low <- NA;
                    y.high <- NA;
                }
                else if(int.type=="s"){
                    y.var <- colMeans(errors.mat^2,na.rm=T);
                    if(Etype=="A"){
                        y.low <- ts(y.for + qt((1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var),start=start(y.for),frequency=datafreq);
                        y.high <- ts(y.for + qt(1-(1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var),start=start(y.for),frequency=datafreq);
                    }
                    else{
                        y.low <- ts(y.for*(1 + qt((1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var)),start=start(y.for),frequency=datafreq);
                        y.high <- ts(y.for*(1 + qt(1-(1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var)),start=start(y.for),frequency=datafreq);
                    }
                }
                else{
                    ye <- errors.mat;
                    xe <- matrix(c(1:h),byrow=TRUE,ncol=h,nrow=nrow(errors.mat));
                    xe <- xe[!is.na(ye)];
                    ye <- ye[!is.na(ye)];

                    A <- rep(1,3);
                    quant <- (1-int.w)/2;
                    A1 <- nlminb(A,quantfunc)$par;
                    quant <- 1-(1-int.w)/2;
                    A2 <- nlminb(A,quantfunc)$par;
                    if(Etype=="A"){
                        y.low <- y.for + A1[1] + A1[2]*c(1:h) + A1[3]*c(1:h)^2;
                        y.high <- ts(y.for + A2[1] + A2[2]*c(1:h) + A2[3]*c(1:h)^2,start=start(y.for),frequency=frequency(data));
                    }
                    else{
                        y.low <- ts(y.for*(1 + A1[1] + A1[2]*c(1:h) + A1[3]*c(1:h)^2),start=start(y.for),frequency=frequency(data));
                        y.high <- ts(y.for*(1 + A2[1] + A2[2]*c(1:h) + A2[3]*c(1:h)^2),start=start(y.for),frequency=frequency(data));
                    }
                }
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
        errors <- ts(y[1:obs] - y.fit,start=start(data),frequency=frequency(data));
        names(IC.weights) <- model.current;
        if(intervals==TRUE){
            y.low <- ts(lower.list %*% IC.weights,start=start(y.for),frequency=frequency(data));
            y.high <- ts(upper.list %*% IC.weights,start=start(y.for),frequency=frequency(data));
        }
        else{
            y.low <- NA;
            y.high <- NA;
        }
    }

    y <- data;

    if(any(is.na(y.fit),is.na(y.for))){
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

    if(holdout==TRUE){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- c(MAPE(as.vector(y.holdout),as.vector(y.for),round=5),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(diff(as.vector(data)[1:obs])))),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(as.vector(data)[1:obs]))),
                           MPE(as.vector(y.holdout),as.vector(y.for),round=5),
                           SMAPE(as.vector(y.holdout),as.vector(y.for),round=5));
        names(errormeasures) <- c("MAPE","MASE","MASALE","MPE","SMAPE");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

if(silent==FALSE){
# Print time elapsed on the construction
    print(paste0("Time elapsed: ",round(as.numeric(Sys.time() - start.time,units="secs"),2)," seconds"));
    if(all(unlist(strsplit(model,""))!="C")){
        print(paste0("Model constructed: ",model));
        print(paste0("Persistence vector: ", paste(round(vecg,3),collapse=", ")));
        if(damped==TRUE){
            print(paste0("Damping parameter: ", round(phi,3)));
        }
        print(paste0("Initial components: ", paste(round(matxt[seasfreq,1:(n.components - (Stype!="N"))],3),collapse=", ")));
        if(Stype!="N"){
            print(paste0("Initial seasonal components: ", paste(round(matxt[1:seasfreq,n.components],3),collapse=", ")));
        }
        if(!is.null(xreg)){
            print(paste0("Xreg coefficients: ", paste(round(matxtreg[seasfreq,],3),collapse=", ")));
        }
        print(paste0("Residuals sigma: ",round(sqrt(mean(errors^2)),3)));
        if(trace==TRUE){
            print(paste0("CF type: trace with ",CF.type, "; CF value is: ",round(CF.objective,0)));
        }
        else{
            print(paste0("CF type: one step ahead using ",CF.type,"; CF value is: ",round(CF.objective,0)));
        }
        print(paste0("AIC: ",round(ICs["AIC"],3)," AICc: ", round(ICs["AICc"],3)));
    }
    else{
        print(paste0("AIC weights were used to produce the combination of forecasts"));
        print(paste0("Residuals sigma: ",round(sqrt(mean(errors^2)),3)));
    }
    if(intervals==TRUE){
        if(int.type=="p"){
            int.type <- "parametric";
        }
        else if(int.type=="s"){
            int.type <- "semiparametric";
        }
        if(int.type=="n"){
            int.type <- "nonparametric";
        }
        print(paste0(int.w*100,"% ",int.type," intervals were constructed"));
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                   lower=y.low,upper=y.high,int.w=int.w,legend=legend);
    }
    else{
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,legend=legend);
    }
#    print(paste0("Biased log-likelihood: ",round((llikelihood - n.param*h^trace),0)))
    if(holdout==TRUE){
        print(paste0("MPE: ",errormeasures["MPE"]*100,"%"));
        print(paste0("MAPE: ",errormeasures["MAPE"]*100,"%"));
        print(paste0("SMAPE: ",errormeasures["SMAPE"]*100,"%"));
        print(paste0("MASE: ",errormeasures["MASE"]));
        print(paste0("MASALE: ",errormeasures["MASALE"]*100,"%"));
        if(intervals==TRUE){
            print(paste0(round(sum(as.vector(data)[(obs+1):obs.all]<y.high &
                    as.vector(data)[(obs+1):obs.all]>y.low)/h*100,0),
                    "% of values are in the interval"));
        }
    }
}

    if(all(unlist(strsplit(model,""))!="C")){
        return(list(model=model,persistence=as.vector(vecg),phi=phi,states=matxt,fitted=y.fit,
                    forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                    errors=errors.mat,actuals=data,holdout=y.holdout,ICs=ICs,
                    CF=CF.objective,FI=FI,xreg=xreg,accuracy=errormeasures));
    }
    else{
        return(list(model=model,fitted=y.fit,forecast=y.for,
                    lower=y.low,upper=y.high,residuals=errors,
                    actuals=data,holdout=y.holdout,ICs=IC.weights,
                    xreg=xreg,accuracy=errormeasures));
    }
}