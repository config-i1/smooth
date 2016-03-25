sim.ets <- function(model="ANN",frequency=1, persistence=NULL, phi=1,
             initial=NULL, initial.season=NULL,
             bounds=c("usual","admissible","restricted"),
             obs=10, nseries=1, silent=FALSE,
             randomizer=c("rnorm","runif","rbeta","rt"),
             iprob=1, ...){
# Function generates data using ETS with Single Source of Error as a data generating process.
#    Copyright (C) 2015 - 2016 Ivan Svetunkov

    bounds <- substring(bounds[1],1,1);
    randomizer <- randomizer[1];

# If chosen model is "AAdN" or anything like that, we are taking the appropriate values
    if(nchar(model)==4){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,4,4);
        if(substring(model,3,3)!="d"){
            if(silent == FALSE){
                message(paste0("You have defined a strange model: ",model));
                sowhat(model);
            }
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        if(Ttype!="N" & phi==1){
            model <- paste0(Etype,Ttype,Stype);
            if(silent == FALSE){
                message(paste0("Damping parameter is set to 1. Changing model to: ",model));
            }
        }
    }
    else if(nchar(model)==3){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,3,3);
        if(phi!=1 & Ttype!="N"){
            model <- paste0(Etype,Ttype,"d",Stype);
            if(silent == FALSE){
                message(paste0("Damping parameter is set to ",phi,". Changing model to: ",model));
            }
        }
    }
    else{
        message(paste0("You have defined a strange model: ",model));
        stop("Cannot proceed.",call.=FALSE);
    }

# In the case of wrong nseries, make it natural number. The same is for obs and frequency.
    nseries <- abs(round(nseries,0));
    obs <- abs(round(obs,0));
    frequency <- abs(round(frequency,0));

    if(!is.null(persistence) & length(persistence)>3){
        stop("The length of persistence vector is wrong! It should not be greater than 3.",call.=FALSE);
    }

    if(phi<0 | phi>2){
        if(silent == FALSE){
            message(paste0("Damping parameter should lie in (0, 2) region! You have chosen phi=",phi,". Be careful!"));
        }
    }

r.value <- function(Etype, Ttype, Stype, xt){
# Function returns the value of r for the error term for the inclusion in transition equation depending on several parameters
    if(Etype=="A"){
# AZZ
        r <- 1;
        r <- rep(r,persistence.length);
        if(Stype=="N" & Ttype=="M"){
            r <- 1 / c(1,xt[1]);
        }
        else if(Stype=="A" & Ttype=="M"){
            r <- 1 / c(1,xt[1],1);
        }
        else if(Stype=="M"){
            if(Ttype=="N"){
                r <- 1 / c(xt[2],xt[1]);
            }
            else if(Ttype=="A"){
                r <- 1 / c(xt[3],xt[3],(matw[1:2] %*% xt[1:2]));
            }
            else {
                r <- 1 / c(xt[3],(xt[1] * xt[3]),(exp(matw[1:2] %*% log(xt[1:2]))));
            }
        }
    }
    else{
        if(Ttype!="M" & Stype!="M"){
# MNN, MAN, MNA, MAA
                r <- matw %*% xt;
                r <- rep(r,persistence.length);
        }
        else if((Ttype=="M" | Ttype=="N") & (Stype=="M" | Stype=="N")){
# MNN, MMN, MNM, MMM
            r <- exp(matF %*% log(xt));
        }
        else if(Ttype=="A" & Stype=="M"){
# MAM
            r <- matw[1:2] %*% xt[1:2];
            r <- c(r,r,xt[3]);
        }
        else if(Ttype=="M" & Stype=="A"){
# MMA
            r <- exp(matw[1:2] %*% log(xt[1:2])) + xt[3];
            r <- c(r, r/xt[1], xt[3]);
        }
    }
    return(r);
}

ry.value <- function(Etype, Ttype, Stype, xt){
# Function returns the value of r for the error term for the inclusion in measurement equation, depending on several parameters
    if(Etype=="A"){
# AZZ
        r <- 1;
    }
    else{
        if(Ttype!="M" & Stype!="M"){
# MNN, MAN, MNA, MAA
                r <- matw %*% xt;
        }
        else if((Ttype=="M" | Ttype=="N") & (Stype=="M" | Stype=="N")){
# MNN, MMN, MNM, MMM
            r <- exp(matw %*% log(xt));
        }
        else if(Ttype=="A" & Stype=="M"){
# MAM
            r <- (matw[1:2] %*% xt[1:2]) * xt[3];
        }
        else if(Ttype=="M" & Stype=="A"){
# MMA
            r <- exp(matw[1:2] %*% log(xt[1:2])) + xt[3];
        }
    }
    return(r);
}

# Check the used model and estimate the length of needed persistence vector.
    if(Etype!="A" & Etype!="M"){
        stop("Wrong error type! Should be 'A' or 'M'.",call.=FALSE);
    }
    else{
# The number of the smoothing parameters needed
        persistence.length <- 1;
# The number initial values of the state vector
        n.components <- 1;
# The lag of components (needed for the seasonal models)
        lags <- 1;
# The names of the state vector components
        component.names <- "level";
        matw <- 1;
# The transition matrix
        matF <- matrix(1,1,1);
# The matrix used for the multiplicative error models. Should contain ^yt
        mat.r <- 1;
    }

# Check the trend type of the model
    if(Ttype!="N" & Ttype!="A" & Ttype!="M"){
        stop("Wrong trend type! Should be 'N', 'A' or 'M'.",call.=FALSE);
    }
    else if(Ttype!="N"){
        if(is.na(phi) | is.null(phi)){
            phi <- 1;
        }
        persistence.length <- persistence.length + 1;
        n.components <- n.components + 1;
        lags <- c(lags,1);
        component.names <- c(component.names,"trend");
        matw <- c(matw,phi);
        matF <- matrix(c(1,0,phi,phi),2,2);
        trend.component=TRUE;
        if(phi!=1){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
    }
    else{
        trend.component=FALSE;
    }

# Check the seasonaity type of the model
    if(Stype!="N" & Stype!="A" & Stype!="M"){
        stop("Wrong seasonality type! Should be 'N', 'A' or 'M'.",call.=FALSE);
    }

    if(Stype!="N" & frequency==1){
        stop("Cannot create the seasonal model with the data frequency 1!",call.=FALSE);
    }

    if(Stype!="N"){
        persistence.length <- persistence.length + 1;
# modelfreq is used in the cases of seasonal models.
#   if modelfreq==1 then non-seasonal data will be produced with the defined frequency.
        modelfreq <- frequency;
        lags <- c(lags,frequency);
        component.names <- c(component.names,"seasonality");
        matw <- c(matw,1);
        seasonal.component <- TRUE;

        if(trend.component==FALSE){
            matF <- matrix(c(1,0,0,1),2,2);
        }
        else{
            matF <- matrix(c(1,0,0,phi,phi,0,0,0,1),3,3);
        }
    }
    else{
        seasonal.component <- FALSE;
        modelfreq <- 1;
    }
# Create vector for the series
    y <- rep(NA,obs);

# Create the matrix of state vectors
    matxt <- matrix(NA,nrow=(obs+modelfreq),ncol=persistence.length);
    colnames(matxt) <- component.names;

# Check the persistence vector length
    if(!is.null(persistence)){
        if(persistence.length != length(persistence)){
            if(silent == FALSE){
                message("The length of persistence vector does not correspond to the chosen model!");
                message("Falling back to random number generator in... now!");
            }
            persistence <- NULL;
        }
    }

# Check the inital vector length
    if(!is.null(initial)){
        if(length(initial)>2){
            stop("The length of the initial value is wrong! It should not be greater than 2.",call.=FALSE);
        }
        if(n.components!=length(initial)){
            if(silent == FALSE){
                message("The length of initial state vector does not correspond to the chosen model!");
                message("Falling back to random number generator in... now!");
            }
            initial <- NULL;
        }
    }

    if(!is.null(initial.season)){
        if(modelfreq!=length(initial.season)){
            if(silent == FALSE){
                message("The length of seasonal initial states does not correspond to the chosen frequency!");
                message("Falling back to random number generator in... now!");
            }
            initial.season <- NULL;
        }
    }

# If the seasonal model is chosen, fill in the first "frequency" values of seasonal component.
    if(seasonal.component==TRUE & !is.null(initial.season)){
        matxt[1:modelfreq,(n.components+1)] <- initial.season;
    }

    if(nseries > 1){
# The array of the components
        arr.xt <- array(NA,c(obs+modelfreq,persistence.length,nseries));
        dimnames(arr.xt) <- list(NULL,component.names,NULL);
# The matrix of the final data
        mat.yt <- matrix(NA,obs,nseries);
# The matrix of the error term
        mat.errors <- matrix(NA,obs,nseries);
# The matrix of smoothing parameters
        mat.g <- matrix(NA,nseries,persistence.length);
        colnames(mat.g) <- c(component.names);
# The vector of likelihoods
        vec.likelihood <- rep(NA,nseries);
# The matrix of ones for intermittency
        mat.ot <- matrix(NA,obs,nseries);

        if(silent == FALSE){
          cat("Series simulated:  ");
        }
    }

# If the chosen randomizer is not rnorm, rt and runif and no parameters are provided, change to rnorm.
    if(randomizer!="rnorm" & randomizer!="rt" & randomizer!="runif" & (any(names(match.call(expand.dots=FALSE)[-1]) == "...")==FALSE)){
        if(silent == FALSE){
            warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
        }
        randomizer = "rnorm";
    }

##### Start the loop #####
for(k in 1:nseries){
###### Produce several matrices with the pregenerated or predefined data, then pass it to Rcpp ######
##### If the vectors are predefined, just copy them #####
# If the persistence is NULL or was of the wrong length, generate the values
    if(is.null(persistence)){
# For the case of "usual" bounds make restrictions on the generated smoothing parameters so the ETS can be "averaging" model.
        if(bounds=="u"){
            vecg <- runif(persistence.length,0,1);
            if(Ttype!="N"){
                vecg[2] <- runif(1,0,vecg[1]);
            }
            if(Stype!="N"){
                vecg[persistence.length] <- runif(1,0,max(0,1-vecg[1]));
            }
        }
        else if(bounds=="r"){
            vecg <- runif(persistence.length,0,0.3);
            if(Ttype!="N"){
                vecg[2] <- runif(1,0,vecg[1]);
            }
            if(Stype!="N"){
                vecg[persistence.length] <- runif(1,0,max(0,1-vecg[1]));
            }
        }
        else if(bounds=="a"){
            vecg <- runif(persistence.length,1-1/phi,1+1/phi);
            if(Ttype!="N"){
                vecg[2] <- runif(1,vecg[1]*(phi-1),(2-vecg[1])*(1+phi));
                if(Stype!="N"){
                    vecg[3] <- runif(1,max(1-1/phi-vecg[1],0),1+1/phi-vecg[1]);
                    B <- phi*(4-3*vecg[3])+vecg[3]*(1-phi)/modelfreq;
                    C <- sqrt(B^2-8*(phi^2*(1-vecg[3])^2+2*(phi-1)*(1-vecg[3])-1)+8*vecg[3]^2*(1-phi)/modelfreq);
                    vecg[1] <- runif(1,1-1/phi-vecg[3]*(1-modelfreq+phi*(1+modelfreq))/(2*phi*modelfreq),(B+C)/(4*phi));
# Solve the equation to get Theta value. Theta
                    Theta.func <- function(Theta){
                        result <- (phi*vecg[1]+phi+1)/(vecg[3]) +
                            ((phi-1)*(1+cos(Theta)-cos(modelfreq*Theta))+cos((modelfreq-1)*Theta)-phi*cos((modelfreq+1)*Theta))/(2*(1+cos(Theta))*(1-cos(modelfreq*Theta)));
                        return(abs(result));
                    }
                    Theta <- 0.1;
                    Theta <- optim(Theta,Theta.func,method="Brent",lower=0,upper=1)$par;

                    D <- (phi*(1-vecg[1])+1)*(1-cos(Theta)) - vecg[3]*((1+phi)*(1-cos(Theta)-cos(modelfreq*Theta))+cos((modelfreq-1)*Theta)+phi*cos((modelfreq+1)*Theta))/(2*(1+cos(Theta))*(1-cos(modelfreq*Theta)));
                    vecg[2] <- runif(1,-(1-phi)*(vecg[3]/modelfreq+vecg[1]),D+(phi-1)*vecg[1]);
                }
            }
            else{
                if(Stype!="N"){
                    vecg[1] <- runif(1,-2/(modelfreq-1),2);
                    vecg[2] <- runif(1,max(-modelfreq*vecg[1],0),2-vecg[1]);
                    vecg[1] <- runif(1,-2/(modelfreq-1),2-vecg[2]);
                }
            }
        }
        else{
            vecg <- runif(persistence.length,0,1);
        }
    }
    else{
        vecg <- persistence;
    }

# Generate initial states of level and trend if they were not supplied
    if(is.null(initial)){
        if(Ttype=="N"){
            matxt[1:modelfreq,1] <- runif(1,0,1000);
        }
        else if(Ttype=="A"){
            matxt[1:modelfreq,1] <- runif(1,0,5000);
            matxt[1:modelfreq,2] <- runif(1,-100,100);
        }
        else{
            matxt[1:modelfreq,1] <- runif(1,500,5000);
            matxt[1:modelfreq,2] <- 1;
        }
    }
    else{
        matxt[1:modelfreq,1:n.components] <- rep(initial,each=modelfreq);
    }

# Generate seasonal states if they were not supplied
    if(seasonal.component==TRUE & is.null(initial.season)){
# Create and normalize seasonal components. Use geometric mean for multiplicative case
        if(Stype == "A"){
            matxt[1:modelfreq,n.components+1] <- runif(modelfreq,-500,500);
            matxt[1:modelfreq,n.components+1] <- matxt[1:modelfreq,n.components+1] - mean(matxt[1:modelfreq,n.components+1]);
        }
        else{
            matxt[1:modelfreq,n.components+1] <- runif(modelfreq,0.3,1.7);
            matxt[1:modelfreq,n.components+1] <- matxt[1:modelfreq,n.components+1] / exp(mean(log(matxt[1:modelfreq,n.components+1])));
        }
    }

# Check if any argument was passed in dots
    if(any(names(match.call(expand.dots=FALSE)[-1]) == "...")==FALSE){
# Create vector of the errors
        if(randomizer=="rnorm" | randomizer=="runif"){
          errors <- eval(parse(text=paste0(randomizer,"(n=",obs,")")));
        }
        else if(randomizer=="rt"){
# The degrees of freedom are df = n - k.
          errors <- rt(obs,obs-(persistence.length + modelfreq));
        }

# Center errors just in case
        errors <- errors - mean(errors);
# Change variance to make some sense. Errors should not be rediculously high and not too low.
        errors <- errors * sqrt(abs(matxt[1,1]));
# If the error is multiplicative, scale it!
        if(Etype=="M" & max(abs(errors))>0.05){
            errors <- 0.05 * errors / max(abs(errors));
        }
    }
# If arguments are passed, use them.
    else{
        errors <- eval(parse(text=paste0(randomizer,"(n=",obs,",", toString(as.character(list(...))),")")));

        if(randomizer=="rbeta"){
# Center the errors around 0.5
          errors <- errors - 0.5;
# Make a meaningful variance of data. Something resembling to var=1.
          errors <- errors / sqrt(var(errors)) * sqrt(abs(matxt[1,1]));
# If the error is multiplicative, scale it!
            if(Etype=="M" & max(abs(errors))>0.05){
                errors <- 0.05 * errors / max(abs(errors));
            }
        }
        else if(randomizer=="rt"){
# Make a meaningful variance of data.
          errors <- errors * sqrt(abs(matxt[1,1]));
# If the error is multiplicative, scale it!
            if(Etype=="M" & max(abs(errors))>0.05){
                errors <- 0.05 * errors / max(abs(errors));
            }
        }

# Center errors in case all of them are positive or negative to get rid of systematic bias.
        if(all(errors>0) | all(errors<0)){
            errors <- errors - mean(errors);
        }
    }

# Generate ones for the possible intermittency
    ot <- rbinom(obs,1,iprob);

############## This part should be rewritten in Rcpp. The list should be returned...  ##############
###### Simulate the data #####
    j <- modelfreq + 1;
    if(Stype=="N"){
        if(Ttype!="M"){
### ZNN and ZAN
            while(j<=(obs+modelfreq)){
                y[j-modelfreq] <- matw %*% matxt[cbind((j-lags),c(1:persistence.length))] + errors[j-modelfreq] * ry.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                matxt[j,] <- matF %*% matxt[cbind((j-lags),c(1:persistence.length))] + vecg * errors[j-modelfreq] * r.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                j <- j + 1;
            }
        }
        else{
### ZMN
            while(j<=(obs+modelfreq)){
                y[j-modelfreq] <- exp(matw %*% log(matxt[cbind((j-lags),c(1:persistence.length))])) + errors[j-modelfreq] * ry.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                matxt[j,] <- exp(matF %*% log(matxt[cbind((j-lags),c(1:persistence.length))])) + vecg * errors[j-modelfreq] * r.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
#Failsafe for the negative components
                if(matxt[j,1] < 0){
                    matxt[j,1] <- matxt[j-1,1];
                }
                if(matxt[j,2] < 0){
                    matxt[j,2] <- matxt[j-1,2];
                }
                j <- j + 1;
            }
        }
    }
    else if(Stype=="A"){
        if(Ttype!="M"){
### ZNA and ZAA
            while(j<=(obs+modelfreq)){
                y[j-modelfreq] <- matw %*% matxt[cbind((j-lags),c(1:persistence.length))] + errors[j-modelfreq] * ry.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                matxt[j,] <- matF %*% matxt[cbind((j-lags),c(1:persistence.length))] + vecg * errors[j-modelfreq] * r.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
# Renormalize seasonal component
                at <- vecg[n.components+1] / frequency * errors[j-modelfreq] * ry.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                matxt[j,1] <- matxt[j,1] + at;
                matxt[(j-frequency+1):(j),n.components+1] <- matxt[(j-frequency+1):(j),n.components+1] - at;
                j <- j + 1;
            }
        }
        else{
### ZMA
            while(j<=(obs+modelfreq)){
                y[j-modelfreq] <- exp(matw[1:n.components] %*% log(matxt[cbind((j-lags[1:n.components]),c(1:n.components))])) + matxt[j-frequency,n.components+1] + errors[j-modelfreq] * ry.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                matxt[j,] <- Re(exp(matF %*% log(as.complex(matxt[cbind((j-lags),c(1:persistence.length))])))) + vecg * errors[j-modelfreq] * r.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
#Failsafe for the negative components
                if(matxt[j,1] < 0){
                    matxt[j,1] <- matxt[j-1,1];
                }
                if(matxt[j,2] < 0){
                    matxt[j,2] <- matxt[j-1,2];
                }
# Renormalize seasonal component
                at <- vecg[n.components+1] / frequency * errors[j-modelfreq] * ry.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                matxt[j,1] <- matxt[j,1] + at;
                matxt[(j-frequency+1):(j),n.components+1] <- matxt[(j-frequency+1):(j),n.components+1] - at;
                j <- j + 1;
            }
        }
    }
    else if(Stype=="M"){
        if(Ttype!="M"){
### ZNM and ZAM
            while(j<=(obs+modelfreq)){
                vec.r <- r.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                y[j-modelfreq] <- matw[1:n.components] %*% matxt[cbind((j-lags[1:n.components]),c(1:n.components))] * matxt[j-frequency,n.components+1] + errors[j-modelfreq] * ry.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                matxt[j,1:n.components] <- matF[1:n.components,1:n.components] %*% matxt[cbind((j-lags[1:n.components]),c(1:n.components))] + vecg[1:n.components] * errors[j-modelfreq] * vec.r[1:n.components];
                matxt[j,(n.components+1)] <- matxt[j-frequency,(n.components+1)] + vecg[n.components+1] * errors[j-modelfreq] * vec.r[n.components+1];
# Failsafe mechanism for the cases with negative multiplicative seasonals
                if(matxt[j,(n.components+1)] < 0){
                    matxt[j,(n.components+1)] <- matxt[j-frequency,(n.components+1)];
                }
# Renormalize seasonal component. It is done differently comparing with Hyndman et. al. 2008!
                matxt[(j-frequency+1):(j),n.components+1] <- matxt[(j-frequency+1):(j),n.components+1] / exp(mean(log(matxt[(j-frequency+1):(j),n.components+1])));
                j <- j + 1;
            }
        }
        else{
### ZMM
            while(j<=(obs+modelfreq)){
                y[j-modelfreq] <- exp(matw %*% log(matxt[cbind((j-lags),c(1:persistence.length))])) + errors[j-modelfreq] * ry.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
                matxt[j,] <- Re(exp(matF %*% log(as.complex(matxt[cbind((j-lags),c(1:persistence.length))])))) + vecg * errors[j-modelfreq] * r.value(Etype=Etype, Ttype=Ttype, Stype=Stype, xt=matxt[cbind((j-lags),c(1:persistence.length))]);
# Failsafe mechanism for the cases with negative components
                if(matxt[j,1] < 0){
                    matxt[j,1] <- matxt[j-1,1];
                }
                if(matxt[j,2] < 0){
                    matxt[j,2] <- matxt[j-1,2];
                }
                if(matxt[j,3] < 0){
                    matxt[j,3] <- matxt[j-frequency,3];
                }
# Renormalize seasonal component. It is done differently comparing with Hyndman et. al. 2008!
                matxt[(j-frequency+1):(j),3] <- matxt[(j-frequency+1):(j),3] / exp(mean(log(matxt[(j-frequency+1):(j),3])));
                j <- j + 1;
            }
        }
    }

    y <- ot * y;
    if(iprob!=1){
        y <- round(y,0);
    }

    likelihood <- -obs/2 *(log(2*pi*exp(1)) + log(mean(errors^2)));

    if(nseries > 1){
        mat.yt[,k] <- y;
        mat.ot[,k] <- ot;
        mat.errors[,k] <- errors;
        arr.xt[,,k] <- matxt;
        mat.g[k,] <- vecg;
        vec.likelihood[k] <- likelihood;

# Print the number of processed series
        if (silent == FALSE){
            cat(paste0(rep("\b",nchar(k-1)),collapse=""));
            cat(k);
        }
    }
}
##### End of loop #####

    if(nseries==1){
        y <- ts(y,frequency=frequency);
        errors <- ts(errors,frequency=frequency);
        matxt <- ts(matxt,frequency=frequency,start=c(0,frequency-modelfreq+1));
        return(list(model=model,data=y,states=matxt,persistence=vecg,residuals=errors,
                    intermittency=ot,likelihood=likelihood));
    }
    else{
        mat.yt <- ts(mat.yt,frequency=frequency);
        mat.errors <- ts(mat.errors,frequency=frequency);
        return(list(model=model,data=mat.yt,states=arr.xt,persistence=mat.g,residuals=mat.errors,
                    intermittency=mat.ot,likelihood=vec.likelihood));
    }
}
