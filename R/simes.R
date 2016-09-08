sim.es <- function(model="ANN",frequency=1, persistence=NULL, phi=1,
             initial=NULL, initialSeason=NULL,
             bounds=c("usual","admissible","restricted"),
             obs=10, nsim=1, silent=FALSE,
             randomizer=c("rnorm","rlnorm","runif","rbeta","rt"),
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
                warning(paste0("You have defined a strange model: ",model),call.=FALSE);
                sowhat(model);
            }
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        if(Ttype!="N" & phi==1){
            model <- paste0(Etype,Ttype,Stype);
            if(silent == FALSE){
                warning(paste0("Damping parameter is set to 1. Changing model to: ",model),call.=FALSE);
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
                warning(paste0("Damping parameter is set to ",phi,". Changing model to: ",model),call.=FALSE);
            }
        }
    }
    else{
        stop(paste0("You have defined a strange model: ",model,". Cannot proceed"),call.=FALSE);
    }

# In the case of wrong nsim, make it natural number. The same is for obs and frequency.
    nsim <- abs(round(nsim,0));
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
        modellags <- 1;
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
        modellags <- c(modellags,1);
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
# maxlag is used in the cases of seasonal models.
#   if maxlag==1 then non-seasonal data will be produced with the defined frequency.
        modellags <- c(modellags,frequency);
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
    }

# Make matrices
    modellags <- matrix(modellags,persistence.length,1);
    maxlag <- max(modellags);
    matw <- matrix(matw,1,persistence.length);

# Check the persistence vector length
    if(!is.null(persistence)){
        if(persistence.length != length(persistence)){
            if(silent == FALSE){
                warning(paste0("The length of persistence vector does not correspond to the chosen model!\n",
                               "Falling back to random number generator."),call.=FALSE);
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
                warning(paste0("The length of initial state vector does not correspond to the chosen model!\n",
                               "Falling back to random number generator."),call.=FALSE);
            }
            initial <- NULL;
        }
        else{
            if(Ttype=="M" & initial[2]<=0){
                if(silent == FALSE){
                    warning(paste0("Wrong initial value for multiplicative trend! It should be greater than zero!\n",
                                   "Falling back to random number generator."),call.=FALSE);
                }
                initial <- NULL;
            }
        }
    }

    if(!is.null(initialSeason)){
        if(maxlag!=length(initialSeason)){
            if(silent == FALSE){
                    warning(paste0("The length of seasonal initial states does not correspond to the chosen frequency!\n",
                                   "Falling back to random number generator."),call.=FALSE);
            }
            initialSeason <- NULL;
        }
    }

# If the chosen randomizer is not rnorm, rt and runif and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rlnorm","rt","runif")) & (any(names(match.call(expand.dots=FALSE)[-1]) == "...")==FALSE)){
        if(silent == FALSE){
            warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
        }
        randomizer = "rnorm";
    }

##### Let's make sum fun #####

    matg <- matrix(NA,persistence.length,nsim);
    arrvt <- array(NA,c(obs+maxlag,persistence.length,nsim),dimnames=list(NULL,component.names,NULL));
    materrors <- matrix(NA,obs,nsim);
    matyt <- matrix(NA,obs,nsim);
    matot <- matrix(NA,obs,nsim);

# If the persistence is NULL or was of the wrong length, generate the values
    if(is.null(persistence)){
### For the case of "usual" bounds make restrictions on the generated smoothing parameters so the ETS can be "averaging" model.
        if(bounds=="u"){
            matg[,] <- runif(persistence.length*nsim,0,1);
        }
### These restrictions are even touhger
        else if(bounds=="r"){
            matg[,] <- runif(persistence.length*nsim,0,0.3);
        }
### Fill in the other smoothing parameters
        if(bounds!="a"){
            if(Ttype!="N"){
                matg[2,] <- runif(nsim,0,matg[1,]);
            }
            if(Stype!="N"){
                matg[persistence.length,] <- runif(nsim,0,max(0,1-matg[1]));
            }
        }
### In case of admissible bounds, do some stuff
        else{
            matg[,] <- runif(persistence.length*nsim,1-1/phi,1+1/phi);
            if(Ttype!="N"){
                matg[2,] <- runif(nsim,matg[1,]*(phi-1),(2-matg[1,])*(1+phi));
                if(Stype!="N"){
                    Theta.func <- function(Theta){
                        result <- (phi*matg[1,i]+phi+1)/(matg[3,i]) +
                            ((phi-1)*(1+cos(Theta)-cos(maxlag*Theta)) +
                                 cos((maxlag-1)*Theta)-phi*cos((maxlag+1)*Theta))/(2*(1+cos(Theta))*(1-cos(maxlag*Theta)));
                        return(abs(result));
                    }

                    for(i in 1:nsim){
                        matg[3,i] <- runif(1,max(1-1/phi-matg[1,i],0),1+1/phi-matg[1,i]);

                        B <- phi*(4-3*matg[3,i])+matg[3,i]*(1-phi)/maxlag;
                        C <- sqrt(B^2-8*(phi^2*(1-matg[3,i])^2+2*(phi-1)*(1-matg[3,i])-1)+8*matg[3,i]^2*(1-phi)/maxlag);
                        matg[1,i] <- runif(1,1-1/phi-matg[3,i]*(1-maxlag+phi*(1+maxlag))/(2*phi*maxlag),(B+C)/(4*phi));
# Solve the equation to get Theta value. Theta

                        Theta <- 0.1;
                        Theta <- optim(Theta,Theta.func,method="Brent",lower=0,upper=1)$par;

                        D <- (phi*(1-matg[1,i])+1)*(1-cos(Theta)) - matg[3,i]*((1+phi)*(1-cos(Theta) - cos(maxlag*Theta)) +
                                                                                   cos((maxlag-1)*Theta)+phi*cos((maxlag+1)*Theta))/
                            (2*(1+cos(Theta))*(1-cos(maxlag*Theta)));
                        matg[2,i] <- runif(1,-(1-phi)*(matg[3,i]/maxlag+matg[1,i]),D+(phi-1)*matg[1,i]);
                    }
                }
            }
            else{
                if(Stype!="N"){
                    matg[1,] <- runif(nsim,-2/(maxlag-1),2);
                    for(i in 1:nsim){
                        matg[2,i] <- runif(1,max(-maxlag*matg[1,i],0),2-matg[1,i]);
                    }
                    matg[1,] <- runif(nsim,-2/(maxlag-1),2-matg[2,]);
                }
            }
        }
    }
    else{
        matg[,] <- rep(persistence,nsim);
    }

# Generate initial states of level and trend if they were not supplied
    if(is.null(initial)){
        if(Ttype=="N"){
            arrvt[1:maxlag,1,] <- runif(nsim,0,1000);
        }
        else if(Ttype=="A"){
            arrvt[1:maxlag,1,] <- runif(nsim,0,5000);
            arrvt[1:maxlag,2,] <- runif(nsim,-100,100);
        }
        else{
            arrvt[1:maxlag,1,] <- runif(nsim,500,5000);
            arrvt[1:maxlag,2,] <- 1;
        }
    }
    else{
        arrvt[,1:n.components,] <- rep(rep(initial,each=(obs+maxlag)),nsim);
    }

# Generate seasonal states if they were not supplied
    if(seasonal.component==TRUE & is.null(initialSeason)){
# Create and normalize seasonal components. Use geometric mean for multiplicative case
        if(Stype == "A"){
            arrvt[1:maxlag,n.components+1,] <- runif(nsim*maxlag,-500,500);
            for(i in 1:nsim){
                arrvt[1:maxlag,n.components+1,i] <- arrvt[1:maxlag,n.components+1,i] - mean(arrvt[1:maxlag,n.components+1,i]);
            }
        }
        else{
            arrvt[1:maxlag,n.components+1,] <- runif(nsim*maxlag,0.3,1.7);
            for(i in 1:nsim){
                arrvt[1:maxlag,n.components+1,i] <- arrvt[1:maxlag,n.components+1,i] / exp(mean(log(arrvt[1:maxlag,n.components+1,i])));
            }
        }
    }
# If the seasonal model is chosen, fill in the first "frequency" values of seasonal component.
    else if(seasonal.component==TRUE & !is.null(initialSeason)){
        arrvt[1:maxlag,n.components+1,] <- rep(initialSeason,nsim);
    }

# Check if any argument was passed in dots
    if(any(names(match.call(expand.dots=FALSE)[-1]) == "...")==FALSE){
# Create vector of the errors
        if(any(randomizer==c("rnorm","runif"))){
            materrors[,] <- eval(parse(text=paste0(randomizer,"(n=",nsim*obs,")")));
        }
        else if(randomizer=="rlnorm"){
            materrors[,] <- rlnorm(n=nsim*obs,0,0.01+(1-iprob));
            materrors <- materrors - 1;
        }
        else if(randomizer=="rt"){
# The degrees of freedom are df = n - k.
            materrors[,] <- rt(nsim*obs,obs-(persistence.length + maxlag));
        }

        if(randomizer!="rlnorm"){
# Center errors just in case
            materrors <- materrors - colMeans(materrors);
# If the error is multiplicative, scale it!
            if(Etype=="M"){
# Errors will be lognormal, decrease variance, so it behaves better
                if(iprob!=1){
                    materrors <- materrors * 0.5;
                }
                else{
                    materrors <- materrors * 0.01;
                }
                materrors <- exp(materrors) - 1;
#            exceedingerrors <- apply(abs(materrors),2,max)>1;
#            materrors[,exceedingerrors] <- 0.95 * materrors[,exceedingerrors] / apply(abs(matrix(materrors[,exceedingerrors],obs)),2,max);
            }
            else if(Etype=="A"){
# Change variance to make some sense. Errors should not be rediculously high and not too low.
                materrors <- materrors * sqrt(abs(arrvt[1,1,]));
            }
        }
    }
# If arguments are passed, use them. WE ASSUME HERE THAT USER KNOWS WHAT HE'S DOING!
    else{
        materrors[,] <- eval(parse(text=paste0(randomizer,"(n=",nsim*obs,",", toString(as.character(list(...))),")")));
        if(randomizer=="rbeta"){
# Center the errors around 0
            materrors <- materrors - 0.5;
# Make a meaningful variance of data. Something resembling to var=1.
            materrors <- materrors / rep(sqrt(colMeans(materrors^2)) * sqrt(abs(arrvt[1,1,])),each=obs);
        }
        else if(randomizer=="rt"){
# Make a meaningful variance of data.
            materrors <- materrors * rep(sqrt(abs(arrvt[1,1,])),each=obs);
        }
        else if(randomizer=="rlnorm"){
            materrors <- materrors - 1;
        }
    }

    veclikelihood <- -obs/2 *(log(2*pi*exp(1)) + log(colMeans(materrors^2)));

# Generate ones for the possible intermittency
    if((iprob < 1) & (iprob > 0)){
        matot[,] <- rbinom(obs*nsim,1,iprob);
    }
    else{
        matot[,] <- 1;
    }

    simulateddata <- simulateETSwrap(arrvt,materrors,matot,matF,matw,matg,Etype,Ttype,Stype,modellags);

    if((iprob < 1) & (iprob > 0)){
        matyt <- round(simulateddata$matyt,0);
    }
    else{
        matyt <- simulateddata$matyt;
    }
    arrvt <- simulateddata$arrvt;
    dimnames(arrvt) <- list(NULL,component.names,NULL);

    if(nsim==1){
        matyt <- ts(matyt[,1],frequency=frequency);
        materrors <- ts(materrors[,1],frequency=frequency);
        arrvt <- ts(arrvt[,,1],frequency=frequency,start=c(0,frequency-maxlag+1));
        matot <- ts(matot[,1],frequency=frequency);
    }
    else{
        matyt <- ts(matyt,frequency=frequency);
        materrors <- ts(materrors,frequency=frequency);
        matot <- ts(matot,frequency=frequency);
    }
    return(list(model=model,data=matyt,states=arrvt,persistence=matg,residuals=materrors,
                occurrences=matot,likelihood=veclikelihood));
}
