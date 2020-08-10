#' Simulate Exponential Smoothing
#'
#' Function generates data using ETS with Single Source of Error as a data
#' generating process.
#'
#' For the information about the function, see the vignette:
#' \code{vignette("simulate","smooth")}
#'
#' @template ssSimParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#'
#' @param model Type of ETS model according to [Hyndman et. al., 2008]
#' taxonomy. Can consist of 3 or 4 chars: \code{ANN}, \code{AAN}, \code{AAdN},
#' \code{AAA}, \code{AAdA}, \code{MAdM} etc.
#' @param persistence Persistence vector, which includes all the smoothing
#' parameters. Must correspond to the chosen model. The maximum length is 3:
#' level, trend and seasonal smoothing parameters. If \code{NULL}, values are
#' generated.
#' @param phi Value of damping parameter. If trend is not chosen in the model,
#' the parameter is ignored.
#' @param initial Vector of initial states of level and trend. The maximum
#' length is 2. If \code{NULL}, values are generated.
#' @param initialSeason Vector of initial states for seasonal coefficients.
#' Should have length equal to \code{frequency} parameter. If \code{NULL},
#' values are generated.
#' @param bounds Type of bounds to use for persistence vector if values are
#' generated. \code{"usual"} - bounds from p.156 by Hyndman et. al., 2008.
#' \code{"restricted"} - similar to \code{"usual"} but with upper bound equal
#' to 0.3. \code{"admissible"} - bounds from tables 10.1 and 10.2 of Hyndman
#' et. al., 2008. Using first letter of the type of bounds also works. These
#' bounds are also used for multiplicative models, but the models are much
#' more restrictive, so weird results might be obtained. Be careful!
#' @param ...  Additional parameters passed to the chosen randomizer. All the
#' parameters should be passed in the order they are used in chosen randomizer.
#' For example, passing just \code{sd=0.5} to \code{rnorm} function will lead
#' to the call \code{rnorm(obs, mean=0.5, sd=1)}.  ATTENTION! When generating
#' the multiplicative errors some tuning might be needed to obtain meaningful
#' data. \code{sd=0.1} is usually already a high value for such models. ALSO
#' NOTE: In case of multiplicative error model, the randomizer will generate
#' \code{1+e_t} error, not \code{e_t}. This means that the mean should
#' typically be equal to 1, not zero.
#'
#' @return List of the following values is returned:
#' \itemize{
#' \item \code{model} - Name of ETS model.
#' \item \code{data} - Time series vector (or matrix if \code{nsim>1}) of the generated
#' series.
#' \item \code{states} - Matrix (or array if \code{nsim>1}) of states. States are in
#' columns, time is in rows.
#' \item \code{persistence} - Vector (or matrix if \code{nsim>1}) of smoothing
#' parameters used in the simulation.
#' \item \code{phi} - Value of damping parameter used in time series generation.
#' \item \code{initial} - Vector (or matrix) of initial values.
#' \item \code{initialSeason} - Vector (or matrix) of initial seasonal coefficients.
#' \item \code{probability} - vector of probabilities used in the simulation.
#' \item \code{intermittent} - type of the intermittent model used.
#' \item \code{residuals} - Error terms used in the simulation. Either vector or matrix,
#' depending on \code{nsim}.
#' \item \code{occurrence} - Values of occurrence variable. Once again, can be either
#' a vector or a matrix...
#' \item \code{logLik} - Log-likelihood of the constructed model.
#' }
#'
#' @seealso \code{\link[smooth]{es}, \link[forecast]{ets},
#' \link[forecast]{forecast}, \link[stats]{ts}, \link[stats]{Distributions}}
#'
#' @examples
#'
#' # Create 40 observations of quarterly data using AAA model with errors from normal distribution
#' ETSAAA <- sim.es(model="AAA",frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100)
#'
#' # Create 50 series of quarterly data using AAA model
#' # with 40 observations each with errors from normal distribution
#' ETSAAA <- sim.es(model="AAA",frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100,nsim=50)
#'
#' # Create 50 series of quarterly data using AAdA model
#' # with 40 observations each with errors from normal distribution
#' # and smoothing parameters lying in the "admissible" range.
#' ETSAAA <- sim.es(model="AAA",phi=0.9,frequency=4,obs=40,bounds="admissible",
#'                   randomizer="rnorm",mean=0,sd=100,nsim=50)
#'
#' # Create 60 observations of monthly data using ANN model
#' # with errors from beta distribution
#' ETSANN <- sim.es(model="ANN",persistence=c(1.5),frequency=12,obs=60,
#'                   randomizer="rbeta",shape1=1.5,shape2=1.5)
#' plot(ETSANN$states)
#'
#' # Create 60 observations of monthly data using MAM model
#' # with errors from uniform distribution
#' ETSMAM <- sim.es(model="MAM",persistence=c(0.3,0.2,0.1),initial=c(2000,50),
#'            phi=0.8,frequency=12,obs=60,randomizer="runif",min=-0.5,max=0.5)
#'
#' # Create 80 observations of quarterly data using MMM model
#' # with predefined initial values and errors from the normal distribution
#' ETSMMM <- sim.es(model="MMM",persistence=c(0.1,0.1,0.1),initial=c(2000,1),
#'            initialSeason=c(1.1,1.05,0.9,.95),frequency=4,obs=80,mean=0,sd=0.01)
#'
#' # Generate intermittent data using AAdN
#' iETSAAdN <- sim.es("AAdN",obs=30,frequency=1,probability=0.1,initial=c(3,0),phi=0.8)
#'
#' # Generate iETS(MNN) with TSB style probabilities
#' oETSMNN <- sim.oes("MNN",obs=50,occurrence="d",persistence=0.2,initial=1,
#'                    randomizer="rlnorm",meanlog=0,sdlog=0.3)
#' iETSMNN <- sim.es("MNN",obs=50,frequency=12,persistence=0.2,initial=4,
#'                   probability=oETSMNN$probability)
#'
#' @importFrom stats optim
#' @importFrom greybox rlaplace rs
#' @export sim.es
sim.es <- function(model="ANN", obs=10, nsim=1,
                   frequency=1, persistence=NULL, phi=1,
                   initial=NULL, initialSeason=NULL,
                   bounds=c("usual","admissible","restricted"),
                   randomizer=c("rnorm","rlnorm","rt","rlaplace","rs"),
                   probability=1, ...){
# Function generates data using ETS with Single Source of Error as a data generating process.
#    Copyright (C) 2015 - Inf Ivan Svetunkov

    randomizer <- randomizer[1];
    ellipsis <- list(...);
    bounds <- bounds[1];
    # If R decided that by "b" we meant "bounds", fix this!
    if(is.numeric(bounds)){
        ellipsis$b <- bounds;
        bounds <- "u";
    }

    if(all(bounds!=c("u","a","r","usual","admissible","restricted"))){
        warning(paste0("Strange type of bounds provided: ",bounds,". Switching to 'usual'."),
                call.=FALSE);
        bounds <- "u";
    }

    bounds <- substring(bounds[1],1,1);

# If chosen model is "AAdN" or anything like that, we are taking the appropriate values
    if(nchar(model)==4){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,4,4);
        if(substring(model,3,3)!="d"){
            warning(paste0("You have defined a strange model: ",model),call.=FALSE);
            # if(!silent){
            #     sowhat(model);
            # }
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        if(Ttype!="N" & phi==1){
            model <- paste0(Etype,Ttype,Stype);
            warning(paste0("Damping parameter is set to 1. Changing model to: ",model),call.=FALSE);
        }
    }
    else if(nchar(model)==3){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,3,3);
        if(phi!=1 & Ttype!="N"){
            model <- paste0(Etype,Ttype,"d",Stype);
            warning(paste0("Damping parameter is set to ",phi,". Changing model to: ",model),call.=FALSE);
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
        warning(paste0("Damping parameter should lie in (0, 2) region! You have chosen phi=",phi,
                       ". Be careful!"),call.=FALSE);
    }

# Check the used model and estimate the length of needed persistence vector.
    if(Etype!="A" & Etype!="M"){
        stop("Wrong error type! Should be 'A' or 'M'.",call.=FALSE);
    }
    else{
# The number of the smoothing parameters needed
        persistenceLength <- 1;
# The number initial values of the state vector
        componentsNumber <- 1;
# The lag of components (needed for the seasonal models)
        lagsModel <- 1;
# The names of the state vector components
        componentsNames <- "level";
        matw <- 1;
# The transition matrix
        matF <- matrix(1,1,1);
    }

# Check the trend type of the model
    if(Ttype!="N" & Ttype!="A" & Ttype!="M"){
        stop("Wrong trend type! Should be 'N', 'A' or 'M'.",call.=FALSE);
    }
    else if(Ttype!="N"){
        if(is.na(phi) | is.null(phi)){
            phi <- 1;
        }
        persistenceLength <- persistenceLength + 1;
        componentsNumber <- componentsNumber + 1;
        lagsModel <- c(lagsModel,1);
        componentsNames <- c(componentsNames,"trend");
        matw <- c(matw,phi);
        matF <- matrix(c(1,0,phi,phi),2,2);
        componentTrend=TRUE;
        if(phi!=1){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
    }
    else{
        componentTrend=FALSE;
    }

# Check the seasonaity type of the model
    if(Stype!="N" & Stype!="A" & Stype!="M"){
        stop("Wrong seasonality type! Should be 'N', 'A' or 'M'.",call.=FALSE);
    }

    if(Stype!="N" & frequency==1){
        stop("Cannot create the seasonal model with the data frequency 1!",call.=FALSE);
    }

    if(Stype!="N"){
        persistenceLength <- persistenceLength + 1;
# lagsModelMax is used in the cases of seasonal models.
#   if lagsModelMax==1 then non-seasonal data will be produced with the defined frequency.
        lagsModel <- c(lagsModel,frequency);
        componentsNames <- c(componentsNames,"seasonality");
        matw <- c(matw,1);
        componentSeasonal <- TRUE;

        if(!componentTrend){
            matF <- matrix(c(1,0,0,1),2,2);
        }
        else{
            matF <- matrix(c(1,0,0,phi,phi,0,0,0,1),3,3);
        }
    }
    else{
        componentSeasonal <- FALSE;
    }

# Make matrices
    lagsModel <- matrix(lagsModel,persistenceLength,1);
    lagsModelMax <- max(lagsModel);
    matw <- matrix(matw,1,persistenceLength);
    arrF <- array(matF,c(dim(matF),nsim));

# Check the persistence vector length
    if(!is.null(persistence)){
        if(persistenceLength != length(persistence)){
            if(length(persistence)!=1){
                warning(paste0("The length of persistence vector does not correspond to the chosen model!\n",
                               "Falling back to random number generator."),call.=FALSE);
                persistence <- NULL;
            }
            else{
                persistence <- rep(persistence,persistenceLength);
            }
        }
    }

# Check the inital vector length
    if(!is.null(initial)){
        if(length(initial)>2){
            stop("The length of the initial value is wrong! It should not be greater than 2.",call.=FALSE);
        }
        if(componentsNumber!=length(initial)){
            warning(paste0("The length of initial state vector does not correspond to the chosen model!\n",
                           "Falling back to random number generator."),call.=FALSE);
            initial <- NULL;
        }
        else{
            if(Ttype=="M" & initial[2]<=0){
                warning(paste0("Wrong initial value for multiplicative trend! It should be greater than zero!\n",
                               "Falling back to random number generator."),call.=FALSE);
                initial <- NULL;
            }
        }
    }

# Check the inital seasonal vector length
    if(!is.null(initialSeason)){
        if(lagsModelMax!=length(initialSeason)){
            warning(paste0("The length of seasonal initial states does not correspond to the chosen frequency!\n",
                           "Falling back to random number generator."),call.=FALSE);
            initialSeason <- NULL;
        }
    }

    # If the chosen randomizer is not default and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rt","rlaplace","rs","rlnorm")) & (length(ellipsis)==0)){
        warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
        randomizer = "rnorm";
    }

# Check the vector of probabilities
    if(is.vector(probability)){
        if(any(probability!=probability[1])){
            if(length(probability)!=obs){
                warning("Length of probability does not correspond to number of observations.",call.=FALSE);
                if(length(probability)>obs){
                    warning("We will cut off the excessive ones.",call.=FALSE);
                    probability <- probability[1:obs];
                }
                else{
                    warning("We will duplicate the last one.",call.=FALSE);
                    probability <- c(probability,rep(probability[length(probability)],obs-length(probability)));
                }
            }
        }
        else{
            probability <- probability[1];
        }
    }

# Check the probabilities and try to assign the type of intermittent model
    if(length(probability)==1){
        intermittent <- "fixed";
    }
    else{
        # This is a strong assumption!
        intermittent <- "tsb";
    }

    if(all(probability==1)){
        intermittent <- "none";
    }

##### Let's make sum fun #####
    matg <- matrix(NA,persistenceLength,nsim);
    arrvt <- array(NA,c(obs+lagsModelMax,persistenceLength,nsim),dimnames=list(NULL,componentsNames,NULL));
    materrors <- matrix(NA,obs,nsim);
    matyt <- matrix(NA,obs,nsim);
    matot <- matrix(NA,obs,nsim);

# If the persistence is NULL or was of the wrong length, generate the values
    if(is.null(persistence)){
### For the case of "usual" bounds make restrictions on the generated smoothing parameters so the ETS can be "averaging" model.

### First generate the first smoothing parameter.
        if(bounds=="u"){
            matg[1,] <- runif(nsim,0,1);
        }
### These restrictions are even touhger
        else if(bounds=="r"){
            matg[1,] <- runif(nsim,0,0.3);
        }

### Fill in the other smoothing parameters
        if(bounds!="a"){
            if(Ttype!="N"){
                matg[2,] <- runif(nsim,0,matg[1,]);
            }
            if(Stype!="N"){
                matg[persistenceLength,] <- runif(nsim,0,max(0,1-matg[1]));
            }
        }
### In case of admissible bounds, do some stuff
        else{
            matg[,] <- runif(persistenceLength*nsim,1-1/phi,1+1/phi);
            if(Ttype!="N"){
                matg[2,] <- runif(nsim,matg[1,]*(phi-1),(2-matg[1,])*(1+phi));
                if(Stype!="N"){
                    Theta.func <- function(Theta){
                        result <- (phi*matg[1,i]+phi+1)/(matg[3,i]) +
                            ((phi-1)*(1+cos(Theta)-cos(lagsModelMax*Theta)) +
                                 cos((lagsModelMax-1)*Theta)-phi*cos((lagsModelMax+1)*Theta))/(2*(1+cos(Theta))*(1-cos(lagsModelMax*Theta)));
                        return(abs(result));
                    }

                    for(i in 1:nsim){
                        matg[3,i] <- runif(1,max(1-1/phi-matg[1,i],0),1+1/phi-matg[1,i]);

                        B <- phi*(4-3*matg[3,i])+matg[3,i]*(1-phi)/lagsModelMax;
                        C <- sqrt(B^2-8*(phi^2*(1-matg[3,i])^2+2*(phi-1)*(1-matg[3,i])-1)+8*matg[3,i]^2*(1-phi)/lagsModelMax);
                        matg[1,i] <- runif(1,1-1/phi-matg[3,i]*(1-lagsModelMax+phi*(1+lagsModelMax))/(2*phi*lagsModelMax),(B+C)/(4*phi));
# Solve the equation to get Theta value. Theta

                        Theta <- 0.1;
                        Theta <- optim(Theta,Theta.func,method="Brent",lower=0,upper=1)$par;

                        D <- (phi*(1-matg[1,i])+1)*(1-cos(Theta)) - matg[3,i]*((1+phi)*(1-cos(Theta) - cos(lagsModelMax*Theta)) +
                                                                                   cos((lagsModelMax-1)*Theta)+phi*cos((lagsModelMax+1)*Theta))/
                            (2*(1+cos(Theta))*(1-cos(lagsModelMax*Theta)));
                        matg[2,i] <- runif(1,-(1-phi)*(matg[3,i]/lagsModelMax+matg[1,i]),D+(phi-1)*matg[1,i]);
                    }
                }
            }
            else{
                if(Stype!="N"){
                    matg[1,] <- runif(nsim,-2/(lagsModelMax-1),2);
                    for(i in 1:nsim){
                        matg[2,i] <- runif(1,max(-lagsModelMax*matg[1,i],0),2-matg[1,i]);
                    }
                    matg[1,] <- runif(nsim,-2/(lagsModelMax-1),2-matg[2,]);
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
            arrvt[1:lagsModelMax,1,] <- runif(nsim,0,1000);
        }
        else if(Ttype=="A"){
            arrvt[1:lagsModelMax,1,] <- runif(nsim,0,5000);
            arrvt[1:lagsModelMax,2,] <- runif(nsim,-100,100);
        }
        else{
            arrvt[1:lagsModelMax,1,] <- runif(nsim,500,5000);
            arrvt[1:lagsModelMax,2,] <- 1;
        }
        initial <- matrix(arrvt[1,1:componentsNumber,],ncol=nsim);
    }
    else{
        arrvt[,1:componentsNumber,] <- rep(rep(initial,each=(obs+lagsModelMax)),nsim);
        initial <- matrix(arrvt[1,1:componentsNumber,],ncol=nsim);
    }

# Generate seasonal states if they were not supplied
    if(componentSeasonal & is.null(initialSeason)){
# Create and normalize seasonal components. Use geometric mean for multiplicative case
        if(Stype == "A"){
            arrvt[1:lagsModelMax,componentsNumber+1,] <- runif(nsim*lagsModelMax,-500,500);
            for(i in 1:nsim){
                arrvt[1:lagsModelMax,componentsNumber+1,i] <- arrvt[1:lagsModelMax,componentsNumber+1,i] - mean(arrvt[1:lagsModelMax,componentsNumber+1,i]);
            }
        }
        else{
            arrvt[1:lagsModelMax,componentsNumber+1,] <- runif(nsim*lagsModelMax,0.3,1.7);
            for(i in 1:nsim){
                arrvt[1:lagsModelMax,componentsNumber+1,i] <- arrvt[1:lagsModelMax,componentsNumber+1,i] / exp(mean(log(arrvt[1:lagsModelMax,componentsNumber+1,i])));
            }
        }
        initialSeason <- matrix(arrvt[1:lagsModelMax,componentsNumber+1,],ncol=nsim);
    }
# If the seasonal model is chosen, fill in the first "frequency" values of seasonal component.
    else if(componentSeasonal & !is.null(initialSeason)){
        arrvt[1:lagsModelMax,componentsNumber+1,] <- rep(initialSeason,nsim);
        initialSeason <- matrix(arrvt[1:lagsModelMax,componentsNumber+1,],ncol=nsim);
    }

# Check if any argument was passed in dots
    if(length(ellipsis)==0){
        ellipsis$n <- nsim*obs;
# Create vector of the errors
        if(any(randomizer==c("rnorm","rlaplace","rs"))){
            materrors[,] <- do.call(randomizer,ellipsis);
        }
        else if(randomizer=="rt"){
            # The degrees of freedom are df = n - k.
            materrors[,] <- rt(nsim*obs,obs-(persistenceLength + lagsModelMax));
        }
        else if(randomizer=="rlnorm"){
            materrors[,] <- rlnorm(n=nsim*obs,0,0.01+(1-probability));
            materrors <- materrors - 1;
        }

        if(randomizer!="rlnorm"){
            # If the error is multiplicative, scale it!
            if(Etype=="M"){
                # Errors will be lognormal, decrease variance, so it behaves better
                if(any(probability!=1)){
                    materrors <- materrors * 0.5;
                }
                else{
                    materrors <- materrors * 0.1;
                }
                materrors <- exp(materrors) - 1;
#            exceedingerrors <- apply(abs(materrors),2,max)>1;
#            materrors[,exceedingerrors] <- 0.95 * materrors[,exceedingerrors] / apply(abs(matrix(materrors[,exceedingerrors],obs)),2,max);
            }
            else if(Etype=="A"){
# Change variance to make some sense. Errors should not be rediculously high and not too low.
                if(all(arrvt[1,1,]!=0)){
                    materrors <- materrors * sqrt(abs(arrvt[1,1,]));
                }

                if(randomizer=="rs"){
                    materrors <- materrors / 4;
                }
            }
        }
    }
# If arguments are passed, use them. WE ASSUME HERE THAT USER KNOWS WHAT HE'S DOING!
    else{
        ellipsis$n <- nsim*obs;
        materrors[,] <- do.call(randomizer,ellipsis);
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

        # Substitute 1 to get epsilon_t
        if(Etype=="M"){
            materrors <- materrors - 1;
        }
    }

# Generate ones for the possible intermittency
    if(all(probability == 1)){
        matot[,] <- 1;
    }
    else{
        matot[,] <- rbinom(obs*nsim,1,probability);
    }

#### Simulate the data ####
    simulateddata <- simulatorwrap(arrvt,materrors,matot,arrF,matw,matg,Etype,Ttype,Stype,lagsModel);

    # if(all(probability == 1)){
        matyt <- simulateddata$matyt;
    # }
    # else{
        # matyt <- round(simulateddata$matyt,0);
    # }
    arrvt <- simulateddata$arrvt;
    dimnames(arrvt) <- list(NULL,componentsNames,NULL);

    if(any(randomizer==c("rnorm","rt"))){
        veclikelihood <- -obs/2 *(log(2*pi*exp(1)) + log(colMeans(materrors^2)));
    }
    else if(randomizer=="rlaplace"){
        veclikelihood <- -obs*(log(2*exp(1)) + log(colMeans(abs(materrors))));
    }
    else if(randomizer=="rs"){
        veclikelihood <- -2*obs*(log(2*exp(1)) + log(0.5*colMeans(sqrt(abs(materrors)))));
    }
    else if(randomizer=="rlnorm"){
        veclikelihood <- -obs/2 *(log(2*pi*exp(1)) + log(colMeans(materrors^2))) - colSums(log(matyt));
    }
    else if(randomizer=="rinvgauss"){
        veclikelihood <- -0.5*(obs*(log(colMeans(materrors^2/(1+materrors))/(2*pi))-1) +
                                   sum(log(matyt/(1+materrors))) - 3*sum(log(matyt)));
    }
    # If this is something unknown, forget about it
    else{
        veclikelihood <- NA;
    }

    if(nsim==1){
        matyt <- ts(matyt[,1],frequency=frequency);
        materrors <- ts(materrors[,1],frequency=frequency);
        arrvt <- ts(arrvt[,,1],frequency=frequency,start=c(0,frequency-lagsModelMax+1));
        matot <- ts(matot[,1],frequency=frequency);
    }
    else{
        matyt <- ts(matyt,frequency=frequency);
        materrors <- ts(materrors,frequency=frequency);
        matot <- ts(matot,frequency=frequency);
    }

    if(Ttype!="N"){
        rownames(matg) <- c("alpha","beta","gamma")[1:persistenceLength];
    }
    else{
        rownames(matg) <- c("alpha","gamma")[1:persistenceLength];
    }

    model <- paste0("ETS(",model,")");
    if(any(probability!=1)){
        model <- paste0("i",model);
    }

    if(any(is.nan(matyt))){
        warning("NaN values were produced by the simulator.",call.=FALSE);
    }

    model <- list(model=model, data=matyt, states=arrvt, persistence=matg, phi=phi,
                  initial=initial, initialSeason=initialSeason, probability=probability, intermittent=intermittent,
                  residuals=materrors, occurrence=matot, logLik=veclikelihood, other=ellipsis);
    return(structure(model,class="smooth.sim"));
}
