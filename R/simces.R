#' Simulate Complex Exponential Smoothing
#'
#' Function generates data using CES with Single Source of Error as a data
#' generating process.
#'
#'
#' @template ssSimParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssCESRef
#'
#' @param seasonality The type of seasonality used in CES. Can be: \code{none}
#' - No seasonality; \code{simple} - Simple seasonality, using lagged CES
#' (based on \code{t-m} observation, where \code{m} is the seasonality lag);
#' \code{partial} - Partial seasonality with real seasonal components
#' (equivalent to additive seasonality); \code{full} - Full seasonality with
#' complex seasonal components (can do both multiplicative and additive
#' seasonality, depending on the data). First letter can be used instead of
#' full words.  Any seasonal CES can only be constructed for time series
#' vectors.
#' @param A First complex smoothing parameter. Should be a complex number.
#'
#' NOTE! CES is very sensitive to A and B values so it is advised to use values
#' from previously estimated model.
#' @param B Second complex smoothing parameter. Can be real if
#' \code{seasonality="partial"}. In case of \code{seasonality="full"} must be
#' complex number.
#' @param initial A matrix with initial values for CES. In case with
#' \code{seasonality="partial"} and \code{seasonality="full"} first two columns
#' should contain initial values for non-seasonal components, repeated
#' \code{frequency} times.
#' @param ...  Additional parameters passed to the chosen randomizer. All the
#' parameters should be passed in the order they are used in chosen randomizer.
#' For example, passing just \code{sd=0.5} to \code{rnorm} function will lead
#' to the call \code{rnorm(obs, mean=0.5, sd=1)}.
#'
#' @return List of the following values is returned:
#' \itemize{
#' \item \code{model} - Name of CES model.
#' \item \code{A} - Value of complex smoothing parameter A. If \code{nsim>1}, then
#' this is a vector.
#' \item \code{B} - Value of complex smoothing parameter B. If \code{seasonality="n"}
#' or \code{seasonality="s"}, then this is equal to NULL. If \code{nsim>1},
#' then this is a vector.
#' \item \code{initial} - Initial values of CES in a form of matrix. If \code{nsim>1},
#' then this is an array.
#' \item \code{data} - Time series vector (or matrix if \code{nsim>1}) of the generated
#' series.
#' \item \code{states} - Matrix (or array if \code{nsim>1}) of states. States are in
#' columns, time is in rows.
#' \item \code{residuals} - Error terms used in the simulation. Either vector or matrix,
#' depending on \code{nsim}.
#' \item \code{occurrences} - Values of occurrence variable. Once again, can be either
#' a vector or a matrix...
#' \item \code{logLik} - Log-likelihood of the constructed model.
#' }
#'
#' @seealso \code{\link[smooth]{sim.es}, \link[smooth]{sim.ssarima},
#' \link[smooth]{ces}, \link[stats]{Distributions}}
#'
#' @examples
#'
#' # Create 120 observations from CES(n). Generate 100 time series of this kind.
#' x <- sim.ces("n",obs=120,nsim=100)
#'
#' # Generate similar thing for seasonal series of CES(s)_4
#' x <- sim.ces("s",frequency=4,obs=80,nsim=100)
#'
#' # Estimate model and then generate 10 time series from it
#' ourModel <- ces(rnorm(100,100,5))
#' simulate(ourModel,nsim=10)
#'
#' @export sim.ces
sim.ces <- function(seasonality=c("none","simple","partial","full"),
                    obs=10, nsim=1,
                    frequency=1, A=NULL, B=NULL,
                    initial=NULL,
                    randomizer=c("rnorm","runif","rbeta","rt"),
                    iprob=1, ...){
# Function simulates the data using CES state space framework
#
# seasonality - the type of seasonality to produce.
# frequency - the frequency of the data. In the case of seasonal models must be > 1.
# A, B - complex smoothing parameters.
# initial - the vector of initial states,
#    If NULL it will be generated.
# obs - the number of observations in each time series.
# nsim - the number of series needed to be generated.
# randomizer - the type of the random number generator function
# ... - the parameters passed to the randomizer.

    randomizer <- randomizer[1];

    args <- list(...);

    AGenerator <- function(nsim=nsim){
        AValue <- matrix(NA,2,nsim);
        ANonStable <- rep(TRUE,nsim);
        for(i in 1:nsim){
            while(ANonStable[i]){
                AValue[1,i] <- runif(1,0.9,2.5);
                AValue[2,i] <- runif(1,0.9,1.1);

                if(((AValue[1,i]-2.5)^2 + AValue[2,i]^2 > 1.25) &
                   ((AValue[1,i]-0.5)^2 + (AValue[2,i]-1)^2 > 0.25) &
                   (AValue[1,i]-1.5)^2 + (AValue[2,i]-0.5)^2 < 1.5){
                    ANonStable[i] <- FALSE;
                }
            }
        }
        return(AValue);
    }

#### Check values and preset parameters ####
# If the user typed wrong seasonality, use "none" instead
    if(all(seasonality!=c("n","s","p","f","none","simple","partial","full"))){
        warning(paste0("Wrong seasonality type: '",seasonality, "'. Changing to 'none'"), call.=FALSE);
        seasonality <- "n";
    }
    seasonality <- substring(seasonality[1],1,1);

    if(seasonality!="n" & frequency==1){
        stop("Can't simulate seasonal data with frequency=1!",call.=FALSE)
    }

    A <- list(value=A);
    B <- list(value=B);

    if(is.null(A$value)){
        A$generate <- TRUE;
    }
    else{
        A$generate <- FALSE;
        if(!(((Re(A$value)-2.5)^2 + Im(A$value)^2 > 1.25) &
                   ((Re(A$value)-0.5)^2 + (Im(A$value)-1)^2 > 0.25) &
                   (Re(A$value)-1.5)^2 + (Im(A$value)-0.5)^2 < 1.5)){
            warning("The provided complex smoothing parameter A leads to non-stable model!",call.=FALSE);
        }
    }

    if(all(is.null(B$value),any(seasonality==c("p","f")))){
        B$generate <- TRUE;
    }
    else{
        B$generate <- FALSE;
        if(seasonality=="f"){
            if(!(((Re(B$value)-2.5)^2 + Im(B$value)^2 > 1.25) &
                 ((Re(B$value)-0.5)^2 + (Im(B$value)-1)^2 > 0.25) &
                 (Re(B$value)-1.5)^2 + (Im(B$value)-0.5)^2 < 1.5)){
                warning("The provided complex smoothing parameter B leads to non-stable model!",call.=FALSE);
            }
        }
        else if(seasonality=="p"){
            if((B$value<0) | (B$value>1)){
                warning("Be careful with the provided B parameter - the model can be unstable.",call.=FALSE);
            }
        }
    }

    A$number <- 2;
# Define lags, number of components and number of parameters
    if(seasonality=="n"){
        # No seasonality
        maxlag <- 1;
        modellags <- c(1,1);
        # Define the number of all the parameters (smoothing parameters + initial states). Used in AIC mainly!
        componentsNumber <- 2;
        B$number <- 0;
        componentsNames <- c("level","potential");
        matw <- matrix(c(1,0),1,2);
    }
    else if(seasonality=="s"){
        # Simple seasonality, lagged CES
        maxlag <- frequency;
        modellags <- c(maxlag,maxlag);
        componentsNumber <- 2;
        B$number <- 0;
        componentsNames <- c("seasonal level","seasonal potential");
        matw <- matrix(c(1,0),1,2);
    }
    else if(seasonality=="p"){
        # Partial seasonality with a real part only
        maxlag <- frequency;
        modellags <- c(1,1,maxlag);
        componentsNumber <- 3;
        B$number <- 1;
        componentsNames <- c("level","potential","seasonal");
        matw <- matrix(c(1,0,1),1,3);
    }
    else if(seasonality=="f"){
        # Full seasonality with both real and imaginary parts
        maxlag <- frequency;
        modellags <- c(1,1,maxlag,maxlag);
        componentsNumber <- 4;
        B$number <- 2;
        componentsNames <- c("level","potential","seasonal level","seasonal potential");
        matw <- matrix(c(1,0,1,0),1,4);
    }

    initialValue <- initial;
# Initial values
    if(!is.null(initialValue)){
        if(length(initialValue) != maxlag*componentsNumber){
            warning(paste0("Wrong length of initial vector. Should be ",maxlag*componentsNumber,
                           " instead of ",length(initial),".\n",
                           "Values of initial vector will be generated"),call.=FALSE);
            initialValue <- NULL;
            initialGenerate <- TRUE;
        }
        else{
            initialGenerate <- FALSE;
            initialValue <- initial;
        }
    }
    else{
        initialGenerate <- TRUE;
    }

# In the case of wrong nsim, make it natural number. The same is for obs and frequency.
    nsim <- abs(round(nsim,0));
    obs <- abs(round(obs,0));
    obsStates <- obs + maxlag;
    frequency <- abs(round(frequency,0));

# Define arrays
    arrvt <- array(NA,c(obsStates,componentsNumber,nsim),dimnames=list(NULL,componentsNames,NULL));
    arrF <- array(0,c(componentsNumber,componentsNumber,nsim));
    matg <- matrix(0,componentsNumber,nsim);

    materrors <- matrix(NA,obs,nsim);
    matyt <- matrix(NA,obs,nsim);
    matot <- matrix(NA,obs,nsim);
    matInitialValue <- array(NA,c(maxlag,componentsNumber,nsim));
    AValue <- matrix(NA,2,nsim);
    BValue <- matrix(NA,B$number,nsim);

# Check the vector of probabilities
    if(is.vector(iprob)){
        if(any(iprob!=iprob[1])){
            if(length(iprob)!=obs){
                warning("Length of iprob does not correspond to number of observations.",call.=FALSE);
                if(length(iprob)>obs){
                    warning("We will cut off the excessive ones.",call.=FALSE);
                    iprob <- iprob[1:obs];
                }
                else{
                    warning("We will duplicate the last one.",call.=FALSE);
                    iprob <- c(iprob,rep(iprob[length(iprob)],obs-length(iprob)));
                }
            }
        }
    }

#### Generate stuff if needed ####
# First deal with initials
    if(initialGenerate){
        matInitialValue[,,] <- runif(componentsNumber*nsim*maxlag,0,1000);
        if(all(seasonality!=c("n","s"))){
            matInitialValue[1:maxlag,1:2,] <- rep(matInitialValue[maxlag,1:2,],each=maxlag);
        }
    }
    else{
        matInitialValue[1:maxlag,,] <- rep(initialValue,nsim);
    }
    arrvt[1:maxlag,,] <- matInitialValue;

# Now let's do parameters with transition + persistence
    if(A$generate){
        AValue[,] <- AGenerator(nsim);
    }
    else{
        AValue[1,] <- Re(A$value);
        AValue[2,] <- Im(A$value);
    }

    if(B$number!=0){
        if(B$generate){
            if(seasonality=="f"){
                BValue[,] <- AGenerator(nsim);
            }
            else{
                BValue[,] <- runif(nsim,0,1);
            }
        }
        else{
            if(seasonality=="f"){
                BValue[1,] <- Re(B$value);
                BValue[2,] <- Im(B$value);
            }
            else{
                BValue[1,] <- B$value;
            }
        }
    }

    arrF[1:2,1,] <- 1;
    for(i in 1:nsim){
        arrF[1:2,2,i] <- c(AValue[2,i]-1,1-AValue[1,i]);
        matg[1:2,i] <- c(AValue[1,i]-AValue[2,i],AValue[1,i]+AValue[2,i]);
    }

    if(seasonality=="p"){
        arrF[3,3,] <- 1;
        matg[3,] <- BValue[1,];
    }
    else if(seasonality=="f"){
        arrF[3:4,3,] <- 1;
        for(i in 1:nsim){
            arrF[3:4,4,i] <- c(BValue[2,i]-1,1-BValue[1,i]);
            matg[3:4,i] <- c(BValue[1,i]-BValue[2,i],BValue[1,i]+BValue[2,i]);
        }
    }

# If the chosen randomizer is not rnorm, rt and runif and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rlnorm","rt","runif")) & (length(args)==0)){
        warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
        randomizer = "rnorm";
    }

# Check if no argument was passed in dots
    if(length(args)==0){
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
            materrors[,] <- rt(nsim*obs,obs-(componentsNumber + maxlag));
        }

        if(randomizer!="rlnorm"){
# Center errors just in case
            materrors <- materrors - colMeans(materrors);
# Change variance to make some sense. Errors should not be rediculously high and not too low.
            materrors <- materrors * sqrt(abs(colMeans(as.matrix(arrvt[1:maxlag,1,]))));
        }
    }
# If arguments are passed, use them. WE ASSUME HERE THAT USER KNOWS WHAT HE'S DOING!
    else{
        materrors[,] <- eval(parse(text=paste0(randomizer,"(n=",nsim*obs,",", toString(as.character(args)),")")));
        if(randomizer=="rbeta"){
# Center the errors around 0
            materrors <- materrors - 0.5;
# Make a meaningful variance of data. Something resembling to var=1.
            materrors <- materrors / rep(sqrt(colMeans(materrors^2)) * sqrt(abs(arrvt[1,1,])),each=obs);
        }
        else if(randomizer=="rt"){
# Make a meaningful variance of data.
            materrors <- materrors * rep(sqrt(abs(colMeans(as.matrix(arrvt[1:maxlag,1,])))),each=obs);
        }
        else if(randomizer=="rlnorm"){
            materrors <- materrors - 1;
        }
    }

    veclikelihood <- -obs/2 *(log(2*pi*exp(1)) + log(colMeans(materrors^2)));

# Generate ones for the possible intermittency
    if(all(iprob == 1)){
        matot[,] <- 1;
    }
    else{
        matot[,] <- rbinom(obs*nsim,1,iprob);
    }

#### Simulate the data ####
    simulateddata <- simulatorwrap(arrvt,materrors,matot,arrF,matw,matg,"A","N","N",modellags);

    if(all(iprob == 1)){
        matyt <- simulateddata$matyt;
    }
    else{
        matyt <- round(simulateddata$matyt,0);
    }
    arrvt <- simulateddata$arrvt;
    dimnames(arrvt) <- list(NULL,componentsNames,NULL);

    if(nsim==1){
        matyt <- ts(matyt[,1],frequency=frequency);
        materrors <- ts(materrors[,1],frequency=frequency);
        arrvt <- ts(arrvt[,,1],frequency=frequency,start=c(0,frequency-maxlag+1));
        matot <- ts(matot[,1],frequency=frequency);
        matInitialValue <- matInitialValue[,,1];
    }
    else{
        matyt <- ts(matyt,frequency=frequency);
        materrors <- ts(materrors,frequency=frequency);
        matot <- ts(matot,frequency=frequency);
    }

    modelname <- paste0("CES(",seasonality,")");
    if(any(iprob!=1)){
        modelname <- paste0("i",modelname);
    }

    AValue <- complex(real=AValue[1,],imaginary=AValue[2,]);
    if(any(seasonality==c("n","s"))){
        BValue <- NULL;
    }
    else if(seasonality=="f"){
        BValue <- complex(real=BValue[1,],imaginary=BValue[2,]);
    }

    model <- list(model=modelname,
                  A=AValue, B=BValue, initial=matInitialValue,
                  data=matyt, states=arrvt, residuals=materrors,
                  occurrences=matot, logLik=veclikelihood);
    return(structure(model,class="smooth.sim"));
}
