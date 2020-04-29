#' Simulate Complex Exponential Smoothing
#'
#' Function generates data using CES with Single Source of Error as a data
#' generating process.
#'
#' For the information about the function, see the vignette:
#' \code{vignette("simulate","smooth")}
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
#' @param a First complex smoothing parameter. Should be a complex number.
#'
#' NOTE! CES is very sensitive to a and b values so it is advised to use values
#' from previously estimated model.
#' @param b Second complex smoothing parameter. Can be real if
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
#' \item \code{a} - Value of complex smoothing parameter a. If \code{nsim>1}, then
#' this is a vector.
#' \item \code{b} - Value of complex smoothing parameter b. If \code{seasonality="n"}
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
#' \item \code{occurrence} - Values of occurrence variable. Once again, can be either
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
                    frequency=1, a=NULL, b=NULL,
                    initial=NULL,
                    randomizer=c("rnorm","rt","rlaplace","rs"),
                    probability=1, ...){
# Function simulates the data using CES state space framework
#
# seasonality - the type of seasonality to produce.
# frequency - the frequency of the data. In the case of seasonal models must be > 1.
# a, b - complex smoothing parameters.
# initial - the vector of initial states,
#    If NULL it will be generated.
# obs - the number of observations in each time series.
# nsim - the number of series needed to be generated.
# randomizer - the type of the random number generator function
# ... - the parameters passed to the randomizer.

    randomizer <- randomizer[1];

    ellipsis <- list(...);

    AGenerator <- function(nsim=nsim){
        aValue <- matrix(NA,2,nsim);
        ANonStable <- rep(TRUE,nsim);
        for(i in 1:nsim){
            while(ANonStable[i]){
                aValue[1,i] <- runif(1,0.9,2.5);
                aValue[2,i] <- runif(1,0.9,1.1);

                if(((aValue[1,i]-2.5)^2 + aValue[2,i]^2 > 1.25) &
                   ((aValue[1,i]-0.5)^2 + (aValue[2,i]-1)^2 > 0.25) &
                   (aValue[1,i]-1.5)^2 + (aValue[2,i]-0.5)^2 < 1.5){
                    ANonStable[i] <- FALSE;
                }
            }
        }
        return(aValue);
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

    a <- list(value=a);
    b <- list(value=b);

    if(is.null(a$value)){
        a$generate <- TRUE;
    }
    else{
        a$generate <- FALSE;
        if(!(((Re(a$value)-2.5)^2 + Im(a$value)^2 > 1.25) &
                   ((Re(a$value)-0.5)^2 + (Im(a$value)-1)^2 > 0.25) &
                   (Re(a$value)-1.5)^2 + (Im(a$value)-0.5)^2 < 1.5)){
            warning("The provided complex smoothing parameter a leads to non-stable model!",call.=FALSE);
        }
    }

    if(all(is.null(b$value),any(seasonality==c("p","f")))){
        b$generate <- TRUE;
    }
    else{
        b$generate <- FALSE;
        if(seasonality=="f"){
            if(!(((Re(b$value)-2.5)^2 + Im(b$value)^2 > 1.25) &
                 ((Re(b$value)-0.5)^2 + (Im(b$value)-1)^2 > 0.25) &
                 (Re(b$value)-1.5)^2 + (Im(b$value)-0.5)^2 < 1.5)){
                warning("The provided complex smoothing parameter b leads to non-stable model!",call.=FALSE);
            }
        }
        else if(seasonality=="p"){
            if((b$value<0) | (b$value>1)){
                warning("Be careful with the provided b parameter - the model can be unstable.",call.=FALSE);
            }
        }
    }

    a$number <- 2;
# Define lags, number of components and number of parameters
    if(seasonality=="n"){
        # No seasonality
        lagsModelMax <- 1;
        lagsModel <- c(1,1);
        # Define the number of all the parameters (smoothing parameters + initial states). Used in AIC mainly!
        componentsNumber <- 2;
        b$number <- 0;
        componentsNames <- c("level","potential");
        matw <- matrix(c(1,0),1,2);
    }
    else if(seasonality=="s"){
        # Simple seasonality, lagged CES
        lagsModelMax <- frequency;
        lagsModel <- c(lagsModelMax,lagsModelMax);
        componentsNumber <- 2;
        b$number <- 0;
        componentsNames <- c("seasonal level","seasonal potential");
        matw <- matrix(c(1,0),1,2);
    }
    else if(seasonality=="p"){
        # Partial seasonality with a real part only
        lagsModelMax <- frequency;
        lagsModel <- c(1,1,lagsModelMax);
        componentsNumber <- 3;
        b$number <- 1;
        componentsNames <- c("level","potential","seasonal");
        matw <- matrix(c(1,0,1),1,3);
    }
    else if(seasonality=="f"){
        # Full seasonality with both real and imaginary parts
        lagsModelMax <- frequency;
        lagsModel <- c(1,1,lagsModelMax,lagsModelMax);
        componentsNumber <- 4;
        b$number <- 2;
        componentsNames <- c("level","potential","seasonal level","seasonal potential");
        matw <- matrix(c(1,0,1,0),1,4);
    }

    initialValue <- initial;
# Initial values
    if(!is.null(initialValue)){
        if(length(initialValue) != lagsModelMax*componentsNumber){
            warning(paste0("Wrong length of initial vector. Should be ",lagsModelMax*componentsNumber,
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
    obsStates <- obs + lagsModelMax;
    frequency <- abs(round(frequency,0));

# Define arrays
    arrvt <- array(NA,c(obsStates,componentsNumber,nsim),dimnames=list(NULL,componentsNames,NULL));
    arrF <- array(0,c(componentsNumber,componentsNumber,nsim));
    matg <- matrix(0,componentsNumber,nsim);

    materrors <- matrix(NA,obs,nsim);
    matyt <- matrix(NA,obs,nsim);
    matot <- matrix(NA,obs,nsim);
    matInitialValue <- array(NA,c(lagsModelMax,componentsNumber,nsim));
    aValue <- matrix(NA,2,nsim);
    bValue <- matrix(NA,b$number,nsim);

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
    }

#### Generate stuff if needed ####
# First deal with initials
    if(initialGenerate){
        matInitialValue[,,] <- runif(componentsNumber*nsim*lagsModelMax,0,1000);
        if(all(seasonality!=c("n","s"))){
            matInitialValue[1:lagsModelMax,1:2,] <- rep(matInitialValue[lagsModelMax,1:2,],each=lagsModelMax);
        }
    }
    else{
        matInitialValue[1:lagsModelMax,,] <- rep(initialValue,nsim);
    }
    arrvt[1:lagsModelMax,,] <- matInitialValue;

# Now let's do parameters with transition + persistence
    if(a$generate){
        aValue[,] <- AGenerator(nsim);
    }
    else{
        aValue[1,] <- Re(a$value);
        aValue[2,] <- Im(a$value);
    }

    if(b$number!=0){
        if(b$generate){
            if(seasonality=="f"){
                bValue[,] <- AGenerator(nsim);
            }
            else{
                bValue[,] <- runif(nsim,0,1);
            }
        }
        else{
            if(seasonality=="f"){
                bValue[1,] <- Re(b$value);
                bValue[2,] <- Im(b$value);
            }
            else{
                bValue[1,] <- b$value;
            }
        }
    }

    arrF[1:2,1,] <- 1;
    for(i in 1:nsim){
        arrF[1:2,2,i] <- c(aValue[2,i]-1,1-aValue[1,i]);
        matg[1:2,i] <- c(aValue[1,i]-aValue[2,i],aValue[1,i]+aValue[2,i]);
    }

    if(seasonality=="p"){
        arrF[3,3,] <- 1;
        matg[3,] <- bValue[1,];
    }
    else if(seasonality=="f"){
        arrF[3:4,3,] <- 1;
        for(i in 1:nsim){
            arrF[3:4,4,i] <- c(bValue[2,i]-1,1-bValue[1,i]);
            matg[3:4,i] <- c(bValue[1,i]-bValue[2,i],bValue[1,i]+bValue[2,i]);
        }
    }

    # If the chosen randomizer is not default and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rt","rlaplace","rs")) & (length(ellipsis)==0)){
        warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
        randomizer = "rnorm";
    }

    # Check if no argument was passed in dots
    if(length(ellipsis)==0){
        ellipsis$n <- nsim*obs;
        # Create vector of the errors
        if(any(randomizer==c("rnorm","rlaplace","rs"))){
            materrors[,] <- do.call(randomizer,ellipsis);
        }
        else if(randomizer=="rt"){
            # The degrees of freedom are df = n - k.
            materrors[,] <- rt(nsim*obs,obs-(componentsNumber + lagsModelMax));
        }

        # Center errors just in case
        materrors <- materrors - colMeans(materrors);
        # Change variance to make some sense. Errors should not be rediculously high and not too low.
        materrors <- materrors * sqrt(abs(colMeans(as.matrix(arrvt[1:lagsModelMax,1,]))));
        if(randomizer=="rs"){
            materrors <- materrors / 4;
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
            materrors <- materrors / rep(sqrt(colMeans(materrors^2)) *
                                             sqrt(abs(colMeans(as.matrix(arrvt[1:lagsModelMax,1,])))),each=obs);
        }
        else if(randomizer=="rt"){
            # Make a meaningful variance of data.
            materrors <- materrors * rep(sqrt(abs(colMeans(as.matrix(arrvt[1:lagsModelMax,1,])))),each=obs);
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
    simulateddata <- simulatorwrap(arrvt,materrors,matot,arrF,matw,matg,"A","N","N",lagsModel);

    if(all(probability == 1)){
        matyt <- simulateddata$matyt;
    }
    else{
        matyt <- round(simulateddata$matyt,0);
    }
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
    # If this is something unknown, forget about it
    else{
        veclikelihood <- NA;
    }

    if(nsim==1){
        matyt <- ts(matyt[,1],frequency=frequency);
        materrors <- ts(materrors[,1],frequency=frequency);
        arrvt <- ts(arrvt[,,1],frequency=frequency,start=c(0,frequency-lagsModelMax+1));
        matot <- ts(matot[,1],frequency=frequency);
        matInitialValue <- matInitialValue[,,1];
    }
    else{
        matyt <- ts(matyt,frequency=frequency);
        materrors <- ts(materrors,frequency=frequency);
        matot <- ts(matot,frequency=frequency);
    }

    modelname <- paste0("CES(",seasonality,")");
    if(any(probability!=1)){
        modelname <- paste0("i",modelname);
    }

    aValue <- complex(real=aValue[1,],imaginary=aValue[2,]);
    if(any(seasonality==c("n","s"))){
        bValue <- NULL;
    }
    else if(seasonality=="f"){
        bValue <- complex(real=bValue[1,],imaginary=bValue[2,]);
    }

    model <- list(model=modelname,
                  a=aValue, b=bValue, initial=matInitialValue,
                  data=matyt, states=arrvt, residuals=materrors,
                  occurrence=matot, logLik=veclikelihood);
    return(structure(model,class="smooth.sim"));
}
