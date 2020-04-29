utils::globalVariables(c("mvrnorm"));

#' Simulate Vector Exponential Smoothing
#'
#' Function generates data using VES model as a data generating process.
#'
#' For the information about the function, see the vignette:
#' \code{vignette("simulate","smooth")}
#'
#' @template ssAuthor
#' @template vssKeywords
#'
#' @template vssGeneralRef
#'
#' @param model Type of ETS model. This can consist of 3 or 4 chars:
#' \code{ANN}, \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA} etc.
#' Only pure additive models are supported. If you want to have multiplicative
#' one, then just take exponent of the generated data.
#' @param obs Number of observations in each generated time series.
#' @param nsim Number of series to generate (number of simulations to do).
#' @param nSeries Number of series in each generated group of series.
#' @param frequency Frequency of generated data. In cases of seasonal models
#' must be greater than 1.
#' @param persistence Matrix of smoothing parameters for all the components
#' of all the generated time series.
#' @param phi Value of damping parameter. If trend is not chosen in the model,
#' the parameter is ignored. If vector is provided, then several parameters
#' are used for different series.
#' @param transition Transition matrix. This should have the size appropriate
#' to the selected model and \code{nSeries}. e.g. if ETS(A,A,N) is selected
#' and \code{nSeries=3}, then the transition matrix should be 6 x 6. In case
#' of damped trend, the phi parameter should be placed in the matrix manually.
#' if \code{NULL}, then the default transition matrix for the selected type
#' of model is used. If both \code{phi} and \code{transition} are provided,
#' then the value of \code{phi} is ignored.
#' @param initial Vector of initial states of level and trend. The minimum
#' length is one (in case of ETS(A,N,N), the initial is used for all the
#' series), the maximum length is 2 x nSeries. If \code{NULL}, values are
#' generated for each series.
#' @param initialSeason Vector or matrix of initial states for seasonal
#' coefficients. Should have number of rows equal to \code{frequency}
#' parameter. If \code{NULL}, values are generated for each series.
#' @param seasonal The type of seasonal component across the series. Can be
#' \code{"individual"}, so that each series has its own component or \code{"common"},
#' so that the component is shared across the series.
#' @param weights The weights for the errors between the series with the common
#' seasonal component. Ignored if \code{seasonal="individual"}.
#' @param bounds Type of bounds to use for persistence vector if values are
#' generated. \code{"usual"} - bounds from p.156 by Hyndman et. al., 2008.
#' \code{"restricted"} - similar to \code{"usual"} but with upper bound equal
#' to 0.3. \code{"admissible"} - bounds from tables 10.1 and 10.2 of Hyndman
#' et. al., 2008. Using first letter of the type of bounds also works.
#' @param randomizer Type of random number generator function used for error
#' term. Defaults are: \code{rnorm}, \code{rt}, \code{rlaplace}, \code{rs}. But
#' any function from \link[stats]{Distributions} will do the trick if the
#' appropriate parameters are passed. \code{mvrnorm} from MASS package can also
#' be used.
#' @param ...  Additional parameters passed to the chosen randomizer. All the
#' parameters should be passed in the order they are used in chosen randomizer.
#' For example, passing just \code{sd=0.5} to \code{rnorm} function will lead
#' to the call \code{rnorm(obs, mean=0.5, sd=1)}. ATTENTION! When generating
#' the multiplicative errors some tuning might be needed to obtain meaningful
#' data. \code{sd=0.1} is usually already a high value for such models.
#'
#' @return List of the following values is returned:
#' \itemize{
#' \item \code{model} - Name of ETS model.
#' \item \code{data} - The matrix (or an array if \code{nsim>1}) of the
#' generated series.
#' \item \code{states} - The matrix (or array if \code{nsim>1}) of states.
#' States are in columns, time is in rows.
#' \item \code{persistence} - The matrix (or array if \code{nsim>1}) of
#' smoothing parameters used in the simulation.
#' \item \code{transition} - The transition matrix (or array if \code{nsim>1}).
#' \item \code{initial} - Vector (or matrix) of initial values.
#' \item \code{initialSeason} - Vector (or matrix) of initial seasonal
#' coefficients.
#' \item \code{residuals} - Error terms used in the simulation. Either matrix
#' or array, depending on \code{nsim}.
#' }
#'
#' @seealso \code{\link[smooth]{es}, \link[forecast]{ets},
#' \link[forecast]{forecast}, \link[stats]{ts}, \link[stats]{Distributions}}
#'
#' @examples
#'
#' # Create 40 observations of quarterly data using AAA model with errors
#' # from normal distribution
#' \dontrun{VESAAA <- sim.ves(model="AAA",frequency=4,obs=40,nSeries=3,
#'                    randomizer="rnorm",mean=0,sd=100)}
#'
#' # You can also use mvrnorm function from MASS package as randomizer,
#' # but you need to provide mu and Sigma explicitly
#' \dontrun{VESANN <- sim.ves(model="ANN",frequency=4,obs=40,nSeries=2,
#'                    randomizer="mvrnorm",mu=c(100,50),Sigma=matrix(c(40,20,20,30),2,2))}
#'
#' # When generating the data with multiplicative model a more diligent definitiion
#' # of parameters is needed. Here's an example with MMM model:
#'
#' VESMMM <- sim.ves("AAA", obs=120, nSeries=2, frequency=12, initial=c(10,0),
#'           initialSeason=runif(12,-1,1), persistence=c(0.06,0.05,0.2), mean=0, sd=0.03)
#' VESMMM$data <- exp(VESMMM$data)
#'
#' # Note that smoothing parameters should be low and the standard diviation should
#' # definitely be less than 0.1. Otherwise you might face the explosions.
#'
#' @export sim.ves
sim.ves <- function(model="ANN", obs=10, nsim=1, nSeries=2,
                    frequency=1, persistence=NULL, phi=1,
                    transition=NULL,
                    initial=NULL, initialSeason=NULL,
                    seasonal=c("individual, common"),
                    weights=rep(1/nSeries,nSeries),
                    bounds=c("usual","admissible","restricted"),
                    randomizer=c("rnorm","rt","rlaplace","rs"),
                    ...){
    # Function generates data using VES model.
    #    Copyright (C) 2018 - Inf Ivan Svetunkov

    randomizer <- randomizer[1];
    ellipsis <- list(...);
    seasonalType <- substr(seasonal,1,1)[1];
    bounds <- bounds[1];
    # If R decided that by "b" we meant "bounds", fix this!
    if(is.numeric(bounds)){
        ellipsis$b <- bounds;
        bounds <- "u";
    }
    bounds <- substring(bounds[1],1,1);

    # If the chosen randomizer is not rnorm, rt and runif and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rt","rlaplace","rs")) & (length(ellipsis)==0)){
        warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
        randomizer = "rnorm";
    }

    # If chosen model is "AAdN" or anything like that, we are taking the appropriate values
    if(nchar(model)==4){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,4,4);
        if(substring(model,3,3)!="d"){
            warning(paste0("You have defined a strange model: ",model),call.=FALSE);
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        if(Ttype!="N" & all(phi==1)){
            model <- paste0(Etype,Ttype,Stype);
            warning(paste0("Damping parameter is set to 1. Changing model to: ",model),call.=FALSE);
        }
    }
    else if(nchar(model)==3){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,3,3);
        if(any(phi!=1) & Ttype!="N"){
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
    nSeries <- abs(round(nSeries,0));
    damped <- FALSE;

    # Check the used model and estimate the length of needed persistence vector.
    if(all(Etype!=c("A","M"))){
        stop("Wrong error type! Should be 'A' or 'M'.",call.=FALSE);
    }
    else{
        # The number initial values of the state vector
        nComponentsAll <- 1;
        # The lag of components (needed for the seasonal models)
        lagsModel <- 1;
        # The names of the state vector components
        componentsNames <- "level";
        matW <- diag(nSeries);
        # The transition matrix
        transComponents <- matrix(1,nSeries,1);
    }

    # Check the trend type of the model
    if(all(Ttype!=c("A","M","N"))){
        stop("Wrong trend type! Should be 'N', 'A' or 'M'.",call.=FALSE);
    }
    else if(Ttype!="N"){
        nComponentsAll <- nComponentsAll + 1;
        lagsModel <- c(lagsModel,1);
        componentsNames <- c(componentsNames,"trend");
        if(all(is.na(phi)) | all(is.null(phi))){
            phi <- 1;
            damped <- FALSE;
        }
        else{
            if(length(phi)==1){
                if(phi!=1){
                    if(phi<0 | phi>2){
                        warning(paste0("Damping parameter should lie in (0, 2) ",
                                       "region! You have chosen phi=",phi,
                                       ". Be careful!"),
                                call.=FALSE);
                    }
                    model <- paste0(Etype,Ttype,"d",Stype);
                    damped <- TRUE;
                }
            }
            else if(length(phi)==nSeries){
                if(any(phi!=1)){
                    if(any(phi<0 | phi>2)){
                        warning(paste0("Damping parameter should lie in (0, 2) ",
                                       "region! Some of yours don't. Be careful!"),
                                call.=FALSE);
                    }
                    model <- paste0(Etype,Ttype,"d",Stype);
                    damped <- TRUE;
                }
            }
            else{
                warning(paste0("Wrong length of phi. It should be either 1 or ",
                               nSeries), call.=FALSE);
                phi <- 1;
                damped <- FALSE;
            }
        }
        transComponents <- cbind(transComponents,phi);
        componentTrend <- TRUE;
    }
    else{
        componentTrend <- FALSE;
    }

    nComponentsNonSeasonal <- nComponentsAll;

    # Check the seasonaity type of the model
    if(all(Stype!=c("A","M","N"))){
        stop("Wrong seasonality type! Should be 'N', 'A' or 'M'.",call.=FALSE);
    }

    if(Stype!="N" & frequency==1){
        stop("Cannot create the seasonal model with the data frequency 1!",call.=FALSE);
    }

    if(Stype!="N"){
        lagsModel <- c(lagsModel,frequency);
        componentsNames <- c(componentsNames,"seasonal");
        componentSeasonal <- TRUE;
        nComponentsAll <- nComponentsAll + 1;
        transComponents <- cbind(transComponents,1);
    }
    else{
        componentSeasonal <- FALSE;
    }

    # Check if a multiplicative model is needed
    if(any(c(Etype,Ttype,Stype)=="M")){
        if(any(c(Etype,Ttype,Stype)=="A")){
            warning("Mixed models are not available. Switching to pure multiplicative.",call.=FALSE);
        }
        Etype <- "M";
        Ttype <- ifelse(Ttype=="A","M",Ttype);
        Stype <- ifelse(Stype=="A","M",Stype);
        modelIsMultiplicative <- TRUE;
    }
    else{
        modelIsMultiplicative <- FALSE;
    }

    # Check seasonal parameter
    if(all(seasonalType!=c("i","c"))){
        warning("You asked for a strange seasonal value. We don't do that here. Switching to individual.",
                call.=FALSE);
        seasonalType <- "i";
    }
    else{
        if(seasonalType=="c" && Stype=="N"){
            stop("Cannot do anything with common seasonality in non-seasonal model.",call.=FALSE);
        }
    }

    dataNames <- paste0("Series",c(1:nSeries));
    if(seasonalType=="i"){
        componentsNames <- paste0(rep(dataNames,each=nComponentsAll),
                                  "_",componentsNames);
    }
    else{
        componentsNames <- c(paste0(rep(dataNames,each=nComponentsNonSeasonal),
                                    "_",componentsNames[-nComponentsAll]),componentsNames[nComponentsAll]);
    }

    #### Form persistence matrix ####
    # seasonal == "individual"
    if(seasonalType=="i"){
        matG <- matrix(0,nComponentsAll*nSeries,nSeries);
        if(!is.null(persistence)){
            if((length(persistence)==nComponentsAll)){
                for(i in 1:nSeries){
                    matG[(i-1)*nComponentsAll+c(1:nComponentsAll),i] <- c(persistence);
                }
            }
            else if(length(persistence)==nComponentsAll*nSeries){
                for(i in 1:nSeries){
                    matG[(i-1)*nComponentsAll+c(1:nComponentsAll),
                         i] <- c(persistence)[(i-1)*nComponentsAll+c(1:nComponentsAll)];
                }
            }
            else if(length(persistence)==nComponentsAll*nSeries^2){
                matG[,] <- persistence;
            }
            else{
                stop(paste0("The length of persistence matrix is wrong! It should be either ",
                            nComponentsAll,", or ",nComponentsAll*nSeries,", or ",nComponentsAll*nSeries^2,
                            "."),
                     call.=FALSE);
            }
            persistenceGenerate <- FALSE;
        }
        else{
            persistenceGenerate <- TRUE;
        }
    }
    # seasonal == "common"
    else{
        matG <- matrix(0,nComponentsNonSeasonal*nSeries+1,nSeries);
        if(!is.null(persistence)){
            if((length(persistence)==nComponentsAll)){
                for(i in 1:nSeries){
                    matG[(i-1)*nComponentsNonSeasonal+c(1:nComponentsNonSeasonal),
                         i] <- persistence[1:nComponentsNonSeasonal];
                }
                matG[nComponentsNonSeasonal*nSeries+1,] <- weights*persistence[nComponentsAll];
            }
            else if(length(persistence)==(nComponentsNonSeasonal*nSeries+1)*nSeries){
                matG[,] <- persistence;
            }
            else{
                stop(paste0("The length of persistence matrix is wrong! It should be either ",
                            nComponentsAll,", or ",(nComponentsNonSeasonal*nSeries+1)*nSeries,
                            "."),
                     call.=FALSE);
            }
            persistenceGenerate <- FALSE;
        }
        else{
            persistenceGenerate <- TRUE;
        }
    }

    #### Form transition matrix ####
    # seasonal == "individual"
    if(seasonalType=="i"){
        matF <- diag(nSeries*nComponentsAll);
        if(!is.null(transition)){
            if(length(transition)==nComponentsAll^2){
                for(i in 1:nSeries){
                    matF[(i-1)*nComponentsAll+c(1:nComponentsAll),
                         (i-1)*nComponentsAll+c(1:nComponentsAll)] <- c(transition);
                }
            }
            else if(length(transition)==(nComponentsAll*nSeries)^2){
                matF[,] <- c(transition);
            }
            else{
                stop(paste0("The length of transition matrix is wrong! It should be either",
                            nComponentsAll^2,", or ",(nComponentsAll*nSeries)^2,
                            "."),
                     call.=FALSE);
            }

            if(damped){
                phi <- 1;
                damped <- FALSE;
            }
        }
        else{
            for(i in 1:nSeries){
                matF[(i-1)*nComponentsAll+1,
                     (i-1)*nComponentsAll+1] <- transComponents[i,1];
                if(componentTrend){
                    matF[(i-1)*nComponentsAll+1:2,
                         (i-1)*nComponentsAll+2] <- transComponents[i,2];
                }
                if(componentSeasonal){
                    matF[(i-1)*nComponentsAll+nComponentsAll,
                         (i-1)*nComponentsAll+nComponentsAll] <- 1;
                }
            }
        }
    }
    # seasonal == "common"
    else{
        matF <- diag(nComponentsNonSeasonal*nSeries+1);
        if(!is.null(transition)){
            if(length(transition)==(nComponentsNonSeasonal*nSeries+1)^2){
                matF[,] <- c(transition);
            }
            else{
                stop(paste0("The length of transition matrix is wrong! It should be ",
                            (nComponentsNonSeasonal*nSeries+1)^2,"."),
                     call.=FALSE);
            }

            if(damped){
                phi <- 1;
                damped <- FALSE;
            }
        }
        else{
            for(i in 1:nSeries){
                matF[(i-1)*nComponentsNonSeasonal+1,
                     (i-1)*nComponentsNonSeasonal+1] <- transComponents[i,1];
                if(componentTrend){
                    matF[(i-1)*nComponentsNonSeasonal+1:2,
                         (i-1)*nComponentsNonSeasonal+2] <- transComponents[i,2];
                }
            }
            matF[nComponentsNonSeasonal*nSeries+1,
                 nComponentsNonSeasonal*nSeries+1] <- 1;
        }
    }

    #### Form measurement matrix ####
    # seasonal == "individual"
    if(seasonalType=="i"){
        matW <- matrix(0,nSeries,nComponentsAll*nSeries);
        for(i in 1:nSeries){
            matW[i,(i-1)*nComponentsAll+c(1:nComponentsAll)] <- transComponents[i,];
        }
    }
    else{
        matW <- matrix(0,nSeries,nSeries*nComponentsNonSeasonal+1);
        for(i in 1:nSeries){
            matW[i,c(1:nComponentsNonSeasonal)+nComponentsNonSeasonal*(i-1)] <- transComponents[i,-nComponentsAll];
        }
        matW[,nSeries*nComponentsNonSeasonal+1] <- transComponents[,nComponentsAll];
    }

    # Make matrices and arrays
    # seasonal == "individual"
    if(seasonalType=="i"){
        lagsModel <- matrix(lagsModel,nComponentsAll*nSeries,1);
    }
    else{
        lagsModel <- matrix(c(rep(lagsModel[-nComponentsAll],nSeries),lagsModel[nComponentsAll]),
                            nSeries*nComponentsNonSeasonal+1,1);
    }
    lagsModelMax <- max(lagsModel);

    #### Check the initals ####
    if(!is.null(initial)){
        if((length(initial)==nComponentsNonSeasonal) |
           (length(initial)==nComponentsNonSeasonal*nSeries)){
            initial <- matrix(c(initial),nComponentsNonSeasonal*nSeries,1);
            initialGenerate <- FALSE;
        }
        else{
            warning(paste0("The length of initial state vector does not correspond to the chosen model!\n",
                           "Falling back to random number generator."),call.=FALSE);
            initial <- NULL;
            initialGenerate <- TRUE;
        }
    }
    else{
        initialGenerate <- TRUE;
    }

    #### Check the inital seasonal ####
    if(!is.null(initialSeason)){
        if(seasonalType=="c"){
            if(length(initialSeason)==lagsModelMax*nSeries){
                stop(paste0("initialSeason should be a vector of length ", lagsModelMax,
                            " in case of seasonal='c'."),
                     call.=FALSE);
            }
            else if(length(initialSeason)==lagsModelMax){
                initialSeason <- matrix(c(initialSeason),1,lagsModelMax,byrow=TRUE);
                initialSeasonGenerate <- FALSE;
            }
            else{
                warning(paste0("The length of seasonal initial states does not correspond to the chosen frequency!\n",
                               "Falling back to random number generator."),call.=FALSE);
                initialSeason <- NULL;
                initialSeasonGenerate <- TRUE;
            }
        }
        else{
            if((length(initialSeason)==lagsModelMax) |
               (length(initialSeason)==lagsModelMax*nSeries)){
                initialSeason <- matrix(c(initialSeason),nSeries,lagsModelMax,byrow=TRUE);
                initialSeasonGenerate <- FALSE;
            }
            else{
                warning(paste0("The length of seasonal initial states does not correspond to the chosen frequency!\n",
                               "Falling back to random number generator."),call.=FALSE);
                initialSeason <- NULL;
                initialSeasonGenerate <- TRUE;
            }
        }
    }
    else{
        if(componentSeasonal){
            initialSeasonGenerate <- TRUE;
        }
        else{
            initialSeasonGenerate <- FALSE;
        }
    }

    ##### Form arrays #####
    arrayW <- array(matW,c(dim(matW),nsim),
                    dimnames=list(dataNames,componentsNames,NULL));
    arrayF <- array(matF,c(dim(matF),nsim),
                    dimnames=list(componentsNames,componentsNames,NULL));
    arrayG <- array(matG,c(dim(matG),nsim),
                    dimnames=list(componentsNames,dataNames,NULL));
    if(seasonalType=="i"){
        arrayStates <- array(NA,c(nComponentsAll*nSeries,obs+lagsModelMax,nsim),
                             dimnames=list(componentsNames,NULL,NULL));
    }
    else{
        arrayStates <- array(NA,c(nComponentsNonSeasonal*nSeries+1,obs+lagsModelMax,nsim),
                             dimnames=list(componentsNames,NULL,NULL));
    }
    arrayErrors <- array(NA,c(nSeries,obs,nsim),
                         dimnames=list(dataNames,NULL,NULL));
    arrayActuals <- array(NA,c(nSeries,obs,nsim),
                          dimnames=list(dataNames,NULL,NULL));

    #### Generate persistence ####
    # If the persistence is NULL or was of the wrong length, generate the values
    if(persistenceGenerate){
        ### For the case of "usual" bounds make restrictions on the generated smoothing parameters so the ETS can be "averaging" model.

        persistence <- rep(0,nComponentsAll);
        ### First generate the first smoothing parameter.
        if(bounds=="u"){
            persistence[1] <- runif(1,0,1);
        }
        ### These restrictions are even touhger
        else if(bounds=="r"){
            persistence[1] <- runif(1,0,0.3);
        }

        # In case of the multiplicative model, the parameter should be very small
        if(modelIsMultiplicative){
            persistence[1] <- runif(1,0,0.03);
        }

        ### Fill in the other smoothing parameters
        if(bounds!="a"){
            if(componentTrend){
                persistence[2] <- runif(1,0,persistence[1]);
            }
            if(componentSeasonal){
                persistence[nComponentsAll] <- runif(1,0,max(0,1-persistence[1]));
            }
        }
        ### In case of admissible bounds, do some stuff
        else{
            persistence[] <- runif(nComponentsAll,1-1/phi,1+1/phi);
            if(componentTrend){
                persistence[2] <- runif(nsim,persistence[1]*(phi-1),(2-persistence[1])*(1+phi));
                if(componentSeasonal){
                    ThetaFunction <- function(Theta){
                        result <- (phi*persistence[1]+phi+1)/(persistence[3]) +
                            ((phi-1)*(1+cos(Theta)-cos(lagsModelMax*Theta)) +
                                 cos((lagsModelMax-1)*Theta)-phi*cos((lagsModelMax+1)*Theta))/(2*(1+cos(Theta))*(1-cos(lagsModelMax*Theta)));
                        return(abs(result));
                    }

                    persistence[3] <- runif(1,max(1-1/phi-persistence[1],0),1+1/phi-persistence[1]);

                    B <- phi*(4-3*persistence[3])+persistence[3]*(1-phi)/lagsModelMax;
                    C <- sqrt(B^2-8*(phi^2*(1-persistence[3])^2+2*(phi-1)*(1-persistence[3])-1)+8*persistence[3]^2*(1-phi)/lagsModelMax);
                    persistence[1] <- runif(1,1-1/phi-persistence[3]*(1-lagsModelMax+phi*(1+lagsModelMax))/(2*phi*lagsModelMax),(B+C)/(4*phi));

                    # Solve the equation to get Theta value. Theta
                    Theta <- 0.1;
                    Theta <- optim(Theta,ThetaFunction,method="Brent",lower=0,upper=1)$par;

                    D <- (phi*(1-persistence[1])+1)*(1-cos(Theta)) - persistence[3]*((1+phi)*(1-cos(Theta) - cos(lagsModelMax*Theta)) +
                                                                               cos((lagsModelMax-1)*Theta)+phi*cos((lagsModelMax+1)*Theta))/
                        (2*(1+cos(Theta))*(1-cos(lagsModelMax*Theta)));
                    persistence[2] <- runif(1,-(1-phi)*(persistence[3]/lagsModelMax+persistence[1]),D+(phi-1)*persistence[1]);
                }
            }
            else{
                if(Stype!="N"){
                    persistence[1] <- runif(nsim,-2/(lagsModelMax-1),2);
                    persistence[2] <- runif(1,max(-lagsModelMax*persistence[1],0),2-persistence[1]);
                    persistence[1] <- runif(nsim,-2/(lagsModelMax-1),2-persistence[2]);
                }
            }
        }
        # Fill in the matrix
        if(seasonalType=="i"){
            for(i in 1:nSeries){
                matG[(i-1)*nComponentsAll+c(1:nComponentsAll),i] <- persistence;
            }
        }
        else{
            for(i in 1:nSeries){
                matG[(i-1)*nComponentsNonSeasonal+c(1:nComponentsNonSeasonal),i] <- persistence[-nComponentsAll];
            }
            matG[nComponentsNonSeasonal*nSeries+1,] <- weights*persistence[nComponentsAll];
        }
        arrayG[,,] <- matG;
    }

    # If we generate things, set the bounds to generate from
    if(initialGenerate || initialSeasonGenerate){
        # Level, Trend, Seasonality
        if(modelIsMultiplicative){
            initialBounds <- c(0,10,-0.1,0.1,-2,2);
        }
        else{
            initialBounds <- c(0,5000,-100,100,-500,500);
        }
    }

    #### Generate initial states ####
    if(initialGenerate){
        # seasonal == "individual"
        if(seasonalType=="i"){
            arrayStates[c(0:(nSeries-1))*nComponentsAll+1,
                        1:lagsModelMax,] <- runif(nsim,initialBounds[1],initialBounds[2]);
            if(componentTrend){
                arrayStates[c(0:(nSeries-1))*nComponentsAll+2,
                            1:lagsModelMax,] <- runif(nsim,initialBounds[3],initialBounds[4]);
            }
        }
        # seasonal == "common"
        else{
            arrayStates[c(0:(nSeries-1))*nComponentsNonSeasonal+1,
                        1:lagsModelMax,] <- runif(nsim,initialBounds[1],initialBounds[2]);
            if(componentTrend){
                arrayStates[c(0:(nSeries-1))*nComponentsNonSeasonal+2,
                            1:lagsModelMax,] <- runif(nsim,initialBounds[3],initialBounds[4]);
            }

        }
        initial <- matrix(arrayStates[1:nComponentsNonSeasonal,1,],ncol=nsim);
    }
    else{
        # seasonal == "individual"
        if(seasonalType=="i"){
            for(i in 1:nSeries){
                arrayStates[((i-1)*nComponentsAll)+(1:nComponentsNonSeasonal),1:lagsModelMax,] <-
                    initial[(i-1)*nComponentsNonSeasonal+(1:nComponentsNonSeasonal),1];
            }
        }
        # seasonal == "common"
        else{
            for(i in 1:nSeries){
                arrayStates[((i-1)*nComponentsNonSeasonal)+(1:nComponentsNonSeasonal),1:lagsModelMax,] <-
                    initial[(i-1)*nComponentsNonSeasonal+(1:nComponentsNonSeasonal),1];
            }
        }
    }

    #### Generate seasonal initials ####
    if(initialSeasonGenerate){
        # seasonal == "individual"
        if(seasonalType=="i"){
            # Create and normalize seasonal components
            initialSeason <- runif(lagsModelMax,initialBounds[5],initialBounds[6]);
            initialSeason <- initialSeason - mean(initialSeason);
            arrayStates[nComponentsAll*c(1:nSeries),1:lagsModelMax,] <- matrix(initialSeason,nSeries,
                                                                               lagsModelMax,byrow=TRUE);
        }
        # seasonal == "common"
        else{
            # Create and normalize seasonal components
            initialSeason <- runif(lagsModelMax,initialBounds[5],initialBounds[6]);
            initialSeason <- initialSeason - mean(initialSeason);
            arrayStates[nComponentsNonSeasonal*nSeries+1,1:lagsModelMax,] <- initialSeason;
        }
    }
    # If the seasonal model is chosen, fill in the first "frequency" values of seasonal component.
    else{
        if(componentSeasonal){
            # seasonal == "individual"
            if(seasonalType=="i"){
                for(i in 1:nSeries){
                    arrayStates[i*nComponentsAll,1:lagsModelMax,] <- initialSeason[i,];
                }
            }
            # seasonal == "common"
            else{
                arrayStates[nComponentsNonSeasonal*nSeries+1,1:lagsModelMax,] <- initialSeason;
            }
        }
    }

    #### Generate errors ####
    if(length(ellipsis)==0){
        # Create vector of the errors
        if(any(randomizer==c("rnorm"))){
            arrayErrors[,,] <- rnorm(nsim*obs*nSeries);
        }
        else if(randomizer=="rt"){
            # The degrees of freedom are df = n - k.
            arrayErrors[,,] <- rt(nsim*obs*nSeries,obs-(nComponentsAll + lagsModelMax));
        }
        else if(randomizer=="rlaplace"){
            arrayErrors[,,] <- rlaplace(nsim*obs*nSeries);
        }
        else if(randomizer=="rs"){
            arrayErrors[,,] <- rs(nsim*obs*nSeries);
        }

        # Make variance sort of meaningful
        if(modelIsMultiplicative){
            arrayErrors <- arrayErrors * 0.03;
        }
        else{
            if(arrayStates[1,1,1]!=0){
                arrayErrors <- arrayErrors * sqrt(abs(arrayStates[1,1,1]));
            }
        }
    }
    # If arguments are passed, use them. WE ASSUME HERE THAT USER KNOWS WHAT HE'S DOING!
    else{
        ellipsis$n <- nsim*obs;
        arrayErrors[,,] <- do.call(randomizer,ellipsis);
    }

    #### Simulate the data ####
    simulatedData <- vSimulatorWrap(arrayStates,arrayErrors,arrayF,arrayW,arrayG,lagsModel);

    if(modelIsMultiplicative){
        arrayActuals[,,] <- exp(simulatedData$arrayActuals);
    }
    else{
        arrayActuals[,,] <- simulatedData$arrayActuals;
    }
    arrayStates[,,] <- simulatedData$arrayStates;

    if(nsim==1){
        arrayActuals <- ts(t(arrayActuals[,,1]),frequency=frequency);
        arrayErrors <- ts(t(arrayErrors[,,1]),frequency=frequency);
        arrayStates <- ts(t(arrayStates[,,1]),frequency=frequency,start=c(0,frequency-lagsModelMax+1));
    }
    else{
        arrayActuals <- aperm(arrayActuals,c(2,1,3));
        arrayErrors <- aperm(arrayErrors,c(2,1,3));
        arrayStates <- aperm(arrayStates,c(2,1,3));
    }

    model <- paste0("VES(",model,")");
    if(seasonalType=="c"){
        model <- paste0(model,"[CCC]");
    }

    model <- list(model=model, data=arrayActuals, states=arrayStates, persistence=arrayG, phi=phi,
                  transition=arrayF, measurement=arrayW,
                  initial=initial, initialSeason=initialSeason, residuals=arrayErrors);
    return(structure(model,class=c("vsmooth.sim","smooth.sim")));
}
