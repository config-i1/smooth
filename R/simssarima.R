#' Simulate SSARIMA
#'
#' Function generates data using SSARIMA with Single Source of Error as a data
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
#' @template ssARIMARef
#'
#' @param orders List of orders, containing vector variables \code{ar},
#' \code{i} and \code{ma}. Example:
#' \code{orders=list(ar=c(1,2),i=c(1),ma=c(1,1,1))}. If a variable is not
#' provided in the list, then it is assumed to be equal to zero. At least one
#' variable should have the same length as \code{lags}.
#' @param lags Defines lags for the corresponding orders (see examples above).
#' The length of \code{lags} must correspond to the length of \code{orders}.
#' There is no restrictions on the length of \code{lags} vector.
#' It is recommended to order \code{lags} ascending.
#' @param initial Vector of initial values for state matrix. If \code{NULL},
#' then generated using advanced, sophisticated technique - uniform
#' distribution.
#' @param AR Vector or matrix of AR parameters. The order of parameters should
#' be lag-wise. This means that first all the AR parameters of the firs lag
#' should be passed, then for the second etc. AR of another ssarima can be
#' passed here.
#' @param MA Vector or matrix of MA parameters. The order of parameters should
#' be lag-wise. This means that first all the MA parameters of the firs lag
#' should be passed, then for the second etc. MA of another ssarima can be
#' passed here.
#' @param constant If \code{TRUE}, constant term is included in the model. Can
#' also be a number (constant value).
#' @param bounds Type of bounds to use for AR and MA if values are generated.
#' \code{"admissible"} - bounds guaranteeing stability and stationarity of
#' SSARIMA. \code{"none"} - we generate something, but do not guarantee
#' stationarity and stability. Using first letter of the type of bounds also
#' works.
#' @param ...  Additional parameters passed to the chosen randomizer. All the
#' parameters should be passed in the order they are used in chosen randomizer.
#' For example, passing just \code{sd=0.5} to \code{rnorm} function will lead
#' to the call \code{rnorm(obs, mean=0.5, sd=1)}.
#'
#' @return List of the following values is returned:
#' \itemize{
#' \item \code{model} - Name of SSARIMA model.
#' \item \code{AR} - Value of AR parameters. If \code{nsim>1}, then this is a
#' matrix.
#' \item \code{MA} - Value of MA parameters. If \code{nsim>1}, then this is a
#' matrix.
#' \item \code{constant} - Value of constant term. If \code{nsim>1}, then this
#' is a vector.
#' \item \code{initial} - Initial values of SSARIMA. If \code{nsim>1}, then this
#' is a matrix.
#' \item \code{data} - Time series vector (or matrix if \code{nsim>1}) of the
#' generated series.
#' \item \code{states} - Matrix (or array if \code{nsim>1}) of states. States
#' are in columns, time is in rows.
#' \item \code{residuals} - Error terms used in the simulation. Either vector or
#' matrix, depending on \code{nsim}.
#' \item \code{occurrence} - Values of occurrence variable. Once again, can be
#' either a vector or a matrix...
#' \item \code{logLik} - Log-likelihood of the constructed model.
#' }
#'
#' @seealso \code{\link[smooth]{sim.es}, \link[smooth]{ssarima},
#' \link[stats]{Distributions}, \link[smooth]{orders}}
#'
#' @examples
#'
#' # Create 120 observations from ARIMA(1,1,1) with drift. Generate 100 time series of this kind.
#' x <- sim.ssarima(ar.orders=1,i.orders=1,ma.orders=1,obs=120,nsim=100,constant=TRUE)
#'
#' # Generate similar thing for seasonal series of SARIMA(1,1,1)(0,0,2)_4
#' x <- sim.ssarima(ar.orders=c(1,0),i.orders=c(1,0),ma.orders=c(1,2),lags=c(1,4),
#'                  frequency=4,obs=80,nsim=100,constant=FALSE)
#'
#' # Generate 10 series of high frequency data from SARIMA(1,0,2)_1(0,1,1)_7(1,0,1)_30
#' x <- sim.ssarima(ar.orders=c(1,0,1),i.orders=c(0,1,0),ma.orders=c(2,1,1),lags=c(1,7,30),
#'                  obs=360,nsim=10)
#'
#'
#' @export sim.ssarima
sim.ssarima <- function(orders=list(ar=0,i=1,ma=1), lags=1,
                        obs=10, nsim=1,
                        frequency=1, AR=NULL, MA=NULL, constant=FALSE,
                        initial=NULL, bounds=c("admissible","none"),
                        randomizer=c("rnorm","rt","rlaplace","rs"),
                        probability=1, ...){
# Function generates data using SSARIMA in Single Source of Error as a data generating process.
#    Copyright (C) 2015 - Inf Ivan Svetunkov

    randomizer <- randomizer[1];
    ellipsis <- list(...);
    bounds <- bounds[1];
    # If R decided that by "b" we meant "bounds", fix this!
    if(is.numeric(bounds)){
        ellipsis$b <- bounds;
        bounds <- "u";
    }

    if(all(bounds!=c("n","a","none","admissible"))){
        warning(paste0("Strange type of bounds provided: ",bounds,". Switching to 'admissible'."),
                call.=FALSE);
        bounds <- "a";
    }

    bounds <- substring(bounds[1],1,1);

    if(!is.null(orders)){
        ar.orders <- orders$ar;
        i.orders <- orders$i;
        ma.orders <- orders$ma;
    }
    else{
        ar.orders <- 0;
        i.orders <- 0;
        ma.orders <- 0;
    }

    if("ar.orders" %in% names(ellipsis)){
        ar.orders <- ellipsis$ar.orders;
        ellipsis$ar.orders <- NULL;
    }
    if("i.orders" %in% names(ellipsis)){
        i.orders <- ellipsis$i.orders;
        ellipsis$i.orders <- NULL;
    }
    if("ma.orders" %in% names(ellipsis)){
        ma.orders <- ellipsis$ma.orders;
        ellipsis$ma.orders <- NULL;
    }

#### Elements Generator for AR and MA ####
elementsGenerator <- function(ar.orders=ar.orders, ma.orders=ma.orders, i.orders=i.orders,
                              ARValue=ARValue, MAValue=MAValue,
                              ARGenerate=FALSE, MAGenerate=FALSE){
    componentsNumber <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
    matvt <- matrix(1,componentsNumber+constantRequired,componentsNumber+constantRequired);
    vecg <- matrix(0,componentsNumber+constantRequired,1);
    matF <- diag(componentsNumber+constantRequired);

    if(ARGenerate){
        ARRoots <- 0.5;
        while(any(ARRoots<1)){
            ARValue <- runif(ARNumber,-1,1);

            elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, componentsNumber,
                                    ARValue, MAValue, NULL, NULL,
                                    matvt, vecg, matF,
                                    "b", 0, matrix(1,obsStates,1), matrix(1,1,1), matrix(0,1,1),
                                    FALSE, FALSE, FALSE, FALSE,
                                    FALSE, FALSE, FALSE, FALSE, FALSE,
                                    # This is still old ssarima
                                    TRUE, lagsModel, matrix(1,ncol=2), matrix(1,ncol=2));

            if(bounds=="a" & (componentsNumber > 0)){
                ARRoots <- abs(polyroot(elements$arPolynomial));
            }
            else{
                ARRoots <- 1;
            }
        }
    }

    if(MAGenerate){
        MARoots <- 0.5;
        while(any(MARoots<1)){
            MAValue <- runif(MANumber,-1,1);

            elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, componentsNumber,
                                    ARValue, MAValue, NULL, NULL,
                                    matvt, vecg, matF,
                                    "b", 0, matrix(1,obsStates,1), matrix(1,1,1), matrix(0,1,1),
                                    FALSE, FALSE, FALSE, FALSE,
                                    FALSE, FALSE, FALSE, FALSE, FALSE,
                                    # This is still old ssarima
                                    TRUE, lagsModel, matrix(1,ncol=2), matrix(1,ncol=2));

            if(bounds=="a" & (componentsNumber > 0)){
                MARoots <- abs(polyroot(elements$maPolynomial));
            }
            else{
                MARoots <- 1;
            }
        }
    }

    return(list(ARValue=ARValue,MAValue=MAValue));
}

##### Orders and lags for ssarima #####
    if(any(is.complex(c(ar.orders,i.orders,ma.orders,lags)))){
        stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
    }

    if(any(c(ar.orders,i.orders,ma.orders)<0)){
        stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
    }

    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    if(length(lags)!=length(ar.orders) & length(lags)!=length(i.orders) & length(lags)!=length(ma.orders)){
        stop("Seasonal lags do not correspond to any element of SARIMA",call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        ar.orders <- ar.orders[lags!=0];
        i.orders <- i.orders[lags!=0];
        ma.orders <- ma.orders[lags!=0];
        lags <- lags[lags!=0];
    }

    # Define maxorder and make all the values look similar (for the polynomials)
    maxorder <- max(length(ar.orders),length(i.orders),length(ma.orders));
    if(length(ar.orders)!=maxorder){
        ar.orders <- c(ar.orders,rep(0,maxorder-length(ar.orders)));
    }
    if(length(i.orders)!=maxorder){
        i.orders <- c(i.orders,rep(0,maxorder-length(i.orders)));
    }
    if(length(ma.orders)!=maxorder){
        ma.orders <- c(ma.orders,rep(0,maxorder-length(ma.orders)));
    }

    # If zeroes are defined for some orders, drop them.
    if(any((ar.orders + i.orders + ma.orders)==0)){
        orders2leave <- (ar.orders + i.orders + ma.orders)!=0;
        if(all(orders2leave==FALSE)){
            orders2leave <- lags==min(lags);
        }
        ar.orders <- ar.orders[orders2leave];
        i.orders <- i.orders[orders2leave];
        ma.orders <- ma.orders[orders2leave];
        lags <- lags[orders2leave];
    }

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        if(frequency!=1){
            warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),
                           "). Getting rid of some of them."),call.=FALSE);
        }
        lags.new <- unique(lags);
        ar.orders.new <- i.orders.new <- ma.orders.new <- lags.new;
        for(i in 1:length(lags.new)){
            ar.orders.new[i] <- max(ar.orders[which(lags==lags.new[i])]);
            i.orders.new[i] <- max(i.orders[which(lags==lags.new[i])]);
            ma.orders.new[i] <- max(ma.orders[which(lags==lags.new[i])]);
        }
        ar.orders <- ar.orders.new;
        i.orders <- i.orders.new;
        ma.orders <- ma.orders.new;
        lags <- lags.new;
    }

    ARValue <- AR;
    # Check the provided AR matrix / vector
    if(!is.null(ARValue)){
        if((!is.numeric(ARValue) | !is.vector(ARValue)) & !is.matrix(ARValue)){
            warning(paste0("AR should be either vector or matrix. You have provided something strange...\n",
                           "AR will be generated."),call.=FALSE);
            ARRequired <- ARGenerate <- TRUE;
            ARValue <- NULL;
        }
        else{
            if(sum(ar.orders)!=length(ARValue[ARValue!=0])){
                warning(paste0("Wrong number of non-zero elements of AR. Should be ",sum(ar.orders),
                               " instead of ",length(ARValue[ARValue!=0]),".\n",
                               "AR will be generated."),call.=FALSE);
                ARRequired <- ARGenerate <- TRUE;
                ARValue <- NULL;
            }
            else{
                if(all(ar.orders==0)){
                    ARValue <- NULL;
                    ARRequired <- ARGenerate <- FALSE;
                }
                else{
                    ARValue <- as.vector(ARValue[ARValue!=0]);
                    ARGenerate <- FALSE;
                    ARRequired <- TRUE;
                }
            }
        }
    }
    else{
        if(all(ar.orders==0)){
            ARRequired <- ARGenerate <- FALSE;
        }
        else{
            ARRequired <- ARGenerate <- TRUE;
        }
    }
    ARNumber <- sum(ar.orders);

    MAValue <- MA;
    # Check the provided MA matrix / vector
    if(!is.null(MAValue)){
        if((!is.numeric(MAValue) | !is.vector(MAValue)) & !is.matrix(MAValue)){
            warning(paste0("MA should be either vector or matrix. You have provided something strange...\n",
                           "MA will be generated."),call.=FALSE);
            MARequired <- MAGenerate <- TRUE;
            MAValue <- NULL;
        }
        else{
            if(sum(ma.orders)!=length(MAValue[MAValue!=0])){
                warning(paste0("Wrong number of non-zero elements of MA. Should be ",sum(ma.orders),
                               " instead of ",length(MAValue[MAValue!=0]),".\n",
                               "MA will be generated."),call.=FALSE);
                MARequired <- MAGenerate <- TRUE;
                MAValue <- NULL;
            }
            else{
                if(all(ma.orders==0)){
                    MAValue <- NULL;
                    MARequired <- MAGenerate <- FALSE;
                }
                else{
                    MAValue <- as.vector(MAValue[MAValue!=0]);
                    MAGenerate <- FALSE;
                    MARequired <- TRUE;
                }
            }
        }
    }
    else{
        if(all(ma.orders==0)){
            MARequired <- MAGenerate <- FALSE;
        }
        else{
            MARequired <- MAGenerate <- TRUE;
        }
    }
    MANumber <- sum(ma.orders);

#### Constant ####
    # Check the provided constant
    if(is.numeric(constant)){
        constantGenerate <- FALSE;
        constantRequired <- TRUE;
        constantValue <- constant;
    }
    else if(is.logical(constant)){
        constantRequired <- constantGenerate <- constant;
        constantValue <- NULL;
    }

#### Number of components and observations ####
    # Number of components to use
    componentsNumber <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
    componentsNames <- paste0("Component ",1:(componentsNumber+constantRequired));
    lagsModel <- matrix(rep(1,times=componentsNumber),ncol=1);
    if(constantRequired){
        lagsModel <- rbind(lagsModel,1);
    }
    lagsModelMax <- 1;

#### Initials ####
    initialValue <- initial;
    initialGenerate <- FALSE;
    if(!is.null(initialValue)){
        if(!is.numeric(initialValue)){
            warning(paste0("Initial vector is not numeric!\n",
                           "Initial values will be generated."),call.=FALSE);
            initialValue <- NULL;
            initialGenerate <- TRUE;
        }
        else{
            if(length(initialValue) != componentsNumber){
                warning(paste0("Wrong length of initial vector. Should be ",componentsNumber,
                               " instead of ",length(initialValue),".\n",
                               "Initial values will be generated."),call.=FALSE);
                initialValue <- NULL;
                initialGenerate <- TRUE;
            }
        }
    }
    else{
        initialGenerate <- TRUE;
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
    }

# In the case of wrong nsim, make it natural number. The same is for obs and frequency.
    nsim <- abs(round(nsim,0));
    obs <- abs(round(obs,0));
    obsStates <- obs + 1;
    frequency <- abs(round(frequency,0));

    if(initialGenerate){
        burnInPeriod <- max(lags);
        obs <- obs + burnInPeriod;
        obsStates <- obsStates + burnInPeriod;
    }

    if((componentsNumber==0) & !constantRequired){
        warning("You have not defined any model. So here's series generated from your distribution.", call.=FALSE);
        matyt <- materrors <- matrix(NA,obs,nsim);
        ellipsis$n <- nsim*obs;
        materrors[,] <- do.call(randomizer,ellipsis);

        matot <- matrix(NA,obs,nsim);
        # Generate values for occurence variable
        if(all(probability == 1)){
            matot[,] <- 1;
        }
        else{
            matot[,] <- rbinom(obs*nsim,1,probability);
        }

        matot <- ts(matot,frequency=frequency);
        materrors <- ts(materrors,frequency=frequency);
        matyt <- materrors;

        veclikelihood <- -obs/2 *(log(2*pi*exp(1)) + log(colMeans(materrors^2)));
        modelname <- "ARIMA(0,0,0)";
        model <- list(model=modelname,
                      AR=NULL, MA=NULL, constant=NA, initial=NULL,
                      data=matyt, states=NULL, residuals=materrors,
                      occurrence=matot, likelihood=veclikelihood);
        return(structure(model,class="smooth.sim"));
    }

##### Preset values of matvt and other matrices and arrays ######
    if(componentsNumber > 0){
# Transition matrix, measurement vector and persistence vector + state vector
        matF <- rbind(cbind(rep(0,componentsNumber-1),diag(componentsNumber-1)),rep(0,componentsNumber));
        matw <- matrix(c(1,rep(0,componentsNumber-1)),1,componentsNumber);
        if(constantRequired){
            matF <- cbind(rbind(matF,rep(0,componentsNumber)),c(1,rep(0,componentsNumber-1),1));
            matw <- cbind(matw,0);
        }
    }
    else{
        matw <- matF <- matrix(1,1,1);
    }

    persistenceLength <- componentsNumber + constantRequired;

# Define arrays
    arrvt <- array(NA,c(obsStates,persistenceLength,nsim),dimnames=list(NULL,componentsNames,NULL));
    arrF <- array(0,c(dim(matF),nsim));
    matg <- matrix(0,persistenceLength,nsim);

    materrors <- matrix(NA,obs,nsim);
    matyt <- matrix(NA,obs,nsim);
    matot <- matrix(NA,obs,nsim);
    matARValue <- matrix(NA,max(1,ARNumber),nsim);
    matMAValue <- matrix(NA,max(1,MANumber),nsim);
    vecConstantValue <- rep(NA,nsim);
    matInitialValue <- matrix(NA,componentsNumber,nsim);

    orderPlaceholder <- rep(0,length(ar.orders));
#### Generate stuff if needed ####
    if(componentsNumber>0){
        if(initialGenerate){
            matInitialValue[1:componentsNumber,] <- runif(componentsNumber*nsim,0,1000);
            arrvt[1:componentsNumber,1,] <- matInitialValue[1:componentsNumber,];
        }
        else{
            matInitialValue[1:componentsNumber,] <- rep(initialValue,nsim);
            arrvt[1,1:componentsNumber,] <- matInitialValue[1:componentsNumber,];
        }
    }

    if(ARRequired){
        if(ARGenerate){
            for(i in 1:nsim){
                elements <- elementsGenerator(ar.orders=ar.orders, ma.orders=orderPlaceholder, i.orders=orderPlaceholder,
                                              ARValue=NULL, MAValue=NULL,
                                              ARGenerate=TRUE, MAGenerate=FALSE);
                matARValue[,i] <- elements$ARValue;
            }
        }
        else{
            matARValue[,] <- ARValue;
        }
    }

    if(MARequired){
        if(MAGenerate){
            for(i in 1:nsim){
                elements <- elementsGenerator(ar.orders=orderPlaceholder, ma.orders=ma.orders, i.orders=orderPlaceholder,
                                              ARValue=NULL, MAValue=NULL,
                                              ARGenerate=FALSE, MAGenerate=TRUE);
                matMAValue[,i] <- elements$MAValue;
            }
        }
        else{
            matMAValue[,] <- MAValue;
        }
    }

    if(constantRequired){
        if(constantGenerate){
            if(any(i.orders>0)){
                vecConstantValue <- runif(nsim,-200,200);
            }
            else{
                vecConstantValue <- runif(nsim,100,1000);
            }
        }
        else{
            vecConstantValue[] <- constantValue;
        }
    }

    for(i in 1:nsim){
        elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, componentsNumber,
                                matARValue[,i], matMAValue[,i], vecConstantValue[i], NULL,
                                matrix(arrvt[,,i],obsStates), matrix(matg[,i],ncol=1), matF,
                                "b", 0, matrix(1,obsStates,1), matrix(1,1,1), matrix(0,1,1),
                                FALSE, FALSE, constantRequired, FALSE,
                                FALSE, FALSE, FALSE, FALSE, FALSE,
                                # This is still old ssarima
                                TRUE, lagsModel, matrix(1,ncol=2), matrix(1,ncol=2));

        arrF[,,i] <- elements$matF;
        matg[,i] <- elements$vecg;

# A correction in order to make sense out of generated initial components
        if(initialGenerate){
            arrvt[,,i] <- elements$matvt;
            arrvt[1,,i] <- matrixPowerWrap(as.matrix(arrF[,,i]),componentsNumber+1) %*% arrvt[1,,i];
        }

        if(constantRequired){
            arrvt[1,persistenceLength,i] <- elements$matvt[1,persistenceLength];
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
            materrors[,] <- rt(nsim*obs,obs-(persistenceLength + lagsModelMax));
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

    if(constantRequired){
        dimnames(arrvt)[[2]][persistenceLength] <- "Constant";
    }

    if(initialGenerate){
        if(constantRequired){
            matInitialValue[,] <- arrvt[burnInPeriod+1,-persistenceLength,];
        }
        else{
            matInitialValue[,] <- arrvt[burnInPeriod+1,,];
        }
        arrvtDim <- dim(arrvt);
        arrvtDim[1] <- arrvtDim[1] - burnInPeriod;
        arrvt <- array(arrvt[-c(1:burnInPeriod),,],arrvtDim);
        materrors <- materrors[-c(1:burnInPeriod),];
        matyt <- matyt[-c(1:burnInPeriod),];
        matot <- matot[-c(1:burnInPeriod),];
    }

    if(nsim==1){
        matyt <- ts(matyt,frequency=frequency);
        materrors <- ts(materrors,frequency=frequency);
        arrvt <- ts(arrvt[,,1],frequency=frequency,start=c(0,frequency-lagsModelMax+1));
        matot <- ts(matot,frequency=frequency);
    }
    else{
        matyt <- ts(matyt,frequency=frequency);
        materrors <- ts(materrors,frequency=frequency);
        matot <- ts(matot,frequency=frequency);
    }

# Give model the name
    if((length(ar.orders)==1) && all(lags==1)){
        modelname <- paste0("ARIMA(",ar.orders,",",i.orders,",",ma.orders,")");
    }
    else{
        modelname <- "";
        for(i in 1:length(ar.orders)){
            modelname <- paste0(modelname,"(",ar.orders[i],",");
            modelname <- paste0(modelname,i.orders[i],",");
            modelname <- paste0(modelname,ma.orders[i],")[",lags[i],"]");
        }
        modelname <- paste0("SARIMA",modelname);
    }
    if(any(probability!=1)){
        modelname <- paste0("i",modelname);
    }

    if(constantRequired){
        if(all(i.orders==0)){
            modelname <- paste0(modelname," with constant");
        }
        else{
            modelname <- paste0(modelname," with drift");
        }
        names(vecConstantValue) <- rep("Constant",length(vecConstantValue));
    }
    else{
        const <- FALSE;
        constantValue <- NULL;
    }

    model <- list(model=modelname,
                  AR=matARValue, MA=matMAValue, constant=vecConstantValue, initial=matInitialValue,
                  data=matyt, states=arrvt, residuals=materrors,
                  occurrence=matot, logLik=veclikelihood);
    return(structure(model,class="smooth.sim"));
}
