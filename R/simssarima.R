sim.ssarima <- function(ar.orders=0, i.orders=1, ma.orders=1, lags=1,
                        frequency=1, AR=NULL, MA=NULL, constant=FALSE,
                        initial=NULL, bounds=c("admissible","none"),
                        obs=10, nsim=1, silent=FALSE,
                        randomizer=c("rnorm","runif","rbeta","rt"),
                        iprob=1, ...){
# Function generates data using SSARIMA in Single Source of Error as a data generating process.
#    Copyright (C) 2015 - 2016 Ivan Svetunkov

    bounds <- substring(bounds[1],1,1);
    randomizer <- randomizer[1];

#### Elements Generator for AR and MA ####
elementsGenerator <- function(ar.orders=ar.orders, ma.orders=ma.orders, i.orders=i.orders,
                              ARValue=ARValue, MAValue=MAValue,
                              #matvt, vecg, matF,
                              ARGenerate=FALSE, MAGenerate=FALSE){
    componentsNumber <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
    #if(!exists(matvt,inherits=FALSE)){
    matvt <- matrix(1,5,componentsNumber+constantRequired);
    #}
    #if(!exists(vecg,inherits=FALSE)){
    vecg <- matrix(0,componentsNumber+constantRequired,1);
    #}
    #if(!exists(matF,inherits=FALSE)){
    matF <- diag(componentsNumber+constantRequired);
    #}

    if(ARGenerate){
        ARRoots <- 0.5;
        while(any(ARRoots<1)){
            ARValue <- runif(ARNumber,-1,1);

            elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, componentsNumber,
                                    ARValue, MAValue, NULL, NULL,
                                    matvt, vecg, matF,
                                    "b", 0, matrix(1,obsStates,1), matrix(1,1,1), matrix(0,1,1),
                                    FALSE, FALSE, FALSE, FALSE,
                                    FALSE, FALSE, FALSE, FALSE, FALSE);

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
                                    FALSE, FALSE, FALSE, FALSE, FALSE);

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
        if(frequency(data)!=1){
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
                           "AR will be estimated."),call.=FALSE);
            ARRequired <- ARGenerate <- TRUE;
            ARValue <- NULL;
        }
        else{
            if(sum(ar.orders)!=length(ARValue[ARValue!=0])){
                warning(paste0("Wrong number of non-zero elements of AR. Should be ",sum(ar.orders),
                               " instead of ",length(ARValue[ARValue!=0]),".\n",
                               "AR will be estimated."),call.=FALSE);
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
                           "MA will be estimated."),call.=FALSE);
            MARequired <- MAGenerate <- TRUE;
            MAValue <- NULL;
        }
        else{
            if(sum(ma.orders)!=length(MAValue[MAValue!=0])){
                warning(paste0("Wrong number of non-zero elements of MA. Should be ",sum(ma.orders),
                               " instead of ",length(MAValue[MAValue!=0]),".\n",
                               "MA will be estimated."),call.=FALSE);
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
    modellags <- matrix(rep(1,times=componentsNumber),ncol=1);
    if(constantRequired){
        modellags <- rbind(modellags,1);
    }
    maxlag <- 1;

# In the case of wrong nsim, make it natural number. The same is for obs and frequency.
    nsim <- abs(round(nsim,0));
    obs <- abs(round(obs,0));
    obsStates <- obs + 1;
    frequency <- abs(round(frequency,0));

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
    matInitialValue <- matrix(NA,persistenceLength,nsim);

    orderPlaceholder <- rep(0,length(ar.orders));
#### Generate stuff if needed ####
    if(initialGenerate){
        matInitialValue[1:componentsNumber,] <- runif(componentsNumber*nsim,0,1000);
    }
    else{
        matInitialValue[1:componentsNumber,] <- rep(initialValue,nsim);
    }
    arrvt[1,,] <- matInitialValue;

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
                                FALSE, FALSE, FALSE, FALSE, FALSE);

        arrF[,,i] <- elements$matF;
        matg[,i] <- elements$vecg;
        arrvt[,,i] <- elements$matvt;

# A correction in order to make sense out of generated initial components
        if(initialGenerate){
            arrvt[1,,i] <- matrixPowerWrap(as.matrix(arrF[,,i]),componentsNumber+1) %*% arrvt[1,,i];
        }
    }

# If the chosen randomizer is not rnorm, rt and runif and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rlnorm","rt","runif")) & (any(names(match.call(expand.dots=FALSE)[-1]) == "...")==FALSE)){
        if(silent == FALSE){
            warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
        }
        randomizer = "rnorm";
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
            materrors[,] <- rt(nsim*obs,obs-(persistenceLength + maxlag));
        }

        if(randomizer!="rlnorm"){
# Center errors just in case
            materrors <- materrors - colMeans(materrors);
# Change variance to make some sense. Errors should not be rediculously high and not too low.
            materrors <- materrors * sqrt(abs(arrvt[1,1,]));
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

    simulateddata <- simulatorwrap(arrvt,materrors,matot,arrF,matw,matg,"A","N","N",modellags);

    if((iprob < 1) & (iprob > 0)){
        matyt <- round(simulateddata$matyt,0);
    }
    else{
        matyt <- simulateddata$matyt;
    }
    arrvt <- simulateddata$arrvt;
    dimnames(arrvt) <- list(NULL,componentsNames,NULL);

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
    if(iprob!=1){
        modelname <- paste0("i",modelname);
    }

    if(constantRequired){
        if(all(i.orders==0)){
            modelname <- paste0(modelname," with constant");
        }
        else{
            modelname <- paste0(modelname," with drift");
        }
    }
    else{
        const <- FALSE;
        constantValue <- NULL;
    }

    model <- list(model=modelname,
                  AR=matARValue, MA=matMAValue, constant=vecConstantValue, initial=matInitialValue,
                  data=matyt, states=arrvt, residuals=materrors,
                  occurrences=matot, likelihood=veclikelihood);
    return(structure(model,class="smooth.sim"));
}
