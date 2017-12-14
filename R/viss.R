viss <- function(data, intermittent=c("none","fixed","logistic"),
                 ic=c("AICc","AIC","BIC"), h=10, holdout=FALSE,
                 probability=c("independent","dependent"),
                 model="ANN", persistence=NULL, transition=NULL, phi=NULL,
                 initial=NULL, initialSeason=NULL, xreg=NULL){
# Function returns intermittent State-Space model
# probability="i" - assume that ot[,1] is independent from ot[,2], but has similar dynamics;
# probability="d" - assume that ot[,1] and ot[,2] are dependent, so that sum(P)=1;
    intermittent <- substring(intermittent[1],1,1);
    if(all(intermittent!=c("n","f","l"))){
        warning(paste0("Unknown value of intermittent provided: '",intermittent,"'."));
        intermittent <- "f";
    }

    ic <- ic[1];

    if(is.null(probability)){
        warning("probability value is not specified. Switching to 'independent'.");
        probability <- "i";
    }
    else{
        probability <- substr(probability[1],1,1);
    }

    # There's no difference in probabilities when intermittent=="f" or "n". So use simpler one.
    if((intermittent!="l") & probability=="d"){
        probability <- "i";
    }

    if(is.null(persistence)){
        if(probability=="d"){
            persistence <- "g";
        }
    }
    if(is.null(transition)){
        if(probability=="d"){
            transition <- "g";
        }
    }
    if(is.null(phi)){
        if(probability=="d"){
            phi <- "g";
        }
    }
    if(!is.null(initial)){
        # If a numeric is provided in initial, check it
        if(is.numeric(initial)){
            if((probability=="i" & length(initial)!=nSeries) |
               (probability=="d" & length(initial)!=2^nSeries)){
                warning("Wrong length of the initial vector");
                initial <- NULL;
                initialIsNumeric <- FALSE;
            }
            initialIsNumeric <- TRUE;
        }
        else{
            initialIsNumeric <- FALSE;
        }
    }
    else{
        if(probability=="d"){
            initial <- "i";
            initialIsNumeric <- FALSE;
        }
    }
    if(is.null(initialSeason)){
        if(probability=="d"){
            initialSeason <- "g";
        }
    }

    if(is.data.frame(data)){
        data <- as.matrix(data);
    }

    # Number of series in the matrix
    nSeries <- ncol(data);

    if(is.null(ncol(data))){
        stop("The provided data is not a matrix! Use iss() function instead!", call.=FALSE);
    }
    if(ncol(data)==1){
        stop("The provided data contains only one column. Use iss() function instead!", call.=FALSE);
    }
    # Check the data for NAs
    if(any(is.na(data))){
        if(!silentText){
            warning("Data contains NAs. These observations will be substituted by zeroes.", call.=FALSE);
        }
        data[is.na(data)] <- 0;
    }

    # Define obs, the number of observations of in-sample
    obsInSample <- nrow(data) - holdout*h;

    # Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- nrow(data) + (1 - holdout)*h;

    # If obsInSample is negative, this means that we can't do anything...
    if(obsInSample<=2){
        stop("Not enough observations in sample.", call.=FALSE);
    }
    # Define the actual values.
    y <- matrix(data[1:obsInSample,],obsInSample,nSeries);
    dataFreq <- frequency(data);
    dataDeltat <- deltat(data);
    dataStart <- start(data);

    ot <- (y!=0)*1;
    otAll <- (data!=0)*1;
    obsOnes <- apply(ot,2,sum);

    pFitted <- matrix(NA,obsInSample,nSeries);
    pForecast <- matrix(NA,h,nSeries);

#### Fixed probability ####
    if(intermittent=="f"){
        if(!initialIsNumeric){
            pFitted[,] <- rep(apply(ot,2,mean),each=obsInSample);
            pForecast[,] <- rep(pFitted[obsInSample,],each=h);
            initial <- pFitted[1,];
        }
        else{
            pFitted[,] <- rep(initial,each=obsInSample);
            pForecast[,] <- rep(initial,each=h);
        }
        states <- rbind(pFitted,pForecast);
        logLik <- structure((sum(log(pFitted[ot==1])) + sum(log((1-pFitted[ot==0])))),df=nSeries,class="logLik");

        states <- ts(states, start=dataStart, frequency=dataFreq);
        errors <- ts(ot-pFitted, start=dataStart, frequency=dataFreq);
        pFitted <- ts(pFitted, start=dataStart, frequency=dataFreq);
        pForecast <- ts(pForecast, start=time(data)[obsInSample] + dataDeltat, frequency=dataFreq);

        output <- list(model=model, fitted=pFitted, forecast=pForecast, states=states,
                       variance=pForecast*(1-pForecast), logLik=logLik, nParam=nSeries,
                       residuals=errors, actuals=otAll, persistence=NULL, initial=initial);
    }
#### Logistic probability ####
    else if(intermittent=="l"){
        if(probability=="i"){
            print("as")
            issModel <- list(NA);
            states <- rep(NA,obsAll);
            logLik <- 0;
            errors <- matrix(NA,obsInSample,nSeries);
            initial <- NA;
            for(i in 1:nSeries){
                issModel <- iss(ot[,i],intermittent=intermittent,ic=ic,h=h,model=model,persistence=persistence,
                                     initial=initial,initialSeason=initialSeason,xreg=xreg,holdout=holdout);
                pFitted[,i] <- issModel$fitted;
                pForecast[,i] <- issModel$forecast;
                states <- cbind(states,issModel$states);
                errors[,i] <- issModel$residuals;
                #### This needs to be modified ####
                logLik <- logLik + logLik(issModel);
                ####
                initial <- rbind(initial,issModel$initial);
            }
            states <- states[,-1];
            initial <- initial[-1,];
            nParam <- issModel$nParam;
            model <- issModel$model;
        }
        else{
            # This matrix contains all the possible outcomes for probabilities
            otOutcomes <- matrix(0,2^nSeries,nSeries);
            otFull <- matrix(NA,obsInSample,2^nSeries);
            for(i in 1:(2^nSeries)){
                otOutcomes[i,] <- rev(as.integer(intToBits(i-1))[1:nSeries]);
                otFull[,i] <- apply(ot==matrix(otOutcomes[i,],obsInSample,nSeries,byrow=T),1,all)*1;
            }

            issModel <- ves(otFull,model=model,persistence=persistence,transition=transition,phi=phi,
                            initial=initial,initialSeason=initialSeason,ic=ic,h=h,xreg=xreg,holdout=holdout)

            states <- issModel$states;
            initial <- issModel$initial;
            errors <- issModel$residuals;
            pFitted[,] <- matrix(issModel$fitted %*% otOutcomes,obsInSample,nSeries,byrow=T);
            pForecast[,] <- matrix(issModel$forecast %*% otOutcomes,h,nSeries,byrow=T);

            nParam <- issModel$nParam;
            model <- modelType(issModel);
            logLik <- issModel$logLik
        }

        output <- list(model=model, fitted=pFitted, forecast=pForecast, states=states,
                       variance=pForecast*(1-pForecast), logLik=logLik, nParam=nSeries,
                       residuals=errors, actuals=otAll, persistence=NULL, initial=initial);
    }

    return(structure(output,class="viss"));
}
