utils::globalVariables(c("y","obs","occurrenceModelProvided","occurrenceModel","occurrenceModel"))

intermittentParametersSetter <- function(occurrence="n",...){
# Function returns basic parameters based on occurrence type
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    if(all(occurrence!=c("n","p"))){
        ot <- (y!=0)*1;
        obsNonzero <- sum(ot);
        obsZero <- obsInsample - obsNonzero;
        # 1 parameter for estimating initial probability. Works for the fixed probability model
        nParamOccurrence <- 1;
        if(any(occurrence==c("o","i","d"))){
            # The minimum number of parameters for these models is 2: level, alpha
            nParamOccurrence <- nParamOccurrence + 1;
        }
        else if(any(occurrence==c("g","a"))){
            # In "general" and "auto" the max number is 4
            nParamOccurrence <- nParamOccurrence + 3;
        }
        # Demand sizes
        yot <- matrix(y[y!=0],obsNonzero,1);
        if(!occurrenceModelProvided){
            pFitted <- matrix(mean(ot),obsInsample,1);
            pForecast <- matrix(1,h,1);
        }
        else{
            if(length(fitted(occurrenceModel))>obsInsample){
                pFitted <- matrix(fitted(occurrenceModel)[1:obsInsample],obsInsample,1);
            }
            else if(length(fitted(occurrenceModel))<obsInsample){
                pFitted <- matrix(c(fitted(occurrenceModel),
                               rep(fitted(occurrenceModel)[length(fitted(occurrenceModel))],obsInsample-length(fitted(occurrenceModel)))),
                             obsInsample,1);
            }
            else{
                pFitted <- matrix(fitted(occurrenceModel),obsInsample,1);
            }

            if(length(occurrenceModel$forecast)>=h){
                pForecast <- matrix(occurrenceModel$forecast[1:h],h,1);
            }
            else{
                pForecast <- matrix(c(occurrenceModel$forecast,
                                   rep(occurrenceModel$forecast[1],h-length(occurrenceModel$forecast))),h,1);
            }

        }
    }
    else{
        obsNonzero <- obsInsample;
        obsZero <- 0;
    }

    if(occurrence=="n"){
        ot <- rep(1,obsInsample);
        obsNonzero <- obsInsample;
        yot <- y;
        pFitted <- matrix(1,obsInsample,1);
        pForecast <- matrix(1,h,1);
        nParamOccurrence <- 0;
    }
    ot <- ts(ot,start=dataStart,frequency=dataFreq);

    assign("ot",ot,ParentEnvironment);
    assign("obsNonzero",obsNonzero,ParentEnvironment);
    assign("obsZero",obsZero,ParentEnvironment);
    assign("yot",yot,ParentEnvironment);
    assign("pFitted",pFitted,ParentEnvironment);
    assign("pForecast",pForecast,ParentEnvironment);
    assign("nParamOccurrence",nParamOccurrence,ParentEnvironment);
}

intermittentMaker <- function(occurrence="n",...){
# Function returns all the necessary stuff from occurrence models
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

##### If occurrence is not absent or provided, then work normally #####
    if(all(occurrence!=c("n","p"))){
        if(!occurrenceModelProvided){
            occurrenceModel <- oes(ot, model=occurrenceModel, occurrence=occurrence, h=h);
        }
        else{
            occurrenceModel <- oes(ot, model=occurrenceModel, h=h);
        }
        nParamOccurrence <- nparam(occurrenceModel);
        pFitted[,] <- fitted(occurrenceModel);
        pForecast <- occurrenceModel$forecast;
        occurrence <- occurrenceModel$occurrence;
    }
    else{
        occurrenceModel <- NULL;
        nParamOccurrence <- 0;
    }

    assign("occurrence",occurrence,ParentEnvironment);
    assign("pFitted",pFitted,ParentEnvironment);
    assign("pForecast",pForecast,ParentEnvironment);
    assign("nParamOccurrence",nParamOccurrence,ParentEnvironment);
    assign("occurrenceModel",occurrenceModel,ParentEnvironment);
}



#' Intermittent State Space
#'
#' Function calculates the probability for intermittent state space model. This
#' is needed in order to forecast intermittent demand using other functions.
#'
#' The function estimates probability of demand occurrence, using one of the ETS
#' state space models.
#'
#' @template ssIntermittentRef
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param data Either numeric vector or time series vector.
#' @param intermittent Type of method used in probability estimation. Can be
#' \code{"none"} - none, \code{"fixed"} - constant probability,
#' \code{"croston"} - estimated using Croston, 1972 method and \code{"TSB"} -
#' Teunter et al., 2011 method., \code{"sba"} - Syntetos-Boylan Approximation
#' for Croston's method (bias correction) discussed in Syntetos and Boylan,
#' 2005, \code{"logistic"} - probability based on logit model.
#' @param ic Information criteria to use in case of model selection.
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, holdout sample of size \code{h} is taken from
#' the end of the data.
#' @param model Type of ETS model used for the estimation. Normally this should
#' be either \code{"ANN"} or \code{"MNN"}.
#' @param persistence Persistence vector. If \code{NULL}, then it is estimated.
#' @param initial Initial vector. If \code{NULL}, then it is estimated.
#' @param initialSeason Initial vector of seasonal components. If \code{NULL},
#' then it is estimated.
#' @param xreg Vector of matrix of exogenous variables, explaining some parts
#' of occurrence variable (probability).
#' @return The object of class "iss" is returned. It contains following list of
#' values:
#'
#' \itemize{
#' \item \code{model} - the type of the estimated ETS model;
#' \item \code{fitted} - fitted values of the constructed model;
#' \item \code{forecast} - forecast for \code{h} observations ahead;
#' \item \code{states} - values of states (currently level only);
#' \item \code{variance} - conditional variance of the forecast;
#' \item \code{logLik} - likelihood value for the model
#' \item \code{nParam} - number of parameters used in the model;
#' \item \code{residuals} - residuals of the model;
#' \item \code{actuals} - actual values of probabilities (zeros and ones).
#' \item \code{persistence} - the vector of smoothing parameters;
#' \item \code{initial} - initial values of the state vector;
#' \item \code{initialSeason} - the matrix of initials seasonal states;
#' }
#' @seealso \code{\link[forecast]{ets}, \link[forecast]{forecast},
#' \link[smooth]{es}}
#' @examples
#'
#'     y <- rpois(100,0.1)
#'     iss(y, intermittent="p")
#'
#'     iss(y, intermittent="i", persistence=0.1)
#'
#' @export iss
iss <- function(data, intermittent=c("none","fixed","interval","probability","sba","logistic"),
                ic=c("AICc","AIC","BIC","BICc"), h=10, holdout=FALSE,
                model=NULL, persistence=NULL, initial=NULL, initialSeason=NULL, xreg=NULL){
# Function returns intermittent state space model
#### Add initialSeason to the output? ####
    intermittent <- substring(intermittent[1],1,1);
    if(all(intermittent!=c("n","f","i","p","s","l"))){
        warning(paste0("Unknown value of intermittent provided: '",intermittent,"'."));
        intermittent <- "f";
    }
    if(intermittent=="s"){
        intermittent <- "i";
        sbaCorrection <- TRUE;
    }
    else{
        sbaCorrection <- FALSE;
    }

    ic <- ic[1];

    if(is.smooth.sim(data)){
        data <- data$data;
    }

    obsInsample <- length(data) - holdout*h;
    obsAll <- length(data) + (1 - holdout)*h;
    y <- ts(data[1:obsInsample],frequency=frequency(data),start=start(data));

    ot <- abs((y!=0)*1);
    otAll <- abs((data!=0)*1);
    iprob <- mean(ot);
    obsOnes <- sum(ot);

    if(!is.null(model)){
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
    }
    else{
        model <- "MNN";
        Etype <- "M";
        Ttype <- "N";
        Stype <- "N";
    }

    if(Stype!="N" & intermittent!="l"){
        Stype <- "N";
        substr(model,nchar(model),nchar(model)) <- "N";
        warning("Sorry, but we do not deal with seasonal models in iss yet.",call.=FALSE);
    }

    if(var(ot)==0){
        warning(paste0("There is no variability in the occurrence of the variable in-sample.\n",
                       "Switching to intermittent='none'."),call.=FALSE)
        intermittent <- "n";
    }

#### Fixed probability ####
    if(intermittent=="f"){
        if(!is.null(initial)){
            pFitted <- ts(matrix(rep(initial,obsInsample),obsInsample,1), start=start(y), frequency=frequency(y));
        }
        else{
            initial <- iprob;
            pFitted <- ts(matrix(rep(iprob,obsInsample),obsInsample,1), start=start(y), frequency=frequency(y));
        }
        names(initial) <- "level";
        pForecast <- ts(rep(pFitted[1],h), start=time(y)[obsInsample]+deltat(y), frequency=frequency(y));
        errors <- ts(ot-iprob, start=start(y), frequency=frequency(y));

        output <- list(model=model, fitted=pFitted, forecast=pForecast, states=pFitted,
                       variance=pForecast*(1-pForecast), logLik=NA, nParam=1,
                       residuals=errors, actuals=otAll,
                       persistence=NULL, initial=initial, initialSeason=NULL);
    }
#### Croston's method ####
    else if(intermittent=="i"){
        if(is.null(initial)){
            initial <- "o";
        }
# Define the matrix of states
        ivt <- matrix(rep(iprob,obsInsample+1),obsInsample+1,1);
# Define the matrix of actuals as intervals between demands
        # zeroes <- c(0,which(y!=0),obsInsample+1);
        zeroes <- c(0,which(y!=0));
### With this thing we fit model of the type 1/(1+qt)
#        zeroes <- diff(zeroes)-1;
        zeroes <- diff(zeroes);
# Number of intervals in Croston
        iyt <- matrix(zeroes,length(zeroes),1);
        newh <- which(y!=0);
        newh <- newh[length(newh)];
        newh <- obsInsample - newh + h;
        crostonModel <- es(iyt,model=model,silent=TRUE,h=newh,
                           persistence=persistence,initial=initial,
                           ic=ic,xreg=xreg,initialSeason=initialSeason);

        pFitted <- rep((crostonModel$fitted),zeroes);
        if(any(pFitted<1)){
            pFitted[pFitted<1] <- 1;
        }
        tailNumber <- obsInsample - length(pFitted);
        if(tailNumber>0){
            pForecast <- crostonModel$forecast[1:tailNumber];
            if(any(pForecast<1)){
                pForecast[pForecast<1] <- 1;
            }
            pFitted <- c(pFitted,pForecast);
        }
        pForecast <- crostonModel$forecast[(tailNumber+1):newh];
        if(any(pForecast<1)){
            pForecast[pForecast<1] <- 1;
        }

        if(sbaCorrection){
            pFitted <- ts((1-sum(crostonModel$persistence)/2)/pFitted,start=start(y),frequency=frequency(y));
            pForecast <- ts((1-sum(crostonModel$persistence)/2)/pForecast, start=time(y)[obsInsample]+deltat(y),frequency=frequency(y));
            states <- 1/crostonModel$states;
            intermittent <- "s";
        }
        else{
            pFitted <- ts(1/pFitted,start=start(y),frequency=frequency(y));
            pForecast <- ts(1/pForecast, start=time(y)[obsInsample]+deltat(y),frequency=frequency(y));
            states <- 1/crostonModel$states;
        }

        output <- list(model=model, fitted=pFitted, forecast=pForecast, states=states,
                       variance=pForecast*(1-pForecast), logLik=NA, nParam=nparam(crostonModel),
                       residuals=crostonModel$residuals, actuals=otAll,
                       persistence=crostonModel$persistence, initial=crostonModel$initial,
                       initialSeason=crostonModel$initialSeason);
    }
#### TSB method ####
    else if(intermittent=="p"){
        if(is.null(model)){
            model <- "YYN";
        }
        if(is.null(initial)){
            initial <- "o";
        }

        iyt <- matrix(ot,obsInsample,1);
        iyt <- ts(iyt,frequency=frequency(data));

        kappa <- 1E-5;
        iy_kappa <- ts(iyt*(1 - 2*kappa) + kappa,start=start(y),frequency=frequency(y));

        tsbModel <- es(iy_kappa,model,persistence=persistence,initial=initial,
                       ic=ic,silent=TRUE,h=h,cfType="TSB",xreg=xreg,
                       initialSeason=initialSeason);

        # Correction so we can return from those iy_kappa values
        tsbModel$fitted <- (tsbModel$fitted - kappa) / (1 - 2*kappa);
        tsbModel$forecast <- (tsbModel$forecast - kappa) / (1 - 2*kappa);

        # If bt>1, then at = 0 and pFitted = bt / (at + bt) = 1
        if(any(tsbModel$fitted>1)){
            tsbModel$fitted[tsbModel$fitted>1] <- 1;
        }
        if(any(tsbModel$forecast>1)){
            tsbModel$forecast[tsbModel$forecast>1] <- 1;
        }

        # If at>1, then bt = 0 and pFitted = bt / (at + bt) = 0
        if(any(tsbModel$fitted<0)){
            tsbModel$fitted[tsbModel$fitted<0] <- 0;
        }
        if(any(tsbModel$forecast<0)){
            tsbModel$forecast[tsbModel$forecast<0] <- 0;
        }

        output <- list(model=model, fitted=tsbModel$fitted, forecast=tsbModel$forecast, states=tsbModel$states,
                       variance=tsbModel$forecast*(1-tsbModel$forecast), logLik=NA, nParam=nparam(tsbModel)-1,
                       residuals=tsbModel$residuals, actuals=otAll,
                       persistence=tsbModel$persistence, initial=tsbModel$initial,
                       initialSeason=tsbModel$initialSeason);
    }
#### Logistic ####
    else if(intermittent=="l"){
        if(is.null(model)){
            model <- "XXX";
        }
        if(is.null(initial)){
            initial <- "o";
        }

        # If the underlying model is pure multiplicative, use error "L", otherwise use "D"
        if(all(c(substr(model,1,1)!="A", substr(model,2,2)!="A"),
                 substr(model,nchar(model),nchar(model))!="A") &
           all(c(substr(model,1,1)!="X", substr(model,2,2)!="X"),
                 substr(model,nchar(model),nchar(model))!="X") &
           all(c(substr(model,1,1)!="Z", substr(model,2,2)!="Z"),
                 substr(model,nchar(model),nchar(model))!="Z")){
            cfType <- "LogisticL";
        }
        else if(all(c(substr(model,1,1)!="Z", substr(model,2,2)!="Z"),
                 substr(model,nchar(model),nchar(model))!="Z")){
            cfType <- "LogisticD";
        }
        else{
            cfType <- "LogisticZ";
        }
        ##### Need to introduce also the one with ZZZ #####

        iyt <- ts(matrix(ot,obsInsample,1),start=start(y),frequency=frequency(y));

        if(cfType=="LogisticZ"){
            logisticModel <- list(NA);

            cfType <- "LogisticD";
            modelNew <- gsub("Z","X",model);
            logisticModel[[1]] <- es(iyt,modelNew,persistence=persistence,initial=initial,
                                     ic=ic,silent=TRUE,h=h,cfType=cfType,xreg=xreg,
                                     initialSeason=initialSeason);

            cfType <- "LogisticL";
            modelNew <- gsub("Z","Y",model);
            logisticModel[[2]] <- es(iyt,modelNew,persistence=persistence,initial=initial,
                                     ic=ic,silent=TRUE,h=h,cfType=cfType,xreg=xreg,
                                     initialSeason=initialSeason);

            if(logisticModel[[1]]$ICs[nrow(logisticModel[[1]]$ICs),ic] <
               logisticModel[[2]]$ICs[nrow(logisticModel[[2]]$ICs),ic]){
                logisticModel <- logisticModel[[1]];
            }
            else{
                logisticModel <- logisticModel[[2]];
            }
        }
        else{
            logisticModel <- es(iyt,model,persistence=persistence,initial=initial,
                                ic=ic,silent=TRUE,h=h,cfType=cfType,xreg=xreg,
                                initialSeason=initialSeason);
        }

        output <- list(model=modelType(logisticModel), fitted=logisticModel$fitted, forecast=logisticModel$forecast, states=logisticModel$states,
                       variance=logisticModel$forecast*(1-logisticModel$forecast), logLik=NA, nParam=nparam(logisticModel),
                       residuals=logisticModel$residuals, actuals=otAll,
                       persistence=logisticModel$persistence, initial=logisticModel$initial,
                       initialSeason=logisticModel$initialSeason);
    }
#### None ####
    else{
        pFitted <- ts(y,start=start(y),frequency=frequency(y));
        pForecast <- ts(rep(y[obsInsample],h), start=time(y)[obsInsample]+deltat(y),frequency=frequency(y));
        errors <- ts(rep(0,obsInsample), start=start(y), frequency=frequency(y));
        output <- list(model=NULL, fitted=pFitted, forecast=pForecast, states=pFitted,
                       variance=rep(0,h), logLik=NA, nParam=0,
                       residuals=errors, actuals=pFitted,
                       persistence=NULL, initial=NULL, initialSeason=NULL);
    }
    output$intermittent <- intermittent;
    pFitted <- output$fitted;
    if(any(c(1-pFitted[ot==0]==0,pFitted[ot==1]==0))){
        # return(-Inf);
        ptNew <- pFitted[(pFitted!=0) & (pFitted!=1)];
        otNew <- ot[(pFitted!=0) & (pFitted!=1)];
        output$logLik <- sum(log(ptNew[otNew==1])) + sum(log(1-ptNew[otNew==0]));
    }
    else{
        output$logLik <- (sum(log(pFitted[ot!=0])) + sum(log(1-pFitted[ot==0])));
    }
    return(structure(output,class="iss"));
}
