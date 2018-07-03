utils::globalVariables(c("y","obs","imodelProvided","intermittentModel","imodel"))

intermittentParametersSetter <- function(intermittent="n",...){
# Function returns basic parameters based on intermittent type
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    if(all(intermittent!=c("n","provided"))){
        ot <- (y!=0)*1;
        obsNonzero <- sum(ot);
        # 1 parameter for estimating initial probability
        nParamIntermittent <- 1;
        if(intermittent=="p"){
            # In TSB we only need to estimate smoothing parameter - we do not
            # estimate any parameters of the Beta distribution.
            nParamIntermittent <- nParamIntermittent + 1;
        }
        else if(any(intermittent==c("i","a"))){
            # In Croston we also need to estimate smoothing parameter and variance
            nParamIntermittent <- nParamIntermittent + 2;
        }
        yot <- matrix(y[y!=0],obsNonzero,1);
        if(!imodelProvided){
            pt <- matrix(mean(ot),obsInsample,1);
            pt.for <- matrix(1,h,1);
        }
        else{
            if(length(imodel$fitted)>obsInsample){
                pt <- matrix(imodel$fitted[1:obsInsample],obsInsample,1);
            }
            else if(length(imodel$fitted)<obsInsample){
                pt <- matrix(c(imodel$fitted,
                               rep(imodel$fitted[length(imodel$fitted)],obsInsample-length(imodel$fitted))),
                             obsInsample,1);
            }
            else{
                pt <- matrix(imodel$fitted,obsInsample,1);
            }

            if(length(imodel$forecast)>=h){
                pt.for <- matrix(imodel$forecast[1:h],h,1);
            }
            else{
                pt.for <- matrix(c(imodel$forecast,
                                   rep(imodel$forecast[1],h-length(imodel$forecast))),h,1);
            }

            iprob <- c(pt,pt.for);
        }
    }
    else{
        obsNonzero <- obsInsample;
    }

    if(all(intermittent!=c("n","l","p"))){
# If number of observations is low, set intermittency to "none"
        if(obsNonzero < 3){
            warning(paste0("Not enough non-zero observations for intermittent state space model. We need at least 5.\n",
                           "Changing intermittent to 'n'."),call.=FALSE);
            intermittent <- "n";
        }
    }

    if(intermittent=="n"){
        ot <- rep(1,obsInsample);
        obsNonzero <- obsInsample;
        yot <- y;
        pt <- matrix(1,obsInsample,1);
        pt.for <- matrix(1,h,1);
        nParamIntermittent <- 0;
    }
    iprob <- pt[1];
    ot <- ts(ot,start=dataStart,frequency=datafreq);

    assign("ot",ot,ParentEnvironment);
    assign("obsNonzero",obsNonzero,ParentEnvironment);
    assign("yot",yot,ParentEnvironment);
    assign("pt",pt,ParentEnvironment);
    assign("pt.for",pt.for,ParentEnvironment);
    assign("nParamIntermittent",nParamIntermittent,ParentEnvironment);
    assign("iprob",iprob,ParentEnvironment);
}

intermittentMaker <- function(intermittent="n",...){
# Function returns all the necessary stuff from intermittent models
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

##### If intermittent is not auto, then work normally #####
    if(all(intermittent!=c("n","provided","a"))){
        if(!imodelProvided){
            imodel <- iss(ot, model=intermittentModel, intermittent=intermittent, h=h);
        }
        else{
            imodel <- iss(ot, model=intermittentModel, intermittent=intermittent, h=h,
                          persistence=imodel$persistence, initial=imodel$initial);
        }
        nParamIntermittent <- imodel$nParam;
        pt[,] <- imodel$fitted;
        pt.for <- imodel$forecast;
        iprob <- pt.for[1];
    }
    else{
        imodel <- NULL;
        nParamIntermittent <- 0;
    }

    assign("pt",pt,ParentEnvironment);
    assign("pt.for",pt.for,ParentEnvironment);
    assign("iprob",iprob,ParentEnvironment);
    assign("nParamIntermittent",nParamIntermittent,ParentEnvironment);
    assign("imodel",imodel,ParentEnvironment);
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
#' @keywords iss intermittent demand intermittent demand state space model
#' exponential smoothing forecasting
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

    if(class(data)=="smooth.sim"){
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
            pt <- ts(matrix(rep(initial,obsInsample),obsInsample,1), start=start(y), frequency=frequency(y));
        }
        else{
            initial <- iprob;
            pt <- ts(matrix(rep(iprob,obsInsample),obsInsample,1), start=start(y), frequency=frequency(y));
        }
        pt.for <- ts(rep(pt[1],h), start=time(y)[obsInsample]+deltat(y), frequency=frequency(y));
        errors <- ts(ot-iprob, start=start(y), frequency=frequency(y));

        output <- list(model=model, fitted=pt, forecast=pt.for, states=pt,
                       variance=pt.for*(1-pt.for), logLik=NA, nParam=1,
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

        pt <- rep((crostonModel$fitted),zeroes);
        if(any(pt<1)){
            pt[pt<1] <- 1;
        }
        tailNumber <- obsInsample - length(pt);
        if(tailNumber>0){
            pt.for <- crostonModel$forecast[1:tailNumber];
            if(any(pt.for<1)){
                pt.for[pt.for<1] <- 1;
            }
            pt <- c(pt,pt.for);
        }
        pt.for <- crostonModel$forecast[(tailNumber+1):newh];
        if(any(pt.for<1)){
            pt.for[pt.for<1] <- 1;
        }

        if(sbaCorrection){
            pt <- ts((1-sum(crostonModel$persistence)/2)/pt,start=start(y),frequency=frequency(y));
            pt.for <- ts((1-sum(crostonModel$persistence)/2)/pt.for, start=time(y)[obsInsample]+deltat(y),frequency=frequency(y));
            states <- 1/crostonModel$states;
            intermittent <- "s";
        }
        else{
            pt <- ts(1/pt,start=start(y),frequency=frequency(y));
            pt.for <- ts(1/pt.for, start=time(y)[obsInsample]+deltat(y),frequency=frequency(y));
            states <- 1/crostonModel$states;
        }

        output <- list(model=model, fitted=pt, forecast=pt.for, states=states,
                       variance=pt.for*(1-pt.for), logLik=NA, nParam=nParam(crostonModel),
                       residuals=crostonModel$residuals, actuals=otAll,
                       persistence=crostonModel$persistence, initial=crostonModel$initial,
                       initialSeason=crostonModel$initialSeason);
    }
#### TSB method ####
    else if(intermittent=="p"){
        if(is.null(model)){
            model <- "YYY";
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

        # If bt>1, then at = 0 and pt = bt / (at + bt) = 1
        if(any(tsbModel$fitted>1)){
            tsbModel$fitted[tsbModel$fitted>1] <- 1;
        }
        if(any(tsbModel$forecast>1)){
            tsbModel$forecast[tsbModel$forecast>1] <- 1;
        }

        # If at>1, then bt = 0 and pt = bt / (at + bt) = 0
        if(any(tsbModel$fitted<0)){
            tsbModel$fitted[tsbModel$fitted<0] <- 0;
        }
        if(any(tsbModel$forecast<0)){
            tsbModel$forecast[tsbModel$forecast<0] <- 0;
        }

        output <- list(model=model, fitted=tsbModel$fitted, forecast=tsbModel$forecast, states=tsbModel$states,
                       variance=tsbModel$forecast*(1-tsbModel$forecast), logLik=NA, nParam=nParam(tsbModel)-1,
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
                       variance=logisticModel$forecast*(1-logisticModel$forecast), logLik=NA, nParam=nParam(logisticModel),
                       residuals=logisticModel$residuals, actuals=otAll,
                       persistence=logisticModel$persistence, initial=logisticModel$initial,
                       initialSeason=logisticModel$initialSeason);
    }
#### None ####
    else{
        pt <- ts(y,start=start(y),frequency=frequency(y));
        pt.for <- ts(rep(y[obsInsample],h), start=time(y)[obsInsample]+deltat(y),frequency=frequency(y));
        errors <- ts(rep(0,obsInsample), start=start(y), frequency=frequency(y));
        output <- list(model=NULL, fitted=pt, forecast=pt.for, states=pt,
                       variance=rep(0,h), logLik=NA, nParam=0,
                       residuals=errors, actuals=pt,
                       persistence=NULL, initial=NULL, initialSeason=NULL);
    }
    output$intermittent <- intermittent;
    output$logLik <- (sum(log(output$fitted[ot!=0])) +
                      sum(log(1-output$fitted[ot==0])));
    return(structure(output,class="iss"));
}
