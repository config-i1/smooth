#' Combination of forecasts of state-space models
#'
#' Function constructs ETS, SSARIMA, CES, GES and SMA and combines their
#' forecasts using IC weights.
#'
#' The combination of these models using information criteria weights is
#' possible because they are all formulated in Single Source of Error
#' framework. Due to the the complexity of some of the models, the
#' estimation process may take some time. So be patient.
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#' @template ssETSRef
#' @template ssIntervalsRef
#'
#' @param initial Can be \code{"optimal"}, meaning that the initial
#' states are optimised, or \code{"backcasting"}, meaning that the
#' initials are produced using backcasting procedure.
#' @param ... This currently determines nothing.
#'
#' \itemize{
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{initialType} - type of the initial values used.
#' \item \code{fitted} - fitted values of ETS.
#' \item \code{forecast} - point forecast of ETS.
#' \item \code{lower} - lower bound of prediction interval. When \code{intervals="none"}
#' then NA is returned.
#' \item \code{upper} - higher bound of prediction interval. When \code{intervals="none"}
#' then NA is returned.
#' \item \code{residuals} - residuals of the estimated model.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom into account).
#' \item \code{intervals} - type of intervals asked by user.
#' \item \code{level} - confidence level for intervals.
#' \item \code{cumulative} - whether the produced forecast was cumulative or not.
#' \item \code{actuals} - original data.
#' \item \code{holdout} - holdout part of the original data.
#' \item \code{imodel} - model of the class "iss" if intermittent model was estimated.
#' If the model is non-intermittent, then imodel is \code{NULL}.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If \code{xregDo="s"},
#' then this value will contain only selected exogenous variables.
#' \item \code{updateX} - boolean, defining, if the states of exogenous variables were
#' estimated as well.
#' \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc, BIC and BICc.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample. In
#' case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.
#' }
#'
#' @seealso \code{\link[smooth]{es}, \link[smooth]{auto.ssarima},
#' \link[smooth]{auto.ces}, \link[smooth]{auto.ges}, \link[smooth]{sma}}
#'
#' @examples
#'
#' library(Mcomp)
#'
#' ourModel <- combineSmooth(M3[[578]])
#' plot(ourModel)
#'
#' @export combineSmooth
combineSmooth <- function(data, initial=c("optimal","backcasting"), ic=c("AICc","AIC","BIC","BICc"),
                          cfType=c("MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                          h=10, holdout=FALSE, cumulative=FALSE,
                          intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                          intermittent=c("none","auto","fixed","interval","probability","sba","logistic"),
                          imodel="MNN",
                          bounds=c("admissible","none"),
                          silent=c("all","graph","legend","output","none"),
                          xreg=NULL, xregDo=c("use","select"), initialX=NULL,
                          updateX=FALSE, persistenceX=NULL, transitionX=NULL,
                          ...){
# Copyright (C) 2018 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    thisEnvironment <- environment();
    list2env(list(...),thisEnvironment);

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- thisEnvironment;
    ssInput("smoothC",ParentEnvironment=thisEnvironment);

    if(ic=="AICc"){
        IC <- AICc;
    }
    else if(ic=="AIC"){
        IC <- AIC;
    }
    else if(ic=="BIC"){
        IC <- BIC;
    }
    else if(ic=="BICc"){
        IC <- BICc;
    }

    # These values are needed for the prediction intervals
    nModels <- 5;

    if(!silentText){
        cat("Estimating models... ");
    }
    # This function produces forecasts from several models and then combines them
    #### This is model fitting ####
    if(!silentText){
        cat("ES");
    }
    esModel <- es(data,initial=initial,ic=ic,cfType=cfType,h=h,holdout=holdout,
                  cumulative=cumulative,intervals=intervalsType,level=0.5,intermittent=intermittent,
                  imodel=imodel,bounds=bounds,silent=TRUE,
                  xreg=NULL,xregDo=c("use","select"),updateX=FALSE,
                  initialX=initialX,persistenceX=persistenceX,transitionX=transitionX);
    if(!silentText){
        cat(", CES");
    }
    cesModel <- auto.ces(data,initial=initial,ic=ic,cfType=cfType,h=h,holdout=holdout,
                         cumulative=cumulative,intervals=intervalsType,level=0.5,intermittent=intermittent,
                         imodel=imodel,bounds=bounds,silent=TRUE,
                         xreg=NULL,xregDo=c("use","select"),updateX=FALSE,
                         initialX=initialX,persistenceX=persistenceX,transitionX=transitionX);
    if(!silentText){
        cat(", SSARIMA");
    }
    ssarimaModel <- auto.ssarima(data,initial=initial,ic=ic,cfType=cfType,h=h,holdout=holdout,
                                 cumulative=cumulative,intervals=intervalsType,level=0.5,intermittent=intermittent,
                                 imodel=imodel,bounds=bounds,silent=TRUE,
                                 xreg=NULL,xregDo=c("use","select"),updateX=FALSE,
                                 initialX=initialX,persistenceX=persistenceX,transitionX=transitionX);
    if(!silentText){
        cat(", GES");
    }
    gesModel <- auto.ges(data,initial=initial,ic=ic,cfType=cfType,h=h,holdout=holdout,
                         cumulative=cumulative,intervals=intervalsType,level=0.5,intermittent=intermittent,
                         imodel=imodel,bounds=bounds,silent=TRUE,
                         xreg=NULL,xregDo=c("use","select"),updateX=FALSE,
                         initialX=initialX,persistenceX=persistenceX,transitionX=transitionX);
    if(!silentText){
        cat(", SMA");
    }
    smaModel <- sma(data,ic=ic,h=h,holdout=holdout,
                    cumulative=cumulative,intervals=intervalsType,level=0.5,silent=TRUE);
    if(!silentText){
        cat(". Done!\n");
    }

    # Calculate AIC weights
    ICs <- c(IC(esModel),IC(cesModel),IC(ssarimaModel),IC(gesModel),IC(smaModel));
    names(ICs) <- paste0(c("ETS", "CES", "SSARIMA", "GES", "SMA"), " ", ic);
    icBest <- min(ICs);
    icWeights <- exp(-0.5*(ICs-icBest)) / sum(exp(-0.5*(ICs-icBest)));

    yForecast <- (esModel$forecast * icWeights[1] + cesModel$forecast * icWeights[2] +
                      ssarimaModel$forecast * icWeights[3] + gesModel$forecast * icWeights[4] +
                      smaModel$forecast * icWeights[5]);

    yFitted <- (esModel$fitted * icWeights[1] + cesModel$fitted * icWeights[2] +
                      ssarimaModel$fitted * icWeights[3] + gesModel$fitted * icWeights[4] +
                      smaModel$fitted * icWeights[5]);

    y <- esModel$actuals;
    yHoldout <- esModel$holdout;

    lower <- upper <- NA;

    yForecastStart <- start(yForecast);

    if(intervalsType!="n"){
        #### This part is for combining the prediction intervals ####
        bins <- 1000-1

        # This is needed for appropriate combination of prediction intervals
        ourQuantiles <- array(NA,c(nModels,bins,h),dimnames=list(paste0("Model",c(1:nModels)),
                                                                 c(1:bins)/(bins+1),paste0("h",c(1:h))))
        minMaxQuantiles <- matrix(NA,2,h)
        # Prepare the matrix for the sequences from min to max for each h
        ourSequence <- array(NA,c(bins,h))

        # Write down the median values for all the models
        ourQuantiles[1,"0.5",] <- esModel$lower;
        ourQuantiles[2,"0.5",] <- cesModel$lower;
        ourQuantiles[3,"0.5",] <- ssarimaModel$lower;
        ourQuantiles[4,"0.5",] <- gesModel$lower;
        ourQuantiles[5,"0.5",] <- smaModel$lower;

        if(!silentText){
            cat("Constructing prediction intervals...    ");
        }
        # Do loop writing down all the quantiles
        for(j in 1:((bins-1)/2)){
            if(!silentText){
                if(j==1){
                    cat("\b");
                }
                cat(paste0(rep("\b",nchar(round((j-1)/((bins-1)/2),2)*100)+1),collapse=""));
                cat(paste0(round(j/((bins-1)/2),2)*100,"%"));
            }
            esModel <- es(data,model=esModel,h=h,intervals=intervalsType,level=j*2/(bins+1),
                          holdout=holdout,cumulative=cumulative,silent=T);
            ourQuantiles[1,(bins+1)/2-j,] <- esModel$lower;
            ourQuantiles[1,(bins+1)/2+j,] <- esModel$upper;

            cesModel <- ces(data,model=cesModel,h=h,intervals=intervalsType,level=j*2/(bins+1),
                            holdout=holdout,cumulative=cumulative,silent=T);
            ourQuantiles[2,(bins+1)/2-j,] <- cesModel$lower;
            ourQuantiles[2,(bins+1)/2+j,] <- cesModel$upper;

            ssarimaModel <- ssarima(data,model=ssarimaModel,h=h,intervals=intervalsType,level=j*2/(bins+1),
                                    holdout=holdout,cumulative=cumulative,silent=T);
            ourQuantiles[3,(bins+1)/2-j,] <- ssarimaModel$lower;
            ourQuantiles[3,(bins+1)/2+j,] <- ssarimaModel$upper;

            gesModel <- ges(data,model=gesModel,h=h,intervals=intervalsType,level=j*2/(bins+1),
                            holdout=holdout,cumulative=cumulative,silent=T);
            ourQuantiles[4,(bins+1)/2-j,] <- gesModel$lower;
            ourQuantiles[4,(bins+1)/2+j,] <- gesModel$upper;

            smaModel <- sma(data,model=smaModel,h=h,intervals=intervalsType,level=j*2/(bins+1),
                            holdout=holdout,cumulative=cumulative,silent=T);
            ourQuantiles[5,(bins+1)/2-j,] <- smaModel$lower;
            ourQuantiles[5,(bins+1)/2+j,] <- smaModel$upper;
        }

        # Write down minimum and maximum values between the models for each horizon
        minMaxQuantiles[1,] <- apply(ourQuantiles,3,min);
        minMaxQuantiles[2,] <- apply(ourQuantiles,3,max);
        # Prepare an array with the new combined probabilities
        newProbabilities <- array(NA,c(bins,h),dimnames=list(c(1:bins),dimnames(ourQuantiles)[[3]]));
        for(j in 1:h){
            ourSequence[,j] <- seq(minMaxQuantiles[1,j],minMaxQuantiles[2,j],length.out=bins);
            for(k in 1:bins){
                newProbabilities[k,j] <- sum(icWeights %*% (ourQuantiles[,,j] <= ourSequence[k,j])) / (bins+1);
            }
        }

        # The correct intervals - quantiles, for which the newP is the first time > than selected value
        intervalsCorrect <- matrix(NA,2,h,dimnames=list(c("Lower","Upper"),dimnames(ourQuantiles)[[3]]));
        for(j in 1:h){
            intervalsCorrect[1,j] <- ourSequence[newProbabilities[,j]>=0.05,j][1];
            intervalsCorrect[2,j] <- ourSequence[newProbabilities[,j]>=0.95,j][1];
        }

        if(!silentText){
            cat(" Done!\n");
        }

        lower <- ts(intervalsCorrect[1,],start=yForecastStart,frequency=datafreq);
        upper <- ts(intervalsCorrect[2,],start=yForecastStart,frequency=datafreq);
    }

    errors <- c(y[1:length(yFitted)])-c(yFitted);
    s2 <- mean(errors^2);

    ##### Now let's deal with holdout #####
    if(holdout){
        if(cumulative){
            errormeasures <- Accuracy(sum(yHoldout),yForecast,h*y);
        }
        else{
            errormeasures <- Accuracy(yHoldout,yForecast,y);
        }

        if(cumulative){
            yHoldout <- ts(sum(yHoldout),start=yForecastStart,frequency=datafreq);
        }
    }
    else{
        errormeasures <- NA;
    }

    if(!silentGraph){
        yForecastNew <- yForecast;
        upperNew <- upper;
        lowerNew <- lower;
        if(cumulative){
            yForecastNew <- ts(rep(yForecast/h,h),start=yForecastStart,frequency=datafreq)
            if(intervals){
                upperNew <- ts(rep(upper/h,h),start=yForecastStart,frequency=datafreq)
                lowerNew <- ts(rep(lower/h,h),start=yForecastStart,frequency=datafreq)
            }
        }

        if(intervals){
            graphmaker(actuals=data,forecast=yForecastNew,fitted=yFitted, lower=lowerNew,upper=upperNew,
                       level=level,legend=!silentLegend,main="Combined smooth forecasts",cumulative=cumulative);
        }
        else{
            graphmaker(actuals=data,forecast=yForecastNew,fitted=yFitted,
                       legend=!silentLegend,main="Combined smooth forecasts",cumulative=cumulative);
        }
    }

    model <- list(timeElapsed=Sys.time()-startTime, initialType=initialType, fitted=yFitted,
                  forecast=yForecast, lower=lower, upper=upper, residuals=errors, s2=s2,
                  intervals=intervalsType, level=level, cumulative=cumulative,
                  actuals=data, holdout=yHoldout, ICs=ICs, ICw=icWeights, cfType=cfType,
                  cf=NULL,accuracy=errormeasures);

    return(structure(model,class=c("smoothC","smooth")));
}
