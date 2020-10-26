utils::globalVariables(c("measurementEstimate","transitionEstimate", "B",
                         "persistenceEstimate","obsAll","obsInSample","multisteps","ot","obsNonzero","ICs","cfObjective",
                         "yForecast","yLower","yUpper","normalizer","yForecastStart"));

#' Generalised Univariate Model
#'
#' Function constructs Generalised Univariate Model, estimating matrices F, w,
#' vector g and initial parameters.
#'
#' The function estimates the Single Source of Error state space model of the
#' following type:
#'
#' \deqn{y_{t} = o_{t} (w' v_{t-l} + x_t a_{t-1} + \epsilon_{t})}
#'
#' \deqn{v_{t} = F v_{t-l} + g \epsilon_{t}}
#'
#' \deqn{a_{t} = F_{X} a_{t-1} + g_{X} \epsilon_{t} / x_{t}}
#'
#' Where \eqn{o_{t}} is the Bernoulli distributed random variable (in case of
#' normal data equal to 1), \eqn{v_{t}} is the state vector (defined using
#' \code{orders}) and \eqn{l} is the vector of \code{lags}, \eqn{x_t} is the
#' vector of exogenous parameters. \eqn{w} is the \code{measurement} vector,
#' \eqn{F} is the \code{transition} matrix, \eqn{g} is the \code{persistence}
#' vector, \eqn{a_t} is the vector of parameters for exogenous variables,
#' \eqn{F_{X}} is the \code{transitionX} matrix and \eqn{g_{X}} is the
#' \code{persistenceX} matrix. Finally, \eqn{\epsilon_{t}} is the error term.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("gum","smooth")}
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssIntervals
#' @template ssInitialParam
#' @template ssPersistenceParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template smoothRef
#' @template ssIntervalsRef
#'
#' @param orders Order of the model. Specified as vector of number of states
#' with different lags. For example, \code{orders=c(1,1)} means that there are
#' two states: one of the first lag type, the second of the second type.
#' @param lags Defines lags for the corresponding orders. If, for example,
#' \code{orders=c(1,1)} and lags are defined as \code{lags=c(1,12)}, then the
#' model will have two states: the first will have lag 1 and the second will
#' have lag 12. The length of \code{lags} must correspond to the length of
#' \code{orders}.
#' @param type Type of model. Can either be \code{"A"} - additive - or
#' \code{"M"} - multiplicative. The latter means that the GUM is fitted on
#' log-transformed data.
#' @param transition Transition matrix \eqn{F}. Can be provided as a vector.
#' Matrix will be formed using the default \code{matrix(transition,nc,nc)},
#' where \code{nc} is the number of components in state vector. If \code{NULL},
#' then estimated.
#' @param measurement Measurement vector \eqn{w}. If \code{NULL}, then
#' estimated.
#' @param ...  Other non-documented parameters.  For example parameter
#' \code{model} can accept a previously estimated GUM model and use all its
#' parameters.  \code{FI=TRUE} will make the function produce Fisher
#' Information matrix, which then can be used to calculated variances of
#' parameters of the model.
#' You can also pass two parameters to the optimiser: 1. \code{maxeval} - maximum
#' number of evaluations to carry on; 2. \code{xtol_rel} - the precision of the
#' optimiser. The default values used in es() are \code{maxeval=5000} and
#' \code{xtol_rel=1e-8}. You can read more about these parameters in the
#' documentation of \link[nloptr]{nloptr} function.
#' @return Object of class "smooth" is returned. It contains:
#'
#' \itemize{
#' \item \code{model} - name of the estimated model.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{states} - matrix of fuzzy components of GUM, where \code{rows}
#' correspond to time and \code{cols} to states.
#' \item \code{initialType} - Type of the initial values used.
#' \item \code{initial} - initial values of state vector (extracted from
#' \code{states}).
#' \item \code{nParam} - table with the number of estimated / provided parameters.
#' If a previous model was reused, then its initials are reused and the number of
#' provided parameters will take this into account.
#' \item \code{measurement} - matrix w.
#' \item \code{transition} - matrix F.
#' \item \code{persistence} - persistence vector. This is the place, where
#' smoothing parameters live.
#' \item \code{fitted} - fitted values.
#' \item \code{forecast} - point forecast.
#' \item \code{lower} - lower bound of prediction interval. When
#' \code{interval="none"} then NA is returned.
#' \item \code{upper} - higher bound of prediction interval. When
#' \code{interval="none"} then NA is returned.
#' \item \code{residuals} - the residuals of the estimated model.
#' \item \code{errors} - matrix of 1 to h steps ahead errors.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom
#' into account).
#' \item \code{interval} - type of interval asked by user.
#' \item \code{level} - confidence level for interval.
#' \item \code{cumulative} - whether the produced forecast was cumulative or not.
#' \item \code{y} - original data.
#' \item \code{holdout} - holdout part of the original data.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If
#' \code{xregDo="s"}, then this value will contain only selected exogenous variables.
#' \item \code{initialX} - initial values for parameters of exogenous variables.
#' \item \code{ICs} - values of information criteria of the model. Includes
#' AIC, AICc, BIC and BICc.
#' \item \code{logLik} - log-likelihood of the function.
#' \item \code{lossValue} - Cost function value.
#' \item \code{loss} - Type of loss function used in the estimation.
#' \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE} or
#' when \code{FI} variable is not provided at all.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample.
#' In case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.
#' \item \code{B} - the vector of all the estimated parameters.
#' }
#' @seealso \code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ces},
#' \link[smooth]{sim.es}}
#'
#' @examples
#'
#' # Something simple:
#' gum(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,bounds="a",interval="p")
#'
#' # A more complicated model with seasonality
#' \dontrun{ourModel <- gum(rnorm(118,100,3),orders=c(2,1),lags=c(1,4),h=18,holdout=TRUE)}
#'
#' # Redo previous model on a new data and produce prediction interval
#' \dontrun{gum(rnorm(118,100,3),model=ourModel,h=18,interval="sp")}
#'
#' # Produce something crazy with optimal initials (not recommended)
#' \dontrun{gum(rnorm(118,100,3),orders=c(1,1,1),lags=c(1,3,5),h=18,holdout=TRUE,initial="o")}
#'
#' # Simpler model estiamted using trace forecast error loss function and its analytical analogue
#' \dontrun{gum(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,bounds="n",loss="TMSE")
#' gum(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,bounds="n",loss="aTMSE")}
#'
#' # Introduce exogenous variables
#' \dontrun{gum(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,xreg=c(1:118))}
#'
#' # Or select the most appropriate one
#' \dontrun{gum(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,xreg=c(1:118),xregDo="s")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))}
#'
#' @rdname gum
#' @export gum
gum <- function(y, orders=c(1,1), lags=c(1,frequency(y)), type=c("additive","multiplicative"),
                persistence=NULL, transition=NULL, measurement=NULL,
                initial=c("optimal","backcasting"), ic=c("AICc","AIC","BIC","BICc"),
                loss=c("MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                h=10, holdout=FALSE, cumulative=FALSE,
                interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
                bounds=c("restricted","admissible","none"),
                silent=c("all","graph","legend","output","none"),
                xreg=NULL, xregDo=c("use","select"), initialX=NULL, ...){
# General Univariate Model function. Crazy thing...
#
#    Copyright (C) 2016 - Inf Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ### Depricate the old parameters
    ellipsis <- list(...)
    ellipsis <- depricator(ellipsis, "occurrence", "es");
    ellipsis <- depricator(ellipsis, "oesmodel", "es");
    ellipsis <- depricator(ellipsis, "updateX", "es");
    ellipsis <- depricator(ellipsis, "persistenceX", "es");
    ellipsis <- depricator(ellipsis, "transitionX", "es");
    updateX <- FALSE;
    persistenceX <- transitionX <- NULL;
    occurrence <- "none";
    oesmodel <- "MNN";

# Add all the variables in ellipsis to current environment
    list2env(ellipsis,environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model",inherits=FALSE)){
        if(is.null(model$model)){
            stop("The provided model is not GUM.",call.=FALSE);
        }
        else if(smoothType(model)!="GUM"){
            stop("The provided model is not GUM.",call.=FALSE);
        }

        type <- errorType(model);

        if(!is.null(model$occurrence)){
            occurrence <- model$occurrence;
        }
        initial <- model$initial;
        persistence <- model$persistence;
        transition <- model$transition;
        measurement <- model$measurement;
        if(is.null(xreg)){
            xreg <- model$xreg;
        }
        else{
            if(is.null(model$xreg)){
                xreg <- NULL;
            }
            else{
                if(ncol(xreg)!=ncol(model$xreg)){
                    xreg <- xreg[,colnames(model$xreg)];
                }
            }
        }
        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
            updateX <- TRUE;
        }
        model <- model$model;
        orders <- as.numeric(substring(model,unlist(gregexpr("\\[",model))-1,unlist(gregexpr("\\[",model))-1));
        lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
    }

    orders <- orders[order(lags)];
    lags <- sort(lags);
    # Remove redundant lags (if present)
    lags <- lags[!is.na(orders)];
    # Remove NAs (if lags are longer than orders)
    orders <- orders[!is.na(orders)];

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput("gum",ParentEnvironment=environment());

##### Initialise gum #####
ElementsGUM <- function(B){
    n.coef <- 0;
    if(measurementEstimate){
        matw <- matrix(B[n.coef+(1:nComponents)],1,nComponents);
        n.coef <- n.coef + nComponents;
    }
    else{
        matw <- matrix(measurement,1,nComponents);
    }

    if(transitionEstimate){
        matF <- matrix(B[n.coef+(1:(nComponents^2))],nComponents,nComponents);
        n.coef <- n.coef + nComponents^2;
    }
    else{
        matF <- matrix(transition,nComponents,nComponents);
    }

    if(persistenceEstimate){
        vecg <- matrix(B[n.coef+(1:nComponents)],nComponents,1);
        n.coef <- n.coef + nComponents;
    }
    else{
        vecg <- matrix(persistence,nComponents,1);
    }

    vt <- matrix(NA,lagsModelMax,nComponents);
    if(initialType!="b"){
        if(initialType=="o"){
            vtvalues <- B[n.coef+(1:(orders %*% lags))];
            n.coef <- n.coef + c(orders %*% lags);

            for(i in 1:nComponents){
                vt[(lagsModelMax - lagsModel + 1)[i]:lagsModelMax,i] <- vtvalues[((cumsum(c(0,lagsModel))[i]+1):cumsum(c(0,lagsModel))[i+1])];
                vt[is.na(vt[1:lagsModelMax,i]),i] <- rep(vt[(lagsModelMax - lagsModel + 1)[i]:lagsModelMax,i],
                                                   ceiling((lagsModelMax - lagsModel + 1) / lagsModel)[i])[is.na(vt[1:lagsModelMax,i])];
            }
        }
        else if(initialType=="p"){
            vt[,] <- initialValue;
        }
    }
    else{
        vt[,] <- matvt[1:lagsModelMax,nComponents];
    }

# If exogenous are included
    if(xregEstimate){
        at <- matrix(NA,lagsModelMax,nExovars);
        if(initialXEstimate){
            at[,] <- rep(B[n.coef+(1:nExovars)],each=lagsModelMax);
            n.coef <- n.coef + nExovars;
        }
        else{
            at <- matat[1:lagsModelMax,];
        }
        if(FXEstimate){
            matFX <- matrix(B[n.coef+(1:(nExovars^2))],nExovars,nExovars);
            n.coef <- n.coef + nExovars^2;
        }

        if(gXEstimate){
            vecgX <- matrix(B[n.coef+(1:nExovars)],nExovars,1);
            n.coef <- n.coef + nExovars;
        }
    }
    else{
        at <- matrix(matat[1:lagsModelMax,],lagsModelMax,nExovars);
    }

    return(list(matw=matw,matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

##### Cost Function for GUM #####
CF <- function(B){
    elements <- ElementsGUM(B);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:lagsModelMax,] <- elements$vt;
    matat[1:lagsModelMax,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    cfRes <- costfunc(matvt, matF, matw, yInSample, vecg,
                       h, lagsModel, Etype, Ttype, Stype,
                       multisteps, loss, normalizer, initialType,
                       matxt, matat, matFX, vecgX, ot,
                       bounds, 0);

    if(is.nan(cfRes) | is.na(cfRes)){
        cfRes <- 1e100;
    }
    return(cfRes);
}

##### Estimate gum or just use the provided values #####
CreatorGUM <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

# If there is something to optimise, let's do it.
    if(any((initialType=="o"),(measurementEstimate),(transitionEstimate),(persistenceEstimate),
       (initialXEstimate),(FXEstimate),(gXEstimate))){

        if(is.null(providedC)){
            ub <- lb <- B <- NULL;
# matw, matF, vecg, vt
            if(measurementEstimate){
                B <- c(B,rep(1,nComponents));
                if(bounds=="r"){
                    lb <- c(lb,rep(0,nComponents));
                    ub <- c(ub,rep(1,nComponents));
                }
                else{
                    lb <- c(lb,rep(-Inf,nComponents));
                    ub <- c(ub,rep(Inf,nComponents));
                }
            }
            if(transitionEstimate){
                # matFInterim <- diag(nComponents);
                # matFInterim[upper.tri(matFInterim)] <- 1;
                # matFInterim[lower.tri(matFInterim)] <- 1;
                # B <- c(B,c(matFInterim));
                B <- c(B,rep(1,nComponents^2));
                if(bounds=="r"){
                    lb <- c(lb,rep(0,nComponents^2));
                    ub <- c(ub,rep(1,nComponents^2));
                }
                else{
                    lb <- c(lb,rep(-Inf,nComponents^2));
                    ub <- c(ub,rep(Inf,nComponents^2));
                }
            }
            if(persistenceEstimate){
                B <- c(B,rep(0.1,nComponents));
                lb <- c(lb,rep(-Inf,nComponents));
                ub <- c(ub,rep(Inf,nComponents));
            }
            if(initialType=="o"){
                B <- c(B,intercept);
                lb <- c(lb,-Inf);
                ub <- c(ub,Inf);
                if((orders %*% lags)>1){
                    B <- c(B,slope);
                    lb <- c(lb,-Inf);
                    ub <- c(ub,Inf);
                }
                if((orders %*% lags)>2){
                    B <- c(B,yot[1:(orders %*% lags-2),]);
                    lb <- c(lb,rep(-Inf,(orders %*% lags-2)));
                    ub <- c(ub,rep(Inf,(orders %*% lags-2)));
                }
            }

# initials, transition matrix and persistence vector
            if(xregEstimate){
                if(initialXEstimate){
                    B <- c(B,matat[lagsModelMax,]);
                    lb <- c(lb,rep(-Inf,nExovars));
                    ub <- c(ub,rep(Inf,nExovars));
                }
                if(updateX){
                    if(FXEstimate){
                        B <- c(B,c(diag(nExovars)));
                        lb <- c(lb,rep(0,nExovars^2));
                        ub <- c(ub,rep(1,nExovars^2));
                    }
                    if(gXEstimate){
                        B <- c(B,rep(0,nExovars));
                        lb <- c(lb,rep(-Inf,nExovars));
                        ub <- c(ub,rep(Inf,nExovars));
                    }
                }
            }
        }

# Optimise model. First run
        res <- nloptr(B, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=xtol_rel, "maxeval"=maxeval),
                      lb=lb, ub=ub);
        B <- res$solution;

# Optimise model. Second run
        res2 <- nloptr(B, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=xtol_rel/100, "maxeval"=maxeval/5),
                       lb=lb, ub=ub);
        # This condition is needed in order to make sure that we did not make the solution worse
        if(res2$objective <= res$objective){
            res <- res2;
        }

        B <- res$solution;
        cfObjective <- res$objective;

        # Parameters estimated + variance
        nParam <- length(B) + 1;
    }
    else{
# matw, matF, vecg, vt
        B <- c(measurement,
               c(transition),
               c(persistence),
               c(initialValue));

        B <- c(B,matat[lagsModelMax,],
               c(transitionX),
               c(persistenceX));

        cfObjective <- CF(B);

        # Only variance is estimated
        nParam <- 1;
    }

    ICValues <- ICFunction(nParam=nParam,nParamOccurrence=nParamOccurrence,
                           B=B,Etype=Etype);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;

    icBest <- ICs[ic];

    return(list(cfObjective=cfObjective,B=B,ICs=ICs,icBest=icBest,nParam=nParam,logLik=logLik));
}

##### Preset yFitted, yForecast, errors and basic parameters #####
    matvt <- matrix(NA,nrow=obsStates,ncol=nComponents);
    yFitted <- rep(NA,obsInSample);
    yForecast <- rep(NA,h);
    errors <- rep(NA,obsInSample);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(y=y, xreg=xreg, updateX=FALSE, ot=ot,
                       persistenceX=NULL, transitionX=NULL, initialX=initialX,
                       obsInSample=obsInSample, obsAll=obsAll, obsStates=obsStates,
                       lagsModelMax=lagsModelMax, h=h, xregDo=xregDo, silent=silentText);

    if(xregDo=="u"){
        nExovars <- xregdata$nExovars;
        matxt <- xregdata$matxt;
        matat <- xregdata$matat;
        xregEstimate <- xregdata$xregEstimate;
        matFX <- xregdata$matFX;
        vecgX <- xregdata$vecgX;
        xregNames <- colnames(matxt);
    }
    else{
        nExovars <- 1;
        nExovarsOriginal <- xregdata$nExovars;
        matxtOriginal <- xregdata$matxt;
        matatOriginal <- xregdata$matat;
        xregEstimateOriginal <- xregdata$xregEstimate;
        matFXOriginal <- xregdata$matFX;
        vecgXOriginal <- xregdata$vecgX;

        matxt <- matrix(1,nrow(matxtOriginal),1);
        matat <- matrix(0,nrow(matatOriginal),1);
        xregEstimate <- FALSE;
        matFX <- matrix(1,1,1);
        vecgX <- matrix(0,1,1);
        xregNames <- NULL;
    }
    xreg <- xregdata$xreg;
    FXEstimate <- xregdata$FXEstimate;
    gXEstimate <- xregdata$gXEstimate;
    initialXEstimate <- xregdata$initialXEstimate;
    if(is.null(xreg)){
        xregDo <- "u";
    }

    # These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

# Check number of parameters vs data
    nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
    nParamOccurrence <- all(occurrence!=c("n","p"))*1;
    nParamMax <- nParamMax + nParamExo + nParamOccurrence;

    if(xregDo=="u"){
        parametersNumber[1,2] <- nParamExo;
        # If transition is provided and not identity, and other things are provided, write them as "provided"
        parametersNumber[2,2] <- (length(matFX)*(!is.null(transitionX) & !all(matFX==diag(ncol(matat)))) +
                                      nrow(vecgX)*(!is.null(persistenceX)) +
                                      ncol(matat)*(!is.null(initialX)));
    }

##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        if(xregDo=="select"){
            if(obsNonzero <= (nParamMax - nParamExo)){
                warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                               nParamMax + nParamExo," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
                tinySample <- TRUE;
            }
            else{
                warning(paste0("The potential number of exogenous variables is higher than the number of observations. ",
                               "This may cause problems in the estimation."),call.=FALSE);
            }
        }
        else{
            warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                           nParamMax," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
            tinySample <- TRUE;
        }
    }
    else{
        tinySample <- FALSE;
    }

# If this is tiny sample, use SES instead
    if(tinySample){
        warning("Not enough observations to fit GUM Switching to ETS(A,N,N).",call.=FALSE);
        return(es(y,"ANN",initial=initial,loss=loss,
                  h=h,holdout=holdout,cumulative=cumulative,
                  interval=interval,level=level,
                  occurrence=occurrence,
                  oesmodel=oesmodel,
                  bounds="u",
                  silent=silent,
                  xreg=xreg,xregDo=xregDo,initialX=initialX,
                  updateX=updateX,persistenceX=persistenceX,transitionX=transitionX));
    }

##### Preset values of matvt ######
    slope <- (cov(yot[1:min(max(12,dataFreq),obsNonzero),],c(1:min(max(12,dataFreq),obsNonzero)))/
                  var(c(1:min(max(12,dataFreq),obsNonzero))));
    intercept <- (sum(yot[1:min(max(12,dataFreq),obsNonzero),])/min(max(12,dataFreq),obsNonzero) -
                      slope * (sum(c(1:min(max(12,dataFreq),obsNonzero)))/
                                   min(max(12,dataFreq),obsNonzero) - 1));

    vtvalues <- intercept;
    if((orders %*% lags)>1){
        vtvalues <- c(vtvalues,slope);
    }
    if((orders %*% lags)>2){
        if(orders %*% lags-2 > obsNonzero){
            vtTail <- orders %*% lags-2 - obsNonzero;
            vtvalues <- c(vtvalues,yot[1:obsNonzero,]);
            vtvalues <- c(vtvalues,rep(yot[obsNonzero],vtTail));
        }
        else{
            vtvalues <- c(vtvalues,yot[1:(orders %*% lags-2),]);
        }
    }

    vt <- matrix(NA,lagsModelMax,nComponents);
    for(i in 1:nComponents){
        vt[(lagsModelMax - lagsModel + 1)[i]:lagsModelMax,i] <- vtvalues[((cumsum(c(0,lagsModel))[i]+1):cumsum(c(0,lagsModel))[i+1])];
        vt[is.na(vt[1:lagsModelMax,i]),i] <- rep(rev(vt[(lagsModelMax - lagsModel + 1)[i]:lagsModelMax,i]),
                                           ceiling((lagsModelMax - lagsModel + 1) / lagsModel)[i])[is.na(vt[1:lagsModelMax,i])];
    }
    matvt[1:lagsModelMax,] <- vt;

#### Deal with provided B ####
    ellipsis <- list(...);
    if(any(names(ellipsis)=="B")){
        providedC <- ellipsis$B;
    }
    else{
        providedC <- NULL;
    }

    if(!is.null(providedC)){
        nParamToEstimate <- (nComponents*measurementEstimate + nComponents*persistenceEstimate +
                                 (nComponents^2)*transitionEstimate);
        if(initialType=="o"){
            nParamToEstimate <- nParamToEstimate + orders %*% lags;
        }

        if(length(providedC)!=nParamToEstimate){
            warning(paste0("Number of parameters to optimise differes from the length of B: ",nParamToEstimate," vs ",length(providedC),".\n",
                           "We will have to drop parameter B."),call.=FALSE);
            providedC <- NULL;
        }
        B <- providedC;
    }

    if(any(names(ellipsis)=="maxeval")){
        maxeval <- ellipsis$maxeval;
    }
    else{
        maxeval <- 5000;
    }
    if(any(names(ellipsis)=="xtol_rel")){
        xtol_rel <- ellipsis$xtol_rel;
    }
    else{
        xtol_rel <- 1e-8;
    }

##### Start the calculations #####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(ssForecaster) <- environment();
    environment(ssFitter) <- environment();

##### If occurrence=="a", run a loop and select the best one #####
    if(occurrence=="a"){
        if(!silentText){
            cat("Selecting the best occurrence model...\n");
        }
        # First produce the auto model
        intermittentParametersSetter(occurrence="a",ParentEnvironment=environment());
        intermittentMaker(occurrence="a",ParentEnvironment=environment());
        intermittentModel <- CreatorGUM(silent=silentText);
        occurrenceBest <- occurrence;
        occurrenceModelBest <- occurrenceModel;

        if(!silentText){
            cat("Comparing it with the best non-intermittent model...\n");
        }
        # Then fit the model without the occurrence part
        occurrence[] <- "n";
        intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
        intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());
        nonIntermittentModel <- CreatorGUM(silent=silentText);

        # Compare the results and return the best
        if(nonIntermittentModel$icBest[ic] <= intermittentModel$icBest[ic]){
            gumValues <- nonIntermittentModel;
        }
        # If this is the "auto", then use the selected occurrence to reset the parameters
        else{
            gumValues <- intermittentModel;
            occurrence[] <- occurrenceBest;
            occurrenceModel <- occurrenceModelBest;
            intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
            intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());
        }
        rm(intermittentModel,nonIntermittentModel,occurrenceModelBest);
    }
    else{
        intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
        intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());

        gumValues <- CreatorGUM(silentText=silentText);
    }

    list2env(gumValues,environment());

    if(xregDo!="u"){
# Prepare for fitting
        elements <- ElementsGUM(B);
        matw <- elements$matw;
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[1:lagsModelMax,] <- elements$vt;
        matat[1:lagsModelMax,] <- elements$at;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;

        ssFitter(ParentEnvironment=environment());

        xregNames <- colnames(matxtOriginal);
        xregNew <- cbind(errors,xreg[1:nrow(errors),]);
        colnames(xregNew)[1] <- "errors";
        colnames(xregNew)[-1] <- xregNames;
        xregNew <- as.data.frame(xregNew);
        xregResults <- stepwise(xregNew, ic=ic, silent=TRUE, df=nParam+nParamOccurrence-1);
        xregNames <- names(coef(xregResults))[-1];
        nExovars <- length(xregNames);
        if(nExovars>0){
            xregEstimate <- TRUE;
            matxt <- as.data.frame(matxtOriginal)[,xregNames];
            matat <- as.data.frame(matatOriginal)[,xregNames];
            matFX <- diag(nExovars);
            vecgX <- matrix(0,nExovars,1);

            if(nExovars==1){
                matxt <- matrix(matxt,ncol=1);
                matat <- matrix(matat,ncol=1);
                colnames(matxt) <- colnames(matat) <- xregNames;
            }
            else{
                matxt <- as.matrix(matxt);
                matat <- as.matrix(matat);
            }
        }
        else{
            nExovars <- 1;
            xreg <- NULL;
        }

        if(!is.null(xreg)){
            gumValues <- CreatorGUM(silentText=TRUE);
            list2env(gumValues,environment());
        }
    }

    if(!is.null(xreg)){
        if(ncol(matat)==1){
            colnames(matxt) <- colnames(matat) <- xregNames;
        }
        xreg <- matxt;
        if(xregDo=="s"){
            nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
            parametersNumber[1,2] <- nParamExo;
        }
    }
# Prepare for fitting
    elements <- ElementsGUM(B);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:lagsModelMax,] <- elements$vt;
    matat[1:lagsModelMax,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

    if(modelIsMultiplicative){
        yInSample <- exp(yInSample);
        yFitted <- exp(yFitted);
        yForecast <- exp(yForecast);
        yLower <- exp(yLower);
        yUpper <- exp(yUpper);

        environment(likelihoodFunction) <- environment();
        environment(ICFunction) <- environment();

        ICValues <- ICFunction(nParam=nParam,nParamOccurrence=nParamOccurrence,
                               B=B,Etype="M");
        ICs <- ICValues$ICs;
        logLik <- ICValues$llikelihood;
    }

##### Do final check and make some preparations for output #####

# Write down initials of states vector and exogenous
    parametersNumber[1,1] <- (nComponents*measurementEstimate + nComponents*persistenceEstimate +
        (nComponents^2)*transitionEstimate);
    # parametersNumber[2,1] <- (nComponents*(!measurementEstimate) + nComponents*(!persistenceEstimate) +
    #                               (nComponents^2)*(!transitionEstimate));

    if(initialType!="p"){
        initialValue <- matrix(matvt[1:lagsModelMax,],lagsModelMax);
        if(initialType!="b"){
            parametersNumber[1,1] <- parametersNumber[1,1] + orders %*% lags;
        }
    }

    if(initialXEstimate){
        initialX <- matat[1,];
        names(initialX) <- colnames(matat);
    }

    # Make initialX NULL if all xreg were dropped
    if(length(initialX)==1){
        if(initialX==0){
            initialX <- NULL;
        }
    }

    if(gXEstimate){
        persistenceX <- vecgX;
    }

    if(FXEstimate){
        transitionX <- matFX;
    }

    # Add variance estimation
    parametersNumber[1,1] <- parametersNumber[1,1] + 1;

    # Write down the number of parameters of occurrence
    if(all(occurrence!=c("n","p")) & !occurrenceModelProvided){
        parametersNumber[1,3] <- nparam(occurrenceModel);
    }

# Make some preparations
    matvt <- ts(matvt,start=(time(y)[1] - deltat(y)*lagsModelMax),frequency=dataFreq);
    if(!is.null(xreg)){
        matvt <- cbind(matvt,matat);
        colnames(matvt) <- c(paste0("Component ",c(1:nComponents)),colnames(matat));
        if(updateX){
            rownames(vecgX) <- xregNames;
            dimnames(matFX) <- list(xregNames,xregNames);
        }
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:nComponents));
    }

    parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
    parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

    # Write down Fisher Information if needed
    if(FI & parametersNumber[1,4]>1){
        environment(likelihoodFunction) <- environment();
        FI <- -numDeriv::hessian(likelihoodFunction,B);
    }
    else{
        FI <- NA;
    }

##### Deal with the holdout sample #####
    if(holdout){
        yHoldout <- ts(y[(obsInSample+1):obsAll],start=yForecastStart,frequency=dataFreq);
        if(cumulative){
            errormeasures <- measures(sum(yHoldout),yForecast,h*yInSample);
        }
        else{
            errormeasures <- measures(yHoldout,yForecast,yInSample);
        }

        if(cumulative){
            yHoldout <- ts(sum(yHoldout),start=yForecastStart,frequency=dataFreq);
        }
    }
    else{
        yHoldout <- NA;
        errormeasures <- NA;
    }

    if(!is.null(xreg)){
        modelname <- "GUMX";
    }
    else{
        modelname <- "GUM";
    }
    modelname <- paste0(modelname,"(",paste(orders,"[",lags,"]",collapse=",",sep=""),")");
    if(all(occurrence!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    if(modelIsMultiplicative){
        modelname <- paste0("M",modelname);
    }

##### Print output #####
    if(!silentText){
        if(any(abs(eigen(matF - vecg %*% matw)$values)>(1 + 1E-10))){
            if(bounds=="n"){
                warning("Unstable model was estimated! Use bounds='admissible' to address this issue!",
                        call.=FALSE);
            }
            else{
                warning("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.",
                        call.=FALSE);
            }
        }
    }

##### Make a plot #####
    if(!silentGraph){
        yForecastNew <- yForecast;
        yUpperNew <- yUpper;
        yLowerNew <- yLower;
        if(cumulative){
            yForecastNew <- ts(rep(yForecast/h,h),start=yForecastStart,frequency=dataFreq)
            if(interval){
                yUpperNew <- ts(rep(yUpper/h,h),start=yForecastStart,frequency=dataFreq)
                yLowerNew <- ts(rep(yLower/h,h),start=yForecastStart,frequency=dataFreq)
            }
        }

        if(interval){
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted, lower=yLowerNew,upper=yUpperNew,
                       level=level,legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
        else{
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted,
                       legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
    }

##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,measurement=matw,transition=matF,persistence=vecg,
                  initialType=initialType,initial=initialValue,
                  nParam=parametersNumber,
                  fitted=yFitted,forecast=yForecast,lower=yLower,upper=yUpper,residuals=errors,
                  errors=errors.mat,s2=s2,interval=intervalType,level=level,cumulative=cumulative,
                  y=y,holdout=yHoldout,
                  xreg=xreg,initialX=initialX,
                  ICs=ICs,logLik=logLik,lossValue=cfObjective,loss=loss,FI=FI,accuracy=errormeasures,
                  B=B);
    return(structure(model,class="smooth"));
}

#' @rdname gum
#' @export
ges <- function(...){
    warning("You are using the old name of the function. Please, use 'gum' instead.", call.=FALSE);
    return(gum(...));
}
