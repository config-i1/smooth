utils::globalVariables(c("damped","matG","initialEstimate","initialSeasonEstimate","xregEstimate","persistenceEstimate","phi",
                         "FXEstimate","gXEstimate","initialXEstimate","matGX","nParamMax"));

#' NOT AVAILABLE YET: Vector Exponential Smoothing in SSOE state-space model
#'
#' Function constructs vector ETS model and returns forecast, fitted values, errors
#' and matrix of states along with other useful variables. THIS IS CURRENTLY UNDER CONSTRUCTION!
#'
#' Function estimates vector ETS in a form of the Single Source of Error State-space
#' model of the following type:
#'
#' \deqn{
#' \mathbf{y}_{t} = \mathbf{o}_{t} (\mathbf{W} \mathbf{v}_{t-l} + \mathbf{x}_t
#' \mathbf{a}_{t-1} + \mathbf{\epsilon}_{t})
#' }{
#' y_{t} = o_{t} (W v_{t-l} + x_t a_{t-1} + \epsilon_{t})
#' }
#'
#' \deqn{
#' \mathbf{v}_{t} = \mathbf{F} \mathbf{v}_{t-l} + \mathbf{G}
#' \mathbf{\epsilon}_{t}
#' }{
#' v_{t} = F v_{t-l} + G \epsilon_{t}
#' }
#'
#' \deqn{\mathbf{a}_{t} = \mathbf{F_{X}} \mathbf{a}_{t-1} + \mathbf{G_{X}}
#' \mathbf{\epsilon}_{t} / \mathbf{x}_{t}}{a_{t} = F_{X} a_{t-1} + G_{X} \epsilon_{t}
#' / x_{t}}
#'
#' Where \eqn{y_{t}} is the vector of time series on observation \eqn{t}, \eqn{o_{t}}
#' is the vector of Bernoulli distributed random variable (in case of normal data it
#' becomes unit vector for all observations), \eqn{\mathbf{v}_{t}} is the matrix of
#' states and \eqn{l} is the matrix of lags, \eqn{\mathbf{x}_t} is the vector of
#' exogenous variables. \eqn{\mathbf{W}} is the measurement matrix, \eqn{\mathbf{F}}
#' is the transition matrix and \eqn{\mathbf{G}} is the persistence matrix.
#' Finally, \eqn{\epsilon_{t}} is the vector of error terms.
#'
#' Conventionally we formulate values as:
#'
#' \deqn{\mathbf{y}'_t = (y_{1,t}, y_{2,t}, \dots, y_{m,t})}{y_t = (y_{1,t}, y_{2,t},
#' \dots, y_{m,t}),}
#' where \eqn{m} is the number of series in the group.
#' \deqn{\mathbf{v}'_t = (v_{1,t}, v_{2,t}, \dots, v_{m,t})}{v'_t = (v_{1,t}, v_{2,t},
#' \dots, v_{m,t}),}
#' where \eqn{v_{i,t}} is vector of components for i-th time series.
#' \deqn{\mathbf{W}' = (w_{1}, \dots , 0;
#' \vdots , \ddots , \vdots;
#' 0 , \vdots , w_{m})}{W' = (w_{1}, ... , 0;
#' ... , ... , ...;
#' 0 , ... , w_{m})} is matrix of measurement vectors.
#'
#' For the details see Hyndman et al. (2008), chapter 17.
#'
#' @template vssBasicParam
#' @template vssAdvancedParam
#' @template ssAuthor
#' @template vssKeywords
#'
#' @template vssGeneralRef
#'
#' @param model The type of ETS model. Can consist of 3 or 4 chars: \code{ANN},
#' \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA}, \code{MAdM} etc.
#' \code{ZZZ} means that the model will be selected based on the chosen
#' information criteria type. ATTENTION! NO MODEL SELECTION IS AVAILABLE AT
#' THIS STAGE!
#'
#' Also \code{model} can accept a previously estimated VES model and use all its
#' parameters.
#'
#' Keep in mind that model selection with "Z" components uses Branch and Bound
#' algorithm and may skip some models that could have slightly smaller
#' information criteria.

#' @param initial Can be either character or a vector / matrix of initial states.
#' If it is character, then it can be \code{"individual"}, individual values of
#' the intial non-seasonal components are udes, or \code{"group"}, meaning that
#' the initials for all the time series are set to be equal to the same value.
#' If vector of states is provided, then it is automatically transformed into
#' a matrix, assuming that these values are provided for the whole group.
#' @param initialSeason Can be either character or a vector / matrix of initial
#' states. Treated the same way as \code{initial}. This means that different time
#' series may share the same initial seasonal component.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of smoothing parameters and initial states of
#' the model.
#' @return Object of class "smooth" is returned. It contains a list of
#' values.
#'
#' @seealso \code{\link[smooth]{es}, \link[forecast]{ets},
#' \link[forecast]{forecast}}
#'
#' @examples
#'
#' library(Mcomp)
#'
#' \dontrun{es(M3$N2568$x,model="MAM",h=18,holdout=TRUE)}
#'
#'
ves <- function(data, model="ANN", persistence=c("group","independent","dependent"),
                transition=c("group","independent","dependent"), damped=c("group","individual"),
                initial=c("group","individual"), initialSeason=c("group","individual"),
                cfType=c("likelihood","diagonal","trace"),
                ic=c("AICc","AIC","BIC"), h=10, holdout=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                intermittent=c("none","auto","fixed","tsb"),
                bounds=c("admissible","usual","none"),
                silent=c("none","all","graph","legend","output"), ...){
# Copyright (C) 2017 - Inf  Ivan Svetunkov

### This should be done as expanded es() function with matrix of states (rows - time, cols - states),
### large transition matrix and a persistence matrix. The returned value of the fit is vector.
### So the vfitter can be based on amended version fitter.

# Start measuring the time of calculations
    startTime <- Sys.time();

# If a previous model provided as a model, write down the variables
    # if(is.list(model)){
    #     if(gregexpr("ETS",model$model)==-1){
    #         stop("The provided model is not ETS.",call.=FALSE);
    #     }
    #     intermittent <- model$intermittent;
    #     if(any(intermittent==c("p","provided"))){
    #         warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
    #         warning("Switching to intermittent='auto'.",call.=FALSE);
    #         intermittent <- "a";
    #     }
    #     persistence <- model$persistence;
    #     transition <- model$transition;
    #     measurement <- model$measurement;
    #     initial <- model$initial;
    #     initialSeason <- model$initialSeason;
    #     if(is.null(xreg)){
    #         xreg <- model$xreg;
    #     }
    #     initialX <- model$initialX;
    #     persistenceX <- model$persistenceX;
    #     transitionX <- model$transitionX;
    #     if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
    #         updateX <- TRUE;
    #     }
    #     model <- model$model;
    #     model <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
    #     if(any(unlist(gregexpr("C",model))!=-1)){
    #         initial <- "o";
    #     }
    # }

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for vssInput and make all the checks #####
    environment(vssInput) <- environment();
    vssInput(modelType="ves",ParentEnvironment=environment());

##### Cost Function for VES #####
CF <- function(C){
    # The following function should be done in R in BasicInitialiserVES
    elements <- etsmatrices(matvt, matG, matrix(C,nrow=1), nComponents,
                            modellags, initialType, Ttype, Stype, nExovars, matat,
                            persistenceEstimate, initialType=="o", initialSeasonEstimate, xregEstimate,
                            matFX, matGX, updateX, FXEstimate, gXEstimate, initialXEstimate);

    cfRes <- vOptimiserWrap(elements$matvt, elements$matF, elements$matw, y, elements$matG,
                      h, modellags, Etype, Ttype, Stype,
                      normalizer,
                      matxt, elements$matat, elements$matFX, elements$matGX, ot);
    # multisteps, cfType, initialType, bounds,

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}




##### C values for estimation #####
# Function constructs default bounds where C values should lie
CValues <- function(bounds,Ttype,Stype,matG,matvt,maxlag,nComponents,matat){
    C <- NA;
    CLower <- NA;
    CUpper <- NA;

    if(persistenceEstimate){
        C <- c(C,matG);
        CLower <- c(CLower,rep(-5,length(matG)));
        CUpper <- c(CUpper,rep(5,length(matG)));
    }
    # if(damped){
    #     C <- c(C,phi);
    #     CLower <- c(CLower,0);
    #     CUpper <- c(CUpper,1);
    # }
    if(initialEstimate){
        if(initialType=="o"){
            C <- c(C,matvt[maxlag,1:(nComponents - (Stype!="N"))]);
            if(Ttype!="M"){
                CLower <- c(CLower,rep(-Inf,(nComponents - (Stype!="N"))));
                CUpper <- c(CUpper,rep(Inf,(nComponents - (Stype!="N"))));
            }
            else{
                CLower <- c(CLower,0.1,0.01);
                CUpper <- c(CUpper,Inf,3);
            }
        }
    }
    if(initialSeasonEstimate){
        C <- c(C,matvt[1:maxlag,nComponents]);
        if(Stype=="A"){
            CLower <- c(CLower,rep(-Inf,maxlag));
            CUpper <- c(CUpper,rep(Inf,maxlag));
        }
        else{
            CLower <- c(CLower,rep(-0.0001,maxlag));
            CUpper <- c(CUpper,rep(20,maxlag));
        }
    }

    # if(xregEstimate){
    #     if(initialXEstimate){
    #         if(Etype=="A" & modelDo!="estimate"){
    #             vecat <- matrix(y[2:obsInsample],nrow=obsInsample-1,ncol=ncol(matxt)) / diff(matxt[1:obsInsample,]);
    #             vecat[is.infinite(vecat)] <- NA;
    #             vecat <- colSums(vecat,na.rm=T);
    #         }
    #         else{
    #             vecat <- matat[maxlag,];
    #         }
    #
    #         C <- c(C,vecat);
    #         CLower <- c(CLower,rep(-Inf,nExovars));
    #         CUpper <- c(CUpper,rep(Inf,nExovars));
    #     }
    #     if(updateX){
    #         if(FXEstimate){
    #             C <- c(C,as.vector(matFX));
    #             CLower <- c(CLower,rep(-Inf,nExovars^2));
    #             CUpper <- c(CUpper,rep(Inf,nExovars^2));
    #         }
    #         if(gXEstimate){
    #             C <- c(C,as.vector(matGX));
    #             CLower <- c(CLower,rep(-Inf,nExovars));
    #             CUpper <- c(CUpper,rep(Inf,nExovars));
    #         }
    #     }
    # }

    C <- C[!is.na(C)];
    CLower <- CLower[!is.na(CLower)];
    CUpper <- CUpper[!is.na(CUpper)];

    return(list(C=C,CLower=CLower,CUpper=CUpper));
}





##### Basic parameter propagator #####
BasicMakerVES <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    basicparams <- initparams(Ttype, Stype, datafreq, obsInsample, obsAll, y,
                              damped, smoothingParameters, initialstates, seasonalCoefs);
    list2env(basicparams,ParentEnvironment);
}

##### Basic parameter propagator #####
BasicInitialiserVES <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    elements <- etsmatrices(matvt, matG, matrix(C,nrow=1), nComponents,
                            modellags, initialType, Ttype, Stype, nExovars, matat,
                            persistenceEstimate, initialType=="o", initialSeasonEstimate, xregEstimate,
                            matFX, matGX, updateX, FXEstimate, gXEstimate, initialXEstimate);

    list2env(elements,ParentEnvironment);
}

##### Basic estimation function for es() #####
EstimatorVES <- function(...){
    environment(BasicMakerVES) <- environment();
    environment(CValues) <- environment();
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();
    environment(CF) <- environment();
    BasicMakerVES(ParentEnvironment=environment());

    Cs <- CValues(bounds,Ttype,Stype,matG,matvt,maxlag,nComponents,matat);

    # Parameters are chosen to speed up the optimisation process and have decent accuracy
    res <- nloptr(Cs$C, CF, lb=Cs$CLower, ub=Cs$CUpper,
                  opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=500));
    C <- res$solution;

    # If the optimisation failed, then probably this is because of smoothing parameters in mixed models. Set them eqaul to zero.
    if(any(C==Cs$C)){
        if(C[1]==Cs$C[1]){
            C[1] <- 0;
        }
        if(Ttype!="N"){
            if(C[2]==Cs$C[2]){
                C[2] <- 0;
            }
            if(Stype!="N"){
                if(C[3]==Cs$C[3]){
                    C[3] <- 0;
                }
            }
        }
        else{
            if(Stype!="N"){
                if(C[2]==Cs$C[2]){
                    C[2] <- 0;
                }
            }
        }
        res <- nloptr(C, CF, lb=Cs$CLower, ub=Cs$CUpper,
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=500));
        C <- res$solution;
    }

    if(any((C>=Cs$CUpper),(C<=Cs$CLower))){
        C[C>=Cs$CUpper] <- Cs$CUpper[C>=Cs$CUpper] * 0.999 - 0.001;
        C[C<=Cs$CLower] <- Cs$CLower[C<=Cs$CLower] * 1.001 + 0.001;
    }

    res2 <- nloptr(C, CF, lb=Cs$CLower, ub=Cs$CUpper,
                  opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-6, "maxeval"=500));
    # This condition is needed in order to make sure that we did not make the solution worse
    if(res2$objective <= res$objective){
        res <- res2;
    }
    C <- res$solution;

    if(all(C==Cs$C) & (initialType!="b")){
        if(any(persistenceEstimate,gXEstimate,FXEstimate)){
            warning(paste0("Failed to optimise the model ETS(", modelCurrent,
                           "). Try different initialisation maybe?\nAnd check all the messages and warnings...",
                           "If you did your best, but the optimiser still fails, report this to the maintainer, please."),
                    call.=FALSE);
        }
    }

    nParam <- 1 + nComponents + damped + (nComponents + (maxlag - 1) * (Stype!="N")) * initialEstimate + (!is.null(xreg)) * nExovars + (updateX)*(nExovars^2 + nExovars);

    # Change cfType for model selection
    if(multisteps){
        if(substring(cfType,1,1)=="a"){
            cfType <- "aTFL";
        }
        else{
            cfType <- "TFL";
        }
    }
    else{
        cfType <- "MSE";
    }

    ICValues <- ICFunction(nParam=nParam+nParamIntermittent,C=res$solution,Etype=Etype);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;

    # Change back
    cfType <- cfTypeOriginal;
    return(list(ICs=ICs,objective=res$objective,C=C,nParam=nParam,FI=FI,logLik=logLik));
}

##### Function constructs the VES function #####
CreatorVES <- function(silent=FALSE,...){
    if(modelDo=="estimate"){
        environment(EstimatorVES) <- environment();
        res <- EstimatorVES(ParentEnvironment=environment());
        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,
                             cfObjective=res$objective,C=res$C,ICs=res$ICs,icBest=res$ICs[ic],
                             nParam=res$nParam,FI=FI,logLik=res$logLik,xreg=xreg,
                             matFX=matFX,matGX=matGX,nExovars=nExovars);
        # if(xregDo!="u"){
        #     listToReturn <- XregSelector(listToReturn=listToReturn);
        # }

        return(listToReturn);
    }
    else{
        environment(CF) <- environment();
        environment(ICFunction) <- environment();
        environment(likelihoodFunction) <- environment();
        environment(BasicMakerVES) <- environment();
        BasicMakerVES(ParentEnvironment=environment());

        C <- c(matG);
        # if(damped){
        #     C <- c(C,phi);
        # }
        C <- c(C,initialValue,initialSeason);
        if(xregEstimate){
            C <- c(C,initialX);
            if(updateX){
                C <- c(C,transitionX,persistenceX);
            }
        }

        cfObjective <- CF(C);

        # Number of parameters
        nParam <- 1 + nComponents + damped + (nComponents + (maxlag-1) * (Stype!="N")) * (initialType!="b") + !is.null(xreg) * nExovars + (updateX)*(nExovars^2 + nExovars);

# Change cfType for model selection
        if(multisteps){
            if(substring(cfType,1,1)=="a"){
                cfType <- "aTFL";
            }
            else{
                cfType <- "TFL";
            }
        }
        else{
            cfType <- "MSE";
        }

        ICValues <- ICFunction(nParam=nParam+nParamIntermittent,C=C,Etype=Etype);
        logLik <- ICValues$llikelihood;
        ICs <- ICValues$ICs;
        icBest <- ICs[ic];
        # Change back
        cfType <- cfTypeOriginal;

        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,
                             cfObjective=cfObjective,C=C,ICs=ICs,icBest=icBest,
                             nParam=nParam,FI=FI,logLik=logLik,xreg=xreg,
                             matFX=matFX,matGX=matGX,nExovars=nExovars);
        return(listToReturn);
    }
}

##### Set initialstates, initialSesons and persistence vector #####
    # If initial values are provided, write them. If not, estimate them.
    # First two columns are needed for additive seasonality, the 3rd and 4th - for the multiplicative
    if(Ttype!="N"){
        if(initialType!="p"){
            initialstates <- matrix(NA,1,4);
            initialstates[1,2] <- cov(yot[1:min(12,obsNonzero)],c(1:min(12,obsNonzero)))/var(c(1:min(12,obsNonzero)));
            initialstates[1,1] <- mean(yot[1:min(12,obsNonzero)]) - initialstates[1,2] * mean(c(1:min(12,obsNonzero)));
            if(allowMultiplicative){
                initialstates[1,4] <- exp(cov(log(yot[1:min(12,obsNonzero)]),c(1:min(12,obsNonzero)))/var(c(1:min(12,obsNonzero))));
                initialstates[1,3] <- exp(mean(log(yot[1:min(12,obsNonzero)])) - log(initialstates[1,4]) * mean(c(1:min(12,obsNonzero))));
            }
        }
        else{
            initialstates <- matrix(rep(initialValue,2),nrow=1);
        }
    }
    else{
        if(initialType!="p"){
            initialstates <- matrix(rep(mean(yot[1:min(12,obsNonzero)]),4),nrow=1);
        }
        else{
            initialstates <- matrix(rep(initialValue,4),nrow=1);
        }
    }

    # Define matrix of seasonal coefficients. The first column consists of additive, the second - multiplicative elements
    # If the seasonal model is chosen and initials are provided, fill in the first "maxlag" values of seasonal component.
    if(Stype!="N"){
        if(is.null(initialSeason)){
            initialSeasonEstimate <- TRUE;
            seasonalCoefs <- decompose(ts(c(y),frequency=datafreq),type="additive")$seasonal[1:datafreq];
            seasonalCoefs <- cbind(seasonalCoefs,decompose(ts(c(y),frequency=datafreq),
                                                           type="multiplicative")$seasonal[1:datafreq]);
        }
        else{
            initialSeasonEstimate <- FALSE;
            seasonalCoefs <- cbind(initialSeason,initialSeason);
        }
    }
    else{
        initialSeasonEstimate <- FALSE;
        seasonalCoefs <- matrix(1,1,1);
    }

    # If the persistence vector is provided, use it
    if(!is.null(persistence)){
        smoothingParameters <- cbind(persistence,persistence);
    }
    else{
        # smoothingParameters <- cbind(c(0.2,0.1,0.05),rep(0.05,3));
        smoothingParameters <- cbind(c(0.3,0.2,0.1),c(0.1,0.05,0.01));
    }

##### Preset y.fit, y.for, errors and basic parameters #####
    y.fit <- rep(NA,obsInsample);
    y.for <- rep(NA,h);
    errors <- rep(NA,obsInsample);

    basicparams <- initparams(Ttype, Stype, datafreq, obsInsample, obsAll, y,
                              damped, smoothingParameters, initialstates, seasonalCoefs);

##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                    nParamMax," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
    }

##### Define modelDo #####
    if(any(persistenceEstimate, transitionEstimate, dampedEstimate, initialEstimate, initialSeasonEstimate,
           FXEstimate, gXEstimate, initialXEstimate)){
        modelDo <- "estimate";
        modelCurrent <- model;
    }
    else{
        modelDo <- "nothing";
    }

##### Now do estimation and model selection #####
    # environment(intermittentParametersSetter) <- environment();
    # environment(intermittentMaker) <- environment();
    environment(BasicInitialiserVES) <- environment();
    environment(vssFitter) <- environment();
    environment(vssForecaster) <- environment();

    EtypeOriginal <- Etype;
    TtypeOriginal <- Ttype;
    StypeOriginal <- Stype;
# If auto intermittent, then estimate model with intermittent="n" first.
    # if(any(intermittent==c("a","n"))){
    #     intermittentParametersSetter(intermittent="n",ParentEnvironment=environment());
    #     if(intermittent=="a"){
    #         if(Etype=="M"){
    #             Etype <- "A";
    #         }
    #         if(Ttype=="M"){
    #             Ttype <- "A";
    #         }
    #         if(Stype=="M"){
    #             Stype <- "A";
    #         }
    #     }
    # }
    # else{
    #     intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
    #     intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    # }
    esValues <- CreatorVES(silent=silentText);

##### If intermittent=="a", run a loop and select the best one #####
#     if(intermittent=="a"){
#         Etype <- EtypeOriginal;
#         Ttype <- TtypeOriginal;
#         Stype <- StypeOriginal;
#         if(cfType!="MSE"){
#             warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. A wrong intermittent model may be selected"),call.=FALSE);
#         }
#         if(!silentText){
#             cat("Selecting appropriate type of intermittency... ");
#         }
# # Prepare stuff for intermittency selection
#         intermittentModelsPool <- c("n","f","c","t","s");
#         intermittentCFs <- intermittentICs <- rep(NA,length(intermittentModelsPool));
#         intermittentModelsList <- list(NA);
#         intermittentICs[1] <- esValues$icBest;
#         intermittentCFs[1] <- esValues$cfObjective;
#
#         for(i in 2:length(intermittentModelsPool)){
#             intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
#             intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
#             intermittentModelsList[[i]] <- CreatorVES(silent=TRUE);
#             intermittentICs[i] <- intermittentModelsList[[i]]$icBest;
#             intermittentCFs[i] <- intermittentModelsList[[i]]$cfObjective;
#         }
#         intermittentICs[is.nan(intermittentICs) | is.na(intermittentICs)] <- 1e+100;
#         intermittentCFs[is.nan(intermittentCFs) | is.na(intermittentCFs)] <- 1e+100;
#         # In cases when the data is binary, choose between intermittent models only
#         if(any(intermittentCFs==0)){
#             if(all(intermittentCFs[2:length(intermittentModelsPool)]==0)){
#                 intermittentICs[1] <- Inf;
#             }
#         }
#         iBest <- which(intermittentICs==min(intermittentICs))[1];
#
#         if(!silentText){
#             cat("Done!\n");
#         }
#         if(iBest!=1){
#             intermittent <- intermittentModelsPool[iBest];
#             intermittentModel <- intermittentModelsList[[iBest]];
#             esValues <- intermittentModelsList[[iBest]];
#         }
#         else{
#             intermittent <- "n"
#         }
#
#         intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
#         intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
#     }

##### Fit the model and produce forecast #####
        list2env(esValues,environment());
        BasicMakerVES(ParentEnvironment=environment());

        # if(!is.null(xregNames)){
        #     matat <- as.matrix(matatOriginal[,xregNames]);
        #     matxt <- as.matrix(matxtOriginal[,xregNames]);
        #     if(ncol(matat)==1){
        #         colnames(matxt) <- xregNames;
        #     }
        #     xreg <- matxt;
        # }
        # else{
        xreg <- NULL;
        # }
        BasicInitialiserVES(ParentEnvironment=environment());

        if(damped){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        else{
            model <- paste0(Etype,Ttype,Stype);
        }

# Write down Fisher Information if needed
        if(FI){
            environment(likelihoodFunction) <- environment();
            FI <- numDeriv::hessian(likelihoodFunction,C);
        }

        ##### These functions need to be rewrittent #####
        vssFitter(ParentEnvironment=environment());
        vssForecaster(ParentEnvironment=environment());

        component.names <- "level";
        if(Ttype!="N"){
            component.names <- c(component.names,"trend");
        }
        if(Stype!="N"){
            component.names <- c(component.names,"seasonality");
        }

        colnames(matvt) <- c(component.names);

# Write down the initials. Done especially for Nikos and issue #10
        if(persistenceEstimate){
            persistence <- as.vector(matG);
        }
        if(Ttype!="N"){
            names(persistence) <- c("alpha","beta","gamma")[1:nComponents];
        }
        else{
            names(persistence) <- c("alpha","gamma")[1:nComponents];
        }

        if(initialType!="p"){
            initialValue <- matvt[maxlag,1:(nComponents - (Stype!="N"))];
        }

        if(initialXEstimate){
            initialX <- matat[1,];
            names(initialX) <- colnames(matat);
        }

        if(initialSeasonEstimate){
            if(Stype!="N"){
                initialSeason <- matvt[1:maxlag,nComponents];
                names(initialSeason) <- paste0("s",1:maxlag);
            }
        }

# Write down the formula of ETS
        # esFormula <- "l[t-1]";
        # if(Ttype=="A"){
        #     esFormula <- paste0(esFormula," + b[t-1]");
        # }
        # else if(Ttype=="M"){
        #     esFormula <- paste0(esFormula," * b[t-1]");
        # }
        # if(Stype=="A"){
        #     esFormula <- paste0(esFormula," + s[t-",maxlag,"]");
        # }
        # else if(Stype=="M"){
        #     if(Ttype=="A"){
        #         esFormula <- paste0("(",esFormula,")");
        #     }
        #     esFormula <- paste0(esFormula," * s[t-",maxlag,"]");
        # }
        # if(Etype=="A"){
        #     if(!is.null(xreg)){
        #         if(updateX){
        #             esFormula <- paste0(esFormula," + ",paste0(paste0("a",c(1:nExovars),"[t-1] * "),paste0(xregNames,"[t]"),collapse=" + "));
        #         }
        #         else{
        #             esFormula <- paste0(esFormula," + ",paste0(paste0("a",c(1:nExovars)," * "),paste0(xregNames,"[t]"),collapse=" + "));
        #         }
        #     }
        #     esFormula <- paste0(esFormula," + e[t]");
        # }
        # else{
        #     if(any(c(Ttype,Stype)=="A") & Stype!="M"){
        #         esFormula <- paste0("(",esFormula,")");
        #     }
        #     if(!is.null(xreg)){
        #         if(updateX){
        #             esFormula <- paste0(esFormula," * exp(",paste0(paste0("a",c(1:nExovars),"[t-1] * "),paste0(xregNames,"[t]"),collapse=" + "),")");
        #         }
        #         else{
        #             esFormula <- paste0(esFormula," * exp(",paste0(paste0("a",c(1:nExovars)," * "),paste0(xregNames,"[t]"),collapse=" + "),")");
        #         }
        #     }
        #     esFormula <- paste0(esFormula," * e[t]");
        # }
        # if(intermittent!="n"){
        #     esFormula <- paste0("o[t] * (",esFormula,")");
        # }
        # esFormula <- paste0("y[t] = ",esFormula);

##### Do final check and make some preparations for output #####

    # Write down the probabilities from intermittent models
    # pt <- ts(c(as.vector(pt),as.vector(pt.for)),start=start(data),frequency=datafreq);
    # if(intermittent=="f"){
    #     intermittent <- "fixed";
    # }
    # else if(intermittent=="c"){
    #     intermittent <- "croston";
    # }
    # else if(intermittent=="t"){
    #     intermittent <- "tsb";
    # }
    # else if(intermittent=="n"){
    #     intermittent <- "none";
    # }
    # else if(intermittent=="p"){
    #     intermittent <- "provided";
    # }

##### Now let's deal with the holdout #####
    if(holdout){
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(y.for),frequency=frequency(data));
        errormeasures <- NA;
        # errormeasures <- errorMeasurer(y.holdout,y.for,y);

# Add PLS
        # errormeasuresNames <- names(errormeasures);
        # if(all(intermittent!=c("n","none"))){
        #     errormeasures <- c(errormeasures, suppressWarnings(pls(actuals=y.holdout, forecasts=y.for, Etype=Etype,
        #                                                            sigma=s2, trace=FALSE, iprob=pt[obsInsample+c(1:h)])));
        # }
        # else{
        #     if(multisteps){
        #         sigma <- t(errors.mat) %*% errors.mat / obsInsample;
        #     }
        #     else{
        #         sigma <- s2;
        #     }
        #     errormeasures <- c(errormeasures, suppressWarnings(pls(actuals=y.holdout, forecasts=y.for, Etype=Etype,
        #                                                            sigma=sigma, trace=multisteps, iprob=pt[obsInsample+c(1:h)])));
        # }
        # names(errormeasures) <- c(errormeasuresNames,"PLS");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    if(!is.null(xreg)){
        modelname <- "VESX";
    }
    else{
        modelname <- "VES";
    }
    modelname <- paste0(modelname,"(",model,")");
    # if(all(intermittent!=c("n","none"))){
    #     modelname <- paste0("i",modelname);
    # }

##### Print output #####
    if(!silentText){
        if(any(abs(eigen(matF - matG %*% matw)$values)>(1 + 1E-10))){
            warning(paste0("Model ETS(",model,") is unstable! Use a different value of 'bounds' parameter to address this issue!"),
                    call.=FALSE);
        }
    }

##### Make a plot #####
    # if(!silentGraph){
    #     if(intervals){
    #         graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
    #                    level=level,legend=!silentLegend,main=modelname);
    #     }
    #     else{
    #         graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
    #                 level=level,legend=!silentLegend,main=modelname);
    #     }
    # }

    ##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,persistence=persistence,
                  initialType=initialType,initial=initialValue,initialSeason=initialSeason,
                  nParam=nParam,
                  fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                  errors=errors.mat,s2=s2,intervals=intervalsType,level=level,
                  actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                  xreg=xreg,updateX=updateX,initialX=initialX,persistenceX=matGX,transitionX=matFX,
                  ICs=ICs,logLik=logLik,cf=cfObjective,cfType=cfType,FI=FI,accuracy=errormeasures);
    return(structure(model,class="smooth"));
}
