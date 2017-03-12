utils::globalVariables(c("vecg","nComponents","modellags","phiEstimate","y","datafreq","initialType",
                         "yot","maxlag","silent","allowMultiplicative","current.model",
                         "nParamIntermittent","cfTypeOriginal","matF","matw","pt.for","errors.mat",
                         "iprob","results","s2","FI","intermittent","normalizer",
                         "persistenceEstimate","initial","multisteps","ot",
                         "silentText","silentGraph","silentLegend"));

#' Exponential Smoothing in SSOE state-space model
#'
#' Function constructs ETS model and returns forecast, fitted values, errors
#' and matrix of states.
#'
#' Function estimates ETS in a form of the Single Source of Error State-space
#' model of the following type:
#'
#' \eqn{y_[t] = o_[t] (w(v_[t-l]) + x_t a_[t-1] + r(v_[t-l]) \epsilon_[t])}
#'
#' \eqn{v_[t] = f(v_[t-l]) + g(v_[t-l]) \epsilon_[t]}
#'
#' \eqn{a_[t] = F_[X] a_[t-1] + g_[X] \epsilon_[t] / x_[t]}
#'
#' Where \eqn{o_[t]} is Bernoulli distributed random variable (in case of
#' normal data it equals to 1 for all observations), \eqn{v_[t]} is a state
#' vector and \eqn{l} is a vector of lags, \eqn{x_t} vector of exogenous
#' parameters. w(.) is measurement function, r(.) is an error function, f(.) is
#' transition function and g(.) is persistence function.
#'
#' For the details see Hyndman et al.(2008).
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssPersistenceParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#' @template ssIntermittentRef
#' @template ssETSRef
#' @template ssIntervalsRef
#'
#' @param model The type of ETS model. Can consist of 3 or 4 chars: \code{ANN},
#' \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA}, \code{MAdM} etc.
#' \code{ZZZ} means that the model will be selected based on the chosen
#' information criteria type. Models pool can be restricted with additive only
#' components. This is done via \code{model="XXX"}. For example, making
#' selection between models with none / additive / damped additive trend
#' component only (i.e. excluding multiplicative trend) can be done with
#' \code{model="ZXZ"}. Furthermore, selection between multiplicative models
#' (excluding additive components) is regulated using \code{model="YYY"}. This
#' can be useful for positive data with low values (for example, slow moving
#' products). Finally, if \code{model="CCC"}, then all the models are estimated
#' and combination of their forecasts using AIC weights is produced (Kolassa,
#' 2011). This can also be regulated. For example, \code{model="CCN"} will
#' combine forecasts of all non-seasonal models and \code{model="CXY"} will
#' combine forecasts of all the models with non-multiplicative trend and
#' non-additive seasonality with either additive or multiplicative error. not
#' sure why anyone whould need this thing, but it is available.
#'
#' The parameter \code{model} can also be a vector of names of models for a
#' finer tuning (pool of models). For example, \code{model=c("ANN","AAA")} will
#' estimate only two models and select the best of them.
#'
#' Also \code{model} can accept a previously estimated ES model and use all its
#' parameters.
#'
#' Keep in mind that model selection with "Z" components uses Branch and Bound
#' algorithm and may skip some models that could have slightly smaller
#' information criteria.
#' @param phi Value of damping parameter. If \code{NULL} then it is estimated.
#' @param initial Can be either character or a vector of initial states. If it
#' is character, then it can be \code{"optimal"}, meaning that the initial
#' states are optimised, or \code{"backcasting"}, meaning that the initials are
#' produced using backcasting procedure (advised for data with high frequency).
#' If character, then \code{initialSeason} will be estimated in the way defined
#' by \code{initial}.
#' @param initialSeason Vector of initial values for seasonal components. If
#' \code{NULL}, they are estimated during optimisation.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of smoothing parameters and initial states of
#' the model.
#' Parameters \code{C}, \code{CLower} and \code{CUpper} can be passed via
#' ellipsis as well. In this case they will be used for optimisation. \code{C}
#' sets the initial values before the optimisation, \code{CLower} and
#' \code{CUpper} define lower and upper bounds for the search inside of the
#' specified \code{bounds}. These values should have exactly the length equal
#' to the number of parameters to estimate.
#' @return Object of class "smooth" is returned. It contains the list of the
#' following values for clasical ETS models:
#'
#' \itemize{
#' \item \code{model} - type of constructed model.
#' \item \code{formula} - mathematical formula, describing interations between
#' components of es() and exogenous variables.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{states} - matrix of the components of ETS.
#' \item \code{persistence} - persistence vector. This is the place, where
#' smoothing parameters live.
#' \item \code{phi} - value of damping parameter.
#' \item \code{initialType} - Type of initial values used.
#' \item \code{initial} - intial values of the state vector (non-seasonal).
#' \item \code{initialSeason} - intial values of the seasonal part of state vector.
#' \item \code{nParam} - number of estimated parameters.
#' \item \code{fitted} - fitted values of ETS.
#' \item \code{forecast} - point forecast of ETS.
#' \item \code{lower} - lower bound of prediction interval. When \code{intervals="none"}
#' then NA is returned.
#' \item \code{upper} - higher bound of prediction interval. When \code{intervals="none"}
#' then NA is returned.
#' \item \code{residuals} - residuals of the estimated model.
#' \item \code{errors} - trace forecast in-sample errors, returned as a matrix. In the
#' case of trace forecasts this is the matrix used in optimisation. In non-trace estimations
#' it is returned just for the information.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom into account).
#' \item \code{intervals} - type of intervals asked by user.
#' \item \code{level} - confidence level for intervals.
#' \item \code{actuals} - original data.
#' \item \code{holdout} - holdout part of the original data.
#' \item \code{iprob} - fitted and forecasted values of the probability of demand occurrence.
#' \item \code{intermittent} - type of intermittent model fitted to the data.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If \code{xregDo="s"},
#' then this value will contain only selected exogenous variables.
#' \item \code{updateX} - boolean, defining, if the states of exogenous variables were
#' estimated as well.
#' \item \code{initialX} - initial values for parameters of exogenous variables.
#' \item \code{persistenceX} - persistence vector g for exogenous variables.
#' \item \code{transitionX} - transition matrix F for exogenous variables.
#' \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc and BIC.
#' \item \code{logLik} - log-likelihood of the function.
#' \item \code{cf} - cost function value.
#' \item \code{cfType} - type of cost function used in the estimation.
#' \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE} or when \code{FI}
#' is not provided at all.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample. In
#' case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.
#' }
#'
#' If combination of forecasts is produced (using \code{model="CCC"}), then a
#' shorter list of values is returned:
#'
#' \itemize{
#' \item \code{model},
#' \item \code{timeElapsed},
#' \item \code{initialType},
#' \item \code{fitted},
#' \item \code{forecast},
#' \item \code{lower},
#' \item \code{upper},
#' \item \code{residuals},
#' \item \code{s2} - variance of additive error of combined one-step-ahead forecasts,
#' \item \code{intervals},
#' \item \code{level},
#' \item \code{actuals},
#' \item \code{holdout},
#' \item \code{iprob},
#' \item \code{intermittent},
#' \item \code{ICs} - combined ic,
#' \item \code{ICw} - ic weights used in the combination,
#' \item \code{cfType},
#' \item \code{xreg},
#' \item \code{accuracy}.
#' }
#' @seealso \code{\link[forecast]{ets}, \link[forecast]{forecast},
#' \link[stats]{ts}, \link[smooth]{sim.es}}
#'
#' @examples
#'
#' library(Mcomp)
#'
#' # See how holdout and trace parameters influence the forecast
#' es(M3$N1245$x,model="AAdN",h=8,holdout=FALSE,cfType="MSE")
#' \dontrun{es(M3$N2568$x,model="MAM",h=18,holdout=TRUE,cfType="MSTFE")}
#'
#' # Model selection example
#' es(M3$N1245$x,model="ZZN",ic="AIC",h=8,holdout=FALSE,bounds="a")
#'
#' # Model selection. Compare AICc of these two models:
#' \dontrun{es(M3$N1683$x,"ZZZ",h=10,holdout=TRUE)
#' es(M3$N1683$x,"MAdM",h=10,holdout=TRUE)}
#'
#' # Model selection, excluding multiplicative trend
#' \dontrun{es(M3$N1245$x,model="ZXZ",h=8,holdout=TRUE)}
#'
#' # Combination example
#' \dontrun{es(M3$N1245$x,model="CCN",h=8,holdout=TRUE)}
#'
#' # Model selection using a specified pool of models
#' ourModel <- es(M3$N1587$x,model=c("ANN","AAM","AMdA"),h=18)
#'
#' # Redo previous model and produce prediction intervals
#' es(M3$N1587$x,model=ourModel,h=18,intervals="p")
#'
#' # Semiparametric intervals example
#' \dontrun{es(M3$N1587$x,h=18,holdout=TRUE,intervals="sp")}
#'
#' # Exogenous variables in ETS example
#' \dontrun{x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)))
#' y <- ts(c(M3$N1457$x,M3$N1457$xx),frequency=12)
#' es(y,h=18,holdout=TRUE,xreg=x,cfType="aMSTFE",intervals="np")
#' ourModel <- es(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,xreg=x,updateX=TRUE)}
#'
#' # This will be the same model as in previous line but estimated on new portion of data
#' \dontrun{es(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),model=ourModel,h=18,holdout=FALSE)}
#'
#' # Intermittent data example
#' x <- rpois(100,0.2)
#' # Croston's method with the best ETS for demand sizes
#' es(x,"ZZN",intermittent="croston")
#' # TSB based on iETS(M,N,N)
#' es(x,"MNN",intermittent="tsb")
#' # Constant probability based on iETS(M,N,N)
#' es(x,"MNN",intermittent="fixed")
#' # Best type of intermittent model based on iETS(Z,Z,N)
#' ourModel <- es(x,"ZZN",intermittent="auto")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#' @export es
es <- function(data, model="ZZZ", persistence=NULL, phi=NULL,
               initial=c("optimal","backcasting"), initialSeason=NULL, ic=c("AICc","AIC","BIC"),
               cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
               h=10, holdout=FALSE,
               intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
               intermittent=c("none","auto","fixed","croston","tsb","sba"),
               bounds=c("usual","admissible","none"),
               silent=c("none","all","graph","legend","output"),
               xreg=NULL, xregDo=c("use","select"), initialX=NULL,
               updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Copyright (C) 2015 - 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# If a previous model provided as a model, write down the variables
    if(is.list(model)){
        if(gregexpr("ETS",model$model)==-1){
            stop("The provided model is not ETS.",call.=FALSE);
        }
        intermittent <- model$intermittent;
        if(any(intermittent==c("p","provided"))){
            warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
            warning("Switching to intermittent='auto'.",call.=FALSE);
            intermittent <- "a";
        }
        persistence <- model$persistence;
        initial <- model$initial;
        initialSeason <- model$initialSeason;
        if(is.null(xreg)){
            xreg <- model$xreg;
        }
        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        phi <- model$phi;
        if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
            updateX <- TRUE;
        }
        model <- model$model;
        model <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        if(any(unlist(gregexpr("C",model))!=-1)){
            initial <- "o";
        }
    }

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput(modelType="es",ParentEnvironment=environment());

##### Cost Function for ES #####
CF <- function(C){
    elements <- etsmatrices(matvt, vecg, phi, matrix(C,nrow=1), nComponents,
                            modellags, initialType, Ttype, Stype, nExovars, matat,
                            persistenceEstimate, phiEstimate, initialType=="o", initialSeasonEstimate, xregEstimate,
                            matFX, vecgX, updateX, FXEstimate, gXEstimate, initialXEstimate);

    cfRes <- costfunc(elements$matvt, elements$matF, elements$matw, y, elements$vecg,
                      h, modellags, Etype, Ttype, Stype,
                      multisteps, cfType, normalizer, initialType,
                      matxt, elements$matat, elements$matFX, elements$vecgX, ot,
                      bounds);

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}

##### C values for estimation #####
# Function constructs default bounds where C values should lie
CValues <- function(bounds,Ttype,Stype,vecg,matvt,phi,maxlag,nComponents,matat){
    C <- NA;
    CLower <- NA;
    CUpper <- NA;

    if(bounds=="u"){
        if(persistenceEstimate){
            C <- c(C,vecg);
            CLower <- c(CLower,rep(0,length(vecg)));
            CUpper <- c(CUpper,rep(1,length(vecg)));
        }
        if(damped & phiEstimate){
            C <- c(C,phi);
            CLower <- c(CLower,0);
            CUpper <- c(CUpper,1);
        }
        if(any(initialType==c("o","p"))){
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
            if(Stype!="N"){
                if(initialSeasonEstimate){
                    C <- c(C,matvt[1:maxlag,nComponents]);
                    if(Stype=="A"){
                        CLower <- c(CLower,rep(-Inf,maxlag));
                        CUpper <- c(CUpper,rep(Inf,maxlag));
                    }
                    else{
                        CLower <- c(CLower,rep(1e-5,maxlag));
                        CUpper <- c(CUpper,rep(10,maxlag));
                    }
                }
            }
        }
    }
    else if(bounds=="a"){
        if(persistenceEstimate){
            C <- c(C,vecg);
            CLower <- c(CLower,rep(-5,length(vecg)));
            CUpper <- c(CUpper,rep(5,length(vecg)));
        }
        if(damped & phiEstimate){
            C <- c(C,phi);
            CLower <- c(CLower,0);
            CUpper <- c(CUpper,1);
        }
        if(any(initialType==c("o","p"))){
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
            if(Stype!="N"){
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
            }
        }
    }
    else{
        if(persistenceEstimate){
            C <- c(C,vecg);
            CLower <- c(CLower,rep(-Inf,length(vecg)));
            CUpper <- c(CUpper,rep(Inf,length(vecg)));
        }
        if(damped & phiEstimate){
            C <- c(C,phi);
            CLower <- c(CLower,-Inf);
            CUpper <- c(CUpper,Inf);
        }
        if(any(initialType==c("o","p"))){
            if(initialType=="o"){
                C <- c(C,matvt[maxlag,1:(nComponents - (Stype!="N"))]);
                if(Ttype!="M"){
                    CLower <- c(CLower,rep(-Inf,(nComponents - (Stype!="N"))));
                    CUpper <- c(CUpper,rep(Inf,(nComponents - (Stype!="N"))));
                }
                else{
                    CLower <- c(CLower,-Inf,-Inf);
                    CUpper <- c(CUpper,Inf,Inf);
                }
            }
            if(Stype!="N"){
                if(initialSeasonEstimate){
                    C <- c(C,matvt[1:maxlag,nComponents]);
                    if(Stype=="A"){
                        CLower <- c(CLower,rep(-Inf,maxlag));
                        CUpper <- c(CUpper,rep(Inf,maxlag));
                    }
                    else{
                        CLower <- c(CLower,rep(-Inf,maxlag));
                        CUpper <- c(CUpper,rep(Inf,maxlag));
                    }
                }
            }
        }
    }

    if(xregEstimate){
        if(initialXEstimate){
            if(Etype=="A" & modelDo!="estimate"){
                vecat <- matrix(y[2:obsInsample],nrow=obsInsample-1,ncol=ncol(matxt)) / diff(matxt[1:obsInsample,]);
                vecat[is.infinite(vecat)] <- NA;
                vecat <- colSums(vecat,na.rm=T);
            }
            else{
                vecat <- matat[maxlag,];
            }

            C <- c(C,vecat);
            CLower <- c(CLower,rep(-Inf,nExovars));
            CUpper <- c(CUpper,rep(Inf,nExovars));
        }
        if(updateX){
            if(FXEstimate){
                C <- c(C,as.vector(matFX));
                CLower <- c(CLower,rep(-Inf,nExovars^2));
                CUpper <- c(CUpper,rep(Inf,nExovars^2));
            }
            if(gXEstimate){
                C <- c(C,as.vector(vecgX));
                CLower <- c(CLower,rep(-Inf,nExovars));
                CUpper <- c(CUpper,rep(Inf,nExovars));
            }
        }
    }

    C <- C[!is.na(C)];
    CLower <- CLower[!is.na(CLower)];
    CUpper <- CUpper[!is.na(CUpper)];

    return(list(C=C,CLower=CLower,CUpper=CUpper));
}

##### Basic parameter propagator #####
BasicMakerES <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    basicparams <- initparams(Ttype, Stype, datafreq, obsInsample, obsAll, y,
                              damped, phi, smoothingParameters, initialstates, seasonalCoefs);
    list2env(basicparams,ParentEnvironment);
}

##### Basic parameter propagator #####
BasicInitialiserES <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    elements <- etsmatrices(matvt, vecg, phi, matrix(C,nrow=1), nComponents,
                            modellags, initialType, Ttype, Stype, nExovars, matat,
                            persistenceEstimate, phiEstimate, initialType=="o", initialSeasonEstimate, xregEstimate,
                            matFX, vecgX, updateX, FXEstimate, gXEstimate, initialXEstimate);

    list2env(elements,ParentEnvironment);
}

##### Basic estimation function for es() #####
EstimatorES <- function(...){
    environment(BasicMakerES) <- environment();
    environment(CValues) <- environment();
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();
    environment(CF) <- environment();
    BasicMakerES(ParentEnvironment=environment());

    Cs <- CValues(bounds,Ttype,Stype,vecg,matvt,phi,maxlag,nComponents,matat);
    if(is.null(providedC)){
        C <- Cs$C;
    }
    if(is.null(providedCLower)){
        CLower <- Cs$CLower;
    }
    if(is.null(providedCUpper)){
        CUpper <- Cs$CUpeer;
    }

    # Parameters are chosen to speed up the optimisation process and have decent accuracy
    res <- nloptr(C, CF, lb=CLower, ub=CUpper,
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
        res <- nloptr(C, CF, lb=CLower, ub=CUpper,
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=500));
        C <- res$solution;
    }

    if(any((C>=CUpper),(C<=CLower))){
        C[C>=CUpper] <- CUpper[C>=CUpper] * 0.999 - 0.001;
        C[C<=CLower] <- CLower[C<=CLower] * 1.001 + 0.001;
    }

    res2 <- nloptr(C, CF, lb=CLower, ub=CUpper,
                  opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-6, "maxeval"=500));
    # This condition is needed in order to make sure that we did not make the solution worse
    if(res2$objective <= res$objective){
        res <- res2;
    }
    C <- res$solution;

    if(all(C==Cs$C) & (initialType!="b")){
        if(any(persistenceEstimate,gXEstimate,FXEstimate)){
            warning(paste0("Failed to optimise the model ETS(", current.model,
                           "). Try different initialisation maybe?\nAnd check all the messages and warnings...",
                           "If you did your best, but the optimiser still fails, report this to the maintainer, please."),
                    call.=FALSE);
        }
    }

    nParam <- 1 + nComponents + damped + (nComponents + (maxlag - 1) * (Stype!="N")) * (initialType!="b") + (!is.null(xreg)) * nExovars + (updateX)*(nExovars^2 + nExovars);

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

##### This function uses residuals in order to determine the needed xreg #####
XregSelector <- function(listToReturn){
# Prepare for fitting
    environment(BasicMakerES) <- environment();
    environment(BasicInitialiserES) <- environment();
    environment(EstimatorES) <- environment();
    environment(ssFitter) <- environment();
    list2env(listToReturn, environment());

    BasicMakerES(ParentEnvironment=environment());
    BasicInitialiserES(ParentEnvironment=environment());
    ssFitter(ParentEnvironment=environment());

    xregNames <- colnames(matxtOriginal);
    xregNew <- cbind(errors,xreg[1:nrow(errors),]);
    colnames(xregNew)[1] <- "errors";
    colnames(xregNew)[-1] <- xregNames;
    xregNew <- as.data.frame(xregNew);
    xregResults <- stepwise(xregNew, ic=ic, silent=TRUE, df=nParam+nParamIntermittent-1);
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
        xregNames <- NULL;
        listToReturn$xregEstimate <- xregEstimate;
    }

    if(!is.null(xreg)){
        res <- EstimatorES(ParentEnvironment=environment());
        icBest <- res$ICs[ic];
        logLik <- res$logLik;
        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                             cfObjective=res$objective,C=res$C,ICs=res$ICs,icBest=icBest,
                             nParam=res$nParam,FI=FI,logLik=logLik,xreg=xreg,xregEstimate=xregEstimate,
                             xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);
    }

    return(listToReturn);
}

##### This function prepares pool of models to use #####
PoolPreparerES <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];
    environment(EstimatorES) <- environment();

    if(!is.null(modelsPool)){
        modelsNumber <- length(modelsPool);

# List for the estimated models in the pool
        results <- as.list(c(1:modelsNumber));
        j <- 0;
    }
    else{
# Define the pool of models in case of "ZZZ" or "CCC" to select from
        if(!allowMultiplicative){
            if(!silent){
                message("Only additive models are allowed with non-positive data.");
            }
            errors.pool <- c("A");
            trends.pool <- c("N","A","Ad");
            season.pool <- c("N","A");
        }
        else{
            errors.pool <- c("A","M");
            trends.pool <- c("N","A","Ad","M","Md");
            season.pool <- c("N","A","M");
        }

        if(Etype!="Z"){
            errors.pool <- Etype;
        }

# List for the estimated models in the pool
        results <- list(NA);

### Use brains in order to define models to estimate ###
        if(modelDo=="select" & any(c(Ttype,Stype)=="Z")){
            if(!silent){
                cat("Forming the pool of models based on... ");
            }

# Some preparation variables
            if(Etype!="Z"){
                small.pool.error <- Etype;
                errors.pool <- Etype;
            }
            else{
                small.pool.error <- "A";
            }

            if(Ttype!="Z"){
                if(damped){
                    small.pool.trend <- paste0(Ttype,"d");
                    trends.pool <- small.pool.trend;
                }
                else{
                    small.pool.trend <- Ttype;
                    trends.pool <- Ttype;
                }
                check.T <- FALSE;
            }
            else{
                small.pool.trend <- c("N","A");
                check.T <- TRUE;
            }

            if(Stype!="Z"){
                small.pool.season <- Stype;
                season.pool <- Stype;
                check.S <- FALSE;
            }
            else{
                small.pool.season <- c("N","A","M");
                check.S <- TRUE;
            }

# If ZZZ, then the vector is: "ANN" "ANA" "ANM" "AAN" "AAA" "AAM"
            small.pool <- paste0(rep(small.pool.error,length(small.pool.trend)*length(small.pool.season)),
                                 rep(small.pool.trend,each=length(small.pool.season)),
                                 rep(small.pool.season,length(small.pool.trend)));
            tested.model <- NULL;

# Counter + checks for the components
            j <- 1;
            i <- 0;
            check <- TRUE;
            besti <- bestj <- 1;

#### Branch and bound is here ####
            while(check){
                i <- i + 1;
                current.model <- small.pool[j];
                if(!silent){
                    cat(paste0(current.model,", "));
                }
                Etype <- substring(current.model,1,1);
                Ttype <- substring(current.model,2,2);
                if(nchar(current.model)==4){
                    damped <- TRUE;
                    phi <- NULL;
                    Stype <- substring(current.model,4,4);
                }
                else{
                    damped <- FALSE;
                    phi <- 1;
                    Stype <- substring(current.model,3,3);
                }
                if(Stype!="N"){
                    initialSeasonEstimate <- TRUE;
                }
                else{
                    initialSeasonEstimate <- FALSE;
                }

                res <- EstimatorES(ParentEnvironment=environment());

                listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                                     cfObjective=res$objective,C=res$C,ICs=res$ICs,icBest=NULL,
                                     nParam=res$nParam,logLik=res$logLik,xreg=xreg,
                                     xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);

                if(xregDo!="u"){
                    listToReturn <- XregSelector(listToReturn=listToReturn);
                }
                results[[i]] <- listToReturn;

                tested.model <- c(tested.model,current.model);

                if(j>1){
# If the first is better than the second, then choose first
                    if(results[[besti]]$ICs[ic] <= results[[i]]$ICs[ic]){
# If Ttype is the same, then we checked seasonality
                        if(substring(current.model,2,2) == substring(small.pool[bestj],2,2)){
                            season.pool <- results[[besti]]$Stype;
                            check.S <- FALSE;
                            j <- which(small.pool!=small.pool[bestj] &
                                           substring(small.pool,nchar(small.pool),nchar(small.pool))==season.pool);
                        }
# Otherwise we checked trend
                        else{
                            trends.pool <- results[[bestj]]$Ttype;
                            check.T <- FALSE;
                        }
                    }
                    else{
                        if(substring(current.model,2,2) == substring(small.pool[besti],2,2)){
                            season.pool <- season.pool[season.pool!=results[[besti]]$Stype];
                            if(length(season.pool)>1){
# Select another seasonal model, that is not from the previous iteration and not the current one
                                bestj <- j;
                                besti <- i;
                                j <- 3;
                            }
                            else{
                                bestj <- j;
                                besti <- i;
                                j <- which(substring(small.pool,nchar(small.pool),nchar(small.pool))==season.pool &
                                          substring(small.pool,2,2)!=substring(current.model,2,2));
                                check.S <- FALSE;
                            }
                        }
                        else{
                            trends.pool <- trends.pool[trends.pool!=results[[bestj]]$Ttype];
                            besti <- i;
                            bestj <- j;
                            check.T <- FALSE;
                        }
                    }

                    if(all(!c(check.T,check.S))){
                        check <- FALSE;
                    }
                }
                else{
                    j <- 2;
                }
            }

            modelsPool <- paste0(rep(errors.pool,each=length(trends.pool)*length(season.pool)),
                                  trends.pool,
                                  rep(season.pool,each=length(trends.pool)));

            modelsPool <- unique(c(tested.model,modelsPool));
            modelsNumber <- length(modelsPool);
            j <- length(tested.model);
        }
        else{
# Make the corrections in the pool for combinations
            if(Etype!="Z"){
                errors.pool <- Etype;
            }
            if(Ttype!="Z"){
                if(damped){
                    trends.pool <- paste0(Ttype,"d");
                }
                else{
                    trends.pool <- Ttype;
                }
            }
            if(Stype!="Z"){
                season.pool <- Stype;
            }

            modelsNumber <- (length(errors.pool)*length(trends.pool)*length(season.pool));
            modelsPool <- paste0(rep(errors.pool,each=length(trends.pool)*length(season.pool)),
                                  trends.pool,
                                  rep(season.pool,each=length(trends.pool)));
            j <- 0;
        }
    }
    assign("modelsPool",modelsPool,ParentEnvironment);
    assign("modelsNumber",modelsNumber,ParentEnvironment);
    assign("j",j,ParentEnvironment);
    assign("results",results,ParentEnvironment);
}

##### Function for estimation of pool of models #####
PoolEstimatorES <- function(silent=FALSE,...){
    environment(EstimatorES) <- environment();
    environment(PoolPreparerES) <- environment();
    esPoolValues <- PoolPreparerES(ParentEnvironment=environment());

    if(!silent){
        cat("Estimation progress:    ");
    }
# Start loop of models
    while(j < modelsNumber){
        j <- j + 1;
        if(!silent){
            if(j==1){
                cat("\b");
            }
            cat(paste0(rep("\b",nchar(round((j-1)/modelsNumber,2)*100)+1),collapse=""));
            cat(paste0(round(j/modelsNumber,2)*100,"%"));
        }

        current.model <- modelsPool[j];
        Etype <- substring(current.model,1,1);
        Ttype <- substring(current.model,2,2);
        if(nchar(current.model)==4){
            damped <- TRUE;
            phi <- NULL;
            Stype <- substring(current.model,4,4);
        }
        else{
            damped <- FALSE;
            phi <- 1;
            Stype <- substring(current.model,3,3);
        }
        if(Stype!="N"){
            initialSeasonEstimate <- TRUE;
        }
        else{
            initialSeasonEstimate <- FALSE;
        }

        res <- EstimatorES(ParentEnvironment=environment());

        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                             cfObjective=res$objective,C=res$C,ICs=res$ICs,icBest=NULL,
                             nParam=res$nParam,logLik=res$logLik,xreg=xreg,
                             xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);
        if(xregDo!="u"){
            listToReturn <- XregSelector(listToReturn=listToReturn);
        }

        results[[j]] <- listToReturn;
    }

    if(!silent){
        cat("... Done! \n");
    }
    icSelection <- matrix(NA,modelsNumber,3);
    for(i in 1:modelsNumber){
        icSelection[i,] <- results[[i]]$ICs;
    }
    colnames(icSelection) <- names(results[[i]]$ICs);

    icSelection[is.nan(icSelection)] <- 1E100;

    return(list(results=results,icSelection=icSelection));
}

##### Function selects the best es() based on IC #####
CreatorES <- function(silent=FALSE,...){
    if(modelDo=="select"){
        if(cfType!="MSE"){
            warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. The results of model selection may be wrong."),call.=FALSE);
        }
        environment(PoolEstimatorES) <- environment();
        esPoolResults <- PoolEstimatorES(silent=silent);
        results <- esPoolResults$results;
        icSelection <- esPoolResults$icSelection;

        icBest <- min(icSelection[,ic]);
        i <- which(icSelection[,ic]==icBest)[1];
        ICs <- icSelection[i,];
        listToReturn <- results[[i]];
        listToReturn$icBest <- icBest;

        return(listToReturn);
    }
    else if(modelDo=="combine"){
        if(cfType!="MSE"){
            warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. The produced combinations weights may be wrong."),call.=FALSE);
        }
        environment(PoolEstimatorES) <- environment();
        esPoolResults <- PoolEstimatorES(silent=silent);
        results <- esPoolResults$results;
        icSelection <- esPoolResults$icSelection;
        icSelection <- icSelection[,ic];
        icBest <- min(icSelection);
        icSelection <- icSelection/(h^multisteps);
        icWeights <- exp(-0.5*(icSelection-icBest))/sum(exp(-0.5*(icSelection-icBest)));
        ICs <- sum(icSelection * icWeights);
        return(list(icWeights=icWeights,ICs=ICs,icBest=icBest,results=results));
    }
    else if(modelDo=="estimate"){
        environment(EstimatorES) <- environment();
        res <- EstimatorES(ParentEnvironment=environment());
        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                             cfObjective=res$objective,C=res$C,ICs=res$ICs,icBest=res$ICs[ic],
                             nParam=res$nParam,FI=FI,logLik=res$logLik,xreg=xreg,
                             xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);
        if(xregDo!="u"){
            listToReturn <- XregSelector(listToReturn=listToReturn);
        }

        return(listToReturn);
    }
    else{
        environment(CF) <- environment();
        environment(ICFunction) <- environment();
        environment(likelihoodFunction) <- environment();
        environment(BasicMakerES) <- environment();
        BasicMakerES(ParentEnvironment=environment());

        C <- c(vecg);
        if(damped){
            C <- c(C,phi);
        }
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

        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                             cfObjective=cfObjective,C=C,ICs=ICs,icBest=icBest,
                             nParam=nParam,FI=FI,logLik=logLik,xreg=xreg,
                             xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);
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
        smoothingParameters <- cbind(c(0.3,0.2,0.1),c(0.3,0.2,0.1));
    }

##### Preset y.fit, y.for, errors and basic parameters #####
    y.fit <- rep(NA,obsInsample);
    y.for <- rep(NA,h);
    errors <- rep(NA,obsInsample);

    basicparams <- initparams(Ttype, Stype, datafreq, obsInsample, obsAll, y,
                              damped, phi, smoothingParameters, initialstates, seasonalCoefs);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(data=data, Etype=Etype, xreg=xreg, updateX=updateX,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obsInsample=obsInsample, obsAll=obsAll, obsStates=obsStates, maxlag=basicparams$maxlag, h=h, silent=silentText);

    if(xregDo=="u"){
        nExovars <- xregdata$nExovars;
        matxtOriginal <- matxt <- xregdata$matxt;
        matatOriginal <- matat <- xregdata$matat;
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

    nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
    nParamMax <- nParamMax + nParamExo + (intermittent!="n");

##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        if(!silentText){
            message(paste0("Number of non-zero observations is ",obsNonzero,
                           ", while the maximum number of parameters to estimate is ", nParamMax,".\n",
                           "Updating pool of models."));
        }

        # We have enough observations for local level model
        if(obsNonzero > (3 + nParamExo) & is.null(modelsPool)){
            modelsPool <- c("ANN");
            if(allowMultiplicative){
                modelsPool <- c(modelsPool,"MNN");
            }
            # We have enough observations for trend model
            if(obsNonzero > (5 + nParamExo)){
                modelsPool <- c(modelsPool,"AAN");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"AMN","MAN","MMN");
                }
            }
            # We have enough observations for damped trend model
            if(obsNonzero > (6 + nParamExo)){
                modelsPool <- c(modelsPool,"AAdN");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"AMdN","MAdN","MMdN");
                }
            }
            # We have enough observations for seasonal model
            if((obsNonzero > (2*datafreq)) & datafreq!=1){
                modelsPool <- c(modelsPool,"ANA");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"ANM","MNA","MNM");
                }
            }
            # We have enough observations for seasonal model with trend
            if((obsNonzero > (6 + datafreq + nParamExo)) & (obsNonzero > 2*datafreq) & datafreq!=1){
                modelsPool <- c(modelsPool,"AAA");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"AAM","AMA","AMM","MAA","MAM","MMA","MMM");
                }
            }

            warning("Not enough observations for the fit of ETS(",model,")! Fitting what we can...",call.=FALSE);
            if(modelDo=="combine"){
                model <- "CNN";
                if(length(modelsPool)>2){
                    model <- "CCN";
                }
                if(length(modelsPool)>10){
                    model <- "CCC";
                }
            }
            else{
                model <- "ZZZ";
            }
        }
        else if(obsNonzero > (3 + nParamExo) & !is.null(modelsPool)){
            modelsPool.new <- modelsPool;
            # We don't have enough observations for seasonal models with damped trend
            if((obsNonzero <= (6 + datafreq + 1 + nParamExo))){
                modelsPool <- modelsPool[!(nchar(modelsPool)==4 &
                                               substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="A")];
                modelsPool <- modelsPool[!(nchar(modelsPool)==4 &
                                               substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="M")];
            }
            # We don't have enough observations for seasonal models with trend
            if((obsNonzero <= (5 + datafreq + 1 + nParamExo))){
                modelsPool <- modelsPool[!(substr(modelsPool,2,2)!="N" &
                                               substr(modelsPool,nchar(modelsPool),nchar(modelsPool))!="N")];
            }
            # We don't have enough observations for seasonal models
            if(obsNonzero <= 2*datafreq){
                modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N"];
            }
            # We don't have enough observations for damped trend
            if(obsNonzero <= (6 + nParamExo)){
                modelsPool <- modelsPool[nchar(modelsPool)!=4];
            }
            # We don't have enough observations for any trend
            if(obsNonzero <= (5 + nParamExo)){
                modelsPool <- modelsPool[substr(modelsPool,2,2)=="N"];
            }

            warning("Not enough observations for the fit of ETS(",model,")! Fitting what we can...",call.=FALSE,immediate.=TRUE);
            if(modelDo=="combine"){
                model <- "CNN";
                if(length(modelsPool)>2){
                    model <- "CCN";
                }
                if(length(modelsPool)>10){
                    model <- "CCC";
                }
            }
            else{
                model <- "ZZZ";
            }
        }
        else{
            stop("Not enough observations... Even for fitting of ETS('ANN')!",call.=FALSE);
        }
    }

##### Define modelDo #####
    if(any(persistenceEstimate, (initialType=="o"), initialSeasonEstimate*(initialType=="o"),
           phiEstimate, FXEstimate, gXEstimate, initialXEstimate)){
        if(all(modelDo!=c("select","combine"))){
            modelDo <- "estimate";
            current.model <- model;
        }
    }
    else{
        modelDo <- "nothing";
    }

    ellipsis <- list(...);
    providedC <- ellipsis$C;
    providedCLower <- ellipsis$CLower;
    providedCUpper <- ellipsis$CUpper;
##### Initials for optimiser #####
    if(!all(c(is.null(providedC),is.null(providedCLower),is.null(providedCUpper)))){
        if((modelDo==c("estimate")) & (xregDo==c("u"))){
            environment(BasicMakerES) <- environment();
            BasicMakerES(ParentEnvironment=environment());

            # Number of parameters
            nParam <- nComponents + damped + (nComponents + (maxlag-1) * (Stype!="N")) * (initialType!="b") + !is.null(xreg) * nExovars + (updateX)*(nExovars^2 + nExovars);
            if(nParam!=length(providedC)){
                warning(paste0("Number of parameters to optimise differes from the length of C:",nParam," vs ",length(providedC),".\n",
                               "We will have to drop parameter C."),call.=FALSE);
                providedC <- NULL;
            }
            if(nParam!=length(providedCLower)){
                warning(paste0("Number of parameters to optimise differes from the length of CLower:",nParam," vs ",length(providedCLower),".\n",
                               "We will have to drop parameter CLower."),call.=FALSE);
                providedCLower <- NULL;
            }
            if(nParam!=length(providedCUpper)){
                warning(paste0("Number of parameters to optimise differes from the length of CUpper:",nParam," vs ",length(providedCUpper),".\n",
                               "We will have to drop parameter CUpper."),call.=FALSE);
                providedCUpper <- NULL;
            }
            C <- providedC;
            CLower <- providedCLower;
            CUpper <- providedCUpper;
        }
        else{
            if(modelDo==c("select")){
                warning("Predefined values of C cannot be used with model selection.",call.=FALSE);
            }
            else if(modelDo==c("combine")){
                warning("Predefined values of C cannot be used with combination of forecasts.",call.=FALSE);
            }
            else if(modelDo==c("nothing")){
                warning("Sorry, but there is nothing to optimise, so we have to drop parameter C.",call.=FALSE);
            }

            if(xregDo==c("select")){
                warning("Predefined values of C cannot be used with xreg selection.",call.=FALSE);
            }
            C <- NULL;
            CLower <- NULL;
            CUpper <- NULL;
        }
    }

##### Now do estimation and model selection #####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(BasicInitialiserES) <- environment();
    environment(ssFitter) <- environment();
    environment(ssForecaster) <- environment();

    EtypeOriginal <- Etype;
    TtypeOriginal <- Ttype;
    StypeOriginal <- Stype;
# If auto intermittent, then estimate model with intermittent="n" first.
    if(any(intermittent==c("a","n"))){
        intermittentParametersSetter(intermittent="n",ParentEnvironment=environment());
        if(intermittent=="a"){
            if(Etype=="M"){
                Etype <- "A";
            }
            if(Ttype=="M"){
                Ttype <- "A";
            }
            if(Stype=="M"){
                Stype <- "A";
            }
        }
    }
    else{
        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }
    esValues <- CreatorES(silent=silentText);

##### If intermittent=="a", run a loop and select the best one #####
    if(intermittent=="a"){
        Etype <- EtypeOriginal;
        Ttype <- TtypeOriginal;
        Stype <- StypeOriginal;
        if(cfType!="MSE"){
            warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. A wrong intermittent model may be selected"),call.=FALSE);
        }
        if(!silentText){
            cat("Selecting appropriate type of intermittency... ");
        }
# Prepare stuff for intermittency selection
        intermittentModelsPool <- c("n","f","c","t","s");
        intermittentCFs <- intermittentICs <- rep(NA,length(intermittentModelsPool));
        intermittentModelsList <- list(NA);
        intermittentICs[1] <- esValues$icBest;
        intermittentCFs[1] <- esValues$cfObjective;

        for(i in 2:length(intermittentModelsPool)){
            intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentModelsList[[i]] <- CreatorES(silent=TRUE);
            intermittentICs[i] <- intermittentModelsList[[i]]$icBest;
            intermittentCFs[i] <- intermittentModelsList[[i]]$cfObjective;
        }
        intermittentICs[is.nan(intermittentICs) | is.na(intermittentICs)] <- 1e+100;
        intermittentCFs[is.nan(intermittentCFs) | is.na(intermittentCFs)] <- 1e+100;
        # In cases when the data is binary, choose between intermittent models only
        if(any(intermittentCFs==0)){
            if(all(intermittentCFs[2:length(intermittentModelsPool)]==0)){
                intermittentICs[1] <- Inf;
            }
        }
        iBest <- which(intermittentICs==min(intermittentICs))[1];

        if(!silentText){
            cat("Done!\n");
        }
        if(iBest!=1){
            intermittent <- intermittentModelsPool[iBest];
            intermittentModel <- intermittentModelsList[[iBest]];
            esValues <- intermittentModelsList[[iBest]];
        }
        else{
            intermittent <- "n"
        }

        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

##### Fit simple model and produce forecast #####
    if(modelDo!="combine"){
        list2env(esValues,environment());
        BasicMakerES(ParentEnvironment=environment());

        if(!is.null(xregNames)){
            matat <- as.matrix(matatOriginal[,xregNames]);
            matxt <- as.matrix(matxtOriginal[,xregNames]);
            if(ncol(matat)==1){
                colnames(matxt) <- xregNames;
            }
            xreg <- matxt;
        }
        else{
            xreg <- NULL;
        }
        BasicInitialiserES(ParentEnvironment=environment());
        if(!is.null(xregNames)){
            colnames(matat) <- xregNames;
        }

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

        ssFitter(ParentEnvironment=environment());
        ssForecaster(ParentEnvironment=environment());

        component.names <- "level";
        if(Ttype!="N"){
            component.names <- c(component.names,"trend");
        }
        if(Stype!="N"){
            component.names <- c(component.names,"seasonality");
        }

        if(!is.null(xregNames)){
            matvt <- cbind(matvt,matat[1:nrow(matvt),]);
            colnames(matvt) <- c(component.names,xregNames);
            if(updateX){
                rownames(vecgX) <- xregNames;
                dimnames(matFX) <- list(xregNames,xregNames);
            }
        }
        else{
            colnames(matvt) <- c(component.names);
        }

# Write down the initials. Done especially for Nikos and issue #10
        if(persistenceEstimate){
            persistence <- as.vector(vecg);
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
        esFormula <- "l[t-1]";
        if(Ttype=="A"){
            esFormula <- paste0(esFormula," + b[t-1]");
        }
        else if(Ttype=="M"){
            esFormula <- paste0(esFormula," * b[t-1]");
        }
        if(Stype=="A"){
            esFormula <- paste0(esFormula," + s[t-",maxlag,"]");
        }
        else if(Stype=="M"){
            if(Ttype=="A"){
                esFormula <- paste0("(",esFormula,")");
            }
            esFormula <- paste0(esFormula," * s[t-",maxlag,"]");
        }
        if(Etype=="A"){
            if(!is.null(xreg)){
                if(updateX){
                    esFormula <- paste0(esFormula," + ",paste0(paste0("a",c(1:nExovars),"[t-1] * "),paste0(xregNames,"[t]"),collapse=" + "));
                }
                else{
                    esFormula <- paste0(esFormula," + ",paste0(paste0("a",c(1:nExovars)," * "),paste0(xregNames,"[t]"),collapse=" + "));
                }
            }
            esFormula <- paste0(esFormula," + e[t]");
        }
        else{
            if(any(c(Ttype,Stype)=="A") & Stype!="M"){
                esFormula <- paste0("(",esFormula,")");
            }
            if(!is.null(xreg)){
                if(updateX){
                    esFormula <- paste0(esFormula," * exp(",paste0(paste0("a",c(1:nExovars),"[t-1] * "),paste0(xregNames,"[t]"),collapse=" + "),")");
                }
                else{
                    esFormula <- paste0(esFormula," * exp(",paste0(paste0("a",c(1:nExovars)," * "),paste0(xregNames,"[t]"),collapse=" + "),")");
                }
            }
            esFormula <- paste0(esFormula," * e[t]");
        }
        if(intermittent!="n"){
            esFormula <- paste0("o[t] * (",esFormula,")");
        }
        esFormula <- paste0("y[t] = ",esFormula);
    }
##### Produce fit and forecasts of combined model #####
    else{
        list2env(esValues,environment());

        if(!is.null(xreg) & (xregDo=="u")){
            colnames(matat) <- xregNames;
            xreg <- matxt;
        }

        modelOriginal <- model;
        # Produce the forecasts using AIC weights
        modelsNumber <- length(icWeights);
        model.current <- rep(NA,modelsNumber);
        fitted.list <- matrix(NA,obsInsample,modelsNumber);
        errors.list <- matrix(NA,obsInsample,modelsNumber);
        forecasts.list <- matrix(NA,h,modelsNumber);
        if(intervals){
             lowerList <- matrix(NA,h,modelsNumber);
             upperList <- matrix(NA,h,modelsNumber);
        }

        for(i in 1:length(icWeights)){
            # Get all the parameters from the model
            Etype <- results[[i]]$Etype;
            Ttype <- results[[i]]$Ttype;
            Stype <- results[[i]]$Stype;
            damped <- results[[i]]$damped;
            phi <- results[[i]]$phi;
            cfObjective <- results[[i]]$cfObjective;
            C <- results[[i]]$C;
            nParam <- results[[i]]$nParam;
            xregNames <- results[[i]]$xregNames
            if(xregDo!="u"){
                if(!is.null(xregNames)){
                    matat <- as.matrix(matatOriginal[,xregNames]);
                    matxt <- as.matrix(matxtOriginal[,xregNames]);
                }
                else{
                    matxt <- matrix(1,nrow(matxtOriginal),1);
                    matat <- matrix(0,nrow(matatOriginal),1);
                }
                nExovars <- results[[i]]$nExovars;
                matFX <- results[[i]]$matFX;
                vecgX <- results[[i]]$vecgX;
                xregEstimate <- results[[i]]$xregEstimate;
            }

            BasicMakerES(ParentEnvironment=environment());
            BasicInitialiserES(ParentEnvironment=environment());
            if(damped){
                model.current[i] <- paste0(Etype,Ttype,"d",Stype);
            }
            else{
                model.current[i] <- paste0(Etype,Ttype,Stype);
            }
            model <- model.current[i];

            ssFitter(ParentEnvironment=environment());
            ssForecaster(ParentEnvironment=environment());

            fitted.list[,i] <- y.fit;
            forecasts.list[,i] <- y.for;
            if(intervals){
                lowerList[,i] <- y.low;
                upperList[,i] <- y.high;
            }
            phi <- NULL;
        }
        badStuff <- apply(is.na(rbind(fitted.list,forecasts.list)),2,any);
        fitted.list <- fitted.list[,!badStuff];
        forecasts.list <- forecasts.list[,!badStuff];
        icWeights <- icWeights[!badStuff];
        model.current <- model.current[!badStuff];
        y.fit <- ts(fitted.list %*% icWeights,start=start(data),frequency=frequency(data));
        y.for <- ts(forecasts.list %*% icWeights,start=time(data)[obsInsample]+deltat(data),frequency=frequency(data));
        errors <- ts(c(y) - y.fit,start=start(data),frequency=frequency(data));
        s2 <- mean(errors^2);
        names(icWeights) <- model.current;
        if(intervals){
            lowerList <- lowerList[,!badStuff];
            upperList <- upperList[,!badStuff];
            y.low <- ts(lowerList %*% icWeights,start=start(y.for),frequency=frequency(data));
            y.high <- ts(upperList %*% icWeights,start=start(y.for),frequency=frequency(data));
        }
        else{
            y.low <- NA;
            y.high <- NA;
        }
        names(ICs) <- paste0("Combined ",ic);
        model <- modelOriginal;

# Write down the formula of ETS
        esFormula <- "y[t] = combination of ";
        if(intermittent!="n"){
            esFormula <- paste0(esFormula,"i");
        }
        esFormula <- paste0(esFormula,"ETS");
        if(!is.null(xreg)){
            esFormula <- paste0(esFormula,"X");
        }
    }

##### Do final check and make some preparations for output #####

    # Write down the probabilities from intermittent models
    pt <- ts(c(as.vector(pt),as.vector(pt.for)),start=start(data),frequency=datafreq);
    if(intermittent=="f"){
        intermittent <- "fixed";
    }
    else if(intermittent=="c"){
        intermittent <- "croston";
    }
    else if(intermittent=="t"){
        intermittent <- "tsb";
    }
    else if(intermittent=="n"){
        intermittent <- "none";
    }
    else if(intermittent=="p"){
        intermittent <- "provided";
    }

##### Now let's deal with holdout #####
    if(holdout){
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(y.for),frequency=frequency(data));
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    if(!is.null(xreg)){
        modelname <- "ETSX";
    }
    else{
        modelname <- "ETS";
    }
    modelname <- paste0(modelname,"(",model,")");
    if(all(intermittent!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

##### Print output #####
    if(!silentText){
        if(modelDo!="combine" & any(abs(eigen(matF - vecg %*% matw)$values)>(1 + 1E-10))){
            warning(paste0("Model ETS(",model,") is unstable! Use a different value of 'bounds' parameter to address this issue!"),
                    call.=FALSE);
        }
    }

##### Make a plot #####
    if(!silentGraph){
        if(intervals){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       level=level,legend=!silentLegend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    level=level,legend=!silentLegend,main=modelname);
        }
    }

    ##### Return values #####
    if(modelDo!="combine"){
        model <- list(model=modelname,formula=esFormula,timeElapsed=Sys.time()-startTime,
                      states=matvt,persistence=persistence,phi=phi,
                      initialType=initialType,initial=initialValue,initialSeason=initialSeason,
                      nParam=nParam,
                      fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                      errors=errors.mat,s2=s2,intervals=intervalsType,level=level,
                      actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                      xreg=xreg,updateX=updateX,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
                      ICs=ICs,logLik=logLik,cf=cfObjective,cfType=cfType,FI=FI,accuracy=errormeasures);
        return(structure(model,class="smooth"));
    }
    else{
        model <- list(model=modelname,formula=esFormula,timeElapsed=Sys.time()-startTime,
                      initialType=initialType,
                      fitted=y.fit,forecast=y.for,
                      lower=y.low,upper=y.high,residuals=errors,s2=s2,intervals=intervalsType,level=level,
                      actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                      xreg=xreg,updateX=updateX,
                      ICs=ICs,ICw=icWeights,cf=NULL,cfType=cfType,accuracy=errormeasures);
        return(structure(model,class="smooth"));
    }
}
