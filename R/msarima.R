utils::globalVariables(c("normalizer","constantValue","constantRequired","constantEstimate","C",
                         "ARValue","ARRequired","AREstimate","MAValue","MARequired","MAEstimate",
                         "yForecastStart","nonZeroARI","nonZeroMA"));

#' Multiple Seasonal ARIMA
#'
#' Function constructs Multiple Seasonal State Space ARIMA, estimating AR, MA
#' terms and initial states.
#'
#' The basic ARIMA(p,d,q) used in the function has the following form:
#'
#' \eqn{(1 - B)^d (1 - a_1 B - a_2 B^2 - ... - a_p B^p) y_[t] = (1 + b_1 B +
#' b_2 B^2 + ... + b_q B^q) \epsilon_[t] + c}
#'
#' where \eqn{y_[t]} is the actual values, \eqn{\epsilon_[t]} is the error term,
#' \eqn{a_i, b_j} are the parameters for AR and MA respectively and \eqn{c} is
#' the constant. In case of non-zero differences \eqn{c} acts as drift.
#'
#' This model is then transformed into ARIMA in the Single Source of Error
#' State space form (based by Snyder, 1985, but in a slightly different
#' formulation):
#'
#' \eqn{y_{t} = o_{t} (w' v_{t-l} + x_t a_{t-1} + \epsilon_{t})}
#'
#' \eqn{v_{t} = F v_{t-l} + g \epsilon_{t}}
#'
#' \eqn{a_{t} = F_{X} a_{t-1} + g_{X} \epsilon_{t} / x_{t}}
#'
#' Where \eqn{o_{t}} is the Bernoulli distributed random variable (in case of
#' normal data equal to 1), \eqn{v_{t}} is the state vector (defined based on
#' \code{orders}) and \eqn{l} is the vector of \code{lags}, \eqn{x_t} is the
#' vector of exogenous parameters. \eqn{w} is the \code{measurement} vector,
#' \eqn{F} is the \code{transition} matrix, \eqn{g} is the \code{persistence}
#' vector, \eqn{a_t} is the vector of parameters for exogenous variables,
#' \eqn{F_{X}} is the \code{transitionX} matrix and \eqn{g_{X}} is the
#' \code{persistenceX} matrix. The main difference from \link[smooth]{ssarima}
#' function is that this implementation skips zero polynomials, substantially
#' decreasing the dimension of the transition matrix. As a result, this
#' function works faster than \link[smooth]{ssarima} on high frequency data,
#' and it is more accurate.
#'
#' Due to the flexibility of the model, multiple seasonalities can be used. For
#' example, something crazy like this can be constructed:
#' SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may
#' take some time... Still this should be estimated in finite time (not like
#' with \code{ssarima}).
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssInitialParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssIntervalsRef
#' @template ssGeneralRef
#'
#' @param orders List of orders, containing vector variables \code{ar},
#' \code{i} and \code{ma}. Example:
#' \code{orders=list(ar=c(1,2),i=c(1),ma=c(1,1,1))}. If a variable is not
#' provided in the list, then it is assumed to be equal to zero. At least one
#' variable should have the same length as \code{lags}. Another option is to
#' specify orders as a vector of a form \code{orders=c(p,d,q)}. The non-seasonal
#' ARIMA(p,d,q) is constructed in this case.
#' @param lags Defines lags for the corresponding orders (see examples above).
#' The length of \code{lags} must correspond to the length of either \code{ar},
#' \code{i} or \code{ma} in \code{orders} variable. There is no restrictions on
#' the length of \code{lags} vector. It is recommended to order \code{lags}
#' ascending.
#' The orders are set by a user. If you want the automatic order selection,
#' then use \link[smooth]{auto.ssarima} function instead.
#' @param constant If \code{TRUE}, constant term is included in the model. Can
#' also be a number (constant value).
#' @param AR Vector or matrix of AR parameters. The order of parameters should
#' be lag-wise. This means that first all the AR parameters of the firs lag
#' should be passed, then for the second etc. AR of another ssarima can be
#' passed here.
#' @param MA Vector or matrix of MA parameters. The order of parameters should
#' be lag-wise. This means that first all the MA parameters of the firs lag
#' should be passed, then for the second etc. MA of another ssarima can be
#' passed here.
#' @param ...  Other non-documented parameters.
#'
#' Parameter \code{model} can accept a previously estimated SARIMA model and
#' use all its parameters.
#'
#' \code{FI=TRUE} will make the function produce Fisher Information matrix,
#' which then can be used to calculated variances of parameters of the model.
#'
#' @return Object of class "smooth" is returned. It contains the list of the
#' following values:
#'
#' \itemize{
#' \item \code{model} - the name of the estimated model.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{states} - the matrix of the fuzzy components of ssarima, where
#' \code{rows} correspond to time and \code{cols} to states.
#' \item \code{transition} - matrix F.
#' \item \code{persistence} - the persistence vector. This is the place, where
#' smoothing parameters live.
#' \item \code{measurement} - measurement vector of the model.
#' \item \code{AR} - the matrix of coefficients of AR terms.
#' \item \code{I} - the matrix of coefficients of I terms.
#' \item \code{MA} - the matrix of coefficients of MA terms.
#' \item \code{constant} - the value of the constant term.
#' \item \code{initialType} - Type of the initial values used.
#' \item \code{initial} - the initial values of the state vector (extracted
#' from \code{states}).
#' \item \code{nParam} - table with the number of estimated / provided parameters.
#' If a previous model was reused, then its initials are reused and the number of
#' provided parameters will take this into account.
#' \item \code{fitted} - the fitted values.
#' \item \code{forecast} - the point forecast.
#' \item \code{lower} - the lower bound of prediction interval. When
#' \code{intervals="none"} then NA is returned.
#' \item \code{upper} - the higher bound of prediction interval. When
#' \code{intervals="none"} then NA is returned.
#' \item \code{residuals} - the residuals of the estimated model.
#' \item \code{errors} - The matrix of 1 to h steps ahead errors.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom into
#' account).
#' \item \code{intervals} - type of intervals asked by user.
#' \item \code{level} - confidence level for intervals.
#' \item \code{cumulative} - whether the produced forecast was cumulative or not.
#' \item \code{actuals} - the original data.
#' \item \code{holdout} - the holdout part of the original data.
#' \item \code{imodel} - model of the class "iss" if intermittent model was estimated.
#' If the model is non-intermittent, then imodel is \code{NULL}.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If
#' \code{xregDo="s"}, then this value will contain only selected exogenous
#' variables.
#' \item \code{updateX} - boolean,
#' defining, if the states of exogenous variables were estimated as well.
#' \item \code{initialX} - initial values for parameters of exogenous
#' variables.
#' \item \code{persistenceX} - persistence vector g for exogenous variables.
#' \item \code{transitionX} - transition matrix F for exogenous variables.
#' \item \code{ICs} - values of information criteria of the model. Includes
#' AIC, AICc, BIC and BICc.
#' \item \code{logLik} - log-likelihood of the function.
#' \item \code{cf} - Cost function value.
#' \item \code{cfType} - Type of cost function used in the estimation.
#' \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE}
#' or when \code{FI} is not provided at all.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample.
#' In case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.
#' }
#'
#' @seealso \code{\link[smooth]{auto.msarima}, \link[smooth]{orders},
#' \link[smooth]{lags}, \link[smooth]{ssarima}, \link[forecast]{auto.arima}}
#'
#' @examples
#'
#' # The previous one is equivalent to:
#' ourModel <- msarima(rnorm(118,100,3),orders=c(1,1,1),lags=1,h=18,holdout=TRUE,intervals="p")
#'
#' # Example of SARIMA(2,0,0)(1,0,0)[4]
#' msarima(rnorm(118,100,3),orders=list(ar=c(2,1)),lags=c(1,4),h=18,holdout=TRUE)
#'
#' # SARIMA of a peculiar order on AirPassengers data with Fisher Information
#' ourModel <- msarima(AirPassengers,orders=list(ar=c(1,0,3),i=c(1,0,1),ma=c(0,1,2)),
#'                     lags=c(1,6,12),h=10,holdout=TRUE,FI=TRUE)
#'
#' # Construct the 95% confidence intervals for the parameters of the model
#' ourCoefs <- coef(ourModel)
#' ourCoefsSD <- sqrt(abs(diag(solve(ourModel$FI))))
#' # Sort values accordingly
#' ourCoefs <- ourCoefs[names(ourCoefsSD)]
#' ourConfInt <- cbind(ourCoefs + qt(0.025,nobs(ourModel)) * ourCoefsSD,
#'                     ourCoefs + qt(0.975,nobs(ourModel)) * ourCoefsSD)
#' colnames(ourConfInt) <- c("2.25%","97.5%")
#' ourConfInt
#'
#' # ARIMA(1,1,1) with Mean Squared Trace Forecast Error
#' msarima(rnorm(118,100,3),orders=list(ar=1,i=1,ma=1),lags=1,h=18,holdout=TRUE,cfType="TMSE")
#'
#' msarima(rnorm(118,100,3),orders=list(ar=1,i=1,ma=1),lags=1,h=18,holdout=TRUE,cfType="aTMSE")
#'
#' # SARIMA(0,1,1) with exogenous variables with crazy estimation of xreg
#' ourModel <- msarima(rnorm(118,100,3),orders=list(i=1,ma=1),h=18,holdout=TRUE,
#'                     xreg=c(1:118),updateX=TRUE)
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#' @export msarima
msarima <- function(data, orders=list(ar=c(0),i=c(1),ma=c(1)), lags=c(1),
                    constant=FALSE, AR=NULL, MA=NULL,
                    initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC","BICc"),
                    cfType=c("MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                    h=10, holdout=FALSE, cumulative=FALSE,
                    intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                    intermittent=c("none","auto","fixed","interval","probability","sba","logistic"),
                    imodel="MNN",
                    bounds=c("admissible","none"),
                    silent=c("all","graph","legend","output","none"),
                    xreg=NULL, xregDo=c("use","select"), initialX=NULL,
                    updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
##### Function constructs SARIMA model (possible triple seasonality) using state space approach
# ar.orders contains vector of seasonal ARs. ar.orders=c(2,1,3) will mean AR(2)*SAR(1)*SAR(3) - model with double seasonality.
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model",inherits=FALSE)){
        if(is.null(model$model)){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }
        else if(smoothType(model)!="ARIMA"){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }

# If this is a normal ARIMA, do things
        if(any(unlist(gregexpr("combine",model$model))==-1)){
            if(!is.null(model$imodel)){
                imodel <- model$imodel;
            }
            if(!is.null(model$initial)){
                initial <- model$initial;
            }
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
            AR <- model$AR;
            MA <- model$MA;
            constant <- model$constant;
            model <- model$model;
            arimaOrders <- paste0(c("",substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1),"")
                                   ,collapse=";");
            comas <- unlist(gregexpr("\\,",arimaOrders));
            semicolons <- unlist(gregexpr("\\;",arimaOrders));
            ar.orders <- as.numeric(substring(arimaOrders,semicolons[-length(semicolons)]+1,comas[2*(1:(length(comas)/2))-1]-1));
            i.orders <- as.numeric(substring(arimaOrders,comas[2*(1:(length(comas)/2))-1]+1,comas[2*(1:(length(comas)/2))-1]+1));
            ma.orders <- as.numeric(substring(arimaOrders,comas[2*(1:(length(comas)/2))]+1,semicolons[-1]-1));
            if(any(unlist(gregexpr("\\[",model))!=-1)){
                lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
            }
            else{
                lags <- 1;
            }
        }
        else{
            stop("The provided model is a combination of ARIMAs. We cannot fit that.",call.=FALSE);
        }
    }
    else if(!is.null(orders)){
        if(is.list(orders)){
            ar.orders <- orders$ar;
            i.orders <- orders$i;
            ma.orders <- orders$ma;
        }
        else if(is.vector(orders)){
            ar.orders <- orders[1];
            i.orders <- orders[2];
            ma.orders <- orders[3];
            lags <- 1;
        }
    }

# If orders are provided in ellipsis via ar.orders, write them down.
    if(exists("ar.orders",inherits=FALSE)){
        if(is.null(ar.orders)){
            ar.orders <- 0;
        }
    }
    else{
        ar.orders <- 0;
    }
    if(exists("i.orders",inherits=FALSE)){
        if(is.null(i.orders)){
            i.orders <- 0;
        }
    }
    else{
        i.orders <- 0;
    }
    if(exists("ma.orders",inherits=FALSE)){
        if(is.null(ma.orders)){
            ma.orders <- 0;
        }
    }
    else{
        ma.orders <- 0;
    }

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput("msarima",ParentEnvironment=environment());

# Cost function for SSARIMA
CF <- function(C){
    cfRes <- costfuncARIMA(ar.orders, ma.orders, i.orders, lags, nComponents,
                           ARValue, MAValue, constantValue, C,
                           matvt, matF, matw, y, vecg,
                           h, modellags, Etype, Ttype, Stype,
                           multisteps, cfType, normalizer, initialType,
                           nExovars, matxt, matat, matFX, vecgX, ot,
                           AREstimate, MAEstimate, constantRequired, constantEstimate,
                           xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate,
                           bounds,
                           # The last bit is "ssarimaOld"
                           FALSE, nonZeroARI, nonZeroMA);

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}

##### Estimate ssarima or just use the provided values #####
CreatorSSARIMA <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    # If there is something to optimise, let's do it.
    if(any((initialType=="o"),(AREstimate),(MAEstimate),
           (initialXEstimate),(FXEstimate),(gXEstimate),(constantEstimate))){

        C <- NULL;
        if(nComponents > 0){
# ar terms, ma terms from season to season...
            if(AREstimate){
                # C <- c(C,c(1:sum(ar.orders))/sum(sum(ar.orders):1));
                C <- c(C,rep(1/sum(ar.orders),sum(ar.orders)));
            }
            if(MAEstimate){
                # C <- c(C,rep(0.1,sum(ma.orders)));
                C <- c(C,rep(1/sum(ma.orders),sum(ma.orders)));
            }

# initial values of state vector and the constant term
            if(initialType=="o"){
                C <- c(C,matvt[1:nComponents,1]);
            }
        }

        if(constantEstimate){
            if(all(i.orders==0)){
                C <- c(C,sum(yot)/obsInsample);
            }
            else{
                C <- c(C,sum(diff(yot))/obsInsample);
            }
        }

# initials, transition matrix and persistence vector
        if(xregEstimate){
            if(initialXEstimate){
                C <- c(C,matat[maxlag,]);
            }
            if(updateX){
                if(FXEstimate){
                    C <- c(C,c(diag(nExovars)));
                }
                if(gXEstimate){
                    C <- c(C,rep(0,nExovars));
                }
            }
        }

# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
        C <- res$solution;
# Optimise model. Second run
        res2 <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
        # This condition is needed in order to make sure that we did not make the solution worse
        if(res2$objective <= res$objective){
            res <- res2;
        }

        C <- res$solution;
        cfObjective <- res$objective;

        # Parameters estimated + variance
        nParam <- length(C) + 1;
    }
    else{
        C <- NULL;

# initial values of state vector and the constant term
        if(nComponents>0 & initialType=="p"){
            matvt[1:maxlag,1:nComponents] <- initialValue;
        }
        if(constantRequired){
            matvt[1:maxlag,(nComponents+1)] <- constantValue;
        }

        cfObjective <- CF(C);

        # Only variance is estimated
        nParam <- 1;
    }

    ICValues <- ICFunction(nParam=nParam,nParamIntermittent=nParamIntermittent,
                           C=C,Etype=Etype);
    ICs <- ICValues$ICs;
    bestIC <- ICs[ic];
    logLik <- ICValues$llikelihood;

    return(list(cfObjective=cfObjective,C=C,ICs=ICs,bestIC=bestIC,nParam=nParam,logLik=logLik));
}

    # Prepare lists for the polynomials
    P <- list(NA);
    D <- list(NA);
    Q <- list(NA);

##### Preset values of matvt and other matrices ######
    if(nComponents > 0){
        # Transition matrix, measurement vector and persistence vector + state vector
        matF <- matrix(0,nComponents,nComponents);
        matw <- matrix(1,1,nComponents);
        vecg <- matrix(0,nComponents,1);
        matvt <- matrix(NA,obsStates,nComponents);
        if(constantRequired){
            matF <- cbind(rbind(matF,rep(0,nComponents)),rep(1,nComponents+1));
            matw <- cbind(matw,1);
            vecg <- rbind(vecg,0);
            matvt <- cbind(matvt,rep(1,obsStates));
        }
        if(initialType=="p"){
            matvt[1:maxlag,1:nComponents] <- initialValue;
        }
        else{
            for(i in 1:nComponents){
                nRepeats <- ceiling(maxlag/modellags[i]);
                matvt[1:maxlag,i] <- rep(y[1:modellags[i]],nRepeats)[nRepeats*modellags[i]+(-maxlag+1):0];
                # matvt[1:maxlag,i] <- rep(y[1:modellags[i]],nRepeats)[1:maxlag];
            }
        }
    }
    else{
        matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0,1,1);
        matvt <- matrix(1,obsStates,1);
        modellags <- matrix(1,1,1);
    }

##### Preset yFitted, yForecast, errors and basic parameters #####
    yFitted <- rep(NA,obsInsample);
    yForecast <- rep(NA,h);
    errors <- rep(NA,obsInsample);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(data=data, xreg=xreg, updateX=updateX, ot=ot,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obsInsample=obsInsample, obsAll=obsAll, obsStates=obsStates,
                       maxlag=maxlag, h=h, xregDo=xregDo, silent=silentText);

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
    nParamIntermittent <- all(intermittent!=c("n","provided"))*1;
    nParamMax <- nParamMax + nParamExo + nParamIntermittent;

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
                               nParamMax," while the number of observations is ",obsNonzero - nParamExo,"!"),call.=FALSE);
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


# If this is tiny sample, use ARIMA with constant instead
    if(tinySample){
        warning("Not enough observations to fit ARIMA. Switching to ARIMA(0,0,0) with constant.",call.=FALSE);
        return(msarima(data,orders=list(ar=0,i=0,ma=0),lags=1,
                       constant=TRUE,
                       initial=initial,cfType=cfType,
                       h=h,holdout=holdout,cumulative=cumulative,
                       intervals=intervals,level=level,
                       intermittent=intermittent,
                       imodel=imodel,
                       bounds="u",
                       silent=silent,
                       xreg=xreg,xregDo=xregDo,initialX=initialX,
                       updateX=updateX,persistenceX=persistenceX,transitionX=transitionX));
    }

#####Start the calculations#####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(ssForecaster) <- environment();
    environment(ssFitter) <- environment();

    # If auto intermittent, then estimate model with intermittent="n" first.
    if(any(intermittent==c("a","n"))){
        intermittentParametersSetter(intermittent="n",ParentEnvironment=environment());
    }
    else{
        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

    ssarimaValues <- CreatorSSARIMA(silentText);

##### If intermittent=="a", run a loop and select the best one #####
    if(intermittent=="a"){
        if(!any(cfType==c("MSE","MAE","HAM","MSEh","MAEh","HAMh","MSCE","MACE","CHAM",
                          "TFL","aTFL","Rounded","TSB","LogisticD","LogisticL"))){
            warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. A wrong intermittent model may be selected"),call.=FALSE);
        }
        if(!silentText){
            cat("Selecting appropriate type of intermittency... ");
        }
# Prepare stuff for intermittency selection
        intermittentModelsPool <- c("n","f","i","p","s","l");
        intermittentCFs <- intermittentICs <- rep(NA,length(intermittentModelsPool));
        intermittentModelsList <- list(NA);
        intermittentICs[1] <- ssarimaValues$bestIC[ic];
        intermittentCFs[1] <- ssarimaValues$cfObjective;

        for(i in 2:length(intermittentModelsPool)){
            intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentModelsList[[i]] <- CreatorSSARIMA(silentText=TRUE);
            intermittentICs[i] <- intermittentModelsList[[i]]$bestIC[ic];
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
            ssarimaValues <- intermittentModelsList[[iBest]];
        }
        else{
            intermittent <- "n"
        }

        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

    list2env(ssarimaValues,environment());

    if(xregDo!="u"){
        # Prepare for fitting
        elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, nComponents,
                                ARValue, MAValue, constantValue, C,
                                matvt, vecg, matF,
                                initialType, nExovars, matat, matFX, vecgX,
                                AREstimate, MAEstimate, constantRequired, constantEstimate,
                                xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate,
                                # The last bit is "ssarimaOld"
                                FALSE, modellags, nonZeroARI, nonZeroMA);
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[,] <- elements$matvt;
        matat[,] <- elements$matat;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;
        polysos.ar <- elements$arPolynomial;
        polysos.ma <- elements$maPolynomial;

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
        }

        if(!is.null(xreg)){
            ssarimaValues <- CreatorSSARIMA(silentText);
            list2env(ssarimaValues,environment());
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
    elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, nComponents,
                            ARValue, MAValue, constantValue, C,
                            matvt, vecg, matF,
                            initialType, nExovars, matat, matFX, vecgX,
                            AREstimate, MAEstimate, constantRequired, constantEstimate,
                            xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate,
                            # The last bit is "ssarimaOld"
                            FALSE, modellags, nonZeroARI, nonZeroMA);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[,] <- elements$matvt;
    matat[,] <- elements$matat;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    polysos.ar <- elements$arPolynomial;
    polysos.ma <- elements$maPolynomial;

    nComponents <- nComponents + constantRequired;

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        if(constantRequired){
            initialValue <- matvt[1:maxlag,-ncol(matvt)];
        }
        else{
            initialValue <- matvt[1:maxlag,];
        }
        if(initialType!="b"){
            parametersNumber[1,1] <- parametersNumber[1,1] + length(initialValue);
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

    # Write down the probabilities from intermittent models
    pt <- ts(c(as.vector(pt),as.vector(pForecast)),start=dataStart,frequency=dataFreq);
    # Write down the number of parameters of imodel
    if(all(intermittent!=c("n","provided")) & !imodelProvided){
        parametersNumber[1,3] <- imodel$nParam;
    }
    # Make nice names for intermittent
    if(intermittent=="f"){
        intermittent <- "fixed";
    }
    else if(intermittent=="i"){
        intermittent <- "interval";
    }
    else if(intermittent=="p"){
        intermittent <- "probability";
    }
    else if(intermittent=="l"){
        intermittent <- "logistic";
    }
    else if(intermittent=="n"){
        intermittent <- "none";
    }

# Fill in the rest of matvt
    matvt <- ts(matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    if(!is.null(xreg)){
        matvt <- cbind(matvt,matat[1:nrow(matvt),]);
        colnames(matvt) <- c(paste0("Component ",c(1:max(1,nComponents))),colnames(matat));
        if(updateX){
            rownames(vecgX) <- xregNames;
            dimnames(matFX) <- list(xregNames,xregNames);
        }
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:max(1,nComponents)));
    }
    if(constantRequired){
        colnames(matvt)[nComponents] <- "Constant";
    }

# AR terms
    if(any(ar.orders!=0)){
        ARterms <- matrix(0,max(ar.orders),sum(ar.orders!=0),
                          dimnames=list(paste0("AR(",c(1:max(ar.orders)),")"),
                                        paste0("Lag ",lags[ar.orders!=0])));
    }
    else{
        ARterms <- NULL;
    }
# Differences
    if(any(i.orders!=0)){
        Iterms <- matrix(0,1,length(i.orders),
                          dimnames=list("I(...)",paste0("Lag ",lags)));
        Iterms[,] <- i.orders;
    }
    else{
        Iterms <- 0;
    }
# MA terms
    if(any(ma.orders!=0)){
        MAterms <- matrix(0,max(ma.orders),sum(ma.orders!=0),
                          dimnames=list(paste0("MA(",c(1:max(ma.orders)),")"),
                                        paste0("Lag ",lags[ma.orders!=0])));
    }
    else{
        MAterms <- NULL;
    }

    nCoef <- arCoef <- maCoef <- 0;
    arIndex <- maIndex <- 1;
    for(i in 1:length(ar.orders)){
        if(ar.orders[i]!=0){
            if(AREstimate){
                ARterms[1:ar.orders[i],arIndex] <- C[nCoef+(1:ar.orders[i])];
                names(C)[nCoef+(1:ar.orders[i])] <- paste0("AR(",1:ar.orders[i],"), ",colnames(ARterms)[arIndex]);
                nCoef <- nCoef + ar.orders[i];
                parametersNumber[1,1] <- parametersNumber[1,1] + ar.orders[i];
            }
            else{
                ARterms[1:ar.orders[i],arIndex] <- ARValue[arCoef+(1:ar.orders[i])];
                arCoef <- arCoef + ar.orders[i];
            }
            arIndex <- arIndex + 1;
        }
        if(ma.orders[i]!=0){
            if(MAEstimate){
                MAterms[1:ma.orders[i],maIndex] <- C[nCoef+(1:ma.orders[i])];
                names(C)[nCoef+(1:ma.orders[i])] <- paste0("MA(",1:ma.orders[i],"), ",colnames(MAterms)[maIndex]);
                nCoef <- nCoef + ma.orders[i];
                parametersNumber[1,1] <- parametersNumber[1,1] + ma.orders[i];
            }
            else{
                MAterms[1:ma.orders[i],maIndex] <- MAValue[maCoef+(1:ma.orders[i])];
                maCoef <- maCoef + ma.orders[i];
            }
            maIndex <- maIndex + 1;
        }
    }

# Give model the name
    if((length(ar.orders)==1) && all(lags==1)){
        if(!is.null(xreg)){
            modelname <- "ARIMAX";
        }
        else{
            modelname <- "ARIMA";
        }
        modelname <- paste0(modelname,"(",ar.orders,",",i.orders,",",ma.orders,")");
    }
    else{
        modelname <- "";
        for(i in 1:length(ar.orders)){
            modelname <- paste0(modelname,"(",ar.orders[i],",");
            modelname <- paste0(modelname,i.orders[i],",");
            modelname <- paste0(modelname,ma.orders[i],")[",lags[i],"]");
        }
        if(!is.null(xreg)){
            modelname <- paste0("SARIMAX",modelname);
        }
        else{
            modelname <- paste0("SARIMA",modelname);
        }
    }
    if(all(intermittent!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    if(constantRequired){
        if(constantEstimate){
            constantValue <- matvt[1,nComponents];
            parametersNumber[1,1] <- parametersNumber[1,1] + 1;
            if(!is.null(names(C))){
                names(C)[is.na(names(C))][1] <- "Constant";
            }
            else{
                names(C)[1] <- "Constant";
            }
        }
        const <- constantValue;

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

    parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
    parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

    # Write down Fisher Information if needed
    if(FI & parametersNumber[1,4]>1){
        environment(likelihoodFunction) <- environment();
        FI <- -numDeriv::hessian(likelihoodFunction,C);
        rownames(FI) <- colnames(FI) <- names(C);
        if(initialType=="o"){
            # Leave only AR and MA parameters. Forget about the initials
            FI <- FI[!is.na(rownames(FI)),!is.na(colnames(FI))];
        }
    }
    else{
        FI <- NA;
    }

##### Deal with the holdout sample #####
    if(holdout){
        yHoldout <- ts(data[(obsInsample+1):obsAll],start=yForecastStart,frequency=frequency(data));
        if(cumulative){
            errormeasures <- Accuracy(sum(yHoldout),yForecast,h*y);
        }
        else{
            errormeasures <- Accuracy(yHoldout,yForecast,y);
        }

        if(cumulative){
            yHoldout <- ts(sum(yHoldout),start=yForecastStart,frequency=dataFreq);
        }
    }
    else{
        yHoldout <- NA;
        errormeasures <- NA;
    }

##### Make a plot #####
    if(!silentGraph){
        yForecastNew <- yForecast;
        yUpperNew <- yUpper;
        yLowerNew <- yLower;
        if(cumulative){
            yForecastNew <- ts(rep(yForecast/h,h),start=yForecastStart,frequency=dataFreq)
            if(intervals){
                yUpperNew <- ts(rep(yUpper/h,h),start=yForecastStart,frequency=dataFreq)
                yLowerNew <- ts(rep(yLower/h,h),start=yForecastStart,frequency=dataFreq)
            }
        }

        if(intervals){
            graphmaker(actuals=data,forecast=yForecastNew,fitted=yFitted, lower=yLowerNew,upper=yUpperNew,
                       level=level,legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
        else{
            graphmaker(actuals=data,forecast=yForecastNew,fitted=yFitted,
                       legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
    }

##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,transition=matF,persistence=vecg,
                  measurement=matw,
                  AR=ARterms,I=Iterms,MA=MAterms,constant=const,
                  initialType=initialType,initial=initialValue,
                  nParam=parametersNumber, modelLags=modellags,
                  fitted=yFitted,forecast=yForecast,lower=yLower,upper=yUpper,residuals=errors,
                  errors=errors.mat,s2=s2,intervals=intervalsType,level=level,cumulative=cumulative,
                  actuals=data,holdout=yHoldout,imodel=imodel,
                  xreg=xreg,updateX=updateX,initialX=initialX,persistenceX=persistenceX,transitionX=transitionX,
                  ICs=ICs,logLik=logLik,cf=cfObjective,cfType=cfType,FI=FI,accuracy=errormeasures);
    return(structure(model,class=c("smooth","msarima")));
}
