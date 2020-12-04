#' @param parallel If TRUE, the estimation of ADAM models is done in parallel (used in \code{auto.adam} only).
#' If the number is provided (e.g. \code{parallel=41}), then the specified number of cores is set up.
#' WARNING! Packages \code{foreach} and either \code{doMC} (Linux and Mac only)
#' or \code{doParallel} are needed in order to run the function in parallel.
#' @param outliers Defines what to do with outliers: \code{"ignore"}, so just returning the model,
#' \code{"detect"} outliers based on specified \code{level} and include dummies for them in the model,
#' or detect and \code{"select"} those of them that reduce \code{ic} value.
#' @param level What confidence level to use for detection of outliers. The default is 99.9%. The statistics
#' values depends on the distribution used in the model.
#' @param fast If \code{TRUE}, then some of the orders of ARIMA are
#' skipped in the order selection. This is not advised for models with \code{lags} greater than 12.
#'
#' @examples
#' ourModel <- auto.adam(rnorm(100,100,10), model="ZZN", lags=c(1,4),
#'                       orders=list(ar=c(2,2),ma=c(2,2),select=TRUE))
#'
#' @rdname adam
#' @export
auto.adam <- function(data, model="ZXZ", lags=c(frequency(data)), orders=list(ar=c(0),i=c(0),ma=c(0),select=FALSE),
                      formula=NULL, outliers=c("ignore","use","select"), level=0.99,
                      distribution=c("dnorm","dlaplace","ds","dgnorm","dlnorm","dinvgauss"),
                      h=0, holdout=FALSE,
                      persistence=NULL, phi=NULL, initial=c("optimal","backcasting"), arma=NULL,
                      occurrence=c("none","auto","fixed","general","odds-ratio","inverse-odds-ratio","direct"),
                      ic=c("AICc","AIC","BIC","BICc"), bounds=c("usual","admissible","none"),
                      regressors=c("use","select","adapt"),
                      silent=TRUE, parallel=FALSE, fast=TRUE, ...){
    # Copyright (C) 2020 - Inf  Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();

    # paste0() is needed in order to get rid of potential issues with names
    responseName <- paste0(deparse(substitute(data)),collapse="");
    responseName <- make.names(responseName,unique=TRUE);

    #### modelDo, ic ####
    if(any(unlist(strsplit(model,""))=="C")){
        modelDo <- "combine";
    }
    else{
        modelDo <- "select";
    }

    ic <- match.arg(ic,c("AICc","AIC","BIC","BICc"));
    ICFunction <- switch(ic,
                         "AIC"=AIC,
                         "AICc"=AICc,
                         "BIC"=BIC,
                         "BICc"=BICc);

    initial <- match.arg(initial);
    outliers <- match.arg(outliers);

    # Check the provided level value.
    if(length(level)>1){
        warning(paste0("Sorry, but we only support scalar for the level, ",
                       "when constructing in-sample prediction interval. ",
                       "Using the first provided value."),
                call.=FALSE);
        level <- level[1];
    }
    # Fix just in case a silly user used 95 etc instead of 0.95
    if(level>1){
        level[] <- level / 100;
    }

    # The function checks the provided parameters of adam and/or oes
    ##### data #####
    # If this is simulated, extract the actuals
    if(is.adam.sim(data) || is.smooth.sim(data)){
        data <- data$data;
    }
    # If this is Mdata, use all the available stuff
    else if(inherits(data,"Mdata")){
        h <- data$h;
        holdout <- TRUE;
        lags <- frequency(data$x);
        data <- ts(c(data$x,data$xx),start=start(data$x),frequency=frequency(data$x));
    }

    # If this is a vector, use length
    # yInSample is needed for checks only
    if(is.null(dim(data))){
        obsInSample <- length(data) - holdout*h;
        yInSample <- data[1:obsInSample];
    }
    else{
        obsInSample <- nrow(data) - holdout*h;
        if(!is.null(formula)){
            yInSample <- data[1:obsInSample,formula[[2]]];
        }
        else{
            yInSample <- data[1:obsInSample,1];
        }
        if(is.null(formula)){
            responseName <- colnames(data)[1];
        }
        else{
            responseName <- all.vars(formula)[1];
        }
    }

    # If this is non-positive data and positive defined distributions are used, fix this
    if(any(yInSample<=0) && any(c("dlnorm","dllaplace","dls","dinvgauss") %in% distribution) &&
       (!is.occurrence(occurrence) && occurrence[1]=="none")){
        distributionToDrop <- c("dlnorm","dllaplace","dls","dinvgauss")[c("dlnorm","dllaplace","dls","dinvgauss") %in% distribution];
        warning(paste0("The data is not strictly positive, so not all the distributions make sense. ",
                       "Dropping ",paste0(distributionToDrop,collapse=", "),"."),
                call.=FALSE);
        distribution <- distribution[!(distribution %in% distributionToDrop)];
    }
    nModels <- length(distribution);

    #### Create logical, determining, what we are dealing with ####
    # ETS
    etsModel <- all(model!="NNN");
    # These values are needed for number of degrees of freedom check
    Etype <- substr(model,1,1);
    Ttype <- substr(model,2,2);
    Stype <- substr(model,nchar(model),nchar(model));
    damped <- (nchar(model)==4);
    if(length(Etype)>1){
        Etype <- "Z";
    }
    if(length(Ttype)>1){
        Ttype <- "Z";
        damped <- TRUE;
    }
    if(length(Stype)>1){
        Stype <- "Z";
    }

    # ARIMA + ARIMA select
    if(is.list(orders)){
        arimaModel <- any(c(orders$ar,orders$i,orders$ma)>0);
    }
    else{
        arimaModel <- any(orders>0);
    }

    # xreg - either as a separate variable or as a matrix for data
    xregModel <- (!is.null(dim(data)) && ncol(data)>1);
    xregNumber <- 0;
    if(xregModel){
        xregNumber[] <- ncol(data)-1;
    }
    regressors <- match.arg(regressors);

    #### Checks of provided parameters for ARIMA selection ####
    if(arimaModel && is.list(orders)){
        arimaModelSelect <- orders$select;
        arMax <- orders$ar;
        iMax <- orders$i;
        maMax <- orders$ma;

        if(is.null(arimaModelSelect)){
            arimaModelSelect <- FALSE;
        }

        if(any(c(arMax,iMax,maMax)<0)){
            stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
        }

        # If there are no lags for the basic components, correct this.
        if(sum(lags==1)==0){
            lags <- c(1,lags);
        }

        # If there are zero lags, drop them
        if(any(lags==0)){
            arMax <- arMax[lags!=0];
            iMax <- iMax[lags!=0];
            maMax <- maMax[lags!=0];
            lags <- lags[lags!=0];
        }

        # Define maxorder and make all the values look similar (for the polynomials)
        maxorder <- max(length(arMax),length(iMax),length(maMax),length(lags));
        if(length(arMax)!=maxorder){
            arMax <- c(arMax,rep(0,maxorder-length(arMax)));
        }
        if(length(iMax)!=maxorder){
            iMax <- c(iMax,rep(0,maxorder-length(iMax)));
        }
        if(length(maMax)!=maxorder){
            maMax <- c(maMax,rep(0,maxorder-length(maMax)));
        }

        # If zeroes are defined as orders for some lags, drop them.
        if(any((arMax + iMax + maMax)==0)){
            orders2leave <- (arMax + iMax + maMax)!=0;
            if(all(!orders2leave)){
                orders2leave <- lags==min(lags);
            }
            arMax <- arMax[orders2leave];
            iMax <- iMax[orders2leave];
            maMax <- maMax[orders2leave];
            lags <- lags[orders2leave];
        }

        # Get rid of duplicates in lags
        if(length(unique(lags))!=length(lags)){
            lagsNew <- unique(lags);
            arMaxNew <- iMaxNew <- maMaxNew <- lagsNew;
            for(i in 1:length(lagsNew)){
                arMaxNew[i] <- max(arMax[which(lags==lagsNew[i])],na.rm=TRUE);
                iMaxNew[i] <- max(iMax[which(lags==lagsNew[i])],na.rm=TRUE);
                maMaxNew[i] <- max(maMax[which(lags==lagsNew[i])],na.rm=TRUE);
            }
            arMax <- arMaxNew;
            iMax <- iMaxNew;
            maMax <- maMaxNew;
            lags <- lagsNew;
        }

        # Order things, so we would deal with the lowest level of seasonality first
        arMax <- arMax[order(lags,decreasing=FALSE)];
        iMax <- iMax[order(lags,decreasing=FALSE)];
        maMax <- maMax[order(lags,decreasing=FALSE)];
        lags <- sort(lags,decreasing=FALSE);
        initialArimaNumber <- max((arMax + iMax) %*% lags, maMax %*% lags);
    }
    else{
        arMax <- iMax <- maMax <- NULL;
        arimaModelSelect <- FALSE;
        initialArimaNumber <- 0;
    }

    #### Maximum number of parameters to estimate ####
    nParamMax <- (1 +
                      # ETS model
                      etsModel*((Etype!="N") + (Ttype!="N") + (Stype!="N")*length(lags) + damped +
                                    (initial=="optimal") * ((Etype!="N") + (Ttype!="N") + (Stype!="N")*sum(lags))) +
                      # ARIMA components: initials + parameters
                      arimaModel*((initial=="optimal")*initialArimaNumber + sum(arMax) + sum(maMax)) +
                      # Xreg initials and smoothing parameters
                      xregModel*(xregNumber*(1+(regressors=="adapt"))));

    # Do something in order to make sure that the stuff works
    if((nParamMax > obsInSample) && arimaModelSelect){
        # If this is ARIMA, remove some orders
        if(arimaModel){
            nParamMaxNonARIMA <- nParamMax - ((initial=="optimal")*initialArimaNumber + sum(arMax) + sum(maMax));
            if(obsInSample > nParamMaxNonARIMA){
                # Drop out some ARIMA orders, start with seasonal
                # Reduce maximum order of AR
                while(nParamMax > obsInSample){
                    arimaTail <- max(tail(which(arMax!=0),1),tail(which(iMax!=0),1),tail(which(maMax!=0),1));
                    if(arMax[arimaTail]>0){
                        arMax[arimaTail] <- arMax[arimaTail]-1;
                        initialArimaNumber[] <- max((arMax + iMax) %*% lags, maMax %*% lags);
                        nParamMax[] <- (1 +
                                            # ETS model
                                            etsModel*((Etype!="N") + (Ttype!="N") + (Stype!="N")*length(lags) + damped +
                                                          (initial=="optimal") * ((Etype!="N") + (Ttype!="N") + (Stype!="N")*sum(lags))) +
                                            # ARIMA components: initials + parameters
                                            arimaModel*((initial=="optimal")*initialArimaNumber + sum(arMax) + sum(maMax)) +
                                            # Xreg initials and smoothing parameters
                                            xregModel*(xregNumber*(1+(regressors=="adapt"))));
                    }

                    # Reduce maximum order of I
                    if(nParamMax > obsInSample && iMax[arimaTail]>0){
                        iMax[arimaTail] <- iMax[arimaTail]-1;
                        initialArimaNumber[] <- max((arMax + iMax) %*% lags, maMax %*% lags);
                        nParamMax[] <- (1 +
                                            # ETS model
                                            etsModel*((Etype!="N") + (Ttype!="N") + (Stype!="N")*length(lags) + damped +
                                                          (initial=="optimal") * ((Etype!="N") + (Ttype!="N") + (Stype!="N")*sum(lags))) +
                                            # ARIMA components: initials + parameters
                                            arimaModel*((initial=="optimal")*initialArimaNumber + sum(arMax) + sum(maMax)) +
                                            # Xreg initials and smoothing parameters
                                            xregModel*(xregNumber*(1+(regressors=="adapt"))));
                    }

                    # Reduce maximum order of MA
                    if(nParamMax > obsInSample && maMax[arimaTail]>0){
                        maMax[arimaTail] <- maMax[arimaTail]-1;
                        initialArimaNumber[] <- max((arMax + iMax) %*% lags, maMax %*% lags);
                        nParamMax[] <- (1 +
                                            # ETS model
                                            etsModel*((Etype!="N") + (Ttype!="N") + (Stype!="N")*length(lags) + damped +
                                                          (initial=="optimal") * ((Etype!="N") + (Ttype!="N") + (Stype!="N")*sum(lags))) +
                                            # ARIMA components: initials + parameters
                                            arimaModel*((initial=="optimal")*initialArimaNumber + sum(arMax) + sum(maMax)) +
                                            # Xreg initials and smoothing parameters
                                            xregModel*(xregNumber*(1+(regressors=="adapt"))));
                    }
                }
            }
        }
    }

    #### Parallel calculations ####
    # Check the parallel parameter and set the number of cores
    if(is.numeric(parallel)){
        nCores <- parallel;
        parallel <- TRUE
    }
    else{
        nCores <- min(parallel::detectCores() - 1, nModels);
    }

    # If this is parallel, then load the required packages
    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }

        # Check the system and choose the package to use
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cat("Setting up", nCores, "clusters using 'doParallel'...\n");
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.",
                     call. = FALSE);
            }
        }
        else{
            if(requireNamespace("doMC", quietly = TRUE)){
                doMC::registerDoMC(nCores);
                cluster <- NULL;
            }
            else if(requireNamespace("doParallel", quietly = TRUE)){
                cat("Setting up", nCores, "clusters using 'doParallel'...\n");
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop(paste0("Sorry, but in order to run the function in parallel, you need either ",
                            "'doMC' (prefered) or 'doParallel' package."),
                     call. = FALSE);
            }
        }
    }

    if(!silent){
        if(!parallel){
            cat("Evaluating models with different distributions... ");
        }
        else{
            cat("Working... ");
        }
    }

    #### The function that does the loop and returns a list of ETS(X) models ####
    adamReturner <- function(data, model, lags, orders,
                             distribution, h, holdout,
                             persistence, phi, initial, arma,
                             occurrence, ic, bounds,
                             regressors, parallel,
                             arimaModelSelect, arMax, iMax, maMax, ...){
        # If we select ARIMA, don't do it in the first step
        if(arimaModelSelect){
            ordersToUse <- c(0,0,0);
        }
        else{
            ordersToUse <- orders;
        }

        if(!parallel){
            # Prepare the list of models
            selectedModels <- vector("list",length(distribution));
            for(i in 1:length(distribution)){
                if(!silent){
                    cat(distribution[i],"\b, ");
                }
                selectedModels[[i]] <- adam(data=data, model=model, lags=lags, orders=ordersToUse,
                                            distribution=distribution[i], formula=formula,
                                            h=h, holdout=holdout,
                                            persistence=persistence, phi=phi, initial=initial, arma=arma,
                                            occurrence=occurrence, ic=ic, bounds=bounds,
                                            regressors=regressors, silent=TRUE, ...);

                if(arimaModelSelect){
                    selectedModels[[i]] <- arimaSelector(data=data, model=model,
                                                         lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                                                         distribution=selectedModels[[i]]$distribution, h=h, holdout=holdout,
                                                         persistence=persistence, phi=phi, initial=initial,
                                                         occurrence=occurrence, ic=ic, bounds=bounds, fast=fast,
                                                         silent=silent, regressors=regressors,
                                                         testModelETS=selectedModels[[i]], ...)
                }
            }
        }
        else{
            selectedModels <- foreach::`%dopar%`(foreach::foreach(i=1:length(distribution)),{
                testModel <- adam(data=data, model=model, lags=lags, orders=ordersToUse,
                                  distribution=distribution[i], formula=formula,
                                  h=h, holdout=holdout,
                                  persistence=persistence, phi=phi, initial=initial, arma=arma,
                                  occurrence=occurrence, ic=ic, bounds=bounds,
                                  regressors=regressors, silent=TRUE, ...)

                if(arimaModelSelect){
                    testModel <- arimaSelector(data=data, model=model,
                                               lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                                               distribution=testModel$distribution, h=h, holdout=holdout,
                                               persistence=persistence, phi=phi, initial=initial,
                                               occurrence=occurrence, ic=ic,
                                               bounds=bounds, fast=fast,
                                               silent=TRUE, regressors=regressors,
                                               testModelETS=testModel, ...)
                }
                return(testModel);
            })
        }
        return(selectedModels);
    }

    #### ARIMA selection script ####
    if(arimaModelSelect){
        if(!is.null(arma)){
            warning("ARIMA order selection cannot be done with the provided arma parameters. Dropping them.",
                    call.=FALSE);
            arma <- NULL;
        }

        #### Function corrects IC taking number of parameters on previous step ####
        icCorrector <- function(llikelihood, ic, nParam, obsNonzero){
            if(ic=="AIC"){
                correction <- 2*nParam - 2*llikelihood;
            }
            else if(ic=="AICc"){
                if(nParam>=obsNonzero-1){
                    correction <- Inf;
                }
                else{
                    correction <- 2*nParam*obsNonzero/(obsNonzero-nParam-1) - 2*llikelihood;
                }
            }
            else if(ic=="BIC"){
                correction <- nParam*log(obsNonzero) - 2*llikelihood;
            }
            else if(ic=="BICc"){
                if(nParam>=obsNonzero-1){
                    correction <- Inf;
                }
                else{
                    correction <- (nParam*log(obsNonzero)*obsNonzero)/(obsNonzero-nParam-1) - 2*llikelihood;
                }
            }

            return(correction);
        }

        #### The function that selects ARIMA orders for the provided data ####
        arimaSelector <- function(data, model, lags, arMax, iMax, maMax,
                                  distribution, h, holdout,
                                  persistence, phi, initial,
                                  occurrence, ic, bounds, fast,
                                  silent, regressors, testModelETS, ...){
            silentDebug <- FALSE;

            # Save the original values
            modelOriginal <- model;
            occurrenceOriginal <- occurrence;
            persistenceOriginal <- persistence;
            phiOriginal <- phi;

            # If the ETS model was done before this, then extract residuals
            if(is.adam(testModelETS)){
                dataAR <- dataI <- dataMA <- yInSample <- residuals(testModelETS);
                model <- "NNN";
                occurrence <- "none"
                persistence <- NULL;
                phi <- NULL;

                # Don't count the scale term
                nParamOriginal <- nparam(testModelETS)-1;
            }
            else{
                # Fit just mean
                testModelETS <- adam(data, model="NNN",lags=1, distribution=distribution, formula=formula,
                                     h=h,holdout=holdout, occurrence=occurrence, bounds=bounds, silent=TRUE);
                dataAR <- dataI <- dataMA <- yInSample <- actuals(testModelETS);

                # Originally, we only have a constant
                nParamOriginal <- 1;
            }
            testModel <- testModelETS;
            bestIC <- bestICI <- ICFunction(testModel);
            obsNonzero <- nobs(testModelETS,all=FALSE);

            if(silentDebug){
                cat("Best IC:",bestIC,"\n");
            }
            if(!silent){
                cat(" Selecting ARIMA orders...    ");
            }

            # 1 stands for constant/no constant, another one stands for ARIMA(0,0,0)
            if(all(maMax==0)){
                nModelsARIMA <- prod(iMax + 1) * (1 + sum(arMax));
            }
            else{
                nModelsARIMA <- prod(iMax + 1) * (1 + sum(maMax*(1 + sum(arMax))));
            }
            ICValue <- 1E+100;
            m <- 0;

            lagsTest <- maTest <- arTest <- rep(0,length(lags));
            arBest <- maBest <- iBest <- rep(0,length(lags));
            arBestLocal <- maBestLocal <- arBest;

            iOrders <- matrix(0,prod(iMax+1),ncol=length(iMax));

            ##### Loop for differences #####
            # Prepare table with differences
            if(any(iMax!=0)){
                iOrders[,1] <- rep(c(0:iMax[1]),times=prod(iMax[-1]+1));
                if(length(iMax)>1){
                    for(seasLag in 2:length(iMax)){
                        iOrders[,seasLag] <- rep(c(0:iMax[seasLag]),each=prod(iMax[1:(seasLag-1)]+1))
                    }
                }
            }
            # Start the loop for differences
            for(d in 1:nrow(iOrders)){
                m <- m + 1;
                if(!silent){
                    cat(paste0(rep("\b",nchar(round(m/nModelsARIMA,2)*100)+1),collapse=""));
                    cat(round((m)/nModelsARIMA,2)*100,"\b%");
                }
                nParamInitial <- 0;
                # If differences are zero, skip this step
                if(!all(iOrders[d,]==0)){
                    # Run the model for differences
                    testModel <- adam(data=yInSample, model=model, lags=lags,
                                      orders=list(ar=0,i=iOrders[d,],ma=0),
                                      distribution=distribution,
                                      h=h, holdout=FALSE,
                                      persistence=persistence, phi=phi, initial=initial,
                                      occurrence=occurrence, ic=ic, bounds=bounds,
                                      regressors=regressors, silent=TRUE, ...);
                    nParamInitial[] <- (initial=="optimal") * (iOrders[d,] %*% lags);
                }
                # Extract Information criteria
                ICValue <- ICFunction(testModel);
                if(silentDebug){
                    cat("I:",iOrders[d,],"\b,",ICValue,"\n");
                }
                if(ICValue < bestICI){
                    bestICI <- ICValue;
                    dataMA <- dataI <- residuals(testModel);
                    if(ICValue < bestIC){
                        iBest <- iOrders[d,];
                        bestIC <- ICValue;
                        maBest <- arBest <- rep(0,length(arTest));
                    }
                }
                else{
                    if(fast){
                        m <- m + sum(maMax*(1 + sum(arMax)));
                        next;
                    }
                    else{
                        dataMA <- dataI <- residuals(testModel);
                    }
                }

                ##### Loop for MA #####
                if(any(maMax!=0)){
                    bestICMA <- bestICI;
                    maBestLocal <- maTest <- rep(0,length(maTest));
                    for(seasSelectMA in 1:length(lags)){
                        if(maMax[seasSelectMA]!=0){
                            for(maSelect in 1:maMax[seasSelectMA]){
                                m <- m + 1;
                                if(!silent){
                                    cat(paste0(rep("\b",nchar(round(m/nModelsARIMA,2)*100)+1),collapse=""));
                                    cat(round((m)/nModelsARIMA,2)*100,"\b%");
                                }
                                maTest[seasSelectMA] <- maMax[seasSelectMA] - maSelect + 1;

                                # Run the model for MA
                                testModel <- adam(data=dataI, model="NNN", lags=lags,
                                                  orders=list(ar=0,i=0,ma=maTest),
                                                  distribution=distribution,
                                                  h=h, holdout=FALSE,
                                                  persistence=NULL, phi=NULL, initial=initial,
                                                  occurrence="none", ic=ic, bounds=bounds,
                                                  regressors="use", silent=TRUE, ...);
                                if(initial=="optimal" && (maTest %*% lags > nParamInitial)){
                                    nParamInitial[] <-  (maTest %*% lags);
                                }
                                # Exclude the initials from the number of parameters
                                nParamMA <- sum(maTest);
                                ICValue <- icCorrector(logLik(testModel), ic,
                                                       nParamOriginal + nParamMA + nParamInitial,
                                                       obsNonzero);
                                if(silentDebug){
                                    cat("MA:",maTest,"\b,",ICValue,"\n");
                                }
                                if(ICValue < bestICMA){
                                    bestICMA <- ICValue;
                                    maBestLocal <- maTest;
                                    if(ICValue < bestIC){
                                        bestIC <- bestICMA;
                                        iBest <- iOrders[d,];
                                        maBest <- maTest;
                                        arBest <- rep(0,length(arTest));
                                    }
                                    dataMA <- residuals(testModel);
                                }
                                else{
                                    if(fast){
                                        m <- m + maTest[seasSelectMA] * (1 + sum(arMax)) - 1;
                                        maTest <- maBestLocal;
                                        break;
                                    }
                                    else{
                                        maTest <- maBestLocal;
                                        dataMA <- residuals(testModel);
                                    }
                                }

                                ##### Loop for AR #####
                                if(any(arMax!=0)){
                                    bestICAR <- bestICMA;
                                    arBestLocal <- arTest <- rep(0,length(arTest));
                                    for(seasSelectAR in 1:length(lags)){
                                        lagsTest[seasSelectAR] <- lags[seasSelectAR];
                                        if(arMax[seasSelectAR]!=0){
                                            for(arSelect in 1:arMax[seasSelectAR]){
                                                m <- m + 1;
                                                if(!silent){
                                                    cat(paste0(rep("\b",nchar(round(m/nModelsARIMA,2)*100)+1),collapse=""));
                                                    cat(round((m)/nModelsARIMA,2)*100,"\b%");
                                                }
                                                arTest[seasSelectAR] <- arMax[seasSelectAR] - arSelect + 1;

                                                # Run the model for AR
                                                testModel <- adam(data=dataMA, model="NNN", lags=lags,
                                                                  orders=list(ar=arTest,i=0,ma=0),
                                                                  distribution=distribution,
                                                                  h=h, holdout=FALSE,
                                                                  persistence=NULL, phi=NULL, initial=initial,
                                                                  occurrence="none", ic=ic, bounds=bounds,
                                                                  regressors="use", silent=TRUE, ...);
                                                if(initial=="optimal" && (arTest %*% lags > nParamInitial)){
                                                    nParamInitial[] <-  (arTest %*% lags);
                                                }
                                                # Exclude the initials (in order not to duplicate them)
                                                nParamAR <- sum(arTest);
                                                ICValue <- icCorrector(logLik(testModel), ic,
                                                                       nParamOriginal + nParamMA + nParamAR + nParamInitial,
                                                                       obsNonzero);
                                                if(silentDebug){
                                                    cat("AR:",arTest,"\b,",ICValue,"\n");
                                                }
                                                if(ICValue < bestICAR){
                                                    bestICAR <- ICValue;
                                                    arBestLocal <- arTest;
                                                    if(ICValue < bestIC){
                                                        bestIC <- ICValue;
                                                        iBest <- iOrders[d,];
                                                        arBest <- arTest;
                                                        maBest <- maTest;
                                                    }
                                                }
                                                else{
                                                    if(fast){
                                                        m <- m + arTest[seasSelectAR] - 1;
                                                        arTest <- arBestLocal;
                                                        break;
                                                    }
                                                    else{
                                                        arTest <- arBestLocal;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else{
                    ##### Loop for AR #####
                    if(any(arMax!=0)){
                        bestICAR <- bestICMA;
                        arBestLocal <- arTest <- rep(0,length(arTest));
                        for(seasSelectAR in 1:length(lags)){
                            lagsTest[seasSelectAR] <- lags[seasSelectAR];
                            if(arMax[seasSelectAR]!=0){
                                for(arSelect in 1:arMax[seasSelectAR]){
                                    m <- m + 1;
                                    if(!silent){
                                        cat(paste0(rep("\b",nchar(round(m/nModelsARIMA,2)*100)+1),collapse=""));
                                        cat(round((m)/nModelsARIMA,2)*100,"\b%");
                                    }
                                    arTest[seasSelectAR] <- arMax[seasSelectAR] - arSelect + 1;

                                    # Run the model for MA
                                    testModel <- adam(data=dataI, model="NNN", lags=lags,
                                                      orders=list(ar=arTest,i=0,ma=0),
                                                      distribution=distribution,
                                                      h=h, holdout=FALSE,
                                                      persistence=NULL, phi=NULL, initial=initial,
                                                      occurrence="none", ic=ic, bounds=bounds,
                                                      regressors="use", silent=TRUE, ...);
                                    if(initial=="optimal" && (arTest %*% lags > nParamInitial)){
                                        nParamInitial[] <-  (arTest %*% lags);
                                    }
                                    # Exclude the initials (in order not to duplicate them)
                                    nParamAR <- sum(arTest);
                                    ICValue <- icCorrector(logLik(testModel), ic,
                                                           nParamOriginal + nParamAR + nParamInitial,
                                                           obsNonzero);
                                    if(silentDebug){
                                        cat("AR:",arTest,"\b,",ICValue,"\n");
                                    }
                                    if(ICValue < bestICAR){
                                        bestICAR <- ICValue;
                                        arBestLocal <- arTest;
                                        if(ICValue < bestIC){
                                            bestIC <- ICValue;
                                            iBest <- iOrders[d,];
                                            arBest <- arTest;
                                            maBest <- maTest;
                                        }
                                    }
                                    else{
                                        if(fast){
                                            m <- m + arTest[seasSelectAR] - 1;
                                            arTest <- arBestLocal;
                                            break;
                                        }
                                        else{
                                            arTest <- arBestLocal;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if(!silent && fast){
                cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
                cat(" ",100,"\b%");
            }

            #### Reestimate the best model in order to get rid of bias ####
            # Run the model for MA
            bestModel <- adam(data=data, model=modelOriginal, lags=lags,
                              orders=list(ar=(arBest),i=(iBest),ma=(maBest)),
                              distribution=distribution, formula=formula,
                              h=h, holdout=holdout,
                              persistence=persistenceOriginal, phi=phiOriginal, initial=initial,
                              occurrence=occurrenceOriginal, ic=ic, bounds=bounds,
                              regressors="use", silent=TRUE, ...);

            if(!silent){
                cat(". The best ARIMA is selected.\n");
            }
            return(bestModel);
        }
    }

    #### The function that extracts outliers and refits the model with the selection ####
    # Outliers types to include: additive outlier (AO) and level shift (LS).
    # Introduce leads and lags of outliers in case of "select"
    outlierDetector <- function(adamModel, outliers="use"){
        if(outliers=="ignore"){
            return(adamModel);
        }
        else{
            outliersModel <- outlierdummy(adamModel, level=level);
            if(length(outliersModel$id)>0){
                if(!silent){
                    cat("\nDealing with outliers...");
                }
                # Create a proper xreg matrix
                if(h>0){
                    outliersXreg <- rbind(outliersModel$outliers, matrix(0,nrow=h,length(outliersModel$id)));
                }
                else{
                    outliersXreg <- outliersModel$outliers;
                }
                # If select outliers, then introduce lags and leads
                if(outliers=="select"){
                    outliersXreg <- xregExpander(outliersXreg,c(-1:1),gaps="zero");
                }
                outliersDo <- outliers;
                data <- cbind(data,outliersXreg);
                # Form new xreg matrix (check data and xreg)
                if(xregModel){
                    # Update formula if it is provided
                    if(!is.null(formula)){
                        formula <- update(as.formula(formula),
                                          as.formula(paste0("~.+",paste0(colnames(outliersXreg),collapse="+"))));
                    }
                    else{
                        formula <- as.formula(paste0(responseName,"~."));
                    }
                    outliersDo <- regressors;
                }
                else{
                    colnames(data)[1] <- responseName;
                }
                adamModel <- suppressWarnings(auto.adam(data, model, lags=lags, orders=orders,
                                                        formula=formula,
                                                        distribution=distribution, h=h, holdout=holdout,
                                                        persistence=persistence, phi=phi, initial=initial, arma=arma,
                                                        occurrence=occurrence, ic=ic, bounds=bounds,
                                                        regressors=outliersDo,
                                                        silent=TRUE, parallel=parallel, fast=fast));
            }
            else{
                if(!silent){
                    cat("No outliers detected.\n");
                }
            }
            return(adamModel);
        }
    }

    #### A simple loop, no ARIMA orders selection ####
    if(!arimaModelSelect){
        selectedModels <- adamReturner(data, model, lags, orders,
                                       distribution, h, holdout,
                                       persistence, phi, initial, arma,
                                       occurrence, ic, bounds,
                                       regressors, parallel,
                                       arimaModelSelect, arMax, iMax, maMax, ...);
    }
    else{
        #### If there is ETS(X), do ARIMA selection on residuals ####
        # Extract residuals from adams for each distribution, fit best ARIMA for each, refit the models.
        if(etsModel || xregModel){
            selectedModels <- adamReturner(data, model, lags, orders,
                                           distribution, h, holdout,
                                           persistence, phi, initial, arma,
                                           occurrence, ic, bounds,
                                           regressors, parallel,
                                           arimaModelSelect, arMax, iMax, maMax, ...);
        }
        #### Otherwise, do the stuff directly ####
        # Do ARIMA selection for each distribution in parallel.
        else{
            if(!parallel){
                # Prepare the list of models
                selectedModels <- vector("list",length(distribution));
                for(i in 1:length(distribution)){
                    if(!silent){
                        cat(distribution[i],": ");
                    }
                    selectedModels[[i]] <- arimaSelector(data=data, model=model,
                                                         lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                                                         distribution=distribution[i], h=h, holdout=holdout,
                                                         persistence=persistence, phi=phi, initial=initial,
                                                         occurrence=occurrence, ic=ic, bounds=bounds, fast=fast,
                                                         silent=TRUE, regressors=regressors, testModelETS=NULL, ...);
                }
            }
            else{
                selectedModels <- foreach::`%dopar%`(foreach::foreach(i=1:length(distribution)),{
                    testModel <- arimaSelector(data=data, model=model,
                                               lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                                               distribution=distribution[i], h=h, holdout=holdout,
                                               persistence=persistence, phi=phi, initial=initial,
                                               occurrence=occurrence, ic=ic, bounds=bounds, fast=fast,
                                               silent=TRUE, regressors=regressors, testModelETS=NULL, ...);
                    return(testModel);
                })
            }
        }
    }

    if(modelDo=="select"){
        ICValues <- sapply(selectedModels, ICFunction);
    }
    else{
        ICValues <- vector("numeric",length(distribution));
        for(i in 1:length(distribution)){
            ICValues[i] <- selectedModels[[i]]$ICs[is.finite(selectedModels[[i]]$ICs)] %*%
                selectedModels[[i]]$ICw[is.finite(selectedModels[[i]]$ICs)];
        }
    }

    selectedModels[[which.min(ICValues)]]$timeElapsed <- Sys.time()-startTime;
    # selectedModels[[which.min(ICValues)]]$formula <- as.formula(do.call("substitute",
    #                                                                     list(expr=selectedModels[[which.min(ICValues)]]$formula,
    #                                                                          env=list(y=as.name(responseName)))));
    # names(ICValues) <- sapply(selectedModels, modelType);
    # selectedModels[[which.min(ICValues)]]$ICValues <- ICValues;

    if(outliers!="ignore"){
        selectedModels[[which.min(ICValues)]] <- outlierDetector(selectedModels[[which.min(ICValues)]], outliers=outliers);
    }

    if(!silent){
        if(outliers=="ignore"){
            cat("Done!\n");
        }
        plot(selectedModels[[which.min(ICValues)]],7);
    }

    return(selectedModels[[which.min(ICValues)]]);
}
