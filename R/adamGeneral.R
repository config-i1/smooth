parametersChecker <- function(data, model, lags, formulaProvided, orders, arma,
                              persistence, phi, initial,
                              distribution=c("default","dnorm","dlaplace","ds","dgnorm","dalaplace",
                                             "dlnorm","dinvgauss"),
                              loss, h, holdout,occurrence,
                              ic=c("AICc","AIC","BIC","BICc"), bounds=c("traditional","usual","admissible","none"),
                              regressors, yName,
                              silent, modelDo, ParentEnvironment,
                              ellipsis, fast=FALSE){

    # The function checks the provided parameters of adam and/or oes
    ##### data #####
    # yName is the name of the object. It might differ from response if matrix is provided
    responseName <- yName;

    # If this is simulated, extract the actuals
    if(is.adam.sim(data) || is.smooth.sim(data)){
        data <- data$data;
    }
    # If this is Mdata, use all the available stuff
    else if(inherits(data,"Mdata")){
        h <- data$h;
        holdout <- TRUE;
        if(modelDo!="use"){
            lags <- frequency(data$x);
        }
        data <- ts(c(data$x,data$xx),start=start(data$x),frequency=frequency(data$x));
    }

    # Extract index from the object in order to use it later
    ### tsibble has its own index function, so shit happens becaus of it...
    if(inherits(data,"tbl_ts")){
        yIndex <- data[[1]];
        if(any(duplicated(yIndex))){
            warning(paste0("You have duplicated time stamps in the variable ",yName,
                           ". We will refactor this."),call.=FALSE);
            yIndex <- yIndex[1] + c(1:length(data[[1]])) * diff(tail(yIndex,2));
        }
    }
    else{
        yIndex <- try(time(data),silent=TRUE);
        # If we cannot extract time, do something
        if(inherits(yIndex,"try-error")){
            if(!is.data.frame(data) && !is.null(dim(data))){
                yIndex <- as.POSIXct(rownames(data));
            }
            else if(is.data.frame(data)){
                yIndex <- c(1:nrow(data));
            }
            else{
                yIndex <- c(1:length(data));
            }
        }
    }
    yClasses <- class(data);

    # If this is something like a matrix
    if(!is.null(ncol(data)) && ncol(data)>1){
        xregData <- data;
        if(!is.null(formulaProvided)){
            responseName <- all.vars(formulaProvided)[1];
            y <- data[,responseName];
        }
        else{
            responseName <- colnames(xregData)[1];
            # If we deal with data.table / tibble / data.frame, the syntax is different.
            # We don't want to import specific classes, so just use inherits()
            if(inherits(data,"tbl_ts")){
                # With tsibble we cannot extract explanatory variables easily...
                y <- data$value;
            }
            else if(inherits(data,"data.table") || inherits(data,"tbl") || inherits(data,"data.frame")){
                y <- data[[1]];
            }
            else if(inherits(data,"zoo")){
                if(ncol(data)>1){
                    xregData <- as.data.frame(data);
                }
                y <- zoo(data[,1],order.by=time(data));
            }
            else{
                y <- data[,1];
            }
        }
        # Give the indeces another try
        yIndex <- try(time(y),silent=TRUE);
        # If we cannot extract time, do something
        if(inherits(yIndex,"try-error")){
            if(!is.null(dim(data))){
                yIndex <- as.POSIXct(rownames(data));
            }
            else{
                yIndex <- c(1:length(y));
            }
        }
    }
    else{
        xregData <- NULL;
        y <- data;
    }

    # Make the response a secure name
    responseName <- make.names(responseName);

    # Substitute NAs with mean values.
    yNAValues <- is.na(y);
    if(any(yNAValues)){
        warning("Data contains NAs. The values will be ignored during the model construction.",call.=FALSE);
        y[yNAValues] <- na.interp(y)[yNAValues];
    }

    # Define obs, the number of observations of in-sample
    obsAll <- length(y) + (1 - holdout)*h;
    obsInSample <- length(y) - holdout*h;

    if(obsInSample<=0){
        stop("The number of in-sample observations is not positive. Cannot do anything.",
             call.=FALSE);
    }

    # If this is just a numeric variable, use ts class
    if(all(yClasses=="integer") || all(yClasses=="numeric") ||
       all(yClasses=="data.frame") || all(yClasses=="matrix")){
        if(any(class(yIndex) %in% c("POSIXct","Date"))){
            yClasses <- "zoo";
        }
        else{
            yClasses <- "ts";
        }
    }
    yFrequency <- frequency(y);
    yStart <- yIndex[1];
    yInSample <- matrix(y[1:obsInSample],ncol=1);
    if(holdout){
        yForecastStart <- yIndex[obsInSample+1];
        yHoldout <- matrix(y[-c(1:obsInSample)],ncol=1);
        yForecastIndex <- yIndex[-c(1:obsInSample)];
        yInSampleIndex <- yIndex[c(1:obsInSample)];
        yIndexAll <- yIndex;
    }
    else{
        yInSampleIndex <- yIndex;
        yIndexDiff <- diff(tail(yIndex,2));
        yForecastStart <- yIndex[obsInSample]+yIndexDiff;
        if(any(yClasses=="ts")){
            yForecastIndex <- yIndex[obsInSample]+as.numeric(yIndexDiff)*c(1:max(h,1));
        }
        else{
            yForecastIndex <- yIndex[obsInSample]+yIndexDiff*c(1:max(h,1));
        }
        yHoldout <- NULL;
        yIndexAll <- c(yIndex,yForecastIndex);
    }

    if(!is.numeric(yInSample)){
        stop("The provided data is not numeric! Can't construct any model!", call.=FALSE);
    }

    # Number of parameters to estimate / provided
    parametersNumber <- matrix(0,2,4,
                               dimnames=list(c("Estimated","Provided"),
                                             c("nParamInternal","nParamXreg","nParamOccurrence","nParamAll")));

    #### Check what is used for the model ####
    if(!is.character(model)){
        stop(paste0("Something strange is provided instead of character object in model: ",
                    paste0(model,collapse=",")),call.=FALSE);
    }

    # Predefine models pool for a model selection
    modelsPool <- NULL;
    if(!fast){
        # Deal with the list of models. Check what has been provided. Stop if there is a mistake.
        if(length(model)>1){
            if(any(nchar(model)>4)){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)>4],collapse=",")),call.=FALSE);
            }
            else if(any(substr(model,1,1)!="A" & substr(model,1,1)!="M" & substr(model,1,1)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,1,1)!="A" & substr(model,1,1)!="M"],collapse=",")),
                     call.=FALSE);
            }
            else if(any(substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                        substr(model,2,2)!="M" & substr(model,2,2)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                                             substr(model,2,2)!="M"],collapse=",")),call.=FALSE);
            }
            else if(any(substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                        substr(model,3,3)!="M" & substr(model,3,3)!="d" & substr(model,3,3)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                                             substr(model,3,3)!="M" & substr(model,3,3)!="d"],collapse=",")),
                     call.=FALSE);
            }
            else if(any(nchar(model)==4 & substr(model,4,4)!="N" & substr(model,4,4)!="A" &
                        substr(model,4,4)!="M" & substr(model,4,4)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)==4 & substr(model,4,4)!="N" &
                                             substr(model,4,4)!="A" & substr(model,4,4)!="M"],collapse=",")),
                     call.=FALSE);
            }
            else{
                modelsPoolCombiner <- (substr(model,1,1)=="C" | substr(model,2,2)=="C" |
                                           substr(model,3,3)=="C" | substr(model,4,4)=="C");
                modelsPool <- model[!modelsPoolCombiner];
                modelsPool <- unique(modelsPool);
                if(any(modelsPoolCombiner)){
                    if(any(substr(model,nchar(model),nchar(model))!="N")){
                        model <- "CCC";
                    }
                    else{
                        model <- "CCN";
                    }
                }
                else{
                    model <- c("Z","Z","Z");
                    if(all(substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N")){
                        model[3] <- "N";
                    }
                    if(all(substr(modelsPool,2,2)=="N")){
                        model[2] <- "N";
                    }
                    model <- paste0(model,collapse="");
                }
            }
        }
    }

    # If chosen model is "AAdN" or anything like that, we are taking the appropriate values
    if(nchar(model)==4){
        Etype <- substr(model,1,1);
        Ttype <- substr(model,2,2);
        Stype <- substr(model,4,4);
        damped <- TRUE;
        if(substr(model,3,3)!="d"){
            message(paste0("You have defined a strange model: ",model));
            model <- paste0(Etype,Ttype,"d",Stype);
        }
    }
    else if(nchar(model)==3){
        Etype <- substr(model,1,1);
        Ttype <- substr(model,2,2);
        Stype <- substr(model,3,3);
        if(any(Ttype==c("Z","X","Y"))){
            damped <- TRUE;
        }
        else{
            damped <- FALSE;
        }
    }
    else{
        message(paste0("You have defined a strange model: ",model));
        message("Switching to 'ZZZ'");
        model <- "ZZZ";

        Etype <- "Z";
        Ttype <- "Z";
        Stype <- "Z";
        damped <- TRUE;
    }

    # Define if we want to select or combine models... or do none of the above.
    if(is.null(modelsPool)){
        if(any(unlist(strsplit(model,""))=="C")){
            modelDo <- "combine";
            if(Etype=="C"){
                Etype <- "Z";
            }
            if(Ttype=="C"){
                Ttype <- "Z";
            }
            if(Stype=="C"){
                Stype <- "Z";
            }
        }
        else if(any(unlist(strsplit(model,""))=="Z") ||
                any(unlist(strsplit(model,""))=="X") ||
                any(unlist(strsplit(model,""))=="Y") ||
                any(unlist(strsplit(model,""))=="F") ||
                any(unlist(strsplit(model,""))=="P")){
            modelDo <- "select";

            # The full test, sidestepping branch and bound
            if(any(unlist(strsplit(model,""))=="F")){
                modelsPool <- c("ANN","AAN","AAdN","AMN","AMdN",
                                "ANA","AAA","AAdA","AMA","AMdA",
                                "ANM","AAM","AAdM","AMM","AMdM",
                                "MNN","MAN","MAdN","MMN","MMdN",
                                "MNA","MAA","MAdA","MMA","MMdA",
                                "MNM","MAM","MAdM","MMM","MMdM");
                Etype[] <- Ttype[] <- Stype[] <- "Z";
                model <- "FFF";
            }

            # The test for pure models only
            if(any(unlist(strsplit(model,""))=="P")){
                modelsPool <- c("ANN","AAN","AAdN","ANA","AAA","AAdA",
                                "MNN","MMN","MMdN","MNM","MMM","MMdM");
                Etype[] <- Ttype[] <- Stype[] <- "Z";
                model <- "PPP";
            }
        }
        else{
            modelDo <- "estimate";
        }

        if(Etype=="X"){
            Etype <- "A";
        }
        else if(Etype=="Y"){
            Etype <- "M";
        }
    }
    else{
        if(any(unlist(strsplit(model,""))=="C")){
            modelDo <- "combine";
        }
        else{
            modelDo <- "select";
        }
    }

    #### Check the components of model ####
    ### Check error type
    if(all(Etype!=c("Z","X","Y","A","M","C","N"))){
        warning(paste0("Wrong error type: ",Etype,". Should be 'Z', 'X', 'Y', 'C', 'A', 'M' or 'N'. ",
                       "Changing to 'Z'"),call.=FALSE);
        Etype <- "Z";
        modelDo <- "select";
    }
    # If the error is "N", then switch off the ETS model
    if(Etype=="N"){
        componentsNamesETS <- NULL;
        componentsNumberETS <- 0;
        Ttype[] <- "N";
        damped[] <- FALSE;
        Stype[] <- "N";
        modelDo[] <- "estimate";
        modelsPool[] <- NULL;
        etsModel <- FALSE;
    }
    else{
        componentsNamesETS <- "level";
        componentsNumberETS <- 1;
        etsModel <- TRUE;
    }

    if(etsModel){
        ### Check trend type
        if(all(Ttype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong trend type: ",Ttype,". Should be 'Z', 'X', 'Y', 'C' 'N', 'A', or 'M'. ",
                           "Changing to 'Z'"),call.=FALSE);
            Ttype <- "Z";
            modelDo <- "select";
        }
        modelIsTrendy <- (Ttype!="N");
        if(modelIsTrendy){
            componentsNamesETS <- c(componentsNamesETS,"trend");
            componentsNumberETS[] <- componentsNumberETS+1;
        }
    }
    else{
        modelIsTrendy <- FALSE;
    }

    #### Check the lags vector ####
    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        lags <- lags[lags!=0];
    }

    # Form the lags based on the provided stuff. Get rid of ones and leave unique seasonals
    # Add one for the level
    lags <- c(1,unique(lags[lags>1]));

    #### ARIMA term ####
    # This should be available for pure models only
    if(is.list(orders)){
        arOrders <- orders$ar;
        iOrders <- orders$i;
        maOrders <- orders$ma;
    }
    else if(is.vector(orders)){
        arOrders <- orders[1];
        iOrders <- orders[2];
        maOrders <- orders[3];
    }

    # If there is arima, prepare orders
    if(sum(c(arOrders,iOrders,maOrders))>0){
        arimaModel <- TRUE;

        # See if AR is needed
        arRequired <- FALSE;
        if(sum(arOrders)>0){
            arRequired[] <- TRUE;
        }

        # See if I is needed
        iRequired <- FALSE;
        if(sum(iOrders)>0){
            iRequired[] <- TRUE;
        }

        # See if I is needed
        maRequired <- FALSE;
        if(sum(maOrders)>0){
            maRequired[] <- TRUE;
        }

        # Define maxOrder and make all the values look similar (for the polynomials)
        maxOrder <- max(length(arOrders),length(iOrders),length(maOrders),length(lags));
        if(length(arOrders)!=maxOrder){
            arOrders <- c(arOrders,rep(0,maxOrder-length(arOrders)));
        }
        if(length(iOrders)!=maxOrder){
            iOrders <- c(iOrders,rep(0,maxOrder-length(iOrders)));
        }
        if(length(maOrders)!=maxOrder){
            maOrders <- c(maOrders,rep(0,maxOrder-length(maOrders)));
        }
        if(length(lags)!=maxOrder){
            lagsNew <- c(lags,rep(0,maxOrder-length(lags)));
            arOrders <- arOrders[lagsNew!=0];
            iOrders <- iOrders[lagsNew!=0];
            maOrders <- maOrders[lagsNew!=0];
        }

        # Define the non-zero values. This is done via the calculation of orders of polynomials
        ariValues <- list(NA);
        maValues <- list(NA);
        for(i in 1:length(lags)){
            ariValues[[i]] <- c(0,min(1,arOrders[i]):arOrders[i])
            if(iOrders[i]!=0){
                ariValues[[i]] <- c(ariValues[[i]],1:iOrders[i]+arOrders[i]);
            }
            ariValues[[i]] <- unique(ariValues[[i]] * lags[i]);
            maValues[[i]] <- unique(c(0,min(1,maOrders[i]):maOrders[i]) * lags[i]);
        }

        # Produce ARI polynomials
        ariLengths <- unlist(lapply(ariValues,length));
        ariPolynomial <- array(0,ariLengths);
        for(i in 1:length(ariValues)){
            if(i==1){
                ariPolynomial <- ariPolynomial + array(ariValues[[i]], ariLengths);
            }
            else{
                ariPolynomial <- ariPolynomial + array(rep(ariValues[[i]],each=prod(ariLengths[1:(i-1)])),
                                                       ariLengths);
            }
        }

        # Produce MA polynomials
        maLengths <- unlist(lapply(maValues,length));
        maPolynomial <- array(0,maLengths);
        for(i in 1:length(maValues)){
            if(i==1){
                maPolynomial <- maPolynomial + array(maValues[[i]], maLengths);
            }
            else{
                maPolynomial <- maPolynomial + array(rep(maValues[[i]],each=prod(maLengths[1:(i-1)])),
                                                     maLengths);
            }
        }

        # What are the non-zero ARI and MA polynomials?
        ### What are their positions in transition matrix?
        nonZeroARI <- unique(matrix(c(ariPolynomial)[-1],ncol=1));
        nonZeroMA <- unique(matrix(c(maPolynomial)[-1],ncol=1));
        # Lags for the ARIMA components
        lagsModelARIMA <- matrix(sort(unique(c(nonZeroARI,nonZeroMA))),ncol=1);
        nonZeroARI <- cbind(nonZeroARI+1,which(lagsModelARIMA %in% nonZeroARI));
        nonZeroMA <- cbind(nonZeroMA+1,which(lagsModelARIMA %in% nonZeroMA));

        # Number of components
        componentsNumberARIMA <- length(lagsModelARIMA);
        # Their names
        componentsNamesARIMA <- paste0("ARIMAState",c(1:componentsNumberARIMA));
        # Number of initials needed. This is based on the longest one. The others are just its transformations
        initialArimaNumber <- max(lagsModelARIMA);

        if(obsInSample < initialArimaNumber){
            warning(paste0("In-sample size is ",obsInSample,", while number of ARIMA components is ",componentsNumberARIMA,
                           ". Cannot fit the model."),call.=FALSE)
            stop("Not enough observations for such a complicated model.",call.=FALSE);
        }
    }
    else{
        arOrders <- NULL;
        iOrders <- NULL;
        maOrders <- NULL;
        arimaModel <- FALSE;
        arRequired <- arEstimate <- FALSE;
        iRequired <- FALSE;
        maRequired <- maEstimate <- FALSE;
        lagsModelARIMA <- initialArimaNumber <- 0;
        componentsNumberARIMA <- 0;
        componentsNamesARIMA <- NULL;
        nonZeroARI <- NULL;
        nonZeroMA <- NULL;
    }

    if(etsModel){
        modelIsSeasonal <- Stype!="N";
    }
    else{
        modelIsSeasonal <- FALSE;
    }

    # Lags of the model used inside the functions
    lagsModel <- matrix(lags,ncol=1);

    # If we have a trend add one more lag
    if(modelIsTrendy){
        lagsModel <- rbind(1,lagsModel);
    }
    # If we don't have seasonality, remove seasonal lag
    if(!modelIsSeasonal && any(lagsModel>1)){
        lagsModel <- lagsModel[lagsModel==1,,drop=FALSE];
    }

    # If this is non-seasonal model and there are no seasonal ARIMA lags, trim the original lags
    if(!modelIsSeasonal && all(c(arOrders[lags>1],iOrders[lags>1],maOrders[lags>1])==0) && any(lags>1)){
        arOrders <- arOrders[lags==1];
        iOrders <- iOrders[lags==1];
        maOrders <- maOrders[lags==1];
        lags <- lags[lags==1];
    }

    # Lags of the model
    lagsModelSeasonal <- lagsModel[lagsModel>1];
    lagsModelMax <- max(lagsModel);
    lagsLength <- length(lagsModel);

    if(etsModel){
        #### Check the seasonal model vs lags ####
        if(all(Stype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong seasonality type: ",Stype,". Should be 'Z', 'X', 'Y', 'C', 'N', 'A' or 'M'. ",
                           "Setting to 'Z'."),call.=FALSE);
            if(lagsModelMax==1){
                Stype <- "N";
                modelIsSeasonal <- FALSE;
            }
            else{
                Stype <- "Z";
                modelDo <- "select";
            }
        }
        if(all(modelIsSeasonal,lagsModelMax==1)){
            if(all(Stype!=c("Z","X","Y"))){
                warning(paste0("Cannot build the seasonal model on data with the unity lags.\n",
                               "Switching to non-seasonal model: ETS(",substr(model,1,nchar(model)-1),"N)"));
            }
            Stype <- "N";
            modelIsSeasonal <- FALSE;
            substr(model,nchar(model),nchar(model)) <- "N";
        }

        # Check the pool of models to combine if it was decided that the data is not seasonal
        if(!modelIsSeasonal && !is.null(modelsPool)){
            modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N"];
        }

        # Check the type of seasonal
        if(Stype!="N"){
            componentsNamesETS <- c(componentsNamesETS,"seasonal");
            componentsNumberETS[] <- componentsNumberETS+1;
            componentsNumberETSSeasonal <- 1;
        }
        else{
            componentsNumberETSSeasonal <- 0;
        }

        # Check, whether the number of lags and the number of components are the same
        if(lagsLength>componentsNumberETS){
            if(Stype!="N"){
                componentsNamesETS <- c(componentsNamesETS[-length(componentsNamesETS)],
                                        paste0("seasonal",c(1:(lagsLength-componentsNumberETS-1))));
                componentsNumberETSSeasonal[] <- lagsLength-componentsNumberETS+1;
                componentsNumberETS[] <- lagsLength;
            }
            else{
                lagsModel <- lagsModel[1:componentsNumberETS,,drop=FALSE];
                lagsModelMax <- max(lagsModel);
                lagsLength <- length(lagsModel);
            }
        }
        else if(lagsLength<componentsNumberETS){
            stop("The number of components of the model is smaller than the number of provided lags", call.=FALSE);
        }
    }
    else{
        componentsNumberETSSeasonal <- 0;
    }

    if(!fast){
        #### Distribution selected ####
        distribution <- match.arg(distribution);
    }

    #### Loss function type ####
    if(is.function(loss)){
        lossFunction <- loss;
        loss <- "custom";
        multisteps <- FALSE;
    }
    else{
        loss <- match.arg(loss[1],c("likelihood","MSE","MAE","HAM","LASSO","RIDGE",
                                    "MSEh","TMSE","GTMSE","MSCE",
                                    "MAEh","TMAE","GTMAE","MACE",
                                    "HAMh","THAM","GTHAM","CHAM","GPL",
                                    "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"));

        if(any(loss==c("MSEh","TMSE","GTMSE","MSCE","MAEh","TMAE","GTMAE","MACE",
                       "HAMh","THAM","GTHAM","CHAM","GPL",
                       "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"))){
            if(!is.null(h) && h>0){
                multisteps <- TRUE;
            }
            else{
                stop("The horizon \"h\" needs to be specified and be positive in order for the multistep loss to work.",
                     call.=FALSE);
                multisteps <- FALSE;
            }
        }
        else{
            multisteps <- FALSE;
        }
        lossFunction <- NULL;
    }

    #### Explanatory variables: xregModel and regressors ####
    regressors <- match.arg(regressors,c("use","select","adapt"));
    xregModel <- !is.null(xregData);

    #### Persistence provided ####
    # Vectors for persistence of different components
    persistenceLevel <- NULL;
    persistenceTrend <- NULL;
    persistenceSeasonal <- NULL;
    persistenceXreg <- NULL;
    # InitialEstimate vectors, defining what needs to be estimated
    persistenceEstimate <- persistenceLevelEstimate <- persistenceTrendEstimate <-
        persistenceXregEstimate <- TRUE;
    # persistence of seasonal is a vector, not a scalar, because we can have several lags
    persistenceSeasonalEstimate <- rep(TRUE,componentsNumberETSSeasonal);
    if(!is.null(persistence)){
        if(all(modelDo!=c("estimate","use"))){
            warning(paste0("Predefined persistence can only be used with ",
                           "preselected ETS model.\n",
                           "Changing to estimation of persistence values."),call.=FALSE);
            persistence <- NULL;
            persistenceEstimate <- TRUE;
        }
        else{
            # If it is a list
            if(is.list(persistence)){
                # If this is a named list, then extract stuff using names
                if(!is.null(names(persistence))){
                    if(!is.null(persistence$level)){
                        persistenceLevel <- persistence$level;
                    }
                    else if(!is.null(persistence$alpha)){
                        persistenceLevel <- persistence$alpha;
                    }
                    if(!is.null(persistence$trend)){
                        persistenceTrend <- persistence$trend;
                    }
                    else if(!is.null(persistence$beta)){
                        persistenceTrend <- persistence$beta;
                    }
                    if(!is.null(persistence$seasonal)){
                        persistenceSeasonal <- persistence$seasonal;
                    }
                    else if(!is.null(persistence$gamma)){
                        persistenceSeasonal <- persistence$gamma;
                    }
                    if(!is.null(persistence$xreg)){
                        persistenceXreg <- persistence$xreg;
                    }
                    else if(!is.null(persistence$delta)){
                        persistenceXreg <- persistence$delta;
                    }
                }
                else{
                    if(!is.null(persistence[[1]])){
                        persistenceLevel <- persistence[[1]];
                    }
                    if(!is.null(persistence[[2]])){
                        persistenceTrend <- persistence[[2]];
                    }
                    if(!is.null(persistence[[3]])){
                        persistenceSeasonal <- persistence[[3]];
                    }
                    if(!is.null(persistence[[4]])){
                        persistenceXreg <- persistence[[4]];
                    }
                }
                # Define estimate variables
                if(!is.null(persistenceLevel)){
                    persistenceLevelEstimate[] <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                }
                if(!is.null(persistenceTrend)){
                    persistenceTrendEstimate[] <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                }
                if(!is.null(persistenceSeasonal)){
                    if(is.list(persistenceSeasonal)){
                        persistenceSeasonalEstimate[] <- length(persistenceSeasonal)==length(lagsModelSeasonal);
                    }
                    else{
                        persistenceSeasonalEstimate[] <- FALSE;
                    }
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(unlist(persistenceSeasonal));
                }
                if(!is.null(persistenceXreg)){
                    persistenceXregEstimate[] <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(persistenceXreg);
                }
            }
            else if(is.numeric(persistence)){
                # If it is smaller... We don't know the length of xreg yet at this stage
                if(length(persistence)<lagsLength){
                    warning(paste0("Length of persistence vector is wrong! ",
                                   "Changing to estimation of persistence vector values."),
                            call.=FALSE);
                    persistence <- NULL;
                    persistenceEstimate <- TRUE;
                }
                else{
                    # If there ARIMA elements, remove them
                    if(any(substr(names(persistence),1,3)=="psi")){
                        persistence <- persistence[substr(names(persistence),1,3)!="psi"];
                    }
                    j <- 0;
                    if(etsModel){
                        j <- j+1;
                        persistenceLevel <- as.vector(persistence)[1];
                        names(persistenceLevel) <- "alpha";
                        if(modelIsTrendy && length(persistence)>j){
                            j <- j+1;
                            persistenceTrend <- as.vector(persistence)[j];
                            names(persistenceTrend) <- "beta";
                        }
                        if(Stype!="N" && length(persistence)>j){
                            j <- j+1;
                            persistenceSeasonal <- as.vector(persistence)[j];
                            names(persistenceSeasonal) <- paste0("gamma",c(1:length(persistenceSeasonal)));
                        }
                    }
                    if(xregModel && length(persistence)>j){
                        if(j>0){
                            persistenceXreg <- as.vector(persistence)[-c(1:j)];
                        }
                        else{
                            persistenceXreg <- as.vector(persistence);
                        }
                        # If there are names, make sure that only deltas are used
                        if(!is.null(names(persistenceXreg))){
                            persistenceXreg <- persistenceXreg[substr(names(persistenceXreg),1,5)=="delta"];
                        }
                        else{
                            names(persistenceXreg) <- paste0("delta",c(1:length(persistenceXreg)));
                        }
                    }

                    persistenceEstimate[] <- persistenceLevelEstimate[] <- persistenceTrendEstimate[] <-
                        persistenceXregEstimate[] <- persistenceSeasonalEstimate[] <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(persistence);
                    # bounds <- "none";
                }
            }
            else{
                warning(paste0("Persistence is not a numeric vector!\n",
                               "Changing to estimation of persistence vector values."),call.=FALSE);
                persistence <- NULL;
                persistenceEstimate <- TRUE;
            }
        }
    }
    else{
        persistenceEstimate <- TRUE;
    }

   # Make sure that only important elements are estimated.
    if(!etsModel){
        persistenceLevelEstimate[] <- FALSE;
    }
    if(!etsModel || Ttype=="N"){
        persistenceTrendEstimate[] <- FALSE;
        persistenceTrend <- NULL;
    }
    if(!etsModel || Stype=="N"){
        persistenceSeasonalEstimate[] <- FALSE;
        persistenceSeasonal <- NULL;
    }
    if(!xregModel){
        persistenceXregEstimate[] <- FALSE;
        persistenceXreg <- NULL;
    }

    #### Phi ####
    if(etsModel){
        if(!is.null(phi)){
            if(all(modelDo!=c("estimate","use"))){
                warning(paste0("Predefined phi can only be used with preselected ETS model.\n",
                               "Changing to estimation."),call.=FALSE);
                phi <- 0.95;
                phiEstimate <- TRUE;
            }
            else{
                if(!is.numeric(phi) & (damped)){
                    warning(paste0("Provided value of phi is meaningless. phi will be estimated."),
                            call.=FALSE);
                    phi <- 0.95;
                    phiEstimate <- TRUE;
                }
                else if(is.numeric(phi) & (phi<0 | phi>2)){
                    warning(paste0("Damping parameter should lie in (0, 2) region. ",
                                   "Changing to the estimation of phi."),call.=FALSE);
                    phi[] <- 0.95;
                    phiEstimate <- TRUE;
                }
                else{
                    phiEstimate <- FALSE;
                    if(damped){
                        parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                    }
                }
            }
        }
        else{
            if(damped){
                phiEstimate <- TRUE;
                phi <- 0.95;
            }
            else{
                phiEstimate <- FALSE;
                phi <- 1;
            }
        }
    }
    else{
        phi <- 1;
        phiEstimate <- FALSE;
    }


    #### Lags for ARIMA ####
    if(arimaModel){
        lagsModelAll <- rbind(lagsModel,lagsModelARIMA);
        lagsModelMax <- max(lagsModel);
    }
    else{
        lagsModelAll <- lagsModel;
    }

    #### This needs to be amended after developing the first prototype! ####
    # If we have the zoo class and weird lags, amend profiles
    # Weird lags (dst and fractional): 24, 24*2 (half hour), 52, 24*4 (15 minutes), 7*24, 7*48, 365, 24*52, 24*365
    if(any(yClasses=="zoo") && any(lags %in% c(24, 48, 52, 96, 168, 336, 365, 1248, 8760))){
        # For hourly, half-hourly and quarter hour data, just amend the profiles for DST.
        # For daily, repeat profile of 28th on 29th February.
        # For weekly, repeat the last week, when we have 53 instead of 52.
    }

    #### Occurrence variable ####
    if(is.occurrence(occurrence)){
        oesModel <- occurrence;
        occurrence <- oesModel$occurrence;
        if(oesModel$occurrence=="provided"){
            occurrenceModelProvided <- FALSE;
        }
        else{
            occurrenceModelProvided <- TRUE;
        }
    }
    else{
        occurrenceModelProvided <- FALSE;
        oesModel <- NULL;
    }
    pFitted <- matrix(1, obsInSample, 1);
    pForecast <- rep(NA,h);

    if(is.numeric(occurrence)){
        # If it is data, then it should correspond to the in-sample.
        if(all(occurrence==1)){
            occurrence <- "none";
        }
        else{
            if(any(occurrence<0,occurrence>1)){
                warning(paste0("Parameter 'occurrence' should contain values between zero and one.\n",
                               "Converting to appropriate vector."),call.=FALSE);
                occurrence[] <- (occurrence!=0)*1;
            }

            # "provided", meaning that we have been provided the values of p
            pFitted[] <- occurrence[1:obsInSample];
            # Create forecasted values for occurrence
            if(h>0){
                if(length(occurrence)>obsInSample){
                    pForecast <- occurrence[-c(1:obsInSample)];
                }
                else{
                    pForecast <- rep(tail(occurrence,1),h);
                }
                if(length(pForecast)>h){
                    pForecast <- pForecast[1:h];
                }
                else if(length(pForecast)<h){
                    pForecast <- c(pForecast,rep(tail(pForecast,1),h-length(pForecast)));
                }
            }
            else{
                pForecast <- NA;
            }
            occurrence <- "provided";
            oesModel <- list(fitted=pFitted,forecast=pForecast,occurrence="provided");
        }
    }

    occurrence <- match.arg(occurrence[1],c("none","auto","fixed","general","odds-ratio",
                                            "inverse-odds-ratio","direct","provided"));

    otLogical <- yInSample!=0;

    # If the data is not occurrence, let's assume that the parameter was switched unintentionally.
    if(all(otLogical) & all(occurrence!=c("none","provided"))){
        occurrence <- "none";
        occurrenceModelProvided <- FALSE;
    }

    # If there were NAs and the occurrence was not specified, do something with it
    # In all the other cases, NAs will be sorted out by the model
    if(any(yNAValues) && (occurrence=="none")){
        otLogical <- (!yNAValues)[1:obsInSample];
        occurrence[] <- "provided";
        pFitted[] <- otLogical*1;
        pForecast[] <- 1;
        occurrenceModel <- FALSE;
        oesModel <- structure(list(y=matrix((otLogical)*1,ncol=1),fitted=pFitted,forecast=pForecast,
                                   occurrence="provided"),class="occurrence");
    }
    else{
        if(occurrence=="none"){
            occurrenceModel <- FALSE;
            otLogical <- rep(TRUE,obsInSample);
        }
        else if(occurrence=="provided"){
            occurrenceModel <- FALSE;
            oesModel$y <- matrix(otLogical*1,ncol=1);
        }
        else{
            occurrenceModel <- TRUE;
        }

        # Include NAs in the zeroes, so that we avoid weird cases
        if(any(yNAValues)){
            otLogical <- !(!otLogical | yNAValues[1:obsInSample]);
        }
    }

    if(any(yClasses=="ts")){
        ot <- ts(matrix(otLogical*1,ncol=1), start=yStart, frequency=yFrequency);
    }
    else{
        ot <- ts(matrix(otLogical*1,ncol=1), start=c(0,0), frequency=lagsModelMax);
    }
    obsNonzero <- sum(ot);
    obsZero <- obsInSample - obsNonzero;

    # Check if multiplicative models can be fitted
    allowMultiplicative <- !((any(yInSample<=0) && !occurrenceModel) || (occurrenceModel && any(yInSample<0)));

    if(etsModel){
        # Clean the pool of models if only additive are allowed
        if(!allowMultiplicative && !is.null(modelsPool)){
            modelsPoolMultiplicative <- ((substr(modelsPool,1,1)=="M") |
                                             substr(modelsPool,2,2)=="M" |
                                             substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="M");
            if(any(modelsPoolMultiplicative)){
                modelsPool <- modelsPool[!modelsPoolMultiplicative];

                if(!any(model==c("PPP","FFF"))){
                    warning("Only additive models are allowed for your data. Amending the pool.",
                            call.=FALSE);
                }
            }
        }
        if((any(model==c("PPP","FFF")) || any(unlist(strsplit(model,""))=="Z")) && !allowMultiplicative){
            model <- "XXX";
            Etype <- "A";
            modelsPool <- NULL;
            warning("Only additive models are allowed for your data. Changing the selection mechanism.",
                    call.=FALSE);
        }
        else if(any(model==c("PPP","FFF")) && allowMultiplicative){
            model <- "ZZZ";
        }
    }

    #### Initial values ####
    # Vectors for initials of different components
    initialLevel <- NULL;
    initialTrend <- NULL;
    initialSeasonal <- NULL;
    initialArima <- NULL;
    initialXreg <- NULL;
    # InitialEstimate vectors, defining what needs to be estimated
    # NOTE: that initial==c("optimal","backcasting") meanst initialEstimate==TRUE!
    initialEstimate <- initialLevelEstimate <- initialTrendEstimate <-
        initialArimaEstimate <- initialXregEstimate <- TRUE;
    # initials of seasonal is a vector, not a scalar, because we can have several lags
    initialSeasonalEstimate <- rep(TRUE,componentsNumberETSSeasonal);

    # This is an initialisation of the variable
    initialType <- "optimal"
    # initial type can be: "o" - optimal, "b" - backcasting, "p" - provided.
    if(any(is.character(initial))){
        initialType[] <- match.arg(initial, c("optimal","backcasting"));
    }
    else if(is.null(initial)){
        if(!silent){
            message("Initial value is not selected. Switching to optimal.");
        }
        initialType[] <- "optimal";
    }
    else if(!is.null(initial)){
        if(all(modelDo!=c("estimate","use"))){
            warning(paste0("Predefined initials vector can only be used with preselected ETS model.\n",
                           "Changing to estimation of initials."),call.=FALSE);
            initialType[] <- "optimal";
            initialEstimate[] <- initialLevelEstimate[] <- initialTrendEstimate[] <-
                initialSeasonalEstimate[] <- initialArimaEstimate[] <- initialXregEstimate[] <- TRUE;
        }
        else{
            # If the list is provided, then check what this is.
            # This should be: level, trend, seasonal[[1]], seasonal[[2]], ..., ARIMA, xreg
            if(is.list(initial)){
                # If this is a named list, then extract stuff using names
                if(!is.null(names(initial))){
                    if(!is.null(initial$level)){
                        initialLevel <- initial$level;
                    }
                    if(!is.null(initial$trend)){
                        initialTrend <- initial$trend;
                    }
                    if(!is.null(initial$seasonal)){
                        initialSeasonal <- initial$seasonal;
                    }
                    if(!is.null(initial$arima)){
                        initialArima <- initial$arima;
                    }
                    if(!is.null(initial$xreg)){
                        initialXreg <- initial$xreg;
                    }
                }
                else{
                    if(!is.null(initial[[1]])){
                        initialLevel <- initial[[1]];
                    }
                    if(!is.null(initial[[2]])){
                        initialTrend <- initial[[2]];
                    }
                    if(!is.null(initial[[3]])){
                        initialSeasonal <- initial[[3]];
                    }
                    if(!is.null(initial[[4]])){
                        initialArima <- initial[[4]];
                    }
                    if(!is.null(initial[[5]])){
                        initialXreg <- initial[[5]];
                    }
                }
            }
            else{
                if(!is.numeric(initial)){
                    warning(paste0("Initial vector is not numeric!\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialType[] <- "optimal";
                }
                else{
                    # If this is a vector, then it should contain values in the order:
                    # level, trend, seasonal1, seasonal2, ..., ARIMA, xreg
                    # if(length(initial)<(sum(lagsModelAll))){
                    #     warning(paste0("The vector of initials contains only values for several components. ",
                    #                    "We will use what we can."),call.=FALSE);
                    # }
                    # else{
                        j <- 0;
                        if(etsModel){
                            initialLevel <- initial[1];
                            initialLevelEstimate[] <- FALSE;
                            if(modelIsTrendy){
                                j <- 2;
                                # If there is something in the vector, use it
                                if(all(!is.na(initial[j]))){
                                    initialTrend <- initial[j];
                                    initialTrendEstimate[] <- FALSE;
                                }
                            }
                            if(Stype!="N"){
                                # If there is something in the vector, use it
                                if(length(initial[-c(1:j)])>0){
                                    initialSeasonal <- vector("list",componentsNumberETSSeasonal);
                                    m <- 0;
                                    for(i in 1:componentsNumberETSSeasonal){
                                        if(all(!is.na(initial[j+m+1:lagsModelSeasonal[i]]))){
                                            initialSeasonal[[i]] <- initial[j+m+1:lagsModelSeasonal[i]];
                                            m <- m + lagsModelSeasonal[i];
                                        }
                                        else{
                                            break;
                                        }
                                    }
                                    j <- j+m;
                                    initialSeasonalEstimate[] <- FALSE;
                                }
                            }
                        }
                        if(arimaModel){
                            # If there is something else left, this must be ARIMA
                            if(all(!is.na(initial[j+c(1:initialArimaNumber)]))){
                                initialArima <- initial[j+c(1:initialArimaNumber)];
                                j <- j+max(lagsModelARIMA);
                                initialArimaEstimate[] <- FALSE;
                            }
                        }
                        if(xregModel){
                            # Something else? xreg for sure!
                            if(length(initial[-c(1:j)])>0){
                                initialXreg <- initial[-c(1:j)];
                                initialXregEstimate[] <- FALSE
                            }
                        }
                        parametersNumber[2,1] <- parametersNumber[2,1] + j;
                    }
                # }
            }
        }
    }

    #### Check the provided initials and define initialEstimate variables ####
    if(etsModel){
        # Level
        if(!is.null(initialLevel)){
            if(length(initialLevel)>1){
                warning("Initial level contains more than one value! Using the first one.",
                        call.=FALSE);
                initialLevel <- initialLevel[1];
            }
            initialLevelEstimate[] <- FALSE;
            parametersNumber[2,1] <- parametersNumber[2,1] + 1;
        }
        # Trend
        if(!is.null(initialTrend)){
            if(length(initialTrend)>1){
                warning("Initial trend contains more than one value! Using the first one.",
                        call.=FALSE);
                initialTrend <- initialTrend[1];
            }
            initialTrendEstimate[] <- FALSE;
            parametersNumber[2,1] <- parametersNumber[2,1] + 1;
        }
        # Seasonal
        if(!is.null(initialSeasonal)){
            # The list means several seasonal lags
            if(is.list(initialSeasonal)){
                # Is the number of seasonal initials correct? If it is bigger, then remove redundant
                if(length(initialSeasonal)>componentsNumberETSSeasonal){
                    warning("Initial seasonals contained more elements than needed! Removing redundant ones.",
                            call.=FALSE);
                    initialSeasonal <- initialSeasonal[1:componentsNumberETSSeasonal];
                }
                # Is the number of initials in each season correct? Use the correct ones only
                if(any(!(sapply(initialSeasonal,length) %in% lagsModelSeasonal))){
                    warning(paste0("Some of initial seasonals have a wrong length, ",
                                   "not corresponding to the provided lags. We will estimate them."),
                            call.=FALSE);
                    initialSeasonalToUse <- sapply(initialSeasonal,length) %in% lagsModelSeasonal;
                    initialSeasonal <- initialSeasonal[initialSeasonalToUse];
                }
                initialSeasonalEstimate[] <- !(lagsModelSeasonal %in% sapply(initialSeasonal,length));
                # If there are some gaps in what to estimate, reform initialSeason to make sense in the future creator function
                if(!all(initialSeasonalEstimate) && !all(!initialSeasonalEstimate)){
                    initialSeasonalCorrect <- vector("list",componentsNumberETSSeasonal);
                    initialSeasonalCorrect[which(!initialSeasonalEstimate)] <- initialSeasonal;
                    initialSeasonal <- initialSeasonalCorrect;
                }
            }
            # The vector implies only one seasonal
            else{
                if(all(length(initialSeasonal)!=lagsModelSeasonal)){
                    warning(paste0("Wrong length of seasonal initial: ",length(initialSeasonal),
                                   "Instead of ",lagsModelSeasonal,". Switching to estimation."),
                            call.=FALSE)
                    initialSeasonalEstimate[] <- TRUE;
                }
                else{
                    initialSeasonalEstimate[] <- FALSE;
                }
                # Create a list from the vector for consistency purposes
                initialSeasonal <- list(initialSeasonal);
            }
            parametersNumber[2,1] <- parametersNumber[2,1] + length(unlist(initialSeasonal));
        }
    }
    else{
        initialLevel <- initialTrend <- initialSeasonal <- NULL;
        initialLevelEstimate <- initialTrendEstimate <- initialSeasonalEstimate <- FALSE

        # If ETS is switched off, set error to whatever, based on the used distribution
        Etype[] <- switch(distribution,
                          "dinvgauss"=,"dlnorm"=,"dllaplace"=,"dls"=,"dlgnorm"="M",
                          "A");
    }

    # ARIMA
    if(!is.null(initialArima)){
        if(length(initialArima)!=initialArimaNumber){
            warning(paste0("The length of ARIMA initials is ",length(initialArima),
                           " instead of ",initialArimaNumber,". Estimating initials instead!"),
                    call.=FALSE);
            initialArimaEstimate[] <- TRUE;
        }
        else{
            initialArimaEstimate[] <- FALSE;
            parametersNumber[2,1] <- parametersNumber[2,1] + length(initialArima);
        }
    }
    # xreg
    if(!is.null(initialXreg)){
        initialXregEstimate[] <- FALSE;
    }

    #### Check ARIMA parameters, if they are provided ####
    if(arimaModel){
        # Check the provided parameters for AR and MA
        if(!is.null(arma)){
            arEstimate <- arRequired;
            maEstimate <- maRequired;
            # If this is a proper list, extract parameters separately
            if(is.list(arma)){
                armaParameters <- vector("numeric",sum(sapply(arma,length)));
                j <- arIndex <- maIndex <- 0;
                # The named list (proper thing)
                if(!is.null(names(arma))){
                    # Check if the length of the provided parameters is correct
                    if(arRequired && !is.null(arma$ar) && length(arma$ar)!=sum(arOrders)){
                        warning(paste0("The number of provided AR parameters is ",length(arma$ar),
                                       "while we need ",sum(arOrders),". ",
                                       "Switching to estimation."),call.=FALSE);
                        arEstimate[] <- TRUE;
                    }
                    if(maRequired && !is.null(arma$ma) && length(arma$ma)!=sum(maOrders)){
                        warning(paste0("The number of provided MA parameters is ",length(arma$ma),
                                       "while we need ",sum(maOrders),". ",
                                       "Switching to estimation."),call.=FALSE);
                        maEstimate[] <- TRUE;
                    }
                    for(i in 1:length(lags)){
                        if(arRequired && !is.null(arma$ar) && arOrders[i]>0){
                            armaParameters[j+c(1:arOrders[i])] <- arma$ar[arIndex+c(1:arOrders[i])]
                            names(armaParameters)[j+c(1:arOrders[i])] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                            j[] <- j+arOrders[i];
                            arIndex[] <- arIndex+arOrders[i];
                            arEstimate[] <- FALSE;
                        }
                        if(maRequired && !is.null(arma$ma) && maOrders[i]>0){
                            armaParameters[j+c(1:maOrders[i])] <- arma$ma[maIndex+c(1:maOrders[i])]
                            names(armaParameters)[j+c(1:maOrders[i])] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                            j[] <- j+maOrders[i];
                            maIndex[] <- maIndex+maOrders[i];
                            maEstimate[] <- FALSE;
                        }
                    }
                }
                # Just a list. Assume that the first element is ar, and the second is ma
                else{
                    for(i in 1:length(lags)){
                        k <- 1;
                        if(arRequired && !is.null(arma[[1]]) && arOrders[i]>0){
                            armaParameters[j+c(1:arOrders[i])] <- arma[[1]][arIndex+c(1:arOrders[i])]
                            names(armaParameters)[j+c(1:arOrders[i])] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                            j[] <- j+arOrders[i];
                            arIndex[] <- arIndex+arOrders[i];
                            arEstimate[] <- FALSE;
                            k[] <- 2;
                        }
                        if(maRequired && !is.null(arma[[k]]) && maOrders[i]>0){
                            armaParameters[j+c(1:maOrders[i])] <- arma[[k]][maIndex+c(1:maOrders[i])]
                            names(armaParameters)[j+c(1:maOrders[i])] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                            j[] <- j+maOrders[i];
                            maIndex[] <- maIndex+maOrders[i];
                            maEstimate[] <- FALSE;
                        }
                    }
                    # Check if the length of the provided parameters is correct
                    if(length(armaParameters)!=sum(arOrders)+sum(maOrders)){
                        warning(paste0("The number of provided ARMA parameters is ",length(armaParameters),
                                       "while we need ",sum(arOrders)+sum(maOrders),". ",
                                       "Switching to estimation."),call.=FALSE);
                        maEstimate <- arEstimate[] <- TRUE;
                        armaParameters <- NULL;
                    }
                }
            }
            else if(is.vector(arma)){
                # Check the length of the vector
                if(length(arma)!=sum(arOrders)+sum(maOrders)){
                    warning(paste0("The number of provided ARMA parameters is ",length(arma),
                                   "while we need ",sum(arOrders)+sum(maOrders),". ",
                                   "Switching to estimation."),call.=FALSE);
                    maEstimate <- arEstimate[] <- TRUE;
                    armaParameters <- NULL;
                }
                else{
                    armaParameters <- arma;
                    j <- 0;
                    for(i in 1:length(lags)){
                        if(arOrders[i]>0){
                            names(armaParameters)[j+1:arOrders[i]] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                            j <- j + arOrders[i];
                        }
                        if(maOrders[i]>0){
                            names(armaParameters)[j+1:maOrders[i]] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                            j <- j + maOrders[i];
                        }
                    }
                    maEstimate <- arEstimate[] <- FALSE;
                }
            }
            parametersNumber[2,1] <- parametersNumber[2,1] + length(armaParameters);
        }
        else{
            arEstimate <- arRequired;
            maEstimate <- maRequired;
            armaParameters <- NULL;
        }
    }
    else{
        armaParameters <- NULL;
    }

    #### xreg preparation ####
    # Check the regressors
    if(!xregModel){
        regressors[] <- "use";
        formulaProvided <- NULL;
    }
    else{
        if(regressors=="select"){
            # If this has not happened by chance, then switch to optimisation
            if(!is.null(initialXreg) && (initialType=="optimal")){
                warning("Variables selection does not work with the provided initials for explantory variables. We will drop them.",
                        call.=FALSE);
                initialXreg <- NULL;
                initialXregEstimate <- TRUE;
            }
            if(!is.null(persistenceXreg) && any(persistenceXreg!=0)){
                warning(paste0("We cannot do variables selection with the provided smoothing parameters ",
                               "for explantory variables. We will estimate them instead."),
                        call.=FALSE);
                persistenceXreg <- NULL;
            }
        }
    }

    # Use alm() in order to fit the preliminary model for xreg
    if(xregModel){
        # List of initials. The first element is additive error, the second is the multiplicative one
        xregModelInitials <- vector("list",2);

        #### Initial xreg are not provided ####
        # If the initials are not provided, estimate them using ALM.
        if(initialXregEstimate){
            initialXregProvided <- FALSE;
            # The function returns an ALM model
            xregInitialiser <- function(Etype,distribution,formulaProvided,subset,responseName){
                # Fix the default distribution for ALM
                if(distribution=="default"){
                    distribution <- switch(Etype,
                                           "A"="dnorm",
                                           "M"="dlnorm");
                }
                else if(distribution=="dllaplace"){
                    distribution <- "dlaplace";
                    Etype <- "M";
                }
                else if(distribution=="dls"){
                    distribution <- "ds";
                    Etype <- "M";
                }
                else if(distribution=="dlgnorm"){
                    distribution <- "dgnorm";
                    Etype <- "M";
                }
                # If the formula is not provided, construct one
                if(is.null(formulaProvided)){
                    if(Etype=="M" && any(distribution==c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace"))){
                        formulaProvided <- as.formula(paste0("log(`",responseName,"`)~."));
                    }
                    else{
                        formulaProvided <- as.formula(paste0("`",responseName,"`~."));
                    }
                }
                else{
                     # If formula contains only one element, or seeral, but no logs, then change response formula
                    if((length(formulaProvided[[2]])==1 ||
                        (length(formulaProvided[[2]])>1 & !any(as.character(formulaProvided[[2]])=="log"))) &&
                       (Etype=="M" && any(distribution==c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace")))){
                        formulaProvided <- update(formulaProvided,log(.)~.);
                    }
                }
                return(do.call(alm,list(formula=formulaProvided,data=xregData,distribution=distribution,subset=which(subset))))
            }
            # Extract names and form a proper matrix for the regression
            if(!is.null(formulaProvided)){
                formulaProvided <- as.formula(formulaProvided);
                responseName <- all.vars(formulaProvided)[1];
            }

            # This allows to save the original data
            xreg <- xregData;
            obsXreg <- nrow(xregData);

            # Form subset in order to use in-sample only
            subset <- rep(FALSE, obsAll);
            subset[1:obsInSample] <- TRUE;
            # Exclude zeroes if this is an occurrence model
            if(occurrenceModel){
                subset[1:obsInSample][!otLogical] <- FALSE;
            }

            #### Pure regression ####
            #### If this is just a regression, use stepwise / ALM
            if((!etsModel && !arimaModel) && regressors!="adapt"){
                # Return the estimated model based on the provided xreg
                if(is.null(formulaProvided)){
                    formulaProvided <- as.formula(paste0("`",responseName,"`~."));
                }
                if(distribution=="default"){
                    distribution[] <- "dnorm";
                }

                # Redefine loss for ALM
                loss <- switch(loss,
                               "MSEh"=,"TMSE"=,"GTMSE"=,"MSCE"="MSE",
                               "MAEh"=,"TMAE"=,"GTMAE"=,"MACE"="MAE",
                               "HAMh"=,"THAM"=,"GTHAM"=,"CHAM"="HAM",
                               loss);
                lossOriginal <- loss;
                if(loss=="custom"){
                    loss <- lossFunction;
                }

                # Either use or select the model via greybox functions
                if(regressors=="use"){
                    warning("The specified model is just a regression. It is recommended to use alm() from greybox instead.",
                            call.=FALSE);
                    # Fisher Information
                    if(is.null(ellipsis$FI)){
                        FI <- FALSE;
                    }
                    else{
                        FI <- ellipsis$FI;
                    }
                    almModel <- do.call("alm", list(formula=formulaProvided, data=xregData,
                                                    distribution=distribution, loss=loss,
                                                    subset=which(subset),
                                                    occurrence=oesModel,FI=FI));
                    almModel$call$data <- as.name(yName);
                    return(almModel);
                }
                else if(regressors=="select"){
                    warning("The specified model is just a stepwise regression. ",
                            "It is advised to use stepwise() function from greybox instead.",
                            call.=FALSE);
                    if(lossOriginal!="likelihood"){
                        warning("Stepwise only works with loss='likelihood'. Switching to it.",
                                call.=FALSE);
                        loss <- "likelihood"
                    }
                    almModel <- stepwise(xregData, distribution=distribution, subset=which(subset), occurrence=oesModel);
                    almModel$call$data <- as.name(yName);
                    return(almModel);
                }
            }

            #### ETSX / ARIMAX ####
            almModel <- NULL;
            if(Etype!="Z"){
                almModel <- xregInitialiser(Etype,distribution,formulaProvided,subset,responseName);
                if(Etype=="A"){
                    # If this is just a regression, include intercept
                    if(!etsModel && !arimaModel){
                        xregModelInitials[[1]]$initialXreg <- almModel$coefficients;
                    }
                    else{
                        xregModelInitials[[1]]$initialXreg <- almModel$coefficients[-1];
                    }
                    if(is.null(formulaProvided)){
                        xregModelInitials[[1]]$formula <- formulaProvided <- formula(almModel);
                    }
                    xregModelInitials[[1]]$other <- almModel$other;
                }
                else{
                    # If this is just a regression, include intercept
                    if(!etsModel && !arimaModel){
                        xregModelInitials[[2]]$initialXreg <- almModel$coefficients;
                    }
                    else{
                        xregModelInitials[[2]]$initialXreg <- almModel$coefficients[-1];
                    }
                    if(is.null(formulaProvided)){
                        xregModelInitials[[2]]$formula <- formulaProvided <- formula(almModel);
                    }
                    xregModelInitials[[2]]$other <- almModel$other;
                }
            }
            # If we are selecting the appropriate error, produce two models: for "M" and for "A"
            else{
                # Additive model
                almModel <- xregInitialiser("A",distribution,formulaProvided,subset,responseName);
                # If this is just a regression, include intercept
                if(!etsModel && !arimaModel){
                    xregModelInitials[[1]]$initialXreg <- almModel$coefficients;
                }
                else{
                    xregModelInitials[[1]]$initialXreg <- almModel$coefficients[-1];
                }
                if(is.null(formulaProvided)){
                    xregModelInitials[[1]]$formula <- formula(almModel);
                }
                xregModelInitials[[1]]$other <- almModel$other;
                # Multiplicative model
                almModel[] <- xregInitialiser("M",distribution,formulaProvided,subset,responseName);
                # If this is just a regression, include intercept
                if(!etsModel && !arimaModel){
                    xregModelInitials[[2]]$initialXreg <- almModel$coefficients;
                }
                else{
                    xregModelInitials[[2]]$initialXreg <- almModel$coefficients[-1];
                }
                if(is.null(formulaProvided)){
                    xregModelInitials[[2]]$formula <- formula(almModel);
                }
                xregModelInitials[[2]]$other <- almModel$other;
            }

            # If this is just a regression, include intercept
            if(!etsModel && !arimaModel){
                xregNumber <- ncol(almModel$data);
                xregNames <- names(coef(almModel));
            }
            else{
                # Write down the number and names of parameters
                xregNumber <- ncol(almModel$data)-1;
                xregNames <- names(coef(almModel))[-1];
            }
            # The original number of obs in xreg
            obsXreg <- nrow(xreg);

            #### Data manipulations for further use ####
            # This formula is needed in order to expand the data
            if(is.null(formulaProvided)){
                formulaProvided <- as.formula(paste0("`",responseName,"`~",
                                                     paste0(colnames(xreg)[colnames(xreg)!=responseName],
                                                            collapse="+")));
            }

            # Robustify the names of variables
            colnames(xreg) <- make.names(colnames(xreg),unique=TRUE);
            # The names of the original variables
            xregNamesOriginal <- colnames(xregData)[-1];
            # Levels for the factors
            xregFactorsLevels <- lapply(xreg,levels);
            xregFactorsLevels[[responseName]] <- NULL;
            # Expand the variables. We cannot use alm, because it is based on obsInSample
            xregData <- model.frame(formulaProvided,data=as.data.frame(xreg));
            # Get the response variable, just in case it was transformed
            if(length(formulaProvided[[2]])>1){
                y <- xregData[,1];
                yInSample <- matrix(y[1:obsInSample],ncol=1);
                if(holdout){
                    yHoldout <- y[-c(1:obsInSample)];
                }
            }
            # Binary, flagging factors in the data
            xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];
            # Expanded stuff with all levels for factors
            if(any(xregFactors)){
                xregModelMatrix <- model.matrix(xregData,xregData,
                                                contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                     contrasts, contrasts=FALSE));
                xregNamesModified <- colnames(xregModelMatrix)[-1];
            }
            else{
                xregModelMatrix <- model.matrix(xregData,data=xregData);
                xregNamesModified <- xregNames;
            }
            xregData <- as.matrix(xregModelMatrix);
            # Remove intercept
            interceptIsPresent <- FALSE;
            if(any(colnames(xregData)=="(Intercept)")){
                interceptIsPresent[] <- TRUE;
                xregData <- xregData[,-1,drop=FALSE];
            }
            xregNumber <- ncol(xregData);

            # If there are more obs in xreg than the obsAll cut the thing
            if(obsXreg>=obsAll){
                xregData <- xregData[1:obsAll,,drop=FALSE]
            }
            # If there are less xreg observations than obsAll, use Naive
            else{
                newnRows <- obsAll-obsXreg;
                xregData <- rbind(xregData,matrix(rep(tail(xregData,1),each=newnRows),newnRows,xregNumber));
            }

            #### Fix parameters for factors ####
            # The indices of the original parameters
            xregParametersMissing <- setNames(vector("numeric",xregNumber),xregNamesModified);
            # # The indices of the original parameters
            xregParametersIncluded <- setNames(vector("numeric",xregNumber),xregNamesModified);
            # The vector, marking the same values of smoothing parameters
            if(interceptIsPresent){
                xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign")[-1],xregNamesModified);
            }
            else{
                xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign"),xregNamesModified);
            }
            # If there are factors not in the alm data, create additional initials
            if(any(!(xregNamesModified %in% xregNames))){
                xregAbsent <- !(xregNamesModified %in% xregNames);
                xregParametersNew <- setNames(rep(NA,xregNumber),xregNamesModified);
                # If the first initials are not NULL, fix parameters
                if(!is.null(xregModelInitials[[1]])){
                    xregParametersNew[!xregAbsent] <- xregModelInitials[[1]]$initialXreg;
                    # Go through new names and find, where they came from. Then get the missing parameters
                    for(i in which(xregAbsent)){
                        # Find the name of the original variable
                        # Use only the last value... hoping that the names like x and x1 are not used.
                        xregNameFound <- tail(names(sapply(xregNamesOriginal,grepl,xregNamesModified[i])),1);
                        # Get the indices of all k-1 levels
                        xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                               xregFactorsLevels[[xregNameFound]])]] <- i;
                        # Get the index of the absent one
                        xregParametersMissing[i] <- i;
                        # Fill in the absent one
                        xregParametersNew[i] <- -sum(xregParametersNew[xregNamesModified[xregParametersIncluded==i]],
                                                     na.rm=TRUE);
                    }
                    # Write down the new parameters
                    xregModelInitials[[1]]$initialXreg <- xregParametersNew;
                }
                # If the second initials are not NULL, fix parameters
                if(!is.null(xregModelInitials[[2]])){
                    xregParametersNew[!xregAbsent] <- xregModelInitials[[2]]$initialXreg;
                    # Go through new names and find, where they came from. Then get the missing parameters
                    for(i in which(xregAbsent)){
                        # Find the name of the original variable
                        # Use only the last value... hoping that the names like x and x1 are not used.
                        xregNameFound <- tail(names(sapply(xregNamesOriginal,grepl,xregNamesModified[i])),1);
                        # Get the indices of all k-1 levels
                        xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                               xregFactorsLevels[[xregNameFound]])]] <- i;
                        # Get the index of the absent one
                        xregParametersMissing[i] <- i;
                        # Fill in the absent one
                        xregParametersNew[i] <- -sum(xregParametersNew[xregNamesModified[xregParametersIncluded==i]],
                                                     na.rm=TRUE);
                    }
                    # Write down the new parameters
                    xregModelInitials[[2]]$initialXreg <- xregParametersNew;
                }
                xregNames <- xregNamesModified;
            }
            # The vector of parameters that should be estimated (numeric + original levels of factors)
            xregParametersEstimated <- xregParametersIncluded
            xregParametersEstimated[xregParametersEstimated!=0] <- 1;
            xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;

            # Remove xreg, just to preserve some memory
            rm(xreg);
        }
        #### Initial xreg are provided ####
        else{
            #### Pure regression ####
            #### If this is just a regression, then this must be a reuse of alm.
            if((!etsModel && !arimaModel) && regressors!="adapt"){
                # Return the estimated model based on the provided xreg
                if(is.null(formulaProvided)){
                    formulaProvided <- as.formula(paste0("`",responseName,"`~."));
                }
                if(distribution=="default"){
                    distribution[] <- "dnorm";
                }
                # Redefine loss for ALM
                loss <- switch(loss,
                               "MSEh"=,"TMSE"=,"GTMSE"=,"MSCE"="MSE",
                               "MAEh"=,"TMAE"=,"GTMAE"=,"MACE"="MAE",
                               "HAMh"=,"THAM"=,"GTHAM"=,"CHAM"="HAM",
                               loss);
                lossOriginal <- loss;
                if(loss=="custom"){
                    loss <- lossFunction;
                }

                # Form subset in order to use in-sample only
                subset <- rep(FALSE, obsAll);
                subset[1:obsInSample] <- TRUE;
                # Exclude zeroes if this is an occurrence model
                if(occurrenceModel){
                    subset[1:obsInSample][!otLogical] <- FALSE;
                }
                # Fisher Information
                if(is.null(ellipsis$FI)){
                    FI <- FALSE;
                }
                else{
                    FI <- ellipsis$FI;
                }
                if(is.null(ellipsis$B)){
                    B <- initialXreg;
                }
                else{
                    B <- ellipsis$B;
                }
                almModel <- do.call("alm", list(formula=formulaProvided, data=xregData,
                                                distribution=distribution, loss=loss, subset=which(subset),
                                                occurrence=occurrence,FI=FI,
                                                parameters=B,fast=TRUE));
                almModel$call$data <- as.name(yName);
                return(almModel);
            }
            #### InitialXreg is provided ####
            else{
                # Save the original data
                xreg <- xregData;
                initialXregProvided <- TRUE;
                # Robustify the names of variables
                colnames(xreg) <- make.names(colnames(xreg),unique=TRUE);

                # This formula is needed only to expand xreg
                if(is.null(formulaProvided)){
                    responseName <- colnames(xreg)[1]
                    formulaProvided <- as.formula(paste0("`",responseName,"`~",
                                                         paste0(colnames(xreg)[colnames(xreg)!=responseName],
                                                                collapse="+")));
                }
                # Extract names and form a proper matrix for the regression
                else{
                    responseName <- all.vars(formulaProvided)[1];
                }

                # Get the names of initials
                xregNames <- names(initialXreg);

                # The names of the original variables
                xregNamesOriginal <- colnames(xregData)[-1];
                # Levels for the factors
                xregFactorsLevels <- lapply(xreg[,-1,drop=FALSE],levels);
                # Expand the variables. We cannot use alm, because it is based on obsInSample
                xregData <- model.frame(formulaProvided,data=as.data.frame(xreg));
                # Get the response variable, just in case it was transformed
                if(length(formulaProvided[[2]])>1){
                    y <- xregData[,1];
                    yInSample <- matrix(y[1:obsInSample],ncol=1);
                    if(holdout){
                        yHoldout <- y[-c(1:obsInSample)];
                    }
                }

                # Binary, flagging factors in the data
                xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];
                # Expanded stuff with all levels for factors
                if(any(xregFactors)){
                    xregModelMatrix <- model.matrix(xregData,xregData,
                                                    contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                         contrasts, contrasts=FALSE));
                    xregNamesModified <- colnames(xregModelMatrix)[-1];
                }
                else{
                    xregModelMatrix <- model.matrix(xregData,data=xregData);
                    xregNamesModified <- xregNames;
                }
                xregData <- as.matrix(xregModelMatrix);

                # Remove intercept
                interceptIsPresent <- FALSE;
                if(any(colnames(xregData)=="(Intercept)")){
                    interceptIsPresent[] <- TRUE;
                    xregData <- xregData[,-1,drop=FALSE];
                }
                xregNumber <- ncol(xregData);
                obsXreg <- nrow(xregData);

                # If there are more obs in xreg than the obsAll cut the thing
                if(obsXreg>=obsAll){
                    xregData <- xregData[1:obsAll,,drop=FALSE]
                }
                # If there are less xreg observations than obsAll, use Naive
                else{
                    newnRows <- obsAll-obsXreg;
                    xregData <- rbind(xregData,matrix(rep(tail(xregData,1),each=newnRows),newnRows,xregNumber));
                }

                # This variable is needed in order to do model.matrix only, when required.
                xregDataIsDataFrame <- is.data.frame(xregData);
                if(xregDataIsDataFrame){
                    # Expand xregData if it is a data frame
                    xregData <- model.frame(formulaProvided,data=xregData);
                }
                xregNumber[] <- ncol(xregData);

                #### Fix parameters for factors ####
                # The indices of the original parameters
                xregParametersMissing <- setNames(vector("numeric",xregNumber),xregNamesModified);
                # # The indices of the original parameters
                xregParametersIncluded <- setNames(vector("numeric",xregNumber),xregNamesModified);
                # The vector, marking the same values of smoothing parameters
                if(interceptIsPresent){
                    xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign")[-1],xregNamesModified);
                }
                else{
                    xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign"),xregNamesModified);
                }

                # If there are factors and the number of initials is not the same as the number of parameters needed
                # This stuff assumes that the provided xreg parameters are named.
                if(any(xregFactors)){
                    # Expand the data again in order to find the missing elements
                    xregNames <- colnames(as.matrix(model.matrix(formulaProvided,
                                                                 model.frame(formulaProvided,data=as.data.frame(xreg)))));
                    if(any(xregNames=="(Intercept)")){
                        xregNames <- xregNames[xregNames!="(Intercept)"];
                    }
                    if(length(xregNames)!=length(xregNamesModified)){
                        xregAbsent <- !(xregNamesModified %in% xregNames);
                        # Create the new vector of parameters
                        xregParametersNew <- setNames(rep(NA,xregNumber),xregNamesModified);
                        if(!is.null(names(initialXreg))){
                            xregParametersNew[xregNames] <- initialXreg[xregNames];
                        }
                        else{
                            # Insert values one by one
                            j <- 1;
                            for(i in 1:length(xregNamesModified)){
                                if(!xregAbsent[i]){
                                    xregParametersNew[i] <- initialXreg[j];
                                    j <- j+1;
                                }
                            }
                        }
                        # Go through new names and find, where they came from. Then get the missing parameters
                        for(i in which(xregAbsent)){
                            # Find the name of the original variable
                            # Use only the last value... hoping that the names like x and x1 are not used.
                            xregNameFound <- tail(names(unlist(sapply(xregNamesOriginal,grep,xregNamesModified[i]))),1);
                            # Get the indices of all k-1 levels
                            xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                                   xregFactorsLevels[[xregNameFound]])]] <- i;
                            # Get the index of the absent one
                            xregParametersMissing[i] <- i;
                            # Fill in the absent one
                            xregParametersNew[i] <- -sum(xregParametersNew[xregParametersIncluded==i],
                                                         na.rm=TRUE);
                        }
                        # Write down the new parameters
                        xregModelInitials[[1]]$initialXreg <- xregParametersNew;

                        # Write down the new parameters to the second part
                        if(any(Etype==c("M","Z"))){
                            xregModelInitials[[2]]$initialXreg <- xregParametersNew;
                        }
                        xregNames <- xregNamesModified;
                    }
                }
                else{
                    xregModelInitials[[1]]$initialXreg <- initialXreg;
                    if(any(Etype==c("M","Z"))){
                        xregModelInitials[[2]]$initialXreg <- initialXreg;
                    }
                }
                # The vector of parameters that should be estimated (numeric + original levels of factors)
                xregParametersEstimated <- xregParametersIncluded
                xregParametersEstimated[xregParametersEstimated!=0] <- 1;
                xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;

                parametersNumber[2,2] <- parametersNumber[2,2] + xregNumber;
            }

            # Remove xreg, just to preserve some memory
            rm(xreg);
        }

        #### persistence for xreg ####
        # Process the persistence for xreg
        if(!is.null(persistenceXreg)){
            if(length(persistenceXreg)!=xregNumber && length(persistenceXreg)!=1){
                warning("The length of the provided persistence for the xreg variables is wrong. Reverting to the estimation.",
                        call.=FALSE);
                persistenceXreg <- rep(0.5,xregNumber);
                persistenceXregProvided <- FALSE;
                persistenceXregEstimate <- TRUE;
            }
            else if(length(persistenceXreg)==1){
                persistenceXreg <- rep(persistenceXreg,xregNumber);
                persistenceXregProvided <- TRUE;
                persistenceXregEstimate <- FALSE;
            }
            else{
                persistenceXregProvided <- TRUE;
                persistenceXregEstimate <- FALSE;
            }
        }
        else{
            # The persistenceXregEstimate is provided
            if(regressors=="adapt" && !persistenceXregEstimate){
                persistenceXregProvided <- TRUE;
                persistenceXregEstimate <- FALSE;
            }
            else if(regressors=="adapt" && persistenceXregEstimate){
                persistenceXreg <- rep(0.05,xregNumber);
                persistenceXregProvided <- FALSE;
                persistenceXregEstimate <- TRUE;
            }
            else{
                persistenceXreg <- rep(0,xregNumber);
                persistenceXregProvided <- FALSE;
                persistenceXregEstimate <- FALSE;
            }
        }

        # If this is just a regression, include intercept
        if(!etsModel && !arimaModel){
            lagsModelAll <- matrix(rep(1,xregNumber),ncol=1);
        }
        else{
            lagsModelAll <- matrix(c(lagsModelAll,rep(1,xregNumber)),ncol=1);
        }
        # If there's only one explanatory variable, then there's nothing to select
        if(xregNumber==1){
            regressors[] <- "use";
        }

        # Fix the names of variables
        colnames(xregData) <- make.names(colnames(xregData), unique=TRUE);
        xregNames[] <- make.names(xregNames, unique=TRUE);

        # If there are no variables after all of that, then xreg doesn't exist
        if(xregNumber==0){
            warning("It looks like there are no suitable explanatory variables. Check the xreg! We dropped them out.",
                    call.=FALSE);
            xregModel[] <- FALSE;
            xregData <- NULL;
        }
    }
    else{
        initialXregProvided <- FALSE;
        initialXregEstimate <- FALSE;
        persistenceXregProvided <- FALSE;
        persistenceXregEstimate <- FALSE;
        xregModelInitials <- NULL;
        xregData <- NULL;
        xregNumber <- 0;
        xregNames <- NULL;
        if(is.null(formulaProvided)){
            # if(Etype=="M" && any(distribution==c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace"))){
            #     formulaProvided <- as.formula(paste0("log(`",responseName,"`)~."));
            # }
            # else{
                formulaProvided <- as.formula(paste0("`",responseName,"`~."));
            # }
        }
        xregParametersMissing <- 0;
        xregParametersIncluded <- 0;
        xregParametersEstimated <- 0;
        xregParametersPersistence <- 0;
    }

    # Redefine persitenceEstimate value
    persistenceEstimate[] <- any(c(persistenceLevelEstimate,persistenceTrendEstimate,
                                   persistenceSeasonalEstimate,persistenceXregEstimate));

    #### Conclusions about the initials ####
    # Make sure that only important elements are estimated.
    if(!modelIsTrendy){
        initialTrendEstimate <- FALSE;
        initialTrend <- NULL;
    }
    if(!modelIsSeasonal){
        initialSeasonalEstimate <- FALSE;
        initialSeasonal <- NULL;
    }
    if(!arimaModel){
        initialArimaEstimate <- FALSE;
        initialArima <- NULL;
    }
    if(!xregModel){
        initialXregEstimate <- FALSE;
        initialXreg <- NULL;
    }

    # If we don't need to estimate anything, flag initialEstimate
    if(!any(c(etsModel && initialLevelEstimate,
              (etsModel && modelIsTrendy && initialTrendEstimate),
              (etsModel & modelIsSeasonal & initialSeasonalEstimate),
              (arimaModel && initialArimaEstimate),
              (xregModel && initialXregEstimate)))){
        initialEstimate[] <- FALSE;
    }
    else{
        initialEstimate[] <- TRUE;
    }

    # If at least something is provided, flag it as "provided"
    if((etsModel && !initialLevelEstimate) ||
       (etsModel && modelIsTrendy && !initialTrendEstimate) ||
       any(etsModel & modelIsSeasonal & !initialSeasonalEstimate) ||
       (arimaModel && !initialArimaEstimate) ||
       (xregModel && !initialXregEstimate)){
        initialType[] <- "provided";
    }

    # Observations in the states matrix
    # Define the number of cols that should be in the matvt
    obsStates <- obsInSample + lagsModelMax;

    if(any(yInSample<=0) && any(distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm")) && !occurrenceModel){
        warning(paste0("You have non-positive values in the data. ",
                       "The distribution ",distribution," does not support that. ",
                       "This might lead to problems in the estimation."),
                call.=FALSE);
    }

    # Update the number of parameters
    if(occurrenceModelProvided){
        parametersNumber[2,3] <- nparam(oesModel);
        pForecast <- c(forecast(oesModel, h=h)$mean);
    }


    #### Information Criteria ####
    ic <- match.arg(ic,c("AICc","AIC","BIC","BICc"));
    ICFunction <- switch(ic,
                         "AIC"=AIC,
                         "AICc"=AICc,
                         "BIC"=BIC,
                         "BICc"=BICc);

    #### Bounds for the smoothing parameters ####
    bounds <- match.arg(bounds,c("usual","admissible","none"));


    #### Checks for the potential number of degrees of freedom ####
    # This is needed in order to make the function work on small samples
    # scale parameter, smoothing parameters and phi
    nParamMax <- (1 +
                      # ETS model
                      etsModel*(persistenceLevelEstimate + modelIsTrendy*persistenceTrendEstimate +
                                    modelIsSeasonal*sum(persistenceSeasonalEstimate) +
                                    phiEstimate + (initialType=="optimal") *
                                    (initialLevelEstimate + initialTrendEstimate + sum(initialSeasonalEstimate*lagsModelSeasonal))) +
                      # ARIMA components: initials + parameters
                      arimaModel*((initialType=="optimal")*initialArimaNumber +
                                      arRequired*arEstimate*sum(arOrders) + maRequired*maEstimate*sum(maOrders)) +
                      # Xreg initials and smoothing parameters
                      xregModel*(xregNumber*(initialXregEstimate+persistenceXregEstimate)));

    # If the sample is smaller than the number of parameters
    if(obsNonzero <= nParamMax){
        # If there is both ETS and ARIMA, remove ARIMA
        if(etsModel && arimaModel){
            warning("We don't have enough observations to fit ETS with ARIMA terms. We will construct the simple ETS.",
                    call.=FALSE);
            lagsModelAll <- lagsModelAll[-c(componentsNumberETS+c(1:componentsNumberARIMA)),,drop=FALSE];
            arRequired <- iRequired <- maRequired <- arimaModel <- FALSE;
            arOrders <- iOrders <- maOrders <- NULL;
            nonZeroARI <- nonZeroMA <- lagsModelARIMA <- NULL;
            componentsNamesARIMA <- NULL;
            initialArimaNumber <- componentsNumberARIMA <- 0;
            lagsModelMax <- max(lagsModelAll);
        }
        else if(arimaModel && !etsModel){
            # If the backacasting helps, switch to it.
            if(initialType=="optimal" && (obsNonzero > (nParamMax - (initialType=="optimal")*initialArimaNumber))){
                warning(paste0("The number of parameter to estimate is ",nParamMax,
                            ", while the number of observations is ",obsNonzero,
                            ". Switching initial to 'backcasting' to save some degrees of freedom."), call.=FALSE);
                initialType <- "backcasting";
            }
            else{
                warning(paste0("The number of parameter to estimate is ",nParamMax,
                               ", while the number of observations is ",obsNonzero,
                               ". Switching to ARIMA(0,1,1) model."), call.=FALSE);
                # Add ARIMA(0,1,1) lags
                lagsModelAll <- matrix(1,1,1);
                arRequired <- FALSE;
                iRequired <- maRequired <- arimaModel <- TRUE;
                arOrders <- 0;
                iOrders <- maOrders <- 1;
                nonZeroARI <- nonZeroMA <- matrix(c(2,1),1,2);
                lagsModelARIMA <- matrix(1,1,1);
                componentsNamesARIMA <- 1;
                initialArimaNumber <- componentsNumberARIMA <- 1;
                lagsModelMax <- max(lagsModelAll);
            }
        }
    }

    # Recalculate the number of parameters
    nParamMax[] <- (1 +
                        # ETS model
                        etsModel*(persistenceLevelEstimate + modelIsTrendy*persistenceTrendEstimate +
                                      modelIsSeasonal*sum(persistenceSeasonalEstimate) +
                                      phiEstimate + (initialType=="optimal") *
                                      (initialLevelEstimate + initialTrendEstimate + sum(initialSeasonalEstimate*lagsModelSeasonal))) +
                        # ARIMA components: initials + parameters
                        arimaModel*((initialType=="optimal")*initialArimaNumber +
                                        arRequired*arEstimate*sum(arOrders) + maRequired*maEstimate*sum(maOrders)) +
                        # Xreg initials and smoothing parameters
                        xregModel*(xregNumber*(initialXregEstimate+persistenceXregEstimate)));

    # If the sample is still smaller than the number of parameters (even after removing ARIMA)
    if(etsModel){
        if(obsNonzero <= nParamMax){
            nParamExo <- xregNumber*(initialXregEstimate+persistenceXregEstimate);
            if(!silent){
                message(paste0("Number of non-zero observations is ",obsNonzero,
                               ", while the maximum number of parameters to estimate is ", nParamMax,".\n",
                               "Updating pool of models."));
            }

            # If the number of observations is still enough for the model selection and the pool is not specified
            if(obsNonzero > (3 + nParamExo) && is.null(modelsPool) && any(modelDo==c("select","combine"))){
                # We have enough observations for local level model
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
                if((obsNonzero > (2*lagsModelMax)) && lagsModelMax!=1){
                    modelsPool <- c(modelsPool,"ANA");
                    if(allowMultiplicative){
                        modelsPool <- c(modelsPool,"ANM","MNA","MNM");
                    }
                }
                # We have enough observations for seasonal model with trend
                if((obsNonzero > (6 + lagsModelMax + nParamExo)) &&
                   (obsNonzero > 2*lagsModelMax) && lagsModelMax!=1){
                    modelsPool <- c(modelsPool,"AAA");
                    if(allowMultiplicative){
                        modelsPool <- c(modelsPool,"AAM","AMA","AMM","MAA","MAM","MMA","MMM");
                    }
                }

                warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what we can...",
                        call.=FALSE);
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
                    modelDo <- "select"
                    model <- "ZZZ";
                }
            }
            # If the pool is provided (so, select / combine), amend it
            else if(obsNonzero > (3 + nParamExo) && !is.null(modelsPool)){
                # We don't have enough observations for seasonal models with damped trend
                if((obsNonzero <= (6 + lagsModelMax + 1 + nParamExo))){
                    modelsPool <- modelsPool[!(nchar(modelsPool)==4 &
                                                   substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="A")];
                    modelsPool <- modelsPool[!(nchar(modelsPool)==4 &
                                                   substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="M")];
                }
                # We don't have enough observations for seasonal models with trend
                if((obsNonzero <= (5 + lagsModelMax + 1 + nParamExo))){
                    modelsPool <- modelsPool[!(substr(modelsPool,2,2)!="N" &
                                                   substr(modelsPool,nchar(modelsPool),nchar(modelsPool))!="N")];
                }
                # We don't have enough observations for seasonal models
                if(obsNonzero <= 2*lagsModelMax){
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

                modelsPool <- unique(modelsPool);
                warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what we can...",
                        call.=FALSE);
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
                    modelDo <- "select"
                    model <- "ZZZ";
                }
            }
            # If the model needs to be estimated / used, not selected
            else if(obsNonzero > (3 + nParamExo) && any(modelDo==c("estimate","use"))){
                # We don't have enough observations for seasonal models with damped trend
                if((obsNonzero <= (6 + lagsModelMax + 1 + nParamExo))){
                    model <- model[!(nchar(model)==4 &
                                         substr(model,nchar(model),nchar(model))=="A")];
                    model <- model[!(nchar(model)==4 &
                                         substr(model,nchar(model),nchar(model))=="M")];
                }
                # We don't have enough observations for seasonal models with trend
                if((obsNonzero <= (5 + lagsModelMax + 1 + nParamExo))){
                    model <- model[!(substr(model,2,2)!="N" &
                                         substr(model,nchar(model),nchar(model))!="N")];
                }
                # We don't have enough observations for seasonal models
                if(obsNonzero <= 2*lagsModelMax){
                    model <- model[substr(model,nchar(model),nchar(model))=="N"];
                }
                # We don't have enough observations for damped trend
                if(obsNonzero <= (6 + nParamExo)){
                    model <- model[nchar(model)!=4];
                }
                # We don't have enough observations for any trend
                if(obsNonzero <= (5 + nParamExo)){
                    model <- model[substr(model,2,2)=="N"];
                }
            }
            # Extreme cases of small samples
            else if(obsNonzero==4){
                if(any(Etype==c("A","M"))){
                    modelDo <- "estimate";
                    Ttype <- "N";
                    Stype <- "N";
                }
                else{
                    modelsPool <- c("ANN");
                    if(allowMultiplicative){
                        modelsPool <- c(modelsPool,"MNN");
                    }
                    modelDo <- "select";
                    model <- "ZZZ";
                    Etype <- "Z";
                    Ttype <- "N";
                    Stype <- "N";
                    warning("You have a very small sample. The only available model is level model.",
                            call.=FALSE);
                }
                phiEstimate <- FALSE;
            }
            # Even smaller sample
            else if(obsNonzero==3){
                if(any(Etype==c("A","M"))){
                    modelDo <- "estimate";
                    Ttype <- "N";
                    Stype <- "N";
                    model <- paste0(Etype,"NN");
                }
                else{
                    modelsPool <- c("ANN");
                    if(allowMultiplicative){
                        modelsPool <- c(modelsPool,"MNN");
                    }
                    modelDo <- "select";
                    model <- "ZNN";
                    Etype <- "Z";
                    Ttype <- "N";
                    Stype <- "N";
                }
                persistenceLevel <- 0;
                persistenceEstimate <- persistenceLevelEstimate <- FALSE;
                warning("We did not have enough of non-zero observations, so persistence value was set to zero.",
                        call.=FALSE);
                phiEstimate <- FALSE;
            }
            # Can it be even smaller?
            else if(obsNonzero==2){
                modelsPool <- NULL;
                persistenceLevel <- 0;
                persistenceEstimate <- persistenceLevelEstimate <- FALSE;
                initialLevel <- mean(yInSample);
                initialType <- "provided";
                initialEstimate <- initialLevelEstimate <- FALSE;
                warning("We did not have enough of non-zero observations, so persistence value was set to zero and initial was preset.",
                        call.=FALSE);
                modelDo <- "use";
                model <- "ANN";
                Etype <- "A";
                Ttype <- "N";
                Stype <- "N";
                phiEstimate <- FALSE;
                parametersNumber[1,1] <- 0;
                parametersNumber[2,1] <- 2;
            }
            # And how about now?!
            else if(obsNonzero==1){
                modelsPool <- NULL;
                persistenceLevel <- 0;
                persistenceEstimate <- persistenceLevelEstimate <- FALSE;
                initialLevel <- yInSample[yInSample!=0];
                initialType <- "provided";
                initialEstimate <- initialLevelEstimate <- FALSE;
                warning("We did not have enough of non-zero observations, so we used Naive.",call.=FALSE);
                modelDo <- "nothing"
                model <- "ANN";
                Etype <- "A";
                Ttype <- "N";
                Stype <- "N";
                phiEstimate <- FALSE;
                parametersNumber[1,1] <- 0;
                parametersNumber[2,1] <- 2;
            }
            # Only zeroes in the data...
            else if(obsNonzero==0 && obsInSample>1){
                modelsPool <- NULL;
                persistenceLevel <- 0;
                persistenceEstimate <- persistenceLevelEstimate <- FALSE;
                initialLevel <- 0;
                initialType <- "provided";
                initialEstimate <- initialLevelEstimate <- FALSE;
                occurrenceModelProvided <- occurrenceModel <- FALSE;
                occurrence <- "none";
                warning("You have a sample with zeroes only. Your forecast will be zero.",call.=FALSE);
                modelDo <- "nothing"
                model <- "ANN";
                Etype <- "A";
                Ttype <- "N";
                Stype <- "N";
                phiEstimate <- FALSE;
                parametersNumber[1,1] <- 0;
                parametersNumber[2,1] <- 2;
            }
            # If you don't have observations, then fuck off!
            else{
                stop("Not enough observations... Even for fitting of ETS('ANN')!",call.=FALSE);
            }
        }
    }
    # Reset the maximum lag. This is in order to take potential changes into account
    lagsModelMax[] <- max(lagsModelAll);

    #### Process ellipsis ####
    # Parameters for the optimiser
    if(is.null(ellipsis$maxeval)){
        maxeval <- NULL;
        # Make a warning if this is a big computational task
        if(any(lags>24) && arimaModel && initialType=="optimal"){
            warning(paste0("The estimation of ARIMA model with initial='optimal' on high frequency data might ",
                           "take more time to converge to the optimum. Consider either setting maxeval parameter ",
                           "to a higher value (e.g. maxeval=10000, which will take ~25 times more time than this) ",
                           "or using initial='backcasting'."),
                    call.=FALSE);
        }
    }
    else{
        maxeval <- ellipsis$maxeval;
    }
    if(is.null(ellipsis$maxtime)){
        maxtime <- -1;
    }
    else{
        maxtime <- ellipsis$maxtime;
    }
    if(is.null(ellipsis$xtol_rel)){
        xtol_rel <- 1E-6;
    }
    else{
        xtol_rel <- ellipsis$xtol_rel;
    }
    if(is.null(ellipsis$xtol_abs)){
        xtol_abs <- 1E-8;
    }
    else{
        xtol_abs <- ellipsis$xtol_abs;
    }
    if(is.null(ellipsis$ftol_rel)){
        ftol_rel <- 1E-8;
    }
    else{
        ftol_rel <- ellipsis$ftol_rel;
    }
    if(is.null(ellipsis$ftol_abs)){
        ftol_abs <- 0;
    }
    else{
        ftol_abs <- ellipsis$ftol_abs;
    }
    if(is.null(ellipsis$algorithm)){
        algorithm <- "NLOPT_LN_SBPLX";
    }
    else{
        algorithm <- ellipsis$algorithm;
    }
    if(is.null(ellipsis$print_level)){
        print_level <- 0;
    }
    else{
        print_level <- ellipsis$print_level;
    }
    # The following three arguments are used for the function itself, not the options
    if(is.null(ellipsis$lb)){
        lb <- NULL;
    }
    else{
        lb <- ellipsis$lb;
    }
    if(is.null(ellipsis$ub)){
        ub <- NULL;
    }
    else{
        ub <- ellipsis$ub;
    }
    if(is.null(ellipsis$B)){
        B <- NULL;
    }
    else{
        B <- ellipsis$B;
    }
    # Initialise parameters
    lambda <- other <- NULL;
    otherParameterEstimate <- FALSE
    # lambda for LASSO
    if(any(loss==c("LASSO","RIDGE"))){
        if(is.null(ellipsis$lambda)){
            warning(paste0("You have not provided lambda parameter. We will set it to zero."), call.=FALSE);
            lambda <- 0;
        }
        else{
            lambda <- ellipsis$lambda;
        }
    }
    # Parameters for distributions
    if(distribution=="dalaplace"){
        if(is.null(ellipsis$alpha)){
            other <- 0.5
            otherParameterEstimate <- TRUE;
        }
        else{
            other <- ellipsis$alpha;
            otherParameterEstimate <- FALSE;
        }
    }
    else if(any(distribution==c("dgnorm","dlgnorm"))){
        if(is.null(ellipsis$beta)){
            other <- 2
            otherParameterEstimate <- TRUE;
        }
        else{
            other <- ellipsis$beta;
            otherParameterEstimate <- FALSE;
        }
    }
    else if(distribution=="dt"){
        if(is.null(ellipsis$nu)){
            other <- 2
            otherParameterEstimate <- TRUE;
        }
        else{
            other <- ellipsis$nu;
            otherParameterEstimate <- FALSE;
        }
    }
    # Fisher Information
    if(is.null(ellipsis$FI)){
        FI <- FALSE;
    }
    else{
        FI <- ellipsis$FI;
    }
    # Step size for the hessian
    if(is.null(ellipsis$stepSize)){
        stepSize <- .Machine$double.eps^(1/4);
    }
    else{
        stepSize <- ellipsis$stepSize;
    }

    # See if the estimation of the model is not needed (do we estimate anything?)
    if(!any(c(etsModel & c(persistenceLevelEstimate, persistenceTrendEstimate,
                           persistenceSeasonalEstimate, phiEstimate,
                           (initialType!="backcasting") & c(initialLevelEstimate,
                                                            initialTrendEstimate,
                                                            initialSeasonalEstimate)),
              arimaModel & c(arEstimate, maEstimate, (initialType!="backcasting") & initialArimaEstimate),
              xregModel & c(persistenceXregEstimate, (initialType!="backcasting") & initialXregEstimate),
              otherParameterEstimate))){
        modelDo <- "use";
    }

    # If there is no model, return a constant level
    if(!etsModel && !arimaModel && !xregModel){
        etsModel <- TRUE;
        modelsPool <- NULL;
        persistenceLevel <- 0;
        persistenceEstimate <- persistenceLevelEstimate <- FALSE;
        initialLevel <- NULL;
        initialType <- "provided";
        initialEstimate <- initialLevelEstimate <- TRUE;
        model <- "NNN";
        if(is.null(B)){
            modelDo <- "estimate";
        }
        Etype <- switch(distribution,
                        "default"=,"dnorm"=,"dlaplace"=,"ds"=,"dgnorm"=,"dlogis"=,"dt"=,"dalaplace"="A",
                        "dlnorm"=,"dllaplace"=,"dls"=,"dlgnorm"=,"dinvgauss"="M");
        Ttype <- "N";
        Stype <- "N";
        phiEstimate <- FALSE;
        parametersNumber[1,1] <- 0;
        parametersNumber[2,1] <- 2;
    }

    # Switch usual bounds to the admissible if there's no ETS - this speeds up ARIMA
    if(!etsModel && bounds=="usual"){
        bounds[] <- "admissible";
    }

    # If we do model selection / combination with non-standard losses, complain
    if(any(modelDo==c("select","combine")) &&
       ((any(loss==c("MSE","MSEh","MSCE","GPL")) && all(distribution!=c("default","dnorm"))) ||
        (any(loss==c("MAE","MAEh","MACE")) && all(distribution!=c("default","dlaplace"))) ||
        (any(loss==c("HAM","HAMh","CHAM")) && all(distribution!=c("default","ds"))))){
        warning("The model selection only works in case of loss='likelihood'. We hope you know what you are doing.",
                call.=FALSE);
    }

    # Just in case, give names to yHoldout and yInSample
    colnames(yInSample) <- responseName;
    if(holdout){
        colnames(yHoldout) <- responseName;
    }

    #### Return the values to the previous environment ####
    ### Actuals
    assign("y",y,ParentEnvironment);
    assign("yHoldout",yHoldout,ParentEnvironment);
    assign("yInSample",yInSample,ParentEnvironment);
    assign("yNAValues",yNAValues,ParentEnvironment);

    ### Index and all related structure variables
    assign("yClasses",yClasses,ParentEnvironment);
    assign("yIndex",yIndex,ParentEnvironment);
    assign("yInSampleIndex",yInSampleIndex,ParentEnvironment);
    assign("yForecastIndex",yForecastIndex,ParentEnvironment);
    assign("yIndexAll",yIndexAll,ParentEnvironment);
    assign("yFrequency",yFrequency,ParentEnvironment);
    assign("yStart",yStart,ParentEnvironment);
    assign("yForecastStart",yForecastStart,ParentEnvironment);

    # The rename of the variable is needed for the hessian to work
    assign("horizon",h,ParentEnvironment);
    assign("h",h,ParentEnvironment);
    assign("holdout",holdout,ParentEnvironment);

    ### Number of observations and parameters
    assign("obsInSample",obsInSample,ParentEnvironment);
    assign("obsAll",obsAll,ParentEnvironment);
    assign("obsStates",obsStates,ParentEnvironment);
    assign("obsNonzero",obsNonzero,ParentEnvironment);
    assign("obsZero",obsZero,ParentEnvironment);
    assign("parametersNumber",parametersNumber,ParentEnvironment);

    ### Model type
    assign("etsModel",etsModel,ParentEnvironment);
    assign("model",model,ParentEnvironment);
    assign("Etype",Etype,ParentEnvironment);
    assign("Ttype",Ttype,ParentEnvironment);
    assign("Stype",Stype,ParentEnvironment);
    assign("modelIsTrendy",modelIsTrendy,ParentEnvironment);
    assign("modelIsSeasonal",modelIsSeasonal,ParentEnvironment);
    assign("modelsPool",modelsPool,ParentEnvironment);
    assign("damped",damped,ParentEnvironment);
    assign("modelDo",modelDo,ParentEnvironment);
    assign("allowMultiplicative",allowMultiplicative,ParentEnvironment);

    ### Numbers and names of components
    assign("componentsNumberETS",componentsNumberETS,ParentEnvironment);
    assign("componentsNamesETS",componentsNamesETS,ParentEnvironment);
    assign("componentsNumberETSNonSeasonal",componentsNumberETS-componentsNumberETSSeasonal,ParentEnvironment);
    assign("componentsNumberETSSeasonal",componentsNumberETSSeasonal,ParentEnvironment);
    # The number and names of ARIMA components
    assign("componentsNumberARIMA",componentsNumberARIMA,ParentEnvironment);
    assign("componentsNamesARIMA",componentsNamesARIMA,ParentEnvironment);

    ### Lags
    # This is the original vector of lags, modified for the level components.
    # This can be used in ARIMA
    assign("lags",lags,ParentEnvironment);
    # This is the vector of lags of ETS components
    assign("lagsModel",lagsModel,ParentEnvironment);
    # This is the vector of seasonal lags
    assign("lagsModelSeasonal",lagsModelSeasonal,ParentEnvironment);
    # This is the vector of lags for ARIMA components (not lags of ARIMA)
    assign("lagsModelARIMA",lagsModelARIMA,ParentEnvironment);
    # This is the vector of all the lags of model (ETS + ARIMA + X)
    assign("lagsModelAll",lagsModelAll,ParentEnvironment);
    # This is the maximum lag
    assign("lagsModelMax",lagsModelMax,ParentEnvironment);

    ### Persistence
    assign("persistence",persistence,ParentEnvironment);
    assign("persistenceEstimate",persistenceEstimate,ParentEnvironment);
    assign("persistenceLevel",persistenceLevel,ParentEnvironment);
    assign("persistenceLevelEstimate",persistenceLevelEstimate,ParentEnvironment);
    assign("persistenceTrend",persistenceTrend,ParentEnvironment);
    assign("persistenceTrendEstimate",persistenceTrendEstimate,ParentEnvironment);
    assign("persistenceSeasonal",persistenceSeasonal,ParentEnvironment);
    assign("persistenceSeasonalEstimate",persistenceSeasonalEstimate,ParentEnvironment);
    assign("persistenceXreg",persistenceXreg,ParentEnvironment);
    assign("persistenceXregEstimate",persistenceXregEstimate,ParentEnvironment);
    assign("persistenceXregProvided",persistenceXregProvided,ParentEnvironment);

    ### phi
    assign("phi",phi,ParentEnvironment);
    assign("phiEstimate",phiEstimate,ParentEnvironment);

    ### Initials
    assign("initial",initial,ParentEnvironment);
    assign("initialType",initialType,ParentEnvironment);
    assign("initialEstimate",initialEstimate,ParentEnvironment);
    assign("initialLevel",initialLevel,ParentEnvironment);
    assign("initialLevelEstimate",initialLevelEstimate,ParentEnvironment);
    assign("initialTrend",initialTrend,ParentEnvironment);
    assign("initialTrendEstimate",initialTrendEstimate,ParentEnvironment);
    assign("initialSeasonal",initialSeasonal,ParentEnvironment);
    assign("initialSeasonalEstimate",initialSeasonalEstimate,ParentEnvironment);
    assign("initialArima",initialArima,ParentEnvironment);
    assign("initialArimaEstimate",initialArimaEstimate,ParentEnvironment);
    # Number of initials that the ARIMA has (either provided or to estimate)
    assign("initialArimaNumber",initialArimaNumber,ParentEnvironment);
    assign("initialXregEstimate",initialXregEstimate,ParentEnvironment);
    assign("initialXregProvided",initialXregProvided,ParentEnvironment);

    ### Occurrence model
    assign("oesModel",oesModel,ParentEnvironment);
    assign("occurrenceModel",occurrenceModel,ParentEnvironment);
    assign("occurrenceModelProvided",occurrenceModelProvided,ParentEnvironment);
    assign("occurrence",occurrence,ParentEnvironment);
    assign("pFitted",pFitted,ParentEnvironment);
    assign("pForecast",pForecast,ParentEnvironment);
    assign("ot",ot,ParentEnvironment);
    assign("otLogical",otLogical,ParentEnvironment);

    ### Distribution, loss, bounds and IC
    assign("distribution",distribution,ParentEnvironment);
    assign("loss",loss,ParentEnvironment);
    assign("lossFunction",lossFunction,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("ic",ic,ParentEnvironment);
    assign("ICFunction",ICFunction,ParentEnvironment);
    assign("bounds",bounds,ParentEnvironment);

    ### ARIMA components
    assign("arimaModel",arimaModel,ParentEnvironment);
    assign("arOrders",arOrders,ParentEnvironment);
    assign("iOrders",iOrders,ParentEnvironment);
    assign("maOrders",maOrders,ParentEnvironment);
    assign("arRequired",arRequired,ParentEnvironment);
    assign("iRequired",iRequired,ParentEnvironment);
    assign("maRequired",maRequired,ParentEnvironment);
    assign("arEstimate",arEstimate,ParentEnvironment);
    assign("maEstimate",maEstimate,ParentEnvironment);
    assign("armaParameters",armaParameters,ParentEnvironment);
    assign("nonZeroARI",nonZeroARI,ParentEnvironment);
    assign("nonZeroMA",nonZeroMA,ParentEnvironment);

    ### Explanatory variables
    assign("xregModel",xregModel,ParentEnvironment);
    assign("regressors",regressors,ParentEnvironment);
    assign("xregModelInitials",xregModelInitials,ParentEnvironment);
    assign("xregData",xregData,ParentEnvironment);
    assign("xregNumber",xregNumber,ParentEnvironment);
    assign("xregNames",xregNames,ParentEnvironment);
    assign("responseName",responseName,ParentEnvironment);
    assign("formula",formulaProvided,ParentEnvironment);
    assign("xregParametersMissing",xregParametersMissing,ParentEnvironment);
    assign("xregParametersIncluded",xregParametersIncluded,ParentEnvironment);
    assign("xregParametersEstimated",xregParametersEstimated,ParentEnvironment);
    assign("xregParametersPersistence",xregParametersPersistence,ParentEnvironment);

    ### Ellipsis thingies
    # Optimisation related
    assign("maxeval",maxeval,ParentEnvironment);
    assign("maxtime",maxtime,ParentEnvironment);
    assign("xtol_rel",xtol_rel,ParentEnvironment);
    assign("xtol_abs",xtol_abs,ParentEnvironment);
    assign("ftol_rel",ftol_rel,ParentEnvironment);
    assign("ftol_abs",ftol_abs,ParentEnvironment);
    assign("algorithm",algorithm,ParentEnvironment);
    assign("print_level",print_level,ParentEnvironment);
    assign("B",B,ParentEnvironment);
    assign("lb",lb,ParentEnvironment);
    assign("ub",ub,ParentEnvironment);
    # Parameters for distributions
    assign("other",other,ParentEnvironment);
    assign("otherParameterEstimate",otherParameterEstimate,ParentEnvironment);
    # LASSO / RIDGE
    assign("lambda",lambda,ParentEnvironment);
    # Fisher Information
    assign("FI",FI,ParentEnvironment);
    # Step size for the hessian
    assign("stepSize",stepSize,ParentEnvironment);
}
