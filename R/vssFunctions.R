utils::globalVariables(c("initialSeason","persistence","phi","otObs","iprobability",
                         "intermittent","intermittentModel","imodelProvided","seasonal"));

##### *Checker of input of vector functions* #####
vssInput <- function(smoothType=c("ves"),...){
    smoothType <- smoothType[1];

    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ##### silent #####
    silent <- silent[1];
    # Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output","debugging","n","a","g","l","o","d"))){
            warning(paste0("Sorry, I have no idea what 'silent=",silent,
                           "' means. Switching to 'none'."),call.=FALSE);
            silent <- "none";
        }
        silent <- substring(silent,1,1);
    }
    silentValue <- silent;

    if(silentValue==FALSE | silentValue=="n"){
        silentText <- FALSE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }
    else if(silentValue==TRUE | silentValue=="a"){
        silentText <- TRUE;
        silentGraph <- TRUE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="g"){
        silentText <- FALSE;
        silentGraph <- TRUE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="l"){
        silentText <- FALSE;
        silentGraph <- FALSE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="o"){
        silentText <- TRUE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }
    else if(silentValue=="d"){
        silentText <- TRUE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }

    #### Check horizon ####
    if(h<=0){
        warning(paste0("You have set forecast horizon equal to ",h,". We hope you know, what you are doing."),
                call.=FALSE);
        if(h<0){
            warning("And by the way, we can't do anything with negative horizon, so we will set it equal to zero.",
                    call.=FALSE);
            h <- 0;
        }
    }

    #### Check data ####
    if(any(is.vsmooth.sim(y))){
        y <- y$data;
        if(length(dim(y))==3){
            warning("Simulated data contains several samples. Selecting a random one.",call.=FALSE);
            y <- ts(y[,,runif(1,1,dim(y)[3])]);
        }
    }

    if(!is.data.frame(y)){
        if(!is.numeric(y)){
            stop("The provided data is not a numeric matrix! Can't construct any model!", call.=FALSE);
        }
    }

    if(is.null(dim(y))){
        stop("The provided data is not a matrix or a data.frame! If it is a vector, please use es() function instead.",
             call.=FALSE);
    }

    if(is.data.frame(y)){
        y <- as.matrix(y);
    }

    # Number of series in the matrix
    nSeries <- ncol(y);

    correlatedSeries <- cor(y)[upper.tri(cor(y))];
    if(any(correlatedSeries>0.999)){
        warning(paste0("Some of series are almost perfectly correlated. This might cause difficulties in the estimation. ",
                       "Please, try removing some of them if you encounter any problems."),
                call.=FALSE);
    }

    if(is.null(ncol(y))){
        stop("The provided data is not a matrix! Use es() function instead!", call.=FALSE);
    }
    if(ncol(y)==1){
        stop("The provided data contains only one column. Use es() function instead!", call.=FALSE);
    }
    # Check the data for NAs
    if(any(is.na(y))){
        if(!silentText){
            warning("Data contains NAs. These observations will be substituted by zeroes.", call.=FALSE);
        }
        y[is.na(y)] <- 0;
    }

    # Define obs, the number of observations of in-sample
    obsInSample <- nrow(y) - holdout*h;

    # Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- nrow(y) + (1 - holdout)*h;

    # If obsInSample is negative, this means that we can't do anything...
    if(obsInSample<=0){
        stop("Not enough observations in sample.", call.=FALSE);
    }
    # Define the actual values. Transpose the matrix!
    yInSample <- matrix(y[1:obsInSample,],nSeries,obsInSample,byrow=TRUE);
    dataFreq <- frequency(y);
    dataDeltat <- deltat(y);
    dataStart <- start(y);
    yForecastStart <- time(y)[obsInSample]+deltat(y);
    dataNames <- colnames(y);
    if(!is.null(dataNames)){
        dataNames <- gsub(" ", "_", dataNames, fixed = TRUE);
        dataNames <- gsub(":", "_", dataNames, fixed = TRUE);
        dataNames <- gsub("$", "_", dataNames, fixed = TRUE);
    }
    else{
        dataNames <- paste0("Series",c(1:nSeries));
    }

    # Number of parameters to estimate / provided
    parametersNumber <- matrix(0,2,4,
                               dimnames=list(c("Estimated","Provided"),
                                             c("nParamInternal","nParamXreg","nParamIntermittent","nParamAll")));

    ##### model for VES #####
    if(smoothType=="ves"){
        if(!is.character(model)){
            stop(paste0("Something strange is provided instead of character object in model: ",
                        paste0(model,collapse=",")),call.=FALSE);
        }

        # If chosen model is "AAdN" or anything like that, we are taking the appropriate values
        if(nchar(model)==4){
            Etype <- substring(model,1,1);
            Ttype <- substring(model,2,2);
            Stype <- substring(model,4,4);
            damped <- TRUE;
            if(substring(model,3,3)!="d"){
                warning(paste0("You have defined a strange model: ",model));
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
        else{
            warning(paste0("You have defined a strange model: ",model));
            sowhat(model);
            warning("Switching to 'ZZZ'");
            model <- "ZZZ";

            Etype <- "Z";
            Ttype <- "Z";
            Stype <- "Z";
            damped <- TRUE;
        }

        #### Check error type ####
        if(all(Etype!=c("A","M","L"))){
            warning(paste0("Wrong error type: ",Etype,". Should be 'A' or 'M'.\n",
                           "Changing to 'A'"),call.=FALSE);
            Etype <- "A";
        }

        #### Check trend type ####
        if(all(Ttype!=c("N","A","M"))){
            warning(paste0("Wrong trend type: ",Ttype,". Should be 'N', 'A' or 'M'.\n",
                           "Changing to 'A'"),call.=FALSE);
            Ttype <- "A";
        }

        #### Check seasonality type ####
        # Check if the data is ts-object
        if(!is.ts(y) & Stype!="N"){
            warning("The provided data is not ts object. Only non-seasonal models are available.");
            Stype <- "N";
            substr(model,nchar(model),nchar(model)) <- "N";
        }

        # Check if seasonality makes sense
        if(all(Stype!=c("N","A","M"))){
            warning(paste0("Wrong seasonality type: ",Stype,". Should be 'N', 'A' or 'M'.",
                           "Setting to 'A'."),call.=FALSE);
            if(dataFreq==1){
                Stype <- "N";
            }
            else{
                Stype <- "A";
            }
        }
        if(Stype!="N" & dataFreq==1){
            warning(paste0("Cannot build the seasonal model on data with frequency 1.\n",
                           "Switching to non-seasonal model: ETS(",substring(model,1,nchar(model)-1),"N)"));
            Stype <- "N";
        }

        if(Stype=="N"){
            initialSeason <- NULL;
            modelIsSeasonal <- FALSE;
        }
        else{
            modelIsSeasonal <- TRUE;
        }

        # if(any(c(Etype,Ttype,Stype)=="Z")){
        #     stop("Sorry we don't do model selection for VES yet.", call.=FALSE);
        # }

        lagsModelMax <- dataFreq * modelIsSeasonal + 1 * (!modelIsSeasonal);

        # Define the number of rows that should be in the matvt
        obsStates <- max(obsAll + lagsModelMax, obsInSample + 2*lagsModelMax);

        nComponentsNonSeasonal <- 1 + (Ttype!="N")*1;
        nComponentsAll <- nComponentsNonSeasonal + modelIsSeasonal*1;
    }

    ##### imodel #####
    if(is.viss(imodel)){
        intermittentModel <- imodel$model;
        intermittent <- imodel$intermittent;
        imodelProvided <- TRUE;
    }
    else{
        intermittentModel <- imodel;
        imodelProvided <- FALSE;
        imodel <- NULL;
    }

    ##### intermittent #####
    intermittent <- substring(intermittent[1],1,1);
    if(intermittent!="n"){
        ot <- (yInSample!=0)*1;
        # Matrix of non-zero observations for the loss function
        otObs <- diag(rowSums(ot));
        for(i in 1:nSeries){
            for(j in 1:nSeries){
                if(i==j){
                    next;
                }
                otObs[i,j] <- min(otObs[i,i],otObs[j,j]);
            }
        }
    }
    else{
        ot <- matrix(1,nrow=nrow(yInSample),ncol=ncol(yInSample));
        otObs <- matrix(obsInSample,nSeries,nSeries);
    }

    # If the data is not intermittent, let's assume that the parameter was switched unintentionally.
    if(all(ot==1) & intermittent!="n"){
        intermittent <- "n";
        imodelProvided <- FALSE;
    }

    iprobability <- substring(iprobability[1],1,1)

    # Check if multiplicative model can be applied
    if(any(c(Etype,Ttype,Stype)=="M")){
        if(all(yInSample>0)){
            if(any(c(Etype,Ttype,Stype)=="A")){
                warning("Mixed models are not available. Switching to pure multiplicative.",call.=FALSE);
            }
            yInSample <- log(yInSample);
            Etype <- "M";
            Ttype <- ifelse(Ttype=="A","M",Ttype);
            Stype <- ifelse(Stype=="A","M",Stype);
            modelIsMultiplicative <- TRUE;
        }
        else{
            if(intermittent=="n"){
                warning("Sorry, but we cannot construct multiplicative model on non-positive data. Changing to additive.",
                        call.=FALSE);
                Etype <- "A";
                Ttype <- ifelse(Ttype=="M","A",Ttype);
                Stype <- ifelse(Stype=="M","A",Stype);
                modelIsMultiplicative <- FALSE;
            }
            else{
                yInSample[ot==1] <- log(yInSample[ot==1]);
                Etype <- "M";
                Ttype <- ifelse(Ttype=="A","M",Ttype);
                Stype <- ifelse(Stype=="A","M",Stype);
                modelIsMultiplicative <- TRUE;
            }
        }
    }
    else{
        modelIsMultiplicative <- FALSE;
    }

    # This is the number of parameters to estimate per series
    nParamMax <- 0;

    ##### Persistence matrix ####
    # persistence type can be: "i" - individual, "d" - dependent, "c" - common (all),
    # "s" - seasonal smoothing parameter is the same
    persistenceValue <- persistence;
    if(is.null(persistenceValue)){
        warning("persistence value is not selected. Switching to group.");
        persistenceType <- "c";
        persistenceEstimate <- TRUE;
    }
    else{
        if(is.character(persistenceValue)){
            persistenceValue <- substring(persistenceValue[1],1,1);
            if(all(persistenceValue!=c("c","i","d","s"))){
                warning("You asked for a strange persistence value. We don't do that here. Switching to group",
                        call.=FALSE);
                persistenceType <- "c";
            }
            else{
                if(persistenceValue=="s" & Stype=="N"){
                    warning(paste0("Non-seasonal model is selected, but you've asked for common ",
                                   "seasonal smoothing parameter. Switching to persistence='individual'."),
                            call.=FALSE);
                    persistenceValue <- "i";
                }
                persistenceType <- persistenceValue;
            }
            persistenceValue <- NULL;
            persistenceEstimate <- TRUE;
        }
        else if(is.numeric(persistenceValue)){
            if(all(length(persistenceValue) != c(nComponentsAll*nSeries^2,nComponentsAll))){
                warning(paste0("Length of persistence matrix is wrong! It should be either ",
                               nComponentsAll*nSeries^2, " or ", nComponentsAll,
                               " instead of ",length(persistenceValue),".\n",
                               "Values of persistence matrix will be estimated as group."),call.=FALSE);
                persistenceValue <- NULL;
                persistenceType <- "c";
                persistenceEstimate <- TRUE;
            }
            else{
                persistenceType <- "p";
                persistenceEstimate <- FALSE;

                if(length(persistenceValue)==nComponentsAll){
                    persistenceBuffer <- matrix(0,nSeries*nComponentsAll,nSeries);
                    for(i in 1:nSeries){
                        persistenceBuffer[1:nComponentsAll+nComponentsAll*(i-1),i] <- persistenceValue;
                    }
                    persistenceValue <- persistenceBuffer;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(as.vector(persistenceValue));
                }
                else{
                    ### Check the persistence matrix in order to decide number of parameters
                    persistencePartial <- matrix(persistenceValue[1:nComponentsAll,1:nSeries],
                                                 nComponentsAll,nSeries);
                    persistenceValue <- matrix(persistenceValue,nSeries*nComponentsAll,nSeries);

                    # Check if persistence is dependent
                    if(all(persistencePartial[,nSeries]==0)){
                        # Check if persistence is grouped
                        if(persistenceValue[1,1]==persistenceValue[1+nComponentsAll,nSeries]){
                            parametersNumber[2,1] <- parametersNumber[2,1] + nSeries;
                        }
                        else{
                            parametersNumber[2,1] <- parametersNumber[2,1] + nSeries*nComponentsAll;
                        }
                    }
                    else{
                        parametersNumber[2,1] <- parametersNumber[2,1] + length(as.vector(persistenceValue));
                    }
                }
            }
        }
        else if(!is.numeric(persistenceValue)){
            warning(paste0("persistence matrix is not numeric!\n",
                           "Values of persistence matrix will be estimated as group."),call.=FALSE);
            persistenceValue <- NULL;
            persistenceType <- "c";
            persistenceEstimate <- TRUE;
        }
    }

    # If it is individual, then it increases by nComponentsAll
    if(persistenceType=="i"){
    # if(any(persistenceType==c("c","i","s"))){
        nParamMax <- nParamMax + nComponentsAll;
    }
    # The seasonal is shared across series, the other parameters are individual
    else if(persistenceType=="s"){
        nParamMax <- nParamMax + nComponentsNonSeasonal + 1/nSeries;
    }
    # All parameters are shared
    else if(persistenceType=="c"){
        nParamMax <- nParamMax + nComponentsAll/nSeries;
    }
    else if(persistenceType=="d"){
        # In case with "dependent" the whole matrix needs to be estimated
        nParamMax <- nParamMax + nComponentsAll*nSeries;
    }

    ##### Transition matrix ####
    # transition type can be: "i" - individual, "d" - dependent, "c" - common
    transitionValue <- transition;
    if(is.null(transitionValue)){
        warning("transition value is not selected. Switching to common");
        transitionType <- "c";
        transitionEstimate <- FALSE;
    }
    else{
        if(is.character(transitionValue)){
            transitionValue <- substring(transitionValue[1],1,1);
            if(all(transitionValue!=c("c","i","d"))){
                warning("You asked for a strange transition value. We don't do that here. Switching to common",
                        call.=FALSE);
                transitionType <- "c";
            }
            else{
                transitionType <- transitionValue;
            }
            transitionValue <- NULL;
            transitionEstimate <- FALSE;
        }
        else if(is.numeric(transitionValue)){
            if(all(length(transitionValue) != c((nSeries*nComponentsAll)^2,nComponentsAll^2))){
                warning(paste0("Length of transition matrix is wrong! It should be either ",
                               (nSeries*nComponentsAll)^2, " or ", nComponentsAll^2,
                               " instead of ",length(transitionValue),".\n",
                               "Values of transition matrix will be estimated as a common one."),call.=FALSE);
                transitionValue <- NULL;
                transitionType <- "c";
                transitionEstimate <- FALSE;
            }
            else{
                transitionType <- "p";
                transitionEstimate <- FALSE;
                ### Check the transition matrix in order to decide number of parameters
                transitionPartial <- matrix(transitionValue[1:nComponentsAll,1:nComponentsAll],
                                            nComponentsAll,nComponentsAll);
                transitionIsStandard <- FALSE;
                transitionContainsPhi <- FALSE;
                if(ncol(transitionPartial)==3){
                    if(all(transitionPartial[,c(1,3)]==matrix(c(1,0,0,1,1,0,0,0,1),3,3)[,c(1,3)])){
                        transitionIsStandard <- TRUE;
                        if(transitionPartial[2,2]!=1){
                            # If there is phi in the matrix, add it
                            transitionContainsPhi <- TRUE;
                        }
                    }
                }
                else if(ncol(transitionPartial)==2){
                    if(all(transitionPartial[,1]==c(1,0))){
                        transitionIsStandard <- TRUE;
                        if(transitionPartial[2,2]!=1){
                            # If there is phi in the matrix, add it
                            transitionContainsPhi <- TRUE;
                        }
                    }
                }
                else{
                    if(transitionPartial[1,1]==1){
                        transitionIsStandard <- TRUE;
                    }
                }
                # If transition is not standard, take unique values from it.
                if(!transitionIsStandard){
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(as.vector(transitionValue));
                }
                else{
                    # If there is phi, check if it is grouped
                    if(transitionContainsPhi){
                        # If phi is grouped, add one parameter
                        if(transitionValue[2,2]==transitionValue[2+nComponentsAll,2+nComponentsAll]){
                            parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                            phi <- transitionValue[2,2];
                        }
                        # Else phi is individual
                        else{
                            parametersNumber[2,1] <- parametersNumber[2,1] + nSeries;
                            phi <- rep(NA,nSeries);
                            for(i in 1:nSeries){
                                phi[i] <- transitionValue[2+(i-1)*nComponentsAll,2+(i-1)*nComponentsAll];
                            }
                        }
                    }
                }

                if(length(transitionValue) == nComponentsAll^2){
                    transitionValue <- matrix(transitionValue,nComponentsAll,nComponentsAll);
                    transitionBuffer <- diag(nSeries*nComponentsAll);
                    for(i in 1:nSeries){
                        transitionBuffer[c(1:nComponentsAll)+nComponentsAll*(i-1),
                                         c(1:nComponentsAll)+nComponentsAll*(i-1)] <- transitionValue;
                    }
                    transitionValue <- transitionBuffer;
                }
                else{
                    transitionValue <- matrix(transitionValue,nSeries*nComponentsAll,nSeries*nComponentsAll);
                }
            }
        }
        else if(!is.numeric(transitionValue)){
            warning(paste0("transition matrix is not numeric!\n",
                           "Values of transition vector will be estimated as a common group."),call.=FALSE);
            transitionValue <- NULL;
            transitionType <- "c";
            transitionEstimate <- FALSE;
        }
    }

    if(transitionType=="d"){
        ## !!! Each separate transition matrix is not evaluated, but the off-diagonals are
        transitionEstimate <- TRUE;
        nParamMax <- nParamMax + (nSeries-1)*nSeries*nComponentsAll^2;
    }

    ##### Damping parameter ####
    # phi type can be: "i" - individual, "c" - common
    dampedValue <- phi;
    if(transitionType!="p"){
        if(damped){
            if(is.null(dampedValue)){
                warning("phi value is not selected. Switching to common.");
                dampedType <- "c";
                dampedEstimate <- TRUE;
            }
            else{
                if(is.character(dampedValue)){
                    dampedValue <- substring(dampedValue[1],1,1);
                    if(all(dampedValue!=c("i","c"))){
                        warning("You asked for a strange phi value. We don't do that here. Switching to common.",
                                call.=FALSE);
                        dampedType <- "c";
                    }
                    else{
                        dampedType <- dampedValue;
                    }
                    dampedValue <- matrix(1,nSeries,1);
                    dampedEstimate <- TRUE;
                }
                else if(is.numeric(dampedValue)){
                    if((length(dampedValue) != nSeries) & (length(dampedValue)!= 1)){
                        warning(paste0("Length of phi vector is wrong! It should be ",
                                       nSeries,
                                       " instead of ",length(dampedValue),".\n",
                                       "Values of phi vector will be estimated as a common one."),call.=FALSE);
                        dampedValue <- matrix(1,nSeries,1);
                        dampedType <- "c";
                        dampedEstimate <- TRUE;
                    }
                    else{
                        dampedType <- "p";
                        dampedValue <- matrix(dampedValue,nSeries,1);
                        dampedEstimate <- FALSE;
                        parametersNumber[2,1] <- parametersNumber[2,1] + length(as.vector(dampedValue));
                    }
                }
                else if(!is.numeric(dampedValue)){
                    warning(paste0("phi vector is not numeric!\n",
                                   "Values of phi vector will be estimated as a common one."),call.=FALSE);
                    dampedValue <- matrix(1,nSeries,1);
                    dampedType <- "c";
                    dampedEstimate <- TRUE;
                }
            }

            if(any(dampedType==c("c","i"))){
                dampedValue <- matrix(1,nSeries,1);
                # In case of common, the parameter is shared.
                if(dampedType=="c"){
                    nParamMax <- nParamMax + 1/nSeries;
                }
                else{
                    nParamMax <- nParamMax + 1;
                }
            }
        }
        else{
            dampedValue <- matrix(1,nSeries,1);
            dampedType <- "c";
            dampedEstimate <- FALSE;
        }
    }
    else{
        dampedType <- "c";
        dampedEstimate <- FALSE;
    }

    ##### initials ####
    # initial type can be: "i" - individual, "c" - common
    initialValue <- initial;
    if(is.null(initialValue)){
        warning("Initial value is not selected. Switching to individual.");
        initialType <- "i";
        initialEstimate <- TRUE;
    }
    else{
        if(is.character(initialValue)){
            initialValue <- substring(initialValue[1],1,1);
            if(all(initialValue!=c("i","c"))){
                warning("You asked for a strange initial value. We don't do that here. Switching to individual.",
                        call.=FALSE);
                initialType <- "i";
            }
            else{
                initialType <- initialValue;
            }
            initialValue <- NULL;
            initialEstimate <- TRUE;
        }
        else if(is.numeric(initialValue)){
            if(smoothType=="ves"){
                if(length(initialValue)>2*nSeries){
                    warning(paste0("Length of initial vector is wrong! It should not be greater than",
                                   2*nSeries,"\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "i";
                    initialEstimate <- TRUE;
                }
                else{
                    if(all(length(initialValue) != c(nComponentsNonSeasonal,nComponentsNonSeasonal * nSeries))){
                        warning(paste0("Length of initial vector is wrong! It should be either ",
                                       nComponentsNonSeasonal*nSeries, " or ", nComponentsNonSeasonal,
                                       " instead of ",length(initialValue),".\n",
                                       "Values of initial vector will be estimated."),call.=FALSE);
                        initialValue <- NULL;
                        initialType <- "i";
                        initialEstimate <- TRUE;
                    }
                    else{
                        initialType <- "p";
                        initialValue <- matrix(initialValue,nComponentsNonSeasonal * nSeries,1);
                        initialEstimate <- FALSE;
                        parametersNumber[2,1] <- parametersNumber[2,1] + length(as.vector(initialValue));
                    }
                }
            }
        }
        else if(!is.numeric(initialValue)){
            warning(paste0("Initial vector is not numeric!\n",
                           "Values of initial vector will be estimated."),call.=FALSE);
            initialValue <- NULL;
            initialType <- "i";
            initialEstimate <- TRUE;
        }
    }

    # Individual initials
    if(initialType=="i"){
        nParamMax <- nParamMax + nComponentsNonSeasonal;
    }
    # Common initials are shared across series
    else{
        nParamMax <- nParamMax + nComponentsNonSeasonal / nSeries;
    }

    if(smoothType=="ves"){
    ##### initialSeason and seasonal for VES #####
    # Here we should check if initialSeason is character or not...
    # if length(initialSeason) == dataFreq*nSeries, then ok
    # if length(initialSeason) == dataFreq, then use it for all nSeries
        if(Stype!="N"){
            #### Seasonal component
            seasonalType <- seasonal;
            if(is.null(seasonalType)){
                warning("The type of the seasonal component is not selected. Switching to individual.");
                seasonalType <- "i";
            }
            else{
                if(is.character(seasonalType)){
                    seasonalType <- substring(seasonalType[1],1,1);
                    if(all(seasonalType!=c("i","c"))){
                        warning("You asked for a strange seasonal value. We don't do that here. Switching to individual.",
                                call.=FALSE);
                        seasonalType <- "i";
                    }
                    else if(seasonalType=="c"){
                        if(Stype=="N"){
                            warning("Common seasonal model does not make sense with the non-seasonal ETS. Changing to individual",
                                    call.=FALSE);
                            seasonalType <- "i";
                        }
                        else{
                            # If the transition matrix provide is full, cut off all the seasonals except for the first one.
                            if(transitionType=="p" && nrow(transitionValue)==nSeries*nComponentsAll){
                                warning(paste0("The transition matrix you provided contains too many rows ",
                                               "for the common seasonal model.",
                                               "Using only the first seasonal one."), call.=FALSE);
                                transitionValue <- rbind(cbind(transitionValue[-(c(1:nSeries)*nComponentsAll),
                                                                               -(c(1:nSeries)*nComponentsAll)],
                                                               0),
                                                         c(transitionValue[nComponentsAll*nSeries,
                                                                           -(c(1:nSeries)*nComponentsAll)],1));
                            }
                            # Do similar stuff for the persistence
                            if(persistenceType=="p" && nrow(persistenceValue)==nSeries*nComponentsAll){
                                warning(paste0("The persistence matrix you provided contains too many rows ",
                                               "for the common seasonal model.",
                                               "Using only the first seasonal one."), call.=FALSE);
                                persistenceValue <- rbind(persistenceValue[-(c(1:nSeries)*nComponentsAll),],
                                                          persistenceValue[nComponentsAll,]);
                            }
                        }
                    }
                }
                else{
                    warning("A weird stuff is provided for the seasonal component. Switching to individual.",
                            call.=FALSE);
                    seasonalType <- "i";
                }
            }

            #### Initials
            initialSeasonValue <- initialSeason;
            if(is.null(initialSeasonValue)){
                warning("Initial value is not selected. Switching to common");
                initialSeasonType <- "c";
                initialSeasonEstimate <- TRUE;
            }
            else{
                if(is.character(initialSeasonValue)){
                    initialSeasonValue <- substring(initialSeasonValue[1],1,1);
                    if(all(initialSeasonValue!=c("i","c"))){
                        warning("You asked for a strange initialSeason value. We don't do that here. Switching to common",
                                call.=FALSE);
                        initialSeasonType <- "c";
                    }
                    else{
                        if(seasonalType=="c" && initialSeasonValue=="i"){
                            warning("initialSeason='i' does not work with seasonalType='c'. Switching to common.",
                                    call.=FALSE);
                            initialSeasonValue <- "c";
                        }
                        initialSeasonType <- initialSeasonValue;
                    }
                    initialSeasonValue <- NULL;
                    initialSeasonEstimate <- TRUE;
                }
                else if(is.numeric(initialSeasonValue)){
                    if(smoothType=="ves"){
                        if(all(length(initialSeasonValue)!=c(dataFreq,dataFreq*nSeries))){
                            warning(paste0("The length of initialSeason is wrong! ",
                                           "It should correspond to the frequency of the data.",
                                           "Values of initialSeason will be estimated as a common one."),call.=FALSE);
                            initialSeasonValue <- NULL;
                            initialSeasonType <- "c";
                            initialSeasonEstimate <- TRUE;
                        }
                        else{
                            if(seasonalType=="i"){
                                initialSeasonValue <- matrix(initialSeasonValue,nSeries,dataFreq);
                            }
                            else{
                                if(length(initialSeasonValue)!=dataFreq){
                                    warning(paste0("The initialSeason you provided contains too many elements ",
                                                   "for the common seasonal model.",
                                                   "Using only the first ",dataFreq," values."), call.=FALSE);
                                }
                                initialSeasonValue <- matrix(initialSeasonValue[1:dataFreq],1,dataFreq);
                            }
                            initialSeasonType <- "p";
                            initialSeasonEstimate <- FALSE;
                            parametersNumber[2,1] <- parametersNumber[2,1] + length(as.vector(initialSeasonValue));
                        }
                    }
                }
                else if(!is.numeric(initialSeasonValue)){
                    warning(paste0("Initial vector is not numeric!\n",
                                   "Values of initialSeason vector will be estimated as a common one."),call.=FALSE);
                    initialSeasonValue <- NULL;
                    initialSeasonType <- "c";
                    initialSeasonEstimate <- TRUE;
                }
            }

            if(initialSeasonType=="i"){
                nParamMax <- nParamMax + dataFreq;
            }
            else if(initialSeasonType=="c"){
                nParamMax <- nParamMax + dataFreq / nSeries;
            }
        }
        else{
            initialSeasonValue <- NULL;
            initialSeasonType <- "c";
            initialSeasonEstimate <- FALSE;
            seasonalType <- "i";
        }
    }

    ##### Loss function type #####
    loss <- loss[1];
    if(!any(loss==c("likelihood","diagonal","trace","l","d","t"))){
        warning(paste0("Strange loss function specified: ",loss,". Switching to 'likelihood'."),call.=FALSE);
        loss <- "likelihood";
    }
    loss <- substr(loss,1,1);

    # If it is likelihood, we also need to estimate the full covariance matrix
    if(loss=="l"){
        nParamMax <- nParamMax + nSeries;
    }
    # Otherwise, these are just variances of the data
    else{
        nParamMax <- nParamMax + 1;
    }

    normalizer <- sum(colMeans(abs(diff(t(yInSample))),na.rm=TRUE));

    ##### Information Criteria #####
    ic <- ic[1];
    if(all(ic!=c("AICc","AIC","BIC","BICc"))){
        warning(paste0("Strange type of information criteria defined: ",ic,". Switching to 'AICc'."),call.=FALSE);
        ic <- "AICc";
    }

    ##### interval, intervalType, level #####
    intervalType <- interval[1];
    # Check the provided type of interval

    if(is.logical(intervalType)){
        if(intervalType){
            intervalType <- "c";
        }
        else{
            intervalType <- "none";
        }
    }

    if(all(intervalType!=c("c","u","i","l","n","none","conditional","unconditional","individual","likelihood"))){
        warning(paste0("Wrong type of interval: '",intervalType, "'. Switching to 'conditional'."),call.=FALSE);
        intervalType <- "c";
    }

    if(intervalType=="none"){
        intervalType <- "n";
        interval <- FALSE;
    }
    else if(intervalType=="conditional"){
        intervalType <- "c";
        interval <- TRUE;
    }
    else if(intervalType=="unconditional"){
        intervalType <- "u";
        interval <- TRUE;
    }
    else if(intervalType=="individual"){
        intervalType <- "i";
        interval <- TRUE;
    }
    else if(intervalType=="likelihood"){
        intervalType <- "l";
        interval <- TRUE;
    }
    else{
        interval <- TRUE;
    }

    if(level>1){
        level <- level / 100;
    }

    ##### bounds #####
    bounds <- substring(bounds[1],1,1);
    if(all(bounds!=c("u","a","n"))){
        warning("Strange bounds are defined. Switching to 'admissible'.",call.=FALSE);
        bounds <- "a";
    }

    ##### Check number of observations vs number of max parameters #####
    if(obsInSample <= nParamMax){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters per series is ",
                    nParamMax," while the number of observations is ",obsInSample,"."),call.=FALSE);
    }

    ##### Fisher Information #####
    if(!exists("FI")){
        FI <- FALSE;
    }
    else{
        if(!is.logical(FI)){
            FI <- FALSE;
        }
        if(!requireNamespace("numDeriv",quietly=TRUE) & FI){
            warning(paste0("Sorry, but you don't have 'numDeriv' package, ",
                           "which is required in order to produce Fisher Information.",call.=FALSE));
            FI <- FALSE;
        }
    }

    ##### Ellipsis thingies #####
    if(!is.null(ellipsis$B)){
        B <- ellipsis$B;
    }
    else{
        B <- NULL;
    }
    if(!is.null(ellipsis$ub)){
        ub <- ellipsis$ub;
    }
    else{
        ub <- NULL;
    }
    if(!is.null(ellipsis$lb)){
        lb <- ellipsis$lb;
    }
    else{
        lb <- NULL;
    }
    if(!is.null(ellipsis$maxeval)){
        maxeval <- ellipsis$maxeval;
    }
    else{
        maxeval <- 1000;
    }
    if(!is.null(ellipsis$algorithm1)){
        algorithm1 <- ellipsis$algorithm1;
    }
    else{
        algorithm1 <- "NLOPT_LN_BOBYQA";
    }
    if(!is.null(ellipsis$algorithm2)){
        algorithm2 <- ellipsis$algorithm2;
    }
    else{
        algorithm2 <- "NLOPT_LN_NELDERMEAD";
    }
    if(!is.null(ellipsis$xtol_rel1)){
        xtol_rel1 <- ellipsis$xtol_rel1;
    }
    else{
        xtol_rel1 <- 1e-8;
    }
    if(!is.null(ellipsis$xtol_rel2)){
        xtol_rel2 <- ellipsis$xtol_rel2;
    }
    else{
        xtol_rel2 <- 1e-6;
    }

    ##### Return values to previous environment #####
    assign("h",h,ParentEnvironment);
    assign("silentText",silentText,ParentEnvironment);
    assign("silentGraph",silentGraph,ParentEnvironment);
    assign("silentLegend",silentLegend,ParentEnvironment);
    assign("obsInSample",obsInSample,ParentEnvironment);
    assign("obsAll",obsAll,ParentEnvironment);
    assign("obsStates",obsStates,ParentEnvironment);
    assign("nSeries",nSeries,ParentEnvironment);
    assign("nParamMax",nParamMax,ParentEnvironment);
    assign("y",y,ParentEnvironment);
    assign("yInSample",yInSample,ParentEnvironment);
    assign("dataFreq",dataFreq,ParentEnvironment);
    assign("dataDeltat",dataDeltat,ParentEnvironment);
    assign("dataStart",dataStart,ParentEnvironment);
    assign("yForecastStart",yForecastStart,ParentEnvironment);
    assign("dataNames",dataNames,ParentEnvironment);
    assign("parametersNumber",parametersNumber,ParentEnvironment);

    assign("model",model,ParentEnvironment);
    # assign("modelsPool",modelsPool,ParentEnvironment);
    assign("Etype",Etype,ParentEnvironment);
    assign("Ttype",Ttype,ParentEnvironment);
    assign("Stype",Stype,ParentEnvironment);
    assign("lagsModelMax",lagsModelMax,ParentEnvironment);
    assign("modelIsSeasonal",modelIsSeasonal,ParentEnvironment);
    assign("modelIsMultiplicative",modelIsMultiplicative,ParentEnvironment);
    assign("nComponentsAll",nComponentsAll,ParentEnvironment);
    assign("nComponentsNonSeasonal",nComponentsNonSeasonal,ParentEnvironment);

    assign("persistenceValue",persistenceValue,ParentEnvironment);
    assign("persistenceType",persistenceType,ParentEnvironment);
    assign("persistenceEstimate",persistenceEstimate,ParentEnvironment);

    assign("transitionValue",transitionValue,ParentEnvironment);
    assign("transitionType",transitionType,ParentEnvironment);
    assign("transitionEstimate",transitionEstimate,ParentEnvironment);

    assign("damped",damped,ParentEnvironment);
    assign("dampedValue",dampedValue,ParentEnvironment);
    assign("dampedType",dampedType,ParentEnvironment);
    assign("dampedEstimate",dampedEstimate,ParentEnvironment);

    assign("initialValue",initialValue,ParentEnvironment);
    assign("initialType",initialType,ParentEnvironment);
    assign("initialEstimate",initialEstimate,ParentEnvironment);

    assign("initialSeasonValue",initialSeasonValue,ParentEnvironment);
    assign("initialSeasonType",initialSeasonType,ParentEnvironment);
    assign("initialSeasonEstimate",initialSeasonEstimate,ParentEnvironment);

    assign("seasonalType",seasonalType,ParentEnvironment);

    assign("loss",loss,ParentEnvironment);
    assign("normalizer",normalizer,ParentEnvironment);

    assign("ic",ic,ParentEnvironment);

    assign("intervalType",intervalType,ParentEnvironment);
    assign("interval",interval,ParentEnvironment);

    assign("intermittent",intermittent,ParentEnvironment);
    assign("ot",ot,ParentEnvironment);
    assign("otObs",otObs,ParentEnvironment);
    assign("intermittentModel",intermittentModel,ParentEnvironment);
    assign("imodelProvided",imodelProvided,ParentEnvironment);
    assign("imodel",imodel,ParentEnvironment);
    assign("iprobability",iprobability,ParentEnvironment);

    # assign("yot",yot,ParentEnvironment);
    # assign("pt",pt,ParentEnvironment);
    # assign("pt.for",pt.for,ParentEnvironment);
    # assign("nParamIntermittent",nParamIntermittent,ParentEnvironment);
    # assign("iprob",iprob,ParentEnvironment);

    assign("bounds",bounds,ParentEnvironment);

    # Stuff in ellipsis
    assign("FI",FI,ParentEnvironment);
    assign("B",B,ParentEnvironment);
    assign("ub",ub,ParentEnvironment);
    assign("lb",lb,ParentEnvironment);
    assign("maxeval",maxeval,ParentEnvironment);
    assign("algorithm1",algorithm1,ParentEnvironment);
    assign("algorithm2",algorithm2,ParentEnvironment);
    assign("xtol_rel1",xtol_rel1,ParentEnvironment);
    assign("xtol_rel2",xtol_rel2,ParentEnvironment);
}

##### *Likelihood function* #####
vLikelihoodFunction <- function(B){
    if(Etype=="A"){
        return(- obsInSample/2 * (nSeries*log(2*pi*exp(1)) + CF(B)));
    }
    else if(Etype=="M"){
        return(- obsInSample/2 * (nSeries*log(2*pi*exp(1)) + CF(B)) - sum(yInSample));
    }
    else{
        #### This is not derived yet ####
        return(- obsInSample/2 * (nSeries*log(2*pi*exp(1)) + CF(B)));
    }
}

##### *Function calculates ICs* #####
vICFunction <- function(nParam=nParam,B,Etype=Etype){
    # Information criteria are calculated with the constant part "log(2*pi*exp(1)*h+log(obs))*obs".
    # And it is based on the mean of the sum squared residuals either than sum.
    # Hyndman likelihood is: llikelihood <- obs*log(obs*cfObjective)

    # Number of parameters per series needs to be used in the calculations of information criteria
    nParamPerSeries <- nParam / nSeries;
    llikelihood <- vLikelihoodFunction(B);

    coefAIC <- 2*nParamPerSeries - 2*llikelihood;
    coefBIC <- log(obsInSample)*nParamPerSeries - 2*llikelihood;

    # max here is needed in order to take into account cases with higher number of parameters than observations
    coefAICc <- ((2*obsInSample*(nParamPerSeries*nSeries + nSeries*(nSeries+1)/2)) /
                                 max(obsInSample - (nParamPerSeries + nSeries + 1),0)) -2*llikelihood;

    coefBICc <- (((nParamPerSeries + nSeries*(nSeries+1)/2) *
                      log(obsInSample * nSeries) * obsInSample * nSeries) /
                     (obsInSample * nSeries - nParamPerSeries - nSeries*(nSeries+1)/2)) -2*llikelihood;

    ICs <- c(coefAIC, coefAICc, coefBIC, coefBICc);
    names(ICs) <- c("AIC", "AICc", "BIC", "BICc");

    return(list(llikelihood=llikelihood,ICs=ICs));
}

##### *vssFitter function* #####
vssFitter <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    fitting <- vFitterWrap(yInSample, matvt, matF, matW, matG,
                           lagsModel, Etype, Ttype, Stype, ot);
    statesNames <- rownames(matvt);
    matvt <- fitting$matvt;
    rownames(matvt) <- statesNames;
    yFitted[] <- fitting$yfit;
    errors[] <- fitting$errors;

    if(modelIsMultiplicative){
        yFitted <- exp(yFitted);
        if(intermittent!="n"){
            yFitted[ot==0] <- 0;
        }
    }

    assign("matvt",matvt,ParentEnvironment);
    assign("yFitted",yFitted,ParentEnvironment);
    assign("errors",errors,ParentEnvironment);
}

##### *State space interval* #####
# This is not implemented yet
#' @importFrom stats qchisq
vssIntervals <- function(level=0.95, intervalType=c("c","u","i","l"), Sigma=NULL,
                         measurement=NULL, transition=NULL, persistence=NULL,
                         lagsModel=NULL, cumulative=FALSE, df=0,
                         nComponents=1, nSeries=1, h=1){

    lagsModelMax <- max(lagsModel);
    # nElements <- nComponents*nSeries;
    nElements <- length(lagsModel);

    # This is a temporary solution, needed while we work on other types.
    if(intervalType!="i"){
        intervalType <- "i";
    }
    # In case of likelihood interval, construct the individual ones
    # if(intervalType=="l"){
    #     intervalType <- "i";
    # }

    # In case of individual we use either t distribution or Chebyshev inequality
    if(intervalType=="i"){
        if(df>0){
            quantUpper <- qt((1+level)/2,df=df);
            quantLower <- qt((1-level)/2,df=df);
        }
        else{
            quantUpper <- sqrt(1/((1-level)/2));
            quantLower <- -quantUpper;
        }
    }
    # In case of conditional / unconditional, we use Chi-squared distribution
    else{
        quant <- qchisq(level,df=nSeries);
    }

    nPoints <- 100;
    if(intervalType=="c"){
        # Number of points in the ellipse
        PI <- array(NA, c(h,2*nPoints^(nSeries-1),nSeries),
                    dimnames=list(paste0("h",c(1:h)), NULL,
                                  paste0("Series_",1:nSeries)));
    }
    else{
        PI <- matrix(NA, nrow=h, ncol=nSeries*2,
                     dimnames=list(paste0("h",c(1:h)),
                                   paste0("Series_",rep(c(1:nSeries),each=2),c("_lower","_upper"))));
    }

    # Array of final variance matrices
    varVec <- array(NA,c(h,nSeries,nSeries));
    # This is needed for the first observations, where we do not care about the transition equation
    for(i in 1:min(h,lagsModelMax)){
        varVec[i,,] <- Sigma;
    }

    if(h>1){
        if(cumulative){
            covarVec <- array(NA,c(h,nSeries,nSeries));
        }

        matrixOfVarianceOfStates <- array(0,c(nElements,nElements,h+lagsModelMax));
        # This multiplication does not make sense
        matrixOfVarianceOfStates[,,1:lagsModelMax] <- persistence %*% Sigma %*% t(persistence);
        matrixOfVarianceOfStatesLagged <- as.matrix(matrixOfVarianceOfStates[,,1]);

        # New transition and measurement for the internal use
        transitionNew <- matrix(0,nElements,nElements);
        measurementNew <- matrix(0,nSeries,nElements);

        # selectionMat is needed for the correct selection of lagged variables in the array
        # elementsNew are needed for the correct fill in of all the previous matrices
        selectionMat <- transitionNew;
        elementsNew <- rep(FALSE,nElements);

        # Define chunks, which correspond to the lags with h being the final one
        chuncksOfHorizon <- c(1,unique(lagsModel),h);
        chuncksOfHorizon <- sort(chuncksOfHorizon);
        chuncksOfHorizon <- chuncksOfHorizon[chuncksOfHorizon<=h];
        chuncksOfHorizon <- unique(chuncksOfHorizon);

        # Length of the vector, excluding the h at the end
        chunksLength <- length(chuncksOfHorizon) - 1;

        elementsNew <- lagsModel<=(chuncksOfHorizon[1]);
        measurementNew[,elementsNew] <- measurement[,elementsNew];

        for(j in 1:chunksLength){
            selectionMat[lagsModel==chuncksOfHorizon[j],] <- chuncksOfHorizon[j];
            selectionMat[,lagsModel==chuncksOfHorizon[j]] <- chuncksOfHorizon[j];

            elementsNew <- lagsModel < (chuncksOfHorizon[j]+1);
            transitionNew[,elementsNew] <- transition[,elementsNew];
            measurementNew[,elementsNew] <- measurement[,elementsNew];

            for(i in (chuncksOfHorizon[j]+1):chuncksOfHorizon[j+1]){
                selectionMat[lagsModel>chuncksOfHorizon[j],] <- i;
                selectionMat[,lagsModel>chuncksOfHorizon[j]] <- i;

                matrixOfVarianceOfStatesLagged[elementsNew,
                                               elementsNew] <- matrixOfVarianceOfStates[cbind(rep(c(1:nElements),
                                                                                                  each=nElements),
                                                                                              rep(c(1:nElements),
                                                                                                  nElements),
                                                                                              i - c(selectionMat))];

                matrixOfVarianceOfStates[,,i] <- (transitionNew %*% matrixOfVarianceOfStatesLagged %*% t(transitionNew) +
                                                      persistence %*% Sigma %*% t(persistence));
                varVec[i,,] <- measurementNew %*% matrixOfVarianceOfStatesLagged %*% t(measurementNew) + Sigma;
                if(cumulative){
                    covarVec[i] <- measurementNew %*% transitionNew %*% persistence;
                }
            }
        }

        if(cumulative){
            varVec <- apply(varVec,c(2,3),sum) + 2*Sigma %*% apply(covarVec*array(c(0,h:2),c(h,nSeries,nSeries)),
                                                                   c(2,3),sum);
        }
    }

    # Produce PI matrix
    if(any(intervalType==c("c","u"))){
        # eigensList contains eigenvalues and eigenvectors of the covariance matrix
        eigensList <- apply(varVec,1,eigen);
        # eigenLimits specify the lowest and highest ellipse points in all dimensions
        eigenLimits <- matrix(NA,nSeries,2);
        # ellipsePoints contains coordinates of the ellipse on the eigenvectors basis
        ellipsePoints <- array(NA, c(h, 2*nPoints^(nSeries-1), nSeries));
        for(i in 1:h){
            eigenLimits[,2] <- sqrt(quant / eigensList[[i]]$value);
            eigenLimits[,1] <- -eigenLimits[,2];
            ellipsePoints[i,,nSeries] <- rep(seq(eigenLimits[nSeries,1],
                                                      eigenLimits[nSeries,2],
                                                      length.out=nPoints),nSeries);
            for(j in (nSeries-1):1){
                ellipsePoints[i,,nSeries];
            }
        }
    }
    else if(intervalType=="i"){
        variances <- apply(varVec,1,diag);
        for(i in 1:nSeries){
            PI[,2*i-1] <- quantLower * sqrt(variances[i,]);
            PI[,2*i] <- quantUpper * sqrt(variances[i,]);
        }
    }

    return(PI);
}

##### *Forecaster of state space functions* #####
vssForecaster <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    # Division by nSeries gives the df per series, which agrees with Lutkepohl (2005), p.75
    nParamPerSeries <- nParam / nSeries;
    # df <- (otObs - nParamPerSeries);
    if(intervalType!="l" && any(otObs >= nParamPerSeries)){
        df <- otObs - nParamPerSeries;
    }
    else{
        df <- otObs;
    }

    # Divide each element by each degree of freedom
    Sigma <- (errors %*% t(errors)) / df;
    rownames(Sigma) <- colnames(Sigma) <- dataNames;

    # if(any((otObs - nParamPerSeries)<=0)){
    #     df <- 0;
    # }
    # else{
    # Take the minimum df for the purposes of interval construction
        df <- min(df);
    # }

    PI <- NA;

    if(h>0){
        yForecast[] <- vForecasterWrap(matrix(matvt[,(obsInSample+1):(obsInSample+lagsModelMax)],ncol=lagsModelMax),
                                     matF, matW, nSeries, h, Etype, Ttype, Stype, lagsModel);

        if(cumulative){
            yForecast <- rowSums(yForecast);
        }

        if(interval){
            PI <- vssIntervals(level=level, intervalType=intervalType, Sigma=Sigma,
                               measurement=matW, transition=matF, persistence=matG,
                               lagsModel=lagsModel, cumulative=cumulative, df=df,
                               nComponents=nComponentsAll, nSeries=nSeries, h=h);

            if(any(intervalType==c("i","l","u"))){
                for(i in 1:nSeries){
                    PI[,i*2-1] <- PI[,i*2-1] + yForecast[i,];
                    PI[,i*2] <- PI[,i*2] + yForecast[i,];
                }
            }
        }
    }
    else{
        yForecast[] <- NA;
    }

    if(any(is.na(yFitted),all(is.na(yForecast),h>0))){
        warning("Something went wrong during the optimisation and NAs were produced!",
                call.=FALSE);
        warning("Please check the input and report this error to the maintainer if it persists.",
                call.=FALSE);
    }

    if(modelIsMultiplicative){
        yForecast[] <- exp(yForecast);
        PI[] <- exp(PI);
    }

    if(intermittent!="n"){
        if(!imodelProvided){
            imodel <- viss(ts(t(ot),frequency=dataFreq),
                           intermittent=intermittent, h=h, holdout=FALSE,
                           probability=iprobability, model=intermittentModel);
        }
        yForecast[] <- yForecast * t(imodel$forecast);
    }

    assign("Sigma",Sigma,ParentEnvironment);
    assign("yForecast",yForecast,ParentEnvironment);
    assign("PI",PI,ParentEnvironment);
    assign("imodel",imodel,ParentEnvironment);
}
