utils::globalVariables(c("initialSeason","persistence","phi"));

##### *Checker of input of vector functions* #####
vssInput <- function(modelType=c("ves"),...){
    modelType <- modelType[1];

    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ##### silent #####
    silent <- silent[1];
    # Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output","debugging","n","a","g","l","o","d"))){
            warning(paste0("Sorry, I have no idea what 'silent=",silent,"' means. Switching to 'none'."),call.=FALSE);
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
        warning(paste0("You have set forecast horizon equal to ",h,". We hope you know, what you are doing."), call.=FALSE);
        if(h<0){
            warning("And by the way, we can't do anything with negative horizon, so we will set it equal to zero.", call.=FALSE);
            h <- 0;
        }
    }

    #### Check data ####
    if(!is.numeric(data)){
        stop("The provided data is not a numeric matrix! Can't construct any model!", call.=FALSE);
    }
    if(is.data.frame(data)){
        data <- as.matrix(data);
    }

    # Number of series in the matrix
    nSeries <- ncol(data);

    if(is.null(ncol(data))){
        stop("The provided data is not a matrix! Use es() function instead!", call.=FALSE);
    }
    if(ncol(data)==1){
        stop("The provided data contains only one column. Use es() function instead!", call.=FALSE);
    }
    # Check the data for NAs
    if(any(is.na(data))){
        if(!silentText){
            warning("Data contains NAs. These observations will be substituted by zeroes.", call.=FALSE);
        }
        data[is.na(data)] <- 0;
    }

    # Define obs, the number of observations of in-sample
    obsInsample <- nrow(data) - holdout*h;

    # Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- nrow(data) + (1 - holdout)*h;

    # If obsInsample is negative, this means that we can't do anything...
    if(obsInsample<=0){
        stop("Not enough observations in sample.", call.=FALSE);
    }
    # Define the actual values. Transpose the matrix!
    y <- matrix(data[1:obsInsample,],nSeries,obsInsample,byrow=TRUE);
    datafreq <- frequency(data);

    ##### model for VES #####
    if(modelType=="ves"){
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
        else{
            message(paste0("You have defined a strange model: ",model));
            sowhat(model);
            message("Switching to 'ZZZ'");
            model <- "ZZZ";

            Etype <- "Z";
            Ttype <- "Z";
            Stype <- "Z";
            damped <- TRUE;
        }

        #### Check error type ####
        if(all(Etype!=c("Z","X","Y","A","M","C"))){
            warning(paste0("Wrong error type: ",Etype,". Should be 'Z', 'X', 'Y', 'A' or 'M'.\n",
                           "Changing to 'Z'"),call.=FALSE);
            Etype <- "Z";
        }

        #### Check trend type ####
        if(all(Ttype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong trend type: ",Ttype,". Should be 'Z', 'X', 'Y', 'N', 'A' or 'M'.\n",
                           "Changing to 'Z'"),call.=FALSE);
            Ttype <- "Z";
        }

        #### Check seasonality type ####
        # Check if the data is ts-object
        if(!is.ts(data) & Stype!="N"){
            if(!silentText){
                message("The provided data is not ts object. Only non-seasonal models are available.");
            }
            Stype <- "N";
            substr(model,nchar(model),nchar(model)) <- "N";
        }

        # Check if seasonality makes sense
        if(all(Stype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong seasonality type: ",Stype,". Should be 'Z', 'X', 'Y', 'N', 'A' or 'M'.",
                           "Setting to 'Z'."),call.=FALSE);
            if(datafreq==1){
                Stype <- "N";
            }
            else{
                Stype <- "Z";
            }
        }
        if(all(Stype!="N",datafreq==1)){
            if(all(Stype!=c("Z","X","Y"))){
                warning(paste0("Cannot build the seasonal model on data with frequency 1.\n",
                               "Switching to non-seasonal model: ETS(",substring(model,1,nchar(model)-1),"N)"));
            }
            Stype <- "N";
        }

        if(Stype=="N"){
            initialSeason <- NULL;
            modelIsSeasonal <- FALSE;
        }
        else{
            modelIsSeasonal <- FALSE;
        }

        if(any(c(Etype,Ttype,Stype)=="Z")){
            stop("Sorry we don't do model selection for VES yet.", call.=FALSE);
        }

        maxlag <- datafreq * modelIsSeasonal + 1 * (!modelIsSeasonal);

        # Define the number of rows that should be in the matvt
        obsStates <- max(obsAll + maxlag, obsInsample + 2*maxlag);

        nComponentsNonSeasonal <- 1 + (Ttype!="N")*1;
        nComponentsAll <- nComponentsNonSeasonal + modelIsSeasonal*1;
    }

    #This is the estimation of covariance matrix
    nParamMax <- 1;

    ##### persistence ####
    # persistence type can be: "i" - independent, "d" - dependent, "g" - group.
    persistenceValue <- persistence;
    if(is.null(persistenceValue)){
        if(silentText){
            message("persistence value is not selected. Switching to group.");
        }
        persistenceType <- "g";
        persistenceEstimate <- TRUE;
    }
    else{
        if(is.character(persistenceValue)){
            persistenceValue <- substring(persistenceValue[1],1,1);
            if(all(persistenceValue!=c("g","i","d"))){
                warning("You asked for a strange persistence value. We don't do that here. Switching to group",
                        call.=FALSE);
                persistenceType <- "g";
            }
            else{
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
                persistenceType <- "g";
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
                }
                else{
                    persistenceValue <- matrix(persistenceValue,nSeries*nComponentsAll,nSeries);
                }
            }
        }
        else if(!is.numeric(persistenceValue)){
            warning(paste0("persistence matrix is not numeric!\n",
                           "Values of persistence matrix will be estimated as group."),call.=FALSE);
            persistenceValue <- NULL;
            persistenceType <- "g";
            persistenceEstimate <- TRUE;
        }
    }

    if(any(persistenceType==c("g","i"))){
        # Whether individual or group, this thing reduces number of degrees of freedom in the same way.
        nParamMax <- nParamMax + nComponentsAll;
    }
    else if(persistenceType=="d"){
        # In case with "dependent" the whol matrix needs to be estimated
        nParamMax <- nParamMax + nComponentsAll*nSeries;
    }

    ##### transition ####
    # transition type can be: "i" - independent, "d" - dependent, "g" - group.
    transitionValue <- transition;
    if(is.null(transitionValue)){
        if(silentText){
            message("transition value is not selected. Switching to group");
        }
        transitionType <- "g";
        transitionEstimate <- FALSE;
    }
    else{
        if(is.character(transitionValue)){
            transitionValue <- substring(transitionValue[1],1,1);
            if(all(transitionValue!=c("g","i","d"))){
                warning("You asked for a strange transition value. We don't do that here. Switching to group",
                        call.=FALSE);
                transitionType <- "g";
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
                               "Values of transition matrix will be estimated as a group."),call.=FALSE);
                transitionValue <- NULL;
                transitionType <- "g";
                transitionEstimate <- FALSE;
            }
            else{
                transitionType <- "p";
                transitionEstimate <- FALSE;
                if(length(transitionValue) == nComponentsAll^2){
                    transitionValue <- matrix(transitionValue,nComponentsAll,nComponentsAll);
                    transitionBuffer <- diag(nSeries*nComponentsAll);
                    for(i in 1:nSeries){
                        transitionBuffer[c(1:nComponentsAll)+nComponentsAll*(i-1),c(1:nComponentsAll)+nComponentsAll*(i-1)] <- transitionValue;
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
                           "Values of transition vector will be estimated as a group."),call.=FALSE);
            transitionValue <- NULL;
            transitionType <- "g";
            transitionEstimate <- FALSE;
        }
    }

    if(transitionType=="d"){
        transitionEstimate <- TRUE;
        # Each separate transition matrix is not evaluated, but the left spaces are...
        nParamMax <- nParamMax + nSeries*nComponentsAll - nComponentsAll^2;
    }

    ##### Damping parameter ####
    # phi type can be: "i" - individual, "g" - group.
    dampedValue <- phi;
    if(is.null(dampedValue)){
        if(silentText){
            message("phi value is not selected. Switching to group");
        }
        dampedType <- "g";
        dampedEstimate <- TRUE;
    }
    else{
        if(is.character(dampedValue)){
            dampedValue <- substring(dampedValue[1],1,1);
            if(all(dampedValue!=c("i","g"))){
                warning("You asked for a strange phi value. We don't do that here. Switching to group.",
                        call.=FALSE);
                dampedType <- "g";
            }
            else{
                dampedType <- dampedValue;
            }
            dampedValue <- NULL;
            dampedEstimate <- TRUE;
        }
        else if(is.numeric(dampedValue)){
            if((length(dampedValue) != nSeries) & (length(dampedValue)!= 1)){
                warning(paste0("Length of phi vector is wrong! It should be ",
                               nSeries,
                               " instead of ",length(dampedValue),".\n",
                               "Values of phi vector will be estimated as a group."),call.=FALSE);
                dampedValue <- NULL;
                dampedType <- "g";
                dampedEstimate <- TRUE;
            }
            else{
                dampedType <- "p";
                dampedValue <- matrix(dampedValue,nSeries,1);
                dampedEstimate <- FALSE;
            }
        }
        else if(!is.numeric(dampedValue)){
            warning(paste0("phi vector is not numeric!\n",
                           "Values of phi vector will be estimated as a group."),call.=FALSE);
            dampedValue <- NULL;
            dampedType <- "g";
            dampedEstimate <- TRUE;
        }
    }

    if(any(dampedType==c("g","i"))){
        dampedValue <- matrix(1,nSeries,1);
        # Whether group or individual the effect on df is the same.
        nParamMax <- nParamMax + 1;
    }

    ##### initials ####
    # initial type can be: "i" - individual, "g" - group.
    initialValue <- initial;
    if(is.null(initialValue)){
        if(silentText){
            message("Initial value is not selected. Switching to individual.");
        }
        initialType <- "i";
        initialEstimate <- TRUE;
    }
    else{
        if(is.character(initialValue)){
            initialValue <- substring(initialValue[1],1,1);
            if(all(initialValue!=c("i","g"))){
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
            if(modelType=="ves"){
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

    if(any(initialType==c("g","i"))){
        nParamMax <- nParamMax + nComponentsNonSeasonal;
    }

    if(modelType=="ves"){
    ##### initialSeason for VES #####
    # Here we should check if initialSeason is character or not...
    # if length(initialSeason) == datafreq*nSeries, then ok
    # if length(initialSeason) == datafreq, then use it for all nSeries
        initialSeasonValue <- initialSeason;
        if(is.null(initialSeasonValue)){
            if(silentText){
                message("Initial value is not selected. Switching to group.");
            }
            initialSeasonType <- "g";
            initialSeasonEstimate <- TRUE;
        }
        else{
            if(is.character(initialSeasonValue)){
                initialSeasonValue <- substring(initialSeasonValue[1],1,1);
                if(all(initialSeasonValue!=c("i","g"))){
                    warning("You asked for a strange initialSeason value. We don't do that here. Switching to group.",
                            call.=FALSE);
                    initialSeasonType <- "g";
                }
                else{
                    initialSeasonType <- initialSeasonValue;
                }
                initialSeasonValue <- NULL;
                initialSeasonEstimate <- TRUE;
            }
            else if(is.numeric(initialSeasonValue)){
                if(modelType=="ves"){
                    if(all(length(initialSeasonValue)!=c(datafreq,datafreq*nSeries))){
                        warning(paste0("The length of initialSeason is wrong! It should correspond to the frequency of the data.",
                                       "Values of initialSeason will be estimated as a group."),call.=FALSE);
                        initialSeasonValue <- NULL;
                        initialSeasonType <- "g";
                        initialSeasonEstimate <- TRUE;
                    }
                    else{
                        initialSeasonValue <- matrix(initialSeasonValue,nSeries,datafreq);
                        initialSeasonType <- "p";
                        initialSeasonEstimate <- FALSE;
                    }
                }
            }
            else if(!is.numeric(initialSeasonValue)){
                warning(paste0("Initial vector is not numeric!\n",
                               "Values of initialSeason vector will be estimated as a group."),call.=FALSE);
                initialSeasonValue <- NULL;
                initialSeasonType <- "g";
                initialSeasonEstimate <- TRUE;
            }
        }

        if(any(initialSeasonType==c("g","i"))){
            nParamMax <- nParamMax + datafreq;
        }
    }

    ##### Cost function type #####
    cfType <- cfType[1];
    if(any(cfType==c("GMSTFE","MSTFE","TFL","MSEh","aGMSTFE","aMSTFE","aTFL","aMSEh"))){
        multisteps <- TRUE;
    }
    else if(any(cfType==c("MSE","MAE","HAM"))){
        multisteps <- FALSE;
    }
    else{
        warning(paste0("Strange cost function specified: ",cfType,". Switching to 'MSE'."),call.=FALSE);
        cfType <- "MSE";
        multisteps <- FALSE;
    }
    cfTypeOriginal <- cfType;

    normalizer <- sum(colMeans(abs(diff(c(y))),na.rm=TRUE));

    ##### Information Criteria #####
    ic <- ic[1];
    if(all(ic!=c("AICc","AIC","BIC"))){
        warning(paste0("Strange type of information criteria defined: ",ic,". Switching to 'AICc'."),call.=FALSE);
        ic <- "AICc";
    }

    ##### intervals, intervalsType, level #####
    intervalsType <- intervals[1];
    # Check the provided type of interval

    if(is.logical(intervalsType)){
        if(intervalsType){
            intervalsType <- "p";
        }
        else{
            intervalsType <- "none";
        }
    }

    if(all(intervalsType!=c("p","s","n","a","sp","np","none","parametric","semiparametric","nonparametric"))){
        warning(paste0("Wrong type of interval: '",intervalsType, "'. Switching to 'parametric'."),call.=FALSE);
        intervalsType <- "p";
    }

    if(intervalsType=="none"){
        intervalsType <- "n";
        intervals <- FALSE;
    }
    else if(intervalsType=="parametric"){
        intervalsType <- "p";
        intervals <- TRUE;
    }
    else if(intervalsType=="semiparametric"){
        intervalsType <- "sp";
        intervals <- TRUE;
    }
    else if(intervalsType=="nonparametric"){
        intervalsType <- "np";
        intervals <- TRUE;
    }
    else{
        intervals <- TRUE;
    }

    if(level>1){
        level <- level / 100;
    }

    ##### intermittent #####
    ##### !!!!! THIS WILL BE FIXED WHEN WE KNOW HOW TO DO THAT
    intermittent <- substring(intermittent[1],1,1);

    ##### Check if multiplicative is applicable #####
    if(any(modelType==c("ves"))){
        # Check if multiplicative models can be fitted
        allowMultiplicative <- !((any(y<=0) & intermittent=="n")| (intermittent!="n" & any(y<0)));
        # If non-positive values are present, check if data is intermittent, if negatives are here, switch to additive models
        if(!allowMultiplicative){
            if(Etype=="M"){
                warning("Can't apply multiplicative model to non-positive data. Switching error type to 'A'", call.=FALSE);
                Etype <- "A";
            }
            if(Ttype=="M"){
                warning("Can't apply multiplicative model to non-positive data. Switching trend type to 'A'", call.=FALSE);
                Ttype <- "A";
            }
            if(Stype=="M"){
                warning("Can't apply multiplicative model to non-positive data. Switching seasonality type to 'A'", call.=FALSE);
                Stype <- "A";
            }
        }
    }

    ##### bounds #####
    bounds <- substring(bounds[1],1,1);
    if(bounds!="u" & bounds!="a" & bounds!="n"){
        warning("Strange bounds are defined. Switching to 'usual'.",call.=FALSE);
        bounds <- "u";
    }

    ##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                    nParamMax," while the number of observations is ",obsNonzero,"."),call.=FALSE);
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
            warning("Sorry, but you don't have 'numDeriv' package, which is required in order to produce Fisher Information.",call.=FALSE);
            FI <- FALSE;
        }
    }

    ##### Return values to previous environment #####
    assign("h",h,ParentEnvironment);
    assign("silentText",silentText,ParentEnvironment);
    assign("silentGraph",silentGraph,ParentEnvironment);
    assign("silentLegend",silentLegend,ParentEnvironment);
    assign("obsInsample",obsInsample,ParentEnvironment);
    assign("obsAll",obsAll,ParentEnvironment);
    assign("obsStates",obsStates,ParentEnvironment);
    assign("obsNonzero",obsNonzero,ParentEnvironment);
    assign("nSeries",nSeries,ParentEnvironment);
    assign("nParamMax",nParamMax,ParentEnvironment);
    assign("data",data,ParentEnvironment);
    assign("y",y,ParentEnvironment);
    assign("datafreq",datafreq,ParentEnvironment);

    assign("model",model,ParentEnvironment);
    # assign("modelsPool",modelsPool,ParentEnvironment);
    assign("Etype",Etype,ParentEnvironment);
    assign("Ttype",Ttype,ParentEnvironment);
    assign("Stype",Stype,ParentEnvironment);
    assign("maxlag",maxlag,ParentEnvironment);
    assign("modelIsSeasonal",modelIsSeasonal,ParentEnvironment);
    assign("nComponentsAll",nComponentsAll,ParentEnvironment);
    assign("nComponentsNonSeasonal",nComponentsNonSeasonal,ParentEnvironment);
    assign("allowMultiplicative",allowMultiplicative,ParentEnvironment);

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

    assign("cfType",cfType,ParentEnvironment);
    assign("cfTypeOriginal",cfTypeOriginal,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("normalizer",normalizer,ParentEnvironment);

    assign("ic",ic,ParentEnvironment);

    assign("intervalsType",intervalsType,ParentEnvironment);
    assign("intervals",intervals,ParentEnvironment);

    assign("intermittent",intermittent,ParentEnvironment);
    # !!!!! THIS WILL BE FIXED WHEN WE KNOW HOW TO DO THAT
    # assign("ot",ot,ParentEnvironment);
    # assign("yot",yot,ParentEnvironment);
    # assign("pt",pt,ParentEnvironment);
    # assign("pt.for",pt.for,ParentEnvironment);
    # assign("nParamIntermittent",nParamIntermittent,ParentEnvironment);
    # assign("iprob",iprob,ParentEnvironment);

    assign("bounds",bounds,ParentEnvironment);

    assign("nParamMax",nParamMax,ParentEnvironment);

    assign("FI",FI,ParentEnvironment);
}
