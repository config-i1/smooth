utils::globalVariables(c("initialSeason","persistence","modelsPool","modelDo"));

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
    # Define the actual values
    y <- matrix(data[1:obsInsample,],obsInsample,nSeries);
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

        if(any(c(Etype,Ttype,Stype)=="Z")){
            stop("Sorry we don't do model selection for VES yet.", call.=FALSE);
        }

        maxlag <- datafreq * (Stype!="N") + 1 * (Stype=="N");

        # Define the number of rows that should be in the matvt
        obsStates <- max(obsAll + maxlag, obsInsample + 2*maxlag);

        nComponents <- 1 + (Ttype!="N")*1 + (Stype!="N")*1;
    }

    ##### persistence ####
    # persistence type can be: "i" - individual, "g" - group.
    persistenceValue <- persistence;
    if(is.null(persistenceValue)){
        if(silentText){
            message("persistence value is not selected. Switching to individual.");
        }
        persistenceType <- "i";
        persistenceEstimate <- TRUE;
    }
    else{
        if(is.character(persistenceValue)){
            persistenceValue <- substring(persistenceValue[1],1,1);
            if(all(persistenceValue!=c("i","g"))){
                warning("You asked for a strange persistence value. We don't do that here. Switching to individual.",
                        call.=FALSE);
                persistenceType <- "i";
            }
            else{
                persistenceType <- persistenceValue;
            }
            persistenceValue <- NULL;
            persistenceEstimate <- TRUE;
        }
        else if(is.numeric(persistenceValue)){
            if(length(persistenceValue) != nSeries^2){
                warning(paste0("Length of persistence matrix is wrong! It should be ",
                               nSeries^2,
                               " instead of ",length(persistenceValue),".\n",
                               "Values of persistence matrix will be estimated."),call.=FALSE);
                persistenceValue <- NULL;
                persistenceType <- "i";
                persistenceEstimate <- TRUE;
            }
            else{
                persistenceType <- "p";
                persistenceValue <- matrix(persistence,nSeries,nSeries);
                persistenceEstimate <- FALSE;
            }
        }
        else if(!is.numeric(persistenceValue)){
            warning(paste0("persistence matrix is not numeric!\n",
                           "Values of persistence matrix will be estimated."),call.=FALSE);
            persistenceValue <- NULL;
            persistenceType <- "i";
            persistenceEstimate <- TRUE;
        }
    }

    ##### transition ####
    # transition type can be: "i" - individual, "g" - group.
    transitionValue <- transition;
    if(is.null(transitionValue)){
        if(silentText){
            message("transition value is not selected. Switching to individual.");
        }
        transitionType <- "i";
        transitionEstimate <- TRUE;
    }
    else{
        if(is.character(transitionValue)){
            transitionValue <- substring(transitionValue[1],1,1);
            if(all(transitionValue!=c("i","g"))){
                warning("You asked for a strange transition value. We don't do that here. Switching to individual.",
                        call.=FALSE);
                transitionType <- "i";
            }
            else{
                transitionType <- transitionValue;
            }
            transitionValue <- NULL;
            transitionEstimate <- TRUE;
        }
        else if(is.numeric(transitionValue)){
            if(length(transitionValue) != (nSeries*nComponents)^2){
                warning(paste0("Length of transition matrix is wrong! It should be ",
                               (nSeries*nComponents)^2,
                               " instead of ",length(transitionValue),".\n",
                               "Values of transition matrix will be estimated."),call.=FALSE);
                transitionValue <- NULL;
                transitionType <- "i";
                transitionEstimate <- TRUE;
            }
            else{
                transitionType <- "p";
                transitionValue <- matrix(transition,nSeries*nComponents,nSeries*nComponents);
                transitionEstimate <- FALSE;
            }
        }
        else if(!is.numeric(transitionValue)){
            warning(paste0("transition vector is not numeric!\n",
                           "Values of transition vector will be estimated."),call.=FALSE);
            transitionValue <- NULL;
            transitionType <- "i";
            transitionEstimate <- TRUE;
        }
    }

    ##### measurement ####
    # measurement type can be: "i" - individual, "g" - group.
    measurementValue <- measurement;
    if(is.null(measurementValue)){
        if(silentText){
            message("measurement value is not selected. Switching to individual.");
        }
        measurementType <- "i";
        measurementEstimate <- TRUE;
    }
    else{
        if(is.character(measurementValue)){
            measurementValue <- substring(measurementValue[1],1,1);
            if(all(measurementValue!=c("i","g"))){
                warning("You asked for a strange measurement value. We don't do that here. Switching to individual.",
                        call.=FALSE);
                measurementType <- "i";
            }
            else{
                measurementType <- measurementValue;
            }
            measurementValue <- NULL;
            measurementEstimate <- TRUE;
        }
        else if(is.numeric(measurementValue)){
            if(length(measurementValue) != nSeries * (nSeries*nComponents)){
                warning(paste0("Length of measurement matrix is wrong! It should be ",
                               nSeries * (nSeries*nComponents),
                               " instead of ",length(measurementValue),".\n",
                               "Values of measurement matrix will be estimated."),call.=FALSE);
                measurementValue <- NULL;
                measurementType <- "i";
                measurementEstimate <- TRUE;
            }
            else{
                measurementType <- "p";
                measurementValue <- matrix(measurement,nSeries,nSeries*nComponents);
                measurementEstimate <- FALSE;
            }
        }
        else if(!is.numeric(measurementValue)){
            warning(paste0("measurement vector is not numeric!\n",
                           "Values of measurement vector will be estimated."),call.=FALSE);
            measurementValue <- NULL;
            measurementType <- "i";
            measurementEstimate <- TRUE;
        }
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
                    if(length(initialValue) != (1*(Ttype!="N") + 1) * nSeries){
                        warning(paste0("Length of initial vector is wrong! It should be ",
                                       (1*(Ttype!="N") + 1)*nSeries,
                                       " instead of ",length(initialValue),".\n",
                                       "Values of initial vector will be estimated."),call.=FALSE);
                        initialValue <- NULL;
                        initialType <- "i";
                        initialEstimate <- TRUE;
                    }
                    else{
                        initialType <- "p";
                        initialValue <- matrix(initial,(1*(Ttype!="N") + 1) * nSeries,1);
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

    if(modelType=="ves"){
    ##### initialSeason for VES #####
    # Here we should check if initialSeason is character or not...
    # if length(initialSeason) == datafreq*nSeries, then ok
    # if length(initialSeason) == datafreq, then use it for all nSeries
        initialSeasonValue <- initialSeason;
        if(is.null(initialSeasonValue)){
            if(silentText){
                message("Initial value is not selected. Switching to individual.");
            }
            initialSeasonType <- "i";
            initialSeasonEstimate <- TRUE;
        }
        else{
            if(is.character(initialSeasonValue)){
                initialSeasonValue <- substring(initialSeasonValue[1],1,1);
                if(all(initialSeasonValue!=c("i","g"))){
                    warning("You asked for a strange initialSeason value. We don't do that here. Switching to individual.",
                            call.=FALSE);
                    initialSeasonType <- "i";
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
                                       "Values of initialSeason will be estimated."),call.=FALSE);
                        initialSeasonValue <- NULL;
                        initialSeasonType <- "i";
                        initialSeasonEstimate <- TRUE;
                    }
                    else{
                        initialSeasonValue <- matrix(initialSeasonValue,nSeries,datafreq);
                        initialSeasonEstimate <- FALSE;
                    }
                }
            }
            else if(!is.numeric(initialSeasonValue)){
                warning(paste0("Initial vector is not numeric!\n",
                               "Values of initialSeason vector will be estimated."),call.=FALSE);
                initialSeasonValue <- NULL;
                initialSeasonType <- "i";
                initialSeasonEstimate <- TRUE;
            }
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

    normalizer <- colMeans(abs(diff(c(y))),na.rm=TRUE);

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

    ##### Define xregDo #####
    if(!exists("xregDo")){
        xregDo <- "u";
    }
    else{
        if(!any(xregDo==c("use","select","u","s"))){
            warning("Wrong type of xregDo parameter. Changing to 'select'.", call.=FALSE);
            xregDo <- "select";
        }
    }
    xregDo <- substr(xregDo[1],1,1);

    if(is.null(xreg)){
        xregDo <- "u";
    }

    ##### Calculate nParamMax for checks #####
    if(modelType=="ves"){
        # 1 - 3: persitence vector;
        # 1 - 2: initials;
        # 1 - 1 phi value;
        # datafreq: datafreq initials for seasonal component;
        # 1: estimation of variance;
        nParamMax <- (1 + (Ttype!="N") + (Stype!="N"))*persistenceEstimate +
            (1 + (Ttype!="N"))*(initialType=="o") +
            phiEstimate*damped + datafreq*(Stype!="N")*initialSeasonEstimate*(initialType=="o") + 1;
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
    assign("data",data,ParentEnvironment);
    assign("y",y,ParentEnvironment);
    assign("datafreq",datafreq,ParentEnvironment);

    assign("model",model,ParentEnvironment);
    assign("modelsPool",modelsPool,ParentEnvironment);
    assign("Etype",Etype,ParentEnvironment);
    assign("Ttype",Ttype,ParentEnvironment);
    assign("Stype",Stype,ParentEnvironment);
    assign("damped",damped,ParentEnvironment);
    assign("modelDo",modelDo,ParentEnvironment);
    assign("nComponents",nComponents,ParentEnvironment);
    assign("allowMultiplicative",allowMultiplicative,ParentEnvironment);

    assign("persistenceValue",persistenceValue,ParentEnvironment);
    assign("persistenceType",persistenceType,ParentEnvironment);
    assign("persistenceEstimate",persistenceEstimate,ParentEnvironment);

    assign("transitionValue",transitionValue,ParentEnvironment);
    assign("transitionType",transitionType,ParentEnvironment);
    assign("transitionEstimate",transitionEstimate,ParentEnvironment);

    assign("measurementValue",measurementValue,ParentEnvironment);
    assign("measurementType",measurementType,ParentEnvironment);
    assign("measurementEstimate",measurementEstimate,ParentEnvironment);

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
    assign("xregDo",xregDo,ParentEnvironment);

    assign("nParamMax",nParamMax,ParentEnvironment);

    assign("FI",FI,ParentEnvironment);
}
