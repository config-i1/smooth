utils::globalVariables(c("initialSeason"));

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

    if(modelType=="ves"){
        ##### model for VES #####
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
    }

    ##### initials ####
    # initial type can be: "i" - individual, "g" - group.
    initialValue <- initial;
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
    else if(is.null(initialValue)){
        if(silentText){
            message("Initial value is not selected. Switching to individual.");
        }
        initialType <- "i";
        initialEstimate <- TRUE;
    }
    else if(!is.null(initialValue)){
        if(!is.numeric(initialValue)){
            warning(paste0("Initial vector is not numeric!\n",
                           "Values of initial vector will be estimated."),call.=FALSE);
            initialValue <- NULL;
            initialType <- "i";
            initialEstimate <- TRUE;
        }
        else{
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
                        initialValue <- initial;
                        initialEstimate <- FALSE;
                    }
                }
            }
        }
    }

    ##### initialSeason for VES #####
    # Here we should check if initialSeason is character or not...
    # if length(initialSeason) == datafreq*nSeries, then ok
    # if length(initialSeason) == datafreq, then use it for all nSeries
    initialSeasonValue <- initialSeason;
    if(is.character(initialSeasonValue)){
        initialSeasonValue <- substring(initialSeasonValue[1],1,1);
        if(all(initialSeasonValue!=c("i","g"))){
            warning("You asked for a strange initial value. We don't do that here. Switching to individual.",
                    call.=FALSE);
            initialSeasonType <- "i";
        }
        else{
            initialSeasonType <- initialSeasonValue;
        }
        initialSeasonValue <- NULL;
        initialSeasonEstimate <- TRUE;
    }
    else if(is.null(initialSeasonValue)){
        if(silentText){
            message("Initial value is not selected. Switching to individual.");
        }
        initialSeasonType <- "i";
        initialSeasonEstimate <- TRUE;
    }
    else if(!is.null(initialSeasonValue)){
        if(!is.numeric(initialSeasonValue)){
            warning(paste0("Initial vector is not numeric!\n",
                           "Values of initial vector will be estimated."),call.=FALSE);
            initialSeasonValue <- NULL;
            initialSeasonType <- "i";
            initialSeasonEstimate <- TRUE;
        }
        else{
            if(modelType=="ves"){
                if(length(initialSeasonValue)>2*nSeries){
                    warning(paste0("Length of initial vector is wrong! It should not be greater than",
                                   2*nSeries,"\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialSeasonValue <- NULL;
                    initialSeasonType <- "i";
                    initialSeasonEstimate <- TRUE;
                }
                else{
                    if(length(initialSeasonValue) != (1*(Ttype!="N") + 1) * nSeries){
                        warning(paste0("Length of initial vector is wrong! It should be ",
                                       (1*(Ttype!="N") + 1)*nSeries,
                                       " instead of ",length(initialSeasonValue),".\n",
                                       "Values of initial vector will be estimated."),call.=FALSE);
                        initialSeasonValue <- NULL;
                        initialSeasonType <- "i";
                        initialSeasonEstimate <- TRUE;
                    }
                    else{
                        initialSeasonType <- "p";
                        initialSeasonValue <- initial;
                        initialSeasonEstimate <- FALSE;
                    }
                }
            }
        }
    }


    if(!is.null(initialSeasonValue)){
        if(!is.numeric(initialSeasonValue)){
            warning(paste0("InitialSeason is not numeric!\n",
                           "Values of initialSeason will be estimated."),call.=FALSE);
            initialSeasonValue <- NULL;
            initialSeasonEstimate <- TRUE;
            initialSeasonType <- "i";
        }
        else{
            if(all(length(initialSeasonValue)!=c(datafreq,datafreq*nSeries))){
                warning(paste0("The length of initialSeason is wrong! It should correspond to the frequency of the data.",
                               "Values of initialSeason will be estimated."),call.=FALSE);
                initialSeasonValue <- NULL;
                initialSeasonEstimate <- TRUE
            }
            else{
                initialSeasonValue <- matrix(initialSeasonValue,datafreq,nSeries);
                initialSeasonEstimate <- FALSE;
            }
        }
    }
    else{
        initialSeasonEstimate <- TRUE;
    }
}
