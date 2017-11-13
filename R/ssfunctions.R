utils::globalVariables(c("h","holdout","orders","lags","transition","measurement","multisteps","ot","obsInsample","obsAll",
                         "obsStates","obsNonzero","pt","cfType","CF","Etype","Ttype","Stype","matxt","matFX","vecgX","xreg",
                         "matvt","nExovars","matat","errors","nParam","intervals","intervalsType","level","ivar","model",
                         "constant","AR","MA","data","y.fit","cumulative","rounded"));

##### *Checker of input of basic functions* #####
ssInput <- function(smoothType=c("es","ges","ces","ssarima"),...){
    # This is universal function needed in order to check the passed arguments to es(), ges(), ces() and ssarima()

    smoothType <- smoothType[1];

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

    # Check what was passed as a horizon
    if(h<=0){
        warning(paste0("You have set forecast horizon equal to ",h,". We hope you know, what you are doing."), call.=FALSE);
        if(h<0){
            warning("And by the way, we can't do anything with negative horizon, so we will set it equal to zero.", call.=FALSE);
            h <- 0;
        }
    }

    ##### data #####
    if(any(class(data)=="smooth.sim")){
        data <- data$data;
    }
    if(!is.numeric(data)){
        stop("The provided data is not a vector or ts object! Can't construct any model!", call.=FALSE);
    }
    if(!is.null(ncol(data))){
        if(ncol(data)>1){
            stop("The provided data is not a vector! Can't construct any model!", call.=FALSE);
        }
    }
    # Check the data for NAs
    if(any(is.na(data))){
        if(!silentText){
            warning("Data contains NAs. These observations will be substituted by zeroes.",call.=FALSE);
        }
        data[is.na(data)] <- 0;
    }

    # Define obs, the number of observations of in-sample
    obsInsample <- length(data) - holdout*h;

    # Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- length(data) + (1 - holdout)*h;

    # If obsInsample is negative, this means that we can't do anything...
    if(obsInsample<=0){
        stop("Not enough observations in sample.",call.=FALSE);
    }
    # Define the actual values
    y <- matrix(data[1:obsInsample],obsInsample,1);
    datafreq <- frequency(data);

    # Number of parameters to estimate / provided
    parametersNumber <- matrix(0,2,4,
                               dimnames=list(c("Estimated","Provided"),
                                             c("nParamInternal","nParamXreg","nParamIntermittent","nParamAll")));

    if(smoothType=="es"){
        ##### model for ES #####
        if(!is.character(model)){
            stop(paste0("Something strange is provided instead of character object in model: ",
                        paste0(model,collapse=",")),call.=FALSE);
        }

        # Predefine models pool for a model selection
        modelsPool <- NULL;
        # Deal with the list of models. Check what has been provided. Stop if there is a mistake.
        if(length(model)>1){
            if(any(nchar(model)>4)){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)>4],collapse=",")),call.=FALSE);
            }
            else if(any(substr(model,1,1)!="A" & substr(model,1,1)!="M" & substr(model,1,1)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,1,1)!="A" & substr(model,1,1)!="M"],collapse=",")),call.=FALSE);
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
                                             substr(model,3,3)!="M" & substr(model,3,3)!="d"],collapse=",")),call.=FALSE);
            }
            else if(any(nchar(model)==4 & substr(model,4,4)!="N" & substr(model,4,4)!="A" &
                        substr(model,4,4)!="M" & substr(model,4,4)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)==4 & substr(model,4,4)!="N" &
                                             substr(model,4,4)!="A" & substr(model,4,4)!="M"],collapse=",")),call.=FALSE);
            }
            else{
                modelsPoolCombiner <- (substr(model,1,1)=="C" | substr(model,2,2)=="C" |
                                         substr(model,3,3)=="C" | substr(model,4,4)=="C");
                modelsPool <- model[!modelsPoolCombiner];
                if(any(modelsPoolCombiner)){
                    model <- "CCC";
                }
                else{
                    model <- "ZZZ";
                }
            }
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
            else if(any(unlist(strsplit(model,""))=="Z") |
                    any(unlist(strsplit(model,""))=="X") |
                    any(unlist(strsplit(model,""))=="Y")){
                modelDo <- "select";
            }
            else{
                modelDo <- "estimate";
            }

            if(any(unlist(strsplit(model,""))=="X") | any(unlist(strsplit(model,""))=="Y")){
                modelsPool <- c("ANN","MNN","AAN","AMN","MAN","MMN","AAdN","AMdN","MAdN","MMdN","ANA","ANM","MNA","MNM",
                                 "AAA","AAM","AMA","AMM","MAA","MAM","MMA","MMM",
                                 "AAdA","AAdM","AMdA","AMdM","MAdA","MAdM","MMdA","MMdM");
                if(datafreq==1 & Stype!="N"){
                    if(!silentText){
                        warning("The provided data has frequency of 1. Only non-seasonal models are available.",call.=FALSE);
                    }
                    Stype <- "N";
                    substr(model,nchar(model),nchar(model)) <- "N";
                }

                if((obsInsample < datafreq*2) & Stype!="N"){
                    warning("Sorry, but we don't have enough data for the seasonal model. Switching to non-seasonal.",call.=FALSE);
                    Stype <- "N";
                }
                # Restrict error types in the pool
                if(Etype=="X"){
                    modelsPool <- modelsPool[substr(modelsPool,1,1)=="A"];
                    Etype <- "Z";
                }
                else if(Etype=="Y"){
                    modelsPool <- modelsPool[substr(modelsPool,1,1)=="M"];
                    Etype <- "Z";
                }
                else{
                    if(Etype!="Z"){
                        modelsPool <- modelsPool[substr(modelsPool,1,1)==Etype];
                    }
                }
                # Restrict trend types in the pool
                if(Ttype=="X"){
                    modelsPool <- modelsPool[substr(modelsPool,2,2)=="A" | substr(modelsPool,2,2)=="N"];
                    Ttype <- "Z";
                }
                else if(Ttype=="Y"){
                    modelsPool <- modelsPool[substr(modelsPool,2,2)=="M" | substr(modelsPool,2,2)=="N"];
                    Ttype <- "Z";
                }
                else{
                    if(Ttype!="Z"){
                        modelsPool <- modelsPool[substr(modelsPool,2,2)==Ttype];
                        if(damped){
                            modelsPool <- modelsPool[nchar(modelsPool)==4];
                        }
                    }
                }
                # Restrict season types in the pool
                if(Stype=="X"){
                    modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="A" |
                                               substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N" ];
                    Stype <- "Z";
                }
                else if(Stype=="Y"){
                    modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="M" |
                                               substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N" ];
                    Stype <- "Z";
                }
                else{
                    if(Stype!="Z"){
                        modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))==Stype];
                    }
                }
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

        ### Check error type
        if(all(Etype!=c("Z","X","Y","A","M","C"))){
            warning(paste0("Wrong error type: ",Etype,". Should be 'Z', 'X', 'Y', 'A' or 'M'.\n",
                           "Changing to 'Z'"),call.=FALSE);
            Etype <- "Z";
        }

        ### Check trend type
        if(all(Ttype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong trend type: ",Ttype,". Should be 'Z', 'X', 'Y', 'N', 'A' or 'M'.\n",
                           "Changing to 'Z'"),call.=FALSE);
            Ttype <- "Z";
        }
    }
    else if(smoothType=="ssarima"){
        ##### Orders and lags for ssarima #####
        if(any(is.complex(c(ar.orders,i.orders,ma.orders,lags)))){
            stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
        }

        if(any(c(ar.orders,i.orders,ma.orders)<0)){
            stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
        }

        if(any(c(lags)<0)){
            stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
        }

        # If there are zero lags, drop them
        if(any(lags==0)){
            ar.orders <- ar.orders[lags!=0];
            i.orders <- i.orders[lags!=0];
            ma.orders <- ma.orders[lags!=0];
            lags <- lags[lags!=0];
        }

        # Define maxorder and make all the values look similar (for the polynomials)
        maxorder <- max(length(ar.orders),length(i.orders),length(ma.orders));
        if(length(ar.orders)!=maxorder){
            ar.orders <- c(ar.orders,rep(0,maxorder-length(ar.orders)));
        }
        if(length(i.orders)!=maxorder){
            i.orders <- c(i.orders,rep(0,maxorder-length(i.orders)));
        }
        if(length(ma.orders)!=maxorder){
            ma.orders <- c(ma.orders,rep(0,maxorder-length(ma.orders)));
        }

        if((length(lags)!=length(ar.orders)) & (length(lags)!=length(i.orders)) & (length(lags)!=length(ma.orders))){
            stop("Seasonal lags do not correspond to any element of SARIMA",call.=FALSE);
        }

        # If zeroes are defined for some orders, drop them.
        if(any((ar.orders + i.orders + ma.orders)==0)){
            orders2leave <- (ar.orders + i.orders + ma.orders)!=0;
            if(all(orders2leave==FALSE)){
                orders2leave <- lags==min(lags);
            }
            ar.orders <- ar.orders[orders2leave];
            i.orders <- i.orders[orders2leave];
            ma.orders <- ma.orders[orders2leave];
            lags <- lags[orders2leave];
        }

        # Get rid of duplicates in lags
        if(length(unique(lags))!=length(lags)){
            if(frequency(data)!=1){
                warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),
                               "). Getting rid of some of them."),call.=FALSE);
            }
            lags.new <- unique(lags);
            ar.orders.new <- i.orders.new <- ma.orders.new <- lags.new;
            for(i in 1:length(lags.new)){
                ar.orders.new[i] <- max(ar.orders[which(lags==lags.new[i])]);
                i.orders.new[i] <- max(i.orders[which(lags==lags.new[i])]);
                ma.orders.new[i] <- max(ma.orders[which(lags==lags.new[i])]);
            }
            ar.orders <- ar.orders.new;
            i.orders <- i.orders.new;
            ma.orders <- ma.orders.new;
            lags <- lags.new;
        }

        ARValue <- AR;
        # Check the provided AR matrix / vector
        if(!is.null(ARValue)){
            if((!is.numeric(ARValue) | !is.vector(ARValue)) & !is.matrix(ARValue)){
                warning(paste0("AR should be either vector or matrix. You have provided something strange...\n",
                               "AR will be estimated."),call.=FALSE);
                ARRequired <- AREstimate <- TRUE;
                ARValue <- NULL;
            }
            else{
                if(is.matrix(ARValue)){
                    ARLength <- length(ARValue[ARValue!=0]);
                }
                else{
                    ARLength <- length(ARValue);
                }
                if(sum(ar.orders)!=ARLength){
                    warning(paste0("Wrong number of non-zero elements of AR. Should be ",sum(ar.orders),
                                    " instead of ",length(ARValue[ARValue!=0]),".\n",
                                   "AR will be estimated."),call.=FALSE);
                    ARRequired <- AREstimate <- TRUE;
                    ARValue <- NULL;
                }
                else{
                    if(is.matrix(ARValue)){
                        ARValue <- ARValue[ARValue!=0];
                    }
                    AREstimate <- FALSE;
                    ARRequired <- TRUE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(ARValue);
                }
            }
        }
        else{
            if(all(ar.orders==0)){
                ARRequired <- AREstimate <- FALSE;
            }
            else{
                ARRequired <- AREstimate <- TRUE;
            }
        }

        MAValue <- MA;
        # Check the provided MA matrix / vector
        if(!is.null(MAValue)){
            if((!is.numeric(MAValue) | !is.vector(MAValue)) & !is.matrix(MAValue)){
                warning(paste0("MA should be either vector or matrix. You have provided something strange...\n",
                               "MA will be estimated."),call.=FALSE);
                MARequired <- MAEstimate <- TRUE;
                MAValue <- NULL;
            }
            else{
                if(is.matrix(MAValue)){
                    MALength <- length(MAValue[MAValue!=0]);
                }
                else{
                    MALength <- length(MAValue);
                }
                if(sum(ma.orders)!=MALength){
                    warning(paste0("Wrong number of non-zero elements of MA. Should be ",sum(ma.orders),
                                    " instead of ",length(MAValue[MAValue!=0]),".\n",
                                   "MA will be estimated."),call.=FALSE);
                    MARequired <- MAEstimate <- TRUE;
                    MAValue <- NULL;
                }
                else{
                    if(is.matrix(MAValue)){
                        MAValue <- MAValue[MAValue!=0];
                    }
                    MAEstimate <- FALSE;
                    MARequired <- TRUE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(MAValue);
                }
            }
        }
        else{
            if(all(ma.orders==0)){
                MARequired <- MAEstimate <- FALSE;
            }
            else{
                MARequired <- MAEstimate <- TRUE;
            }
        }

        constantValue <- constant;
        # Check the provided constant
        if(is.numeric(constantValue)){
            constantEstimate <- FALSE;
            constantRequired <- TRUE;
            parametersNumber[2,1] <- parametersNumber[2,1] + 1;
        }
        else if(is.logical(constantValue)){
            constantRequired <- constantEstimate <- constantValue;
            constantValue <- NULL;
        }

        # Number of components to use
        nComponents <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
        modellags <- matrix(rep(1,times=nComponents),ncol=1);
        if(constantRequired==TRUE){
            modellags <- rbind(modellags,1);
        }
        maxlag <- 1;

        if(obsInsample < nComponents){
            warning(paste0("In-sample size is ",obsInsample,", while number of components is ",nComponents,
                           ". Cannot fit the model."),call.=FALSE)
            stop("Not enough observations for such a complicated model.",call.=FALSE);
        }
    }
    else if(smoothType=="ces"){
        # If the user typed wrong seasonality, use the "Full" instead
        if(all(seasonality!=c("n","s","p","f","none","simple","partial","full"))){
            warning(paste0("Wrong seasonality type: '",seasonality, "'. Changing to 'full'"), call.=FALSE);
            seasonality <- "f";
        }
        seasonality <- substring(seasonality[1],1,1);
    }

    if(smoothType=="es"){
        # Check if the data is ts-object
        if(!is.ts(data) & Stype!="N"){
            if(!silentText){
                message("The provided data is not ts object. Only non-seasonal models are available.");
            }
            Stype <- "N";
            substr(model,nchar(model),nchar(model)) <- "N";
        }

        ### Check seasonality type
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
    }
    else if(smoothType=="sma"){
        maxlag <- 1;
        if(is.null(order)){
            nParamMax <- obsInsample;
        }
        else{
            nParamMax <- order;
        }
    }

    ##### Lags and components for GES #####
    if(smoothType=="ges"){
        if(any(is.complex(c(orders,lags)))){
            stop("Complex values? Right! Come on! Be real!",call.=FALSE);
        }
        if(any(c(orders)<0)){
            stop("Funny guy! How am I gonna construct a model with negative orders?",call.=FALSE);
        }
        if(any(c(lags)<0)){
            stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
        }
        if(length(orders) != length(lags)){
            stop(paste0("The length of 'lags' (",length(lags),
                        ") differes from the length of 'orders' (",length(orders),")."), call.=FALSE);
        }

        # If there are zero lags, drop them
        if(any(lags==0)){
            orders <- orders[lags!=0];
            lags <- lags[lags!=0];
        }
        # If zeroes are defined for some orders, drop them.
        if(any(orders==0)){
            lags <- lags[orders!=0];
            orders <- orders[orders!=0];
        }

        # Get rid of duplicates in lags
        if(length(unique(lags))!=length(lags)){
            lags.new <- unique(lags);
            orders.new <- lags.new;
            for(i in 1:length(lags.new)){
                orders.new[i] <- max(orders[which(lags==lags.new[i])]);
            }
            orders <- orders.new;
            lags <- lags.new;
        }

        modellags <- matrix(rep(lags,times=orders),ncol=1);
        maxlag <- max(modellags);
        nComponents <- sum(orders);

        type <- substr(type[1],1,1);
        if(type=="M"){
            y <- log(y);
            modelIsMultiplicative <- TRUE;
        }
        else{
            modelIsMultiplicative <- FALSE;
        }
    }
    else if(smoothType=="es"){
        maxlag <- datafreq * (Stype!="N") + 1 * (Stype=="N");
    }
    else if(smoothType=="ces"){
        A <- list(value=A);
        B <- list(value=B);

        if(is.null(A$value)){
            A$estimate <- TRUE;
        }
        else{
            A$estimate <- FALSE;
            if(!is.null(A$value)){
                parametersNumber[2,1] <- parametersNumber[2,1] + length(Re(A$value)) + length(Im(A$value));
            }
        }
        if(all(is.null(B$value),any(seasonality==c("p","f")))){
            B$estimate <- TRUE;
        }
        else{
            B$estimate <- FALSE;
            if(!is.null(B$value)){
                parametersNumber[2,1] <- parametersNumber[2,1] + length(Re(B$value)) + length(Im(B$value));
            }
        }

        # Define lags, number of components and number of parameters
        if(seasonality=="n"){
            # No seasonality
            maxlag <- 1;
            modellags <- c(1,1);
            # Define the number of all the parameters (smoothing parameters + initial states). Used in AIC mainly!
            nComponents <- 2;
            A$number <- 2;
            B$number <- 0;
        }
        else if(seasonality=="s"){
            # Simple seasonality, lagged CES
            maxlag <- datafreq;
            modellags <- c(maxlag,maxlag);
            nComponents <- 2;
            A$number <- 2;
            B$number <- 0;
        }
        else if(seasonality=="p"){
            # Partial seasonality with a real part only
            maxlag <- datafreq;
            modellags <- c(1,1,maxlag);
            nComponents <- 3;
            A$number <- 2;
            B$number <- 1;
        }
        else if(seasonality=="f"){
            # Full seasonality with both real and imaginary parts
            maxlag <- datafreq;
            modellags <- c(1,1,maxlag,maxlag);
            nComponents <- 4;
            A$number <- 2;
            B$number <- 2;
        }
    }

    ##### obsStates #####
    # Define the number of rows that should be in the matvt
    obsStates <- max(obsAll + maxlag, obsInsample + 2*maxlag);

    ##### bounds #####
    bounds <- substring(bounds[1],1,1);
    if(bounds!="u" & bounds!="a" & bounds!="n"){
        warning("Strange bounds are defined. Switching to 'usual'.",call.=FALSE);
        bounds <- "u";
    }

    if(any(smoothType==c("es","ges","ces"))){
        ##### Information Criteria #####
        ic <- ic[1];
        if(all(ic!=c("AICc","AIC","BIC"))){
            warning(paste0("Strange type of information criteria defined: ",ic,". Switching to 'AICc'."),call.=FALSE);
            ic <- "AICc";
        }
    }

    ##### Cost function type #####
    cfType <- cfType[1];
    if(any(cfType==c("MSEh","TMSE","GTMSE","MAEh","TMAE","GTMAE","HAMh","THAM","GTHAM",
                     "TFL","aMSEh","aTMSE","aGTMSE","aTFL"))){
        multisteps <- TRUE;
    }
    else if(any(cfType==c("MSE","MAE","HAM","TSB","Rounded"))){
        multisteps <- FALSE;
    }
    else{
        if(cfType=="MSTFE"){
            warning(paste0("This estimator has recently been renamed from \"MSTFE\" to \"TMSE\". ",
                           "Please, use the new name."),call.=FALSE);
            multisteps <- TRUE;
            cfType <- "TMSE";
        }
        else if(cfType=="GMSTFE"){
            warning(paste0("This estimator has recently been renamed from \"GMSTFE\" to \"GTMSE\". ",
                           "Please, use the new name."),call.=FALSE);
            multisteps <- TRUE;
            cfType <- "GTMSE";
        }
        else if(cfType=="aMSTFE"){
            warning(paste0("This estimator has recently been renamed from \"aMSTFE\" to \"aTMSE\". ",
                           "Please, use the new name."),call.=FALSE);
            multisteps <- TRUE;
            cfType <- "aTMSE";
        }
        else if(cfType=="aGMSTFE"){
            warning(paste0("This estimator has recently been renamed from \"aGMSTFE\" to \"aGTMSE\". ",
                           "Please, use the new name."),call.=FALSE);
            multisteps <- TRUE;
            cfType <- "aGTMSE";
        }
        else{
            warning(paste0("Strange cost function specified: ",cfType,". Switching to 'MSE'."),call.=FALSE);
            cfType <- "MSE";
            multisteps <- FALSE;
        }
    }
    cfTypeOriginal <- cfType;

    ##### intervals, intervalsType, level #####
    #intervalsType <- substring(intervalsType[1],1,1);
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

    ##### imodel #####
    if(class(imodel)!="iss"){
        intermittentModel <- imodel;
        imodelProvided <- FALSE;
        imodel <- NULL;
    }
    else{
        intermittentModel <- imodel$model;
        intermittent <- imodel$intermittent;
        imodelProvided <- TRUE;
    }

    ##### intermittent #####
    if(is.numeric(intermittent)){
        # If it is data, then it should either correspond to the whole sample (in-sample + holdout) or be equal to forecating horizon.
        if(all(length(c(intermittent))!=c(h,obsAll))){
            warning(paste0("Length of the provided future occurrences is ",length(c(intermittent)),
                           " while length of forecasting horizon is ",h,".\n",
                           "Where should we plug in the future occurences anyway?\n",
                           "Switching to intermittent='fixed'."),call.=FALSE);
            intermittent <- "f";
            ot <- (y!=0)*1;
            obsNonzero <- sum(ot);
            yot <- matrix(y[y!=0],obsNonzero,1);
            pt <- matrix(mean(ot),obsInsample,1);
            pt.for <- matrix(1,h,1);
            nParamIntermittent <- 1;
        }
        else{
            if(any(intermittent<0,intermittent>1)){
                warning(paste0("Parameter 'intermittent' should contain values between zero and one.\n",
                               "Converting to appropriate vector."),call.=FALSE);
                intermittent <- (intermittent!=0)*1;
            }

            ot <- (y!=0)*1;
            obsNonzero <- sum(ot);
            yot <- matrix(y[y!=0],obsNonzero,1);
            if(length(intermittent)==obsAll){
                pt <- intermittent[1:obsInsample];
                pt.for <- intermittent[(obsInsample+1):(obsInsample+h)];
            }
            else{
                pt <- matrix(ot,obsInsample,1);
                pt.for <- matrix(intermittent,h,1);
            }

            iprob <- pt.for[1];
            # "p" stand for "provided", meaning that we have been provided the future data
            intermittent <- "provided";
            nParamIntermittent <- 0;
        }
    }
    else{
        intermittent <- intermittent[1];
        if(all(intermittent!=c("n","f","i","p","a","s","none","fixed","interval","probability","auto","sba"))){
            ##### !!! This stuff should be removed by 2.5.0 #####
            if(any(intermittent==c("c","croston"))){
                warning(paste0("You are using the old value of intermittent parameter. ",
                               "Please, use 'i' instead of '",intermittent,"'."),
                        call.=FALSE);
                intermittent <- "i";
            }
            else if(any(intermittent==c("t","tsb"))){
                warning(paste0("You are using the old value of intermittent parameter. ",
                               "Please, use 'p' instead of '",intermittent,"'."),
                        call.=FALSE);
                intermittent <- "p";
            }
            else{
                warning(paste0("Strange type of intermittency defined: '",intermittent,"'. Switching to 'fixed'."),
                        call.=FALSE);
                intermittent <- "f";
            }
        }
        intermittent <- substring(intermittent[1],1,1);

        environment(intermittentParametersSetter) <- environment();
        intermittentParametersSetter(intermittent,ParentEnvironment=environment());

        if(obsNonzero <= nParamIntermittent){
            warning(paste0("Not enough observations for estimation of occurence probability.\n",
                           "Switching to simpler model."),
                    call.=FALSE);
            if(obsNonzero > 1){
                intermittent <- "f";
                nParamIntermittent <- 1;
                intermittentParametersSetter(intermittent,ParentEnvironment=environment());
            }
            else{
                intermittent <- "n";
                intermittentParametersSetter(intermittent,ParentEnvironment=environment());
            }
        }
    }

    # If the data is not intermittent, let's assume that the parameter was switched unintentionally.
    if(all(pt==1) & all(intermittent!=c("n","p"))){
        intermittent <- "n";
        imodelProvided <- FALSE;
    }

    if(imodelProvided){
        parametersNumber[2,3] <- imodel$nParam;
    }

    if(any(smoothType==c("es"))){
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

            if(!is.null(modelsPool)){
                if(any(c(substr(modelsPool,1,1),
                         substr(modelsPool,2,2),
                         substr(modelsPool,nchar(modelsPool),nchar(modelsPool)))=="M")){
                    warning("Can't apply multiplicative model to non-positive data. Switching to additive.", call.=FALSE);
                    substr(modelsPool,1,1)[substr(modelsPool,1,1)=="M"] <- "A";
                    substr(modelsPool,2,2)[substr(modelsPool,2,2)=="M"] <- "A";
                    substr(modelsPool,nchar(modelsPool),nchar(modelsPool))[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="M"] <- "A";
                }
            }
        }
    }

    if(any(smoothType==c("es","ges"))){
        ##### persistence for ES & GES #####
        if(!is.null(persistence)){
            if((!is.numeric(persistence) | !is.vector(persistence)) & !is.matrix(persistence)){
                warning(paste0("Persistence is not a numeric vector!\n",
                               "Changing to estimation of persistence vector values."),call.=FALSE);
                persistence <- NULL;
                persistenceEstimate <- TRUE;
            }
            else{
                if(smoothType=="es"){
                    if(modelDo!="estimate"){
                        warning(paste0("Predefined persistence vector can only be used with preselected ETS model.\n",
                                       "Changing to estimation of persistence vector values."),call.=FALSE);
                        persistence <- NULL;
                        persistenceEstimate <- TRUE;
                    }
                    else{
                        if(length(persistence)>3){
                            warning(paste0("Length of persistence vector is wrong! It should not be greater than 3.\n",
                                           "Changing to estimation of persistence vector values."),call.=FALSE);
                            persistence <- NULL;
                            persistenceEstimate <- TRUE;
                        }
                        else{
                            if(length(persistence)!=(1 + (Ttype!="N") + (Stype!="N"))){
                                warning(paste0("Wrong length of persistence vector. Should be ",(1 + (Ttype!="N") + (Stype!="N")),
                                               " instead of ",length(persistence),".\n",
                                               "Changing to estimation of persistence vector values."),call.=FALSE);
                                persistence <- NULL;
                                persistenceEstimate <- TRUE;
                            }
                            else{
                                persistence <- as.vector(persistence);
                                persistenceEstimate <- FALSE;
                                parametersNumber[2,1] <- parametersNumber[2,1] + length(persistence);
                            }
                        }
                    }
                }
                else if(smoothType=="ges"){
                    if(length(persistence) != nComponents){
                        warning(paste0("Wrong length of persistence vector. Should be ",nComponents,
                                       " instead of ",length(persistence),".\n",
                                       "Changing to estimation of persistence vector values."),call.=FALSE);
                        persistence <- NULL;
                        persistenceEstimate <- TRUE;
                    }
                    else{
                        persistenceEstimate <- FALSE;
                        parametersNumber[2,1] <- parametersNumber[2,1] + length(persistence);
                    }
                }
            }
        }
        else{
            persistenceEstimate <- TRUE;
        }
    }

    ##### initials ####
    # initial type can be: "o" - optimal, "b" - backcasting, "p" - provided.
    initialValue <- initial;
    if(is.character(initialValue)){
        initialValue <- substring(initialValue[1],1,1);
        if(all(initialValue!=c("o","b","p"))){
            warning("You asked for a strange initial value. We don't do that here. Switching to optimal.",
                    call.=FALSE);
            initialType <- "o";
        }
        else{
            initialType <- initialValue;
        }
        initialValue <- NULL;
    }
    else if(is.null(initialValue)){
        if(silentText){
            message("Initial value is not selected. Switching to optimal.");
        }
        initialType <- "o";
    }
    else if(!is.null(initialValue)){
        if(!is.numeric(initialValue)){
            warning(paste0("Initial vector is not numeric!\n",
                           "Values of initial vector will be estimated."),call.=FALSE);
            initialValue <- NULL;
            initialType <- "o";
        }
        else{
            if(smoothType=="es"){
                if(modelDo!="estimate"){
                    warning(paste0("Predefined initials vector can only be used with preselected ETS model.\n",
                                   "Changing to estimation of initials."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    if(length(initialValue)>2){
                        warning(paste0("Length of initial vector is wrong! It should not be greater than 2.\n",
                                       "Values of initial vector will be estimated."),call.=FALSE);
                        initialValue <- NULL;
                        initialType <- "o";
                    }
                    else{
                        if(length(initialValue) != (1*(Ttype!="N") + 1)){
                            warning(paste0("Length of initial vector is wrong! It should be ",(1*(Ttype!="N") + 1),
                                           " instead of ",length(initialValue),".\n",
                                           "Values of initial vector will be estimated."),call.=FALSE);
                            initialValue <- NULL;
                            initialType <- "o";
                        }
                        else{
                            initialType <- "p";
                            initialValue <- initial;
                            parametersNumber[2,1] <- parametersNumber[2,1] + length(initial);
                        }
                    }
                }
            }
            else if(smoothType=="ges"){
                if(length(initialValue) != (nComponents*max(lags))){
                    warning(paste0("Wrong length of initial vector. Should be ",(nComponents*max(lags)),
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    initialValue <- initial;
                    parametersNumber[2,1] <- parametersNumber[2,1] + orders %*% lags;
                }
            }
            else if(smoothType=="ssarima"){
                if(length(initialValue) != nComponents){
                    warning(paste0("Wrong length of initial vector. Should be ",nComponents,
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    initialValue <- initial;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(initial);
                }
            }
            else if(smoothType=="ces"){
                if(length(initialValue) != maxlag*nComponents){
                    warning(paste0("Wrong length of initial vector. Should be ",maxlag*nComponents,
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    initialValue <- initial;
                    parametersNumber[2,1] <- (parametersNumber[2,1] + 2*(seasonality!="s") +
                                              maxlag*(seasonality!="n") +
                                              maxlag*any(seasonality==c("f","s")));
                }
            }
        }
    }

    if(any(smoothType==c("es"))){
        # If model selection is chosen, forget about the initial values and persistence
        if(any(Etype=="Z",Ttype=="Z",Stype=="Z")){
            if(any(!is.null(initialValue),!is.null(initialSeason),!is.null(persistence),!is.null(phi))){
                warning(paste0("Model selection doesn't go well with the predefined values.\n",
                               "Switching to estimation of all the parameters."),call.=FALSE);
                initialValue <- NULL;
                initialType <- "o";
                initialSeason <- NULL;
                persistence <- NULL;
                phi <- NULL;
            }
        }

        ##### initialSeason for ES #####
        if(!is.null(initialSeason)){
            if(!is.numeric(initialSeason)){
                warning(paste0("InitialSeason vector is not numeric!\n",
                               "Values of initialSeason vector will be estimated."),call.=FALSE);
                initialSeason <- NULL;
                initialSeasonEstimate <- TRUE;
            }
            else{
                if(length(initialSeason)!=datafreq){
                warning(paste0("The length of initialSeason vector is wrong! It should correspond to the frequency of the data.\n",
                               "Values of initialSeason vector will be estimated."),call.=FALSE);
                    initialSeason <- NULL;
                    initialSeasonEstimate <- TRUE
                }
                else{
                    initialSeasonEstimate <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(initialSeason);
                }
            }
        }
        else{
            initialSeasonEstimate <- TRUE;
        }

        # Check the length of the provided data. Say bad words if:
        # 1. Seasonal model, <=2 seasons of data and no initial seasonals.
        # 2. Seasonal model, <=1 season of data, no initial seasonals and no persistence.
        if(is.null(modelsPool)){
            if((Stype!="N" & (obsInsample <= 2*datafreq) & is.null(initialSeason)) |
               (Stype!="N" & (obsInsample <= datafreq) & is.null(initialSeason) & is.null(persistence))){
                if(is.null(initialSeason)){
                    warning(paste0("Sorry, but we don't have enough observations for the seasonal model!\n",
                                   "Switching to non-seasonal."),call.=FALSE);
                    Stype <- "N";
                    initialSeasonEstimate <- FALSE;
                }
            }
        }

        ##### phi for ES #####
        if(!is.null(phi)){
            if(!is.numeric(phi) & (damped==TRUE)){
                warning(paste0("Provided value of phi is meaningless. phi will be estimated."),call.=FALSE);
                phi <- NULL;
                phiEstimate <- TRUE;
            }
            else if(is.numeric(phi) & (phi<0 | phi>2)){
                warning(paste0("Damping parameter should lie in (0, 2) region. ",
                        "Changing to the estimation of phi."),call.=FALSE);
                phi <- NULL;
                phiEstimate <- TRUE;
            }
            else{
                phiEstimate <- FALSE;
                if(damped){
                    parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                }
            }
        }
        else{
            if(damped){
                phiEstimate <- TRUE;
            }
            else{
                phiEstimate <- FALSE;
            }
        }
    }

    if(smoothType=="ges"){
        ##### transition for GES #####
        # Check the provided vector of initials: length and provided values.
        if(!is.null(transition)){
            if((!is.numeric(transition) | !is.vector(transition)) & !is.matrix(transition)){
                warning(paste0("Transition matrix is not numeric!\n",
                               "The matrix will be estimated!"),call.=FALSE);
                transitionEstimate <- TRUE;
            }
            else if(length(transition) != nComponents^2){
                warning(paste0("Wrong length of transition matrix. Should be ",nComponents^2,
                               " instead of ",length(transition),".\n",
                               "The matrix will be estimated!"),call.=FALSE);
                transitionEstimate <- TRUE;
            }
            else{
                transitionEstimate <- FALSE;
                parametersNumber[2,1] <- parametersNumber[2,1] + length(transition);
            }
        }
        else{
            transitionEstimate <- TRUE;
        }

        ##### measurement for GES #####
        if(!is.null(measurement)){
            if((!is.numeric(measurement) | !is.vector(measurement)) & !is.matrix(measurement)){
                warning(paste0("Measurement vector is not numeric!\n",
                               "The vector will be estimated!"),call.=FALSE);
                measurementEstimate <- TRUE;
            }
            else if(length(measurement) != nComponents){
                warning(paste0("Wrong length of measurement vector. Should be ",nComponents,
                               " instead of ",length(measurement),".\n",
                               "The vector will be estimated!"),call.=FALSE);
                measurementEstimate <- TRUE;
            }
            else{
                measurementEstimate <- FALSE;
                parametersNumber[2,1] <- parametersNumber[2,1] + length(measurement);
            }
        }
        else{
            measurementEstimate <- TRUE;
        }
    }

    if(smoothType=="ssarima"){
        if((nComponents==0) & (constantRequired==FALSE)){
            if(!silentText){
                warning("You have not defined any model! Constructing model with zero constant.",call.=FALSE);
            }
            constantRequired <- TRUE;
            constantValue <- 0;
            initialType <- "p";
        }
    }

    ##### Calculate nParamMax for checks #####
    if(smoothType=="es"){
        # 1: estimation of variance;
        # 1 - 3: persitence vector;
        # 1 - 2: initials;
        # 1 - 1 phi value;
        # datafreq: datafreq initials for seasonal component;
        nParamMax <- (1 + (1 + (Ttype!="N") + (Stype!="N"))*persistenceEstimate +
                          (1 + (Ttype!="N"))*(initialType=="o") + phiEstimate*damped +
                          datafreq*(Stype!="N")*initialSeasonEstimate*(initialType!="b"));
    }
    else if(smoothType=="ges"){
        nParamMax <- (1 + nComponents*measurementEstimate + nComponents*persistenceEstimate +
                          (nComponents^2)*transitionEstimate + (orders %*% lags)*(initialType=="o"));
    }
    else if(smoothType=="ssarima"){
        nParamMax <- (1 + nComponents*(initialType=="o") + sum(ar.orders)*ARRequired*AREstimate +
                          sum(ma.orders)*MARequired*MAEstimate + constantRequired*constantEstimate);
    }
    else if(smoothType=="ces"){
        nParamMax <- (1 + sum(modellags)*(initialType=="o") + A$number*A$estimate + B$number*B$estimate);
    }

    # Stop if number of observations is less than horizon and multisteps is chosen.
    if((multisteps==TRUE) & (obsNonzero < h+1) & all(cfType!=c("aMSEh","aTMSE","aGTMSE","aTFL"))){
        warning(paste0("Do you seriously think that you can use ",cfType,
                       " with h=",h," on ",obsNonzero," non-zero observations?!"),call.=FALSE);
        stop("Not enough observations for multisteps cost function.",call.=FALSE);
    }
    else if((multisteps==TRUE) & (obsNonzero < 2*h) & all(cfType!=c("aMSEh","aTMSE","aGTMSE","aTFL"))){
        warning(paste0("Number of observations is really low for a multisteps cost function! ",
                       "We will, try but cannot guarantee anything..."),call.=FALSE);
    }

    normalizer <- mean(abs(diff(c(y))));

    ##### Define xregDo #####
    if(smoothType!="sma"){
        if(!any(xregDo==c("use","select","u","s"))){
            warning("Wrong type of xregDo parameter. Changing to 'select'.", call.=FALSE);
            xregDo <- "select";
        }
        xregDo <- substr(xregDo[1],1,1);
    }

    if(is.null(xreg)){
        xregDo <- "u";
    }

    ##### Fisher Information #####
    if(!exists("FI",envir=ParentEnvironment,inherits=FALSE)){
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

    ##### Rounded up values #####
    if(!exists("rounded",envir=ParentEnvironment,inherits=FALSE)){
        rounded <- FALSE;
    }
    else{
        if(!is.logical(rounded)){
            rounded <- FALSE;
        }
        else{
            if(rounded){
                cfType <- "Rounded";
                cfTypeOriginal <- cfType;
            }
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
    assign("bounds",bounds,ParentEnvironment);
    assign("cfType",cfType,ParentEnvironment);
    assign("cfTypeOriginal",cfTypeOriginal,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("intervalsType",intervalsType,ParentEnvironment);
    assign("intervals",intervals,ParentEnvironment);
    assign("intermittent",intermittent,ParentEnvironment);
    assign("intermittentModel",intermittentModel,ParentEnvironment);
    assign("imodel",imodel,ParentEnvironment);
    assign("ot",ot,ParentEnvironment);
    assign("yot",yot,ParentEnvironment);
    assign("pt",pt,ParentEnvironment);
    assign("pt.for",pt.for,ParentEnvironment);
    assign("nParamIntermittent",nParamIntermittent,ParentEnvironment);
    assign("iprob",iprob,ParentEnvironment);
    assign("imodelProvided",imodelProvided,ParentEnvironment);
    assign("initialValue",initialValue,ParentEnvironment);
    assign("initialType",initialType,ParentEnvironment);
    assign("normalizer",normalizer,ParentEnvironment);
    assign("nParamMax",nParamMax,ParentEnvironment);
    assign("xregDo",xregDo,ParentEnvironment);
    assign("FI",FI,ParentEnvironment);
    assign("rounded",rounded,ParentEnvironment);
    assign("parametersNumber",parametersNumber,ParentEnvironment);

    if(smoothType=="es"){
        assign("model",model,ParentEnvironment);
        assign("modelsPool",modelsPool,ParentEnvironment);
        assign("Etype",Etype,ParentEnvironment);
        assign("Ttype",Ttype,ParentEnvironment);
        assign("Stype",Stype,ParentEnvironment);
        assign("damped",damped,ParentEnvironment);
        assign("modelDo",modelDo,ParentEnvironment);
        assign("initialSeason",initialSeason,ParentEnvironment);
        assign("phi",phi,ParentEnvironment);
        assign("phiEstimate",phiEstimate,ParentEnvironment);
        assign("allowMultiplicative",allowMultiplicative,ParentEnvironment);
        assign("ic",ic,ParentEnvironment);
    }
    else if(smoothType=="ges"){
        assign("transitionEstimate",transitionEstimate,ParentEnvironment);
        assign("measurementEstimate",measurementEstimate,ParentEnvironment);
        assign("orders",orders,ParentEnvironment);
        assign("lags",lags,ParentEnvironment);
        assign("modelIsMultiplicative",modelIsMultiplicative,ParentEnvironment);
    }
    else if(smoothType=="ssarima"){
        assign("ar.orders",ar.orders,ParentEnvironment);
        assign("i.orders",i.orders,ParentEnvironment);
        assign("ma.orders",ma.orders,ParentEnvironment);
        assign("lags",lags,ParentEnvironment);
        assign("ARValue",ARValue,ParentEnvironment);
        assign("ARRequired",ARRequired,ParentEnvironment);
        assign("AREstimate",AREstimate,ParentEnvironment);
        assign("MAValue",MAValue,ParentEnvironment);
        assign("MARequired",MARequired,ParentEnvironment);
        assign("MAEstimate",MAEstimate,ParentEnvironment);
        assign("constantValue",constantValue,ParentEnvironment);
        assign("constantEstimate",constantEstimate,ParentEnvironment);
        assign("constantRequired",constantRequired,ParentEnvironment);
    }
    else if(smoothType=="ces"){
        assign("seasonality",seasonality,ParentEnvironment);
        assign("A",A,ParentEnvironment);
        assign("B",B,ParentEnvironment);
    }

    if(any(smoothType==c("es","ges"))){
        assign("persistence",persistence,ParentEnvironment);
        assign("persistenceEstimate",persistenceEstimate,ParentEnvironment);
    }

    if(any(smoothType==c("ges","ssarima","ces"))){
        assign("nComponents",nComponents,ParentEnvironment);
        assign("maxlag",maxlag,ParentEnvironment);
        assign("modellags",modellags,ParentEnvironment);
    }
}

##### *Checker for auto. functions* #####
ssAutoInput <- function(smoothType=c("auto.ces","auto.ges","auto.ssarima"),...){
    # This is universal function needed in order to check the passed arguments to auto.ces(), auto.ges() and auto.ssarima()

    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ##### silent #####
    silent <- silent[1];
    # Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output","debugging","n","a","g","l","o","d"))){
            message(paste0("Sorry, I have no idea what 'silent=",silent,"' means. Switching to 'none'."));
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

    # Check what was asked as a horizon
    if(h<=0){
        warning(paste0("You have set forecast horizon equal to ",h,". We hope you know, what you are doing."), call.=FALSE);
        if(h<0){
            warning("And by the way, we can't do anything with negative horizon, so we will set it equal to zero.", call.=FALSE);
            h <- 0;
        }
    }

    ##### Fisher Information #####
    if(!exists("FI",envir=ParentEnvironment,inherits=FALSE)){
        FI <- FALSE;
    }

    ##### data #####
    if(any(class(data)=="smooth.sim")){
        data <- data$data;
    }
    if(!is.numeric(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }
    # Check the data for NAs
    if(any(is.na(data))){
        if(!silentText){
            warning("Data contains NAs. These observations will be substituted by zeroes.",call.=FALSE);
        }
        data[is.na(data)] <- 0;
    }

    ##### Observations #####
# Define obs, the number of observations of in-sample
    obsInsample <- length(data) - holdout*h;

# Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- length(data) + (1 - holdout)*h;

    y <- data[1:obsInsample];
    datafreq <- frequency(data);

# This is the critical minimum needed in order to at least fit ARIMA(0,0,0) with constant
    if(obsInsample < 4){
        stop("Sorry, but your sample is too small. Come back when you have at least 4 observations...",call.=FALSE);
    }

# Check the provided vector of initials: length and provided values.
    initialValue <- initial;
    if(is.character(initialValue)){
        initialValue <- substring(initialValue[1],1,1);
        if(initialValue!="o" & initialValue!="b"){
            warning("You asked for a strange initial value. We don't do that here. Switching to optimal.",call.=FALSE);
            initialType <- "o";
        }
        else{
            initialType <- initialValue;
        }
        initialValue <- NULL;
    }
    else if(is.null(initialValue)){
        if(!silentText){
            warning("Initial value is not selected. Switching to optimal.",call.=FALSE);
        }
        initialType <- "o";
    }
    else{
        warning("Predefined initials don't go well with automatic model selection. Switching to optimal.",call.=FALSE);
        initialType <- "o";
        initialValue <- NULL;
    }

    ##### bounds #####
    bounds <- substring(bounds[1],1,1);
    # Check if "bounds" parameter makes any sense
    if(bounds!="n" & bounds!="a"){
        warning("Strange bounds are defined. Switching to 'admissible'.",call.=FALSE);
        bounds <- "a";
    }

    ##### Information Criteria #####
    ic <- ic[1];
    if(all(ic!=c("AICc","AIC","BIC"))){
        warning(paste0("Strange type of information criteria defined: ",ic,". Switching to 'AICc'."),call.=FALSE);
        ic <- "AICc";
    }

    ##### Cost function type #####
    cfType <- cfType[1];
    if(any(cfType==c("MSEh","TMSE","GTMSE","MAEh","TMAE","GTMAE","HAMh","THAM","GTHAM",
                     "TFL","aMSEh","aTMSE","aGTMSE","aTFL"))){
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

    ##### intervals, intervalsType, level #####
    #intervalsType <- substring(intervalsType[1],1,1);
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

    ##### imodel #####
    if(class(imodel)!="iss"){
        intermittentModel <- imodel;
        imodelProvided <- FALSE;
        imodel <- NULL;
    }
    else{
        intermittentModel <- imodel$model;
        intermittent <- imodel$intermittent;
        imodelProvided <- TRUE;
    }

    ##### intermittent #####
    if(is.numeric(intermittent)){
        obsNonzero <- sum((y!=0)*1);
        # If it is data, then it should either correspond to the whole sample (in-sample + holdout) or be equal to forecating horizon.
        if(all(length(c(intermittent))!=c(h,obsAll))){
            warning(paste0("Length of the provided future occurrences is ",length(c(intermittent)),
                           " while length of forecasting horizon is ",h,".\n",
                           "Where should we plug in the future occurences anyway?\n",
                           "Switching to intermittent='fixed'."),call.=FALSE);
            intermittent <- "f";
        }

        if(any(intermittent!=0 & intermittent!=1)){
            warning(paste0("Parameter 'intermittent' should contain only zeroes and ones.\n",
                           "Converting to appropriate vector."),call.=FALSE);
            intermittent <- (intermittent!=0)*1;
        }
    }
    else{
        obsNonzero <- sum((y!=0)*1);
        intermittent <- intermittent[1];
        if(all(intermittent!=c("n","f","i","p","a","s","none","fixed","interval","probability","auto","sba"))){
            warning(paste0("Strange type of intermittency defined: '",intermittent,"'. Switching to 'fixed'."),
                    call.=FALSE);
            intermittent <- "f";
        }
        intermittent <- substring(intermittent,1,1);
        if(any(intermittent!="n")){
            obsNonzero <- sum((y!=0)*1);
            environment(intermittentParametersSetter) <- environment();
            intermittentParametersSetter(intermittent,ParentEnvironment=environment());

            if(obsNonzero <= nParamIntermittent){
                warning(paste0("Not enough observations for estimation of occurence probability.\n",
                               "Switching to simpler model."),
                        call.=FALSE);
                if(obsNonzero > 1){
                    intermittent <- "f";
                    nParamIntermittent <- 1;
                    intermittentParametersSetter(intermittent,ParentEnvironment=environment());
                }
                else{
                    intermittent <- "n";
                    intermittentParametersSetter(intermittent,ParentEnvironment=environment());
                }
            }
        }
        else{
            obsNonzero <- obsInsample;
        }
    }

    ##### Define xregDo #####
    if(!any(xregDo==c("use","select","u","s"))){
        warning("Wrong type of xregDo parameter. Changing to 'select'.", call.=FALSE);
        xregDo <- "select";
    }
    xregDo <- substr(xregDo[1],1,1);

    if(is.null(xreg)){
        xregDo <- "u";
    }

    ##### Return values to previous environment #####
    assign("h",h,ParentEnvironment);
    assign("silentText",silentText,ParentEnvironment);
    assign("silentGraph",silentGraph,ParentEnvironment);
    assign("silentLegend",silentLegend,ParentEnvironment);
    assign("bounds",bounds,ParentEnvironment);
    assign("FI",FI,ParentEnvironment);
    assign("obsInsample",obsInsample,ParentEnvironment);
    assign("obsAll",obsAll,ParentEnvironment);
    assign("obsNonzero",obsNonzero,ParentEnvironment);
    assign("initialValue",initialValue,ParentEnvironment);
    assign("initialType",initialType,ParentEnvironment);
    assign("ic",ic,ParentEnvironment);
    assign("cfType",cfType,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("intervals",intervals,ParentEnvironment);
    assign("intervalsType",intervalsType,ParentEnvironment);
    assign("intermittent",intermittent,ParentEnvironment);
    assign("y",y,ParentEnvironment);
    assign("data",data,ParentEnvironment);
    assign("datafreq",datafreq,ParentEnvironment);
    assign("xregDo",xregDo,ParentEnvironment);
}

##### *ssFitter function* #####
ssFitter <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    fitting <- fitterwrap(matvt, matF, matw, y, vecg,
                          modellags, Etype, Ttype, Stype, initialType,
                          matxt, matat, matFX, vecgX, ot);
    statesNames <- colnames(matvt);
    matvt <- ts(fitting$matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=datafreq);
    colnames(matvt) <- statesNames;
    y.fit <- ts(fitting$yfit,start=start(data),frequency=datafreq);
    errors <- ts(fitting$errors,start=start(data),frequency=datafreq);

    if(Etype=="M" & any(matvt[,1]<0)){
        matvt[matvt[,1]<0,1] <- 0.001;
        warning(paste0("Negative values produced in the level of state vector of model ",model,".\n",
                       "We had to substitute them by low values. Please, use a different model."),call.=FALSE);
    }

    if(!is.null(xreg)){
        # Write down the matat and copy values for the holdout
        matat[1:nrow(fitting$matat),] <- fitting$matat;
    }

    if(h>0){
        errors.mat <- ts(errorerwrap(matvt, matF, matw, y,
                                     h, Etype, Ttype, Stype, modellags,
                                     matxt, matat, matFX, ot),
                         start=(time(data)[1] - deltat(data)*(h-1)),frequency=datafreq);
        colnames(errors.mat) <- paste0("Error",c(1:h));
    }
    else{
        errors.mat <- NA;
    }

    assign("matvt",matvt,ParentEnvironment);
    assign("y.fit",y.fit,ParentEnvironment);
    assign("matat",matat,ParentEnvironment);
    assign("errors.mat",errors.mat,ParentEnvironment);
    assign("errors",errors,ParentEnvironment);
}

##### *State-space intervals* #####
ssIntervals <- function(errors, ev=median(errors), level=0.95, intervalsType=c("a","p","sp","np"), df=NULL,
                        measurement=NULL, transition=NULL, persistence=NULL, s2=NULL,
                        modellags=NULL, states=NULL, cumulative=FALSE,
                        y.for=rep(0,ncol(errors)), Etype="A", Ttype="N", Stype="N", s2g=NULL,
                        iprob=1, ivar=1){
# Function constructs intervals based on the provided random variable.
# If errors is a matrix, then it is assumed that each column has a variable that needs an interval.
# based on errors the horison is estimated as ncol(errors)

    matrixpower <- function(A,n){
        if(n==0){
            return(diag(nrow(A)));
        }
        else if(n==1){
            return(A);
        }
        else if(n>1){
            return(A %*% matrixpower(A, n-1));
        }
    }

    hsmN <- gamma(0.75)*pi^(-0.5)*2^(-0.75);
    intervalsType <- intervalsType[1]
    # Check the provided type of interval

    if(is.logical(intervalsType)){
        if(intervalsType){
            intervalsType <- "p";
        }
        else{
            intervalsType <- "none";
        }
    }

    if(all(intervalsType!=c("a","p","s","n","a","sp","np","none","parametric","semiparametric","nonparametric","asymmetric"))){
        stop(paste0("What do you mean by 'intervalsType=",intervalsType,"'? I can't work with this!"),call.=FALSE);
    }

    if(intervalsType=="none"){
        intervalsType <- "n";
    }
    else if(intervalsType=="parametric"){
        intervalsType <- "p";
    }
    else if(intervalsType=="semiparametric"){
        intervalsType <- "sp";
    }
    else if(intervalsType=="nonparametric"){
        intervalsType <- "np";
    }

    if(intervalsType=="p"){
        if(any(is.null(measurement),is.null(transition),is.null(persistence),is.null(s2),is.null(modellags))){
            stop("measurement, transition, persistence, s2 and modellags need to be provided in order to construct parametric intervals!",call.=FALSE);
        }

        if(any(!is.matrix(measurement),!is.matrix(transition),!is.matrix(persistence))){
            stop("measurement, transition and persistence must me matrices. Can't do stuff with what you've provided.",call.=FALSE);
        }
    }

# Function allows to estimate the coefficients of the simple quantile regression. Used in intervals construction.
quantfunc <- function(A){
    ee <- ye - (A[1]*xe^A[2]);
    return((1-quant)*sum(abs(ee[which(ee<0)]))+quant*sum(abs(ee[which(ee>=0)])));
}

# Function allows to find the quantiles of Bernoulli-lognormal cumulative distribution
qlnormBinCF <- function(quant, iprob, level=0.95, Etype="M", meanVec=0, sdVec){
    if(Etype=="M"){
        quantiles <- iprob * plnorm(quant, meanlog=meanVec, sdlog=sdVec) + (1 - iprob);
    }
    else{
        quantiles <- iprob * pnorm(quant, mean=meanVec, sd=sdVec) + (1 - iprob)*(quant>0);
    }
    CF <- (level-quantiles)^2;
    return(CF)
}

# Function returns quantiles of Bernoulli-lognormal cumulative distribution for a predefined parameters
qlnormBin <- function(iprob, level=0.95, meanVec=0, sdVec=1, Etype="A"){
    lowerquant <- upperquant <- rep(0,length(sdVec));

# Produce lower quantiles
    if(Etype=="A" | all(Etype=="M",(1-iprob) < (1-level)/2)){
        if(Etype=="M"){
            quantInitials <- qlnorm((1-level)/2,meanVec,sdVec)
        }
        else{
            quantInitials <- qnorm((1-level)/2,meanVec,sdVec)
        }
        for(i in 1:length(sdVec)){
            if(quantInitials[i]==0){
                lowerquant[i] <- 0;
            }
            else{
                lowerquant[i] <- optimize(qlnormBinCF, c(quantInitials[i],0), tol=1e-10, iprob=iprob, level=(1-level)/2, Etype=Etype, meanVec=meanVec[i], sdVec=sdVec[i])[[1]];
            }
            # lowerquant[i] <- nlminb(quantInitials[i], qlnormBinCF, iprob=iprob, level=(1-level)/2, Etype=Etype, meanVec=meanVec[i], sdVec=sdVec[i])$par;
        }
        levelNew <- (1+level)/2;
    }
    else{
        levelNew <- level;
    }

# Produce upper quantiles
    if(Etype=="M"){
        quantInitials <- qlnorm(levelNew,meanVec,sdVec);
    }
    else{
        quantInitials <- qnorm(levelNew,meanVec,sdVec);
    }
    for(i in 1:length(sdVec)){
        if(quantInitials[i]==0){
            upperquant[i] <- 0;
        }
        else{
            upperquant[i] <- optimize(qlnormBinCF, c(0,quantInitials[i]), tol=1e-10, iprob=iprob, level=levelNew, Etype=Etype, meanVec=meanVec[i], sdVec=sdVec[i])[[1]];
            upperquant[i] <- max(0,upperquant[i]);
        }
    }

    return(list(lower=lowerquant,upper=upperquant));
}

# If degrees of freedom are provided, use Student's distribution. Otherwise stick with normal.
    if(is.null(df)){
        upperquant <- qnorm((1+level)/2,0,1);
        lowerquant <- qnorm((1-level)/2,0,1);
    }
    else{
        if(df>0){
            upperquant <- qt((1+level)/2,df=df);
            lowerquant <- qt((1-level)/2,df=df);
        }
        else{
            upperquant <- sqrt(1/((1-level)/2));
            lowerquant <- -upperquant;
        }
    }

##### If they want us to produce several steps ahead #####
    if(is.matrix(errors) | is.data.frame(errors)){
        if(!cumulative){
            nVariables <- ncol(errors);
        }
        else{
            nVariables <- 1;
        }
        obs <- nrow(errors);
        # if(length(ev)!=nVariables & length(ev)!=1){
        #     stop("Provided expected value doesn't correspond to the dimension of errors.", call.=FALSE);
        # }
        # else
        if(length(ev)==1){
            ev <- rep(ev,nVariables);
        }

        upper <- rep(NA,nVariables);
        lower <- rep(NA,nVariables);

#### Asymmetric intervals using HM ####
        if(intervalsType=="a"){
            if(!cumulative){
                for(i in 1:nVariables){
                    upper[i] <- ev[i] + upperquant / hsmN^2 * Re(hm(errors[,i],ev[i]))^2;
                    lower[i] <- ev[i] + lowerquant / hsmN^2 * Im(hm(errors[,i],ev[i]))^2;
                }
            }
            else{
                upper <- ev + upperquant / hsmN^2 * Re(hm(rowSums(errors),sum(ev)))^2;
                lower <- ev + lowerquant / hsmN^2 * Im(hm(rowSums(errors),sum(ev)))^2;
            }
            if(Etype=="M"){
                upper <- 1 + upper;
                lower <- 1 + lower;
            }
            varVec <- NULL;
        }

#### Semiparametric intervals using the variance of errors ####
        else if(intervalsType=="sp"){
            if(Etype=="M"){
                errors[errors < -1] <- -0.999;
                if(!cumulative){
                    varVec <- colSums(log(1+errors)^2,na.rm=T)/df;
                    if(any(iprob!=1)){
                        quants <- qlnormBin(iprob, level=level, meanVec=log(y.for), sdVec=sqrt(varVec), Etype="M");
                        upper <- quants$upper;
                        lower <- quants$lower;
                    }
                    else{
                        upper <- qlnorm((1+level)/2,rep(0,nVariables),sqrt(varVec));
                        lower <- qlnorm((1-level)/2,rep(0,nVariables),sqrt(varVec));
                    }
                }
                else{
                    #This is wrong. And there's not way to make it right.
                    varVec <- sum(rowSums(log(1+errors))^2,na.rm=T)/df;
                    if(any(iprob!=1)){
                        quants <- qlnormBin(iprob, level=level, meanVec=log(sum(y.for)), sdVec=sqrt(varVec), Etype="M");
                        upper <- quants$upper;
                        lower <- quants$lower;
                    }
                    else{
                        upper <- qlnorm((1+level)/2,rep(0,nVariables),sqrt(varVec));
                        lower <- qlnorm((1-level)/2,rep(0,nVariables),sqrt(varVec));
                    }
                }
            }
            else{
                if(!cumulative){
                    errors <- errors - matrix(ev,nrow=obs,ncol=nVariables,byrow=T);
                    varVec <- colSums(errors^2,na.rm=T)/df;
                    if(any(iprob!=1)){
                        quants <- qlnormBin(iprob, level=level, meanVec=ev, sdVec=sqrt(varVec), Etype="A");
                        upper <- ev + quants$upper;
                        lower <- ev + quants$lower;
                    }
                    else{
                        upper <- ev + upperquant * sqrt(varVec);
                        lower <- ev + lowerquant * sqrt(varVec);
                    }
                }
                else{
                    errors <- errors - matrix(ev,nrow=obs,ncol=ncol(errors),byrow=T);
                    varVec <- sum(rowSums(errors,na.rm=T)^2,na.rm=T)/df;
                    if(any(iprob!=1)){
                        quants <- qlnormBin(iprob, level=level, meanVec=sum(ev), sdVec=sqrt(varVec), Etype="A");
                        upper <- sum(ev) + quants$upper;
                        lower <- sum(ev) + quants$lower;
                    }
                    else{
                        upper <- sum(ev) + upperquant * sqrt(varVec);
                        lower <- sum(ev) + lowerquant * sqrt(varVec);
                    }

                }
            }
        }

#### Nonparametric intervals using Taylor and Bunn, 1999 ####
        else if(intervalsType=="np"){
            ye <- errors;
            if(Etype=="M"){
                ye <- 1 + ye;
            }
            if(!cumulative){
                xe <- matrix(c(1:nVariables),byrow=TRUE,ncol=nVariables,nrow=nrow(errors));
                xe <- xe[!is.na(errors)];
                ye <- ye[!is.na(ye)];

                A <- rep(1,2);
                quant <- (1+level)/2;
                A <- nlminb(A,quantfunc)$par;
                upper <- A[1]*c(1:nVariables)^A[2];

                A <- rep(1,2);
                quant <- (1-level)/2;
                A <- nlminb(A,quantfunc)$par;
                lower <- A[1]*c(1:nVariables)^A[2];
            }
            else{
                #This is wrong. And there's not way to make it right.
                upper <- quantile(rowSums(ye),(1+level)/2);
                lower <- quantile(rowSums(ye),(1-level)/2);
            }
            varVec <- NULL;
        }

#### Parametric intervals ####
        else if(intervalsType=="p"){
            nComponents <- nrow(transition);
            maxlag <- max(modellags);
            h <- length(y.for);

            # Vector of final variances
            varVec <- rep(NA,h);
            if(cumulative){
                # covarVec <- rep(0,h);
                cumVarVec <- rep(0,h);
                cumVarVec[1:min(h,maxlag)] <- s2 * (h - 1:min(h,maxlag) + 1);
            }

#### Pure multiplicative models ####
            if(Etype=="M"){
                # Array of variance of states
                matrixOfVarianceOfStates <- array(0,c(nComponents,nComponents,h+maxlag));
                matrixOfVarianceOfStates[,,1:maxlag] <- persistence %*% t(persistence) * s2;
                matrixOfVarianceOfStatesLagged <- as.matrix(matrixOfVarianceOfStates[,,1]);

                # New transition and measurement for the internal use
                transitionnew <- matrix(0,nComponents,nComponents);
                measurementnew <- matrix(0,1,nComponents);

                # selectionmat is needed for the correct selection of lagged variables in the array
                # newelements are needed for the correct fill in of all the previous matrices
                selectionmat <- transitionnew;
                newelements <- rep(FALSE,nComponents);

                # Define chunks, which correspond to the lags with h being the final one
                chuncksofhorizon <- c(1,unique(modellags),h);
                chuncksofhorizon <- sort(chuncksofhorizon);
                chuncksofhorizon <- chuncksofhorizon[chuncksofhorizon<=h];
                chuncksofhorizon <- unique(chuncksofhorizon);

                # Length of the vector, excluding the h at the end
                chunkslength <- length(chuncksofhorizon) - 1;

                newelements <- modellags<=(chuncksofhorizon[1]);
                measurementnew[,newelements] <- measurement[,newelements];
                # This is needed for the first observations, where we do not care about the transition equation
                varVec[1:min(h,maxlag)] <- s2;

                for(j in 1:chunkslength){
                    selectionmat[modellags==chuncksofhorizon[j],] <- chuncksofhorizon[j];
                    selectionmat[,modellags==chuncksofhorizon[j]] <- chuncksofhorizon[j];

                    newelements <- modellags<(chuncksofhorizon[j]+1);
                    transitionnew[,newelements] <- transition[,newelements];
                    measurementnew[,newelements] <- measurement[,newelements];

                    for(i in (chuncksofhorizon[j]+1):chuncksofhorizon[j+1]){
                        selectionmat[modellags>chuncksofhorizon[j],] <- i;
                        selectionmat[,modellags>chuncksofhorizon[j]] <- i;

                        matrixOfVarianceOfStatesLagged[newelements,newelements] <- matrixOfVarianceOfStates[cbind(rep(c(1:nComponents),each=nComponents),
                                                                                                                  rep(c(1:nComponents),nComponents),
                                                                                                                  i - c(selectionmat))];

                        matrixOfVarianceOfStates[,,i] <- transitionnew %*% matrixOfVarianceOfStatesLagged %*% t(transitionnew) + s2g;
                        varVec[i] <- measurementnew %*% matrixOfVarianceOfStatesLagged %*% t(measurementnew) + s2;
                        if(cumulative){
                            # This is just an approximation!
                            cumVarVec[i] <- ((measurementnew %*% matrixOfVarianceOfStatesLagged %*% t(measurementnew)) + s2) * m;
                            m <- m-1;
                        }
                    }
                }

                ### Cumulative variance is different.
                if(cumulative){
                    varVec <- sum(cumVarVec);

                    if(any(iprob!=1)){
                        quants <- qlnormBin(iprob, level=level, meanVec=log(sum(y.for)), sdVec=sqrt(varVec), Etype="M");
                        upper <- quants$upper;
                        lower <- quants$lower;
                    }
                    else{
                        # Produce quantiles for log-normal dist with the specified variance
                        upper <- sum(y.for)*qlnorm((1+level)/2,0,sqrt(varVec));
                        lower <- sum(y.for)*qlnorm((1-level)/2,0,sqrt(varVec));
                    }
                }
                else{
                    if(any(iprob!=1)){
                        quants <- qlnormBin(iprob, level=level, meanVec=log(y.for), sdVec=sqrt(varVec), Etype="M");
                        upper <- quants$upper;
                        lower <- quants$lower;
                    }
                    else{
                        # Produce quantiles for log-normal dist with the specified variance
                        upper <- y.for*qlnorm((1+level)/2,0,sqrt(varVec));
                        lower <- y.for*qlnorm((1-level)/2,0,sqrt(varVec));
                    }
                }
            }
#### Multiplicative error and additive trend / seasonality
            # else if(Etype=="M" & all(c(Ttype,Stype)!="M") & all(c(Ttype,Stype)!="N")){
            # }
#### Pure Additive models ####
            else{
                ### This is an example of the correct variance estimation for h=10
                ### This needs to be implemented here at some point
                # y <- sim.es("ANA",frequency=4,obs=120)
                # test <- ges(y$data,orders=c(1,1,1),lags=c(1,4,9),intervals=T,h=10)
                #
                # F1 <- test$transition
                # F2 <- test$transition
                # F1[,-1] <- 0
                # F2[,-2] <- 0
                # g <- test$persistence
                # w1 <- test$measurement
                # w2 <- test$measurement
                # w1[,-1] <- 0
                # w2[,-2] <- 0
                #
                # vecVar <- rep(1,10)
                # vecVar[1] <- (w1 %*% (matrixPowerWrap(F1,8) %*% g + F2 %*% matrixPowerWrap(F1,4) %*% g + matrixPowerWrap(F2,2) %*% g) +
                #                   w2 %*% (matrixPowerWrap(F1,5) %*% g + F2 %*% matrixPowerWrap(F1,1) %*% g))
                #
                # vecVar[2] <- (w1 %*% (matrixPowerWrap(F1,7) %*% g + F2 %*% matrixPowerWrap(F1,3) %*% g) +
                #                   w2 %*% (matrixPowerWrap(F1,4) %*% g + F2 %*% g))
                #
                # vecVar[3] <- (w1 %*% (matrixPowerWrap(F1,6) %*% g + F2 %*% matrixPowerWrap(F1,2) %*% g) +
                #                   w2 %*% (matrixPowerWrap(F1,3) %*% g))
                #
                # vecVar[4] <- (w1 %*% (matrixPowerWrap(F1,5) %*% g + F2 %*% matrixPowerWrap(F1,1) %*% g) +
                #                   w2 %*% (matrixPowerWrap(F1,2) %*% g))
                #
                # vecVar[5] <- (w1 %*% (matrixPowerWrap(F1,4) %*% g + F2 %*% g) +
                #                   w2 %*% (matrixPowerWrap(F1,1) %*% g))
                #
                # vecVar[6] <- (w1 %*% (matrixPowerWrap(F1,3) %*% g) +
                #                   w2 %*% g)
                #
                # vecVar[7:10] <- c(w1 %*% (matrixPowerWrap(F1,2) %*% g),
                #                   w1 %*% (matrixPowerWrap(F1,1) %*% g),
                #                   w1 %*% g,
                #                   1)
                #
                # vecVar <- vecVar^2
                # sum(vecVar) * test$s2

                # Array of variance of states
                matrixOfVarianceOfStates <- array(0,c(nComponents,nComponents,h+maxlag));
                matrixOfVarianceOfStates[,,1:maxlag] <- persistence %*% t(persistence) * s2;
                matrixOfVarianceOfStatesLagged <- as.matrix(matrixOfVarianceOfStates[,,1]);

                # New transition and measurement for the internal use
                transitionnew <- matrix(0,nComponents,nComponents);
                measurementnew <- matrix(0,1,nComponents);

                # selectionmat is needed for the correct selection of lagged variables in the array
                # newelements are needed for the correct fill in of all the previous matrices
                selectionmat <- transitionnew;
                newelements <- rep(FALSE,nComponents);

                # Define chunks, which correspond to the lags with h being the final one
                chuncksofhorizon <- c(1,unique(modellags),h);
                chuncksofhorizon <- sort(chuncksofhorizon);
                chuncksofhorizon <- chuncksofhorizon[chuncksofhorizon<=h];
                chuncksofhorizon <- unique(chuncksofhorizon);

                # Length of the vector, excluding the h at the end
                chunkslength <- length(chuncksofhorizon) - 1;

                newelements <- modellags<=(chuncksofhorizon[1]);
                measurementnew[,newelements] <- measurement[,newelements];

                # This is needed for the first observations, where we do not care about the transition equation
                varVec[1:min(h,maxlag)] <- s2;

                m <- h-1;
                for(j in 1:chunkslength){
                    selectionmat[modellags==chuncksofhorizon[j],] <- chuncksofhorizon[j];
                    selectionmat[,modellags==chuncksofhorizon[j]] <- chuncksofhorizon[j];

                    newelements <- modellags<(chuncksofhorizon[j]+1);
                    transitionnew[,newelements] <- transition[,newelements];
                    measurementnew[,newelements] <- measurement[,newelements];

                    for(i in (chuncksofhorizon[j]+1):chuncksofhorizon[j+1]){
                        selectionmat[modellags>chuncksofhorizon[j],] <- i;
                        selectionmat[,modellags>chuncksofhorizon[j]] <- i;

                        matrixOfVarianceOfStatesLagged[newelements,newelements] <- matrixOfVarianceOfStates[cbind(rep(c(1:nComponents),each=nComponents),
                                                                                                                  rep(c(1:nComponents),nComponents),
                                                                                                                  i - c(selectionmat))];

                        matrixOfVarianceOfStates[,,i] <- transitionnew %*% matrixOfVarianceOfStatesLagged %*% t(transitionnew) + persistence %*% t(persistence) * s2;
                        varVec[i] <- measurementnew %*% matrixOfVarianceOfStatesLagged %*% t(measurementnew) + s2;
                        if(cumulative){
                            cumVarVec[i] <- ((measurementnew %*% matrixOfVarianceOfStatesLagged %*% t(measurementnew)) + s2) * m;
                            m <- m-1;
                        }
                    }
                }

                ### Cumulative variance is different.
                if(cumulative){
                    varVec <- sum(cumVarVec);
                }

                if(any(iprob!=1)){
                    quants <- qlnormBin(iprob, level=level, meanVec=rep(0,length(varVec)), sdVec=sqrt(varVec), Etype="A");
                    upper <- quants$upper;
                    lower <- quants$lower;
                }
                else{
                    # Take intermittent data into account
                    upper <- upperquant * sqrt(varVec);
                    lower <- lowerquant * sqrt(varVec);
                }
            }
        }
    }
##### If we were asked to produce 1 value #####
    else if(is.numeric(errors) & length(errors)>1 & !is.array(errors)){
        if(length(ev)>1){
            stop("Provided expected value doesn't correspond to the dimension of errors.", call.=FALSE);
        }

        if(intervalsType=="a"){
            upper <- ev + upperquant / hsmN^2 * Re(hm(errors,ev))^2;
            lower <- ev + lowerquant / hsmN^2 * Im(hm(errors,ev))^2;
        }
        else if(any(intervalsType==c("sp","p"))){
            if(Etype=="M"){
                if(any(iprob!=1)){
                    quants <- qlnormBin(iprob, level=level, meanVec=0, sdVec=sqrt(s2), Etype="M");
                    upper <- quants$upper;
                    lower <- quants$lower;
                }
                else{
                    upper <- y.for*qlnorm((1+level)/2,0,sqrt(s2));
                    lower <- y.for*qlnorm((1-level)/2,0,sqrt(s2));
                }
            }
            else{
                if(any(iprob!=1)){
                    quants <- qlnormBin(iprob, level=level, meanVec=ev, sdVec=sqrt(s2), Etype="A");
                    upper <- quants$upper;
                    lower <- quants$lower;
                }
                else{
                    upper <- ev + upperquant * sqrt(s2);
                    lower <- ev + lowerquant * sqrt(s2);
                }
            }
        }
        else if(intervalsType=="np"){
            if(Etype=="M"){
                errors <- errors + 1;
            }
            upper <- quantile(errors,(1+level)/2);
            lower <- quantile(errors,(1-level)/2);
        }
        varVec <- NULL;
    }
    else{
        stop("The provided data is not either vector or matrix. Can't do anything with it!", call.=FALSE);
    }

    return(list(upper=upper,lower=lower,variance=varVec));
}

##### *Forecaster of state-space functions* #####
ssForecaster <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    df <- (obsNonzero - nParam);
    if(df<=0){
        warning(paste0("Number of degrees of freedom is negative. It looks like we have overfitted the data."),call.=FALSE);
        df <- obsNonzero;
    }
# If error additive, estimate as normal. Otherwise - lognormal
    if(Etype=="A"){
        s2 <- as.vector(sum((errors*ot)^2)/df);
        s2g <- 1;
    }
    else{
        s2 <- as.vector(sum(log(1 + errors*ot)^2)/df);
        s2g <- log(1 + vecg %*% as.vector(errors*ot)) %*% t(log(1 + vecg %*% as.vector(errors*ot)))/df;
    }
    if((obsNonzero - nParam)<=0){
        df <- 0;
    }

    if(h>0){
        y.for <- ts(forecasterwrap(matrix(matvt[(obsInsample+1):(obsInsample+maxlag),],nrow=maxlag),
                                   matF, matw, h, Etype, Ttype, Stype, modellags,
                                   matrix(matxt[(obsAll-h+1):(obsAll),],ncol=nExovars),
                                   matrix(matat[(obsAll-h+1):(obsAll),],ncol=nExovars), matFX),
                    start=time(data)[obsInsample]+deltat(data),frequency=datafreq);

        y.forStart <- start(y.for);
        if(Etype=="M" & any(y.for<0)){
            warning(paste0("Negative values produced in forecast. This does not make any sense for model with multiplicative error.\n",
                           "Please, use another model."),call.=FALSE);
            if(intervals){
            warning("And don't expect anything reasonable from the prediction intervals!",call.=FALSE);
            }
        }

        # Write down the forecasting intervals
        if(intervals){
            if(h==1){
                errors.x <- as.vector(errors);
                ev <- median(errors);
            }
            else{
                errors.x <- errors.mat;
                ev <- apply(errors.mat,2,median,na.rm=TRUE);
            }
            if(intervalsType!="a"){
                ev <- 0;
            }

            # We don't simulate pure additive models, pure multiplicative and
            # additive models with multiplicative error on non-intermittent data, because they can be approximated by pure additive
            if(intervalsType=="p"){
                if(all(c(Etype,Stype,Ttype)!="M") |
                   all(c(Etype,Stype,Ttype)!="A") |
                   (all(Etype=="M",any(Ttype==c("A","N")),any(Stype==c("A","N"))) & s2<0.1)){
                    simulateIntervals <- FALSE;
                }
                else{
                    simulateIntervals <- TRUE;
                }
            }
            else{
                simulateIntervals <- FALSE;
            }

            if(cumulative){
               # & Etype=="M"){
               # & intervalsType=="p"){ <--- this is temporary. We do not know what cumulative means for multiplicative models.
                simulateIntervals <- TRUE;
            }

            #If this is integer-valued model, then do simulations
            # if(rounded){
            #     simulateIntervals <- TRUE;
            # }

            if(simulateIntervals==TRUE){
                nSamples <- 10000;
                matg <- matrix(vecg,nComponents,nSamples);
                arrvt <- array(NA,c(h+maxlag,nComponents,nSamples));
                arrvt[1:maxlag,,] <- rep(matvt[obsInsample+(1:maxlag),],nSamples);
                materrors <- matrix(rnorm(h*nSamples,0,sqrt(s2)),h,nSamples);

                if(Etype=="M"){
                    materrors <- exp(materrors) - 1;
                }
                if(all(intermittent!=c("n","p"))){
                    matot <- matrix(rbinom(h*nSamples,1,pt.for),h,nSamples);
                }
                else{
                    matot <- matrix(1,h,nSamples);
                }

                y.simulated <- simulatorwrap(arrvt,materrors,matot,array(matF,c(dim(matF),nSamples)),matw,matg,
                                             Etype,Ttype,Stype,modellags)$matyt;
                if(!is.null(xreg)){
                    y.exo.for <- c(y.for) - forecasterwrap(matrix(matvt[(obsInsample+1):(obsInsample+maxlag),],nrow=maxlag),
                                                           matF, matw, h, Etype, Ttype, Stype, modellags,
                                                           matrix(rep(1,h),ncol=1), matrix(rep(0,h),ncol=1), matrix(1,1,1));
                }
                else{
                    y.exo.for <- rep(0,h);
                }

                if(rounded){
                    y.simulated <- ceiling(y.simulated + matrix(y.exo.for,nrow=h,ncol=nSamples));
                    quantileType <- 1;
                    for(i in 1:h){
                        y.for[i] <- median(y.simulated[i,y.simulated[i,]!=0]);
                    }
                    # NA means that there were no non-zero demands
                    y.for[is.na(y.for)] <- 0;
                }
                else{
                    y.simulated <- y.simulated + matrix(y.exo.for,nrow=h,ncol=nSamples);
                    quantileType <- 7;
                }
                y.for <- c(pt.for)*y.for;

                if(cumulative){
                    y.for <- ts(sum(y.for),start=y.forStart,frequency=datafreq);
                    y.low <- ts(quantile(colSums(y.simulated,na.rm=T),(1-level)/2,type=quantileType),start=y.forStart,frequency=datafreq);
                    y.high <- ts(quantile(colSums(y.simulated,na.rm=T),(1+level)/2,type=quantileType),start=y.forStart,frequency=datafreq);
                }
                else{
                    y.for <- ts(y.for,start=y.forStart + y.exo.for,frequency=datafreq);
                    y.low <- ts(apply(y.simulated,1,quantile,(1-level)/2,na.rm=T,type=quantileType) + y.exo.for,start=y.forStart,frequency=datafreq);
                    y.high <- ts(apply(y.simulated,1,quantile,(1+level)/2,na.rm=T,type=quantileType) + y.exo.for,start=y.forStart,frequency=datafreq);
                }
                # For now we leave it as NULL
                varVec <- NULL;
            }
            else{
                quantvalues <- ssIntervals(errors.x, ev=ev, level=level, intervalsType=intervalsType, df=df,
                                           measurement=matw, transition=matF, persistence=vecg, s2=s2,
                                           modellags=modellags, states=matvt[(obsInsample-maxlag+1):obsInsample,],
                                           cumulative=cumulative,
                                           y.for=y.for, Etype=Etype, Ttype=Ttype, Stype=Stype, s2g=s2g,
                                           iprob=iprob, ivar=ivar);

                if(rounded){
                    y.for <- ceiling(y.for);
                }

                if(!(intervalsType=="sp" & Etype=="M")){
                    y.for <- c(pt.for)*y.for;
                }

                if(cumulative){
                    y.for <- ts(sum(y.for),start=y.forStart,frequency=datafreq);
                }

                if(Etype=="A"){
                    y.low <- ts(c(y.for) + quantvalues$lower,start=y.forStart,frequency=datafreq);
                    y.high <- ts(c(y.for) + quantvalues$upper,start=y.forStart,frequency=datafreq);
                }
                else{
                    if(any(intervalsType==c("np","sp","a"))){
                        quantvalues$upper <- quantvalues$upper * y.for;
                        quantvalues$lower <- quantvalues$lower * y.for;
                    }
                    y.low <- ts(quantvalues$lower,start=y.forStart,frequency=datafreq);
                    y.high <- ts(quantvalues$upper,start=y.forStart,frequency=datafreq);
                }

                if(rounded){
                    y.low <- ceiling(y.low);
                    y.high <- ceiling(y.high);
                }
                varVec <- quantvalues$variance;
            }
        }
        else{
            y.low <- NA;
            y.high <- NA;
            if(rounded){
                y.for <- ceiling(y.for);
            }
            y.for <- c(pt.for)*y.for;
            if(cumulative){
                y.for <- ts(sum(y.for),start=time(data)[obsInsample]+deltat(data),frequency=datafreq);
            }
            else{
                y.for <- ts(y.for,start=time(data)[obsInsample]+deltat(data),frequency=datafreq);
            }
            varVec <- NULL;
        }
    }
    else{
        y.low <- NA;
        y.high <- NA;
        y.for <- ts(NA,start=time(data)[obsInsample]+deltat(data),frequency=datafreq);
        # For now we leave it as NULL, because this thing is estimated in ssIntervals()
        varVec <- NULL;
    }

    if(any(is.na(y.fit),all(is.na(y.for),h>0))){
        warning("Something went wrong during the optimisation and NAs were produced!",call.=FALSE);
        warning("Please check the input and report this error to the maintainer if it persists.",call.=FALSE);
    }

    assign("s2",s2,ParentEnvironment);
    assign("y.for",y.for,ParentEnvironment);
    assign("y.low",y.low,ParentEnvironment);
    assign("y.high",y.high,ParentEnvironment);
    assign("varVec",varVec,ParentEnvironment);
}

##### *Check and initialisation of xreg* #####
ssXreg <- function(data, Etype="A", xreg=NULL, updateX=FALSE, ot=NULL,
                   persistenceX=NULL, transitionX=NULL, initialX=NULL,
                   obsInsample, obsAll, obsStates, maxlag=1, h=1, xregDo="u", silent=FALSE){
# The function does general checks needed for exogenouse variables and returns the list of necessary parameters

    if(!is.null(xreg)){
        xreg <- as.matrix(xreg);
        if(any(is.na(xreg))){
            warning("The exogenous variables contain NAs! This may lead to problems during estimation and in forecasting.\nSubstituting them with 0.",
                    call.=FALSE);
            xreg[is.na(xreg)] <- 0;
        }
        if(!is.null(dim(xreg))){
            if(ncol(xreg)==1){
                xreg <- as.vector(xreg);
            }
            if(length(dim(xreg))>2){
                stop(paste0("Sorry, but we don't deal with multidimensional arrays as exogenous variables here.\n",
                     "Pleas think of your behaviour and come back with matrix!"),call.=FALSE);
            }
        }
##### The case with vectors and ts objects, but not matrices
        if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
# Check if xreg contains something meaningful
            if(is.null(initialX)){
                if(all(xreg[1:obsInsample]==xreg[1])){
                    warning("The exogenous variable has no variability. Cannot do anything with that, so dropping out xreg.",
                            call.=FALSE);
                    xreg <- NULL;
                }
            }

            if(!is.null(xreg)){
                if(length(xreg) < obsAll){
                    warning("xreg did not contain values for the holdout, so we had to predict missing values.", call.=FALSE);
                    xregForecast <- es(xreg,h=obsAll-length(xreg),intermittent="auto",ic="AICc",silent=TRUE)$forecast;
                    xreg <- c(as.vector(xreg),as.vector(xregForecast));
                }
                else if(length(xreg) > obsAll){
                    warning("xreg contained too many observations, so we had to cut off some of them.", call.=FALSE);
                    xreg <- xreg[1:obsAll];
                }

                if(all(data[1:obsInsample]==xreg[1:obsInsample])){
                    warning("The exogenous variable and the forecasted data are exactly the same. What's the point of such a regression?",
                            call.=FALSE);
                    xreg <- NULL;
                }

# Number of exogenous variables
                nExovars <- 1;
# Define matrix w for exogenous variables
                matxt <- matrix(xreg,ncol=1);
# Define the second matat to fill in the coefs of the exogenous vars
                matat <- matrix(NA,obsStates,1);
# Fill in the initial values for exogenous coefs using OLS
                if(is.null(initialX)){
                    if(Etype=="A"){
                        matat[1:maxlag,] <- cov(data[1:obsInsample][ot==1],xreg[1:obsInsample][ot==1])/var(xreg[1:obsInsample][ot==1]);
                    }
                    else{
                        matat[1:maxlag,] <- cov(log(data[1:obsInsample][ot==1]),
                                                xreg[1:obsInsample][ot==1])/var(xreg[1:obsInsample][ot==1]);
                    }
                }
                if(is.null(names(xreg))){
                    colnames(matat) <- "x";
                    colnames(matxt) <- "x";
                }
                else{
                    xregNames <- gsub(" ", "_", names(xreg), fixed = TRUE);
                    colnames(matat) <- xregNames;
                    colnames(matxt) <- xregNames;
                }
            }
            xreg <- as.matrix(xreg);
        }
##### The case with matrices and data frames
        else if(is.matrix(xreg) | is.data.frame(xreg)){
            nExovars <- ncol(xreg);
            if(nrow(xreg) < obsAll){
                warning("xreg did not contain values for the holdout, so we had to predict missing values.", call.=FALSE);
                xregForecast <- matrix(NA,nrow=obsAll-nrow(xreg),ncol=nExovars);
                if(!silent){
                    message("Producing forecasts for xreg variable...");
                }
                for(j in 1:nExovars){
                    if(!silent){
                        cat(paste0(rep("\b",nchar(round((j-1)/nExovars,2)*100)+1),collapse=""));
                        cat(paste0(round(j/nExovars,2)*100,"%"));
                    }
                    xregForecast[,j] <- es(xreg[,j],h=obsAll-nrow(xreg),intermittent="auto",ic="AICc",silent=TRUE)$forecast;
                }
                xreg <- rbind(xreg,xregForecast);
                if(!silent){
                    cat("\b\b\b\bDone!\n");
                }
            }
            else if(nrow(xreg) > obsAll){
                warning("xreg contained too many observations, so we had to cut off some of them.", call.=FALSE);
                xreg <- xreg[1:obsAll,];
            }

            xregEqualToData <- apply(xreg[1:obsInsample,]==data[1:obsInsample],2,all);
            if(any(xregEqualToData)){
                warning("One of exogenous variables and the forecasted data are exactly the same. We have droped it.",
                        call.=FALSE);
                xreg <- matrix(xreg[,!xregEqualToData],nrow=nrow(xreg),ncol=ncol(xreg)-1,dimnames=list(NULL,colnames(xreg[,!xregEqualToData])));
            }

            nExovars <- ncol(xreg);

# If initialX is provided, then probably we don't need to check the xreg on variability and multicollinearity
            if(is.null(initialX)){
                checkvariability <- apply(xreg[1:obsInsample,]==rep(xreg[1,],each=obsInsample),2,all);
                if(any(checkvariability)){
                    if(all(checkvariability)){
                        warning("None of exogenous variables has variability. Cannot do anything with that, so dropping out xreg.",
                                call.=FALSE);
                        xreg <- NULL;
                        nExovars <- 0;
                    }
                    else{
                        warning("Some exogenous variables do not have any variability. Dropping them out.",
                                call.=FALSE);
                        xreg <- as.matrix(xreg[,!checkvariability]);
                        nExovars <- ncol(xreg);
                    }
                }

                if(!is.null(xreg)){
# Check for multicollinearity and drop something if there is a perfect one
                    corMatrix <- cor(xreg);
                    corCheck <- upper.tri(corMatrix) & abs(corMatrix)>=0.999;
                    if(any(corCheck)){
                        removexreg <- unique(which(corCheck,arr.ind=TRUE)[,1]);
                        if(ncol(xreg)-length(removexreg)>1){
                            xreg <- xreg[,-removexreg];
                        }
                        else{
                            xreg <- matrix(xreg[,-removexreg],ncol=ncol(xreg)-length(removexreg),
                                           dimnames=list(NULL,c(colnames(xreg)[-removexreg])));
                        }
                        nExovars <- ncol(xreg)
                        warning("Some exogenous variables were perfectly correlated. We've dropped them out.",
                                call.=FALSE);
                    }
                    # Check multiple correlations. This is needed for cases with dummy variables.
                    # In case with xregDo="select" some of the perfectly correlated things, will be dropped out automatically.
                    if(nExovars>2 & (xregDo=="u")){
                        corMatrix <- cor(xreg);
                        corMulti <- rep(NA,nExovars);
                        if(det(corMatrix)!=0){
                            for(i in 1:nExovars){
                                corMulti[i] <- 1 - det(corMatrix) / det(corMatrix[-i,-i]);
                            }
                            if(any(corMulti>=0.999)){
                                stop(paste0("Some combinations of exogenous variables are perfectly correlated. \n",
                                            "If you use sets of dummy variables, don't forget to drop some of them."),
                                     call.=FALSE);
                            }
                        }
                        else{
                            stop(paste0("Some combinations of exogenous variables are perfectly correlated. \n",
                                        "If you use sets of dummy variables, don't forget to drop some of them."),
                                 call.=FALSE);
                        }
                    }
                }
            }

            if(!is.null(xreg)){
# mat.x is needed for the initial values of coefs estimation using OLS
                mat.x <- as.matrix(cbind(rep(1,obsAll),xreg));
# Define the second matat to fill in the coefs of the exogenous vars
                matat <- matrix(NA,obsStates,nExovars);
# Define matrix w for exogenous variables
                matxt <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
                if(is.null(initialX)){
                    if(Etype=="A"){
                        matat[1:maxlag,] <- rep(t(solve(t(mat.x[1:obsInsample,][ot==1,]) %*% mat.x[1:obsInsample,][ot==1,],tol=1e-50) %*%
                                                      t(mat.x[1:obsInsample,][ot==1,]) %*% data[1:obsInsample][ot==1])[2:(nExovars+1)],
                                                each=maxlag);
                    }
                    else{
                        matat[1:maxlag,] <- rep(t(solve(t(mat.x[1:obsInsample,][ot==1,]) %*% mat.x[1:obsInsample,][ot==1,],tol=1e-50) %*%
                                                      t(mat.x[1:obsInsample,][ot==1,]) %*% log(data[1:obsInsample][ot==1]))[2:(nExovars+1)],
                                                each=maxlag);
                    }
                }
                if(is.null(colnames(xreg))){
                    colnames(matat) <- paste0("x",c(1:nExovars));
                    colnames(matxt) <- paste0("x",c(1:nExovars));
                }
                else{
                    xregNames <- gsub(" ", "_", colnames(xreg), fixed = TRUE);
                    if(xregDo=="s" & any(grepl('[^[:alnum:]]', xregNames))){
                        warning(paste0("There were some special characters in names of ",
                                       "xreg variables. We had to remove them."),call.=FALSE);
                        xregNames <- gsub("[^[:alnum:]]", "", xregNames);
                    }
                    colnames(matat) <- xregNames;
                    colnames(matxt) <- xregNames;
                }
            }
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=FALSE);
        }
        xregEstimate <- TRUE;

# Check the provided initialX vector
        if(!is.null(initialX)){
            if(!is.numeric(initialX) & !is.vector(initialX) & !is.matrix(initialX)){
                stop("The initials for exogenous are not a numeric vector or a matrix!", call.=FALSE);
            }
            else{
                if(length(initialX) != nExovars){
                    stop(paste0("The size of initial vector for exogenous is wrong!\n",
                                "It should correspond to the number of exogenous variables."), call.=FALSE);
                }
                else{
                    matat[1:maxlag,] <- as.vector(rep(initialX,each=maxlag));
                    initialXEstimate <- FALSE;
                }
            }
        }
        else{
            initialXEstimate <- TRUE;
        }
    }
    else{
        updateX <- FALSE;
    }

##### In case we changed xreg to null or if it was like that...
    if(is.null(xreg)){
# "1" is needed for the final forecast simplification
        nExovars <- 1;
        matxt <- matrix(1,obsStates,1);
        matat <- matrix(0,obsStates,1);
        matFX <- matrix(1,1,1);
        vecgX <- matrix(0,1,1);
        xregEstimate <- FALSE;
        FXEstimate <- FALSE;
        gXEstimate <- FALSE;
        initialXEstimate <- FALSE;
    }

# Now check transition and persistence of exogenous variables
    if(xregEstimate & updateX){
# First - transition matrix
        if(!is.null(transitionX)){
            if(!is.numeric(transitionX) & !is.vector(transitionX) & !is.matrix(transitionX)){
                stop("Transition matrix for exogenous is not a numeric vector or a matrix!", call.=FALSE);
            }
            else{
                if(length(transitionX) != nExovars^2){
                    stop(paste0("Size of transition matrix for exogenous is wrong!\n",
                                "It should correspond to the number of exogenous variables."), call.=FALSE);
                }
                else{
                    matFX <- matrix(transitionX,nExovars,nExovars);
                    FXEstimate <- FALSE;
                }
            }
        }
        else{
            matFX <- diag(nExovars);
            FXEstimate <- TRUE;
        }
# Now - persistence vector
        if(!is.null(persistenceX)){
            if(!is.numeric(persistenceX) & !is.vector(persistenceX) & !is.matrix(persistenceX)){
                stop("Persistence vector for exogenous is not numeric!", call.=FALSE);
            }
            else{
                if(length(persistenceX) != nExovars){
                    stop(paste0("Size of persistence vector for exogenous is wrong!\n",
                                "It should correspond to the number of exogenous variables."), call.=FALSE);
                }
                else{
                    vecgX <- matrix(persistenceX,nExovars,1);
                    gXEstimate <- FALSE;
                }
            }
        }
        else{
            vecgX <- matrix(0,nExovars,1);
            gXEstimate <- TRUE;
        }
    }
    else if(xregEstimate & !updateX){
        matFX <- diag(nExovars);
        FXEstimate <- FALSE;

        vecgX <- matrix(0,nExovars,1);
        gXEstimate <- FALSE;
    }

    if(all(!FXEstimate,!gXEstimate,!initialXEstimate)){
        xregEstimate <- FALSE;
    }

    return(list(nExovars=nExovars, matxt=matxt, matat=matat, matFX=matFX, vecgX=vecgX,
                xreg=xreg, xregEstimate=xregEstimate, FXEstimate=FXEstimate,
                gXEstimate=gXEstimate, initialXEstimate=initialXEstimate))
}

##### *Likelihood function* #####
likelihoodFunction <- function(C){
# This block is needed in order to make R CMD to shut up about "no visible binding..."
    if(any(intermittent==c("n","p"))){
        if(cfType=="TFL" | cfType=="aTFL"){
            return(- obsNonzero/2 *(h*log(2*pi*exp(1)) + CF(C)));
        }
        else{
            return(- obsNonzero/2 *(log(2*pi*exp(1)) + log(CF(C))));
        }
    }
    else{
        #Failsafe for exceptional cases when the probability is equal to zero / one, when it should not have been.
        if(any(c(1-pt[ot==0]==0,pt[ot==1]==0))){
            # return(-Inf);
            ptNew <- pt[(pt!=0) & (pt!=1)];
            otNew <- ot[(pt!=0) & (pt!=1)];
            if(length(ptNew)==0){
                return(-obsNonzero/2 *(log(2*pi*exp(1)) + log(CF(C))));
            }
            else{
                return(sum(log(ptNew[otNew==1])) + sum(log(1-ptNew[otNew==0]))
                       - obsNonzero/2 *(log(2*pi*exp(1)) + log(CF(C))));
            }
        }
        #Failsafe for cases, when data has no variability when ot==1.
        if(CF(C)==0){
            if(cfType=="TFL" | cfType=="aTFL"){
                return(sum(log(pt[ot==1]))*h + sum(log(1-pt[ot==0]))*h);
            }
            else{
                return(sum(log(pt[ot==1])) + sum(log(1-pt[ot==0])));
            }
        }
        if(rounded){
            return(sum(log(pt[ot==1])) + sum(log(1-pt[ot==0])) - CF(C));
        }
        if(cfType=="TFL" | cfType=="aTFL"){
            return(sum(log(pt[ot==1]))*h
                   + sum(log(1-pt[ot==0]))*h
                   - obsNonzero/2 * (h*log(2*pi*exp(1)) + CF(C)));
        }
        else{
            return(sum(log(pt[ot==1])) + sum(log(1-pt[ot==0]))
                   - obsNonzero/2 *(log(2*pi*exp(1)) + log(CF(C))));
        }
    }
}

##### *Function calculates ICs* #####
ICFunction <- function(nParam=nParam,C,Etype=Etype){
# Information criteria are calculated with the constant part "log(2*pi*exp(1)*h+log(obs))*obs".
# And it is based on the mean of the sum squared residuals either than sum.
# Hyndman likelihood is: llikelihood <- obs*log(obs*cfObjective)

    llikelihood <- likelihoodFunction(C);

    AIC.coef <- 2*nParam*h^multisteps - 2*llikelihood;
# max here is needed in order to take into account cases with higher number of parameters than observations
    AICc.coef <- AIC.coef + 2 * nParam*h^multisteps * (nParam + 1) / max(obsNonzero - nParam - 1,0);
    BIC.coef <- log(obsNonzero)*nParam*h^multisteps - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

    return(list(llikelihood=llikelihood,ICs=ICs));
}

##### *Ouptut printer* #####
ssOutput <- function(timeelapsed, modelname, persistence=NULL, transition=NULL, measurement=NULL,
                     phi=NULL, ARterms=NULL, MAterms=NULL, constant=NULL, A=NULL, B=NULL, initialType="o",
                     nParam=NULL, s2=NULL, hadxreg=FALSE, wentwild=FALSE,
                     cfType="MSE", cfObjective=NULL, intervals=FALSE, cumulative=FALSE,
                     intervalsType=c("n","p","sp","np","a"), level=0.95, ICs,
                     holdout=FALSE, insideintervals=NULL, errormeasures=NULL,
                     intermittent="n"){
# Function forms the generic output for State-space models.
    if(gregexpr("ETS",modelname)!=-1){
        model <- "ETS";
    }
    else if(gregexpr("CES",modelname)!=-1){
        model <- "CES";
    }
    else if(gregexpr("GES",modelname)!=-1){
        model <- "GES";
    }
    else if(gregexpr("ARIMA",modelname)!=-1){
        model <- "ARIMA";
    }
    else if(gregexpr("SMA",modelname)!=-1){
        model <- "SMA";
    }

    cat(paste0("Time elapsed: ",round(as.numeric(timeelapsed,units="secs"),2)," seconds\n"));
    cat(paste0("Model estimated: ",modelname,"\n"));
    if(all(intermittent!=c("n","none","provided"))){
        if(any(intermittent==c("f","fixed"))){
            intermittent <- "Fixed probability";
        }
        else if(any(intermittent==c("i","interval"))){
            intermittent <- "Interval-based";
        }
        else if(any(intermittent==c("p","probability"))){
            intermittent <- "Probability-based";
        }
        else if(any(intermittent==c("s","sba"))){
            intermittent <- "SBA";
        }
        cat(paste0("Intermittent model type: ",intermittent));
        cat("\n");
    }
    else if(any(intermittent==c("provided"))){
        cat(paste0("Intermittent data provided for holdout.\n"));
    }

### Stuff for ETS
    if(any(model==c("ETS","GES"))){
        if(!is.null(persistence)){
            cat(paste0("Persistence vector g:\n"));
            if(is.matrix(persistence)){
                print(round(t(persistence),3));
            }
            else{
                print(round(persistence,3));
            }
        }
        if(!is.null(phi)){
            if(gregexpr("d",modelname)!=-1){
                cat(paste0("Damping parameter: ", round(phi,3),"\n"));
            }
        }
    }

### Stuff for GES
    if(model=="GES"){
        if(!is.null(transition)){
            cat("Transition matrix F: \n");
            print(round(transition,3));
        }
        if(!is.null(measurement)){
            cat(paste0("Measurement vector w: ",paste(round(measurement,3),collapse=", "),"\n"));
        }
    }

### Stuff for ARIMA
    if(model=="ARIMA"){
        if(all(!is.null(ARterms))){
            cat("Matrix of AR terms:\n");
            print(round(ARterms,3));
        }
        if(all(!is.null(MAterms))){
            cat("Matrix of MA terms:\n");
            print(round(MAterms,3));
        }
        if(!is.null(constant)){
            if(constant!=FALSE){
                cat(paste0("Constant value is: ",round(constant,3),"\n"));
            }
        }
    }
### Stuff for CES
    if(model=="CES"){
        if(!is.null(A)){
            cat(paste0("a0 + ia1: ",round(A,5),"\n"));
        }
        if(!is.null(B)){
            if(is.complex(B)){
                cat(paste0("b0 + ib1: ",round(B,5),"\n"));
            }
            else{
                cat(paste0("b: ",round(B,5),"\n"));
            }
        }
    }

    if(initialType=="o"){
        cat("Initial values were optimised.\n");
    }
    else if(initialType=="b"){
        cat("Initial values were produced using backcasting.\n");
    }
    else if(initialType=="p"){
        cat("Initial values were provided by user.\n");
    }

    if(!is.null(nParam)){
        if(nParam[1,4]==1){
            cat(paste0(nParam[1,4]," parameter was estimated in the process\n"));
        }
        else{
            cat(paste0(nParam[1,4]," parameters were estimated in the process\n"));
        }

        if(nParam[2,4]>1){
            cat(paste0(nParam[2,4]," parameters were provided\n"));
        }
        else if(nParam[2,4]>0){
            cat(paste0(nParam[2,4]," parameter was provided\n"));
        }
    }

    if(!is.null(s2)){
        cat(paste0("Residuals standard deviation: ",round(sqrt(s2),3),"\n"));
    }

    if(hadxreg==TRUE){
        cat("Xreg coefficients were estimated");
        if(wentwild==TRUE){
            cat(" in a crazy style\n");
        }
        else{
            cat(" in a normal style\n");
        }
    }

    cat(paste0("Cost function type: ",cfType))
    if(!is.null(cfObjective)){
        cat(paste0("; Cost function value: ",round(cfObjective,3),"\n"));
    }
    else{
        cat("\n");
    }

    cat("\nInformation criteria:\n");
    print(ICs);

    if(intervals){
        if(intervalsType=="p"){
            intervalsType <- "parametric";
        }
        else if(intervalsType=="sp"){
            intervalsType <- "semiparametric";
        }
        else if(intervalsType=="np"){
            intervalsType <- "nonparametric";
        }
        else if(intervalsType=="a"){
            intervalsType <- "asymmetric";
        }
        if(cumulative){
            intervalsType <- paste0("cumulative ",intervalsType);
        }
        cat(paste0(level*100,"% ",intervalsType," prediction intervals were constructed\n"));
    }

    if(holdout){
        if(intervals & !is.null(insideintervals)){
            cat(paste0(round(insideintervals,0), "% of values are in the prediction interval\n"));
        }
        cat("Forecast errors:\n");
        if(any(intermittent==c("none","n"))){
            cat(paste(paste0("MPE: ",errormeasures["MPE"]*100,"%"),
                      paste0("Bias: ",errormeasures["cbias"]*100,"%"),
                      paste0("MAPE: ",errormeasures["MAPE"]*100,"%"),
                      paste0("SMAPE: ",errormeasures["SMAPE"]*100,"%\n"),sep="; "));
            cat(paste(paste0("MASE: ",errormeasures["MASE"]),
                      paste0("sMAE: ",errormeasures["sMAE"]*100,"%"),
                      paste0("RelMAE: ",errormeasures["RelMAE"]),
                      paste0("sMSE: ",errormeasures["sMSE"]*100,"%\n"),sep="; "));
        }
        else{
            cat(paste(paste0("Bias: ",errormeasures["cbias"]*100,"%"),
                      paste0("sMSE: ",errormeasures["sMSE"]*100,"%"),
                      paste0("sPIS: ",errormeasures["sPIS"]*100,"%"),
                      paste0("sCE: ",errormeasures["sCE"]*100,"%\n"),sep="; "));
        }
    }
}
