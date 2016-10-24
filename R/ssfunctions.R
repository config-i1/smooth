utils::globalVariables(c("h","holdout","orders","lags","transition","measurement","multisteps","ot","obsInsample","obsAll",
                         "obsStates","obsNonzero","pt","cfType","CF","Etype","Ttype","Stype","matxt","matFX","vecgX","xreg",
                         "matvt","n.exovars","matat","errors","n.param","intervals","intervalsType","level","ivar","model",
                         "constant","AR","MA","data"));

##### *Checker of input of basic functions* #####
ssInput <- function(modelType=c("es","ges","ces","ssarima"),...){
    # This is universal function needed in order to check the passed arguments to es(), ges(), ces() and ssarima()

    modelType <- modelType[1];

    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ##### silent #####
    silent <- silent[1];
    # Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output","n","a","g","l","o"))){
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

    ##### data #####
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

    if(modelType=="es"){
        ##### model for ES #####
        if(!is.character(model)){
            stop(paste0("Something strange is provided instead of character object in model: ",
                        paste0(model,collapse=",")),call.=FALSE);
        }

        # Predefine models pool for a model selection
        models.pool <- NULL;
        # Deal with the list of models. Check what has been provided. Stop if there is a mistake.
        if(length(model)>1){
            if(any(nchar(model)>4)){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)>4],collapse=",")),call.=FALSE);
            }
            else if(any(substr(model,1,1)!="A" & substr(model,1,1)!="M")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,1,1)!="A" & substr(model,1,1)!="M"],collapse=",")),call.=FALSE);
            }
            else if(any(substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                        substr(model,2,2)!="M")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                                             substr(model,2,2)!="M"],collapse=",")),call.=FALSE);
            }
            else if(any(substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                        substr(model,3,3)!="M" & substr(model,3,3)!="d")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                                             substr(model,3,3)!="M" & substr(model,3,3)!="d"],collapse=",")),call.=FALSE);
            }
            else if(any(nchar(model)==4 & substr(model,4,4)!="N" &
                        substr(model,4,4)!="A" & substr(model,4,4)!="M")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)==4 & substr(model,4,4)!="N" &
                                             substr(model,4,4)!="A" & substr(model,4,4)!="M"],collapse=",")),call.=FALSE);
            }
            else{
                models.pool <- model;
            }
            model <- "ZZZ";
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
        if(is.null(models.pool)){
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
                models.pool <- c("ANN","MNN","AAN","AMN","MAN","MMN","AAdN","AMdN","MAdN","MMdN","ANA","ANM","MNA","MNM",
                                 "AAA","AAM","AMA","AMM","MAA","MAM","MMA","MMM",
                                 "AAdA","AAdM","AMdA","AMdM","MAdA","MAdM","MMdA","MMdM");
                if(datafreq==1){
                    Stype <- "N";
                }
                # Restrict error types in the pool
                if(Etype=="X"){
                    models.pool <- models.pool[substr(models.pool,1,1)=="A"];
                    Etype <- "Z";
                }
                else if(Etype=="Y"){
                    models.pool <- models.pool[substr(models.pool,1,1)=="M"];
                    Etype <- "Z";
                }
                else{
                    if(Etype!="Z"){
                        models.pool <- models.pool[substr(models.pool,1,1)==Etype];
                    }
                }
                # Restrict trend types in the pool
                if(Ttype=="X"){
                    models.pool <- models.pool[substr(models.pool,2,2)=="A" | substr(models.pool,2,2)=="N"];
                    Ttype <- "Z";
                }
                else if(Ttype=="Y"){
                    models.pool <- models.pool[substr(models.pool,2,2)=="M" | substr(models.pool,2,2)=="N"];
                    Ttype <- "Z";
                }
                else{
                    if(Ttype!="Z"){
                        models.pool <- models.pool[substr(models.pool,2,2)==Ttype];
                        if(damped){
                            models.pool <- models.pool[nchar(models.pool)==4];
                        }
                    }
                }
                # Restrict season types in the pool
                if(Stype=="X"){
                    models.pool <- models.pool[substr(models.pool,nchar(models.pool),nchar(models.pool))=="A" |
                                               substr(models.pool,nchar(models.pool),nchar(models.pool))=="N" ];
                    Stype <- "Z";
                }
                else if(Stype=="Y"){
                    models.pool <- models.pool[substr(models.pool,nchar(models.pool),nchar(models.pool))=="M" |
                                               substr(models.pool,nchar(models.pool),nchar(models.pool))=="N" ];
                    Stype <- "Z";
                }
                else{
                    if(Stype!="Z"){
                        models.pool <- models.pool[substr(models.pool,nchar(models.pool),nchar(models.pool))==Stype];
                    }
                }
            }
        }
        else{
            modelDo <- "select";
        }

        ### Check error type
        if(all(Etype!=c("Z","X","Y","A","M"))){
            warning(paste0("Wrong error type: ",Etype,". Should be 'Z', 'X', 'Y', 'A' or 'M'.\n",
                           "Changing to 'Z'"),call.=FALSE);
            Etype <- "Z";
        }

        ### Check trend type
        if(all(Ttype!=c("Z","X","Y","N","A","M"))){
            warning(paste0("Wrong trend type: ",Ttype,". Should be 'Z', 'X', 'Y', 'N', 'A' or 'M'.\n",
                           "Changing to 'Z'"),call.=FALSE);
            Ttype <- "Z";
        }
    }
    else if(modelType=="ssarima"){
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

        if(length(lags)!=length(ar.orders) & length(lags)!=length(i.orders) & length(lags)!=length(ma.orders)){
            stop("Seasonal lags do not correspond to any element of SARIMA",call.=FALSE);
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
                if(sum(ar.orders)!=length(ARValue[ARValue!=0])){
                    warning(paste0("Wrong number of non-zero elements of AR. Should be ",sum(ar.orders),
                                    " instead of ",length(ARValue[ARValue!=0]),".\n",
                                   "AR will be estimated."),call.=FALSE);
                    ARRequired <- AREstimate <- TRUE;
                    ARValue <- NULL;
                }
                else{
                    ARValue <- ARValue[ARValue!=0];
                    AREstimate <- FALSE;
                    ARRequired <- TRUE;
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
                if(sum(ma.orders)!=length(MAValue[MAValue!=0])){
                    warning(paste0("Wrong number of non-zero elements of MA. Should be ",sum(ma.orders),
                                    " instead of ",length(MAValue[MAValue!=0]),".\n",
                                   "MA will be estimated."),call.=FALSE);
                    MARequired <- MAEstimate <- TRUE;
                    MAValue <- NULL;
                }
                else{
                    MAValue <- MAValue[MAValue!=0];
                    MAEstimate <- FALSE;
                    MARequired <- TRUE;
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
        }
        else if(is.logical(constantValue)){
            constantRequired <- constantEstimate <- constantValue;
            constantValue <- NULL;
        }

        # Number of components to use
        n.components <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
        modellags <- matrix(rep(1,times=n.components),ncol=1);
        if(constantRequired==TRUE){
            modellags <- rbind(modellags,1);
        }
        maxlag <- 1;
    }
    else if(modelType=="ces"){
        # If the user typed wrong seasonality, use the "Full" instead
        if(all(seasonality!=c("n","s","p","f","none","simple","partial","full"))){
            warning(paste0("Wrong seasonality type: '",seasonality, "'. Changing to 'full'"), call.=FALSE);
            seasonality <- "f";
        }
        seasonality <- substring(seasonality[1],1,1);
    }

    if(modelType=="es"){
        # Check if the data is ts-object
        if(!is.ts(data) & Stype!="N"){
            if(!silentText){
                message("The provided data is not ts object. Only non-seasonal models are available.");
            }
            Stype <- "N";
        }

        ### Check seasonality type
        if(all(Stype!=c("Z","X","Y","N","A","M"))){
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
    else if(modelType=="sma"){
        maxlag <- 1;
        if(is.null(order)){
            n.param.max <- obsInsample;
        }
        else{
            n.param.max <- order;
        }
    }

    ##### Lags and components for GES #####
    if(modelType=="ges"){
        if(any(is.complex(c(orders,lags)))){
            stop("Complex values? Right! Come on! Be real!",call.=FALSE);
        }
        if(any(c(orders)<0)){
            stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
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
        n.components <- sum(orders);
    }
    else if(modelType=="es"){
        maxlag <- datafreq * (Stype!="N") + 1 * (Stype=="N");
    }
    else if(modelType=="ces"){
        A <- list(value=A);
        B <- list(value=B);

        if(is.null(A$value)){
            A$estimate <- TRUE;
        }
        else{
            A$estimate <- FALSE;
        }
        if(all(is.null(B$value),any(seasonality==c("p","f")))){
            B$estimate <- TRUE;
        }
        else{
            B$estimate <- FALSE;
        }

        # Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
        # lags is the lags used in pt matrix.
        if(seasonality=="n"){
            # No seasonality
            maxlag <- 1;
            modellags <- c(1,1);
            ces.name <- "Complex Exponential Smoothing";
            # Define the number of all the parameters (smoothing parameters + initial states). Used in AIC mainly!
            n.components <- 2;
            A$number <- 2;
            B$number <- 0;
        }
        else if(seasonality=="s"){
            # Simple seasonality, lagged CES
            maxlag <- datafreq;
            modellags <- c(maxlag,maxlag);
            ces.name <- "Lagged Complex Exponential Smoothing (Simple seasonality)";
            n.components <- 2;
            A$number <- 2;
            B$number <- 0;
        }
        else if(seasonality=="p"){
            # Partial seasonality with a real part only
            maxlag <- datafreq;
            modellags <- c(1,1,maxlag);
            ces.name <- "Complex Exponential Smoothing with a partial (real) seasonality";
            n.components <- 3;
            A$number <- 2;
            B$number <- 1;
        }
        else if(seasonality=="f"){
            # Full seasonality with both real and imaginary parts
            maxlag <- datafreq;
            modellags <- c(1,1,maxlag,maxlag);
            ces.name <- "Complex Exponential Smoothing with a full (complex) seasonality";
            n.components <- 4;
            A$number <- 2;
            B$number <- 2;
        }
    }

    ##### obsStates #####
    # Define the number of rows that should be in the matvt
    obsStates <- max(obsAll + maxlag, obsInsample + 2*maxlag);

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

    ##### bounds #####
    bounds <- substring(bounds[1],1,1);
    if(bounds!="u" & bounds!="a" & bounds!="n"){
        warning("Strange bounds are defined. Switching to 'usual'.",call.=FALSE);
        bounds <- "u";
    }

    if(modelType=="es"){
        ##### Information Criteria #####
        ic <- ic[1];
        if(all(ic!=c("AICc","AIC","BIC"))){
            warning(paste0("Strange type of information criteria defined: ",ic,". Switching to 'AICc'."),call.=FALSE);
            ic <- "AICc";
        }
    }

    ##### Cost function type #####
    cfType <- cfType[1];
    if(any(cfType==c("MLSTFE","MSTFE","TFL","MSEh","aMLSTFE","aMSTFE","aTFL","aMSEh"))){
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
        }

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
        intermittent <- "p";
        n.param.intermittent <- 0;
    }
    else{
        intermittent <- intermittent[1];
        if(all(intermittent!=c("n","f","c","t","a","none","fixed","croston","tsb","auto"))){
            warning(paste0("Strange type of intermittency defined: '",intermittent,"'. Switching to 'fixed'."),
                    call.=FALSE);
            intermittent <- "f";
        }
        intermittent <- substring(intermittent[1],1,1);

        environment(intermittentParametersSetter) <- environment();
        intermittentParametersSetter(intermittent,ParentEnvironment=environment());

        if(obsNonzero <= n.param.intermittent){
            warning(paste0("Not enough observations for estimation of occurence probability.\n",
                           "Switching to simpler model."),
                    call.=FALSE);
            if(obsNonzero > 1){
                intermittent <- "f";
                n.param.intermittent <- 1;
                intermittentParametersSetter(intermittent,ParentEnvironment=environment());
            }
            else{
                intermittent <- "n";
                intermittentParametersSetter(intermittent,ParentEnvironment=environment());
            }
        }
    }

    # If the data is not intermittent, let's assume that the parameter was switched unintentionally.
    if(pt[1,]==1 & all(intermittent!=c("n","p"))){
        intermittent <- "n";
    }

    if(any(modelType==c("es"))){
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

    if(any(modelType==c("es","ges"))){
        ##### persistence for ES & GES #####
        if(!is.null(persistence)){
            if((!is.numeric(persistence) | !is.vector(persistence)) & !is.matrix(persistence)){
                warning(paste0("Persistence is not a numeric vector!\n",
                               "Changing to estimation of persistence vector values."),call.=FALSE);
                persistence <- NULL;
                persistenceEstimate <- TRUE;
            }
            else{
                if(modelType=="es"){
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
                            persistenceEstimate <- FALSE;
                        }
                    }
                }
                else if(modelType=="ges"){
                    if(length(persistence) != n.components){
                        warning(paste0("Wrong length of persistence vector. Should be ",n.components,
                                       " instead of ",length(persistence),".\n",
                                       "Changing to estimation of persistence vector values."),call.=FALSE);
                        persistence <- NULL;
                        persistenceEstimate <- TRUE;
                    }
                    else{
                        persistenceEstimate <- FALSE;
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
            if(modelType=="es"){
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
                    }
                }
            }
            else if(modelType=="ges"){
                if(length(initialValue) != (n.components*max(lags))){
                    warning(paste0("Wrong length of initial vector. Should be ",orders %*% lags,
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    initialValue <- initial;
                }
            }
            else if(modelType=="ssarima"){
                if(length(initialValue) != n.components){
                    warning(paste0("Wrong length of initial vector. Should be ",n.components,
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    initialValue <- initial;
                }
            }
            else if(modelType=="ces"){
                if(length(initialValue) != maxlag*n.components){
                    warning(paste0("Wrong length of initial vector. Should be ",n.components,
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    initialValue <- initial;
                }
            }
        }
    }

    if(any(modelType==c("es"))){
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
                warning(paste0("The length of initialSeason vector is wrong! It should correspond to the frequency of the data.",
                               "Values of initialSeason vector will be estimated."),call.=FALSE);
                    initialSeason <- NULL;
                    initialSeasonEstimate <- TRUE
                }
                else{
                    initialSeasonEstimate <- FALSE;
                }
            }
        }
        else{
            initialSeasonEstimate <- TRUE;
        }

        # Check the length of the provided data. Say bad words if:
        # 1. Seasonal model, <=2 seasons of data and no initial seasonals.
        # 2. Seasonal model, <=1 season of data, no initial seasonals and no persistence.
        if(is.null(models.pool)){
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
            }
        }
        else{
            phiEstimate <- TRUE;
        }
    }

    if(modelType=="ges"){
        ##### transition for GES #####
        # Check the provided vector of initials: length and provided values.
        if(!is.null(transition)){
            if((!is.numeric(transition) | !is.vector(transition)) & !is.matrix(transition)){
                warning(paste0("Transition matrix is not numeric!\n",
                               "The matrix will be estimated!"),call.=FALSE);
                transitionEstimate <- TRUE;
            }
            else if(length(transition) != n.components^2){
                warning(paste0("Wrong length of transition matrix. Should be ",n.components^2,
                               " instead of ",length(transition),".\n",
                               "The matrix will be estimated!"),call.=FALSE);
                transitionEstimate <- TRUE;
            }
            else{
                transitionEstimate <- FALSE;
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
                transitionEstimate <- TRUE;
            }
            else if(length(measurement) != n.components){
                warning(paste0("Wrong length of measurement vector. Should be ",n.components,
                               " instead of ",length(measurement),".\n",
                               "The vector will be estimated!"),call.=FALSE);
                transitionEstimate <- TRUE;
            }
            else{
                measurementEstimate <- FALSE;
            }
        }
        else{
            measurementEstimate <- TRUE;
        }
    }

    if(modelType=="ssarima"){
        if((n.components==0) & (constantRequired==FALSE)){
            warning("You have not defined any model! Constructing model with zero constant.",call.=FALSE);
            constantRequired <- TRUE;
            constantValue <- 0;
            initialType <- "p";
        }
    }

    ##### Calculate n.param.max for checks #####
    if(modelType=="es"){
        # 1 - 3: persitence vector;
        # 1 - 2: initials;
        # 1 - 1 phi value;
        # datafreq: datafreq initials for seasonal component;
        # 1: estimation of variance;
        n.param.max <- (1 + (Ttype!="N") + (Stype!="N"))*persistenceEstimate +
            (1 + (Ttype!="N"))*(initialType=="o") +
            phiEstimate*damped + datafreq*(Stype!="N")*initialSeasonEstimate*(initialType=="o") + 1;
    }
    else if(modelType=="ges"){
        n.param.max <- n.components*measurementEstimate + n.components*(initialType=="o") +
            transitionEstimate*n.components^2 + (orders %*% lags)*persistenceEstimate + 1;
    }
    else if(modelType=="ssarima"){
        n.param.max <- n.components*(initialType=="o") + sum(ar.orders)*ARRequired +
            sum(ma.orders)*MARequired + constantRequired + 1;
    }
    else if(modelType=="ces"){
        n.param.max <- sum(modellags)*(initialType=="o") + A$number + B$number + 1;
    }

    # Stop if number of observations is less than horizon and multisteps is chosen.
    if((multisteps==TRUE) & (obsNonzero < h+1) & all(cfType!=c("aMSEh","aTFL","aMSTFE","aMLSTFE"))){
        warning(paste0("Do you seriously think that you can use ",cfType,
                       " with h=",h," on ",obsNonzero," non-zero observations?!"),call.=FALSE);
        stop("Not enough observations for multisteps cost function.",call.=FALSE);
    }
    else if((multisteps==TRUE) & (obsNonzero < 2*h) & all(cfType!=c("aMSEh","aTFL","aMSTFE","aMLSTFE"))){
        warning(paste0("Number of observations is really low for a multisteps cost function! ",
                       "We will, try but cannot guarantee anything..."),call.=FALSE);
    }

    normalizer <- mean(abs(diff(c(y))));

    ##### Return values to previous environment #####
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
    assign("FI",FI,ParentEnvironment);
    assign("bounds",bounds,ParentEnvironment);
    assign("cfType",cfType,ParentEnvironment);
    assign("cfTypeOriginal",cfTypeOriginal,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("intervalsType",intervalsType,ParentEnvironment);
    assign("intervals",intervals,ParentEnvironment);
    assign("intermittent",intermittent,ParentEnvironment);
    assign("ot",ot,ParentEnvironment);
    assign("yot",yot,ParentEnvironment);
    assign("pt",pt,ParentEnvironment);
    assign("pt.for",pt.for,ParentEnvironment);
    assign("n.param.intermittent",n.param.intermittent,ParentEnvironment);
    assign("iprob",iprob,ParentEnvironment);
    assign("initialValue",initialValue,ParentEnvironment);
    assign("initialType",initialType,ParentEnvironment);
    assign("normalizer",normalizer,ParentEnvironment);
    assign("n.param.max",n.param.max,ParentEnvironment);

    if(modelType=="es"){
        assign("model",model,ParentEnvironment);
        assign("models.pool",models.pool,ParentEnvironment);
        assign("Etype",Etype,ParentEnvironment);
        assign("Ttype",Ttype,ParentEnvironment);
        assign("Stype",Stype,ParentEnvironment);
        assign("damped",damped,ParentEnvironment);
        assign("modelDo",modelDo,ParentEnvironment);
        assign("initialSeason",initialSeason,ParentEnvironment);
        assign("phi",phi,ParentEnvironment);
        assign("allowMultiplicative",allowMultiplicative,ParentEnvironment);
        assign("ic",ic,ParentEnvironment);
    }
    else if(modelType=="ges"){
        assign("transitionEstimate",transitionEstimate,ParentEnvironment);
        assign("measurementEstimate",measurementEstimate,ParentEnvironment);
        assign("orders",orders,ParentEnvironment);
        assign("lags",lags,ParentEnvironment);
    }
    else if(modelType=="ssarima"){
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
    else if(modelType=="ces"){
        assign("seasonality",seasonality,ParentEnvironment);
        assign("ces.name",ces.name,ParentEnvironment);
        assign("A",A,ParentEnvironment);
        assign("B",B,ParentEnvironment);
    }

    if(any(modelType==c("es","ges"))){
        assign("persistence",persistence,ParentEnvironment);
        assign("persistenceEstimate",persistenceEstimate,ParentEnvironment);
    }

    if(any(modelType==c("ges","ssarima","ces"))){
        assign("n.components",n.components,ParentEnvironment);
        assign("maxlag",maxlag,ParentEnvironment);
        assign("modellags",modellags,ParentEnvironment);
    }
}

##### *Checker for auto. functions* #####
ssAutoInput <- function(modelType=c("auto.ces","auto.ges","auto.ssarima"),...){
    # This is universal function needed in order to check the passed arguments to auto.ces(), auto.ges() and auto.ssarima()

    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ##### silent #####
    silent <- silent[1];
    # Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output","n","a","g","l","o"))){
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

    ##### Fisher Information #####
    if(!exists("FI")){
        FI <- FALSE;
    }

    ##### data #####
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
        warning("Predefinde initials don't go well with automatic model selection. Switching to optimal.",call.=FALSE);
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
    if(any(cfType==c("MLSTFE","MSTFE","TFL","MSEh","aMLSTFE","aMSTFE","aTFL","aMSEh"))){
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

    ##### intermittent #####
    if(is.numeric(intermittent)){
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
        y <- data[1:obsInsample];
        obsNonzero <- sum((y!=0)*1);
        intermittent <- intermittent[1];
        if(all(intermittent!=c("n","f","c","t","a","none","fixed","croston","tsb","auto"))){
            warning(paste0("Strange type of intermittency defined: '",intermittent,"'. Switching to 'fixed'."),
                    call.=FALSE);
            intermittent <- "f";
        }
        intermittent <- substring(intermittent[1],1,1);
        environment(intermittentParametersSetter) <- environment();
        intermittentParametersSetter(intermittent,ParentEnvironment=environment());

        if(obsNonzero <= n.param.intermittent){
            warning(paste0("Not enough observations for estimation of occurence probability.\n",
                           "Switching to simpler model."),
                    call.=FALSE);
            if(obsNonzero > 1){
                intermittent <- "f";
                n.param.intermittent <- 1;
                intermittentParametersSetter(intermittent,ParentEnvironment=environment());
            }
            else{
                intermittent <- "n";
                intermittentParametersSetter(intermittent,ParentEnvironment=environment());
            }
        }
    }

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

    if(Etype=="M" & any(matvt[,1]<0)){
        matvt[matvt[,1]<0,1] <- 0.001;
        warning(paste0("Negative values produced in state vector of model ",model,".\n",
                       "Please, use a different model."),call.=FALSE);
    }

    if(!is.null(xreg)){
        # Write down the matat and copy values for the holdout
        matat[1:nrow(fitting$matat),] <- fitting$matat;
    }

    errors.mat <- ts(errorerwrap(matvt, matF, matw, y,
                                 h, Etype, Ttype, Stype, modellags,
                                 matxt, matat, matFX, ot),
                     start=start(data),frequency=frequency(data));
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=datafreq);

    assign("matvt",matvt,ParentEnvironment);
    assign("y.fit",y.fit,ParentEnvironment);
    assign("matat",matat,ParentEnvironment);
    assign("errors.mat",errors.mat,ParentEnvironment);
    assign("errors",errors,ParentEnvironment);
}

##### *State-space intervals* #####
ssIntervals <- function(errors, ev=median(errors), level=0.95, intervalsType=c("a","p","sp","np"), df=NULL,
                        measurement=NULL, transition=NULL, persistence=NULL, s2=NULL,
                        modellags=NULL, states=NULL,
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

#Function allows to estimate the coefficients of the simple quantile regression. Used in intervals construction.
quantfunc <- function(A){
    ee <- ye - (A[1]*xe^A[2]);
    return((1-quant)*sum(abs(ee[which(ee<0)]))+quant*sum(abs(ee[which(ee>=0)])));
}

# If degrees of freedom are provided, use Student's distribution. Otherwise stick with normal.
    if(is.null(df)){
        upperquant <- qnorm((1+level)/2,0,1);
        lowerquant <- qnorm((1-level)/2,0,1);
    }
    else{
        upperquant <- qt((1+level)/2,df=df);
        lowerquant <- qt((1-level)/2,df=df);
    }

##### If they want us to produce several steps ahead #####
    if(is.matrix(errors) | is.data.frame(errors)){
        n.var <- ncol(errors);
        obs <- nrow(errors);
        if(length(ev)!=n.var & length(ev)!=1){
            stop("Provided expected value doesn't correspond to the dimension of errors.", call.=FALSE);
        }
        else if(length(ev)==1){
            ev <- rep(ev,n.var);
        }

        upper <- rep(NA,n.var);
        lower <- rep(NA,n.var);

#### Asymmetric intervals using HM ####
        if(intervalsType=="a"){
            for(i in 1:n.var){
                upper[i] <- ev[i] + upperquant / hsmN^2 * Re(hm(errors[,i],ev[i]))^2;
                lower[i] <- ev[i] + lowerquant / hsmN^2 * Im(hm(errors[,i],ev[i]))^2;
            }
        }

#### Semiparametric intervals using the variance of errors ####
        else if(intervalsType=="sp"){
            errors <- errors - matrix(ev,nrow=obs,ncol=n.var,byrow=T);
            vec.var <- colSums(errors^2,na.rm=T)/df;
            if(Etype=="M"){
                vec.mean <- 1;
                upperquant <- qlnorm((1+level)/2,rep(0,n.var),sqrt(vec.var));
                lowerquant <- qlnorm((1-level)/2,rep(0,n.var),sqrt(vec.var));
                # Return to normal values
                vec.var <- (exp(vec.var) - 1) * exp(vec.var);
                # Standartise quantiles
                upperquant <- (upperquant - vec.mean) / sqrt(vec.var);
                lowerquant <- (lowerquant - vec.mean) / sqrt(vec.var);
                # Take intermittent data into account
                vec.var <- vec.var * ivar + vec.mean^2 * ivar + iprob^2 * vec.var;
                # Write down quantiles with new variances
                upper <- upperquant * sqrt(vec.var);
                lower <- lowerquant * sqrt(vec.var);
            }
            else{
                vec.var <- vec.var * ivar + c(y.for)^2 * ivar + iprob^2 * vec.var;
                upper <- ev + upperquant * sqrt(vec.var);
                lower <- ev + lowerquant * sqrt(vec.var);
            }
        }

#### Nonparametric intervals using Taylor and Bunn, 1999 ####
        else if(intervalsType=="np"){
            ye <- errors;
            xe <- matrix(c(1:n.var),byrow=TRUE,ncol=n.var,nrow=nrow(errors));
            xe <- xe[!is.na(ye)];
            ye <- ye[!is.na(ye)];

            A <- rep(1,2);
            quant <- (1+level)/2;
            A <- nlminb(A,quantfunc)$par;
            upper <- A[1]*c(1:n.var)^A[2];

            A <- rep(1,2);
            quant <- (1-level)/2;
            A <- nlminb(A,quantfunc)$par;
            lower <- A[1]*c(1:n.var)^A[2];
        }

#### Parametric intervals ####
        else if(intervalsType=="p"){
            #s2i <- iprob*(1-iprob);

            n.components <- nrow(transition);
            maxlag <- max(modellags);
            h <- n.var;

            # Vector of final variances
            vec.var <- rep(NA,h);

#### Pure multiplicative models ####
            if(Etype=="M" & all(c(Ttype,Stype)!="A")){
                # Array of variance of states
                mat.var.states <- array(0,c(n.components,n.components,h+maxlag));
                mat.var.states[,,1:maxlag] <- persistence %*% t(persistence) * s2;
                mat.var.states.lagged <- as.matrix(mat.var.states[,,1]);

                # New transition and measurement for the internal use
                transitionnew <- matrix(0,n.components,n.components);
                measurementnew <- matrix(0,1,n.components);

                # selectionmat is needed for the correct selection of lagged variables in the array
                # newelements are needed for the correct fill in of all the previous matrices
                selectionmat <- transitionnew;
                newelements <- rep(FALSE,n.components);

                # Define chunks, which correspond to the lags with h being the final one
                chuncksofhorizon <- c(1,unique(modellags),h);
                chuncksofhorizon <- sort(chuncksofhorizon);
                chuncksofhorizon <- chuncksofhorizon[chuncksofhorizon<=h];
                chuncksofhorizon <- unique(chuncksofhorizon);

                # Length of the vector, excluding the h at the end
                chunkslength <- length(chuncksofhorizon) - 1;

                newelements <- modellags<=(chuncksofhorizon[1]);
                measurementnew[,newelements] <- measurement[,newelements];
                vec.var[1:min(h,maxlag)] <- s2;

                for(j in 1:chunkslength){
                    selectionmat[modellags==chuncksofhorizon[j],] <- chuncksofhorizon[j];
                    selectionmat[,modellags==chuncksofhorizon[j]] <- chuncksofhorizon[j];

                    newelements <- modellags<(chuncksofhorizon[j]+1);
                    transitionnew[newelements,newelements] <- transition[newelements,newelements];
                    measurementnew[,newelements] <- measurement[,newelements];

                    for(i in (chuncksofhorizon[j]+1):chuncksofhorizon[j+1]){
                        selectionmat[modellags>chuncksofhorizon[j],] <- i;
                        selectionmat[,modellags>chuncksofhorizon[j]] <- i;

                        mat.var.states.lagged[newelements,newelements] <- mat.var.states[cbind(rep(c(1:n.components),each=n.components),
                                                                                               rep(c(1:n.components),n.components),
                                                                                               i - c(selectionmat))];

                        mat.var.states[,,i] <- transitionnew %*% mat.var.states.lagged %*% t(transitionnew) + s2g;
                        vec.var[i] <- measurementnew %*% mat.var.states.lagged %*% t(measurementnew) + s2;
                    }
                }
                # Produce quantiles for log-normal dist with the specified variance
                upperquant <- qlnorm(0.975,0,sqrt(vec.var));
                lowerquant <- qlnorm(0.025,0,sqrt(vec.var));
                # These two allow to return mean instead of median...
                #vec.mean <- exp((vec.var)/2);
                #y.for <- exp(log(y.for) + (vec.var)/2);
                # Use median instead of mean and forget about the whole thing
                vec.mean <- 1;
                # Calculate variance for log-normal distribution
                vec.var <- (exp(vec.var) - 1) * exp(vec.var);
                # Standartise quantiles
                upperquant <- (upperquant - vec.mean) / sqrt(vec.var);
                lowerquant <- (lowerquant - vec.mean) / sqrt(vec.var);
                # Take intermittent data into account
                vec.var <- vec.var * ivar + vec.mean^2 * ivar + iprob^2 * vec.var;
                # Write down quantiles with new variances
                upper <- upperquant * sqrt(vec.var);
                lower <- lowerquant * sqrt(vec.var);
            }
#### Multiplicative error and additive trend / seasonality
            # else if(Etype=="M" & all(c(Ttype,Stype)!="M") & all(c(Ttype,Stype)!="N")){
            #     vec.var[1:min(h,maxlag)] <- s2;
            #     for(i in 1:h){
            #
            #     }
            # }
#### Pure Additive models ####
            else{
                # Array of variance of states
                mat.var.states <- array(0,c(n.components,n.components,h+maxlag));
                mat.var.states[,,1:maxlag] <- persistence %*% t(persistence) * s2;
                mat.var.states.lagged <- as.matrix(mat.var.states[,,1]);

                # New transition and measurement for the internal use
                transitionnew <- matrix(0,n.components,n.components);
                measurementnew <- matrix(0,1,n.components);

                # selectionmat is needed for the correct selection of lagged variables in the array
                # newelements are needed for the correct fill in of all the previous matrices
                selectionmat <- transitionnew;
                newelements <- rep(FALSE,n.components);

                # Define chunks, which correspond to the lags with h being the final one
                chuncksofhorizon <- c(1,unique(modellags),h);
                chuncksofhorizon <- sort(chuncksofhorizon);
                chuncksofhorizon <- chuncksofhorizon[chuncksofhorizon<=h];
                chuncksofhorizon <- unique(chuncksofhorizon);

                # Length of the vector, excluding the h at the end
                chunkslength <- length(chuncksofhorizon) - 1;

                newelements <- modellags<=(chuncksofhorizon[1]);
                measurementnew[,newelements] <- measurement[,newelements];
                vec.var[1:min(h,maxlag)] <- s2;

                for(j in 1:chunkslength){
                    selectionmat[modellags==chuncksofhorizon[j],] <- chuncksofhorizon[j];
                    selectionmat[,modellags==chuncksofhorizon[j]] <- chuncksofhorizon[j];

                    newelements <- modellags<(chuncksofhorizon[j]+1);
                    transitionnew[newelements,newelements] <- transition[newelements,newelements];
                    measurementnew[,newelements] <- measurement[,newelements];

                    for(i in (chuncksofhorizon[j]+1):chuncksofhorizon[j+1]){
                        selectionmat[modellags>chuncksofhorizon[j],] <- i;
                        selectionmat[,modellags>chuncksofhorizon[j]] <- i;

                        mat.var.states.lagged[newelements,newelements] <- mat.var.states[cbind(rep(c(1:n.components),each=n.components),
                                                                                               rep(c(1:n.components),n.components),
                                                                                               i - c(selectionmat))];

                        mat.var.states[,,i] <- transitionnew %*% mat.var.states.lagged %*% t(transitionnew) + persistence %*% t(persistence) * s2;
                        vec.var[i] <- measurementnew %*% mat.var.states.lagged %*% t(measurementnew) + s2;
                    }
                }
                # Take intermittent data into account
                vec.var <- vec.var * ivar + c(y.for)^2 * ivar + iprob^2 * vec.var;
                upper <- upperquant * sqrt(vec.var);
                lower <- lowerquant * sqrt(vec.var);
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
                upperquant <- qlnorm((1+level)/2,0,sqrt(s2));
                lowerquant <- qlnorm((1-level)/2,0,sqrt(s2));
                # Return to normal values
                s2 <- (exp(s2) - 1) * exp(s2);
                # Standartise quantiles
                upperquant <- (upperquant - 1) / sqrt(s2);
                lowerquant <- (lowerquant - 1) / sqrt(s2);
                # Take intermittent data into account
                s2 <- s2 * ivar + ivar + iprob^2 * s2;
                # Write down quantiles with new variances
                upper <- upperquant * sqrt(s2);
                lower <- lowerquant * sqrt(s2);
            }
            else{
                s2 <- s2 * ivar + c(y.for)^2 * ivar + iprob^2 * s2;
                upper <- ev + upperquant * sqrt(s2);
                lower <- ev + lowerquant * sqrt(s2);
            }
        }
        else if(intervalsType=="np"){
            upper <- quantile(errors,(1+level)/2);
            lower <- quantile(errors,(1-level)/2);
        }
    }
    else{
        stop("The provided data is not either vector or matrix. Can't do anything with it!", call.=FALSE);
    }

    return(list(upper=upper,lower=lower));
}

##### *Forecaster of state-space functions* #####
ssForecaster <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    y.for <- ts(forecasterwrap(matrix(matvt[(obsInsample+1):(obsInsample+maxlag),],nrow=maxlag),
                                      matF, matw, h, Ttype, Stype, modellags,
                                      matrix(matxt[(obsAll-h+1):(obsAll),],ncol=n.exovars),
                                      matrix(matat[(obsAll-h+1):(obsAll),],ncol=n.exovars), matFX),
                start=time(data)[obsInsample]+deltat(data),frequency=datafreq);

    if(Etype=="M" & any(y.for<0)){
        warning(paste0("Negative values produced in forecast. This does not make any sense for model with multiplicative error.\n",
                       "Please, use another model."),call.=FALSE);
    }

# If error additive, estimate as normal. Otherwise - lognormal
    if(Etype=="A"){
        s2 <- as.vector(sum((errors*ot)^2)/(obsNonzero - n.param));
        s2g <- 1;
    }
    else{
        s2 <- as.vector(sum(log(1 + errors*ot)^2)/(obsNonzero - n.param));
        s2g <- log(1 + vecg %*% as.vector(errors*ot)) %*% t(log(1 + vecg %*% as.vector(errors*ot)))/(obsNonzero - n.param);
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
        if(all(c(Etype,Stype,Ttype)!="M") |
           all(c(Etype,Stype,Ttype)!="A") |
           (all(Etype=="M",any(Ttype==c("A","N")),any(Stype==c("A","N"))) & s2<0.1)){
            simulateint <- FALSE;
        }
        else{
            simulateint <- TRUE;
        }

        if(intervalsType=="p" & simulateint==TRUE){
            n.samples <- 10000;
            matg <- matrix(vecg,n.components,n.samples);
            arrvt <- array(NA,c(h+maxlag,n.components,n.samples));
            arrvt[1:maxlag,,] <- rep(matvt[obsInsample+(1:maxlag),],n.samples);
            materrors <- matrix(rnorm(h*n.samples,0,sqrt(s2)),h,n.samples);

            if(Etype=="M"){
                materrors <- exp(materrors) - 1;
            }
            if(all(intermittent!=c("n","p"))){
                matot <- matrix(rbinom(h*n.samples,1,iprob),h,n.samples);
            }
            else{
                matot <- matrix(1,h,n.samples);
            }

            y.simulated <- simulatorwrap(arrvt,materrors,matot,array(matF,c(dim(matF),n.samples)),matw,matg,
                                           Etype,Ttype,Stype,modellags)$matyt;
            if(!is.null(xreg)){
                y.exo.for <- c(y.for) - forecasterwrap(matrix(matvt[(obsInsample+1):(obsInsample+maxlag),],nrow=maxlag),
                                                       matF, matw, h, Ttype, Stype, modellags,
                                                       matrix(rep(1,h),ncol=1), matrix(rep(0,h),ncol=1), matrix(1,1,1));
            }
            else{
                y.exo.for <- rep(0,h);
            }

            y.for <- c(pt.for)*y.for;
            y.low <- ts(apply(y.simulated,1,quantile,(1-level)/2,na.rm=T) + y.exo.for,start=start(y.for),frequency=frequency(data));
            y.high <- ts(apply(y.simulated,1,quantile,(1+level)/2,na.rm=T) + y.exo.for,start=start(y.for),frequency=frequency(data));
        }
        else{
            vt <- matrix(matvt[cbind(obsInsample-modellags,c(1:n.components))],n.components,1);

            quantvalues <- ssIntervals(errors.x, ev=ev, level=level, intervalsType=intervalsType, df=(obsNonzero - n.param),
                                       measurement=matw, transition=matF, persistence=vecg, s2=s2,
                                       modellags=modellags, states=matvt[(obsInsample-maxlag+1):obsInsample,],
                                       y.for=y.for, Etype=Etype, Ttype=Ttype, Stype=Stype, s2g=s2g,
                                       iprob=iprob, ivar=ivar);

            y.for <- c(pt.for)*y.for;
            if(Etype=="A"){
                y.low <- ts(c(y.for) + quantvalues$lower,start=start(y.for),frequency=frequency(data));
                y.high <- ts(c(y.for) + quantvalues$upper,start=start(y.for),frequency=frequency(data));
            }
            else if(Etype=="M" & all(c(Ttype,Stype)!="A")){
                y.low <- ts(c(y.for)*(1 + quantvalues$lower),start=start(y.for),frequency=frequency(data));
                y.high <- ts(c(y.for)*(1 + quantvalues$upper),start=start(y.for),frequency=frequency(data));
            }
            else{
                y.low <- ts(c(y.for) * (1 + quantvalues$lower),start=start(y.for),frequency=frequency(data));
                y.high <- ts(c(y.for) * (1 + quantvalues$upper),start=start(y.for),frequency=frequency(data));
            }
        }
    }
    else{
        y.low <- NA;
        y.high <- NA;
        y.for <- c(pt.for)*y.for;
    }

    assign("s2",s2,ParentEnvironment);
    assign("y.for",y.for,ParentEnvironment);
    assign("y.low",y.low,ParentEnvironment);
    assign("y.high",y.high,ParentEnvironment);
}

##### *Check and initialisation of xreg* #####
ssXreg <- function(data, xreg=NULL, updateX=FALSE,
                   persistenceX=NULL, transitionX=NULL, initialX=NULL,
                   obsInsample, obsAll, obsStates, maxlag=1, h=1, silent=FALSE){
# The function does general checks needed for exogenouse variables and returns the list of necessary parameters

    if(!is.null(xreg)){
        if(any(is.na(xreg))){
            warning("The exogenous variables contain NAs! This may lead to problems during estimation and forecast.",
                    call.=FALSE);
        }
##### The case with vectors and ts objects, but not matrices
        if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
# Check if xreg contains something meaningful
            if(all(xreg[1:obsInsample]==xreg[1])){
                warning("The exogenous variable has no variability. Cannot do anything with that, so dropping out xreg.",
                        call.=FALSE);
                xreg <- NULL;
            }
            else{
                if(length(xreg)!=obsInsample & length(xreg)!=obsAll){
                    stop("Length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",
                         call.=FALSE);
                }
                if(length(xreg)==obsInsample){
                    warning("No exogenous variable provided for the holdout sample. es() was used in order to forecast it.",call.=FALSE);
                    xregForecast <- es(xreg,h=h,intermittent="auto",ic="AICc",silent=TRUE)$forecast;
                    xreg <- c(as.vector(xreg),as.vector(xregForecast));
#                    xreg <- c(as.vector(xreg),rep(xreg[obsInsample],h));
                }
# Number of exogenous variables
                n.exovars <- 1;
# Define matrix w for exogenous variables
                matxt <- matrix(xreg,ncol=1);
# Define the second matat to fill in the coefs of the exogenous vars
                matat <- matrix(NA,obsStates,1);
# Fill in the initial values for exogenous coefs using OLS
                matat[1:maxlag,] <- cov(data[1:obsInsample],xreg[1:obsInsample])/var(xreg[1:obsInsample]);
                if(is.null(names(xreg))){
                    colnames(matat) <- "x";
                }
                else{
                    colnames(matat) <- names(xreg);
                }
            }
        }
##### The case with matrices and data frames
        else if(is.matrix(xreg) | is.data.frame(xreg)){
            checkvariability <- apply(xreg[1:obsInsample,]==rep(xreg[1,],each=obsInsample),2,all);
            if(any(checkvariability)){
                if(all(checkvariability)){
                    warning("None of exogenous variables has variability. Cannot do anything with that, so dropping out xreg.",
                            call.=FALSE);
                    xreg <- NULL;
                }
                else{
                    warning("Some exogenous variables do not have any variability. Dropping them out.",
                            call.=FALSE);
                    xreg <- as.matrix(xreg[,!checkvariability]);
                }
            }

            if(!is.null(xreg)){
                # Check for multicollinearity and drop something if there is a perfect one
                corMatrix <- cor(xreg);
                corCheck <- upper.tri(corMatrix) & corMatrix==1;
                if(any(corCheck)){
                    removexreg <- unique(which(corCheck,arr.ind=TRUE)[,1]);
                    xreg <- matrix(xreg[,-removexreg],ncol=ncol(xreg)-length(removexreg));
                    warning("Some exogenous variables were perfectly correlated. We've dropped them out.",
                            call.=FALSE);
                }

                if(nrow(xreg)!=obsInsample & nrow(xreg)!=obsAll){
                    stop("Length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",
                         call.=FALSE);
                }
                n.exovars <- ncol(xreg);
                if(nrow(xreg)==obsInsample){
                    warning("No exogenous are provided for the holdout sample. es() was used in order to forecast them.",
                            call.=FALSE);
                    xregForecast <- matrix(NA,nrow=h,ncol=n.exovars);
                    if(!silent){
                        message("Producing forecasts for xreg variable...");
                    }
                    for(j in 1:n.exovars){
                        if(!silent){
                            cat(paste0(rep("\b",nchar(round((j-1)/n.exovars,2)*100)+1),collapse=""));
                            cat(paste0(round(j/n.exovars,2)*100,"%"));
                        }
                        xregForecast[,j] <- es(xreg[,j],h=h,intermittent="auto",ic="AICc",silent=TRUE)$forecast;
                        #xreg <- rbind(xreg,xreg[obsInsample,]);
                    }
                    xreg <- rbind(xreg,xregForecast);
                    if(!silent){
                        cat("\b\b\b\bDone!\n");
                    }
                }
# mat.x is needed for the initial values of coefs estimation using OLS
                mat.x <- as.matrix(cbind(rep(1,obsAll),xreg));
# Define the second matat to fill in the coefs of the exogenous vars
                matat <- matrix(NA,obsStates,n.exovars);
# Define matrix w for exogenous variables
                matxt <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
                matat[1:maxlag,] <- rep(t(solve(t(mat.x[1:obsInsample,]) %*% mat.x[1:obsInsample,],tol=1e-50) %*%
                                                  t(mat.x[1:obsInsample,]) %*% data[1:obsInsample])[2:(n.exovars+1)],
                                          each=maxlag);
                if(is.null(colnames(xreg))){
                    colnames(matat) <- paste0("x",c(1:n.exovars));
                }
                else{
                    colnames(matat) <- colnames(xreg);
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
                if(length(initialX) != n.exovars){
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
        n.exovars <- 1;
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
    if(xregEstimate==TRUE & updateX==TRUE){
# First - transition matrix
        if(!is.null(transitionX)){
            if(!is.numeric(transitionX) & !is.vector(transitionX) & !is.matrix(transitionX)){
                stop("Transition matrix for exogenous is not a numeric vector or a matrix!", call.=FALSE);
            }
            else{
                if(length(transitionX) != n.exovars^2){
                    stop(paste0("Size of transition matrix for exogenous is wrong!\n",
                                "It should correspond to the number of exogenous variables."), call.=FALSE);
                }
                else{
                    matFX <- matrix(transitionX,n.exovars,n.exovars);
                    FXEstimate <- FALSE;
                }
            }
        }
        else{
            matFX <- diag(n.exovars);
            FXEstimate <- TRUE;
        }
# Now - persistence vector
        if(!is.null(persistenceX)){
            if(!is.numeric(persistenceX) & !is.vector(persistenceX) & !is.matrix(persistenceX)){
                stop("Persistence vector for exogenous is not numeric!", call.=FALSE);
            }
            else{
                if(length(persistenceX) != n.exovars){
                    stop(paste0("Size of persistence vector for exogenous is wrong!\n",
                                "It should correspond to the number of exogenous variables."), call.=FALSE);
                }
                else{
                    vecgX <- matrix(persistenceX,n.exovars,1);
                    gXEstimate <- FALSE;
                }
            }
        }
        else{
            vecgX <- matrix(0,n.exovars,1);
            gXEstimate <- TRUE;
        }
    }
    else if(xregEstimate==TRUE & updateX==FALSE){
        matFX <- diag(n.exovars);
        FXEstimate <- FALSE;

        vecgX <- matrix(0,n.exovars,1);
        gXEstimate <- FALSE;
    }

    if(all(!FXEstimate,!gXEstimate,!initialXEstimate)){
        xregEstimate <- FALSE;
    }

    return(list(n.exovars=n.exovars, matxt=matxt, matat=matat, matFX=matFX, vecgX=vecgX,
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
ICFunction <- function(n.param=n.param,C,Etype=Etype){
# Information criteria are calculated with the constant part "log(2*pi*exp(1)*h+log(obs))*obs".
# And it is based on the mean of the sum squared residuals either than sum.
# Hyndman likelihood is: llikelihood <- obs*log(obs*cfObjective)

    llikelihood <- likelihoodFunction(C);

    AIC.coef <- 2*n.param*h^multisteps - 2*llikelihood;
# max here is needed in order to take into account cases with higher number of parameters than observations
    AICc.coef <- AIC.coef + 2 * n.param*h^multisteps * (n.param + 1) / max(obsNonzero - n.param - 1,0);
    BIC.coef <- log(obsNonzero)*n.param*h^multisteps - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

    return(list(llikelihood=llikelihood,ICs=ICs));
}

##### *Ouptut printer* #####
ssOutput <- function(timeelapsed, modelname, persistence=NULL, transition=NULL, measurement=NULL,
                     phi=NULL, ARterms=NULL, MAterms=NULL, constant=NULL, A=NULL, B=NULL, initialType="o",
                     nParam=NULL, s2=NULL, hadxreg=FALSE, wentwild=FALSE,
                     cfType="MSE", cfObjective=NULL, intervals=FALSE,
                     intervalsType=c("n","p","sp","np","a"), level=0.95, ICs,
                     holdout=FALSE, insideintervals=NULL, errormeasures=NULL,
                     intermittent="n", iprob=1){
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
    if(all(intermittent!=c("n","p","none","provided"))){
        if(any(intermittent==c("f","fixed"))){
            intermittent <- "Fixed probability";
        }
        else if(any(intermittent==c("c","croston"))){
            intermittent <- "Croston";
        }
        else if(any(intermittent==c("t","tsb"))){
            intermittent <- "TSB";
        }
        cat(paste0("Intermittent model type: ",intermittent));
        if(iprob!=1){
            cat(paste0(", ",round(iprob,3),"\n"));
        }
        else{
            cat("\n");
        }
    }
    else if(any(intermittent==c("p","provided"))){
        cat(paste0("Intermittent data provided for holdout.\n"));
    }

### Stuff for ETS
    if(any(model==c("ETS","GES"))){
        if(!is.null(persistence)){
            cat(paste0("Persistence vector g:\n"));
            print(t(round(persistence,3)));
        }
        if(!is.null(phi)){
            if(phi!=1){
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
        if(all(!is.null(ARterms),any(ARterms!=0))){
            cat("Matrix of AR terms:\n");
            print(round(ARterms,3));
        }
        if(all(!is.null(MAterms),any(MAterms!=0))){
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
        if(nParam==1){
            cat(paste0(nParam," parameter was estimated in the process\n"));
        }
        else{
            cat(paste0(nParam," parameters were estimated in the process\n"));
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
        cat(paste0("; Cost function value: ",round(cfObjective,0),"\n"));
    }
    else{
        cat("\n");
    }

    cat("\nInformation criteria:\n");
    print(ICs);
    cat("\n");

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
        cat(paste0(level*100,"% ",intervalsType," prediction intervals were constructed\n"));
    }

    if(holdout){
        if(intervals & !is.null(insideintervals)){
            cat(paste0(round(insideintervals,0), "% of values are in the prediction interval\n"));
        }
        cat("Forecast errors:\n");
        cat(paste(paste0("MPE: ",errormeasures["MPE"]*100,"%"),
                  paste0("Bias: ",errormeasures["cbias"]*100,"%"),
                  paste0("MAPE: ",errormeasures["MAPE"]*100,"%"),
                  paste0("SMAPE: ",errormeasures["SMAPE"]*100,"%\n"),sep="; "));
        cat(paste(paste0("MASE: ",errormeasures["MASE"]),
                  paste0("sMAE: ",errormeasures["sMAE"]*100,"%"),
                  paste0("RelMAE: ",errormeasures["RelMAE"]),
                  paste0("sMSE: ",errormeasures["sMSE"]*100,"%\n"),sep="; "));
    }
}
