utils::globalVariables(c("h","holdout","orders","lags","transition","measurement","multisteps","ot","obs.ot","CF.type",
                         "CF","Etype","Ttype","Stype","matxt","matFX","vecgX","xreg","matvt","n.exovars","matat","errors",
                         "n.param","intervals","int.type","int.w"));

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
            message(paste0("Sorry, I have no idea what 'silent=",silent,"' means. Switching to 'none'."));
            silent <- "none";
        }
        silent <- substring(silent,1,1);
    }

    if(silent==FALSE | silent=="n"){
        silent.text <- FALSE;
        silent.graph <- FALSE;
        legend <- TRUE;
    }
    else if(silent==TRUE | silent=="a"){
        silent.text <- TRUE;
        silent.graph <- TRUE;
        legend <- FALSE;
    }
    else if(silent=="g"){
        silent.text <- FALSE;
        silent.graph <- TRUE;
        legend <- FALSE;
    }
    else if(silent=="l"){
        silent.text <- FALSE;
        silent.graph <- FALSE;
        legend <- FALSE;
    }
    else if(silent=="o"){
        silent.text <- TRUE;
        silent.graph <- FALSE;
        legend <- TRUE;
    }

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
            else if(any(unlist(strsplit(model,""))=="Z")){
                modelDo <- "select";
            }
            else{
                modelDo <- "estimate";
            }
        }
        else{
            modelDo <- "select";
        }

        ### Check error type
        if(all(Etype!=c("Z","A","M"))){
            message("Wrong error type! Should be 'Z', 'A' or 'M'.");
            message("Changing to 'Z'");
            Etype <- "Z";
        }

        ### Check trend type
        if(all(Ttype!=c("Z","N","A","M"))){
            message("Wrong trend type! Should be 'Z', 'N', 'A' or 'M'.");
            message("Changing to 'Z'");
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
                warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),"). Getting rid of some of them."),call.=FALSE);
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

        AR <- list(value=AR);
        # Check the provided AR matrix / vector
        if(!is.null(AR$value)){
            if((!is.numeric(AR$value) | !is.vector(AR$value)) & !is.matrix(AR$value)){
                message("AR should be either vector or matrix. You have provided something strange...",call.=FALSE);
                message("AR will be estimated.");
                AR$required <- AR$estimate <- TRUE;
                AR$value <- NULL;
            }
            else{
                if(sum(ar.orders)!=length(AR$value[AR$value!=0])){
                    message(paste0("Wrong number of non-zero elements of AR. Should be ",sum(ar.orders),
                                   " instead of ",length(AR$value[AR$value!=0]),"."),call.=FALSE);
                    message("AR will be estimated.");
                    AR$required <- AR$estimate <- TRUE;
                    AR$value <- NULL;
                }
                else{
                    AR$value <- as.vector(AR$value[AR$value!=0]);
                    AR$estimate <- FALSE;
                    AR$required <- TRUE;
                }
            }
        }
        else{
            if(all(ar.orders==0)){
                AR$required <- AR$estimate <- FALSE;
            }
            else{
                AR$required <- AR$estimate <- TRUE;
            }
        }

        MA <- list(value=MA);
        # Check the provided MA matrix / vector
        if(!is.null(MA$value)){
            if((!is.numeric(MA$value) | !is.vector(MA$value)) & !is.matrix(MA$value)){
                message("MA should be either vector or matrix. You have provided something strange...",call.=FALSE);
                message("MA will be estimated.");
                MA$required <- MA$estimate <- TRUE;
                MA$value <- NULL;
            }
            else{
                if(sum(ma.orders)!=length(MA$value[MA$value!=0])){
                    message(paste0("Wrong number of non-zero elements of MA. Should be ",sum(ma.orders),
                                   " instead of ",length(MA$value[MA$value!=0]),"."),call.=FALSE);
                    message("MA will be estimated.");
                    MA$required <- MA$estimate <- TRUE;
                    MA$value <- NULL;
                }
                else{
                    MA$value <- as.vector(MA$value[MA$value!=0]);
                    MA$estimate <- FALSE;
                    MA$required <- TRUE;
                }
            }
        }
        else{
            if(all(ma.orders==0)){
                MA$required <- MA$estimate <- FALSE;
            }
            else{
                MA$required <- MA$estimate <- TRUE;
            }
        }

        constant <- list(value=constant);
        # Check the provided constant
        if(is.numeric(constant$value)){
            constant$estimate <- FALSE;
            constant$required <- TRUE;
        }
        else if(is.logical(constant$value)){
            constant$required <- constant$estimate <- constant$value;
            constant$value <- NULL;
        }

        # Number of components to use
        n.components <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
        modellags <- matrix(rep(1,times=n.components),ncol=1);
        if(constant$required==TRUE){
            modellags <- rbind(modellags,1);
        }
        maxlag <- 1;

        if((n.components==0) & (constant$required==FALSE)){
            warning("You have not defined any model! Forcing constant=TRUE.",call.=FALSE,immediate.=TRUE);
            constant$required <- constant$estimate <- TRUE;
        }
    }
    else if(modelType=="ces"){
        # If the user typed wrong seasonality, use the "Full" instead
        if(all(seasonality!=c("n","s","p","f","none","simple","partial","full"))){
            message(paste0("Wrong seasonality type: '",seasonality, "'. Changing it to 'full'"));
            seasonality <- "f";
        }
        seasonality <- substring(seasonality[1],1,1);
    }

    ##### data #####
    if(!is.numeric(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }
    # Check the data for NAs
    if(any(is.na(data))){
        if(silent.text==FALSE){
            message("Data contains NAs. These observations will be excluded.");
        }
        datanew <- data[!is.na(data)];
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=frequency(data));
        }
        data <- datanew;
    }

    # Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

    # Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

    # If obs is negative, this means that we can't do anything...
    if(obs<=0){
        stop("Not enough observations in sample.",call.=FALSE);
    }
    # Define the actual values
    y <- matrix(data[1:obs],obs,1);
    datafreq <- frequency(data);

    if(modelType=="es"){
        # Check if the data is ts-object
        if(!is.ts(data) & Stype!="N"){
            message("The provided data is not ts object. Only non-seasonal models are available.");
            Stype <- "N";
        }

        ### Check seasonality type
        if(all(Stype!=c("Z","N","A","M"))){
            message("Wrong seasonality type! Should be 'Z', 'N', 'A' or 'M'.");
            if(datafreq==1){
                if(silent.text==FALSE){
                    message("Data is non-seasonal. Setting seasonal component to 'N'");
                }
                Stype <- "N";
            }
            else{
                if(silent.text==FALSE){
                    message("Changing to 'Z'");
                }
                Stype <- "Z";
            }
        }
        if(all(Stype!="N",datafreq==1)){
            if(silent.text==FALSE){
                message("Cannot build the seasonal model on data with frequency 1.");
                message(paste0("Switching to non-seasonal model: ETS(",substring(model,1,nchar(model)-1),"N)"));
            }
            Stype <- "N";
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
            stop(paste0("The length of 'lags' (",length(lags),") differes from the length of 'orders' (",length(orders),")."), call.=FALSE);
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
        if(is.null(B$value)){
            B$estimate <- TRUE;
        }
        else{
            B$estimate <- FALSE;
        }

        # Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
        #   matvt - the matrix with the components, lags is the lags used in pt matrix.
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
            C <- c(C,0.5);
            ces.name <- "Complex Exponential Smoothing with a partial (real) seasonality";
            n.components <- 3;
            A$number <- 2;
            B$number <- 1;
        }
        else if(seasonality=="f"){
            # Full seasonality with both real and imaginary parts
            maxlag <- datafreq;
            modellags <- c(1,1,maxlag,maxlag);
            C <- c(C,C);
            ces.name <- "Complex Exponential Smoothing with a full (complex) seasonality";
            n.components <- 4;
            A$number <- 2;
            B$number <- 2;
        }
    }

    ##### obs.vt #####
    # Define the number of rows that should be in the matvt
    obs.vt <- max(obs.all + maxlag, obs + 2*maxlag);

    ##### Fisher Information #####
    if(!exists("FI")){
        FI <- FALSE;
    }

    ##### bounds #####
    bounds <- substring(bounds[1],1,1);
    if(bounds!="u" & bounds!="a" & bounds!="n"){
        message("The strange bounds are defined. Switching to 'usual'.");
        bounds <- "u";
    }

    if(modelType=="es"){
        ##### Information Criteria #####
        IC <- IC[1];
        if(all(IC!=c("AICc","AIC","BIC"))){
            message(paste0("Strange type of information criteria defined: ",IC,". Switching to 'AICc'."));
            IC <- "AICc";
        }
    }

    ##### Cost function type #####
    CF.type <- CF.type[1];
    if(any(CF.type==c("MLSTFE","MSTFE","TFL","MSEh","aMLSTFE","aMSTFE","aTFL","aMSEh"))){
        multisteps <- TRUE;
    }
    else if(any(CF.type==c("MSE","MAE","HAM"))){
        multisteps <- FALSE;
    }
    else{
        message(paste0("Strange cost function specified: ",CF.type,". Switching to 'MSE'."));
        CF.type <- "MSE";
        multisteps <- FALSE;
    }
    CF.type.original <- CF.type;

    ##### intervals, int.type, int.w #####
    int.type <- substring(int.type[1],1,1);
    # Check the provided type of interval
    if(all(int.type!=c("a","p","f","n"))){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'parametric'."));
        int.type <- "p";
    }

    ##### intermittent #####
    if(is.numeric(intermittent)){
        # If it is data, then it should either correspond to the whole sample (in-sample + holdout) or be equal to forecating horizon.
        if(all(length(c(intermittent))!=c(h,obs.all))){
            message(paste0("The length of the provided future occurrences is ",length(c(intermittent)),
                           " while the length of forecasting horizon is ",h,"."));
            message(paste0("Where should we plug in the future occurences data?"));
            message(paste0("Switching to intermittent='fixed'."));
            intermittent <- "f";
            ot <- (y!=0)*1;
            obs.ot <- sum(ot);
            yot <- matrix(y[y!=0],obs.ot,1);
            pt <- matrix(mean(ot),obs,1);
            pt.for <- matrix(1,h,1);
        }

        if(any(intermittent!=0 & intermittent!=1)){
            warning(paste0("Parameter 'intermittent' should contain only zeroes and ones."),
                    call.=FALSE, immediate.=FALSE);
            if(silent.text==FALSE){
                message(paste0("Converting to appropriate vector."));
            }
            intermittent <- (intermittent!=0)*1;
        }

        ot <- (y!=0)*1;
        obs.ot <- sum(ot);
        yot <- matrix(y[y!=0],obs.ot,1);
        if(length(intermittent)==obs.all){
            pt <- intermittent[1:obs];
            pt.for <- intermittent[(obs+1):(obs+h)];
        }
        else{
            pt <- matrix(ot,obs,1);
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
                    call.=FALSE, immediate.=FALSE);
            intermittent <- "f";
        }
        intermittent <- substring(intermittent[1],1,1);

        environment(intermittentParametersSetter) <- environment();
        intermittentParametersSetter(intermittent,ParentEnvironment=environment());

        if(obs.ot <= n.param.intermittent){
            warning("Not enough observations for estimation of intermittency probability.",
                    call.=FALSE, immediate.=FALSE);
            if(silent.text==FALSE){
                message("Switching to simpler model.");
            }
            if(obs.ot > 1){
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
                warning("Can't apply multiplicative model to non-positive data. Switching error type to 'A'", call.=FALSE,immediate.=TRUE);
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
                message("The persistence is not a numeric vector!");
                message("Changing to the estimation of persistence vector values.");
                persistence <- NULL;
                estimate.persistence <- TRUE;
            }
            else{
                if(modelType=="es"){
                    if(length(persistence)>3){
                        message("The length of persistence vector is wrong! It should not be greater than 3.");
                        message("Changing to the estimation of persistence vector values.");
                        persistence <- NULL;
                        estimate.persistence <- TRUE;
                    }
                    else{
                        estimate.persistence <- FALSE;
                    }
                }
                else if(modelType=="ges"){
                    if(length(persistence) != n.components){
                        message(paste0("Wrong length of persistence vector. Should be ",n.components," instead of ",length(persistence),"."),call.=FALSE);
                        message("Changing to the estimation of persistence vector values.");
                        persistence <- NULL;
                        estimate.persistence <- TRUE;
                    }
                    else{
                        estimate.persistence <- FALSE;
                    }
                }
            }
        }
        else{
            estimate.persistence <- TRUE;
        }
    }

    ##### initials ####
    if(is.character(initial)){
        initial <- substring(initial[1],1,1);
        if(initial!="o" & initial!="b"){
            warning("You asked for a strange initial value. We don't do that here. Switching to optimal.",
                    call.=FALSE,immediate.=FALSE);
            initial <- "o";
        }
        fittertype <- initial;
        initial <- NULL;
        estimate.initial <- TRUE;
    }
    else if(is.null(initial)){
        message("Initial value is not selected. Switching to optimal.");
        fittertype <- "o";
        estimate.initial <- TRUE;
    }
    else if(!is.null(initial)){
        if(!is.numeric(initial) | !is.vector(initial)){
            message("The initial vector is not numeric!",call.=FALSE);
            message("Values of initial vector will be estimated.");
            initial <- NULL;
            estimate.initial <- TRUE;
        }
        else{
            if(modelType=="es"){
                if(length(initial)>2){
                    message("The length of initial vector is wrong! It should not be greater than 2.");
                    message("Values of initial vector will be estimated.");
                    initial <- NULL;
                    estimate.initial <- TRUE;
                }
                else{
                    estimate.initial <- FALSE;
                }
            }
            else if(modelType=="ges"){
                if(length(initial) != orders %*% lags){
                    message(paste0("Wrong length of initial vector. Should be ",orders %*% lags," instead of ",length(initial),"."),call.=FALSE);
                    message("Values of initial vector will be estimated.");
                    estimate.initial <- TRUE;
                }
                else{
                    estimate.initial <- FALSE;
                }
            }
            else if(modelType=="ssarima"){
                if(length(initial) != n.components){
                    message(paste0("Wrong length of initial vector. Should be ",n.components," instead of ",length(initial),"."),call.=FALSE);
                    message("Values of initial vector will be estimated.");
                    estimate.initial <- TRUE;
                }
                else{
                    estimate.initial <- FALSE;
                }
            }
            else if(modelType=="ces"){
                if(length(initial) != n.components){
                    message(paste0("Wrong length of initial vector. Should be ",n.components," instead of ",length(initial),"."),call.=FALSE);
                    message("Values of initial vector will be estimated.");
                    estimate.initial <- TRUE;
                }
                else{
                    estimate.initial <- FALSE;
                }
            }
        }
        fittertype <- "o";
    }

    if(any(modelType==c("es"))){
        # If model selection is chosen, forget about the initial values and persistence
        if(any(Etype=="Z",Ttype=="Z",Stype=="Z")){
            if(any(!is.null(initial),!is.null(initial.season),!is.null(persistence),!is.null(phi))){
                message("Model selection doesn't go well with the predefined values.");
                message("Switching to the estimation of all the parameters.");
                initial <- NULL;
                initial.season <- NULL;
                persistence <- NULL;
                phi <- NULL;
            }
        }

        ##### initial.season for ES #####
        if(!is.null(initial.season)){
            if(!is.numeric(initial.season)){
                message("The initial.season vector is not numeric!");
                message("Values of initial.season vector will be estimated.");
                initial.season <- NULL;
                estimate.initial.season <- TRUE;
            }
            else{
                if(length(initial.season)!=datafreq){
                    message("The length of initial.season vector is wrong! It should correspond to the frequency of the data.");
                    message("Values of initial.season vector will be estimated.");
                    initial.season <- NULL;
                    estimate.initial.season <- TRUE
                }
                else{
                    estimate.initial.season <- FALSE;
                }
            }
        }
        else{
            estimate.initial.season <- TRUE;
        }

        # Check the length of the provided data. Say bad words if:
        # 1. Seasonal model, <=2 seasons of data and no initial seasonals.
        # 2. Seasonal model, <=1 season of data, no initial seasonals and no persistence.
        if((Stype!="N" & (obs <= 2*datafreq) & is.null(initial.season)) |
           (Stype!="N" & (obs <= datafreq) & is.null(initial.season) & is.null(persistence))){
            if(is.null(initial.season)){
                message("Are you out of your mind?! We don't have enough observations for the seasonal model! Switching to non-seasonal.");
                Stype <- "N";
                estimate.initial.season <- FALSE;
            }
        }

        ##### phi for ES #####
        if(!is.null(phi)){
            if(!is.numeric(phi) & (damped==TRUE)){
                message("The provided value of phi is meaningless.");
                message("phi will be estimated.");
                phi <- NULL;
                estimate.phi <- TRUE;
            }
            else{
                estimate.phi <- FALSE
            }
        }
        else{
            estimate.phi <- TRUE;
        }
    }

    if(modelType=="ges"){
        ##### transition for GES #####
        # Check the provided vector of initials: length and provided values.
        if(!is.null(transition)){
            if((!is.numeric(transition) | !is.vector(transition)) & !is.matrix(transition)){
                message("The transition matrix is not numeric!",call.=FALSE);
                message("The matrix will be estimated!",call.=FALSE);
                estimate.transition <- TRUE;
            }
            else if(length(transition) != n.components^2){
                message(paste0("Wrong length of transition matrix. Should be ",n.components^2," instead of ",length(transition),"."),call.=FALSE);
                message("The matrix will be estimated!",call.=FALSE);
                estimate.transition <- TRUE;
            }
            else{
                estimate.transition <- FALSE;
            }
        }
        else{
            estimate.transition <- TRUE;
        }

        ##### measurement for GES #####
        if(!is.null(measurement)){
            if((!is.numeric(measurement) | !is.vector(measurement)) & !is.matrix(measurement)){
                message("The measurement vector is not numeric!",call.=FALSE);
                message("The vector will be estimated!",call.=FALSE);
                estimate.transition <- TRUE;
            }
            else if(length(measurement) != n.components){
                message(paste0("Wrong length of measurement vector. Should be ",n.components," instead of ",length(measurement),"."),call.=FALSE);
                message("The vector will be estimated!",call.=FALSE);
                estimate.transition <- TRUE;
            }
            else{
                estimate.measurement <- FALSE;
            }
        }
        else{
            estimate.measurement <- TRUE;
        }
    }

    ##### Calculate n.param.max for checks #####
    if(modelType=="es"){
        # 1 - 3: persitence vector;
        # 1 - 2: initials;
        # 1 - 1 phi value;
        # datafreq: datafreq initials for seasonal component;
        # 1: estimation of variance;
        n.param.max <- (1 + (Ttype!="N") + (Stype!="N"))*estimate.persistence +
            (1 + (Ttype!="N"))*estimate.initial*(fittertype=="o") +
            estimate.phi*damped + datafreq*(Stype!="N")*estimate.initial.season*(fittertype=="o") + 1;
    }
    else if(modelType=="ges"){
        n.param.max <- n.components*estimate.measurement + n.components*(fittertype=="o")*estimate.initial +
            estimate.transition*n.components^2 + (orders %*% lags)*estimate.persistence + 1;
    }
    else if(modelType=="ssarima"){
        n.param.max <- n.components*estimate.initial*(fittertype=="o") + sum(ar.orders)*AR$required +
            sum(ma.orders)*MA$required + constant$required + 1;
    }
    else if(modelType=="ces"){
        n.param.max <- sum(modellags)*estimate.initial*(fittertype=="o") + A$number + B$number + 1;
    }

    # Stop if number of observations is less than horizon and multisteps is chosen.
    if((multisteps==TRUE) & (obs.ot < h+1) & all(CF.type!=c("aMSEh","aTFL","aMSTFE","aMLSTFE"))){
        message(paste0("Do you seriously think that you can use ",CF.type," with h=",h," on ",obs.ot," non-zero observations?!"));
        stop("Not enough observations for multisteps cost function.",call.=FALSE);
    }
    else if((multisteps==TRUE) & (obs.ot < 2*h) & all(CF.type!=c("aMSEh","aTFL","aMSTFE","aMLSTFE"))){
        message(paste0("Number of observations is really low for a multisteps cost function! We will, try but cannot guarantee anything..."));
    }

    normalizer <- mean(abs(diff(c(y))));

    ##### Return values to previous environment #####
    assign("silent",silent,ParentEnvironment);
    assign("silent.text",silent.text,ParentEnvironment);
    assign("silent.graph",silent.graph,ParentEnvironment);
    assign("legend",legend,ParentEnvironment);
    assign("data",data,ParentEnvironment);
    assign("obs.all",obs.all,ParentEnvironment);
    assign("obs",obs,ParentEnvironment);
    assign("obs.ot",obs.ot,ParentEnvironment);
    assign("obs.vt",obs.vt,ParentEnvironment);
    assign("y",y,ParentEnvironment);
    assign("datafreq",datafreq,ParentEnvironment);
    assign("FI",FI,ParentEnvironment);
    assign("bounds",bounds,ParentEnvironment);
    assign("CF.type",CF.type,ParentEnvironment);
    assign("CF.type.original",CF.type.original,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("int.type",int.type,ParentEnvironment);
    assign("intermittent",intermittent,ParentEnvironment);
    assign("ot",ot,ParentEnvironment);
    assign("yot",yot,ParentEnvironment);
    assign("pt",pt,ParentEnvironment);
    assign("pt.for",pt.for,ParentEnvironment);
    assign("n.param.intermittent",n.param.intermittent,ParentEnvironment);
    assign("iprob",iprob,ParentEnvironment);
    assign("initial",initial,ParentEnvironment);
    assign("fittertype",fittertype,ParentEnvironment);
    assign("normalizer",normalizer,ParentEnvironment);

    assign("estimate.initial",estimate.initial,ParentEnvironment);
    assign("n.param.max",n.param.max,ParentEnvironment);

    if(modelType=="es"){
        assign("model",model,ParentEnvironment);
        assign("models.pool",models.pool,ParentEnvironment);
        assign("Etype",Etype,ParentEnvironment);
        assign("Ttype",Ttype,ParentEnvironment);
        assign("Stype",Stype,ParentEnvironment);
        assign("damped",damped,ParentEnvironment);
        assign("modelDo",modelDo,ParentEnvironment);
        assign("initial.season",initial.season,ParentEnvironment);
        assign("phi",phi,ParentEnvironment);
        assign("allowMultiplicative",allowMultiplicative,ParentEnvironment);
        assign("IC",IC,ParentEnvironment);
    }
    else if(modelType=="ges"){
        assign("estimate.transition",estimate.transition,ParentEnvironment);
        assign("estimate.measurement",estimate.measurement,ParentEnvironment);
        assign("orders",orders,ParentEnvironment);
        assign("lags",lags,ParentEnvironment);
    }
    else if(modelType=="ssarima"){
        assign("ar.orders",ar.orders,ParentEnvironment);
        assign("i.orders",i.orders,ParentEnvironment);
        assign("ma.orders",ma.orders,ParentEnvironment);
        assign("lags",lags,ParentEnvironment);
        assign("AR",AR,ParentEnvironment);
        assign("MA",MA,ParentEnvironment);
        assign("constant",constant,ParentEnvironment);
    }
    else if(modelType=="ces"){
        assign("seasonality",seasonality,ParentEnvironment);
        assign("ces.name",ces.name,ParentEnvironment);
        assign("A",A,ParentEnvironment);
        assign("B",B,ParentEnvironment);
    }

    if(any(modelType==c("es","ges"))){
        assign("persistence",persistence,ParentEnvironment);
        assign("estimate.persistence",estimate.persistence,ParentEnvironment);
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

    if(silent==FALSE | silent=="n"){
        silent.text <- FALSE;
        silent.graph <- FALSE;
        legend <- TRUE;
    }
    else if(silent==TRUE | silent=="a"){
        silent.text <- TRUE;
        silent.graph <- TRUE;
        legend <- FALSE;
    }
    else if(silent=="g"){
        silent.text <- FALSE;
        silent.graph <- TRUE;
        legend <- FALSE;
    }
    else if(silent=="l"){
        silent.text <- FALSE;
        silent.graph <- FALSE;
        legend <- FALSE;
    }
    else if(silent=="o"){
        silent.text <- TRUE;
        silent.graph <- FALSE;
        legend <- TRUE;
    }

    ##### Fisher Information #####
    if(!exists("FI")){
        FI <- FALSE;
    }

    ##### Observations #####
# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

# This is the critical minimum needed in order to at least fit ARIMA(0,0,0) with constant
    if(obs < 4){
        stop("Sorry, but your sample is too small. Come back when you have at least 4 observations...",call.=FALSE);
    }

# Check the provided vector of initials: length and provided values.
    if(is.character(initial)){
        initial <- substring(initial[1],1,1);
        if(initial!="o" & initial!="b"){
            warning("You asked for a strange initial value. We don't do that here. Switching to optimal.",call.=FALSE,immediate.=TRUE);
            initial <- "o";
        }
        fittertype <- initial;
        initial <- NULL;
    }
    else if(is.null(initial)){
        if(silent.text==FALSE){
            message("Initial value is not selected. Switching to optimal.");
        }
        fittertype <- "o";
    }
    else{
        if(silent.text==FALSE){
            message("Predefinde initials don't go well with automatic model selection. Switching to optimal.");
        }
        fittertype <- "o";
    }

    ##### bounds #####
    bounds <- substring(bounds[1],1,1);
    # Check if "bounds" parameter makes any sense
    if(bounds!="n" & bounds!="a"){
        message("The strange bounds are defined. Switching to 'admissible'.");
        bounds <- "a";
    }

    ##### Information Criteria #####
    IC <- IC[1];
    if(all(IC!=c("AICc","AIC","BIC"))){
        message(paste0("Strange type of information criteria defined: ",IC,". Switching to 'AICc'."));
        IC <- "AICc";
    }

    ##### Cost function type #####
    CF.type <- CF.type[1];
    if(any(CF.type==c("MLSTFE","MSTFE","TFL","MSEh","aMLSTFE","aMSTFE","aTFL","aMSEh"))){
        multisteps <- TRUE;
    }
    else if(any(CF.type==c("MSE","MAE","HAM"))){
        multisteps <- FALSE;
    }
    else{
        message(paste0("Strange cost function specified: ",CF.type,". Switching to 'MSE'."));
        CF.type <- "MSE";
        multisteps <- FALSE;
    }

    ##### intervals, int.type, int.w #####
    int.type <- substring(int.type[1],1,1);
    # Check the provided type of interval
    if(all(int.type!=c("a","p","f","n"))){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'parametric'."));
        int.type <- "p";
    }

    ##### intermittent #####
    if(is.numeric(intermittent)){
        # If it is data, then it should either correspond to the whole sample (in-sample + holdout) or be equal to forecating horizon.
        if(all(length(c(intermittent))!=c(h,obs.all))){
            message(paste0("The length of the provided future occurrences is ",length(c(intermittent)),
                           " while the length of forecasting horizon is ",h,"."));
            message(paste0("Where should we plug in the future occurences data?"));
            message(paste0("Switching to intermittent='fixed'."));
            intermittent <- "f";
        }

        if(any(intermittent!=0 & intermittent!=1)){
            warning(paste0("Parameter 'intermittent' should contain only zeroes and ones."),
                    call.=FALSE, immediate.=FALSE);
            if(silent.text==FALSE){
                message(paste0("Converting to appropriate vector."));
            }
            intermittent <- (intermittent!=0)*1;
        }
    }
    else{
        y <- data[1:obs];
        obs.ot <- sum((y!=0)*1);
        intermittent <- intermittent[1];
        if(all(intermittent!=c("n","f","c","t","a","none","fixed","croston","tsb","auto"))){
            warning(paste0("Strange type of intermittency defined: '",intermittent,"'. Switching to 'fixed'."),
                    call.=FALSE, immediate.=FALSE);
            intermittent <- "f";
        }
        intermittent <- substring(intermittent[1],1,1);
        environment(intermittentParametersSetter) <- environment();
        intermittentParametersSetter(intermittent,ParentEnvironment=environment());

        if(obs.ot <= n.param.intermittent){
            warning("Not enough observations for estimation of intermittency probability.",
                    call.=FALSE, immediate.=FALSE);
            if(silent.text==FALSE){
                message("Switching to simpler model.");
            }
            if(obs.ot > 1){
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

    assign("silent",silent,ParentEnvironment);
    assign("silent.text",silent.text,ParentEnvironment);
    assign("silent.graph",silent.graph,ParentEnvironment);
    assign("legend",legend,ParentEnvironment);
    assign("bounds",bounds,ParentEnvironment);
    assign("FI",FI,ParentEnvironment);
    assign("obs.all",obs.all,ParentEnvironment);
    assign("obs",obs,ParentEnvironment);
    assign("fittertype",fittertype,ParentEnvironment);
    assign("initial",initial,ParentEnvironment);
    assign("IC",IC,ParentEnvironment);
    assign("CF.type",CF.type,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("int.type",int.type,ParentEnvironment);
    assign("intermittent",intermittent,ParentEnvironment);
}

##### *ssFitter function* #####
ssFitter <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    fitting <- fitterwrap(matvt, matF, matw, y, vecg,
                          modellags, Etype, Ttype, Stype, fittertype,
                          matxt, matat, matFX, vecgX, ot);
    matvt <- ts(fitting$matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=datafreq);
    y.fit <- ts(fitting$yfit,start=start(data),frequency=datafreq);

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
ssIntervals <- function(errors, ev=median(errors), int.w=0.95, int.type=c("a","p","s","n"), df=NULL,
                      measurement=NULL, transition=NULL, persistence=NULL, s2=NULL, modellags=NULL,
                      y.for=rep(0,ncol(errors)), iprob=1){
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

    int.type <- int.type[1]
    hsmN <- gamma(0.75)*pi^(-0.5)*2^(-0.75);

    if(all(int.type!=c("a","p","s","n"))){
        stop(paste0("What do you mean by 'int.type=",int.type,"'? I can't work with this!"),call.=FALSE);
    }

    if(int.type=="p"){
        if(any(is.null(measurement),is.null(transition),is.null(persistence),is.null(s2),is.null(modellags))){
            stop("measurement, transition, persistence, s2 and modellags need to be provided in order to construct parametric intervals!",call.=FALSE);
        }

        if(any(!is.matrix(measurement),!is.matrix(transition),!is.matrix(persistence))){
            stop("measurement, transition and persistence must me matrices. Can't do stuff with what you've provided.",call.=FALSE);
        }
    }

#Function allows to estimate the coefficients of the simple quantile regression. Used in intervals construction.
quantfunc <- function(A){
#    ee <- ye - (A[1] + A[2]*xe + A[3]*xe^2);
    ee <- ye - (A[1]*xe^A[2]);
    return((1-quant)*sum(abs(ee[which(ee<0)]))+quant*sum(abs(ee[which(ee>=0)])));
}

# If degrees of freedom are provided, use Student's distribution. Otherwise stick with normal.
    if(is.null(df)){
        upperquant <- qnorm((1+int.w)/2,0,1);
        lowerquant <- qnorm((1-int.w)/2,0,1);
    }
    else{
        upperquant <- qt((1+int.w)/2,df=df);
        lowerquant <- qt((1-int.w)/2,df=df);
    }

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

##### Asymmetric intervals using HM
        if(int.type=="a"){
            for(i in 1:n.var){
                upper[i] <- ev[i] + upperquant / hsmN^2 * Re(hm(errors[,i],ev[i]))^2;
                lower[i] <- ev[i] + lowerquant / hsmN^2 * Im(hm(errors[,i],ev[i]))^2;
            }
        }

##### Semiparametric intervals using the variance of errors
        else if(int.type=="s"){
            errors <- errors - matrix(ev,nrow=obs,ncol=n.var,byrow=T);
            upper <- ev + upperquant * sqrt(colMeans(errors^2,na.rm=T));
            lower <- ev + lowerquant * sqrt(colMeans(errors^2,na.rm=T));
        }

##### Nonparametric intervals using Taylor and Bunn, 1999
        else if(int.type=="n"){
            ye <- errors;
            xe <- matrix(c(1:n.var),byrow=TRUE,ncol=n.var,nrow=nrow(errors));
            xe <- xe[!is.na(ye)];
            ye <- ye[!is.na(ye)];

#            A <- rep(1,3);
            A <- rep(1,2);
            quant <- (1+int.w)/2;
            A <- nlminb(A,quantfunc)$par;
#            upper <- A[1] + A[2]*c(1:n.var) + A[3]*c(1:n.var)^2;
            upper <- A[1]*c(1:n.var)^A[2];

#            A <- rep(1,3);
            A <- rep(1,2);
            quant <- (1-int.w)/2;
            A <- nlminb(A,quantfunc)$par;
#            lower <- A[1] + A[2]*c(1:n.var) + A[3]*c(1:n.var)^2;
            lower <- A[1]*c(1:n.var)^A[2];
        }

##### Parametric intervals from GES
        else if(int.type=="p"){
            s2i <- iprob*(1-iprob);
            s2 <- s2 * iprob;

            n.components <- nrow(transition);
            maxlag <- max(modellags);
            h <- n.var;

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

# Vector of final variances
            vec.var <- rep(NA,h);
            newelements <- modellags<=(chuncksofhorizon[1]);
            measurementnew[,newelements] <- measurement[,newelements];
            vec.var[1:min(h,maxlag)] <- s2 + s2i * (y.for[1])^2;

            for(j in 1:chunkslength){
                selectionmat[modellags==chuncksofhorizon[j],] <- chuncksofhorizon[j];
                selectionmat[,modellags==chuncksofhorizon[j]] <- chuncksofhorizon[j];

                newelements <- modellags<=(chuncksofhorizon[j]+1);
                transitionnew[newelements,newelements] <- transition[newelements,newelements];
                measurementnew[,newelements] <- measurement[,newelements];

                for(i in (chuncksofhorizon[j]+1):chuncksofhorizon[j+1]){
                    selectionmat[modellags>chuncksofhorizon[j],] <- i;
                    selectionmat[,modellags>chuncksofhorizon[j]] <- i;

                    mat.var.states.lagged[newelements,newelements] <- mat.var.states[cbind(rep(c(1:n.components),each=n.components),
                                                              rep(c(1:n.components),n.components),
                                                              i - c(selectionmat))];

                    mat.var.states[,,i] <- transitionnew %*% mat.var.states.lagged %*% t(transitionnew) + persistence %*% t(persistence) * s2;
                    vec.var[i] <- measurementnew %*% mat.var.states.lagged %*% t(measurementnew) + s2 +
                                  s2i * (y.for[i])^2;
                }
            }

            upper <- ev + upperquant * sqrt(vec.var);
            lower <- ev + lowerquant * sqrt(vec.var);
        }
    }
    else if(is.numeric(errors) & length(errors)>1 & !is.array(errors)){
        if(length(ev)>1){
            stop("Provided expected value doesn't correspond to the dimension of errors.", call.=FALSE);
        }

        if(int.type=="a"){
            upper <- ev + upperquant / hsmN^2 * Re(hm(errors,ev))^2;
            lower <- ev + lowerquant / hsmN^2 * Im(hm(errors,ev))^2;
        }
        else if(any(int.type==c("s","p"))){
            s2i <- iprob*(1-iprob);
            newelements <- modellags<=1;
            measurement <- measurement[,newelements];
            s2i <- s2i * (y.for[1])^2;
            upper <- ev + upperquant * sqrt(mean((errors-ev)^2,na.rm=T) * iprob + s2i);
            lower <- ev + lowerquant * sqrt(mean((errors-ev)^2,na.rm=T) * iprob + s2i);
        }
        else if(int.type=="n"){
            upper <- quantile(errors,(1+int.w)/2);
            lower <- quantile(errors,(1-int.w)/2);
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

    y.for <- ts(pt.for*forecasterwrap(matrix(matvt[(obs+1):(obs+maxlag),],nrow=maxlag),
                                      matF, matw, h, Ttype, Stype, modellags,
                                      matrix(matxt[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                                      matrix(matat[(obs.all-h+1):(obs.all),],ncol=n.exovars), matFX),
                start=time(data)[obs]+deltat(data),frequency=datafreq);

    if(Etype=="M" & any(y.for<0)){
        y.for[y.for<0] <- 1;
    }

# If error additive, estimate as normal. Otherwise - lognormal
        if(Etype=="A"){
            s2 <- as.vector(sum((errors*ot)^2)/(obs.ot - n.param));
        }
        else{
            s2 <- as.vector(sum((log(1+errors*ot))^2)/(obs.ot - n.param));
        }

# Write down the forecasting intervals
        if(intervals==TRUE){
            if(h==1){
                errors.x <- as.vector(errors);
                ev <- median(errors);
            }
            else{
                errors.x <- errors.mat;
                ev <- apply(errors.mat,2,median,na.rm=TRUE);
            }
            if(int.type!="a"){
                ev <- 0;
            }

            if(all(c(Etype,Stype,Ttype)!="M") | (all(c(Etype,Stype,Ttype)!="A") & s2 < 1)){
                simulateint <- FALSE;
            }
            else{
                simulateint <- TRUE;
            }

            if(int.type=="p" & simulateint==TRUE){
                n.samples <- 10000
                matg <- matrix(vecg,n.components,n.samples);
                arrvt <- array(NA,c(h+maxlag,n.components,n.samples));
                arrvt[1:maxlag,,] <- rep(matvt[(obs-maxlag+1):obs,],n.samples);
                materrors <- matrix(rnorm(n.samples,0,sqrt(s2)),h,n.samples);
                if(Etype=="M"){
                    materrors <- exp(materrors) - 1;
                }
                if(all(intermittent!=c("n","p"))){
                    matot <- matrix(rbinom(n.samples,1,iprob),h,n.samples);
                }
                else{
                    matot <- matrix(1,h,n.samples);
                }

                y.simulated <- simulateETSwrap(arrvt,materrors,matot,matF,matw,matg,Etype,Ttype,Stype,modellags)$matyt;
                if(!is.null(xreg)){
                    y.exo.for <- c(y.for) - forecasterwrap(matrix(matvt[(obs+1):(obs+maxlag),],nrow=maxlag),
                                                  matF, matw, h, Ttype, Stype, modellags,
                                                  matrix(rep(1,h),ncol=1), matrix(rep(0,h),ncol=1), matrix(1,1,1));
                }
                else{
                    y.exo.for <- rep(0,h);
                }
                y.low <- ts(apply(y.simulated,1,quantile,(1-int.w)/2,na.rm=T) + y.exo.for,start=start(y.for),frequency=frequency(data));
                y.high <- ts(apply(y.simulated,1,quantile,(1+int.w)/2,na.rm=T) + y.exo.for,start=start(y.for),frequency=frequency(data));
            }
            else{
                vt <- matrix(matvt[cbind(obs-modellags,c(1:n.components))],n.components,1);

                quantvalues <- ssIntervals(errors.x, ev=ev, int.w=int.w, int.type=int.type, df=(obs.ot - n.param),
                                          measurement=matw, transition=matF, persistence=vecg, s2=s2, modellags=modellags,
                                          y.for=y.for, iprob=iprob);
                if(Etype=="A"){
                    y.low <- ts(c(y.for) + pt.for*quantvalues$lower,start=start(y.for),frequency=frequency(data));
                    y.high <- ts(c(y.for) + pt.for*quantvalues$upper,start=start(y.for),frequency=frequency(data));
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
        }

    assign("s2",s2,ParentEnvironment);
    assign("y.for",y.for,ParentEnvironment);
    assign("y.low",y.low,ParentEnvironment);
    assign("y.high",y.high,ParentEnvironment);
}

##### *Check and initialisation of xreg* #####
ssXreg <- function(data, xreg=NULL, go.wild=FALSE,
                   persistenceX=NULL, transitionX=NULL, initialX=NULL,
                   obs, obs.all, obs.vt, maxlag=1, h=1, silent=FALSE){
# The function does general checks needed for exogenouse variables and returns the list of necessary parameters

    if(!is.null(xreg)){
        if(any(is.na(xreg)) & silent==FALSE){
            message("The exogenous variables contain NAs! This may lead to problems during estimation and forecast.");
        }
##### The case with vectors and ts objects, but not matrices
        if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
# Check if xreg contains something meaningful
            if(all(xreg[1:obs]==xreg[1])){
                warning("The exogenous variable has no variability. Cannot do anything with that, so dropping out xreg.",
                        call.=FALSE, immediate.=TRUE);
                xreg <- NULL;
            }
            else{
                if(length(xreg)!=obs & length(xreg)!=obs.all){
                    stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!", call.=F);
                }
                if(length(xreg)==obs){
                    if(silent==FALSE){
                        message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
                    }
                    xreg <- c(as.vector(xreg),rep(xreg[obs],h));
                }
# Number of exogenous variables
                n.exovars <- 1;
# Define matrix w for exogenous variables
                matxt <- matrix(xreg,ncol=1);
# Define the second matat to fill in the coefs of the exogenous vars
                matat <- matrix(NA,obs.vt,1);
                exocomponent.names <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
                matat[1:maxlag,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
            }
        }
##### The case with matrices and data frames
        else if(is.matrix(xreg) | is.data.frame(xreg)){
            checkvariability <- apply(xreg[1:obs,]==rep(xreg[1,],each=obs),2,all);
            if(any(checkvariability)){
                if(all(checkvariability)){
                    warning("All exogenous variables have no variability. Cannot do anything with that, so dropping out xreg.",
                            call.=FALSE, immediate.=TRUE);
                    xreg <- NULL;
                }
                else{
                    warning("Some exogenous variables do not have any variability. Dropping them out.",
                            call.=FALSE, immediate.=TRUE);
                    xreg <- as.matrix(xreg[,!checkvariability]);
                }
            }

            if(!is.null(xreg)){
                if(nrow(xreg)!=obs & nrow(xreg)!=obs.all){
                    stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=F);
                }
                if(nrow(xreg)==obs){
                    if(silent==FALSE){
	                    message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
                    }
                    for(j in 1:h){
                    xreg <- rbind(xreg,xreg[obs,]);
                    }
                }
# mat.x is needed for the initial values of coefs estimation using OLS
                mat.x <- as.matrix(cbind(rep(1,obs.all),xreg));
                n.exovars <- ncol(xreg);
# Define the second matat to fill in the coefs of the exogenous vars
                matat <- matrix(NA,obs.vt,n.exovars);
                exocomponent.names <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
                matxt <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
                matat[1:maxlag,] <- rep(t(solve(t(mat.x[1:obs,]) %*% mat.x[1:obs,],tol=1e-50) %*%
                                                  t(mat.x[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],
                                          each=maxlag);
                colnames(matat) <- colnames(xreg);
            }
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=F);
        }
        estimate.xreg <- TRUE;
        colnames(matat) <- exocomponent.names;

# Check the provided initialX vector
        if(!is.null(initialX)){
            if(!is.numeric(initialX) & !is.vector(initialX) & !is.matrix(initialX)){
                stop("The initials for exogenous are not a numeric vector or a matrix!", call.=FALSE);
            }
            else{
                if(length(initialX) != n.exovars){
                    stop("The size of initial vector for exogenous is wrong! It should correspond to the number of exogenous variables.", call.=FALSE);
                }
                else{
                    matat[1:maxlag,] <- as.vector(rep(initialX,each=maxlag));
                    estimate.initialX <- FALSE;
                }
            }
        }
        else{
            estimate.initialX <- TRUE;
        }
    }
    else{
        go.wild <- FALSE;
    }

##### In case we changed xreg to null or if it was like that...
    if(is.null(xreg)){
# "1" is needed for the final forecast simplification
        n.exovars <- 1;
        matxt <- matrix(1,obs.vt,1);
        matat <- matrix(0,obs.vt,1);
        matFX <- matrix(1,1,1);
        vecgX <- matrix(0,1,1);
        estimate.xreg <- FALSE;
        estimate.FX <- FALSE;
        estimate.gX <- FALSE;
        estimate.initialX <- FALSE;
    }

# Now check transition and persistence of exogenous variables
    if(estimate.xreg==TRUE & go.wild==TRUE){
# First - transition matrix
        if(!is.null(transitionX)){
            if(!is.numeric(transitionX) & !is.vector(transitionX) & !is.matrix(transitionX)){
                stop("The transition matrix for exogenous is not a numeric vector or matrix!", call.=FALSE);
            }
            else{
                if(length(transitionX) != n.exovars^2){
                    stop("The size of transition matrix for exogenous is wrong! It should correspond to the number of exogenous variables.", call.=FALSE);
                }
                else{
                    matFX <- matrix(transitionX,n.exovars,n.exovars);
                    estimate.FX <- FALSE;
                }
            }
        }
        else{
            matFX <- diag(n.exovars);
            estimate.FX <- TRUE;
        }
# Now - persistence vector
        if(!is.null(persistenceX)){
            if(!is.numeric(persistenceX) & !is.vector(persistenceX) & !is.matrix(persistenceX)){
                stop("The transition matrix for exogenous is not a numeric vector or matrix!", call.=FALSE);
            }
            else{
                if(length(persistenceX) != n.exovars){
                    stop("The size of persistence vector for exogenous is wrong! It should correspond to the number of exogenous variables.", call.=FALSE);
                }
                else{
                    vecgX <- matrix(persistenceX,n.exovars,1);
                    estimate.gX <- FALSE;
                }
            }
        }
        else{
            vecgX <- matrix(0,n.exovars,1);
            estimate.gX <- TRUE;
        }
    }
    else if(estimate.xreg==TRUE & go.wild==FALSE){
        matFX <- diag(n.exovars);
        estimate.FX <- FALSE;

        vecgX <- matrix(0,n.exovars,1);
        estimate.gX <- FALSE;
    }

    return(list(n.exovars=n.exovars, matxt=matxt, matat=matat, matFX=matFX, vecgX=vecgX,
                estimate.xreg=estimate.xreg, estimate.FX=estimate.FX,
                estimate.gX=estimate.gX, estimate.initialX=estimate.initialX))
}

##### *Likelihood function* #####
likelihoodFunction <- function(C){
# This block is needed in order to make R CMD to shut up about "no visible binding..."
    if(any(intermittent==c("n","p"))){
        if(CF.type=="TFL" | CF.type=="aTFL"){
            return(- obs.ot/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
        }
        else{
            return(- obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
        }
    }
    else{
        if(CF.type=="TFL" | CF.type=="aTFL"){
            return(sum(log(pt[ot==1]))*(h^multisteps)
                   + sum(log(1-pt[ot==0]))*(h^multisteps)
                   - obs.ot/2 * ((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
        }
        else{
            return(sum(log(pt[ot==1])) + sum(log(1-pt[ot==0]))
                   - obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
        }
    }
}

##### *Function calculates ICs* #####
ICFunction <- function(n.param=n.param,C,Etype=Etype){
# Information criteria are calculated with the constant part "log(2*pi*exp(1)*h+log(obs))*obs".
# And it is based on the mean of the sum squared residuals either than sum.
# Hyndman likelihood is: llikelihood <- obs*log(obs*CF.objective)

    llikelihood <- likelihoodFunction(C);

    AIC.coef <- 2*n.param*h^multisteps - 2*llikelihood;
# max here is needed in order to take into account cases with higher number of parameters than observations
    AICc.coef <- AIC.coef + 2 * n.param*h^multisteps * (n.param + 1) / max(obs.ot - n.param - 1,0);
    BIC.coef <- log(obs.ot)*n.param*h^multisteps - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

    return(list(llikelihood=llikelihood,ICs=ICs));
}

##### *Ouptut printer* #####
ssOutput <- function(timeelapsed, modelname, persistence=NULL, transition=NULL, measurement=NULL,
                     phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                     n.components=NULL, s2=NULL, hadxreg=FALSE, wentwild=FALSE,
                     CF.type="MSE", CF.objective=NULL, intervals=FALSE,
                     int.type=c("p","s","n","a"), int.w=0.95, ICs,
                     holdout=FALSE, insideintervals=NULL, errormeasures=NULL, intermittent="n"){
# Function forms the generic output for State-space models.
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
        cat(paste0("Intermittent model type: ",intermittent,"\n"));
    }
    else if(any(intermittent==c("p","provided"))){
        cat(paste0("Intermittent data provided for holdout.\n"));
    }

### Stuff for ETS and GES
    if(!is.null(persistence)){
        cat(paste0("Persistence vector g: ", paste(round(persistence,3),collapse=", "),"\n"));
    }
    if(!is.null(transition)){
        cat("Transition matrix F: \n");
        print(round(transition,3));
    }
    if(!is.null(measurement)){
        cat(paste0("Measurement vector w: ",paste(round(measurement,3),collapse=", "),"\n"));
    }
    if(!is.null(phi)){
        cat(paste0("Damping parameter: ", round(phi,3),"\n"));
    }
### Stuff for ARIMA
    if(all(!is.null(ARterms),any(ARterms!=0))){
        cat("Matrix of AR terms:\n");
        print(round(ARterms,3));
    }
    if(all(!is.null(MAterms),any(MAterms!=0))){
        cat("Matrix of MA terms:\n");
        print(round(MAterms,3));
    }
    if(!is.null(const)){
        cat(paste0("Constant value is: ",round(const,3),"\n"));
    }
### Stuff for CES
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

    if(!is.null(n.components)){
        if(n.components==1){
            cat(paste0(n.components," initial state was estimated in the process\n"));
        }
        else{
            cat(paste0(n.components," initial states were estimated in the process\n"));
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

    cat(paste0("Cost function type: ",CF.type))
    if(!is.null(CF.objective)){
        cat(paste0("; Cost function value: ",round(CF.objective,0),"\n"));
    }
    else{
        cat("\n");
    }

    cat("\nInformation criteria:\n");
    print(ICs);
    cat("\n");

    if(intervals==TRUE){
        if(int.type=="p"){
            int.type <- "parametric";
        }
        else if(int.type=="s"){
            int.type <- "semiparametric";
        }
        else if(int.type=="n"){
            int.type <- "nonparametric";
        }
        else if(int.type=="a"){
            int.type <- "asymmetric";
        }
        cat(paste0(int.w*100,"% ",int.type," prediction intervals were constructed\n"));
    }

    if(holdout==TRUE){
        if(intervals==TRUE){
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
#    message("Don't forget to include the info about the initialisation method used!");
}
