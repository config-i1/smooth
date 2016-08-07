utils::globalVariables(c("h","holdout"))

ssinput <- function(...){
# This is universal function needed in order to check the passed arguments to es(), ges(), ces() and ssarima()

    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

# Variable is needed in order to check if model can be estimated on data
    n.param.test <- 0;

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

##### model #####
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

# Check if the data is ts-object
    if(!is.ts(data) & Stype!="N"){
        message("The provided data is not ts object. Only non-seasonal models are available.");
        Stype <- "N";
    }
    datafreq <- frequency(data);

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

##### Fisher Information #####
    if(!is.null(ellipsis[['FI']])){
        FI <- ellipsis[['FI']];
    }
    else{
        FI <- FALSE;
    }

##### bounds #####
    bounds <- substring(bounds[1],1,1);
    if(bounds!="u" & bounds!="a" & bounds!="n"){
        message("The strange bounds are defined. Switching to 'usual'.");
        bounds <- "u";
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
        else if(length(intermittent)==obs.all){
            intermittent <- intermittent[(obs+1):(obs+h)];
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
        pt <- matrix(ot,obs,1);
        pt.for <- matrix(intermittent,h,1);
        iprob <- 1;
# "p" stand for "provided", meaning that we have been provided the future data
        intermittent <- "p";
        n.param.intermittent <- 0;
    }
    else{
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
    if(pt[1,]==1 & all(intermittent!=c("n","p","a"))){
        intermittent <- "n";
    }

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

##### persistence #####
    if(!is.null(persistence)){
        if(!is.numeric(persistence)){
            message("The persistence is not a numeric vector!");
            message("Changing to the estimation of persistence vector values.");
            persistence <- NULL;
        }
        else{
            if(length(persistence)>3){
                message("The length of persistence vector is wrong! It should not be greater than 3.");
                message("Changing to the estimation of persistence vector values.");
                persistence <- NULL;
            }
        }
    }
# If persistence is null or has been changed in the previous check, write down the number of parameters
    if(is.null(persistence)){
        n.param.test <- n.param.test - length(persistence);
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
    }
    else if(is.null(initial)){
        message("Initial value is not selected. Switching to optimal.");
        fittertype <- "o";
    }
    else if(!is.null(initial)){
        if(!is.numeric(initial) | !is.vector(initial)){
            stop("The initial vector is not numeric!",call.=FALSE);
        }
        else{
            if(length(initial)>2){
                message("The length of initial vector is wrong! It should not be greater than 2.");
                message("Values of initial vector will be estimated.");
                initial <- NULL;
            }
            fittertype <- "o";
        }
    }

# If initial is null or has been changed in the previous check, write down the number of parameters
    if(is.null(initial)){
        n.param.test <- n.param.test - length(initial);
    }

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

##### initial.season #####
    if(!is.null(initial.season)){
        if(!is.numeric(initial.season)){
            message("The initial.season vector is not numeric!");
            message("Values of initial.season vector will be estimated.");
            initial.season <- NULL;
        }
        else{
            if(length(initial.season)!=datafreq){
                message("The length of initial.season vector is wrong! It should correspond to the frequency of the data.");
                message("Values of initial.season vector will be estimated.");
                initial.season <- NULL;
            }
        }
    }
# If the initial.season has been changed to estimation, do things...
    if(Stype!="N" & is.null(initial.season)){
        n.param.test <- n.param.test - length(initial.season);
    }

# Check the length of the provided data. Say bad words if:
# 1. Seasonal model, <=2 seasons of data and no initial seasonals.
# 2. Seasonal model, <=1 season of data, no initial seasonals and no persistence.
    if((Stype!="N" & (obs <= 2*datafreq) & is.null(initial.season)) |
       (Stype!="N" & (obs <= datafreq) & is.null(initial.season) & is.null(persistence))){
    	if(is.null(initial.season)){
        	message("Are you out of your mind?! We don't have enough observations for the seasonal model! Switching to non-seasonal.");
       		Stype <- "N";
    	}
    }

##### phi #####
    if(!is.null(phi)){
        if(!is.numeric(phi) & (damped==TRUE)){
            message("The provided value of phi is meaningless.");
            message("phi will be estimated.");
            phi <- NULL;
        }
        else{
            n.param.test <- n.param.test - 1;
        }
    }

##### Calculate n.param.test #####
# 2: 1 initial and 1 smoothing for level component;
# 2: 1 initial and 1 smoothing for trend component;
# 1: 1 phi value.
# 1 + datafreq: datafreq initials and 1 smoothing for seasonal component;
# 1: estimation of variance;
    n.param.test <- n.param.test + 2 + 2*(Ttype!="N") + damped + (1 + datafreq)*(Stype!="N") + 1;

# Stop if number of observations is less than horizon and multisteps is chosen.
    if((multisteps==TRUE) & (obs.ot < h+1) & all(CF.type!=c("aMSEh","aTFL","aMSTFE","aMLSTFE"))){
        message(paste0("Do you seriously think that you can use ",CF.type," with h=",h," on ",obs.ot," non-zero observations?!"));
        stop("Not enough observations for multisteps cost function.",call.=FALSE);
    }
    else if((multisteps==TRUE) & (obs.ot < 2*h) & all(CF.type!=c("aMSEh","aTFL","aMSTFE","aMLSTFE"))){
        message(paste0("Number of observations is really low for a multisteps cost function! We will, try but cannot guarantee anything..."));
    }


    assign("silent",silent,ParentEnvironment);
    assign("silent.text",silent.text,ParentEnvironment);
    assign("silent.graph",silent.graph,ParentEnvironment);
    assign("legend",legend,ParentEnvironment);
    assign("model",model,ParentEnvironment);
    assign("models.pool",models.pool,ParentEnvironment);
    assign("Etype",Etype,ParentEnvironment);
    assign("Ttype",Ttype,ParentEnvironment);
    assign("Stype",Stype,ParentEnvironment);
    assign("damped",damped,ParentEnvironment);
    assign("modelDo",modelDo,ParentEnvironment);
    assign("data",data,ParentEnvironment);
    assign("obs.all",obs.all,ParentEnvironment);
    assign("obs",obs,ParentEnvironment);
    assign("y",y,ParentEnvironment);
    assign("datafreq",datafreq,ParentEnvironment);
    assign("FI",FI,ParentEnvironment);
    assign("bounds",bounds,ParentEnvironment);
    assign("IC",IC,ParentEnvironment);
    assign("CF.type",CF.type,ParentEnvironment);
    assign("CF.type.original",CF.type.original,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("int.type",int.type,ParentEnvironment);
    assign("intermittent",intermittent,ParentEnvironment);
    assign("ot",ot,ParentEnvironment);
    assign("obs.ot",obs.ot,ParentEnvironment);
    assign("yot",yot,ParentEnvironment);
    assign("pt",pt,ParentEnvironment);
    assign("pt.for",pt.for,ParentEnvironment);
    assign("n.param.intermittent",n.param.intermittent,ParentEnvironment);
    assign("iprob",iprob,ParentEnvironment);
    assign("allowMultiplicative",allowMultiplicative,ParentEnvironment);
    assign("persistence",persistence,ParentEnvironment);
    assign("initial",initial,ParentEnvironment);
    assign("fittertype",fittertype,ParentEnvironment);
    assign("initial.season",initial.season,ParentEnvironment);
    assign("phi",phi,ParentEnvironment);
    assign("n.param.test",n.param.test,ParentEnvironment);
}
