ssarima <- function(data, ar.orders=c(0), i.orders=c(1), ma.orders=c(1), lags=c(1),
                    constant=FALSE, initial=c("backcasting","optimal"), AR=NULL, MA=NULL,
                    CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                    h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                    int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                    intermittent=c("none","simple","croston","tsb"),
                    bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                    xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...){
##### Function constructs SARIMA model (possible triple seasonality) using state-space approach
# ar.orders contains vector of seasonal ars. ar.orders=c(2,1,3) will mean AR(2)+SAR(1)+SAR(3) - model with double seasonality.
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

# See if a user asked for Fisher Information
    if(!is.null(list(...)[['FI']])){
        FI <- list(...)[['FI']];
    }
    else{
        FI <- FALSE;
    }

# Make sense out of silent
    silent <- silent[1];
# Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output"))){
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

    bounds <- substring(bounds[1],1,1);
    intermittent <- substring(intermittent[1],1,1);
# Check if "bounds" parameter makes any sense
    if(bounds!="n" & bounds!="a"){
        message("The strange bounds are defined. Switching to 'admissible'.");
        bounds <- "a";
    }

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

# If zeroes are defined for some lags, drop them.
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

# Number of components to use
    n.components <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
    modellags <- matrix(rep(1,times=n.components),ncol=1);
    if(constant==TRUE){
        modellags <- rbind(modellags,1);
    }
    maxlag <- 1;

    if((n.components==0) & (constant==FALSE)){
        warning("You have not defined any model! Forcing constant=TRUE.",call.=FALSE,immediate.=TRUE);
        constant <- TRUE;
    }

    CF.type <- CF.type[1];
# Check if the appropriate CF.type is defined
    if(any(CF.type==c("MLSTFE","MSTFE","TFL","MSEh"))){
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

    int.type <- substring(int.type[1],1,1);
# Check the provided type of interval
    if(all(int.type!=c("a","p","s","n"))){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'parametric'."));
        int.type <- "p";
    }

# Check if the data is vector
    if(!is.numeric(data) & !is.ts(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }

# Check if the data contains NAs
    if(any(is.na(data))){
        message("Data contains NAs. These observations will be excluded.");
        datanew <- data[!is.na(data)];
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=frequency(data));
        }
        data <- datanew;
    }

# Check the provided vector of initials: length and provided values.
    if(is.character(initial)){
        initial <- substring(initial[1],1,1);
        if(initial!="o" & initial!="b"){
            warning("You asked for a strange initial value. We don't do that here. Switching to optimal.",call.=FALSE,immediate.=TRUE);
            initial <- "o";
        }
        fittertype <- initial;
        estimate.initial <- TRUE;
        if(constant==TRUE){
            estimate.constant <- TRUE;
        }
        else{
            estimate.constant <- FALSE;
        }
    }
    else if(is.null(initial)){
        message("Initial value is not selected. Switching to optimal.");
        fittertype <- "o";
        estimate.initial <- TRUE;
        estimate.constant <- TRUE;
    }
    else if(!is.null(initial)){
        if(!is.numeric(initial) | !is.vector(initial)){
            stop("The initial vector is not numeric!",call.=FALSE);
        }
        if(constant==TRUE){
            if(length(initial)==(n.components+constant)){
                estimate.constant <- FALSE;
            }
            else if(length(initial)<(n.components+constant)){
                estimate.constant <- TRUE
            }
            else if(length(initial)>(n.components+constant)){
                stop(paste0("Wrong length of initial vector. Should be ",n.components+constant," instead of ",length(initial),"."),call.=FALSE);
            }
        }
        else{
            estimate.constant <- FALSE;
            if(length(initial)!=(n.components)){
                stop(paste0("Wrong length of initial vector. Should be ",n.components," instead of ",length(initial),"."),call.=FALSE);
            }
        }
        fittertype <- "o";
        estimate.initial <- FALSE;
    }

# Check the provided AR matrix / vector
    if(!is.null(AR)){
        if((!is.numeric(AR) | !is.vector(AR)) & !is.matrix(AR)){
            stop("AR should be either vector or matrix. You have provided something strange...",call.=FALSE);
        }
        if(sum(ar.orders)!=length(AR[AR!=0])){
            stop(paste0("Wrong number of non-zero elements of AR. Should be ",sum(ar.orders)," instead of ",length(AR[AR!=0]),"."),call.=FALSE);
        }
        AR <- as.vector(AR[AR!=0]);
        estimate.AR <- FALSE;
    }
    else{
        if(all(ar.orders==0)){
            estimate.AR <- FALSE;
        }
        else{
            estimate.AR <- TRUE;
        }
    }

# Check the provided MA matrix / vector
    if(!is.null(MA)){
        if((!is.numeric(MA) | !is.vector(MA)) & !is.matrix(MA)){
            stop("MA should be either vector or matrix. You have provided something strange...",call.=FALSE);
        }
        if(sum(ma.orders)!=length(MA[MA!=0])){
            stop(paste0("Wrong number of non-zero elements of MA. Should be ",sum(ma.orders)," instead of ",length(MA[MA!=0]),"."),call.=FALSE);
        }
        MA <- as.vector(MA[MA!=0]);
        estimate.MA <- FALSE;
    }
    else{
        if(all(ma.orders==0)){
            estimate.MA <- FALSE;
        }
        else{
            estimate.MA <- TRUE;
        }
    }

# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

# If obs is negative, this means that we can't do anything...
    if(obs<=0){
        stop("Not enough observations in sample.",call.=FALSE);
    }

# Define the number of rows that should be in the matvt
    obs.xt <- max(obs.all + maxlag, obs + 2*maxlag);

# Check if the data is vector
    if(!is.numeric(data) & !is.ts(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }

# Define the actual values and write down the original ts frequency
    y <- matrix(data[1:obs],obs,1);
    datafreq <- frequency(data);

    if(intermittent!="n"){
        ot <- (y!=0)*1;
        obs.ot <- sum(ot);
        yot <- matrix(y[y!=0],obs.ot,1);
        pt <- matrix(mean(ot),obs,1);
        pt.for <- matrix(1,h,1);
    }
    else{
        ot <- rep(1,obs);
        obs.ot <- obs;
        yot <- y;
        pt <- matrix(1,obs,1);
        pt.for <- matrix(1,h,1);
    }
    iprob <- pt[1,];

# If the data is not intermittent, let's assume that the parameter was switched unintentionally.
    if(iprob==1){
        intermittent <- "n";
    }

# Stop if number of observations is less than horizon and multisteps is chosen.
    if((multisteps==TRUE) & (obs.ot < h+1)){
        message(paste0("Do you seriously think that you can use ",CF.type," with h=",h," on ",obs.ot," non-zero observations?!"));
        stop("Not enough observations for multisteps cost function.",call.=FALSE);
    }
    else if((multisteps==TRUE) & (obs.ot < 2*h)){
        message(paste0("Number of observations is really low for a multisteps cost function! We will try but cannot guarantee anything..."));
    }

# Prepare lists for the polynomials
    P <- list(NA);
    D <- list(NA);
    Q <- list(NA);

    if(n.components > 0){
# Transition matrix, measurement vector and persistence vector + state vector
        matF <- rbind(cbind(rep(0,n.components-1),diag(n.components-1)),rep(0,n.components));
        matw <- matrix(c(1,rep(0,n.components-1)),1,n.components);
        vecg <- matrix(0.1,n.components,1);
        matvt <- matrix(NA,obs.xt,n.components);
        if(constant==TRUE){
            matF <- cbind(rbind(matF,rep(0,n.components)),c(1,rep(0,n.components-1),1));
            matw <- cbind(matw,0);
            vecg <- rbind(vecg,0);
            matvt <- cbind(matvt,rep(1,obs.xt));
        }
    }
    else{
        matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0,1,1);
        matvt <- matrix(1,obs.xt,1);
        modellags <- matrix(1,1,1);
    }

##### Prepare exogenous variables #####
    xregdata <- ssxreg(data=data, xreg=xreg, go.wild=go.wild,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obs=obs, obs.all=obs.all, obs.xt=obs.xt, maxlag=maxlag, h=h, silent=silent.text);
    n.exovars <- xregdata$n.exovars;
    matxt <- xregdata$matxt;
    matat <- xregdata$matat;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;
    estimate.xreg <- xregdata$estimate.xreg;
    estimate.FX <- xregdata$estimate.FX;
    estimate.gX <- xregdata$estimate.gX;
    estimate.initialX <- xregdata$estimate.initialX;

# 1 stands for the variance
    if(fittertype=="o"){
        n.param <- 1 + n.components*estimate.initial + sum(ar.orders)*estimate.AR + sum(ma.orders)*estimate.MA + (intermittent!="n") + constant;
    }
    else{
# Number of components that really need to be estimated (droping zeroes): (p+d+1)(P+D+1)-1 or (q+1)(Q+1)-1
#        n.components.corrected <- max(prod(ar.orders + i.orders + 1) - 1, prod(ma.orders + 1) - 1);
# Initials are not optimised, so they should not be included in number of parameters calculation
        n.components.corrected <- 0;
        n.param <- 1 + n.components.corrected + sum(ar.orders) + sum(ma.orders) + (intermittent!="n") + constant;
    }

    if(estimate.xreg==TRUE){
        n.param <- n.param + estimate.initialX*n.exovars + estimate.FX*(n.exovars^2) + estimate.gX*(n.exovars);
    }

    if(obs.ot <= n.param){
        if(intermittent!="n"){
            stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                        n.param," while the number of non-zero observations is ",obs.ot,"!"),call.=FALSE);
        }
        else{
            stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                        n.param," while the number of observations is ",obs.ot,"!"),call.=FALSE);
        }
    }

# These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

polyroots <- function(C){
    polysos.ar <- 0;
    polysos.ma <- 0;
    n.coef <- 0;
    matF[,1] <- 0;
    if(n.components > 0){
        ar.inner.coef <- ma.inner.coef <- 0;
        for(i in 1:length(lags)){
            if((ar.orders*lags)[i]!=0){
                armat <- matrix(0,lags[i],ar.orders[i]);
                if(estimate.AR==TRUE){
                    armat[lags[i],] <- -C[n.coef+(1:ar.orders[i])];
                    n.coef <- n.coef + ar.orders[i];
                }
                else{
                    armat[lags[i],] <- -AR[ar.inner.coef+(1:ar.orders[i])];
                    ar.inner.coef <- ar.inner.coef + ar.orders[i];
                    n.coef <- n.coef + ar.orders[i];
                }
                P[[i]] <- c(1,c(armat));

            }
            else{
                P[[i]] <- 1;
            }

            if((i.orders*lags)[i]!=0){
                D[[i]] <- c(1,rep(0,max(lags[i]-1,0)),-1);
            }
            else{
                D[[i]] <- 1;
            }

            if((ma.orders*lags)[i]!=0){
                armat <- matrix(0,lags[i],ma.orders[i]);
                if(estimate.MA==TRUE){
                    armat[lags[i],] <- C[n.coef+(1:ma.orders[i])];
                    n.coef <- n.coef + ma.orders[i];
                }
                else{
                    armat[lags[i],] <- MA[ma.inner.coef+(1:ma.orders[i])];
                    ma.inner.coef <- ma.inner.coef + ma.orders[i];
                    n.coef <- n.coef + ma.orders[i];
                }
                Q[[i]] <- c(1,c(armat));
            }
            else{
                Q[[i]] <- 1;
            }
        }

##### Polynom multiplication is the slowest part now #####
        polysos.i <- as.polynomial(1);
        for(i in 1:length(lags)){
            polysos.i <- polysos.i * polynomial(D[[i]])^i.orders[i];
        }

        polysos.ar <- 1;
        polysos.ma <- 1;
        for(i in 1:length(P)){
            polysos.ar <- polysos.ar * polynomial(P[[i]]);
        }
        polysos.ari <- polysos.ar * polysos.i;

        for(i in 1:length(Q)){
            polysos.ma <- polysos.ma * polynomial(Q[[i]]);
        }

        if(length((polysos.ari))!=1){
            matF[1:(length(polysos.ari)-1),1] <- -(polysos.ari)[2:length(polysos.ari)];
        }

### The MA parameters are in the style "1 + b1 * B".
        vecg[1:n.components,] <- (-polysos.ari + polysos.ma)[2:(n.components+1)];
        vecg[is.na(vecg),] <- 0;

        if(estimate.initial==TRUE){
            if(fittertype=="o"){
                vt <- C[(n.coef + 1):(n.coef + n.components + constant)];
                n.coef <- n.coef + n.components + constant;
            }
            else{
                vt <- matvt[1,];
                vt[-1] <- vt[1] * matF[-1,1];
                if(estimate.constant==TRUE){
                    vt[n.components+constant] <- C[(n.coef + 1)];
                    n.coef <- n.coef + 1;
                }
            }
        }
        else{
            vt <- initial;
            if(estimate.constant==TRUE){
                vt[n.components+constant] <- C[(n.coef + 1)];
                n.coef <- n.coef + 1;
            }
        }
    }
    else{
        matF[1,1] <- 1;
        if(estimate.initial==TRUE){
            vt <- C[n.coef+1];
            n.coef <- n.coef + 1;
        }
        else{
            vt <- initial;
        }
    }

# If exogenous are included
    if(estimate.xreg==TRUE){
        at <- matrix(NA,maxlag,n.exovars);
        if(estimate.initialX==TRUE){
            at[,] <- rep(C[n.coef+(1:n.exovars)],each=maxlag);
            n.coef <- n.coef + n.exovars;
        }
        else{
            at <- matat[1:maxlag,];
        }
        if(estimate.FX==TRUE){
            matFX <- matrix(C[n.coef+(1:(n.exovars^2))],n.exovars,n.exovars);
            n.coef <- n.coef + n.exovars^2;
        }

        if(estimate.gX==TRUE){
            vecgX <- matrix(C[n.coef+(1:n.exovars)],n.exovars,1);
            n.coef <- n.coef + n.exovars;
        }
    }
    else{
        at <- matrix(0,maxlag,n.exovars);
    }

    return(list(matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX,polysos.ar=polysos.ar,polysos.ma=polysos.ma));
}

# Cost function for GES
CF <- function(C){
    elements <- polyroots(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    polysos.ar <- elements$polysos.ar;
    polysos.ma <- elements$polysos.ma;

    if(bounds=="a" & (n.components > 0)){
        arroots <- abs(polyroot(polysos.ar));
        if(any(arroots<1)){
            return(max(arroots)*1E+100);
        }
        maroots <- abs(polyroot(polysos.ma));
        if(any(maroots<1)){
            return(max(maroots)*1E+100);
        }
    }

    CF.res <- optimizerwrap(matvt, matF, matw, y, vecg,
                            h, modellags, Etype, Ttype, Stype,
                            multisteps, CF.type, normalizer, fittertype,
                            matxt, matat, matFX, vecgX, ot);
    if(is.nan(CF.res) | is.na(CF.res) | is.infinite(CF.res)){
        CF.res <- 1e+100;
    }

    return(CF.res);
}

Likelihood.value <- function(C){
    if(intermittent=="n"){
        if(CF.type=="TFL"){
            return(obs*(h^multisteps) - obs/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
        }
        else{
            return(obs - obs/2 *(log(2*pi*exp(1)) + log(CF(C))));
        }
    }
    else{
        if(CF.type=="TFL"){
            return(sum(log(pt)*ot)*(h^multisteps) +
                       sum(log(1-pt)*(1-ot))*(h^multisteps) +
                       -obs.ot/2 * ((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
        }
        else{
            return(sum(log(pt)*ot) + sum(log(1-pt)*(1-ot)) +
                       -obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
        }
    }
}

#####Start the calculations#####
    y.fit <- rep(NA,obs);
    errors <- rep(NA,obs);

    if(multisteps==TRUE){
        normalizer <- sum(abs(diff(y)))/(obs-1);
    }
    else{
        normalizer <- 0;
    }

############################################## To be fixed ##############################################
# Needs to be done properly... does not take into account the provided data...
# If there is something to optimise, let's do it.
    if(((estimate.initial==TRUE) & fittertype=="o") | (estimate.AR==TRUE) | (estimate.MA==TRUE) |
       (estimate.xreg==TRUE) | (estimate.FX==TRUE) | (estimate.gX==TRUE) | (estimate.constant==TRUE) ){

        C <- NULL;
        if(n.components > 0){
# ar terms, ma terms from season to season...
            if(estimate.AR==TRUE){
                C <- c(C,rep(0.1,sum(ar.orders)));
            }
            if(estimate.MA==TRUE){
                C <- c(C,rep(0.1,sum(ma.orders)));
            }

# initial values of state vector and the constant term
            if(estimate.initial==TRUE){
                slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
                intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);
                if(fittertype=="o"){
                    initial.stuff <- c(rep(intercept,n.components));
                    C <- c(C,initial.stuff[1:n.components]);
                }
                else{
                    initial.stuff <- c(intercept,-intercept,rep(slope,n.components));
                    matvt[1,1:n.components] <- initial.stuff[1:(n.components)];
                }
            }
        }

        if(constant==TRUE){
            if(all(i.orders==0)){
                C <- c(C,sum(yot)/obs);
            }
            else{
                C <- c(C,sum(diff(yot))/obs);
            }
        }

# initials, transition matrix and persistence vector
        if(estimate.xreg==TRUE){
            if(estimate.initialX==TRUE){
                C <- c(C,matat[maxlag,]);
            }
            if(go.wild==TRUE){
                if(estimate.FX==TRUE){
                    C <- c(C,c(diag(n.exovars)));
                }
                if(estimate.gX==TRUE){
                    C <- c(C,rep(0,n.exovars));
                }
            }
        }

        elements <- polyroots(C);
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[1:maxlag,] <- elements$vt;
        matat[1:maxlag,] <- elements$at;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;
        polysos.ar <- elements$polysos.ar;
        polysos.ma <- elements$polysos.ma;
        arroots <- abs(polyroot(polysos.ar));
        maroots <- abs(polyroot(polysos.ma));

##### Initialisation needs to be done using backcast! #####
# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
#                              lb=c(rep(-2,2*n.components+n.components^2),rep(-max(abs(y[1:obs]),intercept),orders %*% lags)),
#                              ub=c(rep(2,2*n.components+n.components^2),rep(max(abs(y[1:obs]),intercept),orders %*% lags)));
        C <- res$solution;
        if(fittertype=="o"){
# Optimise model. Second run
            res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
            C <- res$solution;
        }
        CF.objective <- res$objective;
#
#        res <- nlminb(C, CF);
#        C <- res$par;
#        CF.objective <- res$objective;
    }
    else{
        C <- NULL;
# initial values of state vector and the constant term
        slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
        intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);
        initial.stuff <- c(intercept,-intercept,rep(slope,n.components));
        matvt[1,1:n.components] <- initial.stuff[1:(n.components)];

        CF.objective <- CF(C);
    }

# Change the CF.type in orders to calculate likelihood correctly.
    if(multisteps==TRUE){
        CF.type <- "TFL";
    }
    else{
        CF.type <- "MSE";
    }

    if(FI==TRUE){
        FI <- hessian(Likelihood.value,C);
    }
    else{
        FI <- NA;
    }

# Prepare for fitting
    elements <- polyroots(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    polysos.ar <- elements$polysos.ar;
    polysos.ma <- elements$polysos.ma;
    arroots <- abs(polyroot(polysos.ar));
    maroots <- abs(polyroot(polysos.ma));

    if(!is.null(initial)){
        initial <- elements$vt;
    }

    fitting <- fitterwrap(matvt, matF, matw, y, vecg,
                          modellags, Etype, Ttype, Stype, fittertype,
                          matxt, matat, matFX, vecgX, ot);

    matvt <- ts(fitting$matvt,start=(time(data)[1] - deltat(data)),frequency=frequency(data));
    y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));

    if(!is.null(xreg)){
# Write down the matat and produce values for the holdout
        matat[1:nrow(fitting$matat),] <- fitting$matat;
    }

# Produce matrix of errors
    errors.mat <- ts(errorerwrap(matvt, matF, matrix(matw[1,],nrow=1), y,
                                 h, Etype, Ttype, Stype, modellags,
                                 matxt, matat, matFX, ot),
                     start=start(data),frequency=datafreq);
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=datafreq);

# Produce forecast
    y.for <- ts(iprob*forecasterwrap(matrix(matvt[(obs+1):(obs+maxlag),],nrow=maxlag),
                               matF, matrix(matw[1,],nrow=1), h, Ttype, Stype, modellags,
                               matrix(matxt[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                               matrix(matat[(obs.all-h+1):(obs.all),],ncol=n.exovars), matFX),
                start=time(data)[obs]+deltat(data),frequency=datafreq);

#    s2 <- mean(errors^2);
    s2 <- as.vector(sum((errors*ot)^2)/(obs.ot-n.param));

    if(any(is.na(y.fit),is.na(y.for))){
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

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

        vt <- matrix(matvt[cbind(obs-modellags,c(1:max(1,n.components+constant)))],max(1,n.components+constant),1);

        quantvalues <- pintervals(errors.x, ev=ev, int.w=int.w, int.type=int.type, df=(obs.ot - n.param),
                                 measurement=matw, transition=matF, persistence=vecg, s2=s2, modellags=modellags,
                                 y.for=y.for, iprob=iprob);
        y.low <- ts(c(y.for) + quantvalues$lower,start=start(y.for),frequency=frequency(data));
        y.high <- ts(c(y.for) + quantvalues$upper,start=start(y.for),frequency=frequency(data));
    }
    else{
        y.low <- NA;
        y.high <- NA;
    }

# Information criteria
    llikelihood <- Likelihood.value(C);

    AIC.coef <- 2*n.param*h^multisteps - 2*llikelihood;
    AICc.coef <- AIC.coef + 2 * n.param*h^multisteps * (n.param + 1) / (obs.ot - n.param - 1);
    BIC.coef <- log(obs.ot)*n.param - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

# Revert to the provided cost function
    CF.type <- CF.type.original

# Write down initials of states vector and exogenous
    if(estimate.initial==TRUE){
        initial <- matvt[1,];
    }
    if(estimate.initialX==TRUE){
        initialX <- matat[1,];
    }

# Fill in the rest of matvt
    matvt <- ts(matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    if(!is.null(xreg)){
        matvt <- cbind(matvt,matat[1:nrow(matvt),]);
        colnames(matvt) <- c(paste0("Component ",c(1:max(1,n.components+constant))),colnames(matat));
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:max(1,n.components+constant)));
    }
    if(constant==TRUE){
        colnames(matvt)[n.components+constant] <- "Constant";
    }

# AR terms
    if(any(ar.orders!=0)){
        ARterms <- matrix(0,max(ar.orders),sum(ar.orders!=0),
                          dimnames=list(paste0("AR(",c(1:max(ar.orders)),")"),
                                        paste0("Lag ",lags[ar.orders!=0])));
    }
    else{
        ARterms <- 0;
    }
# Differences
    if(any(i.orders!=0)){
        Iterms <- matrix(0,1,length(i.orders),
                          dimnames=list("I(...)",paste0("Lag ",lags)));
        Iterms[,] <- i.orders;
    }
    else{
        Iterms <- 0;
    }
# MA terms
    if(any(ma.orders!=0)){
        MAterms <- matrix(0,max(ma.orders),sum(ma.orders!=0),
                          dimnames=list(paste0("MA(",c(1:max(ma.orders)),")"),
                                        paste0("Lag ",lags[ma.orders!=0])));
    }
    else{
        MAterms <- 0;
    }

    n.coef <- ar.coef <- ma.coef <- 0;
    ar.i <- ma.i <- 1;
    for(i in 1:length(ar.orders)){
        if(ar.orders[i]!=0){
            if(estimate.AR==TRUE){
                ARterms[1:ar.orders[i],ar.i] <- C[n.coef+(1:ar.orders[i])];
            }
            else{
                ARterms[1:ar.orders[i],ar.i] <- AR[ar.coef+(1:ar.orders[i])];
                ar.coef <- ar.coef + ar.orders[i];
            }
            ar.i <- ar.i + 1;
            n.coef <- n.coef + ar.orders[i];
        }
        if(ma.orders[i]!=0){
            if(estimate.MA==TRUE){
                MAterms[1:ma.orders[i],ma.i] <- C[n.coef+(1:ma.orders[i])];
            }
            else{
                MAterms[1:ma.orders[i],ma.i] <- MA[ma.coef+(1:ma.orders[i])];
                ma.coef <- ma.coef + ma.orders[i];
            }
            ma.i <- ma.i + 1;
            n.coef <- n.coef + ma.orders[i];
        }
    }

    if(holdout==T){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- c(MAPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(diff(as.vector(data)[1:obs])))),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(as.vector(data)[1:obs]))),
                           MPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           RelMAE(as.vector(y.holdout),as.vector(y.for),rep(y[obs],h),digits=3),
                           SMAPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           cbias(as.vector(y.holdout)-as.vector(y.for),0,digits=5));
        names(errormeasures) <- c("MAPE","MASE","MAE/mean","MPE","RelMAE","SMAPE","cbias");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

# Give model the name
    if((length(ar.orders)==1) && all(lags==1)){
        modelname <- paste0("ARIMA(",ar.orders,",",i.orders,",",ma.orders,")");
    }
    else{
        modelname <- "";
        for(i in 1:length(ar.orders)){
            modelname <- paste0(modelname,"(",ar.orders[i],",");
            modelname <- paste0(modelname,i.orders[i],",");
            modelname <- paste0(modelname,ma.orders[i],")[",lags[i],"]");
        }
        modelname <- paste0("SARIMA",modelname);
    }

    if(constant==TRUE){
        const <- C[length(C)];
        if(all(i.orders==0)){
            modelname <- paste0(modelname," with constant");
        }
        else{
            modelname <- paste0(modelname," with drift");
        }
    }
    else{
        const <- NULL;
    }

    if(silent.text==FALSE){
        if(any(maroots<1)){
            if(bounds!="a"){
                message("Unstable model was estimated! Use bounds='admissible' to address this issue!");
            }
            else{
                message("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.");
            }
        }
        if(any(arroots<1)){
            if(bounds!="a"){
                message("Non-stationary model was estimated! Beware of explosions! Use bounds='admissible' to address this issue!");
            }
            else{
                message("Something went wrong in optimiser - non-stationary model was estimated! Please report this error to the maintainer.");
            }
        }
# Calculate the number os observations in the interval
        if(all(holdout==TRUE,intervals==TRUE)){
            insideintervals <- sum(as.vector(data)[(obs+1):obs.all]<=y.high &
                                as.vector(data)[(obs+1):obs.all]>=y.low)/h*100;
        }
        else{
            insideintervals <- NULL;
        }
# Print output
        ssoutput(Sys.time() - start.time, modelname, persistence=NULL, transition=NULL, measurement=NULL,
            phi=NULL, ARterms=ARterms, MAterms=MAterms, const=const, A=NULL, B=NULL,
            n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
            CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
            int.type=int.type, int.w=int.w, ICs=ICs,
            holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);
    }

# Make plot
    if(silent.graph==FALSE){
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       int.w=int.w,legend=legend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    int.w=int.w,legend=legend,main=modelname);
        }
    }

return(list(model=modelname,states=matvt,initial=initial,transition=matF,persistence=vecg,
            AR=ARterms,I=Iterms,MA=MAterms,constant=const,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,
            xreg=xreg,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
