ssarima <- function(data, ar.orders=c(0), i.orders=c(1), ma.orders=c(1), lags=c(1),
                    constant=FALSE, initial=NULL, persistence=NULL, transition=NULL,
                    persistence2=NULL, transition2=NULL,
                    CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
                    FI=FALSE, intervals=FALSE, int.w=0.95,
                    int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                    bounds=TRUE, holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
                    xreg=NULL, go.wild=FALSE, ...){
##### Function constructs SARIMA model (possible triple seasonality) using state-space approach
# ar.orders contains vector of seasonal ars. ar.orders=c(2,1,3) will mean AR(2)+SAR(1)+SAR(3) - model with double seasonality.
#
#    Copyright (C) 2016  Ivan Svetunkov

##### Testing period. Switch off several things
    xreg <- NULL;

# Start measuring the time of calculations
    start.time <- Sys.time();

    if(any(ar.orders<0) | any(i.orders<0) | any(ma.orders<0)){
        stop("Wrong order of the model!",call.=FALSE);
    }
    if(length(lags)!=length(ar.orders) & length(lags)!=length(i.orders) & length(lags)!=length(ma.orders)){
        stop("Seasonal lags do not correspond to any element of SARIMA",call.=FALSE);
    }

# Define maxorder and make all the values look similar (for the polynomials)
    maxorder <- max(length(ar.orders),length(i.orders),length(ma.orders))
    if(length(ar.orders)!=maxorder){
        ar.orders <- c(ar.orders,rep(0,maxorder-length(ar.orders)))
    }
    if(length(i.orders)!=maxorder){
        i.orders <- c(i.orders,rep(0,maxorder-length(i.orders)))
    }
    if(length(ma.orders)!=maxorder){
        ma.orders <- c(ma.orders,rep(0,maxorder-length(ma.orders)))
    }

# Number of components to use
    n.components <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
    modellags <- matrix(rep(1,times=n.components),ncol=1);
    maxlag <- 1;

    CF.type <- CF.type[1];
# Check if the appropriate CF.type is defined
    if(any(CF.type==c("trace","TV","GV","MSEh"))){
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
    if(!is.null(initial)){
        if(!is.numeric(initial) | !is.vector(initial)){
            stop("The initial vector is not numeric!",call.=FALSE);
        }
        if(length(initial) != n.components){
            stop(paste0("Wrong length of initial vector. Should be ",n.components," instead of ",length(initial),"."),call.=FALSE);
        }
    }

# Check the provided vector of initials: length and provided values.
    if(!is.null(persistence)){
        if((!is.numeric(persistence) | !is.vector(persistence)) & !is.matrix(persistence)){
            stop("The persistence vector is not numeric!",call.=FALSE);
        }
        if(length(persistence) != n.components){
            stop(paste0("Wrong length of persistence vector. Should be ",n.components," instead of ",length(persistence),"."),call.=FALSE);
        }
    }

# Check the provided vector of initials: length and provided values.
    if(!is.null(transition)){
        if((!is.numeric(transition) | !is.vector(transition)) & !is.matrix(transition)){
            stop("The transition matrix is not numeric!",call.=FALSE);
        }
        if(length(transition) != n.components^2){
            stop(paste0("Wrong length of transition matrix. Should be ",n.components^2," instead of ",length(transition),"."),call.=FALSE);
        }
    }

# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

# Check if the data is vector
    if(!is.numeric(data) & !is.ts(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }

# Define the actual values and write down the original ts frequency
    y <- matrix(data[1:obs],obs,1);
    datafreq <- frequency(data);

# Prepare lists for the polynomials
    P <- list(NA);
    D <- list(NA);
    Q <- list(NA);

# Transition matrix, measurement vector and persistence vector + state vector
    matF <- rbind(cbind(rep(0,n.components-1),diag(n.components-1)),rep(0,n.components));
    matw <- matrix(c(1,rep(0,n.components-1)),1,n.components);
    vecg <- matrix(0.1,n.components,1);
    matxt <- matrix(NA,nrow=(obs+1),ncol=n.components);

# Now let's prepare the provided exogenous data for the inclusion in ETS
# Check the exogenous variable if it is present and
# fill in the values of xreg if it is absent in the holdout sample.
    if(!is.null(xreg)){
        if(any(is.na(xreg))){
            message("The exogenous variables contain NAs! This may lead to problems during estimation and forecast.");
        }
##### The case with vectors and ts objects, but not matrices #####
        if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
# If xreg is vector or simple ts
        if(length(xreg)!=obs & length(xreg)!=obs.all){
            stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=FALSE);
        }
        if(length(xreg)==obs){
            message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
            xreg <- c(as.vector(xreg),rep(xreg[obs],h));
        }
# Number of exogenous variables
        n.exovars <- 1;
# Define matrix w for exogenous variables
        matwex <- matrix(xreg,ncol=1);
# Define the second matxtreg to fill in the coefs of the exogenous vars
        matxtreg <- matrix(NA,max(obs+maxlag,obs.all),1);
        colnames(matxtreg) <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
        matxtreg[1:maxlag,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
# Redefine the number of components of ETS.
        }
##### The case with matrices and data frames #####
        else if(is.matrix(xreg) | is.data.frame(xreg)){
    # If xreg is matrix or data frame
            if(nrow(xreg)!=obs & nrow(xreg)!=obs.all){
                stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=FALSE)
            }
            if(nrow(xreg)==obs){
                message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
                for(j in 1:h){
                xreg <- rbind(xreg,xreg[obs,]);
                }
            }
# matx is needed for the initial values of coefs estimation using OLS
            n.exovars <- ncol(xreg);
            matx <- as.matrix(cbind(rep(1,obs.all),xreg));
# Define the second matxtreg to fill in the coefs of the exogenous vars
            matxtreg <- matrix(NA,max(obs+maxlag,obs.all),n.exovars);
            colnames(matxtreg) <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
            matwex <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
            matxtreg[1:maxlag,] <- rep(t(solve(t(matx[1:obs,]) %*% matx[1:obs,],tol=1e-50) %*% t(matx[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],each=maxlag);
# Redefine the number of components of ETS.
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=FALSE);
        }
        matv <- matwex;
    }
    else{
        n.exovars <- 1;
        matwex <- matrix(1,max(obs+1,obs.all),1);
        matxtreg <- matrix(0,max(obs+1,obs.all),1);
        matv <- matrix(1,max(obs+1,obs.all),1);
        matF2 <- matrix(1,1,1);
        vecg2 <- matrix(0,1,1);
    }

    n.param <- n.components + sum(ar.orders) + sum(ma.orders);
    if(!is.null(xreg)){
        n.param <- n.param + n.exovars;
        if(go.wild==TRUE){
            n.param <- n.exovars^2 + n.exovars;
        }
    }

    if(n.param >= obs-1){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                    n.param," while the number of observations is ",obs,"!"),call.=FALSE)
    }

polyroots <- function(C){
    n.coef <- 0;
    matF[,1] <- 0;
    for(i in 1:length(lags)){
        if((ar.orders*lags)[i]!=0){
            armat <- matrix(0,lags[i],ar.orders[i]);
            armat[lags[i],] <- -C[(n.coef+1):(n.coef + ar.orders[i])];
            P[[i]] <- c(1,c(armat));

            n.coef <- n.coef + ar.orders[i];
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
            armat[lags[i],] <- C[(n.coef+1):(n.coef + ma.orders[i])];
            Q[[i]] <- c(1,c(armat));

            n.coef <- n.coef + ma.orders[i];
        }
        else{
            Q[[i]] <- 1;
        }
    }

    polysos.i <- as.polynomial(1);
    for(i in 1:length(lags)){
        polysos.i <- polysos.i * polynomial(D[[i]])^i.orders[i];
    }

#starttime <- Sys.time();
    polysos.ar <- 1;
    polysos.ma <- 1;
    for(i in 1:length(P)){
        polysos.ar <- polysos.ar * polynomial(P[[i]]);
    }
    polysos.ari <- polysos.ar * polysos.i;

    for(i in 1:length(Q)){
        polysos.ma <- polysos.ma * polynomial(Q[[i]]);
    }
#    polysos.ar <- prod(as.polylist(lapply(P,polynomial))) * polysos.i;
#    polysos.ma <- prod(as.polylist(lapply(Q,polynomial)));
#print(Sys.time() - starttime);

    if(length((polysos.ari))!=1){
        matF[1:(length(polysos.ari)-1),1] <- -(polysos.ari)[2:length(polysos.ari)];
    }
### The MA parameters are in the style "1 + b1 * B".
    vecg[,] <- (-polysos.ari + polysos.ma)[2:(n.components+1)];
    vecg[is.na(vecg),] <- 0;

    if(is.null(initial)){
        xt <- C[(n.coef+1):(n.coef+n.components)];
    }
    else{
        xt <- initial;
    }

    if(constant==TRUE){
        xtreg <- C[length(C)];
    }
    else{
        xtreg <- 0;
    }

    return(list(matF=matF,vecg=vecg,xt=xt,xtreg=xtreg,polysos.ar=polysos.ar,polysos.ma=polysos.ma));
}

# Function creates bounds for the estimates
hin.constrains <- function(C){
### The other way to do it is check abs(polyroot(polysos.ma)) > 1
    elements <- polyroots(C);
    matF <- elements$matF;
    vecg <- elements$vecg;

    if(any(is.nan(matF - vecg %*% matw))){
        D <- -0.1;
    }
    else{
        D <- 1 - abs(eigen(matF - vecg %*% matw)$values);
    }
    return(D);
}

# Cost function for GES
CF <- function(C){
    elements <- polyroots(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matxt[1,] <- elements$xt;
    polysos.ar <- elements$polysos.ar;
    polysos.ma <- elements$polysos.ma;
    matxtreg[1,] <- elements$xtreg;
#    matF2 <- elements$matF2;
#    vecg2 <- elements$vecg2;

    if(bounds==TRUE){
        if(any(abs(polyroot(polysos.ar))<1)){
            return(max(abs(polyroot(polysos.ar)))*1E+100);
        }
        if(any(abs(polyroot(polysos.ma))<1)){
            return(max(abs(polyroot(polysos.ma)))*1E+100);
        }
    }

    CF.res <- ssoptimizerwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE),
                              y, vecg, h, modellags, multisteps, CF.type, normalizer,
                              matwex, matxtreg, matv, matF2, vecg2);

    return(CF.res);
}

Likelihood.value <- function(C){
    if(CF.type=="GV"){
        return(-obs/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(-obs/2 *((h^multisteps)*log(2*pi*exp(1)) + log(CF(C))));
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

# If there is something to optimise, let's do it.
    if(is.null(initial) | is.null(transition) | is.null(persistence) | !is.null(xreg)){

# ar terms, ma terms from season to season...
        C <- c(rep(0.1,sum(ar.orders)),
               rep(0.1,sum(ma.orders)));

# initial values of state vector and the constant term
        slope <- cov(y[1:min(12,obs),],c(1:min(12,obs)))/var(c(1:min(12,obs)));
        intercept <- sum(y[1:min(12,obs),])/min(12,obs) - slope * (sum(c(1:min(12,obs)))/min(12,obs) - 1);
        initial.stuff <- c(intercept,slope,diff(y[1:(n.components-1),]));
        C <- c(C,initial.stuff[1:n.components]);
        if(constant==TRUE){
            C <- c(C,sum(y)/obs);
        }

# xtreg
# initials, transition matrix and persistence vector
        if(!is.null(xreg)){
            C <- c(C,matxtreg[maxlag,]);
            if(go.wild==TRUE){
                C <- c(C,c(diag(n.exovars)));
                C <- c(C,rep(0,n.exovars));
            }
        }

        elements <- polyroots(C);
        matF <- elements$matF;
        vecg <- elements$vecg;
        matxt[1,] <- elements$xt;
        matxtreg[1,] <- elements$xtreg;
#        matF2 <- elements$matF2;
#        vecg2 <- elements$vecg2;

# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=5000));
#                              lb=c(rep(-2,2*n.components+n.components^2),rep(-max(abs(y[1:obs]),intercept),orders %*% lags)),
#                              ub=c(rep(2,2*n.components+n.components^2),rep(max(abs(y[1:obs]),intercept),orders %*% lags)));
        C <- res$solution;

# Optimise model. Second run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
        C <- res$solution;
        CF.objective <- res$objective;
    }
    else{
# matF, vecg, xt
##############################################
# Needs to be done properly... An additional part of code for that...
        C <- c(transition[,1]);
        C <- c(c(transition),
               c(persistence),
               c(initial));

        CF.objective <- CF(C);
    }

    if(any(hin.constrains(C)<0) & silent==FALSE){
        message("Unstable model is estimated! Use 'bounds=TRUE' to address this issue!");
    }

# Change the CF.type in orders to calculate likelihood correctly.
    if(multisteps==TRUE){
        CF.type <- "GV";
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
    matxt[1,] <- elements$xt;
    matxtreg[1,] <- elements$xtreg;
#    matF2 <- elements$matF2;
#    vecg2 <- elements$vecg2;
    if(is.null(initial)){
        initial <- elements$xt;
    }

    fitting <- ssfitterwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE), y,
                            vecg, modellags, matwex, matxtreg, matv, matF2, vecg2);
    matxt <- ts(fitting$matxt,start=(time(data)[1] - deltat(data)),frequency=frequency(data));
    y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));

    if(constant==TRUE){
        const <- C[length(C)];
    }
    else{
        const <- 0;
    }

#    if(!is.null(xreg)){
# Write down the matxtreg and produce values for the holdout
    matxtreg[1:nrow(fitting$xtreg),] <- fitting$xtreg;
    matxtreg[(obs.all-h):(obs.all),] <- ssxtregfitterwrap(matrix(matxtreg[(obs.all-h):(obs.all),],h+1,n.exovars),matF2);
#    }

# Produce matrix of errors
    errors.mat <- ts(sserrorerwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE), y, h,
                                   modellags, matwex, matxtreg),
                     start=start(data), frequency=frequency(data));
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));

# Produce forecast
    y.for <- ts(ssforecasterwrap(matrix(matxt[(obs+1):nrow(matxt),],nrow=1),
                                 matF,matrix(matw,obs.all,n.components,byrow=TRUE),h,
                                 modellags,matrix(matwex[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                                 matrix(matxtreg[(obs.all-h+1):(obs.all),],ncol=n.exovars)),
                start=time(data)[obs]+deltat(data), frequency=frequency(data));

#    s2 <- mean(errors^2);
    s2 <- as.vector(sum(errors^2)/(obs-n.param));

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

        quantvalues <- pintervals(errors.x, ev=ev, int.w=int.w, int.type=int.type, df=(obs - n.param),
                                 measurement=matw, transition=matF, persistence=vecg, s2=s2, modellags=modellags);
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
    AICc.coef <- AIC.coef + 2 * n.param*h^multisteps * (n.param + 1) / (obs - n.param - 1);
    BIC.coef <- log(obs)*n.param - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

# Revert to the provided cost function
    CF.type <- CF.type.original

    if(!is.null(xreg)){
        matxt <- cbind(matxt,matxtreg[1:nrow(matxt),]);
        colnames(matxt) <- c(paste0("Component ",c(1:n.components)),colnames(matxtreg));
    }
    else{
        colnames(matxt) <- paste0("Component ",c(1:n.components));
    }

# AR terms
    if(any(ar.orders!=0)){
        ARterms <- matrix(0,length(ar.orders),max(ar.orders),
                          dimnames=list(paste0("Lag ",lags),
                                        paste0("AR(",c(1:max(ar.orders)),")")));
    }
    else{
        ARterms <- 0;
    }
# Differences
    if(any(i.orders!=0)){
        Iterms <- matrix(0,length(i.orders),1,
                          dimnames=list(paste0("Lag ",lags),"I(...)"));
        Iterms[,] <- i.orders;
    }
    else{
        Iterms <- 0;
    }
# MA terms
    if(any(ma.orders!=0)){
        MAterms <- matrix(0,length(ma.orders),max(ma.orders),
                          dimnames=list(paste0("Lag ",lags),
                                        paste0("MA(",c(1:max(ma.orders)),")")));
    }
    else{
        MAterms <- 0;
    }

    n.coef <- 0;
    for(i in 1:length(ar.orders)){
        if(ar.orders[i]!=0){
            ARterms[i,1:ar.orders[i]] <- C[(n.coef+1):(n.coef + ar.orders[i])];
            n.coef <- n.coef + ar.orders[i];
        }
        if(ma.orders[i]!=0){
            MAterms[i,1:ma.orders[i]] <- C[(n.coef+1):(n.coef + ma.orders[i])];
            n.coef <- n.coef + ma.orders[i];
        }
    }

    if(holdout==T){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- c(MAPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           MASE(as.vector(y.holdout),as.vector(y.for),sum(abs(diff(as.vector(data)[1:obs])))/(obs-1)),
                           MASE(as.vector(y.holdout),as.vector(y.for),sum(abs(as.vector(data)[1:obs]))/obs),
                           MPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           SMAPE(as.vector(y.holdout),as.vector(y.for),digits=5));
        names(errormeasures) <- c("MAPE","MASE","MASALE","MPE","SMAPE");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

# Give model the name
    if(length(ar.orders)==1){
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
    }
    else{
        const <- NULL;
    }

if(silent==FALSE){
# Make plot
    if(intervals==TRUE){
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                   int.w=int.w,legend=legend,main=modelname);
    }
    else{
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                   int.w=int.w,legend=legend,main=modelname);
    }
# Calculate the number os observations in the interval
    if(all(holdout==TRUE,intervals==TRUE)){
        insideintervals <- sum(as.vector(data)[(obs+1):obs.all]<y.high &
                               as.vector(data)[(obs+1):obs.all]>y.low)/h*100;
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

return(list(model=modelname,states=matxt,initial=initial,transition=matF,persistence=vecg,
            AR=ARterms,I=Iterms,MA=MAterms,constant=const,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,xreg=xreg,persistence2=vecg2,transition2=matF2,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
