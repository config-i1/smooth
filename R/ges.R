ges <- function(data, bounds=TRUE, order=c(2), lags=c(1),
                CF.type=c("MSE","MAE","HAM","TLV","GV","TV","hsteps"),
                backcast=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric"),
                xreg=NULL, holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
                ...){
# General Exponential Smoothing function

# Start measuring the time of calculations
    start.time <- Sys.time();

    CF.type <- CF.type[1];
    int.type <- substring(int.type[1],1,1);

    if(length(order) != length(lags)){
        stop(paste0("The length of 'lags' (",length(lags),") differes from the length of 'order' (",length(order),")."), call.=FALSE);
    }

    modellags <- matrix(rep(lags,times=order),ncol=1);
    maxlag <- max(modellags);
    n.components <- sum(order);

    if(CF.type=="TLV" | CF.type=="TV" | CF.type=="GV" | CF.type=="hsteps"){
        trace <- TRUE;
    }
    else if(CF.type=="MSE" | CF.type=="MAE" | CF.type=="HAM"){
        trace <- FALSE;
    }
    else{
        message(paste0("Strange cost function specified: ",CF.type,". Switching to 'MSE'."));
        CF.type <- "MSE";
        trace <- FALSE;
    }

# Check the provided type of interval
    if(int.type!="p" & int.type!="s" & int.type!="n"){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'parametric'."));
        int.type <- "p";
    }

    if(any(is.na(data))){
        message("Data contains NAs. These observations will be excluded.");
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

# Define the actual values
    y <- as.vector(data);

# Check if the data is vector
    if(!is.numeric(data) & !is.ts(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE);
    }

    datafreq <- frequency(data);

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
#        n.components <- n.components + 1;
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
            matx <- as.matrix(cbind(rep(1,obs.all),xreg));
            n.exovars <- ncol(xreg);
# Define the second matxtreg to fill in the coefs of the exogenous vars
            matxtreg <- matrix(NA,max(obs+maxlag,obs.all),n.exovars);
            colnames(matxtreg) <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
            matwex <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
            matxtreg[1:maxlag,] <- rep(t(solve(t(matx[1:obs,]) %*% matx[1:obs,],tol=1e-50) %*% t(matx[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],each=maxlag);
# Redefine the number of components of ETS.
#            n.components <- n.components + n.exovars;
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=FALSE);
        }
# Redefine the number of all the parameters. Used in AIC mainly!
#        n.param <- n.param + n.exovars;
    }
    else{
        n.exovars <- 1;
        matwex <- matrix(0,max(obs+maxlag,obs.all),1);
        matxtreg <- matrix(0,max(obs+maxlag,obs.all),1);
    }

    n.param <- 2*n.components+n.components^2 + order %*% lags;
    if(!is.null(xreg)){
        n.param <- n.param + n.exovars;
    }

    if(n.param >= obs-1){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                    n.param," while the number of observations is ",obs,"!"),call.=FALSE)
    }

elements.ges <- function(C){

    matw <- matrix(C[1:n.components],1,n.components);
    matF <- matrix(C[n.components+(1:(n.components^2))],n.components,n.components);
    vecg <- matrix(C[n.components+n.components^2+(1:n.components)],n.components,1);

    xtvalues <- C[2*n.components+n.components^2+(1:(order %*% lags))];

    xt <- matrix(NA,maxlag,n.components);
    for(i in 1:n.components){
        xt[(maxlag - modellags + 1)[i]:maxlag,i] <- xtvalues[((cumsum(c(0,modellags))[i]+1):cumsum(c(0,modellags))[i+1])];
        xt[is.na(xt[1:maxlag,i]),i] <- rep(rev(xt[(maxlag - modellags + 1)[i]:maxlag,i]),
                                           ceiling((maxlag - modellags + 1) / modellags)[i])[is.na(xt[1:maxlag,i])];
    }

# If exogenous are included
    if(!is.null(xreg)){
        matxtreg[1:maxlag,] <- rep(C[(length(C)-n.exovars+1):length(C)],each=maxlag);
    }

    return(list(matw=matw,matF=matF,vecg=vecg,xt=xt,matxtreg=matxtreg));
}

# Function makes interval forecasts
forec.inter.ges <- function(matw,matF,vecg,h,s2,int.w,y.for){
# Array of variance of states
    mat.var.states <- array(0,c(n.components,n.components,h+maxlag));
# Vector of final variances
    vec.var <- rep(NA,h);
    vec.var[1] <- s2;

    if(h>1){
        for(i in (2+maxlag):(h+maxlag)){
            mat.var.states[,,i] <- matF %*% mat.var.states[,,i-1] %*% t(matF) + vecg %*% t(vecg) * s2;
            vec.var[i-maxlag] <- matw %*% mat.var.states[,,i] %*% t(matw) + s2;
        }
    }

    y.low <- y.for + qt((1-int.w)/2, df=max(obs - n.param,1)) * sqrt(vec.var);
    y.high <- y.for + qt(1-(1-int.w)/2, df=max(obs - n.param,1)) * sqrt(vec.var);

    return(list(y.low=y.low,y.high=y.high));
}

# Function creates bounds for the estimates
hin.constrains <- function(C){

    elements <- elements.ges(C);
    matw <- elements$matw;
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
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matxtreg[1:maxlag,] <- elements$matxtreg[1:maxlag,];
    matxt[1:maxlag,] <- elements$xt;

    if(bounds==TRUE){
        if(any(is.nan(matF - vecg %*% matw))){
            return(1E+300);
        }
        else{
            eigenvalues <- abs(eigen(matF - vecg %*% matw)$values);
            if(any(eigenvalues>1)){
                return(max(eigenvalues)*1E+100);
            }
        }
    }

    CF.res <- ssoptimizerwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE), matrix(1,obs.all,n.components),
                              as.matrix(y[1:obs]), matrix(vecg,n.components,1), h, modellags, CF.type,
                              normalizer, backcast, matwex, matxtreg);

    return(CF.res);
}

Likelihood.value <- function(C){
    if(CF.type=="GV"){
        return(-obs/2 *((h^trace)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(-obs/2 *((h^trace)*log(2*pi*exp(1)) + log(CF(C))));
    }
}

#####Start the calculations#####
# Initial values of matxt
    slope <- cov(y[1:min(12,obs)],c(1:min(12,obs)))/var(c(1:min(12,obs)));
    intercept <- mean(y[1:min(12,obs)]) - slope * (mean(c(1:min(12,obs))) - 1);

# matw, matF, vecg, xt
    C <- c(rep(1,n.components),
#           as.vector(diag(n.components)),
           rep(1,n.components^2),
           rep(0,n.components),
           intercept);

    if((order %*% lags)>1){
        C <- c(C,slope);
    }
    if((order %*% lags)>2){
        C <- c(C,y[1:(order %*% lags-2)]);
#        C <- c(C,rep(intercept,(order %*% lags)-2));
    }
# xtreg
    if(!is.null(xreg)){
        C <- c(C,matxtreg[maxlag,]);
    }

    matxt <- matrix(NA,nrow=(obs+maxlag),ncol=n.components);

    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matxtreg[1:maxlag,] <- elements$matxtreg[1:maxlag,];
    matxt[1:maxlag,] <- elements$xt;

    y.fit <- rep(NA,obs);
    errors <- rep(NA,obs);

    if(trace==TRUE){
        normalizer <- mean(abs(diff(y[1:obs])));
    }
    else{
        normalizer <- 0;
    }

    res <- nloptr::nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-10, "maxeval"=5000),
                          lb=c(rep(-2,2*n.components+n.components^2),rep(-max(abs(y[1:obs]),intercept),order %*% lags)),
                          ub=c(rep(2,2*n.components+n.components^2),rep(max(abs(y[1:obs]),intercept),order %*% lags)));
    C <- res$solution;

    res <- nloptr::nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-8, "maxeval"=5000));
    C <- res$solution;
    CF.objective <- res$objective;

    if(any(hin.constrains(C)<0) & silent==FALSE){
        message("Unstable model is estimated! Use 'bounds=TRUE' to address this issue!");
    }

# Change the CF.type in order to calculate likelihood correctly.
    CF.type.original <- CF.type;
    if(trace==TRUE){
        CF.type <- "GV";
    }
    else{
        CF.type <- "MSE";
    }

    FI <- numDeriv::hessian(Likelihood.value,C);

    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matxt[1:maxlag,] <- elements$xt;
    matxtreg[1:maxlag,] <- elements$matxtreg[1:maxlag,];

    if(backcast==TRUE){
        fitting <- ssfitterbackcastwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE), matrix(1,obs.all,n.components),
                                        as.matrix(y[1:obs]), matrix(vecg,n.components,1), modellags, matwex, matxtreg);
    }
    else{
        fitting <- ssfitterwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE), matrix(1,obs.all,n.components),
                                as.matrix(y[1:obs]), matrix(vecg,n.components,1), modellags, matwex, matxtreg);
    }
    matxt <- ts(fitting$matxt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));

    if(!is.null(xreg)){
# Write down the matxtreg and copy values for the holdout
        matxtreg[1:nrow(fitting$matxtreg),] <- fitting$matxtreg;
        matxtreg[(obs.all-h+1):obs.all,] <- rep(matxtreg[1,],each=h);
    }

    errors.mat <- ts(sserrorerwrap(matxt, matF, matrix(matw,obs.all,n.components,byrow=TRUE), as.matrix(y[1:obs]), h,
                                  modellags, matwex, matxtreg), start=start(data), frequency=frequency(data));
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));

    y.for <- ts(ssforecasterwrap(matrix(matxt[(obs+1):nrow(matxt),],nrow=maxlag),matF,matrix(matw,obs.all,n.components,byrow=TRUE),h,
                                modellags,matrix(matwex[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                               matrix(matxtreg[(obs.all-h+1):(obs.all),],ncol=n.exovars)),start=time(data)[obs]+deltat(data),
                frequency=frequency(data));
    s2 <- mean(errors^2);

    if(any(is.na(y.fit),is.na(y.for))){
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

    if(intervals==TRUE){
        if(int.type=="p"){
            forec.int <- forec.inter.ges(matw,matF,vecg,h,s2,int.w,y.for);
            y.low <- forec.int$y.low;
            y.high <- forec.int$y.high;
        }
        else if(int.type=="s"){
            y.var <- colMeans(errors.mat^2,na.rm=T);
            y.low <- ts(y.for + qt((1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var),start=start(y.for),frequency=frequency(data));
            y.high <- ts(y.for + qt(1-(1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var),start=start(y.for),frequency=frequency(data));
        }
        else{
            y.var <- apply(errors.mat,2,quantile,probs=c((1-int.w)/2,1-(1-int.w)/2),na.rm=T);
            y.low <- ts(y.for + y.var[1,],start=start(y.for),frequency=frequency(data));
            y.high <- ts(y.for + y.var[2,],start=start(y.for),frequency=frequency(data));
        }
    }
    else{
        y.low <- NA;
        y.high <- NA;
    }

# Information criteria
    llikelihood <- Likelihood.value(C);

    AIC.coef <- 2*n.param*h^trace - 2*llikelihood;
    AICc.coef <- AIC.coef + 2 * n.param*h^trace * (n.param + 1) / (obs - n.param - 1);
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

    if(holdout==T){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- c(MAPE(as.vector(y.holdout),as.vector(y.for),round=5),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(diff(as.vector(data)[1:obs])))),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(as.vector(data)[1:obs]))),
                           MPE(as.vector(y.holdout),as.vector(y.for),round=5),
                           SMAPE(as.vector(y.holdout),as.vector(y.for),round=5));
        names(errormeasures) <- c("MAPE","MASE","MASALE","MPE","SMAPE");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

if(silent==FALSE){
# Print time elapsed on the construction
    print(paste0("Time elapsed: ",round(as.numeric(Sys.time() - start.time,units="secs"),2)," seconds"));
    print(paste0("Persistence vector g: ", paste(round(vecg,3),collapse=", ")));
    print("Transition matrix F: ");
    print(round(matF,3));
    print(paste0("Measurement vector w: ",paste(round(matw,3),collapse=", ")));
    print(paste0("Residuals sigma: ",round(sqrt(mean(errors^2)),3)));
#    print(paste0("Initial components: ", paste(round(matxt[maxlag,1:n.components],3),collapse=", ")));
    if(!is.null(xreg)){
        print(paste0("Xreg coefficients: ", paste(round(matxtreg[maxlag,],3),collapse=", ")));
    }
    if(trace==TRUE){
        print(paste0("CF type: trace with ",CF.type, "; CF value is: ",round(CF.objective,0)));
    }
    else{
        print(paste0("CF type: one step ahead; CF value is: ",round(CF.objective,0)));
    }
    if(intervals==TRUE){
        if(int.type=="p"){
            int.type <- "parametric";
        }
        else if(int.type=="s"){
            int.type <- "semiparametric";
        }
        if(int.type=="n"){
            int.type <- "nonparametric";
        }
        print(paste0(int.w*100,"% ",int.type," intervals were constructed"));
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                   lower=y.low,upper=y.high,int.w=int.w,legend=legend);
    }
    else{
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,legend=legend);
    }
    print(paste0("AIC: ",round(ICs["AIC"],3)," AICc: ", round(ICs["AICc"],3)," BIC: ", round(ICs["BIC"],3)));
    if(holdout==T){
        if(intervals==TRUE){
            print(paste0(round(sum(as.vector(data)[(obs+1):obs.all]<y.high &
                    as.vector(data)[(obs+1):obs.all]>y.low)/h*100,0),
                    "% of values are in the interval"));
        }
        print(paste(paste0("MPE: ",errormeasures["MPE"]*100,"%"),
                    paste0("MAPE: ",errormeasures["MAPE"]*100,"%"),
                    paste0("SMAPE: ",errormeasures["SMAPE"]*100,"%"),sep="; "));
        print(paste(paste0("MASE: ",errormeasures["MASE"]),
                    paste0("MASALE: ",errormeasures["MASALE"]*100,"%"),sep="; "));
    }
}

return(list(persistence=vecg,matF=matF,matw=matw,states=matxt,fitted=y.fit,forecast=y.for,
            lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,actuals=data,
            holdout=y.holdout,ICs=ICs,CF=CF.objective,FI=FI,xreg=xreg,accuracy=errormeasures));
}
