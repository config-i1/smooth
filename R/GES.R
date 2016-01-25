ges <- function(data, bounds=FALSE, seasonality=c("N","S","F"),
                trace=FALSE, CF.type=c("TLV","GV","TV","hsteps","MAD","HAM"),
                use.test=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric"),
                xreg=NULL, holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
                ...){
# General Exponential Smoothing function

# Start measuring the time of calculations
    start.time <- Sys.time()

    CF.type <- CF.type[1]
    seasonality <- seasonality[1]
    int.type <- substring(int.type[1],1,1)

# Check the provided type of interval
    if(int.type!="p" & int.type!="s" & int.type!="n"){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'semiparametric'."))
        int.type <- "s";
    }

    if(any(is.na(data))){
        message("Data contains NAs. These observations will be excluded.")
        datanew <- data[!is.na(data)]
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=frequency(data))
        }
        data <- datanew
    }
    
# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h

# Define the actual values
    y <- as.vector(data)

# Check if the data is vector
    if(!is.numeric(data) & !is.ts(data)){
        stop("The provided data is not a vector or ts object! Can't build any model!", call.=FALSE)
    }

    seasfreq <- frequency(data)

### Check the seasonaity type
    if(seasonality!="N" & seasonality!="S" & seasonality!="F"){
        message("Wrong seasonality type! Should be 'N', 'S' or 'F'.")
        if(seasfreq==1){
            message("Data is non-seasonal. Changing seasonal component to 'N'")
            seasonality <- "N"
        }
        else{
        message("Changing to 'F'")
        seasonality <- "F"
        }
    }
    if(seasonality!="N" & seasfreq==1){
        message("Cannot build the seasonal model on the data with the frequency 1.")
        message("Switching to non-seasonal model")
        seasonality <- "N"
    }

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
            stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=F)
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
        matxtreg <- matrix(NA,max(obs+seasfreq,obs.all),1);
        colnames(matxtreg) <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
        matxtreg[1:seasfreq,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
# Redefine the number of components of ETS.
#        n.components <- n.components + 1;
        }
##### The case with matrices and data frames #####
        else if(is.matrix(xreg) | is.data.frame(xreg)){
    # If xreg is matrix or data frame
            if(nrow(xreg)!=obs & nrow(xreg)!=obs.all){
                stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=F)
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
            matxtreg <- matrix(NA,max(obs+seasfreq,obs.all),n.exovars)
            colnames(matxtreg) <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
            matwex <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
            matxtreg[1:seasfreq,] <- rep(t(solve(t(matx[1:obs,]) %*% matx[1:obs,],tol=1e-50) %*% t(matx[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],each=seasfreq);
# Redefine the number of components of ETS.
#            n.components <- n.components + n.exovars;
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=F);
        }
# Redefine the number of all the parameters. Used in AIC mainly!
#        n.param <- n.param + n.exovars;
    }
    else{
        n.exovars <- 1;
        matwex <- matrix(0,max(obs+seasfreq,obs.all),1);
        matxtreg <- matrix(0,max(obs+seasfreq,obs.all),1);
    }

elements.ges <- function(C){
    if(use.test==TRUE){
        C[C.const1] <- 1
        C[C.const0] <- 0
    }
    matw <- matrix(C[1:n.components],1,n.components)
    if(seasonality=="N"){
        matF <- matrix(C[(n.components+1):(n.components+4)],2,2)
        vecg <- matrix(C[7:8],2,1)
        xt <- C[9:10]
    }
    else if(seasonality=="S"){
        matF <- matrix(C[(n.components+1):(n.components+4)],2,2)
        vecg <- matrix(C[7:8],2,1)
        xt <- matrix(NA,seasfreq,n.components)
        xt[1:seasfreq,1:n.components] <- C[8+(1:(2*seasfreq))]
    }
    else{
        matF <- matrix(0,4,4)
        matF[1:2,1:2] <- C[(n.components+1):(n.components+4)]
        matF[3:4,3:4] <- C[(n.components+5):(n.components+8)]
        vecg <- matrix(C[(n.components+9):(n.components+12)],4,1)
        xt <- matrix(NA,seasfreq,n.components)
        xt[1:seasfreq,1:2] <- rep(C[(n.components+13):(n.components+14)],each=seasfreq)
        xt[1:seasfreq,3:n.components] <- C[n.components+14+(1:(2*seasfreq))]
    }
# If exogenous are included
    if(!is.null(xreg)){
        matxtreg[1:seasfreq,] <- rep(C[(length(C)-n.exovars+1):length(C)],each=seasfreq);
    }

    return(list(matw=matw,matF=matF,vecg=vecg,xt=xt,matxtreg=matxtreg))
}


# Function calculates the power of matrix
matrix.power <- function(A, n) {
    if(n==0){
        B <- diag(nrow(A))
    }
    else{
        B <- diag(nrow(A))
        for(i in 1:n){
            B <- A %*% B
        }
    }
    return(B)
}

# Function makes interval forecasts
forec.inter.ges <- function(matw,matF,vecg,h,s2,int.w,y.for){
    # Vector of variance of states
    mat.var.states <- array(NA,c(2,2,h))
    mat.var.states[,,1] <- 0
    # Vector of final variances
    vec.var <- rep(NA,h)
    vec.var[1] <- s2
    
    if(h>1){
        for(i in 2:h){
            mat.var.states[,,i] <- matF %*% mat.var.states[,,i-1] %*% t(matF) + vecg %*% t(vecg) * s2
            vec.var[i] <- matw %*% mat.var.states[,,i] %*% t(matw) + s2
        }
    }
    
    y.low <- y.for + qt((1-int.w)/2, df=max(obs - 10,1)) * sqrt(vec.var)
    y.high <- y.for + qt(1-(1-int.w)/2, df=max(obs - 10,1)) * sqrt(vec.var)
    
    return(list(y.low=y.low,y.high=y.high))
}

# Function creates bounds for the estimates
hin.constrains <- function(C){
    
    elements <- elements.ges(C)
    matw <- elements$matw
    matF <- elements$matF
    vecg <- elements$vecg

    if(any(is.nan(matF - vecg %*% matw))){
        D <- -0.1
    }
    else{
        D <- 1 - abs(eigen(matF - vecg %*% matw)$values)
    }
    return(D)
}

# Cost function for GES
CF <- function(C){
    elements <- elements.ges(C)
    matw <- elements$matw
    matF <- elements$matF
    vecg <- elements$vecg
    matxt[1:seasfreq,] <- elements$xt
    matxtreg[1:seasfreq,] <- elements$matxtreg[1:seasfreq,]

    CF.res <- optimizerwrap(matxt,matF,matrix(matw,1,length(matw)),as.matrix(y[1:obs]),matrix(vecg,length(vecg),1),h,"A","A","N",seasfreq,trace,CF.type,normalizer,matwex,matxtreg)

    return(CF.res)
}

Likelihood.value <- function(C){
    if(trace==TRUE & (CF.type=="TLV" | CF.type=="GV")){
        return(-obs/2 *((h^trace)*log(2*pi*exp(1)) + CF(C)))
    }
    else{
        return(-obs/2 *((h^trace)*log(2*pi*exp(1)) + log(CF(C))))
    }
}

#####Start the calculations#####
# Initial values of matxt
    slope <- cov(y[1:min(12,obs)],c(1:min(12,obs)))/var(c(1:min(12,obs)))
    intercept <- mean(y[1:min(12,obs)]) - slope * (mean(c(1:min(12,obs))) - 1)

    if(seasonality=="N"){
        seasfreq <- 1
        lags <- c(1,1)
        n.components <- 2
# matw, matF, vecg, xt
        C <- c(rep(1,2),c(1,0,1,1),rep(0.3,2),intercept,slope)
        C.lower <- c(rep(-2,6),rep(-1,2),-Inf,-Inf)
        C.upper <- c(rep(2,6),rep(2,2),Inf,Inf)
    }
    else if(seasonality=="S"){
        lags <- c(seasfreq,seasfreq)
        n.components <- 2
# matw, matF, vecg, season.xt
        C <- c(rep(1,2),c(1,0,1,1),rep(0.3,2),y[1:seasfreq],diff(y)[1:seasfreq])
        C.lower <- c(rep(-2,6),rep(0,2),rep(-Inf,2*seasfreq))
        C.upper <- c(rep(2,6),rep(1,2),rep(Inf,2*seasfreq))
    }
    else{
        lags <- c(1,1,seasfreq,seasfreq)
        n.components <- 4
# matw, matF, vecg, xt, season.xt
        C <- c(rep(1,4),c(1,0,1,1,1,0,0,1),rep(0.3,4),intercept,slope,y[1:seasfreq],diff(y)[1:seasfreq])
        C.lower <- c(rep(-2,6),rep(0,2),min(data[1:min(12,obs)]),min(diff(data[1:min(12,obs)])),rep(-Inf,2*seasfreq))
        C.upper <- c(rep(2,6),rep(1,2),max(data[1:min(12,obs)]),max(diff(data[1:min(12,obs)])),rep(Inf,2*seasfreq))
    }

    if(!is.null(xreg)){
        C <- c(C,matxtreg[seasfreq,])
        C.lower <- c(C.lower,rep(-Inf,n.exovars))
        C.upper <- c(C.upper,rep(Inf,n.exovars))
    }

    C.const1 <- rep(FALSE,length(C))
    C.const0 <- C.const1
#    C.orig <- C

    matxt <- matrix(NA,nrow=(obs+seasfreq),ncol=n.components)
    elements <- elements.ges(C)
    matw <- elements$matw
    matF <- elements$matF
    vecg <- elements$vecg
    matxt[1:seasfreq,] <- elements$xt
    matxtreg[1:seasfreq,] <- elements$matxtreg[1:seasfreq,]

    y.fit <- rep(NA,obs)
    errors <- rep(NA,obs)

    if(trace==TRUE & CF.type=="GV"){
        normalizer <- mean(abs(diff(y[1:obs])))
    }
    else{
        normalizer <- 0;
    }

    if(bounds==FALSE){
        res <- nlminb(C, CF)
        CF.objective <- res$objective
    }
    else{
        res <- nloptr::cobyla(C, CF, hin=hin.constrains, lower=C.lower, upper=C.upper)
        CF.objective <- res$value
    }

    C <- res$par

    if(any(hin.constrains(C)<0) & silent!=FALSE){
        message("Unstable model is estimated! Use 'bounds=TRUE' to address this issue!")
    }

    n.param <- length(C)
    if(!is.null(xreg)){
        n.param <- n.param + n.exovars
    }

    FI <- numDeriv::hessian(Likelihood.value,C)

########## Statistical test of the imaginary part. If it is not far from 1, use 1.
    if(use.test==TRUE){
        FIvar <- solve(-FI)
        C.constvals <- C
        for(i in 1:8){
            if(abs((C[i]-1)/sqrt(abs(FIvar[i,i])))<qt(0.975,obs-n.param)){
                C.constvals[i] <- 31337.31337
            }
            if(i!=1){
                if(abs((C[i])/sqrt(abs(FIvar[i,i])))<qt(0.975,obs-n.param)){
                    C.constvals[i] <- 31337.31338
                }
            }
        }
        C.const1 <- C.constvals==31337.31337
        C.const0 <- C.constvals==31337.31338
        C.const1 <- !(C.const1 & C.const0) & C.const1
        C.const1 <- !(C.const1 & C.const0) & C.const0
#        C <- C.orig
        C[C.const1] <- 1
        C[C.const0] <- 0

# If any variable needs to be set constant, reestimate the model
        if(any(C.const0+C.const1)){
            if(bounds==FALSE){
                res <- nlminb(C, CF)
                CF.objective <- res$objective
            }
            else{
                res <- nloptr::cobyla(C, CF, hin=hin.constrains, lower=C.lower, upper=C.upper)
                CF.objective <- res$value
            }

            C <- res$par
            C[C.const1] <- 1
            C[C.const0] <- 0

            n.param <- length(C)
            if(!is.null(xreg)){
                n.param <- n.param + n.exovars
            }
        }
    }

    elements <- elements.ges(C)
    matw <- elements$matw
    matF <- elements$matF
    vecg <- elements$vecg
    matxt[1:seasfreq,] <- elements$xt
    matxtreg[1:seasfreq,] <- elements$matxtreg[1:seasfreq,]

    fitting <- fitterwrap(matxt,matF,matrix(matw,1,length(matw)),as.matrix(y[1:obs]),matrix(vecg,length(vecg),1),"A","A","N",seasfreq,matwex,matxtreg)
    matxt <- ts(fitting$matxt,start=(time(data)[1] - deltat(data)*seasfreq),frequency=frequency(data))
    y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data))

    if(!is.null(xreg)){
# Write down the matxtreg and copy values for the holdout
        matxtreg[1:nrow(fitting$matxtreg),] <- fitting$matxtreg
        matxtreg[(obs.all-h+1):obs.all,] <- rep(matxtreg[1,],each=h)
    }

    errors.mat <- ts(errorerwrap(matxt,matF,matrix(matw,1,length(matw)),as.matrix(y[1:obs]),h,"A","A","N",seasfreq,TRUE,matwex,matxtreg),start=start(data),frequency=frequency(data))
    colnames(errors.mat) <- paste0("Error",c(1:h))
    errors <- ts(errors.mat[,1],start=start(data),frequency=frequency(data))

    y.for <- ts(forecasterwrap(matrix(matxt[(obs+1):(obs+seasfreq),],nrow=seasfreq),matF,matrix(matw,nrow=1),h,"A","N",seasfreq,matrix(matwex[(obs.all-h+1):(obs.all),],ncol=n.exovars),matrix(matxtreg[(obs.all-h+1):(obs.all),],ncol=n.exovars)),start=time(data)[obs]+deltat(data),frequency=frequency(data))
    s2 <- mean(errors^2)

    if(any(is.na(y.fit),is.na(y.for))){
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

    if(intervals==TRUE){
        if(int.type=="p"){
            forec.int <- forec.inter.ges(matw,matF,vecg,h,s2,int.w,y.for)
            y.low <- forec.int$y.low
            y.high <- forec.int$y.high
        }
        else if(int.type=="s"){
            y.var <- colMeans(errors.mat^2,na.rm=T)
            y.low <- ts(y.for + qt((1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var),start=start(y.for),frequency=frequency(data))
            y.high <- ts(y.for + qt(1-(1-int.w)/2,df=(obs - n.components - n.exovars))*sqrt(y.var),start=start(y.for),frequency=frequency(data))
        }
        else{
            y.var <- apply(errors.mat,2,quantile,probs=c((1-int.w)/2,1-(1-int.w)/2),na.rm=T)
            y.low <- ts(y.for + y.var[1,],start=start(y.for),frequency=frequency(data))
            y.high <- ts(y.for + y.var[2,],start=start(y.for),frequency=frequency(data))
        }
    }
    else{
        y.low <- NA
        y.high <- NA
    }

# Information criteria
    llikelihood <- Likelihood.value(C)

    AIC.coef <- 2*n.param*h^trace - 2*llikelihood
    AICc.coef <- AIC.coef + 2 * n.param*h^trace * (n.param + 1) / (obs - n.param - 1)
    BIC.coef <- log(obs)*n.param - 2*llikelihood

    ICs <- c(AIC.coef, AICc.coef, BIC.coef)
    names(ICs) <- c("AIC", "AICc", "BIC")

    if(!is.null(xreg)){
        matxt <- cbind(matxt,matxtreg[1:nrow(matxt),])
        colnames(matxt) <- c(paste0("Component ",c(1:n.components)),colnames(matxtreg))
    }
    else{
        colnames(matxt) <- paste0("Component ",c(1:n.components))
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
    print(paste0("Time elapsed: ",round(as.numeric(Sys.time() - start.time,units="secs"),2)," seconds"))
    print(paste0("Persistence vector g: ", paste(round(vecg,3),collapse=", ")))
    print("Transition matrix F: ")
    print(round(matF,3))
    print(paste0("Measurement vector w: ",paste(round(matw,3),collapse=", ")))
    print(paste0("Residuals sigma: ",round(sqrt(mean(errors^2)),3)))
    print(paste0("Initial components: ", paste(round(matxt[seasfreq,1:length(vecg)],3),collapse=", ")))
    if(!is.null(xreg)){
        print(paste0("Xreg coefficients: ", paste(round(matxtreg[seasfreq,],3),collapse=", ")))
    }
    if(trace==TRUE){
        print(paste0("CF type: trace with ",CF.type, "; CF value is: ",round(CF.objective,0)))
    }
    else{
        print(paste0("CF type: one step ahead; CF value is: ",round(CF.objective,0)))
    }
    if(intervals==TRUE){
        if(int.type=="p"){
            int.type <- "parametric"
        }
        else if(int.type=="s"){
            int.type <- "semiparametric"
        }
        if(int.type=="n"){
            int.type <- "nonparametric"
        }
        print(paste0(int.w*100,"% ",int.type," intervals were constructed"))
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                   lower=y.low,upper=y.high,int.w=int.w,legend=legend);
    }
    else{
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,legend=legend);
    }
#    print(paste0("Unbiased log-likelihood: ",round((llikelihood - n.param),0)))
    print(paste0("AIC: ",round(AIC.coef,3)," AICc: ", round(AICc.coef,3)))
    if(holdout==T){
        print(paste0("MAPE: ",errormeasures["MAPE"]*100,"%"));
        print(paste0("MASE: ",errormeasures["MASE"]));
        print(paste0("MASALE: ",errormeasures["MASALE"]*100,"%"));
        if(intervals==TRUE){
            print(paste0(round(sum(as.vector(data)[(obs+1):obs.all]<y.high &
                    as.vector(data)[(obs+1):obs.all]>y.low)/h*100,0),
                    "% of values are in the interval"))
        }
    }
}

return(list(persistence=vecg,matF=matF,states=matxt,matw=matw,fitted=y.fit,forecast=y.for,
            lower=y.low,upper=y.high,residuals=errors,actuals=data,holdout=y.holdout,
            ICs=ICs,CF=CF.objective,FI=FI,xreg=xreg,accuracy=errormeasures))
}