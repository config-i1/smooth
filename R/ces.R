ces <- function(data, C=c(1.1, 1), seasonality=c("N","S","P","F"),
                initial=c("backcasting","optimal"),
                CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                intermittent=FALSE,
                bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Function estimates CES in state-space form with sigma = error
#  and returns complex smoothing parameter value, fitted values,
#  residuals, point and interval forecasts, matrix of CES components and values of
#  information criteria.
#
#    Copyright (C) 2015 - 2016i  Ivan Svetunkov

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
# Check if "bounds" parameter makes any sense
    if(bounds!="n" & bounds!="a"){
        message("The strange bounds are defined. Switching to 'admissible'.");
        bounds <- "a";
    }

    seasonality <- seasonality[1];
# If the user typed wrong seasonality, use the "Full" instead
    if(seasonality!="N" & seasonality!="S" & seasonality!="P" & seasonality!="F"){
        message(paste0("Wrong seasonality type: '",seasonality, "'. Changing it to 'F'"));
        seasonality <- "F";
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
# Check the provided type of intervals
    if(all(int.type!=c("a","p","s","n"))){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'parametric'."));
        int.type <- "p";
    }

    if(any(is.na(data))){
        if(silent.text==FALSE){
            message("Data contains NAs. These observations will be excluded.")
        }
        datanew <- data[!is.na(data)]
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=datafreq)
        }
        data <- datanew
    }

# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

# If obs is negative, this means that we can't do anything...
    if(obs<=0){
        stop("Not enough observations in sample.",call.=FALSE);
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
        message("Initial value is not selected. Switching to optimal.");
        fittertype <- "o";
    }
    else if(!is.null(initial)){
        if(!is.numeric(initial) | !is.vector(initial)){
            stop("The initial vector is not numeric!",call.=FALSE);
        }
        fittertype <- "o";
    }

# Define the actual values
  y <- matrix(as.vector(data[1:obs]),obs,1);
  datafreq <- frequency(data);

    if(intermittent==TRUE){
        ot <- (y!=0)*1;
        iprob <- mean(ot);
        obs.ot <- sum(ot);
        yot <- matrix(y[y!=0],obs.ot,1);
    }
    else{
        ot <- rep(1,obs);
        iprob <- 1;
        obs.ot <- obs;
        yot <- y;
    }

# If the data is not intermittent, let's assume that the parameter was switched unintentionally.
    if(iprob==1){
        intermittent <- FALSE;
    }

# Stop if number of observations is less than horizon and multisteps is chosen.
    if((multisteps==TRUE) & (obs.ot < h+1)){
        message(paste0("Do you seriously think that you can use ",CF.type," with h=",h," on ",obs.ot," non-zero observations?!"));
        stop("Not enough observations for multisteps cost function.",call.=FALSE);
    }
    else if((multisteps==TRUE) & (obs.ot < 2*h)){
        message(paste0("Number of observations is really low for a multisteps cost function! We will try but cannot guarantee anything..."));
    }


# Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
#   matvt - the matrix with the components, lags is the lags used in pt matrix.
    if(seasonality=="N"){
# No seasonality
        maxlag <- 1;
        modellags <- c(1,1);
        obs.xt <- max(obs.all+maxlag,obs+2*maxlag);
        matw <- matrix(c(1,0,0,1),2,2);
        matvt <- matrix(NA,obs.xt,2);
        colnames(matvt) <- c("level","potential");
        matvt[1,] <- c(mean(yot[1:min(10,obs.ot)]),mean(yot[1:min(10,obs.ot)])/C[1]);
        ces.name <- "Complex Exponential Smoothing";
# Define the number of all the parameters (smoothing parameters + initial states). Used in AIC mainly!
        n.param <- length(C);
        if(fittertype=="o"){
            n.param <- n.param + 2;
            C <- c(C,matvt[1:maxlag,]);
        }
        n.components <- 2;
    }
    else if(seasonality=="S"){
# Simple seasonality, lagged CES
        maxlag <- datafreq;
        modellags <- c(maxlag,maxlag);
        obs.xt <- max(obs.all+maxlag,obs+2*maxlag);
        matw <- matrix(c(1,0,0,1),2,2);
        matvt <- matrix(NA,obs.xt,2);
        colnames(matvt) <- c("level.s","potential.s");
        matvt[1:maxlag,1] <- y[1:maxlag];
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/C[1];
        ces.name <- "Lagged Complex Exponential Smoothing (Simple seasonality)";
        n.param <- length(C);
        if(fittertype=="o"){
            n.param <- n.param + 2*maxlag;
            C <- c(C,matvt[1:maxlag,]);
        }
        n.components <- 2;
    }
    else if(seasonality=="P"){
# Partial seasonality with a real part only
        maxlag <- datafreq;
        modellags <- c(1,1,maxlag);
        obs.xt <- max(obs.all+maxlag,obs+2*maxlag);
        C <- c(C,0.5);
        matw <- matrix(c(1,0,0,1,1,0),2,3);
        lags <- c(1,1,maxlag);
        matvt <- matrix(NA,obs.xt,3);
        colnames(matvt) <- c("level","potential","seasonal");
        matvt[1:maxlag,1] <- mean(y[1:maxlag]);
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/C[1];
        matvt[1:maxlag,3] <- decompose(data,type="a")$figure;
        ces.name <- "Complex Exponential Smoothing with a partial (real) seasonality";
        n.param <- length(C);
        if(fittertype=="o"){
            n.param <- n.param + 2 + maxlag
            C <- c(C,matvt[1,1:2]);
            C <- c(C,matvt[1:maxlag,3]);
        }
        n.components <- 3;
    }
    else if(seasonality=="F"){
# Full seasonality with both real and imaginary parts
        maxlag <- datafreq;
        modellags <- c(1,1,maxlag,maxlag);
        obs.xt <- max(obs.all+maxlag,obs+2*maxlag);
        C <- c(C,C);
        matw <- matrix(c(1,0,0,1,1,0,0,1),2,4);
        lags <- c(1,1,maxlag,maxlag);
        matvt <- matrix(NA,obs.xt,4);
        colnames(matvt) <- c("level","potential","seasonal 1", "seasonal 2");
        matvt[1:maxlag,1] <- mean(y[1:maxlag]);
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/C[1];
        matvt[1:maxlag,3] <- decompose(data,type="a")$figure;
        matvt[1:maxlag,4] <- matvt[1:maxlag,3]/C[3];
        ces.name <- "Complex Exponential Smoothing with a full (complex) seasonality";
        n.param <- length(C);
        if(fittertype=="o"){
            n.param <- n.param + 2 + 2*maxlag
            C <- c(C,matvt[1,1:2]);
            C <- c(C,matvt[1:maxlag,3:4]);
        }
        n.components <- 4;
    }

# 1 stands for variance
    n.param <- n.param + 1;

    # Stop if number of observations is less than number of parameters
    if(obs.ot <= n.param){
        message(paste0("Number of non-zero observations is ",obs.ot,", while the number of parameters to estimate is ", n.param,"."));
        stop(paste0("Not enough observations for the fit of CES(",seasonality,") model!"),call.=FALSE);
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

# Define the vector of fitted, forecasted values and overall
    y.fit <- rep(NA,obs);
    y.for <- rep(NA,h);

# Define vector of all the errors
    errors <- rep(NA,obs);

# 1 stands for the variance
    n.param <- n.param + intermittent + 1;

    if(estimate.xreg==TRUE){
        n.param <- n.param + estimate.initialX*n.exovars + estimate.FX*(n.exovars^2) + estimate.gX*(n.exovars);
    }

# Define "F" and "g" matrices for the state-space CES
state.space.elements <- function(seasonality, C){
    vt <- matrix(matvt[1:maxlag,],maxlag);
    if(seasonality=="N" | seasonality=="S"){
    # No seasonality or Simple seasonality, lagged CES
        matF <- matrix(c(1,1,C[2]-1,1-C[1]),2,2);
        vecg <- matrix(c(C[1]-C[2],C[1]+C[2]),2,1);
##### Making SES with backcast #####
#      matF <- matrix(c(1,0,0,0),2,2);
#      vecg <- matrix(c(C[1],0),2,1);
        n.coef <- 2;
        if(fittertype=="o"){
            vt[1:maxlag,] <- C[n.coef+(1:2)*maxlag];
            n.coef <- n.coef + maxlag*2;
        }
    }
    else if(seasonality=="P"){
    # Partial seasonality with a real part only
        matF <- matrix(c(1,1,0,C[2]-1,1-C[1],0,0,0,1),3,3);
        vecg <- matrix(c(C[1]-C[2],C[1]+C[2],C[3]),3,1);
        n.coef <- 3;
        if(fittertype=="o"){
            vt[,1:2] <- C[n.coef+(1:2)];
            n.coef <- n.coef + 2;
            vt[,3] <- C[n.coef+(1:maxlag)];
            n.coef <- n.coef + maxlag;
        }
    }
    else if(seasonality=="F"){
    # Full seasonality with both real and imaginary parts
        matF <- matrix(c(1,1,0,0,C[2]-1,1-C[1],0,0,0,0,1,1,0,0,C[4]-1,1-C[3]),4,4);
        vecg <- matrix(c(C[1]-C[2],C[1]+C[2],C[3]-C[4],C[3]+C[4]),4,1);
        n.coef <- 4;
        if(fittertype=="o"){
            vt[,1:2] <- C[n.coef+(1:2)];
            n.coef <- n.coef + 2;
            vt[,3:4] <- C[n.coef+(1:(maxlag*2))];
            n.coef <- n.coef + maxlag*2;
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

    return(list(matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

# Cost function for CES
CF <- function(C){
# Obtain the elements of CES

    elements <- state.space.elements(seasonality, C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    CF.res <- costfunc(matvt, matF, matrix(matw[1,],nrow=1), y, vecg,
                       h, modellags, "A", "N", "N",
                       multisteps, CF.type, normalizer, fittertype,
                       matxt, matat, matFX, vecgX, ot,
                       bounds);

    if(is.nan(CF.res) | is.na(CF.res)){
        CF.res <- 1e100;
    }
    return(CF.res);
}

# Create a function for constrains for CES based on eigenvalues of discount matrix of partial state-space CES
constrains <- function(C){
    elements <- state.space.elements(seasonality, C);
    matF <- elements$matF;
    vecg <- elements$vecg;
# Stability region can not be estimated when exogenous variables are included,
#   that is why we do not include mat.q in the constrains and take the original matw
    if(any(is.nan(matF - vecg %*% matw[1,]))){
        constr <- -0.1;
    }
    else{
        constr <- 1 - abs(eigen(matF - vecg %*% matw[1,])$values);
    }
    return(constr);
}

# Likelihood function
Likelihood.value <- function(C){
    if(CF.type=="TFL"){
        return(obs.ot*log(iprob)*(h^multisteps)
               -obs.ot/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(obs.ot*log(iprob)
               -obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
    }
}

##### Prepare for the optimisation #####
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

    if(CF.type=="TFL"){
        normalizer <- mean(abs(diff(y)));
    }
    else{
        normalizer <- 0;
    }

# Estimate CES
#  if(bounds=="a"){
#    res <- cobyla(C, CF, hin=constrains, lower=lowerb, upper=upperb);
#    C <- res$par;
#    CF.objective <- res$value;
#  }
#  else{
#    res <- nlminb(C, CF, lower=lowerb, upper=upperb);
#    C <- res$par;
#    CF.objective <- res$objective;
#  }
    res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
    C <- res$solution;
    CF.objective <- res$objective;

    llikelihood <- Likelihood.value(C);

    if(FI==TRUE){
        FI <- hessian(Likelihood.value,C);
    }
    else{
        FI <- NULL;
    }

# Information criteria are calculated here with the constant part "log(2*pi*exp(1)/obs)*obs".
    AIC.coef <- 2*n.param*h^multisteps - 2*llikelihood;
    AICc.coef <- AIC.coef + 2 * n.param * (n.param + 1) / (obs.ot - n.param - 1);
    BIC.coef <- log(obs.ot)*n.param*h^multisteps - 2 * llikelihood;
# Information criterion derived and used especially for CES
#   k here is equal to number of coefficients/2 (number of numbers) + number of complex initial states of CES.
    CIC.coef <- 2 * (ceiling(length(C)/2) + maxlag) * h ^ multisteps - 2 * llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef,CIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC","CIC");

# Obtain the elements of CES
    elements <- state.space.elements(seasonality, C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

# Change F and g matrices if exogenous variables are presented
    if(!is.null(xreg)){
        matat[1:maxlag,] <- rep(C[(n.components+1):(n.components+n.exovars)],each=maxlag);
    }

# Estimate the elements of the transitional equation, fitted values and errors
    fitting <- fitterwrap(matvt, matF, matrix(matw[1,],nrow=1), y, vecg,
                          modellags, "A", "N", "N", fittertype,
                          matxt, matat, matFX, vecgX, ot);
    matvt[,] <- fitting$matvt;
    y.fit <- ts(fitting$yfit,start=start(data),frequency=datafreq);
    matat[,] <- fitting$matat;

# Produce matrix of errors
    errors.mat <- ts(errorerwrap(matvt, matF, matrix(matw[1,],nrow=1), y,
                                 h, "A", "N", "N", modellags,
                                 matxt, matat, matFX, ot),
                     start=start(data),frequency=datafreq);
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=datafreq);

# Produce forecast
    y.for <- ts(iprob*forecasterwrap(matrix(matvt[(obs+1):(obs+maxlag),],nrow=maxlag),
                               matF, matrix(matw[1,],nrow=1), h, "N", "N", modellags,
                               matrix(matxt[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                               matrix(matat[(obs.all-h+1):(obs.all),],ncol=n.exovars), matFX),
                start=time(data)[obs]+deltat(data),frequency=datafreq);

    s2 <- as.vector(sum((errors*ot)^2)/(obs.ot-n.param));
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

        vt <- matrix(matvt[cbind(obs-modellags,c(1:n.components))],n.components,1);

        quantvalues <- ssintervals(errors.x, ev=ev, int.w=int.w, int.type=int.type, df=(obs.ot - n.param),
                                 measurement=matrix(matw[1,],nrow=1), transition=matF, persistence=vecg,
                                 s2=s2, modellags=modellags, y.for=y.for, iprob=iprob);
        y.low <- ts(c(y.for) + quantvalues$lower,start=start(y.for),frequency=datafreq);
        y.high <- ts(c(y.for) + quantvalues$upper,start=start(y.for),frequency=datafreq);
    }
    else{
        y.low <- NA;
        y.high <- NA;
    }

    if(any(is.na(y.fit),is.na(y.for))){
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

    y.for <- ts(y.for,start=time(data)[obs]+deltat(data),frequency=datafreq);
    matvt <- ts(matvt,start=start(data),frequency=datafreq);
    if(!is.null(xreg)){
        statenames <- c(colnames(matvt),colnames(matat));
        matvt <- cbind(matvt,matat);
        colnames(matvt) <- statenames;
    }

# Write down initials of states vector and exogenous
    initial <- matvt[1,];
    if(estimate.initialX==TRUE){
        initialX <- matat[1,];
    }

# Right down the smoothing parameters
    A <- complex(real=C[1],imaginary=C[2]);

    if(seasonality=="P"){
        B <- C[3];
        modelname <- "CES(P)";
    }
    else if(seasonality=="F"){
        B <- complex(real=C[3],imaginary=C[4]);
        modelname <- "CES(F)";
    }
    else{
        B <- NULL;
        if(seasonality=="N"){
            modelname <- "CES(N)";
        }
        else{
            modelname <- "CES(S)";
        }
    }

    if(holdout==TRUE){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=datafreq);
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    if(silent.text==FALSE){
        if(bounds!="a" & sum(1-constrains(C)>1)>=1){
            message("Non-stable model was estimated! Use with care! To avoid that reestimate ces using admissible bounds.");
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
                 phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=A, B=B,
                 n.components=sum(modellags), s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
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

return(list(model=modelname,states=matvt,A=A,B=B,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,
            xreg=xreg,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
