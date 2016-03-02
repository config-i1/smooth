ces <- function(data, h=1, holdout=FALSE, C=c(1.1, 1), bounds=FALSE,
                seasonality=c("N","S","P","F"), xreg=NULL, trace=FALSE,
                CF.type=c("TLV","TV","GV","hsteps"), use.test=FALSE,
                intervals=FALSE, int.w=0.95,
                silent=FALSE, legend=TRUE){

# Function estimates CES in state-space form with sigma = error
#  and returns complex smoothing parameter value, fitted values,
#  residuals, point and interval forecasts, matrix of CES components and values of
#  information criteria.
#
#    Copyright (C) 2015  Ivan Svetunkov

# Start measuring the time of calculations
  start.time <- Sys.time();

  seasonality <- seasonality[1];
  CF.type <- CF.type[1];
# If the user typed wrong seasonality, use the "Full" instead
  if(seasonality!="N" & seasonality!="S" & seasonality!="P" & seasonality!="F"){
    message(paste0("Wrong seasonality type: '",seasonality, "'. Changing it to 'F'"));
    seasonality <- "F";
  }

# If the wrong CF.type is defined, change it back to default
  if(trace==TRUE & all(CF.type!=c("TLV","TV","GV","hsteps"))){
    message(paste0("Wrong cost function type defined: '",CF.type, "'. Changing it to 'TLV'"));
    CF.type <- "TLV";
  }

  if(any(is.na(data))){
    if(silent==FALSE){
      message("Data contains NAs. These observations will be excluded.")
    }
    datanew <- data[!is.na(data)]
    if(is.ts(data)){
      datanew <- ts(datanew,start=start(data),frequency=frequency(data))
    }
    data <- datanew
  }

# Define obs.all, the overal number of observations (in-sample + holdout)
  obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
  obs <- length(data) - holdout*h;

# Define the actual values
  y <- as.vector(data);

# Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
#   matxt - the matrix with the components, pt.lags is the lags used in pt matrix.
  if(seasonality=="N"){
  # No seasonality
    seas.lag <- 1;
    matw <- matrix(c(1,0,0,1),2,2);
    matxt <- matrix(NA,max(obs.all+seas.lag,obs+2*seas.lag),2);
    colnames(matxt) <- c("level","potential");
    matxt[1,] <- c(mean(y[1:min(10,obs)]),mean(y[1:min(10,obs)])/C[1]);
    ces.name <- "Complex Exponential Smoothing";
# Define the number of all the parameters (smoothing parameters + initial states). Used in AIC mainly!
    n.param <- length(C) + 2;
    n.components <- 2;
  }
  else if(seasonality=="S"){
  # Simple seasonality, lagged CES
    seas.lag <- frequency(data);
    matw <- matrix(c(1,0,0,1),2,2);
    matxt <- matrix(NA,max(obs.all+seas.lag,obs+2*seas.lag),2);
    colnames(matxt) <- c("level.s","potential.s");
    matxt[1:seas.lag,1] <- y[1:seas.lag];
    matxt[1:seas.lag,2] <- matxt[1:seas.lag,1]/C[1];
    ces.name <- "Lagged Complex Exponential Smoothing (Simple seasonality)";
    n.param <- length(C) + 2*seas.lag;
    n.components <- 2;
  }
  else if(seasonality=="P"){
  # Partial seasonality with a real part only
    seas.lag <- frequency(data);
    C <- c(C,0.5);
    matw <- matrix(c(1,0,0,1,1,0),2,3);
    pt.lags <- c(1,1,seas.lag);
    matxt <- matrix(NA,max(obs.all+seas.lag,obs+2*seas.lag),3);
    colnames(matxt) <- c("level","potential","seasonal");
    matxt[1:seas.lag,1] <- mean(y[1:seas.lag]);
    matxt[1:seas.lag,2] <- matxt[1:seas.lag,1]/C[1];
    matxt[1:seas.lag,3] <- decompose(data,type="a")$figure;
    ces.name <- "Complex Exponential Smoothing with a partial (real) seasonality";
    n.param <- length(C) + 2 + seas.lag;
    n.components <- 3;
  }
  else if(seasonality=="F"){
  # Full seasonality with both real and imaginary parts
    seas.lag <- frequency(data);
    C <- c(C,C);
    matw <- matrix(c(1,0,0,1,1,0,0,1),2,4);
    pt.lags <- c(1,1,seas.lag,seas.lag);
    matxt <- matrix(NA,max(obs.all+seas.lag,obs+2*seas.lag),4);
    colnames(matxt) <- c("level","potential","seasonal 1", "seasonal 2");
    matxt[1:seas.lag,1] <- mean(y[1:seas.lag]);
    matxt[1:seas.lag,2] <- matxt[1:seas.lag,1]/C[1];
    matxt[1:seas.lag,3] <- decompose(data,type="a")$figure;
    matxt[1:seas.lag,4] <- matxt[1:seas.lag,3]/C[3];
    ces.name <- "Complex Exponential Smoothing with a full (complex) seasonality";
    n.param <- length(C) + 2 + 2*seas.lag;
    n.components <- 4;
  }

# Check the exogenous variable if it is present and
# fill in the values of xreg if it is absent in the holdout sample.
  if(!is.null(xreg)){
    if(any(is.na(xreg))){
      message("The exogenous variables contain NAs! This may lead to problems during estimation and forecast.");
    }
    if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
# If xreg is vector or simple ts
      if(length(xreg)!=obs & length(xreg)!=obs.all){
        stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=F)
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
      matwex <- matrix(xreg,ncol=1);
# Define the second matxtreg to fill in the coefs of the exogenous vars
      matxtreg <- matrix(NA,max(obs.all+seas.lag,obs+2*seas.lag),1);
      colnames(matxtreg) <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
      matxtreg[1:seas.lag,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
# Redefine the number of components of CES.
      n.components <- n.components + 1;
    }
    else if(is.matrix(xreg) | is.data.frame(xreg)){
    # If xreg is matrix or data frame
      if(nrow(xreg)!=obs & nrow(xreg)!=obs.all){
        stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=F)
      }
      if(nrow(xreg)==obs){
        if(silent==FALSE){
	        message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
        }
        for(j in 1:h){
          xreg <- rbind(xreg,xreg[obs,]);
        }
#        rownames(xreg) <- c(1:obs.all);
      }
# mat.x is needed for the initial values of coefs estimation using OLS
      mat.x <- as.matrix(cbind(rep(1,obs.all),xreg));
      n.exovars <- ncol(xreg);
# Define the second matxtreg to fill in the coefs of the exogenous vars
      matxtreg <- matrix(NA,max(obs.all+seas.lag,obs+2*seas.lag),n.exovars)
      colnames(matxtreg) <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
      matwex <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
      matxtreg[1:seas.lag,] <- rep(t(solve(t(mat.x[1:obs,]) %*% mat.x[1:obs,],tol=1e-50) %*%
                                         t(mat.x[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],
                                   each=seas.lag);
# Redefine the number of components of CES.
      n.components <- n.components + n.exovars;
    }
    else{
      stop("Unknown format of xreg. Aborting!",call.=F);
    }
# Redefine the number of all the parameters. Used in AIC mainly!
    n.param <- n.param + n.exovars;
  }
  else{
    n.exovars <- 1;
    matwex <- matrix(0,obs.all,1);
    matxtreg <- matrix(0,max(obs.all+seas.lag,obs+2*seas.lag),1);
  }

# Define the vector of fitted, forecasted values and overall
  y.fit <- rep(NA,obs);
  y.for <- rep(NA,h);

# Define vector of all the errors
  errors <- rep(NA,obs);

# Define "F" and "g" matrices for the state-space CES
  state.space.elements <- function(seasonality, C){
    if(seasonality=="N" | seasonality=="S"){
    # No seasonality or Simple seasonality, lagged CES
      matF <- matrix(c(1,1,C[2]-1,1-C[1]),2,2);
      vecg <- matrix(c(C[1]-C[2],C[1]+C[2]),2,1);
##### Making SES with backcast #####
#      matF <- matrix(c(1,0,0,0),2,2);
#      vecg <- matrix(c(C[1],0),2,1);
    }
    else if(seasonality=="P"){
    # Partial seasonality with a real part only
      matF <- matrix(c(1,1,0,C[2]-1,1-C[1],0,0,0,1),3,3);
      vecg <- matrix(c(C[1]-C[2],C[1]+C[2],C[3]),3,1);
    }
    else if(seasonality=="F"){
    # Full seasonality with both real and imaginary parts
      matF <- matrix(c(1,1,0,0,C[2]-1,1-C[1],0,0,0,0,1,1,0,0,C[4]-1,1-C[3]),4,4);
      vecg <- matrix(c(C[1]-C[2],C[1]+C[2],C[3]-C[4],C[3]+C[4]),4,1);
    }
    return(list(matF=matF,vecg=vecg));
  }

# Cost function for CES
  CF <- function(C){
    # Obtain the elements of CES
    if(!is.null(xreg)){
      matxtreg[1:seas.lag,] <- rep(C[(n.components-n.exovars+1):n.components],each=seas.lag);
    }

    ces.elements <- state.space.elements(seasonality, C);
    matF <- ces.elements$matF;
    vecg <- ces.elements$vecg;

    CF.res <- cesoptimizerwrap(matxt,matF,matrix(matw[1,],nrow=1),matrix(y[1:obs],ncol=1),
                               vecg,h,seasonality,seas.lag,trace,CF.type,normalizer,matwex,matxtreg);

#    CF.res <- ssoptimizerwrap(matxt, matF, matrix(matw[1,],obs.all,length(matw),byrow=TRUE), matrix(1,obs.all,length(matw)),
#                              as.matrix(y[1:obs]), matrix(vecg,length(vecg),1), h, matrix(1,2,1), CF.type,
#                              normalizer, matwex, matxtreg);

    if(is.nan(CF.res) | is.na(CF.res)){
        CF.res <- 1e100;
    }
    return(CF.res);
  }

# Create a function for constrains for CES based on eigenvalues of discount matrix of partial state-space CES
  constrains <- function(C){
    ces.elements <- state.space.elements(seasonality, C);
    matF <- ces.elements$matF;
    vecg <- ces.elements$vecg;
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
  likelihood <- function(C){
    if(trace==TRUE & (CF.type=="GV" | CF.type=="TLV")){
        return(-obs/2 *((h^trace)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(-obs/2 *((h^trace)*log(2*pi*exp(1)) + log(CF(C))));
    }
  }

  if(!is.null(xreg)){
    lowerb <- c(rep(0,length(C)),rep(-Inf,n.exovars));
    upperb <- c(rep(2,length(C)),rep(Inf,n.exovars));
    C <- c(C,matxtreg[1,]);
  }
  else{
    lowerb <- rep(0,n.components);
    upperb <- rep(2,n.components);
  }

  if(trace==TRUE & CF.type=="GV"){
    normalizer <- mean(abs(diff(y[1:obs])));
  }
  else{
    normalizer <- 0;
  }

# Estimate CES
  if(bounds==TRUE){
    res <- nloptr::cobyla(C, CF, hin=constrains, lower=lowerb, upper=upperb);
    C <- res$par;
    CF.objective <- res$value;
  }
  else{
    res <- nlminb(C, CF, lower=lowerb, upper=upperb);
    C <- res$par;
    CF.objective <- res$objective;
  }

  llikelihood <- likelihood(C);
  FI <- numDeriv::hessian(likelihood,C);

  if(!is.null(xreg)){
    FI <- FI[1:(n.components-n.exovars),1:(n.components-n.exovars)];
  }

# Information criteria are calculated here with the constant part "log(2*pi*exp(1)/obs)*obs".
  AIC.coef <- 2*n.param*h^trace - 2*llikelihood;
  AICc.coef <- AIC.coef + 2 * n.param * (n.param + 1) / (obs - n.param - 1);
  BIC.coef <- log(obs)*n.param*h^trace - 2 * llikelihood;
# Information criterion derived and used especially for CES
#   k here is equal to number of coefficients/2 + number of complex initial states of CES.
  CIC.coef <- 2*(length(C)/2 + seas.lag)*h^trace - 2*llikelihood;

  ICs <- c(AIC.coef, AICc.coef, BIC.coef,CIC.coef);
  names(ICs) <- c("AIC", "AICc", "BIC","CIC");

########## Statistical test of the imaginary part. If it is not far from 1, use 1.
##### Should be redone in the style of GES test.
  if(use.test==TRUE){
    if(abs((C[2]-1)/sqrt(abs(solve(-FI)[2,2])))<qt(0.975,50)){
      C[2] <- 1;
#      seas.test <- FALSE;
    }
    if(seasonality=="F"){
      if(abs((C[4]-1)/sqrt(abs(solve(-FI)[4,4])))<qt(0.975,50)){
        C[4] <- 1;
#        seas.test <- TRUE;
      }
    }
  }

# Obtain the elements of CES
  ces.elements <- state.space.elements(seasonality, C);
  matF <- ces.elements$matF;
  vecg <- ces.elements$vecg;

# Change F and g matrices if exogenous variables are presented
  if(!is.null(xreg)){
    matxtreg[1:seas.lag,] <- rep(C[(n.components-n.exovars+1):n.components],each=seas.lag);
  }

# Estimate the elements of the transitional equation, fitted values and errors
  fitting <- cesfitterwrap(matxt,matF,matrix(matw[1,],nrow=1),as.matrix(y[1:obs]),vecg,
                           seasonality,seas.lag,matwex,matxtreg)
  matxt[,] <- fitting$matxt;
  y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
  matxtreg[,] <- fitting$xtreg;

  errors.mat <- ts(ceserrorerwrap(matxt,matF,matrix(matw[1,],nrow=1),as.matrix(y[1:obs]),h,
                                  seasonality,seas.lag,matwex,matxtreg),start=start(data),
                   frequency=frequency(data));
  colnames(errors.mat) <- paste0("Error",c(1:h));
  errors.mat <- ts(errors.mat,start=start(data),frequency=frequency(data));
  errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));

  y.for <- ts(cesforecasterwrap(matrix(matxt[((obs-seas.lag+1):obs)+seas.lag,],nrow=seas.lag),
                                matF,matrix(matw[1,],nrow=1),h,seasonality,seas.lag,
                                matrix(matwex[(obs.all-h+1):obs.all,],ncol=n.exovars),
                                matrix(matxtreg[(obs.all-h+1):obs.all,],ncol=n.exovars)),
              start=time(data)[obs]+deltat(data),frequency=frequency(data));


  if(intervals==TRUE){
    y.var <- cesforecastervar(matF,matrix(matw[1,],nrow=1),vecg,h,var(errors),seasonality,seas.lag);
    y.low <- ts(y.for + qt((1-int.w)/2,df=(obs - n.components))*sqrt(y.var),start=start(y.for),frequency=frequency(data));
    y.high <- ts(y.for + qt(1-(1-int.w)/2,df=(obs - n.components))*sqrt(y.var),start=start(y.for),frequency=frequency(data));
  }
  else{
    y.low=NA;
    y.high=NA;
  }

  if(any(is.na(y.fit),is.na(y.for))){
    message("Something went wrong during the optimisation and NAs were produced!");
    message("Please check the input and report this error if it persists to the maintainer.");
  }

  y.for <- ts(y.for,start=time(data)[obs]+deltat(data),frequency=frequency(data));
  matxt <- ts(matxt,start=start(data),frequency=frequency(data));
  if(!is.null(xreg)){
    matxt <- cbind(matxt,matxtreg);
  }

  if(silent==FALSE){
    if(bounds==FALSE & sum(1-constrains(C)>1)>=1){
      message("Non-stable model estimated! Use with care! To avoid that reestimate ces using 'bounds=TRUE'.");
    }
  }

# Right down the smoothing parameters
  A <- complex(real=C[1],imaginary=C[2]);

  if(seasonality=="P"){
    B <- C[3];
  }
  else if(seasonality=="F"){
    B <- complex(real=C[3],imaginary=C[4]);
  }
  else{
    B <- NA;
  }

  if(holdout==TRUE){
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
    print("Model constructed:");
    print(ces.name);
    print(paste0("a0 + ia1: ",A));

    if(seasonality=="P"){
      print(paste0("b: ",B));
    }
    else if(seasonality=="F"){
      print(paste0("b0 + ib1: ",B));
    }

    print("ABS Eigenvalues for stability condition:");
    print(1-constrains(C));
    if(trace==FALSE){
      CF.type <- "1 step ahead";
    }
    print(paste0("Cost function used: ",CF.type,". CF value is: ",round(CF.objective,0)));
    print(paste0("AIC: ",round(AIC.coef,3),"; AICc: ", round(AICc.coef,3),
                 "; BIC: ", round(BIC.coef,3), "; CIC:", round(CIC.coef,3)));

    if(intervals==TRUE){
        print(paste0(int.w*100,"% intervals were constructed"));
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                   lower=y.low,upper=y.high,int.w=int.w,legend=legend);
    }
    else{
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,legend=legend);
    }
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

return(list(A=A,B=B,residuals=errors,errors=errors.mat,holdout=y.holdout,
            actuals=data,fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,
            states=matxt,ICs=ICs,FI=FI,xreg=matwex,accuracy=errormeasures));
}
