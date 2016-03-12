ces <- function(data, C=c(1.1, 1), seasonality=c("N","S","P","F"),
                CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
                use.test=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                bounds=FALSE, holdout=FALSE, h=1, silent=FALSE, legend=TRUE,
                xreg=NULL){
# Function estimates CES in state-space form with sigma = error
#  and returns complex smoothing parameter value, fitted values,
#  residuals, point and interval forecasts, matrix of CES components and values of
#  information criteria.
#
#    Copyright (C) 2015 - 2016i  Ivan Svetunkov

    go.wild <- FALSE;

# Start measuring the time of calculations
    start.time <- Sys.time();

    seasonality <- seasonality[1];
# If the user typed wrong seasonality, use the "Full" instead
    if(seasonality!="N" & seasonality!="S" & seasonality!="P" & seasonality!="F"){
        message(paste0("Wrong seasonality type: '",seasonality, "'. Changing it to 'F'"));
        seasonality <- "F";
    }

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
# Check the provided type of intervals
    if(all(int.type!=c("a","p","s","n"))){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'parametric'."));
        int.type <- "p";
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
    maxlag <- 1;
    modellags <- c(1,1);
    matw <- matrix(c(1,0,0,1),2,2);
    matxt <- matrix(NA,max(obs.all+maxlag,obs+2*maxlag),2);
    colnames(matxt) <- c("level","potential");
    matxt[1,] <- c(mean(y[1:min(10,obs)]),mean(y[1:min(10,obs)])/C[1]);
    ces.name <- "Complex Exponential Smoothing";
# Define the number of all the parameters (smoothing parameters + initial states). Used in AIC mainly!
    n.param <- length(C) + 2;
    n.components <- 2;
  }
  else if(seasonality=="S"){
  # Simple seasonality, lagged CES
    maxlag <- frequency(data);
    modellags <- c(maxlag,maxlag);
    matw <- matrix(c(1,0,0,1),2,2);
    matxt <- matrix(NA,max(obs.all+maxlag,obs+2*maxlag),2);
    colnames(matxt) <- c("level.s","potential.s");
    matxt[1:maxlag,1] <- y[1:maxlag];
    matxt[1:maxlag,2] <- matxt[1:maxlag,1]/C[1];
    ces.name <- "Lagged Complex Exponential Smoothing (Simple seasonality)";
    n.param <- length(C) + 2*maxlag;
    n.components <- 2;
  }
  else if(seasonality=="P"){
  # Partial seasonality with a real part only
    maxlag <- frequency(data);
    modellags <- c(1,1,maxlag);
    C <- c(C,0.5);
    matw <- matrix(c(1,0,0,1,1,0),2,3);
    pt.lags <- c(1,1,maxlag);
    matxt <- matrix(NA,max(obs.all+maxlag,obs+2*maxlag),3);
    colnames(matxt) <- c("level","potential","seasonal");
    matxt[1:maxlag,1] <- mean(y[1:maxlag]);
    matxt[1:maxlag,2] <- matxt[1:maxlag,1]/C[1];
    matxt[1:maxlag,3] <- decompose(data,type="a")$figure;
    ces.name <- "Complex Exponential Smoothing with a partial (real) seasonality";
    n.param <- length(C) + 2 + maxlag;
    n.components <- 3;
  }
  else if(seasonality=="F"){
  # Full seasonality with both real and imaginary parts
    maxlag <- frequency(data);
    modellags <- c(1,1,maxlag,maxlag);
    C <- c(C,C);
    matw <- matrix(c(1,0,0,1,1,0,0,1),2,4);
    pt.lags <- c(1,1,maxlag,maxlag);
    matxt <- matrix(NA,max(obs.all+maxlag,obs+2*maxlag),4);
    colnames(matxt) <- c("level","potential","seasonal 1", "seasonal 2");
    matxt[1:maxlag,1] <- mean(y[1:maxlag]);
    matxt[1:maxlag,2] <- matxt[1:maxlag,1]/C[1];
    matxt[1:maxlag,3] <- decompose(data,type="a")$figure;
    matxt[1:maxlag,4] <- matxt[1:maxlag,3]/C[3];
    ces.name <- "Complex Exponential Smoothing with a full (complex) seasonality";
    n.param <- length(C) + 2 + 2*maxlag;
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
            matxtreg <- matrix(NA,max(obs.all+maxlag,obs+2*maxlag),1);
            colnames(matxtreg) <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
            matxtreg[1:maxlag,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
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
            }
# mat.x is needed for the initial values of coefs estimation using OLS
            mat.x <- as.matrix(cbind(rep(1,obs.all),xreg));
            n.exovars <- ncol(xreg);
# Define the second matxtreg to fill in the coefs of the exogenous vars
            matxtreg <- matrix(NA,max(obs.all+maxlag,obs+2*maxlag),n.exovars)
            colnames(matxtreg) <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
            matwex <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
            matxtreg[1:maxlag,] <- rep(t(solve(t(mat.x[1:obs,]) %*% mat.x[1:obs,],tol=1e-50) %*%
                                             t(mat.x[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],
                                       each=maxlag);
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
        matxtreg <- matrix(0,max(obs.all+maxlag,obs+2*maxlag),1);
        matv <- matrix(1,max(obs+maxlag,obs.all),1);
    }
##### Let's not go wild with xreg for now! #####
    matF2 <- matrix(1,1,1);
    vecg2 <- matrix(1,1,1);

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
      matxtreg[1:maxlag,] <- rep(C[(n.components-n.exovars+1):n.components],each=maxlag);
    }

    ces.elements <- state.space.elements(seasonality, C);
    matF <- ces.elements$matF;
    vecg <- ces.elements$vecg;

    CF.res <- cesoptimizerwrap(matxt,matF,matrix(matw[1,],nrow=1),matrix(y[1:obs],ncol=1),
                               vecg,h,seasonality,maxlag,multisteps,CF.type,normalizer,matwex,matxtreg);

#    CF.res <- ssoptimizerwrap(matxt, matF, matrix(matw[1,],obs.all,n.components,byrow=TRUE),
#                              as.matrix(y[1:obs]), as.matrix(vecg),
#                              h, modellags, multisteps, CF.type, normalizer,
#                              matwex, matxtreg, matv, matF2, vecg2);

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
    if(CF.type=="GV"){
        return(-obs/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(-obs/2 *((h^multisteps)*log(2*pi*exp(1)) + log(CF(C))));
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

  if(CF.type=="GV"){
    normalizer <- mean(abs(diff(y[1:obs])));
  }
  else{
    normalizer <- 0;
  }

# Estimate CES
  if(bounds==TRUE){
    res <- cobyla(C, CF, hin=constrains, lower=lowerb, upper=upperb);
    C <- res$par;
    CF.objective <- res$value;
  }
  else{
    res <- nlminb(C, CF, lower=lowerb, upper=upperb);
    C <- res$par;
    CF.objective <- res$objective;
  }

  llikelihood <- likelihood(C);
  FI <- hessian(likelihood,C);

  if(!is.null(xreg)){
    FI <- FI[1:(n.components-n.exovars),1:(n.components-n.exovars)];
  }

# Information criteria are calculated here with the constant part "log(2*pi*exp(1)/obs)*obs".
  AIC.coef <- 2*n.param*h^multisteps - 2*llikelihood;
  AICc.coef <- AIC.coef + 2 * n.param * (n.param + 1) / (obs - n.param - 1);
  BIC.coef <- log(obs)*n.param*h^multisteps - 2 * llikelihood;
# Information criterion derived and used especially for CES
#   k here is equal to number of coefficients/2 + number of complex initial states of CES.
  CIC.coef <- 2*(length(C)/2 + maxlag)*h^multisteps - 2*llikelihood;

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
    matxtreg[1:maxlag,] <- rep(C[(n.components-n.exovars+1):n.components],each=maxlag);
  }

# Estimate the elements of the transitional equation, fitted values and errors
  fitting <- cesfitterwrap(matxt,matF,matrix(matw[1,],nrow=1),as.matrix(y[1:obs]),vecg,
                           seasonality,maxlag,matwex,matxtreg)
#  fitting <- ssfitterwrap(matxt, matF, matrix(matw[1,],obs.all,n.components,byrow=TRUE), as.matrix(y[1:obs]),
#                          as.matrix(vecg), modellags, matwex, matxtreg, matv, matF2, vecg2);
  matxt[,] <- fitting$matxt;
  y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
  matxtreg[,] <- fitting$xtreg;

  errors.mat <- ts(ceserrorerwrap(matxt,matF,matrix(matw[1,],nrow=1),as.matrix(y[1:obs]),h,
                                  seasonality,maxlag,matwex,matxtreg),start=start(data),
                   frequency=frequency(data));
  colnames(errors.mat) <- paste0("Error",c(1:h));
  errors.mat <- ts(errors.mat,start=start(data),frequency=frequency(data));
  errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));

  y.for <- ts(cesforecasterwrap(matrix(matxt[((obs-maxlag+1):obs)+maxlag,],nrow=maxlag),
                                matF,matrix(matw[1,],nrow=1),h,seasonality,maxlag,
                                matrix(matwex[(obs.all-h+1):obs.all,],ncol=n.exovars),
                                matrix(matxtreg[(obs.all-h+1):obs.all,],ncol=n.exovars)),
              start=time(data)[obs]+deltat(data),frequency=frequency(data));


    s2 <- as.vector(sum(errors^2)/(obs-n.param));
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
                                 measurement=matrix(matw[1,],nrow=1), transition=matF, persistence=vecg, s2=s2, modellags=modellags);
        y.low <- ts(c(y.for) + quantvalues$lower,start=start(y.for),frequency=frequency(data));
        y.high <- ts(c(y.for) + quantvalues$upper,start=start(y.for),frequency=frequency(data));
    }
    else{
        y.low <- NA;
        y.high <- NA;
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
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- c(MAPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(diff(as.vector(data)[1:obs])))),
                           MASE(as.vector(y.holdout),as.vector(y.for),mean(abs(as.vector(data)[1:obs]))),
                           MPE(as.vector(y.holdout),as.vector(y.for),digits=5),
                           SMAPE(as.vector(y.holdout),as.vector(y.for),digits=5));
        names(errormeasures) <- c("MAPE","MASE","MASALE","MPE","SMAPE");
  }
  else{
        y.holdout <- NA;
        errormeasures <- NA;
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
             phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=A, B=B,
             n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
             CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
             int.type=int.type, int.w=int.w, ICs=ICs,
             holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);

#    cat("ABS Eigenvalues for stability condition:\n");
#    cat(1-constrains(C));
  }

return(list(model=modelname,A=A,B=B,residuals=errors,errors=errors.mat,holdout=y.holdout,
            actuals=data,fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,
            states=matxt,ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,xreg=matwex,accuracy=errormeasures));
}
