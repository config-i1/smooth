iss <- function(data, intermittent=c("simple","croston","tsb"),
                h=10, imodel=NULL, ipersistence=NULL){
# Function estimates and returns mean and variance of probability for intermittent State-Space model based on the chosen method
    intermittent <- substring(intermittent[1],1,1);
    y <- data;
    obs <- length(y);
    ot <- abs((y!=0)*1);
    iprob <- mean(ot);
    obs.ones <- sum(ot);
    obs.zeroes <- obs - obs.ones;
# Sizes of demand
    yot <- matrix(y[y!=0],obs.ones,1);

    if(intermittent=="s"){
        return(list(fitted=iprob,forecast=iprob,variance=iprob*(1-iprob)));
    }
    else if(intermittent=="c"){
# Define the matrix of states
        ivt <- matrix(rep(iprob,obs+1),obs+1,1);
# Define the matrix of actuals as intervals between demands
        zeroes <- c(0,which(y!=0),obs+1);
        zeroes <- diff(zeroes)-1;
# Number of intervals in Croston
        obs.int <- length(zeroes);
        iyt <- matrix(zeroes,obs.int,1);
        if(is.null(imodel)){
            if(any(iyt==0)){
                return(es(iyt,"ANN",intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence));
            }
            else{
                return(es(iyt,"MNN",intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence));
            }
        }
        else{
            return(es(iyt,model=imodel,intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence));
        }
    }
    else if(intermittent=="t"){
        ivt <- matrix(rep(iprob,obs+1),obs+1,1);
        iyt <- matrix(ot,obs,1);
        modellags <- matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0.01,1,1);
        errors <- matrix(NA,obs,1);
        iyt.fit <- matrix(NA,obs,1);

        CF <- function(C){
            vecg[,] <- C[1];
            ivt[1,] <- C[2];
            iy_kappa <- iyt*(1 - 2*kappa) + kappa;
            fitting <- fitterwrap(ivt, matF, matw, iy_kappa, vecg,
                                  modellags, "M", "N", "N", "o",
                                  matrix(0,obs,1), matrix(0,obs+1,1), matrix(1,1,1), matrix(1,1,1), matrix(1,obs,1));

            iyt.fit <- fitting$yfit;
            errors <- fitting$errors;

            CF.res <- -(C[3]-1)*sum(log(iyt.fit*(1+errors))) -
                      (C[4]-1)*sum(1-log(iyt.fit*(1+errors))) +
                      obs * log(beta(C[3],C[4]));
            return(CF.res);
        }

# Smoothing parameter, initial, alpha, betta, kappa
        kappa <- 1E-5;
        C <- c(vecg[1],ivt[1],0.1,0.1);
        res <- nloptr(C, CF, lb=c(0,0,0,0), ub=c(1,1,1000,1000),
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=500));
        C <- res$solution;

        vecg[,] <- C[1];
        ivt[1,] <- C[2];
        iy_kappa <- iyt*(1 - 2*kappa) + kappa;
        fitting <- fitterwrap(ivt, matF, matw, iy_kappa, vecg,
                              modellags, "M", "N", "N", "o",
                              matrix(0,obs,1), matrix(0,obs+1,1), matrix(1,1,1), matrix(1,1,1), matrix(1,obs,1));

        ivt <- ts(fitting$matvt,start=(time(data)[1] - deltat(data)),frequency=frequency(data));
        iyt.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
        iy.for <- ts(rep(iyt.fit[obs],h),
                     start=time(data)[obs]+deltat(data),frequency=frequency(data));

        return(list(fitted=iyt.fit,states=ivt,forecast=iy.for,variance=iy.for*(1-iy.for)));
    }
}
