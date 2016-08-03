ssintermittent <- function(data, intermittent=c("fixed","croston","tsb")){
# Function checks the provided parameters and data for intermittency

}

iss <- function(data, intermittent=c("fixed","croston","tsb"),
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

    if(intermittent=="f"){
        pt <- ts(matrix(rep(iprob,obs),obs,1), start=start(data), frequency=frequency(data));
        pt.for <- ts(rep(iprob,h), start=time(data)[obs]+deltat(data), frequency=frequency(data));
        return(list(fitted=pt,forecast=pt.for,variance=pt.for*(1-pt.for),likelihood=0));
    }
### Croston's method
    else if(intermittent=="c"){
# Define the matrix of states
        ivt <- matrix(rep(iprob,obs+1),obs+1,1);
# Define the matrix of actuals as intervals between demands
        zeroes <- c(0,which(y!=0),obs+1);
### With this thing we fit model of the type 1/(1+qt)
#        zeroes <- diff(zeroes)-1;
        zeroes <- diff(zeroes);
# Number of intervals in Croston
        obs.int <- length(zeroes);
        iyt <- matrix(zeroes,obs.int,1);
        if(is.null(imodel)){
            crostonModel <- es(iyt,"MNN",intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence);
        }
        else{
            crostonModel <- es(iyt,model=imodel,intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence);
        }

        zeroes[length(zeroes)] <- zeroes[length(zeroes)] - 1;
        zeroes <- zeroes;
        pt <- ts(rep(1/(crostonModel$fitted),zeroes),start=start(data),frequency=frequency(data));
        pt.for <- ts(1/(crostonModel$forecast), start=time(data)[obs]+deltat(data),frequency=frequency(data));
        likelihood <- - (crostonModel$ICs["AIC"]/2 - 3);

        return(list(fitted=pt,forecast=pt.for,states=crostonModel$states,variance=pt.for*(1-pt.for),likelihood=likelihood));
    }
### TSB method
    else if(intermittent=="t"){
        ivt <- matrix(rep(iprob,obs+1),obs+1,1);
        iyt <- matrix(ot,obs,1);
        modellags <- matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0.01,1,1);
        errors <- matrix(NA,obs,1);
        iyt.fit <- matrix(NA,obs,1);

        if(!is.null(imodel)){
# If chosen model is "AAdN" or anything like that, we are taking the appropriate values
            if(nchar(imodel)==4){
                Etype <- substring(imodel,1,1);
                Ttype <- substring(imodel,2,2);
                Stype <- substring(imodel,4,4);
                damped <- TRUE;
                if(substring(imodel,3,3)!="d"){
                    message(paste0("You have defined a strange imodel: ",imodel));
                    sowhat(imodel);
                    imodel <- paste0(Etype,Ttype,"d",Stype);
                }
            }
            else if(nchar(imodel)==3){
                Etype <- substring(imodel,1,1);
                Ttype <- substring(imodel,2,2);
                Stype <- substring(imodel,3,3);
                damped <- FALSE;
            }
        }
        else{
            Etype <- "M";
            Ttype <- "N";
            Stype <- "N";
        }

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

        return(list(fitted=iyt.fit,states=ivt,forecast=iy.for,variance=iy.for*(1-iy.for),likelihood=-res$objective));
    }
}
